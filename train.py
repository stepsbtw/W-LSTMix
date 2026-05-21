import os
import json
from time import time
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from models import W_LSTMix
# my_utils.tools import adjust_learning_rate 
from dataset import AnomalyDataset

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")


def resolve_processed_split_dir(args, split_name):
    explicit_keys = (
        f"processed_{split_name}_dataset_path",
        f"processed_{split_name}_path",
        f"{split_name}_processed_dataset_path",
        f"{split_name}_processed_path",
    )

    for key in explicit_keys:
        split_dir = args.get(key)
        if split_dir:
            return split_dir

    processed_root = args.get("processed_dataset_root")
    if processed_root:
        return os.path.join(processed_root, split_name)

    raise KeyError(f"Missing processed dataset path for split '{split_name}'. Set processed_dataset_root or an explicit processed split path in configs/W_LSTMix.json.")


def compute_automated_pos_weight(dataset, dataloader):
    print("Calculating automated class weights from training dataset...")
    
    if hasattr(dataset, 'labels') and dataset.labels is not None:
        labels = torch.tensor(dataset.labels)
    elif hasattr(dataset, 'targets') and dataset.targets is not None:
        labels = torch.tensor(dataset.targets)
    else:
        print("Dataset properties not exposed. Performing a fast dataloader label sweep...")
        label_list = []
        for batch in tqdm(dataloader, desc="Label Sweep", leave=False):
            label_list.append(batch["label"].view(-1))
        labels = torch.cat(label_list)
        
    num_normals = (labels == 0).sum().item()
    num_anomalies = (labels == 1).sum().item()
    
    if num_anomalies == 0:
        print("Warning: No anomalies found in training set. Setting pos_weight to 1.0")
        return torch.tensor([1.0])
        
    pos_weight_val = num_normals / num_anomalies
    print(f"--> Found {num_normals} normal samples and {num_anomalies} anomalies.")
    print(f"--> Automated pos_weight value: {pos_weight_val:.4f}")
    return torch.tensor([pos_weight_val])


def evaluate(model, criterion, loader, device, threshold=0.5):
    model.eval()

    losses = []
    tp = fp = fn = 0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        with tqdm(loader, desc="Eval", leave=False) as pbar:
            for batch in pbar:
                trend_input = batch["trend_input"].to(device, non_blocking=True)
                season_input = batch["season_input"].to(device, non_blocking=True)
                label = batch["label"].to(device, non_blocking=True)

                with autocast('cuda'):
                    logits = model(trend_input, season_input)
                    loss = criterion(logits, label)

                losses.append(loss.item())

                probs = torch.sigmoid(logits)
                preds = (probs >= threshold).float()

                total_correct += (preds == label).sum().item()
                total_samples += label.numel()

                tp += ((preds == 1) & (label == 1)).sum().item()
                fp += ((preds == 1) & (label == 0)).sum().item()
                fn += ((preds == 0) & (label == 1)).sum().item()

    avg_loss = float(np.mean(losses)) if len(losses) > 0 else 0.0
    acc = total_correct / total_samples if total_samples > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return avg_loss, acc, precision, recall, f1


def save_checkpoint(path, epoch, model, optimizer, scaler, best_val_loss, counter, train_loss, val_loss, test_loss, param):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "best_val_loss": best_val_loss,
        "early_stop_counter": counter,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "test_loss": test_loss,
        "param": param,
    }
    torch.save(checkpoint, path)


def load_checkpoint(path, model, optimizer=None, scaler=None, device="cpu"):
    print(f"Loading checkpoint: {path}")

    checkpoint = torch.load(path, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scaler is not None and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    start_epoch = checkpoint.get("epoch", -1) + 1
    best_val_loss = checkpoint.get("best_val_loss", float("inf"))
    counter = checkpoint.get("early_stop_counter", 0)

    train_loss = checkpoint.get("train_loss", [])
    val_loss = checkpoint.get("val_loss", [])
    test_loss = checkpoint.get("test_loss", [])

    print(f"Checkpoint loaded successfully. Resuming from epoch {start_epoch}")

    return start_epoch, best_val_loss, counter, train_loss, val_loss, test_loss


def save_logs(log_path, log_data):
    """Overwrite the log file with the latest log_data dict."""
    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2)


def train(args, model, criterion, optimizer, device, train_loader, val_loader, test_loader, param):
    scaler = GradScaler('cuda')

    patience = args["patience"]
    threshold = args.get("threshold", 0.5)
    num_epochs = args["num_epochs"]

    os.makedirs(args["model_save_path"], exist_ok=True)
    save_dir = args["model_save_path"]

    checkpoint_path = os.path.join(save_dir, "latest_checkpoint.pth")

    start_epoch = 0
    best_val_loss = float("inf")
    counter = 0
    t_loss = []
    v_loss = []
    test_loss_hist = []

    if os.path.isfile(checkpoint_path):
        start_epoch, best_val_loss, counter, t_loss, v_loss, test_loss_hist = load_checkpoint(
            checkpoint_path, model, optimizer, scaler, device
        )
    else:
        print("No checkpoint found — starting from scratch.")

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3
    )

    log_path = os.path.join(save_dir, "training_log.json")
    epoch_logs = []
    prior_best_epoch = None
    prior_run_start = datetime.now().isoformat(timespec="seconds")

    if os.path.isfile(log_path):
        try:
            with open(log_path, "r") as f:
                prior = json.load(f)
            epoch_logs = prior.get("epochs", [])
            prior_best_epoch = prior.get("best_epoch", None)
            prior_run_start = prior.get("run_start", prior_run_start)
            
            epoch_logs = [e for e in epoch_logs if e["epoch"] <= start_epoch]
            print(f"Restored {len(epoch_logs)} sanitized epoch log(s) from {log_path}")
        except Exception as e:
            print(f"Warning: could not restore training log ({e}), starting fresh log.")

    log_data = {
        "run_start": prior_run_start,
        "param": param,
        "args": args,
        "best_val_loss": best_val_loss,
        "best_epoch": prior_best_epoch,
        "total_training_time_s": None,
        "epochs": epoch_logs,
        "train_loss": t_loss,
        "val_loss": v_loss,
        "test_loss": test_loss_hist,
    }

    train_start_time = time()

    for epoch in range(start_epoch, num_epochs):
        model.train()

        train_losses = []
        epoch_start = time()

        with tqdm(train_loader, desc=f"Train {epoch+1}/{num_epochs}", leave=False) as pbar:
            for i, batch in enumerate(pbar):
                trend_input = batch["trend_input"].to(device, non_blocking=True)
                season_input = batch["season_input"].to(device, non_blocking=True)
                label = batch["label"].to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                with autocast('cuda'):
                    logits = model(trend_input, season_input)
                    loss = criterion(logits, label)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                train_losses.append(loss.item())

                if i % 10 == 0:
                    pbar.set_postfix(
                        loss=f"{loss.item():.4f}",
                        gpu_mem=f"{torch.cuda.memory_allocated()/1024**3:.2f}GB",
                        elapsed=f"{time() - epoch_start:.1f}s",
                    )

        epoch_time = time() - epoch_start
        avg_train_loss = float(np.mean(train_losses)) if train_losses else 0.0
        t_loss.append(avg_train_loss)

        val_loss, val_acc, val_prec, val_rec, val_f1 = evaluate(model, criterion, val_loader, device, threshold=threshold)
        v_loss.append(val_loss)

        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f}")

        epoch_record = {
            "epoch": epoch + 1,
            "epoch_time_s": round(epoch_time, 2),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": {
                "loss": round(avg_train_loss, 6),
            },
            "val": {
                "loss": round(val_loss, 6),
                "accuracy": round(val_acc, 6),
                "precision": round(val_prec, 6),
                "recall": round(val_rec, 6),
                "f1": round(val_f1, 6),
            },
            "test": None,
        }

        if test_loader is not None:
            test_loss, test_acc, test_prec, test_rec, test_f1 = evaluate(model, criterion, test_loader, device, threshold=threshold)
            test_loss_hist.append(test_loss)
            print(f"           Test  Loss: {test_loss:.4f} | Acc: {test_acc:.4f} | F1: {test_f1:.4f}")

            epoch_record["test"] = {
                "loss": round(test_loss, 6),
                "accuracy": round(test_acc, 6),
                "precision": round(test_prec, 6),
                "recall": round(test_rec, 6),
                "f1": round(test_f1, 6),
            }

        epoch_logs.append(epoch_record)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0

            best_model_path = os.path.join(save_dir, "best_model.pth")
            save_checkpoint(best_model_path, epoch, model, optimizer, scaler, best_val_loss, counter, t_loss, v_loss, test_loss_hist, param)
            print(f"Saved best model to: {best_model_path}")

            log_data["best_val_loss"] = round(best_val_loss, 6)
            log_data["best_epoch"] = epoch + 1

        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered.")
                epoch_logs[-1]["early_stop"] = True
                save_checkpoint(checkpoint_path, epoch, model, optimizer, scaler, best_val_loss, counter, t_loss, v_loss, test_loss_hist, param)
                break

        save_checkpoint(checkpoint_path, epoch, model, optimizer, scaler, best_val_loss, counter, t_loss, v_loss, test_loss_hist, param)

        scheduler.step(val_loss)

        log_data["total_training_time_s"] = round(time() - train_start_time, 2)
        save_logs(log_path, log_data)

    total_time = time() - train_start_time
    log_data["total_training_time_s"] = round(total_time, 2)
    log_data["run_end"] = datetime.now().isoformat(timespec="seconds")
    save_logs(log_path, log_data)

    print(f"Total Training Time: {total_time:.2f}s")
    print(f"Training log saved to: {log_path}")


if __name__ == "__main__":
    config_file = Path(__file__).resolve().parent / "configs" / "W_LSTMix.json"

    with open(config_file, "r") as f:
        args = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Using device:", device)

    train_dir = resolve_processed_split_dir(args, "train")
    val_dir = resolve_processed_split_dir(args, "val")
    test_dir = resolve_processed_split_dir(args, "test")

    train_dataset = AnomalyDataset(train_dir)
    val_dataset = AnomalyDataset(val_dir)
    test_dataset = AnomalyDataset(test_dir) if os.path.isdir(test_dir) else None

    num_workers = args.get("num_workers", 16)
    persistent_workers = num_workers > 0

    train_loader = DataLoader(train_dataset, batch_size=args.get("batch_size", 256), shuffle=True, num_workers=num_workers, pin_memory=True, persistent_workers=persistent_workers, prefetch_factor=args.get("prefetch_factor", 4) if num_workers > 0 else None, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.get("batch_size", 256), shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=persistent_workers, prefetch_factor=args.get("prefetch_factor", 4) if num_workers > 0 else None, drop_last=False)

    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(test_dataset, batch_size=args.get("batch_size", 256), shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=persistent_workers, prefetch_factor=args.get("prefetch_factor", 4) if num_workers > 0 else None, drop_last=False)

    model = W_LSTMix.Model(
        device=device,
        num_blocks_per_stack=args["num_blocks_per_stack"],
        backcast_length=args["backcast_length"],
        patch_size=args["patch_size"],
        num_patches=args["backcast_length"] // args["patch_size"],
        thetas_dim=args["thetas_dim"],
        hidden_dim=args["hidden_dim"],
        embed_dim=args["embed_dim"],
        num_heads=args["num_heads"],
        ff_hidden_dim=args["ff_hidden_dim"],
        context_length=args.get("context_length", args["backcast_length"]),
        num_classes=args.get("num_classes", 1),
    ).to(device)

    param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Trainable params:", param)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args["learning_rate"])

    loss_name = args.get("loss", "bce").lower()

    if loss_name in {"bce", "bcewithlogits", "bcewithlogitsloss"}:
        pos_weight = compute_automated_pos_weight(train_dataset, train_loader).to(device)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        raise ValueError(f"Unsupported loss '{args.get('loss')}'. Expected 'bce' for this training script.")

    train(args, model, criterion, optimizer, device, train_loader, val_loader, test_loader, param)