
import os
import json
from time import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from models import W_LSTMix
from my_utils.tools import adjust_learning_rate
from dataset import AnomalyDataset

#performance
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

    raise KeyError(
        f"Missing processed dataset path for split '{split_name}'. "
        "Set processed_dataset_root or an explicit processed split path in configs/W_LSTMix.json."
    )

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

                with autocast():
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

def train(args, model, criterion, optimizer, device, train_loader, val_loader, test_loader, param):
    scaler = GradScaler()

    patience = args["patience"]
    threshold = args.get("threshold", 0.5)
    num_epochs = args["num_epochs"]

    best_val_loss = float("inf")
    counter = 0

    train_start_time = time()
    t_loss = []
    v_loss = []
    test_loss_hist = []

    os.makedirs(args["model_save_path"], exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        epoch_start = time()

        with tqdm(train_loader, desc=f"Train {epoch+1}/{num_epochs}", leave=False) as pbar:
            for i, batch in enumerate(pbar):
                trend_input = batch["trend_input"].to(device, non_blocking=True)
                season_input = batch["season_input"].to(device, non_blocking=True)
                label = batch["label"].to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                with autocast():
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
                        elapsed=f"{time() - epoch_start:.1f}s"
                    )

        avg_train_loss = float(np.mean(train_losses)) if len(train_losses) > 0 else 0.0
        t_loss.append(avg_train_loss)

        val_loss, val_acc, val_prec, val_rec, val_f1 = evaluate(model, criterion, val_loader, device, threshold=threshold)
        v_loss.append(val_loss)

        msg = (
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Acc: {val_acc:.4f} | "
            f"F1: {val_f1:.4f}"
        )
        print(msg)

        if test_loader is not None:
            test_loss, test_acc, test_prec, test_rec, test_f1 = evaluate(
                model, criterion, test_loader, device, threshold=threshold
            )
            test_loss_hist.append(test_loss)
            print(
                f"           Test Loss: {test_loss:.4f} | "
                f"Acc: {test_acc:.4f} | "
                f"F1: {test_f1:.4f}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), f'{args["model_save_path"]}/best_model.pth')
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered.")
                break

        adjust_learning_rate(optimizer, epoch + 1, args)

    total_time = time() - train_start_time
    print(f"Total Training Time: {total_time:.2f}s")

    loss_data = {
        "param": param,
        "train_loss": t_loss,
        "val_loss": v_loss,
        "test_loss": test_loss_hist
    }

    with open(f'{args["model_save_path"]}/loss_data.json', 'w') as f:
        json.dump(loss_data, f, indent=2)

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

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.get("batch_size", 256),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent_workers,
        prefetch_factor=args.get("prefetch_factor", 4) if num_workers > 0 else None,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.get("batch_size", 256),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent_workers,
        prefetch_factor=args.get("prefetch_factor", 4) if num_workers > 0 else None,
        drop_last=False
    )

    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.get("batch_size", 256),
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=persistent_workers,
            prefetch_factor=args.get("prefetch_factor", 4) if num_workers > 0 else None,
            drop_last=False
        )

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

    loss_name = args.get("loss", "bce").lower()
    if loss_name in {"bce", "bcewithlogits", "bcewithlogitsloss"}:
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        raise ValueError(f"Unsupported loss '{args.get('loss')}'. Expected 'bce' for this training script.")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args["learning_rate"])

    train(
        args,
        model,
        criterion,
        optimizer,
        device,
        train_loader,
        val_loader,
        test_loader,
        param
    )