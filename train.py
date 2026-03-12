#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import argparse
import json
import os
import sys
sys.path.append('./models')

import pywt
from statsmodels.tsa.seasonal import STL

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from models import W_LSTMix
from statsmodels.tsa.seasonal import seasonal_decompose

from tqdm import tqdm
from time import time
#from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from my_utils.tools import EarlyStopping, adjust_learning_rate, visual


# In[12]:


def standardize_series(series, eps=1e-8):
    mean = np.mean(series)
    std = np.std(series)
    standardized_series = (series - mean) / (std + eps)
    return standardized_series, mean, std

# def unscale_predictions(predictions, mean, std, eps=1e-8):
#     return predictions * (std+eps) + mean


# In[ ]:


def decompose_series(series, method_decom, period=24, wavelet='db4', level=5):
    """
    Decomposes a time series into trend and seasonal+residual components.
    Assumes hourly data by default (period=24).
    """
    if method_decom == 'seasonal_decompose':
       result = seasonal_decompose(series, model='additive', period=period, extrapolate_trend='freq')
       trend = result.trend
       seasonal_plus_resid = series - trend

       # Handle NaNs from the trend's boundary effects
       # trend = pd.Series(trend).fillna(method='bfill').fillna(method='ffill').values
       trend = pd.Series(trend).bfill().ffill().values
       seasonal_plus_resid = pd.Series(seasonal_plus_resid).fillna(0).values

       return trend, seasonal_plus_resid
    
   
    ##Decomposes a time series into trend and seasonal+residual components using wavelet transform, adjust level to get more in depth decompostion.

    elif method_decom == 'wavelet':
        if level is None:
            level = pywt.dwt_max_level(len(series), pywt.Wavelet(wavelet).dec_len)

        coeffs = pywt.wavedec(series, wavelet, level=level)

        # Keep only the approximation, set detail coeffs to zero for clean trend
        trend_coeffs = [coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]]
        trend = pywt.waverec(trend_coeffs, wavelet)[:len(series)]

        seasonal_plus_resid = series - trend
        seasonal_plus_resid = pd.Series(seasonal_plus_resid).fillna(0).values

        return trend, seasonal_plus_resid




# In[ ]:


# class DecomposedTimeSeriesDataset(Dataset):
class AnomalyDetectionDataset(Dataset):
    # def __init__(self, series, backcast_length, forecast_length, method_decom, stride=1, period=24):
    def __init__(self, series, labels, backcast_length, method_decom, stride=1, period=24):
        self.backcast_length = backcast_length
        # self.forecast_length = forecast_length
        self.stride = stride
        self.method_decom = method_decom
        # Decompose the series into trend and seasonality+residual
        trend, seasonality = decompose_series(series, method_decom, period=period)

        # Standardize each component
        self.trend, self.trend_mean, self.trend_std = standardize_series(trend)
        self.season, self.season_mean, self.season_std = standardize_series(seasonality)

        self.labels = labels

    def __len__(self):
        # return (len(self.trend) - self.backcast_length - self.forecast_length) // self.stride + 1
        return (len(self.trend) - self.backcast_length) // self.stride + 1

    def __getitem__(self, idx):
        start = idx * self.stride

        # Inputs
        trend_input = self.trend[start : start + self.backcast_length]
        season_input = self.season[start : start + self.backcast_length]

        # # Targets
        # trend_target = self.trend[start + self.backcast_length : start + self.backcast_length + self.forecast_length]
        # season_target = self.season[start + self.backcast_length : start + self.backcast_length + self.forecast_length]

        # Per-point labels for the window
        window_labels = self.labels[start : start + self.backcast_length]

        return {
            'trend_input': torch.tensor(trend_input, dtype=torch.float32),
            'season_input': torch.tensor(season_input, dtype=torch.float32),
            # 'trend_target': torch.tensor(trend_target, dtype=torch.float32),
            # 'season_target': torch.tensor(season_target, dtype=torch.float32),
            'label': torch.tensor(window_labels, dtype=torch.float32),
        }


# In[ ]:


#def load_datasets(folder_path, backcast_length, forecast_length, method_decom, stride=1, period=24):
def load_datasets(folder_path, backcast_length, method_decom, stride=1, period=24):
    datasets = []

    for region in os.listdir(folder_path):
        region_path = os.path.join(folder_path, region)

        for building in os.listdir(region_path):

            if building.endswith('.csv'):
                file_path = os.path.join(region_path, building)
                df = pd.read_csv(file_path)
                energy_data = df['energy'].values
                #dataset = DecomposedTimeSeriesDataset(energy_data, backcast_length, forecast_length, method_decom, stride)
                labels = df['label'].values if 'label' in df.columns else np.zeros(len(energy_data))
                dataset = AnomalyDetectionDataset(energy_data, labels, backcast_length, method_decom, stride, period)
                datasets.append(dataset)


            elif building.endswith('.parquet'):
                file_path = os.path.join(region_path, building)
                df = pd.read_parquet(file_path)

                if 'energy' not in df.columns:
                    continue  # Skip if energy column is missing

                energy_data = df['energy'].values
                #dataset = DecomposedTimeSeriesDataset(energy_data, backcast_length, forecast_length, method_decom,  stride, period)
                labels = df['label'].values if 'label' in df.columns else np.zeros(len(energy_data))
                dataset = AnomalyDetectionDataset(energy_data, labels, backcast_length, method_decom, stride, period)
                datasets.append(dataset)

            else:
                print("Wrong file format!")

    if len(datasets) == 0:
        raise RuntimeError("No valid parquet datasets found.")

    return ConcatDataset(datasets)


# ## Dynamic Coefficient Based on Loss Magnitude

# In[16]:


def train(args, model, criterion, optimizer, device, train_loader, val_loader, param):

    # Early stopping parameters
    patience = args['patience']
    best_val_loss = float('inf')
    counter = 0
    early_stop = False

    threshold = args.get('threshold', 0.5)

    num_epochs = args["num_epochs"]
    train_start_time = time()  # Start timer 

    t_loss = []
    v_loss = []

    for epoch in range(num_epochs):

        if early_stop:
            print(f"Early stopping at epoch {epoch + 1}")
            break  

        model.train()
        train_losses = []

        epoch_start_time = time()  # Start epoch timer

        # Progress bar for the training loop
        with tqdm(train_loader, desc=f'Training Epoch {epoch+1}/{num_epochs}', leave=False) as pbar:
            for i, batch in enumerate(pbar):
                trend_input = batch['trend_input'].to(device)
                season_input = batch['season_input'].to(device)
                # trend_target = batch['trend_target'].to(device)
                # season_target = batch['season_target'].to(device)
                label = batch['label'].to(device)

                optimizer.zero_grad()

                # # Forward pass: Get trend and season predictions
                # trend_pred, season_pred = model(trend_input, season_input)

                # # Calculate loss for trend and season separately (you could also add weightings)
                # loss_trend = criterion(trend_pred, trend_target)
                # loss_season = criterion(season_pred, season_target)
                
                # # Total loss is the sum of trend and season losses
                # # total_loss = 0.3 * loss_trend + 0.7 * loss_season

                # sum_loss = loss_trend + loss_season
                # alpha = loss_season / sum_loss
                # beta = loss_trend / sum_loss

                # total_loss = alpha * loss_trend + beta * loss_season

                logits = model(trend_input, season_input)
                loss = criterion(logits, label)

                # total_loss.backward()
                loss.backward()
                optimizer.step()

                # train_losses.append(total_loss.item())
                train_losses.append(loss.item())

                if i % 5 ==0:
                    #pbar.set_postfix(loss=total_loss.item(), elapsed=f"{time() - epoch_start_time:.2f}s")
                    pbar.set_postfix(loss=loss.item(), elapsed=f"{time() - epoch_start_time:.2f}s")

        # Calculate average training loss
        avg_train_loss = np.mean(train_losses)
        t_loss.append(avg_train_loss)

        # Validation phase
        model.eval()
        val_losses = []
        # y_true_val = []
        # y_pred_val = []
        all_labels = []
        all_preds = []

        # Progress bar for the validation loop
        with tqdm(val_loader, desc=f'Validation Epoch {epoch+1}/{num_epochs}', leave=False) as pbar:
            for batch in pbar:
                trend_input = batch['trend_input'].to(device)
                season_input = batch['season_input'].to(device)
                # trend_target = batch['trend_target'].to(device)
                # season_target = batch['season_target'].to(device)
                label = batch['label'].to(device)

                with torch.no_grad():
                    # trend_pred, season_pred = model(trend_input, season_input)
                    # loss_trend = criterion(trend_pred, trend_target)
                    # loss_season = criterion(season_pred, season_target)
                    logits = model(trend_input, season_input)
                    val_loss = criterion(logits, label)

                    # val_loss = 0.3 * loss_trend + 0.7 * loss_season


                    # sum_loss = loss_trend + loss_season
                    # alpha = loss_season / sum_loss
                    # beta = loss_trend / sum_loss

                    # val_loss = alpha * loss_trend + beta * loss_season
                    val_losses.append(val_loss.item())

                    # # Collect true and predicted values for RMSE calculation
                    # y_true_val.extend(trend_target.cpu().numpy())
                    # y_pred_val.extend(trend_pred.cpu().numpy())
                    # y_true_val.extend(season_target.cpu().numpy())
                    # y_pred_val.extend(season_pred.cpu().numpy())

                    probs = torch.sigmoid(logits)
                    preds = (probs >= threshold).float()
                    # Flatten to point-level: each point gets its own prediction
                    all_labels.extend(label.cpu().numpy().flatten())
                    all_preds.extend(preds.cpu().numpy().flatten())

        # Calculate average validation loss and RMSE
        avg_val_loss = np.mean(val_losses)
        v_loss.append(avg_val_loss)

        # rmse_val = np.sqrt(mean_squared_error(y_true_val, y_pred_val))
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, zero_division=0)

        # Print epoch summary
        # print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, RMSE: {rmse_val:.4f}')
        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Acc: {acc:.4f}, F1: {f1:.4f}')

        # Save the best model parameters
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            counter = 0
            os.makedirs(args["model_save_path"], exist_ok=True)
            torch.save(model.state_dict(), f'{args["model_save_path"]}/best_model.pth')
        else:
            counter += 1
            if counter >= patience:
                early_stop = True

        # Adjust learning rate
        adjust_learning_rate(optimizer, epoch + 1, args)


    total_training_time = time() - train_start_time
    print(f'Total Training Time: {total_training_time:.2f}s')

    # Save loss data
    loss_data = {
        "param": param,
        "train_loss": t_loss,
        "val_loss": v_loss
    }

    loss_data_path = f'{args["model_save_path"]}/loss_data.json'
    with open(loss_data_path, "w") as f:
        json.dump(loss_data, f)


# In[ ]:


config_file = "./configs/W_LSTMix.json"
with open(config_file, 'r') as f:
    args = json.load(f)

# train_datasets = load_datasets(args['train_dataset_path'], args['backcast_length'], args['forecast_length'],args['method_decom'], args['stride'])
# val_datasets = load_datasets(args['val_dataset_path'], args['backcast_length'], args['forecast_length'],args['method_decom'],  args['stride'])

train_datasets = load_datasets(args['train_dataset_path'], args['backcast_length'], args['method_decom'], args['stride'])
val_datasets = load_datasets(args['val_dataset_path'], args['backcast_length'], args['method_decom'], args['stride'])


# Create data loaders
train_loader = DataLoader(train_datasets, batch_size=args['batch_size'], shuffle=True)
val_loader = DataLoader(val_datasets, batch_size=args['batch_size'], shuffle=True)
# val_loader = DataLoader(val_datasets, batch_size=args['batch_size'], shuffle=False)  SHUFFLE OR NOT?



# check device 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define N-BEATS model
model = W_LSTMix.Model(
    device=device,
    num_blocks_per_stack=args['num_blocks_per_stack'],
    # forecast_length=args['forecast_length'],
    backcast_length=args['backcast_length'],
    patch_size=args['patch_size'],
    num_patches=args['backcast_length'] // args['patch_size'],
    thetas_dim=args['thetas_dim'],
    hidden_dim=args['hidden_dim'],
    embed_dim=args['embed_dim'],
    num_heads=args['num_heads'],
    ff_hidden_dim=args['ff_hidden_dim'],
    num_classes=args.get('num_classes', 1),
).to(device)

# model's parameters
param = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Model's parameter count is:", param)

# Define loss and optimizer
# if args['loss'] == 'mse':
#     criterion = torch.nn.MSELoss()
# else:
#     criterion = torch.nn.HuberLoss(reduction="mean", delta=1)

criterion = torch.nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=args["learning_rate"])

# training the model and save best parameters
train(args, model, criterion, optimizer, device, train_loader, val_loader, param)



# In[ ]:




