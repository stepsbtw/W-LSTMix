#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
# from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
# from sklearn.model_selection import train_test_split
from my_utils.tools import EarlyStopping, adjust_learning_rate, visual


# In[3]:


def standardize_series(series, eps=1e-8):
    mean = np.mean(series)
    std = np.std(series)
    standardized_series = (series - mean) / (std + eps)
    return standardized_series, mean, std

# def unscale_predictions(predictions, mean, std, eps=1e-8):
#     return predictions * (std+eps) + mean


# In[4]:


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


# In[5]:


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


# In[6]:


#def fixed_time_split(series, train_len=4320, val_len=2160):
def fixed_time_split(series, labels, train_len=4320, val_len=2160):
    total_len = len(series)

    if total_len < train_len + val_len + 1:
        # Fallback when data is too short
        half_point = total_len // 2
        train_data = series[:half_point]
        train_labels = labels[:half_point]
        val_data = series[half_point:]
        val_labels = labels[half_point:]
        test_data = series[half_point:]
        test_labels = labels[half_point:]
    else:
        train_data = series[:train_len]
        train_labels = labels[:train_len]
        val_data = series[train_len:train_len + val_len]
        val_labels = labels[train_len:train_len + val_len]
        test_data = series[train_len:]
        test_labels = labels[train_len:]

    # return train_data, val_data, test_data
    return train_data, train_labels, val_data, val_labels, test_data, test_labels

# In[ ]:


'''def fixed_time_split(series, train_len=4320, val_len=2160):
    total_len = len(series)

    if total_len < train_len + val_len + 1:  # not enough for all three splits
        return None, None, None
    
    train_data = series[:train_len]
    val_data = series[train_len:train_len+val_len]
    test_data = series[train_len:]

    return train_data, val_data, test_data'''



# In[7]:


#def load_datasets(folder_path, backcast_length, forecast_length, method_decom, stride=1, period=24):
def load_datasets(folder_path, backcast_length, method_decom, stride=1, period=24):
   
    train_datasets = []
    val_datasets = []
    test_datasets = []

    for region in os.listdir(folder_path):
        region_path = os.path.join(folder_path, region)

        for building in os.listdir(region_path):

            if building.endswith('.csv'):
                file_path = os.path.join(region_path, building)
                df = pd.read_csv(file_path)
                energy_data = df['energy'].values
                
                labels = df['label'].values if 'label' in df.columns else np.zeros(len(energy_data))
                
                # train_data, val_data, test_data = fixed_time_split(energy_data)
                train_data, train_labels, val_data, val_labels, test_data, test_labels = fixed_time_split(energy_data, labels)

                if train_data is None:
                    continue

                # train_dataset = DecomposedTimeSeriesDataset(train_data, backcast_length, forecast_length, method_decom,stride,period)
                # val_dataset = DecomposedTimeSeriesDataset(val_data, backcast_length, forecast_length,method_decom, stride,period)
                # test_dataset = DecomposedTimeSeriesDataset(test_data, backcast_length, forecast_length,method_decom, stride,period)
             
                # train_datasets.append(train_dataset)
                # val_datasets.append(val_dataset)
                # test_datasets.append(test_dataset)
                                
                train_datasets.append(AnomalyDetectionDataset(train_data, train_labels, backcast_length, method_decom, stride, period))
                val_datasets.append(AnomalyDetectionDataset(val_data, val_labels, backcast_length, method_decom, stride, period))
                test_datasets.append(AnomalyDetectionDataset(test_data, test_labels, backcast_length, method_decom, stride, period))


            elif building.endswith('.parquet'):
                file_path = os.path.join(region_path, building)
                df = pd.read_parquet(file_path)

                if 'energy' not in df.columns:
                    continue  # Skip if energy column is missing

                energy_data = df['energy'].values
                # train_data, val_data, test_data = fixed_time_split(energy_data)
                labels = df['label'].values if 'label' in df.columns else np.zeros(len(energy_data))
                train_data, train_labels, val_data, val_labels, test_data, test_labels = fixed_time_split(energy_data, labels)

                if train_data is None:
                    continue

                # # Create TimeSeriesDataset for each split
                # train_dataset = DecomposedTimeSeriesDataset(train_data, backcast_length, forecast_length, method_decom,stride,period)
                # val_dataset = DecomposedTimeSeriesDataset(val_data, backcast_length, forecast_length,method_decom, stride,period)
                # test_dataset = DecomposedTimeSeriesDataset(test_data, backcast_length, forecast_length, method_decom,stride,period)
             
                # train_datasets.append(train_dataset)
                # val_datasets.append(val_dataset)
                # test_datasets.append(test_dataset)

                train_datasets.append(AnomalyDetectionDataset(train_data, train_labels, backcast_length, method_decom, stride, period))
                val_datasets.append(AnomalyDetectionDataset(val_data, val_labels, backcast_length, method_decom, stride, period))
                test_datasets.append(AnomalyDetectionDataset(test_data, test_labels, backcast_length, method_decom, stride, period))

            else:
                print("Wrong file format!")

    if len(train_datasets) == 0:
        raise RuntimeError("No valid parquet datasets found.")
    
    # print(len(train_dataset), len(val_dataset), len(test_dataset))
    # Combine all datasets for each split
    combined_train_dataset = ConcatDataset(train_datasets)
    combined_val_dataset = ConcatDataset(val_datasets)
    combined_test_dataset = ConcatDataset(test_datasets)
    print(len(combined_train_dataset), len(combined_val_dataset), len(combined_test_dataset))

    return combined_train_dataset, combined_val_dataset, combined_test_dataset


# In[7]:


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

                logits = model(trend_input, season_input)
                # total_loss = alpha * loss_trend + beta * loss_season
                loss = criterion(logits, label)

                # total_loss.backward()
                loss.backward()
                optimizer.step()

                #train_losses.append(total_loss.item())
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
                    # # val_loss = 0.3 * loss_trend + 0.7 * loss_season

                    # sum_loss = loss_trend + loss_season
                    # alpha = loss_season / sum_loss
                    # beta = loss_trend / sum_loss

                    # val_loss = alpha * loss_trend + beta * loss_season
                    logits = model(trend_input, season_input)
                    val_loss = criterion(logits, label)
                    val_losses.append(val_loss.item())

                    # Collect true and predicted values for RMSE calculation
                    # y_true_val.extend(trend_target.cpu().numpy())
                    # y_pred_val.extend(trend_pred.cpu().numpy())
                    # y_true_val.extend(season_target.cpu().numpy())
                    # y_pred_val.extend(season_pred.cpu().numpy())
                    probs = torch.sigmoid(logits)
                    preds = (probs >= threshold).float()
                    # Flatten to point-level
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
        #print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, RMSE: {rmse_val:.4f}')
        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Acc: {acc:.4f}, F1: {f1:.4f}')


        # Save the best model parameters
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            counter = 0
            os.makedirs(args["finetuned_model_save_path"], exist_ok=True)
            torch.save(model.state_dict(), f'{args["finetuned_model_save_path"]}/best_model.pth')
          
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

    loss_data_path = f'{args["finetuned_model_save_path"]}/loss_data.json'
    with open(loss_data_path, "w") as f:
        json.dump(loss_data, f)


# In[8]:


config_file = "./configs/W_LSTMix.json"
with open(config_file, 'r') as f:
    args = json.load(f)

#train_datasets, val_datasets,_ = load_datasets(args['test_dataset_path'], args['backcast_length'], args['forecast_length'],args['method_decom'], args['stride'])
train_datasets, val_datasets, _ = load_datasets(args['test_dataset_path'], args['backcast_length'], args['method_decom'], args['stride'])


# Create data loaders
train_loader = DataLoader(train_datasets, batch_size=args['batch_size'], shuffle=True)
val_loader = DataLoader(val_datasets, batch_size=args['batch_size'], shuffle=False)



# check device 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define N-BEATS model
model = W_LSTMix.Model(
    device=device,
    num_blocks_per_stack=args['num_blocks_per_stack'],
    #forecast_length=args['forecast_length'],
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

model.load_state_dict(torch.load(f'{args["pretrained_model_path"]}/best_model.pth'))
#model.load_state_dict(torch.load(f'{args["pretrained_model_path"]}/best_model.pth'), strict=False)

# model's parameters
param = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Model's parameter count is:", param)

# # Define loss and optimizer
# if args['loss'] == 'mse':
#     criterion = torch.nn.MSELoss()
# else:
#     criterion = torch.nn.HuberLoss(reduction="mean", delta=1)

criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args["learning_rate"])

# training the model and save best parameters
train(args, model, criterion, optimizer, device, train_loader, val_loader, param)


# ### Testing

# In[19]:


import numpy as np
import pandas as pd
import argparse
import json
import os
import sys
sys.path.append('./model')

import torch
from torch.utils.data import Dataset, DataLoader
from models import W_LSTMix
from torch.utils.data import ConcatDataset

from tqdm import tqdm
from my_utils.metrics import cal_cvrmse, cal_mae, cal_mse, cal_nrmse
from my_utils.decompose_normalize import standardize_series, unscale_predictions, decompose_series


# In[20]:


import matplotlib.pyplot as plt

def plot_forecast_with_context(backcast, forecast_true, forecast_pred, building_id, idx, save_path):
    """
    Plots backcast (input context), true forecast, and predicted forecast for a time series sample.
    """
    plt.figure(figsize=(10, 4))
    
    # Plot context (backcast) in blue
    context_range = list(range(len(backcast)))
    plt.plot(context_range, backcast, label='Context (Past)', color='blue')
    
    # Plot true forecast in blue (dashed)
    forecast_range = list(range(len(backcast), len(backcast) + len(forecast_true)))
    plt.plot(forecast_range, forecast_true, label='True Forecast', color='blue', linestyle='--')
    
    # Plot predicted forecast in red
    plt.plot(forecast_range, forecast_pred, label='Predicted Forecast', color='red')
    
    plt.title(f"{building_id} - Forecast {idx}")
    plt.xlabel("Time")
    plt.ylabel("Energy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, f"{building_id}_forecast_{idx}.png"))
    plt.close()


# In[42]:


# import matplotlib.pyplot as plt
# import os

# def plot_forecast_grid(building_forecast_data, save_path, filename='forecast_grid.png'):
#     """
#     Plots a 2x3 grid of forecasts for selected buildings.

#     Args:
#         building_forecast_data: List of tuples -> (building_id, idx, backcast, forecast_true, forecast_pred)
#         save_path: Directory to save the final image
#         filename: Name of the final saved image file
#     """
#     fig, axs = plt.subplots(2, 3, figsize=(18, 8))
#     axs = axs.flatten()

#     for i, (building_id, idx, backcast, forecast_true, forecast_pred) in enumerate(building_forecast_data):
#         ax = axs[i]
#         context_range = list(range(len(backcast)))
#         forecast_range = list(range(len(backcast), len(backcast) + len(forecast_true)))

#         # Context
#         ax.plot(context_range, backcast, label='Context (Past)',color='blue')

#         # True forecast
#         ax.plot(forecast_range, forecast_true, label='True Forecast',color='blue', linestyle='--')

#         # Predicted forecast
#         ax.plot(forecast_range, forecast_pred, label='Predicted Forecast', color='orangered')

#         building_name = building_id.rsplit('_forecast_', 1)[0]  # removes _forecast_6
#         building_name = building_name.replace('all_buildings_power_processed_', '')  # optional cleanup
#         ax.set_title(building_name, fontsize=14, fontweight='bold', color='black')


#         #ax.set_title(f"{building_id.replace('_forecast_', ' [#')})", fontsize=11)
#         ax.grid(True)

#         # Only label outer plots to reduce clutter
#         if i % 3 == 0:
#             ax.set_ylabel("Energy(kWh)", fontsize=10, fontweight='bold', color='black')
#         if i >= 3:
#             ax.set_xlabel("Time", fontsize=10, fontweight='bold', color='black')

#     # Common legend (outside of subplots)
#     handles, labels = axs[0].get_legend_handles_labels()
#     fig.legend(handles, labels, loc='upper center', ncol=3, fontsize=11, frameon=False)

#     plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for legend
#     os.makedirs(save_path, exist_ok=True)

#     # Save in high resolution
#     plt.savefig(os.path.join(save_path, filename), dpi=300, bbox_inches='tight')
#     plt.savefig(os.path.join(save_path, filename.replace(".png", ".pdf")), bbox_inches='tight')
#     plt.close()


# In[43]:


def test(args, model, criterion, device):

    folder_path = args['test_dataset_path']
    result_path = args['result_path_finetune']
    backcast_length = args['backcast_length']
    #forecast_length = args['forecast_length']
    threshold = args.get('threshold', 0.5)
    stride = args['stride']
    period = 24
    method_decom = args['method_decom']


    # median_res = []  
    # building_forecast_data = []  # To collect selected forecasts for final ICML plot
    # selected_buildings = set([
    #     'all_buildings_power_processed_Boys_backup_forecast_8',
    #     'all_buildings_power_processed_Girls_backup_forecast_0',
    #     'all_buildings_power_processed_Library_forecast_6',
    #     'enernoc_45_forecast_2',
    #     'enernoc_136_forecast_3',
    #     'enernoc_304_forecast_0'
    # ])

    all_region_res = []

    for region in os.listdir(folder_path):

        region_path = os.path.join(folder_path, region)

        results_path = os.path.join(result_path, region)
        os.makedirs(results_path, exist_ok=True)

        res = []

        for building in os.listdir(region_path):

            

            if building.endswith('.csv') or building.endswith('.parquet'):
                file_path = os.path.join(region_path, building)
                if building.endswith('.csv'):
                    building_id = building.rsplit(".csv",1)[0]
                    df = pd.read_csv(file_path)
                else:
                    building_id = building.rsplit(".parquet",1)[0]
                    df = pd.read_parquet(file_path)
                energy_data = df['energy'].values
                # train_data, val_data, test_data = fixed_time_split(energy_data)

                labels = df['label'].values if 'label' in df.columns else np.zeros(len(energy_data))
                _, _, _, _, test_data, test_labels = fixed_time_split(energy_data, labels)
            
                if test_data is None:
                    continue
                #dataset = DecomposedTimeSeriesDataset(test_data, backcast_length, forecast_length, method_decom, stride, period)
                dataset = AnomalyDetectionDataset(test_data, test_labels, backcast_length, method_decom, stride, period)

                # test phase
                model.eval()
                test_losses = []
                # y_true_trend = []
                # y_true_seasonal = []
                # y_pred_trend = []
                # y_pred_seasonal = []
                # sample_plot_count = 0
                # # test loop
                # for batch in tqdm(DataLoader(dataset, batch_size=1, num_workers=4), desc=f"Testing {building_id}", leave=False):

                all_labels = []
                all_probs = []
                all_preds = []

                for batch in tqdm(DataLoader(dataset, batch_size=64), desc=f"Testing {building_id}", leave=False):
                    trend_input = batch['trend_input'].to(device)
                    season_input = batch['season_input'].to(device)
                    # trend_target = batch['trend_target'].to(device)
                    # season_target = batch['season_target'].to(device)

                    label = batch['label'].to(device)

                    with torch.no_grad():
                    #     trend_pred, season_pred = model(trend_input, season_input)
                    #     loss_trend = criterion(trend_pred, trend_target)
                    #     loss_season = criterion(season_pred, season_target)

                    #     sum_loss = loss_trend + loss_season
                    #     alpha = loss_season / sum_loss
                    #     beta = loss_trend / sum_loss
                        
                    #    loss = alpha * loss_trend + beta * loss_season
                        logits = model(trend_input, season_input)
                        loss = criterion(logits, label)
                        
                        test_losses.append(loss.item())
                        
                        # Collect true and predicted values for RMSE calculation
                        # y_true_trend.extend(trend_target.cpu().numpy())
                        # y_true_seasonal.extend(season_target.cpu().numpy())
                        # y_pred_trend.extend(trend_pred.cpu().numpy())
                        # y_pred_seasonal.extend(season_pred.cpu().numpy())

                        # if len(y_pred_seasonal) <= 3:
                        #     forecast_idx = len(y_pred_seasonal)
                        #     true_combined = (season_target + trend_target).squeeze().cpu().numpy()
                        #     pred_combined = (season_pred + trend_pred).squeeze().cpu().numpy()
                            
                        #     true_combined_unscaled = unscale_predictions(true_combined, dataset.season_mean + dataset.trend_mean, 1.0)
                        #     pred_combined_unscaled = unscale_predictions(pred_combined, dataset.season_mean + dataset.trend_mean, 1.0)
                        probs = torch.sigmoid(logits)
                        preds = (probs >= threshold).float()
                        # Flatten to point-level
                        all_labels.extend(label.cpu().numpy().flatten())
                        all_probs.extend(probs.cpu().numpy().flatten())
                        all_preds.extend(preds.cpu().numpy().flatten())
                        
        #             backcast_combined = (trend_input + season_input).squeeze().cpu().numpy()
        #             true_forecast_combined = (trend_target + season_target).squeeze().cpu().numpy()
        #             pred_forecast_combined = (trend_pred + season_pred).squeeze().cpu().numpy()

        #             # Unscale all if needed
        #             backcast_combined_unscaled = unscale_predictions(backcast_combined, dataset.trend_mean + dataset.season_mean, 1.0)
        #             true_forecast_combined_unscaled = unscale_predictions(true_forecast_combined, dataset.trend_mean + dataset.season_mean, 1.0)
        #             pred_forecast_combined_unscaled = unscale_predictions(pred_forecast_combined, dataset.trend_mean + dataset.season_mean, 1.0)
                    
        #             building_forecast_key = f"{building_id}_forecast_{sample_plot_count}"

        #             if building_forecast_key in selected_buildings:
        #                 building_forecast_data.append((
        #                     building_id, sample_plot_count,
        #                     backcast_combined_unscaled,
        #                     true_forecast_combined_unscaled,
        #                     pred_forecast_combined_unscaled
        #                 ))

        #             if sample_plot_count < 10:
        #                 plot_dir = os.path.join(results_path, "plots")
        #                 plot_forecast_with_context(
        #                     backcast=backcast_combined_unscaled,
        #                     forecast_true=true_forecast_combined_unscaled,
        #                     forecast_pred=pred_forecast_combined_unscaled,
        #                     building_id=building_id,
        #                     idx=sample_plot_count,
        #                     save_path=plot_dir
        #                 )
        #                 sample_plot_count += 1
                        
        #         # Calculate average validation loss and RMSE
        #         y_true_combine_trend = np.concatenate(y_true_trend, axis=0)
        #         y_true_combine_seasonal = np.concatenate(y_true_seasonal, axis=0)
        #         y_pred_combine_trend = np.concatenate(y_pred_trend, axis=0)
        #         y_pred_combine_seasonal = np.concatenate(y_pred_seasonal, axis=0)
        #         avg_test_loss = np.mean(test_losses)

        #         y_pred_combine = y_pred_combine_seasonal + y_pred_combine_trend
        #         y_true_combine = y_true_combine_seasonal + y_true_combine_trend
                
        #         y_true_combine_trend_unscaled = unscale_predictions(y_true_combine_trend, dataset.trend_mean, dataset.trend_std)
        #         y_pred_combine_trend_unscaled = unscale_predictions(y_pred_combine_trend, dataset.trend_mean, dataset.trend_std)
        #         y_true_combine_seasonal_unscaled = unscale_predictions(y_true_combine_seasonal, dataset.season_mean, dataset.season_std)
        #         y_pred_combine_seasonal_unscaled = unscale_predictions(y_pred_combine_seasonal, dataset.season_mean, dataset.season_std)

        #         y_pred_combine_unscaled = y_pred_combine_seasonal_unscaled + y_pred_combine_trend_unscaled
        #         y_true_combine_unscaled = y_true_combine_seasonal_unscaled + y_true_combine_trend_unscaled

                
        #         # Calculate CVRMSE, NRMSE, MAE on unscaled data
        #         cvrmse = cal_cvrmse(y_pred_combine_unscaled, y_true_combine_unscaled)
        #         nrmse = cal_nrmse(y_pred_combine_unscaled, y_true_combine_unscaled)
        #         mae = cal_mae(y_pred_combine_unscaled, y_true_combine_unscaled)
        #         mse = cal_mse(y_pred_combine_unscaled, y_true_combine_unscaled)
        #         mae_norm = cal_mae(y_pred_combine, y_true_combine)
        #         mse_norm = cal_mse(y_pred_combine, y_true_combine)

        #         res.append([building_id, cvrmse, nrmse, mae, mae_norm, mse, mse_norm, avg_test_loss])

        #     # Plot final ICML-style 2x3 grid
        # if len(building_forecast_data) == 6:
        #     plot_forecast_grid(
        #         building_forecast_data=building_forecast_data,
        #         save_path=result_path,
        #         filename='forecast_icml_grid.png'
        #     )

                all_labels = np.array(all_labels)
                all_preds = np.array(all_preds)
                all_probs = np.array(all_probs)
                avg_test_loss = np.mean(test_losses)
                acc = accuracy_score(all_labels, all_preds)
                prec = precision_score(all_labels, all_preds, zero_division=0)
                rec = recall_score(all_labels, all_preds, zero_division=0)
                f1 = f1_score(all_labels, all_preds, zero_division=0)
                if len(np.unique(all_labels)) > 1:
                    auc = roc_auc_score(all_labels, all_probs)
                else:
                    auc = float('nan')
                tn, fp, fn, tp = confusion_matrix(all_labels, all_preds, labels=[0, 1]).ravel()

                res.append([building_id, acc, prec, rec, f1, auc, tp, fp, tn, fn, avg_test_loss])

        # columns = ['building_ID', 'CVRMSE', 'NRMSE', 'MAE', 'MAE_NORM', 'MSE', 'MSE_NORM', 'Avg_Test_Loss']
        # df = pd.DataFrame(res, columns=columns)
        # df.to_csv("{}/{}.csv".format(results_path, 'result'), index=False)

        columns = ['building_ID', 'Accuracy', 'Precision', 'Recall', 'F1', 'AUC_ROC', 'TP', 'FP', 'TN', 'FN', 'Avg_Test_Loss']
        df_res = pd.DataFrame(res, columns=columns)
        df_res.to_csv("{}/{}.csv".format(results_path, 'result'), index=False)

        # med_nrmse = df['NRMSE'].median()
        # med_mae = df['MAE'].median()
        # med_mae_norm = df['MAE_NORM'].median()
        # med_mse = df['MSE'].median()
        # med_mse_norm = df['MSE_NORM'].median()

        med_acc = df_res['Accuracy'].median()
        med_prec = df_res['Precision'].median()
        med_rec = df_res['Recall'].median()
        med_f1 = df_res['F1'].median()
        med_auc = df_res['AUC_ROC'].median()

        # median_res.append([region, med_nrmse, med_mae, med_mae_norm, med_mse, med_mse_norm])

        all_region_res.append([region, med_acc, med_prec, med_rec, med_f1, med_auc])

#     #med_columns = ['Dataset','NRMSE', 'MAE', 'MAE_NORM', 'MSE', 'MSE_NORM']
#     median_df = pd.DataFrame(median_res, columns=med_columns)
#    # median_df.to_csv("{}/{}.csv".format(result_path, 'median_results_of_buildings'), index=False)

    med_columns = ['Dataset', 'Accuracy', 'Precision', 'Recall', 'F1', 'AUC_ROC']
    median_df = pd.DataFrame(all_region_res, columns=med_columns)
    median_df.to_csv("{}/{}.csv".format(result_path, 'median_results_of_buildings'), index=False)


# In[44]:


if __name__ == '__main__':

  
    config_file = "./configs/W_LSTMix.json"
    with open(config_file, 'r') as f:
        args = json.load(f)

    # check device 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define W_LSTMIx model
    model = W_LSTMix.Model(
        device=device,
        num_blocks_per_stack=args['num_blocks_per_stack'],
        #forecast_length=args['forecast_length'],
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

    model_load_path = '{}/best_model.pth'.format(args['finetuned_model_save_path'])
    model.load_state_dict(torch.load(model_load_path, weights_only=True))

    # # Define loss
    # if args['loss'] == 'mse':
    #     criterion = torch.nn.MSELoss()
    # else:
    #     criterion = torch.nn.HuberLoss(reduction="mean", delta=1)

    criterion = torch.nn.BCEWithLogitsLoss()

    # training the model and save best parameters
    test(args, model, criterion, device)


# In[ ]:




