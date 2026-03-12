
#!/usr/bin/env python
# coding: utf-8

# In[ ]:



############ changes for different models
# 1. change the model name in import statement for model
# 2. change the config file name and change the parameters and folders path in the config file

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
#from my_utils.metrics import cal_cvrmse, cal_mae, cal_mse, cal_nrmse
from my_utils.decompose_normalize import standardize_series, decompose_series#, unscale_predictions

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix




# In[ ]:


#class DecomposedTimeSeriesDataset(Dataset):
class AnomalyDetectionDataset(Dataset):
    #def __init__(self, series, backcast_length, forecast_length, method_decom, stride=1, period=24):
    def __init__(self, series, labels, backcast_length, method_decom, stride=1, period=24):
        self.backcast_length = backcast_length
        # self.forecast_length = forecast_length
        self.stride = stride
        self.method_decom = method_decom

        # Decompose the series into trend and seasonality+residual
        trend, seasonality = decompose_series(series, method_decom, period=period)

        self.labels = labels

        # Standardize each component
        self.trend, self.trend_mean, self.trend_std = standardize_series(trend)
        self.season, self.season_mean, self.season_std = standardize_series(seasonality)

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


def test(args, model, criterion, device):

    folder_path = args['test_dataset_path']
    result_path = args['result_path']
    backcast_length = args['backcast_length']
    # forecast_length = args['forecast_length']
    stride = args['stride']
    period = 24
    method_decom = args['method_decom']
    threshold = args.get('threshold', 0.5)


    # median_res = []  
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

                labels = df['label'].values if 'label' in df.columns else np.zeros(len(energy_data))
                #dataset = DecomposedTimeSeriesDataset(energy_data, backcast_length, forecast_length, method_decom, stride, period)
                dataset = AnomalyDetectionDataset(energy_data, labels, backcast_length, method_decom, stride, period)

                # test phase
                model.eval()
                test_losses = []
                # y_true_trend = []
                # y_true_seasonal = []
                # y_pred_trend = []
                # y_pred_seasonal = []
                all_labels = []
                all_probs = []
                all_preds = []

                # test loop
                for batch in tqdm(DataLoader(dataset, batch_size=1, num_workers=4), desc=f"Testing {building_id}", leave=False):
                #for batch in tqdm(DataLoader(dataset, batch_size=64), desc=f"Testing {building_id}", leave=False):
                    trend_input = batch['trend_input'].to(device)
                    season_input = batch['season_input'].to(device)
                    # trend_target = batch['trend_target'].to(device)
                    # season_target = batch['season_target'].to(device)
                    label = batch['label'].to(device)

                    with torch.no_grad():
                        # trend_pred, season_pred = model(trend_input, season_input)
                        # loss_trend = criterion(trend_pred, trend_target)
                        # loss_season = criterion(season_pred, season_target)

                        # sum_loss = loss_trend + loss_season
                        # alpha = loss_season / sum_loss
                        # beta = loss_trend / sum_loss

                        # loss = alpha * loss_trend + beta * loss_season
                        
                        logits = model(trend_input, season_input)
                        loss = criterion(logits, label)
                        test_losses.append(loss.item())
                        
                        # # Collect true and predicted values for RMSE calculation
                        # y_true_trend.extend(trend_target.cpu().numpy())
                        # y_true_seasonal.extend(season_target.cpu().numpy())
                        # y_pred_trend.extend(trend_pred.cpu().numpy())
                        # y_pred_seasonal.extend(season_pred.cpu().numpy())

                        probs = torch.sigmoid(logits)
                        preds = (probs >= threshold).float()

                        # Flatten to point-level
                        all_labels.extend(label.cpu().numpy().flatten())
                        all_probs.extend(probs.cpu().numpy().flatten())
                        all_preds.extend(preds.cpu().numpy().flatten())
                        
                # # Calculate average validation loss and RMSE
                # y_true_combine_trend = np.concatenate(y_true_trend, axis=0)
                # y_true_combine_seasonal = np.concatenate(y_true_seasonal, axis=0)
                # y_pred_combine_trend = np.concatenate(y_pred_trend, axis=0)
                # y_pred_combine_seasonal = np.concatenate(y_pred_seasonal, axis=0)
                # avg_test_loss = np.mean(test_losses)

                # y_pred_combine = y_pred_combine_seasonal + y_pred_combine_trend
                # y_true_combine = y_true_combine_seasonal + y_true_combine_trend
                
                # y_true_combine_trend_unscaled = unscale_predictions(y_true_combine_trend, dataset.trend_mean, dataset.trend_std)
                # y_pred_combine_trend_unscaled = unscale_predictions(y_pred_combine_trend, dataset.trend_mean, dataset.trend_std)
                # y_true_combine_seasonal_unscaled = unscale_predictions(y_true_combine_seasonal, dataset.season_mean, dataset.season_std)
                # y_pred_combine_seasonal_unscaled = unscale_predictions(y_pred_combine_seasonal, dataset.season_mean, dataset.season_std)

                # y_pred_combine_unscaled = y_pred_combine_seasonal_unscaled + y_pred_combine_trend_unscaled
                # y_true_combine_unscaled = y_true_combine_seasonal_unscaled + y_true_combine_trend_unscaled

                
                # # Calculate CVRMSE, NRMSE, MAE on unscaled data
                # cvrmse = cal_cvrmse(y_pred_combine_unscaled, y_true_combine_unscaled)
                # nrmse = cal_nrmse(y_pred_combine_unscaled, y_true_combine_unscaled)
                # mae = cal_mae(y_pred_combine_unscaled, y_true_combine_unscaled)
                # mse = cal_mse(y_pred_combine_unscaled, y_true_combine_unscaled)
                # mae_norm = cal_mae(y_pred_combine, y_true_combine)
                # mse_norm = cal_mse(y_pred_combine, y_true_combine)

                # res.append([building_id, cvrmse, nrmse, mae, mae_norm, mse, mse_norm, avg_test_loss])

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

        # median_res.append([region, med_nrmse, med_mae, med_mae_norm, med_mse, med_mse_norm])

        med_acc = df_res['Accuracy'].median()
        med_prec = df_res['Precision'].median()
        med_rec = df_res['Recall'].median()
        med_f1 = df_res['F1'].median()
        med_auc = df_res['AUC_ROC'].median()
        all_region_res.append([region, med_acc, med_prec, med_rec, med_f1, med_auc])

    # med_columns = ['Dataset','NRMSE', 'MAE', 'MAE_NORM', 'MSE', 'MSE_NORM']
    # median_df = pd.DataFrame(median_res, columns=med_columns)

    med_columns = ['Dataset', 'Accuracy', 'Precision', 'Recall', 'F1', 'AUC_ROC']
    median_df = pd.DataFrame(all_region_res, columns=med_columns)
    median_df.to_csv("{}/{}.csv".format(result_path, 'median_results_of_buildings'), index=False)



# In[ ]:


if __name__ == '__main__':


    config_file = "./configs/W_LSTMix.json"
    with open(config_file, 'r') as f:
        args = json.load(f)

    # check device 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define W_LSTMix model
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

    model_load_path = '{}/best_model.pth'.format(args['model_save_path'])
    model.load_state_dict(torch.load(model_load_path, weights_only=True))

    criterion = torch.nn.BCEWithLogitsLoss()

    # Define loss
    if args['loss'] == 'mse':
        criterion = torch.nn.MSELoss()
    else:
        criterion = torch.nn.HuberLoss(reduction="mean", delta=1)


    # training the model and save best parameters
    test(args, model, criterion, device)






