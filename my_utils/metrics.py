import numpy as np

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

def classification_metrics(y_true, y_pred, y_probs=None):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    results = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
    }

    if y_probs is not None and len(np.unique(y_true)) > 1:
        results['auc_roc'] = roc_auc_score(y_true, np.array(y_probs))
    else:
        results['auc_roc'] = float('nan')
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        results['tp'] = int(tp)
        results['fp'] = int(fp)
        results['tn'] = int(tn)
        results['fn'] = int(fn)

    return results

def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2).sum(0) * ((pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))

def NRMSE(pred, true):
    return (RMSE(pred, true) / np.mean(true))


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    nrmse = NRMSE(pred, true)

    return mae, mse, rmse, mape, mspe, nrmse


# metrics used for evaluation
def cal_cvrmse(pred, true, eps=1e-8):
    pred = np.array(pred)
    true = np.array(true)
    return np.power(np.square(pred - true).sum() / pred.shape[0], 0.5) / (true.sum() / pred.shape[0] + eps)

def cal_mse(pred, true):
    pred = np.array(pred)
    true = np.array(true)
    return np.mean((pred - true) ** 2)

def cal_mae(pred, true):
    pred = np.array(pred)
    true = np.array(true)
    return np.mean(np.abs(pred - true))

def cal_nrmse(pred, true, eps=1e-8):
    true = np.array(true)
    pred = np.array(pred)

    M = len(true) // 24
    y_bar = np.mean(true)
    NRMSE = 100 * (1/ (y_bar+eps)) * np.sqrt((1 / (24 * M)) * np.sum((true - pred) ** 2))
    return NRMSE
