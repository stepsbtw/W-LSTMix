# processed_dataset/
#     train/
#         X_trend.npy
#         X_season.npy
#         Y.npy
#
#     val/
#         X_trend.npy
#         X_season.npy
#         Y.npy

import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import pywt
from pathlib import Path

from statsmodels.tsa.seasonal import seasonal_decompose

# CONFIG

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "configs" / "W_LSTMix.json"

with open(CONFIG_PATH, "r") as f:
    args = json.load(f)

BACKCAST_LENGTH = args["backcast_length"]
STRIDE = args["stride"]
METHOD_DECOM = args["method_decom"]

TRAIN_INPUT = args["train_dataset_path"]
VAL_INPUT = args["val_dataset_path"]

OUTPUT_DIR = Path(args.get("processed_dataset_root"))

os.makedirs(OUTPUT_DIR, exist_ok=True)

# HELPERS

def standardize_series(series, eps=1e-8):

    mean = np.mean(series)
    std = np.std(series)

    return ((series - mean) / (std + eps)).astype(np.float32)


def decompose_series(
    series,
    method_decom,
    period=24,
    wavelet='db4',
    level=5
):

    if method_decom == "seasonal_decompose":

        result = seasonal_decompose(
            series,
            model='additive',
            period=period,
            extrapolate_trend='freq'
        )

        trend = pd.Series(result.trend).bfill().ffill().values

        seasonal_plus_resid = (
            series - trend
        )

        seasonal_plus_resid = pd.Series(
            seasonal_plus_resid
        ).fillna(0).values

        return trend.astype(np.float32), seasonal_plus_resid.astype(np.float32)

    elif method_decom == "wavelet":

        coeffs = pywt.wavedec(
            series,
            wavelet,
            level=level
        )

        trend_coeffs = [coeffs[0]] + [
            np.zeros_like(c)
            for c in coeffs[1:]
        ]

        trend = pywt.waverec(
            trend_coeffs,
            wavelet
        )[:len(series)]

        seasonal_plus_resid = (
            series - trend
        )

        return trend.astype(np.float32), seasonal_plus_resid.astype(np.float32)

    else:
        raise ValueError("Unknown decomposition method")


def to_binary_labels(values):

    labels = pd.to_numeric(
        values,
        errors='coerce'
    ).to_numpy()

    out = np.zeros(len(labels), dtype=np.float32)

    valid = ~pd.isna(labels)

    out[valid] = (
        labels[valid] > 0
    ).astype(np.float32)

    return out, valid


def extract_series_with_labels(df):

    if 'energy' in df.columns:

        energy = pd.to_numeric(
            df['energy'],
            errors='coerce'
        ).to_numpy()

        if 'label' in df.columns:
            labels, label_valid = to_binary_labels(df['label'])
        else:
            labels = np.zeros(len(energy), dtype=np.float32)
            label_valid = np.ones(len(energy), dtype=bool)

        valid = (~pd.isna(energy)) & label_valid

        if valid.any():
            yield energy[valid].astype(np.float32), labels[valid]

        return

    for col in df.columns:

        label_col = f'label_{col}'

        if col.startswith('label_'):
            continue

        if label_col not in df.columns:
            continue

        series = pd.to_numeric(
            df[col],
            errors='coerce'
        ).to_numpy()

        labels, label_valid = to_binary_labels(
            df[label_col]
        )

        valid = (~pd.isna(series)) & label_valid

        if valid.any():
            yield (
                series[valid].astype(np.float32),
                labels[valid]
            )


def load_dataframe(file_path):

    if file_path.endswith(".csv"):
        return pd.read_csv(file_path)

    elif file_path.endswith(".parquet"):
        return pd.read_parquet(file_path)

    return None


# WINDOW CREATION

def create_windows(trend, season, labels):

    X_trend = []
    X_season = []
    Y = []

    total = len(trend)

    if total < BACKCAST_LENGTH:
        return None, None, None

    for start in range(
        0,
        total - BACKCAST_LENGTH + 1,
        STRIDE
    ):

        end = start + BACKCAST_LENGTH

        X_trend.append(
            trend[start:end]
        )

        X_season.append(
            season[start:end]
        )

        Y.append(
            labels[start:end]
        )

    return (
        np.array(X_trend, dtype=np.float32),
        np.array(X_season, dtype=np.float32),
        np.array(Y, dtype=np.float32)
    )


# PROCESS SPLIT

def process_split(input_dir, split_name):

    split_dir = os.path.join(
        OUTPUT_DIR,
        split_name
    )

    os.makedirs(split_dir, exist_ok=True)

    all_X_trend = []
    all_X_season = []
    all_Y = []

    files = []

    for root, _, filenames in os.walk(input_dir):

        for filename in filenames:

            if (
                filename.endswith(".csv")
                or filename.endswith(".parquet")
            ):

                files.append(
                    os.path.join(root, filename)
                )

    print(f"\nFound {len(files)} files for {split_name}")

    for file_path in tqdm(files):

        df = load_dataframe(file_path)

        if df is None:
            continue

        for series, labels in extract_series_with_labels(df):

            if len(series) < BACKCAST_LENGTH:
                continue

            # DECOMPOSE
            trend, season = decompose_series(
                series,
                METHOD_DECOM
            )

            # STANDARDIZE
            trend = standardize_series(trend)
            season = standardize_series(season)

            # WINDOWS
           
            X_t, X_s, Y = create_windows(
                trend,
                season,
                labels
            )

            if X_t is None:
                continue

            all_X_trend.append(X_t)
            all_X_season.append(X_s)
            all_Y.append(Y)

    # CONCAT EVERYTHING

    print("\nConcatenating arrays...")

    X_trend = np.concatenate(
        all_X_trend,
        axis=0
    )

    X_season = np.concatenate(
        all_X_season,
        axis=0
    )

    Y = np.concatenate(
        all_Y,
        axis=0
    )

    print("\nFinal shapes:")
    print("X_trend:", X_trend.shape)
    print("X_season:", X_season.shape)
    print("Y:", Y.shape)

    # SAVE

    print("\nSaving arrays...")

    np.save(
        os.path.join(split_dir, "X_trend.npy"),
        X_trend
    )

    np.save(
        os.path.join(split_dir, "X_season.npy"),
        X_season
    )

    np.save(
        os.path.join(split_dir, "Y.npy"),
        Y
    )

    print(f"\nSaved {split_name} split.")


# MAIN

if __name__ == "__main__":

    process_split(
        TRAIN_INPUT,
        "train"
    )

    process_split(
        VAL_INPUT,
        "val"
    )

    print("\nDONE.")
