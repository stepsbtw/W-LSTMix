import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path


def to_binary_labels(values):
    """Convert values to binary labels (0 or 1)."""
    labels = pd.to_numeric(values, errors='coerce').to_numpy()
    out = np.zeros(len(labels), dtype=np.float32)
    valid = ~pd.isna(labels)
    out[valid] = (labels[valid] > 0).astype(np.float32)
    return out, valid


def extract_series_with_labels(df):
    """Yield (series_name, series_array, labels_array) for long or wide formats.

    Long format: expects 'energy' and optional 'label' columns -> yields ('energy', series, labels)
    Wide format: yields one tuple per data column that has a matching 'label_<col>' column.
    """
    # Long format
    if 'energy' in df.columns:
        energy = pd.to_numeric(df['energy'], errors='coerce').to_numpy()
        if 'label' in df.columns:
            labels, label_valid = to_binary_labels(df['label'])
        else:
            labels = np.zeros(len(energy), dtype=np.float32)
            label_valid = np.ones(len(energy), dtype=bool)

        valid = (~pd.isna(energy)) & label_valid
        if valid.any():
            yield 'energy', energy[valid].astype(np.float32), labels[valid]
        return

    # Wide format
    for col in df.columns:
        label_col = f'label_{col}'
        if col.startswith('label_') or label_col not in df.columns:
            continue

        series = pd.to_numeric(df[col], errors='coerce').to_numpy()
        labels, label_valid = to_binary_labels(df[label_col])
        valid = (~pd.isna(series)) & label_valid
        if valid.any():
            yield col, series[valid].astype(np.float32), labels[valid]


def fixed_time_split(series, labels, train_len=4320, val_len=2160):
    """Split series into train/val/test. Uses fixed time windows with 50/50 fallback."""
    total_len = len(series)

    if total_len < train_len + val_len + 1:
        # Fallback when data is too short: 50/50 split
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

    return train_data, train_labels, val_data, val_labels, test_data, test_labels


def split_and_save_dataset(
    input_root="./dataset",
    output_root="./dataset_split",
    train_len=4320,
    val_len=2160,
):
    """
    Read all dataset files and for each series (long or wide) save train/val/test parquet files.
    Output filenames preserve source filename and series name when applicable.
    """
    splits = ['train', 'val', 'test']
    regions = ['Commercial', 'Residential']

    print(f"Starting dataset split from {input_root} to {output_root}...")
    
    for split in splits:
        for region in regions:
            os.makedirs(os.path.join(output_root, split, region), exist_ok=True)

    stats = {'train': 0, 'val': 0, 'test': 0, 'skipped': 0}

    for root, _, files in os.walk(input_root):
        for filename in files:
            if not (filename.endswith('.csv') or filename.endswith('.parquet')):
                continue

            filepath = os.path.join(root, filename)

            # Determine region from path (Commercial or Residential)
            path_parts = filepath.split(os.sep)
            try:
                region_idx = path_parts.index('dataset') + 1
                region = path_parts[region_idx]
            except (ValueError, IndexError):
                region = 'unknown'

            if region not in regions:
                continue

            try:
                if filename.endswith('.parquet'):
                    df = pd.read_parquet(filepath)
                else:
                    df = pd.read_csv(filepath)
            except Exception as e:
                print(f"  ⚠ Error reading {filepath}: {e}")
                stats['skipped'] += 1
                continue

            base_name = filename.replace('.parquet', '').replace('.csv', '')

            # Extract series (supports long and wide formats)
            series_found = False
            for series_name, series_data, labels in extract_series_with_labels(df):
                series_found = True
                train_data, train_labels, val_data, val_labels, test_data, test_labels = fixed_time_split(
                    series_data, labels, train_len, val_len
                )

                if train_data is None or len(train_data) < 2:
                    stats['skipped'] += 1
                    continue

                # Create output filename
                safe_series = series_name.replace('/', '_') if series_name is not None else ''
                out_name = f"{base_name}__{safe_series}.parquet" if safe_series else f"{base_name}.parquet"

                # Save train
                train_df = pd.DataFrame({'energy': train_data, 'label': train_labels})
                train_df.to_parquet(os.path.join(output_root, 'train', region, out_name))
                stats['train'] += 1

                # Save val
                if len(val_data) > 1:
                    val_df = pd.DataFrame({'energy': val_data, 'label': val_labels})
                    val_df.to_parquet(os.path.join(output_root, 'val', region, out_name))
                    stats['val'] += 1

                # Save test
                if len(test_data) > 1:
                    test_df = pd.DataFrame({'energy': test_data, 'label': test_labels})
                    test_df.to_parquet(os.path.join(output_root, 'test', region, out_name))
                    stats['test'] += 1

            if not series_found:
                # No recognizable series in file
                print(f"  ⚠ No energy/label series found in {filepath}")
                stats['skipped'] += 1

    print(f"\n✅ Dataset split complete!")
    print(f"  Train splits saved: {stats['train']}")
    print(f"  Val splits saved: {stats['val']}")
    print(f"  Test splits saved: {stats['test']}")
    print(f"  Skipped: {stats['skipped']}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Split dataset into train/val/test (wide+long formats)")
    parser.add_argument("--input", default="./dataset", help="Input dataset root")
    parser.add_argument("--output", default="./dataset_split", help="Output split dataset root")
    parser.add_argument("--train-len", type=int, default=4320, help="Train split length (hours)")
    parser.add_argument("--val-len", type=int, default=2160, help="Val split length (hours)")

    args = parser.parse_args()

    split_and_save_dataset(input_root=args.input, output_root=args.output, train_len=args.train_len, val_len=args.val_len)
