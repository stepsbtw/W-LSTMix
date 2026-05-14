import json
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

def to_binary_labels(values):
    labels = pd.to_numeric(values, errors='coerce').to_numpy()
    out = np.zeros(len(labels), dtype=np.float32)
    valid = ~pd.isna(labels)
    out[valid] = (labels[valid] > 0).astype(np.float32)
    return out, valid

def extract_series_with_labels(df):
    # uma serie por arquivo, com coluna 'energy'
    if 'energy' in df.columns:
        energy = pd.to_numeric(df['energy'], errors='coerce').to_numpy().astype(np.float32)
        if 'label' in df.columns:
            labels, label_valid = to_binary_labels(df['label'])
        else:
            labels = np.zeros(len(energy), dtype=np.float32)
            label_valid = np.ones(len(energy), dtype=bool)

        valid = (~np.isnan(energy)) & label_valid
        if valid.any():
            yield 'energy', energy[valid], labels[valid]
        return

    # multiplas series por arquivo, ja rotuladas
    for col in df.columns:
        label_col = f'label_{col}'
        if col.startswith('label_') or label_col not in df.columns:
            continue

        series = pd.to_numeric(df[col], errors='coerce').to_numpy().astype(np.float32)
        labels, label_valid = to_binary_labels(df[label_col])
        valid = (~np.isnan(series)) & label_valid
        if valid.any():
            yield col, series[valid], labels[valid]

def fixed_time_split(series, labels, train_len, val_len):
    total_len = len(series)

    if total_len < train_len + val_len + 1:
        half = total_len // 2
        return (series[:half], labels[:half], 
                series[half:], labels[half:], 
                series[half:], labels[half:])

    train_data = series[:train_len]
    train_labels = labels[:train_len]
    val_data = series[train_len:train_len + val_len]
    val_labels = labels[train_len:train_len + val_len]
    test_data = series[train_len:]
    test_labels = labels[train_len:]

    return train_data, train_labels, val_data, val_labels, test_data, test_labels

def process_single_file(filepath, filename, input_root, output_root, train_len, val_len, regions):
    stats = {'train': 0, 'val': 0, 'test': 0, 'skipped': 0}
    
    # Path handling to find 'Commercial' or 'Residential'
    path_parts = Path(filepath).parts
    region = next((p for p in path_parts if p in regions), 'unknown')

    if region not in regions:
        return stats

    try:
        if filename.endswith('.parquet'):
            df = pd.read_parquet(filepath, engine='pyarrow')
        else:
            df = pd.read_csv(filepath)
    except Exception as e:
        print(f"\nError reading {filename}: {e}")
        stats['skipped'] += 1
        return stats

    base_name = Path(filename).stem
    series_found = False

    for series_name, series_data, labels in extract_series_with_labels(df):
        series_found = True
        
        # Split logic
        tr_d, tr_l, val_d, val_l, te_d, te_l = fixed_time_split(series_data, labels, train_len, val_len)

        if len(tr_d) < 1:
            continue

        # Naming logic
        safe_series = series_name.replace('/', '_')
        out_name = f"{base_name}__{safe_series}.parquet" if safe_series != 'energy' else f"{base_name}.parquet"

        # Save results
        splits_to_save = [('train', tr_d, tr_l), ('val', val_d, val_l), ('test', te_d, te_l)]
        for split_type, data, lbls in splits_to_save:
            if len(data) > 0:
                save_path = os.path.join(output_root[split_type], region, out_name)
                pd.DataFrame({'energy': data, 'label': lbls}).to_parquet(save_path, index=False, engine='pyarrow')
                stats[split_type] += 1

    if not series_found:
        stats['skipped'] += 1

    return stats

def main():
    BASE_DIR = Path(__file__).resolve().parent
    CONFIG_PATH = BASE_DIR / "configs" / "W_LSTMix.json"

    with open(CONFIG_PATH, "r") as f:
        args = json.load(f)

    INPUT_DIR = args.get("raw_dataset_path", "./dataset")
    OUTPUT_DIRS = {
        'train': args.get("train_dataset_path", "./dataset_split/train"),
        'val': args.get("val_dataset_path", "./dataset_split/val"),
        'test': args.get("test_dataset_path", "./dataset_split/test"),
    }
    TRAIN_HRS = 4320
    VAL_HRS = 2160
    REGIONS = ['Commercial', 'Residential']
    NUM_WORKERS = 8 

    # data structure
    for s in ['train', 'val', 'test']:
        for r in REGIONS:
            os.makedirs(os.path.join(OUTPUT_DIRS[s], r), exist_ok=True)

    # load
    target_files = []
    for root, _, files in os.walk(INPUT_DIR):
        for f in files:
            if f.endswith(('.csv', '.parquet')):
                target_files.append((os.path.join(root, f), f))

    print(f"processing {len(target_files)} files")

    # paralelo
    total_stats = {'train': 0, 'val': 0, 'test': 0, 'skipped': 0}
    
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {
            executor.submit(
                process_single_file, fp, fn, INPUT_DIR, OUTPUT_DIRS, TRAIN_HRS, VAL_HRS, REGIONS
            ): fn for fp, fn in target_files
        }

        for i, future in enumerate(as_completed(futures), 1):
            try:
                res = future.result()
                for k in total_stats:
                    total_stats[k] += res[k]
                
                if i % 10 == 0 or i == len(target_files):
                    print(f"Progress: {i}/{len(target_files)} files processed...", end='\r')
            except Exception as e:
                print(f"\nWorker error: {e}")

    print(f"Summary: {total_stats}")

if __name__ == "__main__":
    main()
