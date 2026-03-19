import os
import sys
import argparse
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from my_utils.decompose_normalize import decompose_series


def label_anomalies(series, lower=1, upper=99, method='wavelet', period=24):
    trend, season = decompose_series(series, method, period=period)

    trend_out = (trend < np.percentile(trend, lower)) | (trend > np.percentile(trend, upper))
    season_out = (season < np.percentile(season, lower)) | (season > np.percentile(season, upper))

    return (trend_out | season_out).astype(int)


def label_dataset(input_path, output_path, lower=1, upper=99, method='wavelet', period=24):
    total = 0

    for region in sorted(os.listdir(input_path)):
        region_dir = os.path.join(input_path, region)
        if not os.path.isdir(region_dir):
            continue

        out_dir = os.path.join(output_path, region)
        os.makedirs(out_dir, exist_ok=True)

        for f in sorted(os.listdir(region_dir)):
            path = os.path.join(region_dir, f)

            if f.endswith('.csv'):
                df = pd.read_csv(path)
            elif f.endswith('.parquet'):
                df = pd.read_parquet(path)
            else:
                continue

            if 'energy' not in df.columns:
                continue

            df['label'] = label_anomalies(df['energy'].values, lower, upper, method, period)
            n_anom = int(df['label'].sum())
            total += 1

            print(f"  {region}/{f}: {len(df)} pts, {n_anom} anomalies ({100*n_anom/len(df):.1f}%)")

            out = os.path.join(out_dir, f)
            if f.endswith('.csv'):
                df.to_csv(out, index=False)
            else:
                df.to_parquet(out, index=False)

    print(f"\nDone — {total} files labeled → {output_path}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input',  type=str, required=True)
    p.add_argument('--output', type=str, default=None, help="Output path (default: <input>_labeled)")
    args = p.parse_args()

    output = args.output or args.input.rstrip('/') + '_labeled'

    print(f"Input: {args.input}  |  Output: {output}\n")
    label_dataset(args.input, output)
