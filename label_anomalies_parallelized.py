import os
import argparse
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count

from my_utils.decompose_normalize import decompose_series


def label_anomalies(series, lower=1, upper=99, method='wavelet', period=24):
    trend, season = decompose_series(series, method, period=period)

    trend_out = (trend < np.percentile(trend, lower)) | (trend > np.percentile(trend, upper))
    season_out = (season < np.percentile(season, lower)) | (season > np.percentile(season, upper))

    return (trend_out | season_out).astype(np.int8)


def read_file(path):
    if path.endswith('.csv'):
        return pd.read_csv(path)
    if path.endswith('.parquet'):
        return pd.read_parquet(path)
    return None


def write_file(df, path):
    if path.endswith('.csv'):
        df.to_csv(path, index=False)
    else:
        df.to_parquet(path, index=False)


def get_series_columns(df):
    cols = []

    for col in df.columns:
        s = pd.to_numeric(df[col], errors='coerce')
        if s.notna().any():
            cols.append(col)

    return cols


def process_one_series(args):
    col, values, valid_mask, lower, upper, method, period = args

    labels = label_anomalies(
        values,
        lower=lower,
        upper=upper,
        method=method,
        period=period
    )

    n_anom = int(labels.sum())
    pct = 100 * n_anom / len(labels) if len(labels) else 0.0

    return col, valid_mask, labels, len(labels), n_anom, pct


def label_dataset(
    input_path,
    output_path,
    lower=1,
    upper=99,
    method='wavelet',
    period=24,
    dry_run=False,
    workers=None,
):
    total_files = 0
    total_series = 0

    if workers is None:
        workers = max(1, cpu_count() - 1)

    for root, _, files in os.walk(input_path):
        out_dir = os.path.join(output_path, os.path.relpath(root, input_path))

        if not dry_run:
            os.makedirs(out_dir, exist_ok=True)

        for filename in sorted(files):
            file_path = os.path.join(root, filename)
            df = read_file(file_path)

            if df is None:
                continue

            total_files += 1
            rel_path = os.path.relpath(file_path, input_path)
            print(f"\nFILE: {rel_path}")

            series_cols = get_series_columns(df)
            print(f"  -> found {len(series_cols)} series columns")

            if not series_cols:
                continue

            if dry_run:
                total_series += len(series_cols)
                continue

            jobs = []
            for col in series_cols:
                series = pd.to_numeric(df[col], errors='coerce')
                valid_mask = series.notna().to_numpy()

                if not valid_mask.any():
                    continue

                values = series.to_numpy()[valid_mask]
                jobs.append((col, values, valid_mask, lower, upper, method, period))

            if not jobs:
                continue

            with Pool(processes=workers) as pool:
                results = pool.map(process_one_series, jobs)

            out_df = df.copy()

            for col, valid_mask, labels, n_points, n_anom, pct in results:
                label_col = f'label_{col}'
                out_df[label_col] = pd.Series(pd.NA, index=df.index, dtype='Int64')
                out_df.loc[valid_mask, label_col] = labels

                total_series += 1

            out_file = os.path.join(out_dir, filename)
            write_file(out_df, out_file)

            print(f"  -> labeled {len(results)} series")

    if dry_run:
        print(f"\nDRY RUN COMPLETE -> {total_files} files scanned, {total_series} series found")
    else:
        print(f"\nDONE -> {total_files} files scanned, {total_series} series labeled")
        print(f"OUTPUT: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', default=None)
    parser.add_argument('--lower', type=float, default=1)
    parser.add_argument('--upper', type=float, default=99)
    parser.add_argument('--method', default='wavelet')
    parser.add_argument('--period', type=int, default=24)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--workers', type=int, default=None)
    args = parser.parse_args()

    output_path = args.output or args.input.rstrip('/\\') + '_labeled'

    label_dataset(
        input_path=args.input,
        output_path=output_path,
        lower=args.lower,
        upper=args.upper,
        method=args.method,
        period=args.period,
        dry_run=args.dry_run,
        workers=args.workers,
    )