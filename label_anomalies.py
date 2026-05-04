import os
import argparse
import numpy as np
import pandas as pd

from my_utils.decompose_normalize import decompose_series


def label_anomalies(series, lower=1, upper=99, method='wavelet', period=24):
    trend, season = decompose_series(series, method, period=period)

    trend_out = (trend < np.percentile(trend, lower)) | (trend > np.percentile(trend, upper))
    season_out = (season < np.percentile(season, lower)) | (season > np.percentile(season, upper))

    return (trend_out | season_out).astype(int)


def get_series_columns(df):
    cols = []

    for col in df.columns:
        if pd.to_numeric(df[col], errors='coerce').notna().any():
            cols.append(col)

    return cols


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


def label_dataset(input_path, output_path, lower=1, upper=99, method='wavelet', period=24, dry_run=False):
    total_files = 0
    total_series = 0

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
            print(f"\nFILE: {os.path.relpath(file_path, input_path)}")

            series_cols = get_series_columns(df)
            print(f"  -> found {len(series_cols)} series columns")

            if not series_cols:
                continue

            out_df = df.copy()

            for col in series_cols:
                series = pd.to_numeric(df[col], errors='coerce')
                valid = series.notna()

                if not valid.any():
                    continue

                labels = label_anomalies(
                    series[valid].to_numpy().copy(),
                    lower=lower,
                    upper=upper,
                    method=method,
                    period=period
                )

                out_df[f'label_{col}'] = pd.Series(pd.NA, index=df.index, dtype='Int64')
                out_df.loc[valid, f'label_{col}'] = labels

                # n_anom = int(labels.sum())
                # pct = 100 * n_anom / len(labels)

                # print(f"  {col} -> {len(labels)} points, {n_anom} anomalies ({pct:.1f}%)")
                total_series += 1

            if not dry_run:
                out_file = os.path.join(out_dir, filename)
                write_file(out_df, out_file)

    if dry_run:
        print(f"\nDRY RUN COMPLETE -> {total_files} files scanned, {total_series} series would be labeled")
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
    args = parser.parse_args()

    output_path = args.output or args.input.rstrip('/\\') + '_labeled'

    label_dataset(
        input_path=args.input,
        output_path=output_path,
        lower=args.lower,
        upper=args.upper,
        method=args.method,
        period=args.period,
        dry_run=args.dry_run
    )