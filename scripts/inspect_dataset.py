import os
import argparse
import pandas as pd


def read_file(path):
    if path.endswith(".csv"):
        return pd.read_csv(path)
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return None


def inspect_dataframe(df, max_cols=10):
    all_cols = list(df.columns)

    time_cols = []
    numeric_cols = []
    other_cols = []

    for col in all_cols:
        name = str(col).lower()

        if "time" in name or "date" in name or "stamp" in name:
            time_cols.append(col)
            continue

        s = pd.to_numeric(df[col], errors="coerce")
        if s.notna().any():
            numeric_cols.append(col)
        else:
            other_cols.append(col)

    if "energy" in df.columns:
        format_type = "long_or_single_series"
    elif len(numeric_cols) > 1:
        format_type = "wide_multi_series"
    elif len(numeric_cols) == 1:
        format_type = "single_numeric_series"
    else:
        format_type = "unknown"

    print(f"  rows: {len(df)}")
    print(f"  cols: {len(all_cols)}")
    print(f"  format: {format_type}")
    print(f"  time/date cols: {len(time_cols)} -> {time_cols[:max_cols]}")
    print(f"  numeric series cols: {len(numeric_cols)} -> {numeric_cols[:max_cols]}")
    print(f"  other cols: {len(other_cols)} -> {other_cols[:max_cols]}")

    if numeric_cols:
        print("  sample series stats:")
        for col in numeric_cols[:max_cols]:
            s = pd.to_numeric(df[col], errors="coerce")
            valid = s.dropna()

            if len(valid) == 0:
                continue

            missing_pct = 100.0 * s.isna().sum() / len(s)

            print(
                f"    {col}: "
                f"valid={len(valid)}, "
                f"missing={missing_pct:.1f}%, "
                f"min={valid.min():.3f}, "
                f"max={valid.max():.3f}, "
                f"mean={valid.mean():.3f}"
            )


def inspect_dataset(input_path, max_files_per_folder=None):
    total_files = 0

    for root, _, files in os.walk(input_path):
        data_files = [f for f in sorted(files) if f.endswith(".csv") or f.endswith(".parquet")]

        if max_files_per_folder is not None:
            data_files = data_files[:max_files_per_folder]

        for filename in data_files:
            path = os.path.join(root, filename)
            rel_path = os.path.relpath(path, input_path)

            print(f"\nFILE: {rel_path}")

            try:
                df = read_file(path)
                if df is None:
                    print("  skipped: unsupported file")
                    continue

                inspect_dataframe(df)
                total_files += 1

            except Exception as e:
                print(f"  error: {e}")

    print(f"\nDONE -> inspected {total_files} files")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Dataset root folder")
    parser.add_argument(
        "--max-files-per-folder",
        type=int,
        default=None,
        help="Limit files inspected in each folder"
    )
    args = parser.parse_args()

    inspect_dataset(
        input_path=args.input,
        max_files_per_folder=args.max_files_per_folder
    )