import pandas as pd
import os
from collections import defaultdict


def check_dataset_hours(dataset_root="./dataset", min_hours=6480):    
    results = defaultdict(list)
    below_threshold = []

    for root, _, files in os.walk(dataset_root):
        for f in files:
            if f.endswith('.parquet') or f.endswith('.csv'):
                path = os.path.join(root, f)
                try:
                    if f.endswith('.parquet'):
                        df = pd.read_parquet(path)
                    else:
                        df = pd.read_csv(path)
                    
                    n_rows = len(df)
                    region = root.split('/')[-2]  # Get region (Commercial/Residential)
                    dataset_name = root.split('/')[-1]  # Get dataset name
                    
                    results[region].append((dataset_name, f, n_rows))
                    
                    if n_rows < min_hours:
                        months = (n_rows / 24) / 30
                        below_threshold.append((path, n_rows, months))
                except Exception as e:
                    pass

    print("Dataset Duration Summary\n")
    for region in ['Commercial', 'Residential']:
        if region in results:
            datasets = results[region]
            lengths = [r[2] for r in datasets]
            print(f"{region}:")
            print(f"  Count: {len(datasets)} datasets")
            print(f"  Min: {min(lengths):,} hours ({min(lengths)/24/30:.1f} months)")
            print(f"  Max: {max(lengths):,} hours ({max(lengths)/24/30:.1f} months)")
            print(f"  Avg: {sum(lengths)/len(lengths):.0f} hours ({sum(lengths)/len(lengths)/24/30:.1f} months)")
            print()

    if below_threshold:
        print(f"{len(below_threshold)} datasets < {min_hours} hours ({min_hours/24/30:.1f} months):\n")
        below_threshold.sort(key=lambda x: x[1])
        for path, hours, months in below_threshold:
            print(f"  {path}")
            print(f"    → {hours:,} hours = {months:.1f} months\n")
    else:
        print(f"All datasets >= {min_hours} hours!")


if __name__ == "__main__":
    check_dataset_hours()
