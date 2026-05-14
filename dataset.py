import numpy as np
import torch
from torch.utils.data import Dataset


class AnomalyDataset(Dataset):

    def __init__(self, root_dir):

        self.X_trend = np.load(
            f"{root_dir}/X_trend.npy",
            mmap_mode='r'
        )

        self.X_season = np.load(
            f"{root_dir}/X_season.npy",
            mmap_mode='r'
        )

        self.Y = np.load(
            f"{root_dir}/Y.npy",
            mmap_mode='r'
        )

    def __len__(self):
        return len(self.X_trend)

    def __getitem__(self, idx):

        return {
            "trend_input": torch.from_numpy(
                self.X_trend[idx]
            ),

            "season_input": torch.from_numpy(
                self.X_season[idx]
            ),

            "label": torch.from_numpy(
                self.Y[idx]
            )
        }
