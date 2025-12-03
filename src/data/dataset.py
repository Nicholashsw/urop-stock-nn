from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class WindowDataset(Dataset):
    """
    Simple PyTorch Dataset that reads preprocessed numpy arrays.
    """

    def __init__(self, X_path: str, y_path: str):
        self.X = np.load(X_path)
        self.y = np.load(y_path)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(self.X[idx]).float()      # shape (seq_len, features)
        y = torch.tensor(self.y[idx]).float()          # scalar
        return x, y
