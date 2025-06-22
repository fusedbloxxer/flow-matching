from pathlib import Path

import torch

from scipy.io import loadmat
from torch.utils.data import Dataset


class FlowersDataset(Dataset):
    def __init__(self, *args, path: Path, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        labels = loadmat(path / "labels.mat")["labels"]
        labels = torch.tensor(labels, dtype=torch.int32)
        print(labels.shape)

    def __len__(self) -> int:
        return 0
