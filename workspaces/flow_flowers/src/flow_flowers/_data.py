from pathlib import Path
from typing import List, Tuple

import torch

from torch import Tensor
from torch.utils.data import Dataset
from torchvision.io import read_image


class FlowersDataset(Dataset):
    def __init__(self, *args, path: Path, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.labels_: Tensor = torch.load(path / "labels.pt")
        self.images_: List[Path] = list(sorted((path / "images").glob("*.jpg"), key=lambda x: x.name))

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        image = read_image(str(self.images_[index]))
        label = self.labels_[index]
        return image, label

    def __len__(self) -> int:
        return self.labels_.nelement()
