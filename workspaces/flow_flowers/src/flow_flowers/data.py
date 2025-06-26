from pathlib import Path
from typing import List, Tuple

import torch
import torchvision.transforms.v2 as T

from torch import Tensor
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms.v2 import InterpolationMode as IM


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


class DataTransform:
    def __init__(self, *args, size: int, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        assert size > 0, "size must be greater than zero"

        transforms = []
        transforms.append(T.ToDtype(torch.float32, scale=True))
        transforms.append(T.Resize(size, IM.NEAREST))
        transforms.append(T.RandomCrop(size))
        self.transforms_ = T.Compose(transforms)

    def __call__(self, x: Tensor) -> Tensor:
        return self.transforms_(x)
