import torch
import torchvision.transforms.v2 as T

from pathlib import Path
from typing import Tuple
from torch import Tensor
from torchvision.datasets import MNIST
from torch.utils.data import Dataset, DataLoader

from .utils import image2norm
from .config import DataConfig


class MNISTDataset(Dataset):
    def __init__(self, *args, path: Path, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.data_ = MNIST(root=path, download=True)
        self.transform_ = T.Compose(
            [
                T.ToImage(),
                T.ToDtype(dtype=torch.float32, scale=True),
            ]
        )

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        sample = self.data_[index]

        label: Tensor = torch.tensor(sample[1])
        image: Tensor = sample[0]

        image = self.transform_(image)
        return image, label

    def __len__(self) -> int:
        return len(self.data_)
