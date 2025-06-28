from pathlib import Path
from typing import List, Literal, Tuple, overload

import albumentations as A
import torch

from torch import Tensor
from torch.utils.data import Dataset
from torchvision.io import read_image

from .utils import to_numpy


class FlowersDataset(Dataset):
    def __init__(self, path: Path, transform=None) -> None:
        super().__init__()
        self.transform_ = transform
        self.labels_: Tensor = torch.load(path / "labels.pt") - 1
        self.images_: List[Path] = list(sorted((path / "images").glob("*.jpg"), key=lambda x: x.name))

    @overload
    def sample(self, batch_size: int, data: Literal["label"]) -> Tensor: ...
    @overload
    def sample(self, batch_size: int, data: Literal["image"]) -> List[Tensor]: ...
    @overload
    def sample(self, batch_size: int, data: Literal["both"]) -> Tuple[List[Tensor], Tensor]: ...
    def sample(self, batch_size: int, data: Literal["image", "label", "both"] = "both") -> Tensor | List[Tensor] | Tuple[List[Tensor], Tensor]:
        indices = torch.randperm(len(self))[:batch_size].tolist()
        samples = [self[index] for index in indices]

        y_samples = torch.stack([y for _, y in samples])
        x_samples = [x for x, _ in samples]

        match data:
            case "both":
                return x_samples, y_samples
            case "label":
                return y_samples
            case "image":
                return x_samples
            case _:
                raise ValueError(f"Invalid data type {data}")

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        image = read_image(str(self.images_[index]))

        if self.transform_:
            image = self.transform_(image=image)["image"]

        return image, self.labels_[index]

    def __len__(self) -> int:
        return self.labels_.nelement()


def get_transform(*, size: int, crop: Literal["center", "random"], augment: bool = False, norm: bool = False):
    pipe = A.Compose([A.Lambda(to_numpy), A.SmallestMaxSize(size)])

    match crop:
        case "center":
            pipe += A.CenterCrop(height=size, width=size)
        case "random":
            pipe += A.RandomCrop(height=size, width=size)
        case _:
            raise ValueError(f"Invalid crop type: {crop}")

    if augment:
        pipe += A.HorizontalFlip()

    if norm:
        pipe += A.Normalize(mean=0.5, std=0.5)

    pipe += A.ToTensorV2()
    return pipe


__all__ = ["FlowersDataset", "get_transform"]
