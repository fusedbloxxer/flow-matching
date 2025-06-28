import os
import random

from pathlib import Path

import numpy.typing as npt
import torch

from einops import rearrange
from torch import Tensor


def find_and_chdir(filename: str) -> None:
    path = Path.cwd()

    while True:
        if (path / filename).is_file():
            os.chdir(path)
            return None
        path_next = path.parent
        if path == path_next:
            raise FileNotFoundError(f"File {filename} was not found!")
        path = path_next


def set_manual_seed(seed: int) -> None:
    torch.manual_seed(seed)
    random.seed(seed)


def to_numpy(image: Tensor, **kwargs) -> npt.NDArray:
    return rearrange(image.numpy(), "c h w -> h w c")


def norm2img(image: Tensor) -> Tensor:
    image = image * 0.5 + 0.5
    image = torch.clamp(image, 0.0, 1.0)
    return image


__all__ = ["find_and_chdir", "set_manual_seed", "to_numpy", "norm2img"]
