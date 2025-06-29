import os
import random

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy.typing as npt
import torch

from box import Box
from einops import rearrange
from torch import Tensor


@dataclass
class MeanVariance:
    """Inspired from: https://notmatthancock.github.io/2017/03/23/simple-batch-stat-updates.html"""

    def __post_init__(self) -> None:
        self.obs_: Tensor = torch.tensor(0, dtype=torch.int32)
        self.mean_: Tensor = torch.tensor(0.0)
        self.var_: Tensor = torch.tensor(0.0)

    def update(self, x: Tensor) -> None:
        n = torch.tensor(1 if x.ndim == 0 else x.size(0), dtype=torch.int32, device=x.device)

        if self.obs_ == 0:
            self.mean_ = torch.mean(x)
            self.var_ = torch.var(x)
            self.obs_ = n
            return
        m = self.obs_

        mu_n = torch.mean(x)
        mu_m = self.mean_
        mu = (m * mu_m + n * mu_n) * (m + n) ** -1.0

        var_n = torch.var(x)
        var_m = self.var_
        var = m * var_m / (m + n) + n * var_n / (m + n) + m * n * (mu_m - mu_n) ** 2 / (m + n) ** 2

        self.obs_ += n
        self.mean_ = mu
        self.var_ = var

    def compute(self) -> Dict[str, Tensor]:
        box = Box()
        if self.obs_ == 0:
            box.mean = torch.tensor(torch.nan)
            box.std = torch.tensor(torch.nan)
            box.var = torch.tensor(torch.nan)
        else:
            box.std = torch.sqrt(self.var_)
            box.mean = self.mean_
            box.var = self.var_
        return box.to_dict()


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
