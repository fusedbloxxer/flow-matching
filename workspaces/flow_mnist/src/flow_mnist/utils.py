import matplotlib.pyplot as plt

from typing import Tuple
from einops import rearrange
from torch import Tensor
from torchvision.utils import make_grid


def image2norm(x: Tensor) -> Tensor:
    return x * 2.0 - 1.0


def norm2image(x: Tensor) -> Tensor:
    return (x * 0.5 + 0.5).clip(0.0, 1.0)


def plot_batch(x: Tensor, nrow: int=5, padding: int=5, figsize: Tuple | None=None, **kwargs):
    x_grid = make_grid(x, nrow=nrow, padding=padding, **kwargs)
    x_grid = rearrange(x_grid, "c h w -> h w c")
    plt.figure(figsize=figsize)
    plt.imshow(x_grid, cmap="gray")
    plt.show()
