import torch

from torch import Tensor


def equal_select(tensor, total) -> Tensor:
    if total <= 0:
        return torch.empty(0)
    indices = torch.linspace(0, len(tensor) - 1, steps=total).long()
    return tensor[indices]
