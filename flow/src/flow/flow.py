import abc
import torch

from typing import cast
from torch import Tensor
from torch.autograd.functional import jvp
from einops import rearrange


class Schedule(abc.ABC):
    pass


class LinearSchedule(Schedule):
    def alpha(self, t: Tensor) -> Tensor:
        return rearrange(t, "n -> n 1")

    def beta(self, t: Tensor) -> Tensor:
        return rearrange(1 - t, "n -> n 1")

    def alpha_dt(self, t: Tensor) -> Tensor:
        _, d_alpha = jvp(self.alpha, t, torch.ones_like(t))
        return cast(Tensor, d_alpha)

    def beta_dt(self, t: Tensor) -> Tensor:
        _, d_beta = jvp(self.beta, t, torch.ones_like(t))
        return cast(Tensor, d_beta)
