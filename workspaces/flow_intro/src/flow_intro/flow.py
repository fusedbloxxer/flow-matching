import abc
import torch

from typing import cast
from torch import Tensor
from torch.autograd.functional import jvp


class Schedule(abc.ABC):
    pass


class LinearSchedule(Schedule):
    def alpha(self, t: Tensor) -> Tensor:
        return t

    def beta(self, t: Tensor) -> Tensor:
        return 1 - t

    def alpha_dt(self, t: Tensor) -> Tensor:
        _, d_alpha = jvp(self.alpha, t, torch.ones_like(t))
        return cast(Tensor, d_alpha)

    def beta_dt(self, t: Tensor) -> Tensor:
        _, d_beta = jvp(self.beta, t, torch.ones_like(t))
        return cast(Tensor, d_beta)


class RootSchedule(Schedule):
    def alpha(self, t: Tensor) -> Tensor:
        return torch.sqrt(t)

    def beta(self, t: Tensor) -> Tensor:
        return torch.sqrt(1 - t)

    def alpha_dt(self, t: Tensor) -> Tensor:
        _, d_alpha = jvp(self.alpha, t, torch.ones_like(t))
        return cast(Tensor, d_alpha)

    def beta_dt(self, t: Tensor) -> Tensor:
        _, d_beta = jvp(self.beta, t, torch.ones_like(t))
        return cast(Tensor, d_beta)
