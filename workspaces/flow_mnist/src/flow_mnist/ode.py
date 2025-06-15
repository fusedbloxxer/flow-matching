import torch
import torch.nn as nn

from typing import cast
from torch import Tensor
from torchdiffeq import odeint
from dataclasses import dataclass


@dataclass
class ODE:
    model: nn.Module

    @torch.inference_mode()
    def __call__(self, t: Tensor, x_t: Tensor) -> Tensor:
        t = t.repeat(x_t.size(0), 1)
        v_pred = self.model.forward(x=x_t, t=t)
        return v_pred

    def sample_trajectory(self, x_0: Tensor, t: Tensor) -> Tensor:
        x_t = odeint(self, x_0, t)
        x_t = cast(Tensor, x_t)
        return x_t

    def sample(self, x_0: Tensor, t: Tensor) -> Tensor:
        return self.sample_trajectory(x_0=x_0, t=t)[-1]
