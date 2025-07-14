from dataclasses import dataclass

import torch

from torch import Tensor

from .diffusion import DiffusionModel
from .field import VectorField


@dataclass(kw_only=True)
class ODE:
    vector_field: VectorField

    def step(self, x_t: Tensor, t: Tensor, h: Tensor) -> Tensor:
        x_t_h = x_t + h * self.vector_field.drift(x_t, t)
        return x_t_h


@dataclass(kw_only=True)
class SDE:
    sigma: Tensor
    diffusion_model: DiffusionModel

    def step(self, x_t: Tensor, t: Tensor, h: Tensor) -> Tensor:
        u_t = self.diffusion_model.drift(x_t=x_t, t=t)
        s_t = self.diffusion_model.diffusion(x_t=x_t, t=t)
        dW_t = torch.sqrt(h) * torch.randn_like(x_t, device=x_t.device)
        x_t_h = x_t + h * u_t + h * 0.5 * self.sigma**2 * s_t + self.sigma * dW_t
        return x_t_h
