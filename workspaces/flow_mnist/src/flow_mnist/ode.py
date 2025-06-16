import einx
import torch
import torch.nn as nn

from typing import cast
from torch import Tensor
from torchdiffeq import odeint
from dataclasses import dataclass

from .model import FlowModel


@dataclass(kw_only=True)
class ODE:
    model: FlowModel
    method: str = "euler"

    def __post_init__(self) -> None:
        self.pad_idx = self.model.class_embedding.padding_idx

    def sample_trajectory(self, x_0: Tensor, t: Tensor, y: Tensor | None = None, w: float | None = None) -> Tensor:
        @torch.inference_mode()
        def ode_uncond(t: Tensor, x_t: Tensor) -> Tensor:
            t = t.repeat(x_t.size(0), 1)
            v_pred = self.model.forward(x=x_t, t=t, y=None)
            return v_pred

        @torch.inference_mode()
        def ode_cond(t: Tensor, x_t: Tensor) -> Tensor:
            assert y is not None
            t = t.repeat(x_t.size(0), 1)
            v_pred = self.model.forward(x=x_t, t=t, y=y)
            return v_pred

        @torch.inference_mode()
        def ode_cfg(t: Tensor, x_t: Tensor) -> Tensor:
            assert y is not None
            assert w is not None

            t = t.repeat(x_t.size(0), 1)
            t = cast(Tensor, einx.rearrange("b ... -> (2 b) ...", t))
            x_t = cast(Tensor, einx.rearrange("b ... -> (2 b) ...", x_t))

            y_c = y
            y_u = torch.full_like(y_c, fill_value=self.pad_idx, device=y_c.device)
            y_cu = cast(Tensor, einx.rearrange("b ..., b ... -> (b + b) ...", y_c, y_u))

            v_pred = self.model.forward(x=x_t, t=t, y=y_cu)
            v_c, v_u = einx.rearrange('(b + b) ... -> b ..., b ...', v_pred)
            v_pred = v_u + w * (v_c - v_u)

            return v_pred

        match (y, w):
            case (None, None):
                ode_func = ode_uncond
            case (c, None) if c is not None:
                ode_func = ode_cond
            case (c, w) if c is not None and w is not None:
                ode_func = ode_cfg
            case _:
                raise ValueError("Invalid combination of y and cfg")

        x_t = odeint(ode_func, x_0, t, method=self.method)
        x_t = cast(Tensor, x_t)
        return x_t

    def sample(self, x_0: Tensor, t: Tensor, y: Tensor | None = None, w: float | None = None) -> Tensor:
        return self.sample_trajectory(x_0=x_0, t=t, y=y, w=w)[-1]
