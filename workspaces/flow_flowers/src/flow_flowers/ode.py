from dataclasses import dataclass
from typing import Optional, cast

import einx
import torch
import torch.nn as nn

from einops import rearrange, repeat
from torch import Tensor
from torchdiffeq import odeint


@dataclass(kw_only=True, frozen=True)
class ODE:
    pad_idx: int
    u_theta: nn.Module

    @torch.inference_mode()
    def sample(
        self,
        *,
        x_t: Tensor,
        t: Tensor,
        y: Tensor,
        w: Optional[float] = None,
        **kwargs,
    ) -> Tensor:
        def ode_cnd(t: Tensor, x_t: Tensor) -> Tensor:
            t = repeat(t, " -> b 1 1 1", b=x_t.size(0))
            return self.u_theta(x_t=x_t, t=t, y=y)

        def ode_cfg(t: Tensor, x_t: Tensor) -> Tensor:
            t = rearrange(t, " -> 1 1 1 1")
            x_t_cu = repeat(x_t, "b c h w -> (2 b) c h w")

            y_c = y
            y_u = torch.full_like(y_c, self.pad_idx, device=y.device)
            y_cu = einx.rearrange("b ..., b ... -> (b + b) ...", y_c, y_u, b=y.size(0))

            v_pred_cu = self.u_theta(x_t=x_t_cu, t=t, y=y_cu)
            v_pred_c, v_pred_u = einx.rearrange("(b + b) ... -> b ..., b ...", v_pred_cu)
            v_pred = v_pred_u + w * (v_pred_c - v_pred_u)

            return v_pred

        if w is None:
            func = ode_cnd
        else:
            func = ode_cfg

        x_f = odeint(func, x_t, t, method="euler", **kwargs)
        x_f = cast(Tensor, x_f)
        return x_f
