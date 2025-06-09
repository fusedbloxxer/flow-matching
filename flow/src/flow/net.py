import torch
import torch.nn as nn

from torch import Tensor
from einops import rearrange, repeat


class MLP(nn.Module):
    def __init__(
        self,
        *args,
        in_dim: int,
        out_dim: int,
        h_dim: int = 256,
        num_layers: int = 4,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.proj_in = nn.Sequential(
            nn.Linear(in_dim + 1, h_dim),
            nn.SiLU(),
        )

        layers = []
        for _ in range(num_layers):
            layers.append(nn.Linear(h_dim, h_dim))
            layers.append(nn.SiLU())
        self.layers = nn.ModuleList(layers)

        self.proj_out = nn.Sequential(nn.Linear(h_dim, out_dim))

    def forward(self, x_t: Tensor, t: Tensor) -> Tensor:
        if t.ndim == 1:
            t = rearrange(t, 'b -> b 1')
        if t.ndim == 0:
            t = repeat(t, '  -> b 1', b=x_t.size(0))
        x = torch.cat([x_t, t], dim=-1)
        x = self.proj_in(x)

        for layer in self.layers:
            x = layer.forward(x)
        x = self.proj_out(x)

        return x
