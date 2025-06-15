import torch
import torch.nn as nn

from torch import Tensor, FloatTensor
from typing import Unpack, TypedDict, cast
from diffusers.models.autoencoders.vae import DecoderOutput
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL

from .utils import image2norm, norm2image


class ConditionArgs(TypedDict):
    y: Tensor


class AutoEncoder(nn.Module):
    scale_factor: Tensor

    def __init__(self, *args, repo: str, scale: float, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.vae = AutoencoderKL.from_pretrained(repo)
        self.register_buffer("scale_factor", torch.tensor(scale))
        self.vae.requires_grad_(False)

    @torch.inference_mode()
    def forward(self, x: Tensor) -> Tensor:
        return self.encode(x)

    @torch.inference_mode()
    def encode(self, x: Tensor) -> Tensor:
        x_data = image2norm(x)
        x_latent_dist = self.vae.encode(x_data)
        x_latent_dist = cast(AutoencoderKLOutput, x_latent_dist).latent_dist
        x_latent: Tensor = x_latent_dist.sample()
        x_latent = x_latent * self.scale_factor
        return x_latent

    @torch.inference_mode()
    def decode(self, x: Tensor) -> Tensor:
        x_latent = x * self.scale_factor**-1.0
        x_latent = cast(FloatTensor, x_latent)
        x_latent = self.vae.decode(x_latent)
        x_data = cast(DecoderOutput, x_latent).sample
        x_data = norm2image(x_data)
        return x_data

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device


class TimeEmbedding(nn.Module):
    def __init__(self, *args, dim_timesteps: int, emb_dim: int, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.mlp = nn.Sequential(
            nn.Linear(dim_timesteps, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(self, t: Tensor) -> Tensor:
        return self.mlp(t)


class ResidualBlock(nn.Module):
    def __init__(
        self,
        *args,
        dim: int,
        n_layers: int,
        dim_hidden: int,
        time_emb_dim: int,
        num_groups: int = 8,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.n_layers = n_layers
        self.time_proj = nn.Linear(time_emb_dim, dim_hidden)
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            in_dim = dim if i == 0 else dim_hidden
            out_dim = dim_hidden
            block = nn.Sequential(nn.Linear(in_dim, out_dim), nn.GroupNorm(num_groups, out_dim), nn.SiLU())
            self.layers.append(block)
        self.out = nn.Linear(dim_hidden, dim)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i == self.n_layers // 2:
                h = h + self.time_proj(t_emb)
        return x + self.out(h)


class FlowModel(nn.Module):
    def __init__(
        self,
        *args,
        dim_in: int = 64,
        dim_hidden: int = 128,
        time_emb_dim: int = 128,
        n_blocks: int = 4,
        n_layers: int = 2,
        num_groups: int = 8,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.time_embedding = TimeEmbedding(dim_timesteps=1, emb_dim=time_emb_dim)

        self.blocks = nn.ModuleList(
            [
                ResidualBlock(
                    dim=dim_in,
                    n_layers=n_layers,
                    num_groups=num_groups,
                    dim_hidden=dim_hidden,
                    time_emb_dim=time_emb_dim,
                )
                for _ in range(n_blocks)
            ]
        )

        self.out = nn.Sequential(
            nn.Linear(dim_in, dim_hidden),
            nn.SiLU(),
            nn.Linear(dim_hidden, dim_in),
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        **cond: Unpack[ConditionArgs],
    ) -> torch.Tensor:
        t_emb = self.time_embedding(t)
        h = x
        for block in self.blocks:
            h = block(h, t_emb)
        return self.out(h)
