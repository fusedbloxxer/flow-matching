from typing import TypedDict, Unpack, cast

import torch
import torch.nn as nn

from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.models.autoencoders.vae import DecoderOutput
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from torch import FloatTensor, Tensor

from .utils import image2norm, norm2image


class ConditionArgs(TypedDict):
    y: Tensor | None


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


class LabelEmbedding(nn.Module):
    def __init__(self, *args, dim_class: int, emb_dim: int, num_classes: int, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.num_embeddings = num_classes + 1
        self.padding_idx = num_classes

        self.embd_table = nn.Embedding(
            num_embeddings=self.num_embeddings,
            padding_idx=self.padding_idx,
            embedding_dim=dim_class,
        )

        self.mlp = nn.Sequential(
            nn.Linear(dim_class, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(self, y: Tensor) -> Tensor:
        y_embd = self.embd_table(y)
        y_proj = self.mlp(y_embd)
        return y_proj


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
        class_emb_dim: int,
        num_groups: int = 8,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.n_layers = n_layers
        self.layers = nn.ModuleList()
        self.time_proj = nn.Linear(time_emb_dim, dim_hidden)
        self.class_proj = nn.Linear(class_emb_dim, dim_hidden)
        for i in range(n_layers):
            in_dim = dim if i == 0 else dim_hidden
            out_dim = dim_hidden
            block = nn.Sequential(nn.Linear(in_dim, out_dim), nn.GroupNorm(num_groups, out_dim), nn.SiLU())
            self.layers.append(block)
        self.out = nn.Linear(dim_hidden, dim)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor, y_emb: Tensor) -> torch.Tensor:
        h: Tensor = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i == self.n_layers // 2:
                h = h * self.time_proj(t_emb) + self.class_proj(y_emb)
        return x + self.out(h)


class FlowModel(nn.Module):
    def __init__(
        self,
        *args,
        dim_in: int = 64,
        dim_hidden: int = 128,
        time_emb_dim: int = 128,
        class_emb_dim: int = 128,
        n_blocks: int = 4,
        n_layers: int = 2,
        num_groups: int = 8,
        num_classes: int = 10,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.class_embedding = LabelEmbedding(dim_class=1, emb_dim=class_emb_dim, num_classes=num_classes)
        self.time_embedding = TimeEmbedding(dim_timesteps=1, emb_dim=time_emb_dim)

        self.blocks = nn.ModuleList(
            [
                ResidualBlock(
                    dim=dim_in,
                    n_layers=n_layers,
                    num_groups=num_groups,
                    dim_hidden=dim_hidden,
                    time_emb_dim=time_emb_dim,
                    class_emb_dim=class_emb_dim,
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
        B = x.size(0)

        if "y" not in cond or cond["y"] is None:
            y = torch.full((B,), self.class_embedding.padding_idx, device=x.device)
        else:
            y = cond["y"]

        y_emb: Tensor = self.class_embedding(y)
        t_emb: Tensor = self.time_embedding(t)

        h: Tensor = x
        for block in self.blocks:
            h = block.forward(x=h, t_emb=t_emb, y_emb=y_emb)
        return self.out(h)
