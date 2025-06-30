from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.models.autoencoders.autoencoder_dc import AutoencoderDC
from diffusers.models.autoencoders.vae import DecoderOutput, EncoderOutput
from torch import Tensor


class AutoEncoder(nn.Module):
    std: Tensor
    mu: Tensor

    def __init__(self, *, id: str, mu: float = 0.0, std: float = 1.0, **kwargs) -> None:
        super().__init__()

        # Used to re-scale latents
        self.register_buffer("std", torch.tensor(std))
        self.register_buffer("mu", torch.tensor(mu))

        # DC-AE from SANA paper
        self.dc_ae = AutoencoderDC.from_pretrained(id)
        self.dc_ae.requires_grad_(False)
        self.dc_ae.eval()

    def encode(self, x: Tensor) -> Tensor:
        x_enc = self.dc_ae.encode(x)
        x_lat = cast(EncoderOutput, x_enc).latent
        x_lat = (x_lat - self.mu) / self.std
        return x_lat

    def decode(self, x: Tensor) -> Tensor:
        x = (x * self.std) + self.mu
        x_dec = self.dc_ae.decode(x)
        x_out = cast(DecoderOutput, x_dec).sample
        return x_out


class LabelEmbedder(nn.Module):
    def __init__(self, *, in_dim: int, emb_dim: int, **kwargs) -> None:
        super().__init__()
        self.pad_idx = 0
        self.num_emb = in_dim + 1
        self.embd_proj = nn.Embedding(self.num_emb, emb_dim, self.pad_idx)

    def forward(self, x: Tensor) -> Tensor:
        return self.embd_proj(x)


class MLP(nn.Module):
    def __init__(self, *, in_dim: int, h_dim: int, out_dim: int, layers: int, **kwargs) -> None:
        super().__init__()
        self.mlp = nn.Sequential()

        for i in range(layers):
            if i == 0:
                self.mlp.extend(
                    [
                        nn.Linear(in_dim, h_dim),
                        nn.SiLU(),
                    ]
                )
                continue
            if i == layers - 1:
                self.mlp.extend(
                    [
                        nn.Linear(h_dim, out_dim),
                        nn.SiLU(),
                    ]
                )
                continue
            self.mlp.extend(
                [
                    nn.Linear(h_dim, h_dim),
                    nn.SiLU(),
                ]
            )

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(x)


class CompactChannelAttn(nn.Module):
    def __init__(self, *, pool_kernel_size: int, in_dim: int, out_dim: int, **kwargs) -> None:
        super().__init__()
        self.gap = nn.AvgPool2d(pool_kernel_size, stride=1)
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1, padding=0)

    def forward(self, x: Tensor) -> Tensor:
        return x * F.sigmoid(self.conv(self.gap(x)))


__all__ = ["AutoEncoder"]
