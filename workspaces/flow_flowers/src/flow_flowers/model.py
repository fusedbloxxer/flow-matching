from typing import TypedDict, cast

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
        assert layers >= 1, "MLP must have at least one layer"
        assert layers > 1 or h_dim != out_dim, "MLP must have h_dim == out_dim when layers == 1"

        if layers == 1:
            self.mlp = nn.Sequential(
                *[
                    nn.Linear(in_dim, out_dim),
                    nn.SiLU(),
                ]
            )
            return

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
    def __init__(self, *, channels: int, **kwargs) -> None:
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.conv = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        gap_features: Tensor = self.gap(x)
        attn_weights: Tensor = F.sigmoid(self.conv(gap_features))
        return x * attn_weights


class ConvModule(nn.Module):
    def __init__(self, *, channels: int, **kwargs) -> None:
        super().__init__()
        self.conv_p1 = nn.Conv2d(channels, channels, kernel_size=1)
        self.conv_d = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.cca = CompactChannelAttn(channels=channels)
        self.conv_p2 = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        h: Tensor = x
        h = self.conv_p1(x)
        h = self.conv_d(x)
        h = F.silu(h)
        h = self.cca(x)
        h = self.conv_p2(x)
        return h


class ModulateOutput(TypedDict):
    scale: Tensor
    shift: Tensor
    gate: Tensor


class AdaLNOutput(TypedDict):
    msa: ModulateOutput
    mlp: ModulateOutput


class AdaLN(nn.Module):
    def __init__(self, *, h_dim: int, **kwargs) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            *[
                nn.SiLU(),
                nn.Linear(h_dim, 6 * h_dim),
                nn.Unflatten(dim=1, unflattened_size=(6, h_dim)),
            ]
        )

    def forward(self, x: Tensor) -> AdaLNOutput:
        h = self.mlp(x)
        h = torch.unbind(h, dim=1)
        shift_msa, scale_msa, gate_msa = h[:3]
        shift_mlp, scale_mlp, gate_mlp = h[3:]
        msa: ModulateOutput = {"shift": shift_msa, "scale": scale_msa, "gate": gate_msa}
        mlp: ModulateOutput = {"shift": shift_mlp, "scale": scale_mlp, "gate": gate_mlp}
        return {"msa": msa, "mlp": mlp}

    @staticmethod
    def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiCo(nn.Module):
    def __init__(self, *, in_dim: int, d_dim: int, out_dim: int, **kwargs) -> None:
        super().__init__()

        self.in_proj = nn.Conv2d(in_dim, d_dim, kernel_size=3, stride=1, padding=1)

    def forward(self, x_t: Tensor, t: Tensor, y: Tensor) -> Tensor:
        return torch.zeros()


__all__ = ["AutoEncoder"]
