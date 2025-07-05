from typing import Optional, TypedDict, Unpack, cast

import torch
import torch.nn as nn

from diffusers.models.autoencoders.autoencoder_dc import AutoencoderDC
from diffusers.models.autoencoders.vae import DecoderOutput, EncoderOutput
from einops import rearrange
from einops.layers.torch import Rearrange
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
    def __init__(self, *, n_class: int, emb_dim: int, **kwargs) -> None:
        super().__init__()
        self.pad_idx = 0
        self.num_emb = n_class + 1
        self.embd_proj = nn.Embedding(self.num_emb, emb_dim, self.pad_idx)

    def forward(self, x: Tensor) -> Tensor:
        return rearrange(self.embd_proj(x), "b h -> b h 1 1")


class TimestepEmbedder(nn.Module):
    def __init__(self, *, t_dim: int, emb_dim: int, **kwargs) -> None:
        super().__init__()
        self.time_proj = MLP(in_dim=t_dim, out_dim=emb_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.time_proj(x)


class MLP(nn.Module):
    def __init__(self, *, in_dim: Optional[int] = None, h_dim: Optional[int] = None, out_dim: Optional[int] = None, layers: int = 1, kernel: int = 1, **kwargs) -> None:
        super().__init__()

        if h_dim is not None and in_dim is None and out_dim is None:
            in_dim = out_dim = h_dim
        if in_dim is None or out_dim is None:
            raise ValueError("in_dim and out_dim must be provided when h_dim is not set")
        if kernel % 2 == 0:
            raise ValueError("kernel size must be odd")
        padding = kernel // 2

        if layers == 1:
            self.mlp = nn.Sequential(
                *[
                    nn.Conv2d(in_dim, out_dim, kernel_size=kernel, padding=padding),
                    nn.SiLU(),
                ]
            )
            return

        assert h_dim is not None, "h_dim must be provided when layers > 1"
        self.mlp = nn.Sequential()

        for i in range(layers):
            if i == 0:
                self.mlp.extend(
                    [
                        nn.Conv2d(in_dim, h_dim, kernel_size=kernel, padding=padding),
                        nn.SiLU(),
                    ]
                )
                continue
            if i == layers - 1:
                self.mlp.extend(
                    [
                        nn.Conv2d(h_dim, out_dim, kernel_size=kernel, padding=padding),
                        nn.SiLU(),
                    ]
                )
                continue
            self.mlp.extend(
                [
                    nn.Conv2d(h_dim, h_dim, kernel_size=kernel, padding=padding),
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
        self.sigmod = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        gap_features: Tensor = self.gap(x)
        attn_weights: Tensor = self.sigmod(self.conv(gap_features))
        return x * attn_weights


class ConvModule(nn.Module):
    def __init__(self, *, channels: int, **kwargs) -> None:
        super().__init__()
        self.conv_p1 = nn.Conv2d(channels, channels, kernel_size=1)
        self.conv_d = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.cca = CompactChannelAttn(channels=channels)
        self.silu = nn.SiLU()
        self.conv_p2 = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        h: Tensor = x
        h = self.conv_p1(x)
        h = self.conv_d(x)
        h = self.silu(h)
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
                nn.Conv2d(h_dim, 6 * h_dim, kernel_size=1),
                Rearrange("b (n c) h w -> b n c h w", n=6),
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
        return x * (1 + scale) + shift


class CondInput(TypedDict):
    t: Tensor
    y: Tensor


class DiCoBlock(nn.Module):
    def __init__(self, *, h_dim: int, h_size: int, w_size: int, mlp_layers: int, **kwargs) -> None:
        super().__init__()
        self.adaLN = AdaLN(h_dim=h_dim)
        self.norm_msa = nn.LayerNorm([h_dim, h_size, w_size])
        self.conv_mod = ConvModule(channels=h_dim)
        self.norm_mlp = nn.LayerNorm([h_dim, h_size, w_size])
        self.feed_fwd = MLP(h_dim=h_dim, layers=mlp_layers, kernel=1)

    def forward(self, x: Tensor, **cond: Unpack[CondInput]) -> Tensor:
        adaLN = self.adaLN.forward(cond["t"] + cond["y"])

        # MSA
        h_msa = AdaLN.modulate(self.norm_msa(x), shift=adaLN["msa"]["shift"], scale=adaLN["msa"]["scale"])
        h_msa = self.conv_mod(x)
        h_msa = x + adaLN["msa"]["gate"] * h_msa

        # MLP
        h_mlp = AdaLN.modulate(self.norm_mlp(h_msa), shift=adaLN["mlp"]["shift"], scale=adaLN["mlp"]["scale"])
        h_mlp = self.feed_fwd(h_mlp)
        h_mlp = h_msa + adaLN["mlp"]["gate"] * h_mlp

        return h_mlp


class DiCo(nn.Module):
    def __init__(self, *, in_dim: int, h_dim: int, out_dim: int, h_size: int, w_size: int, mlp_layers: int, blocks: int, **kwargs) -> None:
        super().__init__()
        self.in_proj = nn.Conv2d(in_dim, h_dim, kernel_size=3, padding=1)
        self.layers = nn.ModuleList([DiCoBlock(h_dim=h_dim, h_size=h_size, w_size=w_size, mlp_layers=mlp_layers) for _ in range(blocks)])
        self.out_proj = nn.Sequential(
            *[
                nn.LayerNorm([h_dim, h_size, w_size]),
                nn.Conv2d(h_dim, out_dim, kernel_size=3, padding=1),
            ]
        )

    def initialize_weights(self) -> None:
        def _basic_init(module: nn.Module) -> None:
            if not isinstance(module, nn.Conv2d):
                return
            if module.bias is not None:
                nn.init.zeros_(module.bias)
            nn.init.xavier_uniform_(module.weight)

        def _adaLN_init(module: nn.Module) -> None:
            if not isinstance(module, nn.Conv2d):
                return
            if module.bias is not None:
                nn.init.zeros_(module.bias)
            nn.init.zeros_(module.weight)

        # Init layers with uniform xavier
        self.apply(_basic_init)

        # Init AdaLN-Zero
        for module in self.modules():
            if isinstance(module, AdaLN):
                module.apply(_adaLN_init)

    def forward(self, x_t: Tensor, **cond: Unpack[CondInput]) -> Tensor:
        h_t: Tensor = self.in_proj(x_t)

        for layer in self.layers:
            h_t = layer(h_t, **cond)

        h_t = self.out_proj(h_t)
        return h_t


__all__ = ["AutoEncoder", "DiCo"]
