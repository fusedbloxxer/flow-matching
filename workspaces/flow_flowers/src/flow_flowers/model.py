from typing import cast

import torch
import torch.nn as nn

from diffusers.models.autoencoders.autoencoder_dc import AutoencoderDC
from diffusers.models.autoencoders.vae import DecoderOutput, EncoderOutput
from torch import Tensor


class AutoEncoder(nn.Module):
    std: Tensor
    mu: Tensor

    def __init__(self, *args, id: str, mu: float = 0.0, std: float = 1.0, **kwargs) -> None:
        super().__init__(*args, **kwargs)

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


__all__ = ["AutoEncoder"]
