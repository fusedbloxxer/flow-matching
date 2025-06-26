from typing import cast

import torch.nn as nn

from diffusers.models.autoencoders.autoencoder_dc import AutoencoderDC
from diffusers.models.autoencoders.vae import DecoderOutput, EncoderOutput
from torch import Tensor


class AutoEncoder(nn.Module):
    def __init__(self, *args, id: str, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.dc_ae = AutoencoderDC.from_pretrained(id)
        self.dc_ae.requires_grad_(False)
        self.dc_ae.eval()

    def encode(self, x: Tensor) -> Tensor:
        x_enc = self.dc_ae.encode(x)
        x_lat = cast(EncoderOutput, x_enc).latent
        return x_lat

    def decode(self, x: Tensor) -> Tensor:
        x_dec = self.dc_ae.decode(x)
        x_out = cast(DecoderOutput, x_dec).sample
        return x_out


__all__ = ["AutoEncoder"]
