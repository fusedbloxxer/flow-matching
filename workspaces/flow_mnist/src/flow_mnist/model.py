import torch
import torch.nn as nn

from typing import cast
from torch import Tensor, FloatTensor
from diffusers.models.autoencoders.vae import DecoderOutput
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL

from .utils import image2norm, norm2image


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
