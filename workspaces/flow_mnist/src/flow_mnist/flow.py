import abc
import torch

from einops import rearrange
from dataclasses import dataclass
from typing import List, Tuple, Any, cast

from torch.utils.data import Dataset
from torch import Tensor, FloatTensor

from diffusers.models.autoencoders.vae import DecoderOutput
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL

from .utils import image2norm, norm2image


@dataclass
class Sampleable(abc.ABC):
    @abc.abstractmethod
    def sample(self, samples: int) -> Tensor:
        raise NotImplementedError()


@dataclass
class GaussianSampleable(Sampleable):
    size: Tuple[int, ...]

    def sample(self, samples: int) -> Tensor:
        return torch.randn((samples, *self.size))


@dataclass
class MNISTSampleable(Sampleable):
    dataset: Dataset[Tuple[Tensor, Tensor]]

    def sample(self, samples: int) -> Tensor:
        size = len(cast(Any, self.dataset))
        index = torch.randperm(size).tolist()
        index = index[:samples]
        x_data = [self.dataset[i][0] for i in index]
        x_data = rearrange(x_data, "b c h w -> b c h w")
        return x_data


@dataclass
class LatentSampleable(Sampleable):
    dataset: Sampleable
    vae: AutoencoderKL
    scale: float

    @torch.inference_mode()
    def sample(self, samples: int) -> Tensor:
        x_data = self.dataset.sample(samples).to(self.vae.device)
        x_data_latent = self.encode(x_data)
        return x_data_latent

    @torch.inference_mode()
    def encode(self, x: Tensor) -> Tensor:
        x_data = image2norm(x)
        x_latent_dist = self.vae.encode(x_data)
        x_latent_dist = cast(AutoencoderKLOutput, x_latent_dist).latent_dist
        x_latent: Tensor = x_latent_dist.sample()
        x_latent = x_latent * self.scale
        return x_latent

    @torch.inference_mode()
    def decode(self, x: Tensor) -> Tensor:
        x_latent = x * self.scale**-1.0
        x_latent = cast(FloatTensor, x_latent)
        x_latent = self.vae.decode(x_latent)
        x_data = cast(DecoderOutput, x_latent).sample
        x_data = norm2image(x_data)
        return x_data


@dataclass
class ProbPath(abc.ABC):
    p_init: Sampleable
    p_data: Sampleable

    @abc.abstractmethod
    def cond_flow(self, x_0: Tensor, x_1: Tensor, t: Tensor) -> Tensor:
        raise NotImplementedError()

    @abc.abstractmethod
    def cond_prob_path(self, x_1: Tensor, t: Tensor) -> Tensor:
        raise NotImplementedError()

    @abc.abstractmethod
    def marg_prob_path(self, t: Tensor) -> Tensor:
        raise NotImplementedError()

    @abc.abstractmethod
    def cond_vector_field(self, x_t: Tensor, t: Tensor) -> Tensor:
        raise NotImplementedError()


@dataclass
class OTProbPath(ProbPath):
    def cond_flow(self, x_0: Tensor, x_1: Tensor, t: Tensor) -> Tensor:
        return t * x_1 + (1.0 - t) * x_0

    def cond_prob_path(self, x_1: Tensor, t: Tensor) -> Tensor:
        B = t.size(0)
        x_0 = self.p_init.sample(B).to(t.device)
        x_t = self.cond_flow(x_0=x_0, x_1=x_1, t=t)
        return x_t

    def marg_prob_path(self, t: Tensor) -> Tensor:
        B = t.size(0)
        x_1 = self.p_data.sample(B).to(t.device)
        x_t = self.cond_prob_path(x_1=x_1, t=t)
        return x_t

    def cond_vector_field(self, x_1: Tensor, x_t: Tensor, t: Tensor) -> Tensor:
        return (x_1 - x_t) / (1.0 - t)
