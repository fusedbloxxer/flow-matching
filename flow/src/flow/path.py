import abc
import torch

from typing import Tuple
from torch import Tensor
from dataclasses import dataclass

from .distrib import Sampleable
from .flow import LinearSchedule, RootSchedule
from .field import CondGaussVectorField, CondOTVectorField


class ProbPath(abc.ABC):
    @abc.abstractmethod
    def sample_init(self, n_samples: int) -> Tensor:
        raise NotImplementedError()

    @abc.abstractmethod
    def sample_data(self, n_samples: int) -> Tensor:
        raise NotImplementedError()

    @abc.abstractmethod
    def sample_path(self, x: Tensor, z: Tensor, t: Tensor) -> Tensor:
        raise NotImplementedError()

    @abc.abstractmethod
    def sample_conditional(self, z: Tensor, t: Tensor) -> Tensor:
        raise NotImplementedError()

    @abc.abstractmethod
    def sample_marginal(self, t: Tensor) -> Tensor:
        raise NotImplementedError()

    @abc.abstractmethod
    def conditional_vector_field(self, x_t: Tensor, z: Tensor, t: Tensor) -> Tensor:
        raise NotImplementedError()


@dataclass(kw_only=True)
class OTProbPath(ProbPath):
    p_init: Sampleable
    p_data: Sampleable

    def __post_init__(self) -> None:
        self.schedule = LinearSchedule()

    def sample_init(self, n_samples: int) -> Tensor:
        return self.p_init.sample((n_samples,))

    def sample_data(self, n_samples: int) -> Tensor:
        return self.p_data.sample((n_samples,))

    def sample_path(self, x: Tensor, z: Tensor, t: Tensor) -> Tensor:
        beta_t = self.schedule.beta(t)
        alpha_t = self.schedule.alpha(t)
        x_t = alpha_t * z + beta_t * x
        return x_t

    def sample_conditional(self, z: Tensor, t: Tensor) -> Tensor:
        n_samples = (t.size(0),)
        x_0 = self.p_init.sample(n_samples)
        x_t = self.sample_path(x_0, z, t)
        return x_t

    def sample_marginal(self, t: Tensor) -> Tensor:
        n_samples = (t.size(0),)
        z = self.p_data.sample(n_samples)
        x_t = self.sample_conditional(z, t)
        return x_t

    def conditional_vector_field(self, x_t: Tensor, z: Tensor, t: Tensor) -> Tensor:
        return CondOTVectorField(z=z).drift(x_t, t)


@dataclass(kw_only=True)
class GaussProbPath(ProbPath):
    p_data: Sampleable

    def __post_init__(self) -> None:
        self.schedule = RootSchedule()

    def sample_init(self, n_samples: int) -> Tensor:
        return torch.randn((n_samples, 2))

    def sample_data(self, n_samples: int) -> Tensor:
        return self.p_data.sample((n_samples,))

    def sample_path(self, x: Tensor, z: Tensor, t: Tensor) -> Tensor:
        beta_t = self.schedule.beta(t)
        alpha_t = self.schedule.alpha(t)
        x_t = alpha_t * z + beta_t * x
        return x_t

    def sample_conditional(self, z: Tensor, t: Tensor) -> Tensor:
        x_0 = torch.randn_like((z * t), device=z.device)
        x_t = self.sample_path(x_0, z, t)
        return x_t

    def sample_marginal(self, t: Tensor) -> Tensor:
        n_samples = (t.size(0),)
        z = self.p_data.sample(n_samples)
        x_t = self.sample_conditional(z, t)
        return x_t

    def conditional_vector_field(self, x_t: Tensor, z: Tensor, t: Tensor) -> Tensor:
        return CondGaussVectorField(z=z).drift(x_t, t)
