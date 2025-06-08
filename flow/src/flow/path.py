import abc
import torch

from torch import Tensor
from dataclasses import dataclass

from .distrib import Sampleable
from .flow import LinearSchedule


class ProbPath(abc.ABC):
    @abc.abstractmethod
    def sample_conditional(self, z: Tensor, t: Tensor) -> Tensor:
        raise NotImplementedError()

    @abc.abstractmethod
    def sample_marginal(self, t: Tensor) -> Tensor:
        raise NotImplementedError()


@dataclass(kw_only=True)
class CondOTProbPath(ProbPath):
    p_init: Sampleable
    p_data: Sampleable

    def __post_init__(self) -> None:
        self.schedule = LinearSchedule()

    def sample_conditional(self, z: Tensor, t: Tensor) -> Tensor:
        n_samples = t.size()

        beta_t = self.schedule.beta(t)
        alpha_t = self.schedule.alpha(t)

        x_0 = self.p_init.sample(n_samples)
        x_t = alpha_t * z + beta_t * x_0

        return x_t

    def sample_marginal(self, t: Tensor) -> Tensor:
        n_samples = t.size()
        z = self.p_data.sample(n_samples)
        x_t = self.sample_conditional(z, t)
        return x_t


@dataclass(kw_only=True)
class GaussProbPath(ProbPath):
    p_data: Sampleable

    def __post_init__(self) -> None:
        self.schedule = LinearSchedule()

    def sample_conditional(self, z: Tensor, t: Tensor) -> Tensor:
        beta_t = self.schedule.beta(t)
        alpha_t = self.schedule.alpha(t)

        x_data = alpha_t * z
        x_init = beta_t * torch.randn_like(x_data)
        x_t = x_data + x_init

        return x_t

    def sample_marginal(self, t: Tensor) -> Tensor:
        n_samples = t.size()
        z = self.p_data.sample(n_samples)
        x_t = self.sample_conditional(z, t)
        return x_t
