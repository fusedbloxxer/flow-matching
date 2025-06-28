from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import override

from torch import Tensor


@dataclass
class Sampleable(ABC):
    @abstractmethod
    def sample(self, batch_size: int) -> Tensor:
        raise NotImplementedError()


@dataclass
class ProbPath(ABC):
    p_init: Sampleable
    p_data: Sampleable

    @abstractmethod
    def prob_path_flow(self, *, x_0: Tensor, x_1: Tensor, t: Tensor) -> Tensor:
        raise NotImplementedError()

    @abstractmethod
    def cond_vect_field(self, *, x_1: Tensor, x_t: Tensor, t: Tensor) -> Tensor:
        raise NotImplementedError()

    def cond_prob_path(self, *, x_1: Tensor, t: Tensor) -> Tensor:
        x_0 = self.p_init.sample(t.size(0))
        x_t = self.prob_path_flow(x_0=x_0, x_1=x_1, t=t)
        return x_t

    def marg_prob_path(self, *, t: Tensor) -> Tensor:
        x_1 = self.p_data.sample(t.size(0))
        x_t = self.cond_prob_path(x_1=x_1, t=t)
        return x_t

    def loss(self, *, x_1: Tensor, x_t: Tensor, t: Tensor) -> Tensor:
        return self.cond_vect_field(x_1=x_1, x_t=x_t, t=t)


@dataclass
class OTProbPath(ProbPath):
    def cond_vect_field(self, *, x_1: Tensor, x_t: Tensor, t: Tensor) -> Tensor:
        return (x_1 - x_t) * (1.0 - t) ** -1.0

    def prob_path_flow(self, *, x_0: Tensor, x_1: Tensor, t: Tensor) -> Tensor:
        return (1.0 - t) * x_0 + t * x_1

    @override
    def loss(self, *, x_1: Tensor, x_t: Tensor, t: Tensor) -> Tensor:
        return x_1 - x_t
