from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple, cast

import torch

from torch import Tensor
from torch.utils.data import Dataset


@dataclass(kw_only=True)
class Sampleable(ABC):
    device: Optional[torch.device] = None

    @abstractmethod
    def sample(self, batch_size: int) -> Tensor:
        raise NotImplementedError()


@dataclass(kw_only=True)
class GaussianSampleable(Sampleable):
    size: Tuple

    def sample(self, batch_size: int) -> Tensor:
        x = torch.randn((batch_size, *self.size))
        if self.device:
            x = x.to(self.device)
        return x


@dataclass(kw_only=True)
class DatasetSampleable(Sampleable):
    data: Dataset
    map_entry: Optional[Callable[[Any], Tensor]] = None
    map_batch: Optional[Callable[[Tensor], Tensor]] = None

    def sample(self, batch_size: int) -> Tensor:
        size = len(cast(Any, self.data))
        indices = torch.randperm(size)[:batch_size].tolist()

        x = []
        for i in indices:
            x_i = self.data[i]
            x.append(self.map_entry(x_i) if self.map_entry else x_i)

        x = torch.stack(x, dim=0).to(self.device)
        x = self.map_batch(x) if self.map_batch else x
        return x


@dataclass(kw_only=True)
class ProbPath(ABC):
    p_init: Optional[Sampleable] = None
    p_data: Optional[Sampleable] = None

    @abstractmethod
    def prob_path_flow(self, *, x_0: Tensor, x_1: Tensor, t: Tensor) -> Tensor:
        raise NotImplementedError()

    @abstractmethod
    def cond_vect_field(self, *, x_1: Tensor, x_t: Tensor, t: Tensor) -> Tensor:
        raise NotImplementedError()

    @abstractmethod
    def target(self, *, x_1: Tensor, x_0: Tensor) -> Tensor:
        raise NotImplementedError()

    def cond_prob_path(self, *, x_1: Tensor, t: Tensor) -> Tensor:
        assert self.p_init is not None, "p_init must be specified"
        x_0 = self.p_init.sample(x_1.size(0))
        x_t = self.prob_path_flow(x_0=x_0, x_1=x_1, t=t)
        return x_t

    def marg_prob_path(self, *, n: int, t: Tensor) -> Tensor:
        assert self.p_data is not None, "p_data must be specified"
        x_1 = self.p_data.sample(n)
        x_t = self.cond_prob_path(x_1=x_1, t=t)
        return x_t


@dataclass(kw_only=True)
class OTProbPath(ProbPath):
    def target(self, *, x_1: Tensor, x_0: Tensor) -> Tensor:
        return x_1 - x_0

    def cond_vect_field(self, *, x_1: Tensor, x_t: Tensor, t: Tensor) -> Tensor:
        return (x_1 - x_t) * (1.0 - t) ** -1.0

    def prob_path_flow(self, *, x_0: Tensor, x_1: Tensor, t: Tensor) -> Tensor:
        return (1.0 - t) * x_0 + t * x_1
