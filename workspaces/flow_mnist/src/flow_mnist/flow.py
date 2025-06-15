import abc
import torch

from einops import rearrange
from dataclasses import dataclass
from typing import Callable, Tuple, Any, cast

from torch import Tensor
from torch.utils.data import Dataset


@dataclass
class Sampleable[T](abc.ABC):
    @abc.abstractmethod
    def sample(self, num_samples: int) -> T:
        raise NotImplementedError()


@dataclass
class BatchSampleable(Sampleable[Tensor]):
    batch: Tensor

    def sample(self, num_samples: int) -> Tensor:
        assert num_samples <= self.batch.size(0)
        return self.batch[:num_samples]


@dataclass
class NormalSampleable(Sampleable[Tensor]):
    size: Tuple[int, ...]

    def sample(self, num_samples: int) -> Tensor:
        return torch.randn((num_samples, *self.size))


@dataclass
class MNISTSampleable(Sampleable[Tuple[Tensor, Tensor]]):
    dataset: Dataset[Tuple[Tensor, Tensor]]

    def sample(self, num_samples: int) -> Tuple[Tensor, Tensor]:
        size = len(cast(Any, self.dataset))
        index = torch.randperm(size).tolist()
        index = index[:num_samples]
        x_pairs = [self.dataset[i] for i in index]
        x_label = [y for _, y in x_pairs]
        x_data = [x for x, _ in x_pairs]
        x_data = rearrange(x_data, "b c h w -> b c h w")
        x_label = rearrange(x_label, "b -> b")
        return x_data, x_label


@dataclass
class MNISTImageSampleable(Sampleable[Tensor]):
    dataset: Dataset[Tuple[Tensor, Tensor]]

    def __post_init__(self) -> None:
        self.sampleable = MNISTSampleable(self.dataset)

    def sample(self, num_samples: int) -> Tensor:
        return self.sampleable.sample(num_samples)[0]


@dataclass
class LambdaSampleable(Sampleable[Tensor]):
    sampleable: Sampleable
    transform: Callable[[Tensor], Tensor]

    def sample(self, num_samples: int) -> Tensor:
        samples = self.sampleable.sample(num_samples=num_samples)
        samples = self.transform(samples)
        return samples


@dataclass
class FlowPath(abc.ABC):
    @abc.abstractmethod
    def flow(self, x_0: Tensor, x_1: Tensor, t: Tensor) -> Tensor:
        raise NotImplementedError()

    @abc.abstractmethod
    def cond_vector_field(self, x_1: Tensor, x_t: Tensor, t: Tensor) -> Tensor:
        raise NotImplementedError()


@dataclass
class OTFlowPath(FlowPath):
    def flow(self, x_0: Tensor, x_1: Tensor, t: Tensor) -> Tensor:
        return t * x_1 + (1.0 - t) * x_0

    def cond_vector_field(self, x_1: Tensor, x_t: Tensor, t: Tensor) -> Tensor:
        return (x_1 - x_t) / (1.0 - t)


@dataclass
class ProbPath(FlowPath):
    p_init: Sampleable[Tensor]
    p_data: Sampleable[Tensor]
    f_path: FlowPath

    def flow(self, x_0: Tensor, x_1: Tensor, t: Tensor) -> Tensor:
        return self.f_path.flow(x_0=x_0, x_1=x_1, t=t)

    def cond_prob_path(self, x_1: Tensor, t: Tensor) -> Tensor:
        B = x_1.size(0)
        x_0 = self.p_init.sample(B).to(t.device)
        x_t = self.f_path.flow(x_0=x_0, x_1=x_1, t=t)
        return x_t

    def marg_prob_path(self, t: Tensor) -> Tensor:
        B = t.size(0)
        x_1 = self.p_data.sample(B).to(t.device)
        x_t = self.cond_prob_path(x_1=x_1, t=t)
        return x_t

    def cond_vector_field(self, x_1: Tensor, x_t: Tensor, t: Tensor) -> Tensor:
        return self.f_path.cond_vector_field(x_1=x_1, x_t=x_t, t=t)
