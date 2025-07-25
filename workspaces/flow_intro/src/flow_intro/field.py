import abc

from dataclasses import dataclass

from torch import Tensor

from .flow import LinearSchedule
from .net import MLP


class VectorField(abc.ABC):
    @abc.abstractmethod
    def drift(self, x_t: Tensor, t: Tensor) -> Tensor:
        pass


@dataclass(kw_only=True)
class CondOTVectorField(VectorField):
    z: Tensor

    def __post_init__(self) -> None:
        self.schedule = LinearSchedule()

    def drift(self, x_t: Tensor, t: Tensor) -> Tensor:
        return (self.z - x_t) / (1 - t)


@dataclass(kw_only=True)
class CondGaussVectorField(VectorField):
    z: Tensor

    def __post_init__(self) -> None:
        self.schedule = LinearSchedule()

    def drift(self, x_t: Tensor, t: Tensor) -> Tensor:
        alpha_dt = self.schedule.alpha_dt(t)
        beta_dt = self.schedule.beta_dt(t)

        alpha_t = self.schedule.alpha(t)
        beta_t = self.schedule.beta(t)

        x_data = (alpha_dt - beta_dt / beta_t * alpha_t) * self.z
        x_init = beta_dt / beta_t * x_t
        u_xtz = x_data + x_init

        return u_xtz


@dataclass(kw_only=True)
class MLPVectorField(VectorField):
    model: MLP

    def drift(self, x_t: Tensor, t: Tensor) -> Tensor:
        return self.model.forward(x_t, t)
