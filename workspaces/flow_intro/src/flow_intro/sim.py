import abc
import torch

from torch import Tensor
from einops import rearrange
from dataclasses import dataclass

from .ode import ODE
from .ode import SDE


class Simulator(abc.ABC):
    @abc.abstractmethod
    def step(self, x_t: Tensor, t: Tensor, h: Tensor) -> Tensor:
        raise NotImplementedError()

    @abc.abstractmethod
    def simulate(self, x_t: Tensor, t: Tensor) -> Tensor:
        raise NotImplementedError()

    @abc.abstractmethod
    def trajectory(self, x_t: Tensor, t: Tensor) -> Tensor:
        raise NotImplementedError()


@dataclass(kw_only=True)
class EulerSimulator(Simulator):
    ode: ODE

    def step(self, x_t: Tensor, t: Tensor, h: Tensor) -> Tensor:
        return self.ode.step(x_t=x_t, t=t, h=h)

    def simulate(self, x_t: Tensor, t: Tensor) -> Tensor:
        return self.trajectory(x_t=x_t, t=t)[-1]

    def trajectory(self, x_t: Tensor, t: Tensor) -> Tensor:
        h = torch.diff(t)
        x = [x_t]

        for step in range(h.nelement()):
            x_t = self.step(x_t, t[step], h[step])
            x.append(x_t)

        return rearrange(x, "h n d -> h n d")


@dataclass(kw_only=True)
class EulerMaruyamaSimulator(Simulator):
    sde: SDE

    def step(self, x_t: Tensor, t: Tensor, h: Tensor) -> Tensor:
        return self.sde.step(x_t=x_t, t=t, h=h)

    def simulate(self, x_t: Tensor, t: Tensor) -> Tensor:
        return self.trajectory(x_t=x_t, t=t)[-1]

    def trajectory(self, x_t: Tensor, t: Tensor) -> Tensor:
        h = torch.diff(t)
        x = [x_t]

        for step in range(h.nelement()):
            x_t = self.step(x_t, t[step], h[step])
            x.append(x_t)

        return rearrange(x, "h n d -> h n d")
