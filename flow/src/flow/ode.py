from torch import Tensor
from dataclasses import dataclass

from .field import VectorField


@dataclass(kw_only=True)
class ODE:
    vector_field: VectorField

    def step(self, x_t: Tensor, t: Tensor, h: Tensor) -> Tensor:
        return x_t + h * self.vector_field.drift(x_t, t)
