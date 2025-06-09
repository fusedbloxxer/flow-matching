import abc

from torch import Tensor
from dataclasses import dataclass

from .field import VectorField
from .path import GaussProbPath


class ScoreMatching(abc.ABC):
    @abc.abstractmethod
    def diffusion(self, x_t: Tensor, t: Tensor) -> Tensor:
        raise NotImplementedError()


@dataclass(kw_only=True)
class ScoreMatchingFromVectorField(ScoreMatching):
    prob_path: GaussProbPath
    vector_field: VectorField

    def diffusion(self, x_t: Tensor, t: Tensor) -> Tensor:
        u_t = self.vector_field.drift(x_t=x_t, t=t)

        beta_t = self.prob_path.schedule.beta(t)
        alpha_t = self.prob_path.schedule.alpha(t)

        beta_dt = self.prob_path.schedule.beta_dt(t)
        alpha_dt = self.prob_path.schedule.alpha_dt(t)

        s_t_0 = alpha_t * u_t - alpha_dt * x_t
        s_t_1 = beta_t**2 * alpha_dt - alpha_t * beta_dt * beta_t
        s_t = s_t_0 / s_t_1

        return s_t


class DiffusionModel(VectorField, ScoreMatching):
    pass


@dataclass(kw_only=True)
class DenoisingDiffusionModel(DiffusionModel):
    prob_path: GaussProbPath
    vector_field: VectorField

    def __post_init__(self) -> None:
        self.score_matching = ScoreMatchingFromVectorField(
            vector_field=self.vector_field,
            prob_path=self.prob_path,
        )

    def diffusion(self, x_t: Tensor, t: Tensor) -> Tensor:
        return self.score_matching.diffusion(x_t=x_t, t=t)

    def drift(self, x_t: Tensor, t: Tensor) -> Tensor:
        return self.vector_field.drift(x_t=x_t, t=t)
