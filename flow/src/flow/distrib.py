import abc
import torch
import torch.distributions as D

from torch import Tensor
from torch.nn import Module
from typing import Tuple
from einops import rearrange, repeat


class Sampleable(abc.ABC):
    @abc.abstractmethod
    def sample(self, samples: torch.Size | Tuple) -> Tensor:
        raise NotImplementedError()


class Circles(Module, Sampleable):
    _dummy: Tensor

    def __init__(
        self,
        *args,
        n_modes: int,
        mean: Tensor | None = None,
        scale: Tensor | None = None,
        radius: float | None = None,
        offset_degree: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        # Default init
        mean = torch.zeros((2,)) if mean is None else mean
        scale = torch.ones((2,)) if scale is None else scale

        # Validate args
        assert mean.size() == (2,)
        assert scale.size() == (2,)

        if n_modes == 1:
            assert radius is None
            radius = 0
        if n_modes != 1 and radius is None:
            radius = torch.pi

        # Split a circle in n_modes then obtain (x, y) for those points
        theta = 2 * torch.pi * torch.arange(n_modes) / n_modes
        mean_x = radius * torch.cos(theta)
        mean_y = radius * torch.sin(theta)
        means = rearrange([mean_x, mean_y], "b n -> b n")

        # Obtain rotation matrix from offset_degrees
        rot_degree = torch.tensor(offset_degree)
        rot = torch.deg2rad(rot_degree)
        rot_mat = torch.tensor(
            [[torch.cos(rot), -torch.sin(rot)], [torch.sin(rot), torch.cos(rot)]]
        )

        # Rotate the points on circle
        means = rot_mat @ means
        means = (mean + means.T).T

        # Reshape data to (n_modes, 2D)
        means = rearrange(means, "d n -> n d")
        scale = repeat(scale, "d -> n d", n=n_modes)

        # Compose distribution
        com = D.Independent(D.Normal(means, scale), 1)
        mix = D.Categorical(torch.ones(n_modes))
        self._dist = D.MixtureSameFamily(mix, com)
        self.register_buffer("_dummy", torch.empty((0,)))

    def sample(self, n_samples: torch.Size | Tuple | int) -> Tensor:
        n_samples = (n_samples,) if isinstance(n_samples, int) else n_samples
        samples = self._dist.sample(n_samples)
        return samples.to(self._dummy.device)


class Checkers(Module, Sampleable):
    _dummy: Tensor

    def __init__(
        self,
        *args,
        n_patches: int,
        patch_size: float,
        patch_fill: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        # Validate args
        assert n_patches >= 0, "number of patches must be positive"
        assert patch_size > 0, "a patch size must be strictly positive"

        # Save parameters
        self.p_size_ = patch_size
        self.n_patches_ = n_patches
        self.t_patches_ = (n_patches * 2) ** 2 // 2
        self.fill_toggle_ = 0 if patch_fill else 1

        # Total number of patches per row
        self.patch_dist = D.Categorical(torch.ones(self.t_patches_))
        self.point_dist = D.Uniform(
            low=torch.zeros(2), high=torch.full((2,), patch_size)
        )
        self.register_buffer("_dummy", torch.empty((0,)))

    def sample(self, n_samples: torch.Size | Tuple | int):
        n_samples = (n_samples,) if isinstance(n_samples, int) else n_samples

        # Sample independent class and points
        cls = self.patch_dist.sample(n_samples)
        pts = self.point_dist.sample(n_samples)

        # Determine offsets
        y_offset = cls // self.n_patches_
        x_offset = (cls % self.n_patches_) * 2 + (
            (y_offset + self.fill_toggle_) % 2 != 0
        ).float()

        # Translate all points according to class
        pts[:, 0] = pts[:, 0] + x_offset * self.p_size_
        pts[:, 1] = pts[:, 1] - y_offset * self.p_size_

        # Translate all points relative to center
        pts[:, 0] = pts[:, 0] - (self.n_patches_ * self.p_size_)
        pts[:, 1] = pts[:, 1] + ((self.n_patches_ - 1) * self.p_size_)
        samples = pts

        return samples.to(self._dummy.device)
