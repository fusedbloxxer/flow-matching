from torch import Tensor


def image2norm(x: Tensor) -> Tensor:
    return x * 2.0 - 1.0


def norm2image(x: Tensor) -> Tensor:
    return 0.5 * (x + 1.0)
