from typing import Annotated

from msgspec import Struct
from tyro.conf import arg, subcommand


class WandbArgs(Struct):
    """Arguments for tracking experiments"""

    # The name of the experiment
    exp_name: str


_WandbArgs = Annotated[WandbArgs, subcommand(name="wandb")]


class SDXLAutoEncoderArgs(Struct, tag="sdxl"):
    """Arguments for Stable Diffusion XL AutoEncoder"""


_SDXLAutoEncoderArgs = Annotated[SDXLAutoEncoderArgs, subcommand(name="sdxl")]


class DCAutoEncoderArgs(Struct, tag="dc"):
    """Arguments for Deep Compression AutoEncoder"""


_DCAutoEncoderArgs = Annotated[DCAutoEncoderArgs, subcommand(name="dc")]


class T5GemmaTextEncoderArgs(Struct, tag="t5gemma"):
    """Arguments for T5 Gemma Text Encoder"""

    # The latent size of the text encoder
    latent_size: int


_T5GemmaTextEncoderArgs = Annotated[T5GemmaTextEncoderArgs, subcommand(name="t5gemma")]


class CLIPTextEncoderArgs(Struct, tag="clip"):
    """Arguments for CLIP Text Encoder"""

    # The latent size of the text encoder
    latent_size: int


_CLIPTextEncoderArgs = Annotated[CLIPTextEncoderArgs, subcommand(name="clip")]


class T5CLIPTextEncoderArgs(Struct, tag="t5clip"):
    """Arguments for T5 and CLIP Text Encoders"""

    # T5 Gemma Text Encoder
    t5gemma: T5GemmaTextEncoderArgs

    # CLIP Text Encoder
    clip: CLIPTextEncoderArgs


_T5CLIPTextEncoderArgs = Annotated[T5CLIPTextEncoderArgs, subcommand(name="t5clip")]


class TrainArgs(Struct, kw_only=True):
    """Arguments for training"""

    # Enable verbose logging
    verbose: bool = True

    # The seed of the experiment
    seed: int = 42

    # Tracking provider for experiments
    track: _WandbArgs | None = None

    # AutoEncoder used for encoding images into latents
    ae: _SDXLAutoEncoderArgs | _DCAutoEncoderArgs

    # Text Encoders used for encoding text into latents
    te: _CLIPTextEncoderArgs | _T5GemmaTextEncoderArgs | _T5CLIPTextEncoderArgs


TrainArgs = Annotated[TrainArgs, arg(name="")]  # type: ignore


__all__ = ["TrainArgs"]
