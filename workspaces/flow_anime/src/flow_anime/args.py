from typing import Annotated, Literal

from msgspec import Struct
from tyro.conf import arg, subcommand


class WandbArgs(Struct):
    """Arguments for tracking experiments"""

    # The name of the experiment
    exp_name: str = ""

    # The name of the run
    run_name: str = ""

    # Log interval in steps
    interval: int = 1


_WandbArgs = Annotated[WandbArgs, subcommand(name="wandb")]


class DatasetArgs(Struct):
    """Arguments for dataset"""

    # The path to the dataset directory
    path: str = "data"


class DanbooruDatasetArgs(DatasetArgs):
    """Arguments for Danbooru dataset"""


_DanbooruDatasetArgs = Annotated[DanbooruDatasetArgs, arg(name="danbooru")]


class SDXLAutoEncoderArgs(Struct, tag="sdxl"):
    """Arguments for Stable Diffusion XL AutoEncoder"""


_SDXLAutoEncoderArgs = Annotated[SDXLAutoEncoderArgs, subcommand(name="sdxl")]


class DCAutoEncoderArgs(Struct, tag="dc"):
    """Arguments for Deep Compression AutoEncoder"""


_DCAutoEncoderArgs = Annotated[DCAutoEncoderArgs, subcommand(name="dc")]


class T5GemmaTextEncoderArgs(Struct, tag="t5gemma"):
    """Arguments for T5 Gemma Text Encoder"""

    # The latent size of the text encoder
    latent_size: int = 224


_T5GemmaTextEncoderArgs = Annotated[T5GemmaTextEncoderArgs, subcommand(name="t5gemma")]


class CLIPTextEncoderArgs(Struct, tag="clip"):
    """Arguments for CLIP Text Encoder"""

    # The latent size of the text encoder
    latent_size: int = 224


_CLIPTextEncoderArgs = Annotated[CLIPTextEncoderArgs, subcommand(name="clip")]


class T5CLIPTextEncoderArgs(Struct, tag="t5clip"):
    """Arguments for T5 and CLIP Text Encoders"""

    # T5 Gemma Text Encoder
    t5gemma: T5GemmaTextEncoderArgs

    # CLIP Text Encoder
    clip: CLIPTextEncoderArgs


_T5CLIPTextEncoderArgs = Annotated[T5CLIPTextEncoderArgs, subcommand(name="t5clip")]


class TrainParamsArgs(Struct):
    """Arguments for training parameters"""

    # The learning rate of the optimizer
    lr: float = 1e-4

    # The batch size of the training
    bs: int = 8

    # The number of epochs
    epochs: int | None = None

    # The number of steps
    steps: int | None = 1

    # Evaluate split ratio
    eval_split: float | None = 0.1

    # Evaluate every n steps
    eval_every: int | None = 10

    # Checkpoint name
    ckpt_name: str | None = None

    # Checkpoint directory
    ckpt_dir: str | None = None

    # Checkpoint every n steps
    ckpt_every: int | None = 10

    # Resume from checkpoint
    ckpt_resume: bool = False


_TrainParametersArgs = Annotated[TrainParamsArgs, arg(name="train")]


class TrainArgs(Struct, kw_only=True):
    """Arguments for training"""

    # The devices to be used
    device: Literal["cpu", "cuda:0", "cuda:1"] = "cuda:0"

    # Enable verbose logging
    verbose: bool = True

    # The seed of the experiment
    seed: int = 42

    # Training parameters
    train: _TrainParametersArgs

    # Dataset used for training
    data: _DanbooruDatasetArgs

    # Tracking provider for experiments
    track: _WandbArgs

    # AutoEncoder used for encoding images into latents
    ae: _SDXLAutoEncoderArgs | _DCAutoEncoderArgs

    # Text Encoders used for encoding text into latents
    te: _CLIPTextEncoderArgs | _T5GemmaTextEncoderArgs | _T5CLIPTextEncoderArgs


TrainArgs = Annotated[TrainArgs, arg(name="")]  # type: ignore


__all__ = ["TrainArgs"]
