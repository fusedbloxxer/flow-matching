from typing import Annotated, Literal, Tuple

from msgspec import Struct
from tyro.conf import arg, subcommand


class TrackArgs(Struct):
    """Arguments for tracking experiments"""

    # Logs location
    dir: str

    # Name of the owner entity
    entity_name: str

    # Name of the project
    project_name: str = "flow_anime"

    # Name of the experiment
    exp_name: str = ""

    # Name of the run
    run_name: str = ""

    # Log interval in steps
    interval: int = 1


_TrackArgs = Annotated[TrackArgs, subcommand(name="track")]


class DatasetArgs(Struct):
    """Arguments for Danbooru dataset"""

    # The path to the dataset directory
    path: str = "data"


_DatasetArgs = Annotated[DatasetArgs, arg(name="danbooru")]


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

    # Freeze text encoder(s) during training
    freeze_te: bool = True

    # Use cached embeddings during training
    use_cached_embeddings: bool = False

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

    # Device that will be used to train upon
    device: Literal["cpu", "cuda:0", "cuda:1"] = "cuda:0"

    # Enable verbose logging
    verbose: bool = True

    # Seed of the experiment
    seed: int = 42

    # Training parameters
    train: _TrainParametersArgs

    # Dataset used for training
    data: _DatasetArgs

    # Tracking provider for experiments
    track: _TrackArgs

    # AutoEncoder used for encoding images into latents
    ae: _SDXLAutoEncoderArgs | _DCAutoEncoderArgs

    # Text Encoders used for encoding text into latents
    te: _CLIPTextEncoderArgs | _T5GemmaTextEncoderArgs | _T5CLIPTextEncoderArgs


TrainArgs = Annotated[TrainArgs, arg(name="")]  # type: ignore


class WandbArgs(Struct, kw_only=True, tag="wandb"):
    """Arguments for wandb server"""

    # Local path for the volume
    path: str

    # Port to bind the server to
    port: int = 8080

    # Official docker image name for wandb
    image: str = "wandb/local"

    # Volume name to identify the host storage
    volume: str = "wandb-flow-anime"

    # Container name to identify the wandb instance
    container = "wandb-flow-anime"


_WandbServerArgs = Annotated[WandbArgs, subcommand(name="wandb")]


class ServerArgs(Struct, kw_only=True):
    """Arguments for servers"""

    # Type of server to run
    server_args: Annotated[_WandbServerArgs, arg(name="")]


ServerArgs = Annotated[ServerArgs, arg(name="", prefix_name=False)]  # type: ignore


class DatasetDownloadToolArgs(Struct, tag="download"):
    """Arguments for downloading datasets"""


_DatasetDownloadToolArgs = Annotated[DatasetDownloadToolArgs, subcommand(name="download")]


class DatasetConvertToolArgs(Struct, tag="convert"):
    """Arguments for dataset conversion"""

    # Path to the source dataset directory
    src: str

    # Path to the destination dataset directory
    dst: str

    # Resolution to resize images to (height, width)
    res: Tuple[int, int] = (256, 256)

    # Transformation to apply to images
    transform: Literal["resize_center_crop"] = "resize_center_crop"


_DatasetConvertToolArgs = Annotated[DatasetConvertToolArgs, subcommand(name="convert")]


class DatasetEncodeToolArgs(Struct, tag="encode"):
    """Arguments for dataset encoding"""

    # Path to the source dataset directory
    src: str

    # Path to the destination dataset directory
    dst: str

    # AutoEncoder arguments to use for encoding the dataset
    ae: _SDXLAutoEncoderArgs | _DCAutoEncoderArgs


_DatasetEncodeToolArgs = Annotated[DatasetEncodeToolArgs, subcommand(name="encode")]


class DatasetEmbedToolArgs(Struct, tag="embed"):
    """Arguments for embedding the captions in the dataset"""

    # Path to the source dataset directory
    src: str

    # Path to the destination dataset directory
    dst: str

    # Text Encoders used for encoding text into embeddings
    te: _CLIPTextEncoderArgs | _T5GemmaTextEncoderArgs | _T5CLIPTextEncoderArgs


_DatasetEmbedToolArgs = Annotated[DatasetEmbedToolArgs, subcommand(name="embed")]


class DatasetToolsArgs(Struct, kw_only=True):
    """Arguments for datasets"""

    # Arguments for the specific dataset tool to use
    tool_args: Annotated[_DatasetDownloadToolArgs | _DatasetConvertToolArgs | _DatasetEncodeToolArgs | _DatasetEmbedToolArgs, arg(name="", prefix_name=False)]


DatasetToolsArgs = Annotated[DatasetToolsArgs, arg(name="", prefix_name=False)]  # type: ignore
