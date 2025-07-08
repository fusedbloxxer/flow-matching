from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Dict, Optional

from box import Box
from cyclopts import Group, Parameter, validators


log_group = Group(name="Logging")
model_group = Group(name="Model")
data_group = Group(name="Dataset")
train_group = Group(name="Training")
ckpt_group = Group(name="Checkpoint")


def validate_crop_type(typ, val):
    if val is None:
        return
    if val not in ["random", "center"]:
        raise ValueError(f"Invalid value {val} for {typ}. Expected one of ['random', 'center']")


class ConfigAdapter(ABC):
    @abstractmethod
    def get_cli_cfg(self) -> Dict:
        return {}


@Parameter("*")
@dataclass
class CommonParam:
    config_path: Annotated[Path, Parameter(name=["--config", "-c"], help="Path to the YAML file", validator=validators.Path(exists=True))] = Path(".")


@Parameter("*")
@dataclass
class ServerParam(CommonParam, ConfigAdapter):
    store: Annotated[Optional[Path], Parameter(name=["--store", "-s"], help="The path to the storage directory", validator=validators.Path(exists=True))] = None
    port: Annotated[Optional[int], Parameter(name=["--port", "-p"], help="The port for the server")] = None
    host: Annotated[Optional[str], Parameter(name=["--host", "-h"], help="The host for the server")] = None

    def get_cli_cfg(self) -> Dict:
        cli_cfg = Box(default_box=True)
        if self.port is not None:
            cli_cfg.track.server.port = self.port
        if self.host is not None:
            cli_cfg.track.server.host = self.host
        if self.store is not None:
            cli_cfg.track.server.store = self.store
        return cli_cfg.to_dict()


@Parameter("*")
@dataclass
class TrainParam(CommonParam, ConfigAdapter):
    debug: Annotated[Optional[bool], Parameter(name=["--debug"], help="Enable debug mode")] = None

    lr: Annotated[Optional[float], Parameter(name=["--lr"], help="Learning rate for training", group=train_group)] = None
    seed: Annotated[Optional[int], Parameter(name=["--seed"], help="Random seed for reproducibility", group=log_group)] = None
    steps: Annotated[Optional[int], Parameter(name=["--steps", "-s"], help="The number of steps for training", group=train_group)] = None
    epochs: Annotated[Optional[int], Parameter(name=["--epochs", "-e"], help="The number of epochs for training", group=train_group)] = None
    batch_size: Annotated[Optional[int], Parameter(name=["--batch-size", "-b"], help="The batch size for training", group=train_group)] = None
    eval_every: Annotated[Optional[int], Parameter(name=["--eval-every"], help="Evaluate the model every N steps", group=train_group)] = None
    eval_split: Annotated[Optional[float], Parameter(name=["--eval-split"], help="Fraction of data to use for evaluation", group=train_group)] = None
    cfg: Annotated[Optional[float], Parameter(name=["--cfg"], help="Classifier-Free Guidance drop probability for training", group=train_group)] = None
    vae_batch_size: Annotated[Optional[int], Parameter(name=["--batch-size-vae"], help="The batch size for VAE during training", group=train_group)] = None

    n_class: Annotated[Optional[int], Parameter(name=["--num-classes"], help="Number of classes in the dataset", group=model_group)] = None
    h_dim: Annotated[Optional[int], Parameter(name=["--hidden-dim"], help="Dimension of the hidden layers in the model", group=model_group)] = None
    blocks: Annotated[Optional[int], Parameter(name=["--dico-layers"], help="Number of layers in the DiCo model", group=model_group)] = None
    mlp_layers: Annotated[Optional[int], Parameter(name=["--mlp-layers"], help="Number of layers in the feedforward layer", group=model_group)] = None
    in_dim: Annotated[Optional[int], Parameter(name=["--input-dim"], help="Input dimension of the model", group=model_group)] = None
    w_size: Annotated[Optional[int], Parameter(name=["--input-width"], help="Input width of the model", group=model_group)] = None
    h_size: Annotated[Optional[int], Parameter(name=["--input-height"], help="Input height of the model", group=model_group)] = None

    ddt: Annotated[Optional[bool], Parameter(name=["--ddt"], help="Enable Decoupled-Diffusion Transformer for training", group=train_group)] = None
    ddt_encoder: Annotated[Optional[int], Parameter(name=["--ddt-encoder"], help="Number of encoder layers for DDT", group=train_group)] = None
    ddt_decoder: Annotated[Optional[int], Parameter(name=["--ddt-decoder"], help="Number of decoder layers for DDT", group=train_group)] = None

    run_id: Annotated[Optional[str], Parameter(name=["--run-id"], help="The ID of the MLflow run", group=log_group)] = None
    run_nest: Annotated[Optional[bool], Parameter(name=["--run-nest"], help="Whether to nest the run under a parent run", group=log_group)] = None
    run_name: Annotated[Optional[str], Parameter(name=["--run-name"], help="The name of the MLflow run", group=log_group)] = None
    exp_name: Annotated[Optional[str], Parameter(name=["--exp-name"], help="Experiment name for MLflow tracking", group=log_group)] = None
    log_every: Annotated[Optional[int], Parameter(name=["--log-every"], help="Log metrics every N steps/epochs", group=log_group)] = None

    ckpt_dir: Annotated[Optional[Path], Parameter(name=["--ckpt-dir"], help="The directory to save checkpoints", group=ckpt_group)] = None
    ckpt_name: Annotated[Optional[str], Parameter(name=["--ckpt-name"], help="The name of the checkpoint file", group=ckpt_group)] = None
    ckpt_every: Annotated[Optional[int], Parameter(name=["--ckpt-every"], help="Save checkpoint every N steps/epochs", group=ckpt_group)] = None
    ckpt_resume: Annotated[Optional[bool], Parameter(name=["--ckpt-resume"], help="Resume training from checkpoint", group=ckpt_group)] = None

    augment: Annotated[Optional[bool], Parameter(name=["--augment"], help="Enable data augmentation", group=data_group)] = None
    crop_size: Annotated[Optional[int], Parameter(name=["--crop-size"], help="Size to crop images to", group=data_group)] = None
    crop_type: Annotated[Optional[str], Parameter(name=["--crop-type"], help="Type of cropping (random, center)", validator=validate_crop_type, group=data_group)] = None

    def get_cli_cfg(self) -> Dict:
        if self.steps is not None and self.epochs is not None:
            raise ValueError("Cannot specify both steps and epochs")

        cli_cfg = Box(default_box=True)
        if self.debug is not None:
            cli_cfg.base.debug = self.debug

        if self.lr is not None:
            cli_cfg.train.params.lr = self.lr
        if self.seed is not None:
            cli_cfg.base.seed = self.seed
        if self.steps is not None:
            cli_cfg.train.params.steps = self.steps
        if self.epochs is not None:
            cli_cfg.train.params.epochs = self.epochs
        if self.batch_size is not None:
            cli_cfg.train.params.batch_size = self.batch_size
        if self.eval_every is not None:
            cli_cfg.train.eval.every = self.eval_every
        if self.eval_split is not None:
            cli_cfg.train.eval.split = self.eval_split
        if self.cfg is not None:
            cli_cfg.train.params.cfg = self.cfg
        if self.vae_batch_size is not None:
            cli_cfg.train.params.vae_batch_size = self.vae_batch_size
        if self.ddt is not None:
            cli_cfg.model.ddt.active = self.ddt
        if self.ddt_encoder is not None:
            cli_cfg.model.ddt.encoder.active = self.ddt_encoder
        if self.ddt_decoder is not None:
            cli_cfg.model.ddt.decoder.active = self.ddt_decoder

        if self.n_class is not None:
            cli_cfg.model.vector_field.n_class = self.n_class
        if self.h_dim is not None:
            cli_cfg.model.vector_field.h_dim = self.h_dim
        if self.blocks is not None:
            cli_cfg.model.vector_field.blocks = self.blocks
        if self.mlp_layers is not None:
            cli_cfg.model.vector_field.mlp_layers = self.mlp_layers
        if self.in_dim is not None:
            cli_cfg.model.vector_field.in_dim = self.in_dim
        if self.w_size is not None:
            cli_cfg.model.vector_field.w_size = self.w_size
        if self.h_size is not None:
            cli_cfg.model.vector_field.h_size = self.h_size

        if self.crop_size is not None:
            cli_cfg.data.preprocess.size = self.crop_size
        if self.crop_type is not None:
            cli_cfg.data.preprocess.crop = self.crop_type
        if self.augment is not None:
            cli_cfg.data.preprocess.augment = self.augment

        if self.run_id is not None:
            cli_cfg.track.run.id = self.run_id
        if self.run_nest is not None:
            cli_cfg.track.run.nest = self.run_nest
        if self.run_name is not None:
            cli_cfg.track.run.name = self.run_name
        if self.log_every is not None:
            cli_cfg.track.run.log_every = self.log_every
        if self.exp_name is not None:
            cli_cfg.track.run.experiment = self.exp_name

        if self.ckpt_dir is not None:
            cli_cfg.train.ckpt.dir = self.ckpt_dir
        if self.ckpt_name is not None:
            cli_cfg.train.ckpt.name = self.ckpt_name
        if self.ckpt_every is not None:
            cli_cfg.train.ckpt.every = self.ckpt_every
        if self.ckpt_resume is not None:
            cli_cfg.train.ckpt.resume = self.ckpt_resume
        return cli_cfg.to_dict()


__all__ = ["CommonParam", "ServerParam", "TrainParam"]
