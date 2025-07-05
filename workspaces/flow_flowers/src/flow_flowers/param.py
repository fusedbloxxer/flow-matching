from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Dict, Optional

from box import Box
from cyclopts import Group, Parameter, validators


log_group = Group(name="Logging")
data_group = Group(name="Dataset")
train_group = Group(name="Training")
ckpt_group = Group(name="Checkpoint")


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
    lr: Annotated[Optional[float], Parameter(name=["--lr"], help="Learning rate for training", group=train_group)] = None
    seed: Annotated[Optional[int], Parameter(name=["--seed", "-s"], help="Random seed for reproducibility", group=log_group)] = None
    steps: Annotated[Optional[int], Parameter(name=["--steps", "-s"], help="The number of steps for training", group=train_group)] = None
    epochs: Annotated[Optional[int], Parameter(name=["--epochs", "-e"], help="The number of epochs for training", group=train_group)] = None
    batch_size: Annotated[Optional[int], Parameter(name=["--batch-size", "-b"], help="The batch size for training", group=train_group)] = None

    run_name: Annotated[Optional[str], Parameter(name=["--run-name"], help="The name of the MLflow run", group=log_group)] = None
    exp_name: Annotated[Optional[str], Parameter(name=["--exp-name"], help="Experiment name for MLflow tracking", group=log_group)] = None
    log_every: Annotated[Optional[int], Parameter(name=["--log-every"], help="Log metrics every N steps/epochs", group=log_group)] = None

    ckpt_dir: Annotated[Optional[Path], Parameter(name=["--ckpt-dir"], help="The directory to save checkpoints", group=ckpt_group)] = None
    ckpt_name: Annotated[Optional[str], Parameter(name=["--ckpt-name"], help="The name of the checkpoint file", group=ckpt_group)] = None
    ckpt_every: Annotated[Optional[int], Parameter(name=["--ckpt-every"], help="Save checkpoint every N steps/epochs", group=ckpt_group)] = None
    resume_from: Annotated[Optional[str], Parameter(name=["--resume"], help="Resume training from a checkpoint file", group=ckpt_group)] = None

    augment: Annotated[Optional[bool], Parameter(name=["--augment"], help="Enable data augmentation", group=data_group)] = None
    crop_size: Annotated[Optional[int], Parameter(name=["--crop-size"], help="Size to crop images to", group=data_group)] = None
    crop_type: Annotated[Optional[str], Parameter(name=["--crop-type"], help="Type of cropping (random, center)", group=data_group)] = None

    def get_cli_cfg(self) -> Dict:
        if self.steps is not None and self.epochs is not None:
            raise ValueError("Cannot specify both steps and epochs")

        cli_cfg = Box(default_box=True)
        if self.lr is not None:
            cli_cfg.train.params.lr = self.lr
        if self.seed is not None:
            cli_cfg.train.params.seed = self.seed
        if self.steps is not None:
            cli_cfg.train.params.steps = self.steps
        if self.epochs is not None:
            cli_cfg.train.params.epochs = self.epochs
        if self.batch_size is not None:
            cli_cfg.train.params.batch_size = self.batch_size

        if self.crop_size is not None:
            cli_cfg.data.preprocess.size = self.crop_size
        if self.crop_type is not None:
            cli_cfg.data.preprocess.crop = self.crop_type
        if self.augment is not None:
            cli_cfg.data.preprocess.augment = self.augment

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
        if self.resume_from is not None:
            cli_cfg.train.ckpt.resume = self.resume_from
        return cli_cfg.to_dict()


__all__ = ["CommonParam", "ServerParam", "TrainParam"]
