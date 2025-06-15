from typing import cast
from pathlib import Path
from dataclasses import dataclass
from omegaconf import OmegaConf, SCMode


@dataclass
class DataConfig:
    path: Path


@dataclass
class PathConfig:
    storage: Path
    root: Path


@dataclass
class VAEConfig:
    scale: float
    repo: str


@dataclass
class FlowConfig:
    dim_in: int
    dim_hidden: int
    time_emb_dim: int
    n_blocks: int
    n_layers: int
    num_groups: int


@dataclass
class ModelConfig:
    vae: VAEConfig
    flow: FlowConfig


@dataclass
class ServerConfig:
    path: Path
    host: str
    port: int
    uri: str


@dataclass
class LogParamConfig:
    step_interval: int
    sample_steps: int
    batch_size: int


@dataclass
class LogConfig:
    param: LogParamConfig
    server: ServerConfig


@dataclass
class TrainConfig:
    batch_size: int
    steps: int
    lr: float


@dataclass
class Config:
    train: TrainConfig
    model: ModelConfig
    path: PathConfig
    data: DataConfig
    log: LogConfig


def load_config(path: str | Path) -> Config:
    schema = OmegaConf.structured(Config)
    config = OmegaConf.load(path)
    config = OmegaConf.merge(schema, config)
    config = OmegaConf.to_container(config, structured_config_mode=SCMode.INSTANTIATE)
    return cast(Config, config)
