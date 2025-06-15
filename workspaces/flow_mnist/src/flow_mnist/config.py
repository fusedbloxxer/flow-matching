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
class ModelConfig:
    vae: VAEConfig


@dataclass
class ServerConfig:
    path: Path
    host: str
    port: int
    uri: str


@dataclass
class LogConfig:
    server: ServerConfig


@dataclass
class Config:
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
