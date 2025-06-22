import os

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, cast

from omegaconf import OmegaConf, SCMode


@dataclass
class BaseConfig:
    store: Path


@dataclass
class DataConfig:
    path: Path


@dataclass
class ServerConfig:
    store: Path
    name: str
    host: str
    port: int
    uri: str


@dataclass
class TrackConfig:
    server: ServerConfig


@dataclass
class TrainConfig:
    epochs: int


@dataclass
class Config:
    train: TrainConfig
    track: TrackConfig
    data: DataConfig
    base: BaseConfig

    @staticmethod
    def init(path: Path, conf_cli: Optional[Dict] = None) -> "Config":
        base_type = OmegaConf.structured(Config)
        conf_cli = dict() if conf_cli is None else conf_cli
        conf_base = OmegaConf.load(path)
        conf_dict = OmegaConf.merge(base_type, conf_base, conf_cli)
        conf_data = OmegaConf.to_container(conf_dict, structured_config_mode=SCMode.INSTANTIATE, resolve=False)
        conf_data = cast(Config, conf_data)
        os.chdir(path.parent)
        return conf_data


__all__ = ["Config"]
