import os

from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, cast
from omegaconf import OmegaConf, SCMode


@dataclass
class BaseConfig:
    store: Path


@dataclass
class DataConfig:
    store: Path


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
    def init(path: Path, conf: Optional[Dict] = None) -> "Config":
        base_type = OmegaConf.structured(Config)
        base_conf = OmegaConf.load(path)
        conf_dict = OmegaConf.merge(base_type, base_conf, conf)
        conf_data = OmegaConf.to_container(
            conf_dict, structured_config_mode=SCMode.INSTANTIATE, resolve=False
        )
        conf_data = cast(Config, conf_data)
        os.chdir(path.parent)
        return conf_data


__all__ = ["Config"]
