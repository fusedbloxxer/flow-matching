from typing import cast
from pathlib import Path
from dataclasses import dataclass
from omegaconf import OmegaConf, SCMode


@dataclass
class PathConfig:
    store: Path


@dataclass
class BaseConfig:
    path: PathConfig


@dataclass
class ServerConfig:
    store: Path
    name: str
    host: str
    port: int
    uri: str


@dataclass
class LoggingConfig:
    server: ServerConfig


@dataclass
class Config:
    log: LoggingConfig
    base: BaseConfig


def load_config(config_path: Path) -> Config:
    base_type = OmegaConf.structured(Config)
    base_conf = OmegaConf.load(config_path)
    conf_dict = OmegaConf.merge(base_type, base_conf)
    conf_data = OmegaConf.to_container(conf_dict, structured_config_mode=SCMode.INSTANTIATE)
    conf_data = cast(Config, conf_data)
    return conf_data


__all__ = ["load_config"]
