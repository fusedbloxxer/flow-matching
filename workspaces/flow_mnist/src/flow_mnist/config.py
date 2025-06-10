from pathlib import Path
from typing import cast, Dict
from omegaconf import OmegaConf, SCMode
from dataclasses import dataclass, field


@dataclass
class DataConfig:
    path: Path


@dataclass
class PathConfig:
    root: Path


@dataclass
class Config:
    path: PathConfig
    data: DataConfig


def load_config(path: Path) -> Config:
    schema = OmegaConf.structured(Config)
    config = OmegaConf.load(path)
    config = OmegaConf.merge(schema, config)
    config = OmegaConf.to_container(config, structured_config_mode=SCMode.INSTANTIATE)
    return cast(Config, config)
