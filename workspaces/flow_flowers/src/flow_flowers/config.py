from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, cast

from omegaconf import OmegaConf, SCMode


@dataclass
class BaseConfig:
    store: Path
    debug: bool
    seed: int


@dataclass
class PreprocessConfig:
    augment: bool
    norm: bool
    crop: str
    size: int


@dataclass
class DataConfig:
    path: Path
    preprocess: PreprocessConfig


@dataclass
class ServerConfig:
    store: Path
    name: str
    host: str
    port: int
    uri: str


@dataclass
class RunConfig:
    name: str
    log_every: int
    experiment: str
    id: Optional[str] = None
    nest: Optional[bool] = None


@dataclass
class TrackConfig:
    server: ServerConfig
    run: RunConfig


@dataclass
class TrainParamsConfig:
    lr: float
    cfg: float
    batch_size: int
    vae_batch_size: int
    steps: Optional[int] = None
    epochs: Optional[int] = None


@dataclass
class TrainCkptConfig:
    dir: Path
    name: str
    every: int
    resume: bool


@dataclass
class TrainEvalConfig:
    split: float
    every: int


@dataclass
class TrainConfig:
    params: TrainParamsConfig
    ckpt: TrainCkptConfig
    eval: TrainEvalConfig


@dataclass
class AutoEncoderConfig:
    std: float
    mu: float
    id: str


@dataclass
class VectorFieldConfig:
    mlp_layers: int
    n_class: int
    out_dim: int
    h_size: int
    w_size: int
    in_dim: int
    blocks: int
    h_dim: int


@dataclass
class DDTConfig:
    encoder: int
    decoder: int


@dataclass
class ContrastiveLossConfig:
    w: float


@dataclass
class ModelConfig:
    autoencoder: AutoEncoderConfig
    vector_field: VectorFieldConfig
    ddt: Optional[DDTConfig] = None
    cfm: Optional[ContrastiveLossConfig] = None


@dataclass
class Config:
    model: ModelConfig
    train: TrainConfig
    track: TrackConfig
    data: DataConfig
    base: BaseConfig

    @staticmethod
    def init(path: str | Path, conf_cli: Optional[Dict] = None) -> "Config":
        base_type = OmegaConf.structured(Config)
        conf_cli = dict() if conf_cli is None else conf_cli
        conf_base = OmegaConf.load(Path(path).absolute())
        conf_dict = OmegaConf.merge(base_type, conf_base, conf_cli)
        conf_data = OmegaConf.to_container(conf_dict, structured_config_mode=SCMode.INSTANTIATE, resolve=False)
        conf_data = cast(Config, conf_data)
        return conf_data


__all__ = ["Config"]
