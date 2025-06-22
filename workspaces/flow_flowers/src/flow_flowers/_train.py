from dataclasses import asdict

from ._config import Config
from ._data import FlowersDataset


def train(cfg: Config):
    print(cfg)
    data = FlowersDataset(**asdict(cfg.data))
    print("done")
