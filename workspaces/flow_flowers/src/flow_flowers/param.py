from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Dict, Optional

from box import Box
from cyclopts import Parameter


class ConfigAdapter(ABC):
    @abstractmethod
    def get_cli_cfg(self) -> Dict:
        "Transform CLI args to config"
        return dict()


@Parameter("*")
@dataclass(kw_only=True)
class CommonParam:
    config_path: Annotated[Path, Parameter(name=["--config", "-c"], help="Path to the YAML file")] = Path(".")


@Parameter("*")
@dataclass(kw_only=True)
class ServerParam(CommonParam, ConfigAdapter):
    store: Annotated[Optional[Path], Parameter(name=["--store", "-s"], help="The path to the server storage directory")] = None
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
@dataclass(kw_only=True)
class TrainParam(CommonParam, ConfigAdapter):
    epochs: Annotated[Optional[int], Parameter(name=["--epochs", "-e"], help="The number of epochs for training")] = None

    def get_cli_cfg(self) -> Dict:
        cli_cfg = Box(default_box=True)
        if self.epochs is not None:
            cli_cfg.train.epochs = self.epochs
        return cli_cfg.to_dict()


__all__ = ["CommonParam", "ServerParam", "TrainParam"]
