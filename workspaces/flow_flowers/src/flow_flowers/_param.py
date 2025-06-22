from pathlib import Path
from cyclopts import Parameter
from typing import Annotated, Optional
from dataclasses import dataclass, field


@Parameter("*")
@dataclass(kw_only=True)
class CommonParam:
    config_path: Annotated[Path, Parameter(name=["--config", "-c"], help="Path to the YAML file")] = Path(".")


@Parameter("*")
@dataclass(kw_only=True)
class ServerParam(CommonParam):
    store: Annotated[Optional[Path], Parameter(name=["--store", "-s"], help="The path to the server storage directory")] = None
    port: Annotated[Optional[int], Parameter(name=["--port", "-p"], help="The port for the server")] = None
    host: Annotated[Optional[str], Parameter(name=["--host", "-h"], help="The host for the server")] = None


@Parameter("*")
@dataclass(kw_only=True)
class TrainParam(CommonParam):
    epochs: Annotated[Optional[int], Parameter(name=["--epochs", "-e"], help="The number of epochs for training")] = None


__all__ = ["CommonParam", "ServerParam", "TrainParam"]
