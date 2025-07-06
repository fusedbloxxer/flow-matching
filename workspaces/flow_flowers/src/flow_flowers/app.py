import os

from typing import Optional

import mlflow.cli

from cyclopts import App

from .config import Config
from .param import ServerParam, TrainParam
from .train import train


app = App()
app_server = App(name="server")
app.command(app_server)


@app_server.command(name="start")
def server_start(param: Optional[ServerParam] = None) -> None:
    param = param or ServerParam()
    config = Config.init(param.config_path, param.get_cli_cfg())
    server = config.track.server
    mlflow.cli.server(
        [
            "--backend-store-uri",
            server.store,
            "--host",
            server.host,
            "--port",
            server.port,
        ]
    )


@app.command(name="train")
def train_model(param: Optional[TrainParam] = None) -> None:
    param = param or TrainParam()
    config = Config.init(param.config_path, param.get_cli_cfg())
    os.chdir(param.config_path.parent)
    train(config)


__all__ = ["app"]
