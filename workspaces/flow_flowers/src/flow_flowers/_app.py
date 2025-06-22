from typing import Optional

import mlflow.cli

from box import Box
from cyclopts import App

from ._config import Config
from ._param import ServerParam, TrainParam


app = App()
app_server = App(name="server")
app.command(app_server)


@app_server.command(name="start")
def server_start(param: Optional[ServerParam] = None) -> None:
    param = param or ServerParam()

    cli_cfg = Box(default_box=True)
    if param.port is not None:
        cli_cfg.track.server.port = param.port
    if param.host is not None:
        cli_cfg.track.server.host = param.host
    if param.store is not None:
        cli_cfg.track.server.store = param.store
    config = Config.init(param.config_path, cli_cfg.to_dict())

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
def train(param: Optional[TrainParam] = None) -> None:
    param = param or TrainParam()

    cli_cfg = Box(default_box=True)
    if param.epochs is not None:
        cli_cfg.train.epochs = param.epochs
    config = Config.init(param.config_path, cli_cfg.to_dict())

    print(config)


__all__ = ["app"]
