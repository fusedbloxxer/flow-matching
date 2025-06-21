import mlflow.cli

from cyclopts import App
from typing import Optional

from ._param import CommonParam
from ._config import load_config


app = App()

app_server = App(name="server")
app.command(app_server)


@app_server.command(name="start")
def server_start(param: Optional[CommonParam] = None) -> None:
    param = param or CommonParam()
    server = param.config.log.server
    mlflow.cli.server(["--backend-store-uri", server.store, "--host", server.host, "--port", server.port])


@app.command(name="train")
def train(param: Optional[CommonParam] = None) -> None:
    param = param or CommonParam()
    print(param.config)


__all__ = ["app"]
