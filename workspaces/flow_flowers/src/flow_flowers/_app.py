from dataclasses import asdict
from typing import Optional

import matplotlib.pyplot as plt
import mlflow.cli

from cyclopts import App
from tqdm import trange

from ._config import Config
from ._data import FlowersDataset
from ._param import ServerParam, TrainParam


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
def train(param: Optional[TrainParam] = None) -> None:
    param = param or TrainParam()
    config = Config.init(param.config_path, param.get_cli_cfg())
    print(config.data)
    data = FlowersDataset(**asdict(config.data))
    image, label = data[0]

    for i in trange(len(data)):
        image, label = data[i]
        plt.imshow(image.permute((1, 2, 0)))
        plt.title(str(label.item()))
        plt.show()


__all__ = ["app"]
