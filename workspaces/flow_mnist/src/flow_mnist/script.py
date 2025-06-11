import os
import mlflow.cli

from pathlib import Path

from .config import load_config


def start_server() -> None:
    root = Path(__file__, "..", "..", "..").resolve()
    os.chdir(root)

    config = load_config("config.yaml")
    server_config = config.log.server

    mlflow.cli.server(
        [
            "--backend-store-uri",
            server_config.path,
            "--host",
            server_config.host,
            "--port",
            server_config.port,
        ]
    )
