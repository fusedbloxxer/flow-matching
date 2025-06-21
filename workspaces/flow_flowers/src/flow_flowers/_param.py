import os

from pathlib import Path
from typing import Annotated
from cyclopts import Parameter
from dataclasses import dataclass, field

from ._config import load_config


@Parameter("*")
@dataclass(kw_only=True)
class CommonParam:
    config_path: Annotated[Path, Parameter(name=["--config", "-c"])] = field(default=Path("."))
    "The path to the configuration YAML file"

    def __post_init__(self) -> None:
        assert self.config_path.exists(), "config_file not found at {}".format(self.config_path)
        assert self.config_path.is_file(), "config_file is not a file: {}".format(self.config_path)

        # Load and parse the configuration file
        self.config = load_config(self.config_path)

        # Change path to the configuration directory
        os.chdir(self.config_path.absolute().parent)


__all__ = ["CommonParam"]
