import tyro

from .eval import eval
from .sample import sample
from .server import server
from .train import train
from .workflow import workflow


tyro.extras.subcommand_cli_from_dict(
    {
        "eval": eval,
        "sample": sample,
        "server": server,
        "train": train,
        "workflow": workflow,
    },
)
