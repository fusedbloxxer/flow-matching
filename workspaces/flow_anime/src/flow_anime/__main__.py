import tyro

from .eval import eval
from .sample import sample
from .train import train
from .workflow import workflow


tyro.extras.subcommand_cli_from_dict(
    {
        "eval": eval,
        "sample": sample,
        "train": train,
        "workflow": workflow,
    },
)
