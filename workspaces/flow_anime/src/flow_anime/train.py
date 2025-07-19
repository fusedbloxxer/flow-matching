import random

import wandb

from .args import TrainArgs


def train(args: TrainArgs) -> None:
    # Start a new wandb run to track this script.
    run = wandb.init(
        project=args.track.project_name,
        entity=args.track.entity_name,
        dir=args.track.dir,
        config={
            "learning_rate": 0.02,
            "architecture": "CNN",
            "dataset": "CIFAR-100",
            "epochs": 10,
        },
    )

    # Simulate training.
    epochs = 10
    offset = random.random() / 5
    for epoch in range(2, epochs):
        acc = 1 - 2**-epoch - random.random() / epoch - offset
        loss = 2**-epoch + random.random() / epoch + offset

        # Log metrics to wandb.
        run.log({"acc": acc, "loss": loss})

    # Finish the run and upload any remaining data.
    run.finish()
