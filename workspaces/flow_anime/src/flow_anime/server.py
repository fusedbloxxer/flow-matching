import pathlib
import signal
import sys

import docker
import docker.errors

from .args import ServerArgs


def server(args: ServerArgs) -> None:
    WANDB_DOCKER_IMAGE = "wandb/local"
    WANDB_VOLUME_NAME = "wandb-flow-anime"
    WANDB_CONTAINER_NAME = "wandb-flow-anime"
    client = docker.from_env()

    try:
        client.images.get(WANDB_DOCKER_IMAGE)
    except docker.errors.ImageNotFound:
        client.images.pull(WANDB_DOCKER_IMAGE)

    host_path = pathlib.Path(args.path)
    host_path.mkdir(parents=True, exist_ok=True)
    host_path = str(host_path.absolute())

    try:
        volume = client.volumes.get(WANDB_VOLUME_NAME)
    except docker.errors.NotFound:
        volume = client.volumes.create(
            name=WANDB_VOLUME_NAME,
            driver="local",
            driver_opts={
                "device": host_path,
                "type": "none",
                "o": "bind",
            },
        )

    try:
        container = client.containers.get(WANDB_CONTAINER_NAME)
        container.remove(force=True)
        container.wait(condition="removed")
    except docker.errors.NotFound:
        pass

    container = client.containers.create(
        volumes={f"{volume.name}": {"bind": "/vol", "mode": "rw"}},
        ports={"8080/tcp": args.port},
        name=WANDB_CONTAINER_NAME,
        image=WANDB_DOCKER_IMAGE,
        auto_remove=True,
        detach=True,
    )

    def clean_on_exit() -> None:
        try:
            container.remove(force=True)
            container.wait(condition="removed")
        finally:
            sys.exit(0)

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: clean_on_exit())

    try:
        container.start()
        container.wait()
    finally:
        clean_on_exit()
