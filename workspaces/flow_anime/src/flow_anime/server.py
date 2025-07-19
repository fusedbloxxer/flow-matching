import pathlib
import signal
import sys

import docker
import docker.errors

from .args import ServerArgs, WandbArgs


def server(args: ServerArgs) -> None:
    match args.server_args:
        case WandbArgs() as wandb_args:
            wandb(wandb_args)


def wandb(args: WandbArgs) -> None:
    client = docker.from_env()

    try:
        client.images.get(args.image)
    except docker.errors.ImageNotFound:
        client.images.pull(args.image)

    host_path = pathlib.Path(args.path)
    host_path.mkdir(parents=True, exist_ok=True)
    host_path = str(host_path.absolute())

    try:
        volume = client.volumes.get(args.volume)
    except docker.errors.NotFound:
        volume = client.volumes.create(
            name=args.volume,
            driver="local",
            driver_opts={
                "device": host_path,
                "type": "none",
                "o": "bind",
            },
        )

    try:
        container = client.containers.get(args.container)
        container.remove(force=True)
        container.wait(condition="removed")
    except docker.errors.NotFound:
        pass

    container = client.containers.create(
        volumes={f"{volume.name}": {"bind": "/vol", "mode": "rw"}},
        ports={"8080/tcp": args.port},
        name=args.container,
        image=args.image,
        auto_remove=True,
        detach=True,
    )

    def clean_on_exit() -> None:
        try:
            container.remove(force=True)
            container.wait(condition="removed")
        finally:
            sys.exit(0)

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, lambda *_: clean_on_exit())

    try:
        container.start()
        container.wait()
    finally:
        clean_on_exit()


__all__ = ["server"]
