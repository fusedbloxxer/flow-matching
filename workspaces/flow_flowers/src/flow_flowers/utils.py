import os

from pathlib import Path


def find_and_chdir(filename: str):
    path = Path.cwd()

    while True:
        if (path / filename).is_file():
            os.chdir(path)
            return None
        path_next = path.parent
        if path == path_next:
            raise FileNotFoundError(f"File {filename} was not found!")
        path = path_next


__all__ = ["find_and_chdir"]
