import hashlib
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Literal, overload

import requests


@overload
def download(url: str, path: None = ..., return_bytes: Literal[True] = ...) -> bytes:
    ...


@overload
def download(url: str, path: None = ..., return_bytes: Literal[False] = ...) -> str:
    ...


@overload
def download(url: str, path: Path | str) -> None:
    ...


def download(url: str, path: str | Path | None = None, return_bytes: bool = False) -> str | bytes | None:
    with requests.get(url, stream=True) as r:
        if path is None:
            return r.content if return_bytes else r.text
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / url.split("/")[-1].split("?")[0], "wb") as f:
            shutil.copyfileobj(r.raw, f)


def get_file_name(url: str) -> str:
    return url.split("/")[-1]


@contextmanager
def set_cwd(path: Path):
    cwd = Path.cwd()
    try:
        Path(path).mkdir(exist_ok=True, parents=True)
        os.chdir(path)
        yield
    finally:
        os.chdir(cwd)
