from __future__ import annotations

import os
import shutil
from contextlib import contextmanager
from pathlib import Path

import requests
from loguru import logger
from tqdm.auto import tqdm


def download(url: str, path: str | Path, name: str | None = None) -> None:
    with requests.get(url, stream=True, allow_redirects=True, timeout=30) as response:
        response.raise_for_status()
        size = int(response.headers.get("Content-Length", 0))
        name = name or url.split("/")[-1]

        destination = Path(path).expanduser().resolve()
        destination.mkdir(parents=True, exist_ok=True)
        destination = destination / name
        if destination.exists() and size == destination.stat().st_size:
            logger.info(f"File {destination} already exists. Skipping.")
            return
        if destination.exists():
            logger.warning(f"File {destination} already exists but size is different. Redownloading.")
        else:
            logger.info(f"Downloading {url}")

        temporary = destination.with_name(f"{destination.name}.tmp")
        with tqdm.wrapattr(response.raw, "read", total=size) as source:
            with temporary.open("wb") as output:
                shutil.copyfileobj(source, output)
        temporary.replace(destination)

    logger.info(f"Finished downloading {url}")


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

