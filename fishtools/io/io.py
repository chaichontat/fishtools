import os
import re
import shutil
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import cast

import requests
from loguru import logger
from tqdm.auto import tqdm


def download(url: str, path: str | Path, name: str | None = None) -> None:
    with requests.get(url, stream=True, allow_redirects=True) as r:
        if r.status_code >= 400:
            r.raise_for_status()
            raise RuntimeError(f"Request to {url} returned status code {r.status_code}")

        size = int(r.headers.get("Content-Length", 0))
        if name is None:
            name = url.split("/")[-1]

        path = Path(path).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        path = path / name
        if path.exists():
            if size == path.stat().st_size:
                logger.info(f"File {path} already exists. Skipping.")
                return
            else:
                logger.warning(f"File {path} already exists but size is different. Redownloading.")
        else:
            logger.info(f"Downloading {url}")

        with tqdm.wrapattr(r.raw, "read", total=size) as r_raw:
            with open(f"{path}.tmp", "wb") as f:
                shutil.copyfileobj(r_raw, f)
                os.rename(f"{path}.tmp", path)

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
