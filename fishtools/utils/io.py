import os
import shutil
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, overload

import numpy as np
import numpy.typing as npt
import requests
from loguru import logger
from tifffile import imread
from tqdm.auto import tqdm

from fishtools.preprocess.tileconfig import TileConfiguration


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


@dataclass
class OptimizePath:
    path: Path

    @property
    def mse(self):
        return self.path / "mse.txt"

    @property
    def scaling_factor(self):
        return self.path / "global_scale.txt"


@dataclass
class Workspace:
    path: Path

    def __str__(self):
        return str(self.path)

    def __repr__(self):
        return f"Workspace({self.path})"

    def __init__(self, path: Path | str, deconved: bool = False):
        path = Path(path)
        self.path = Path(path if not deconved else path.parent.parent).expanduser().resolve()

    @property
    def rounds(self):
        FORBIDDEN = {"10x", "analysis", "shifts", "stitch", "fid", "registered", "old", "basic"}
        path = self.path if not self.deconved.exists() else self.deconved
        res = sorted(
            {
                p.name.split("--")[0]
                for p in path.iterdir()
                if "--" in p.name and p.is_dir() and not any(p.name.startswith(bad) for bad in FORBIDDEN)
            },
            key=lambda x: f"{int(x.split('_')[0]):02d}" if x.split("_")[0].isdigit() else x,
        )
        if not res:
            raise ValueError(f"No round subdirectories found in {path}.")
        return res

    @property
    def rois(self):
        path = self.path if not self.deconved.exists() else self.deconved
        return sorted({
            p.name.split("--")[1].split("+")[0] for p in path.iterdir() if p.is_dir() and "--" in p.name
        })

    @property
    def deconved(self):
        return self.path / "analysis" / "deconv"

    @overload
    def img(self, round_: str, roi: str, idx: int, *, read: Literal[False] = ...) -> Path: ...
    @overload
    def img(self, round_: str, roi: str, idx: int, *, read: Literal[True]) -> npt.NDArray[np.uint16]: ...
    def img(self, round_: str, roi: str, idx: int, *, read: bool = False):
        path = self.deconved / f"{round_}--{roi}/{round_}-{idx:04d}.tif"
        if read:
            return imread(path)
        return path

    def registered(self, roi: str, codebook: str):
        return self.deconved / f"registered--{roi}+{codebook}"

    @overload
    def regimg(self, roi: str, codebook: str, idx: int, *, read: Literal[False] = ...) -> Path: ...
    @overload
    def regimg(self, roi: str, codebook: str, idx: int, *, read: Literal[True]) -> npt.NDArray[np.uint16]: ...
    def regimg(self, roi: str, codebook: str, idx: int, *, read: bool = False):
        path = self.registered(roi, codebook) / f"reg-{idx:04d}.tif"
        if read:
            return imread(path)
        return path

    def stitch(self, roi: str, codebook: str | None = None):
        if codebook is None:
            return self.deconved / f"stitch--{roi}"
        return self.deconved / f"stitch--{roi}+{codebook}"

    def tileconfig(self, roi: str):
        try:
            return TileConfiguration.from_file(
                self.path / f"stitch--{roi}" / "TileConfiguration.registered.txt"
            )
        except FileNotFoundError:
            raise FileNotFoundError("Haven't stitch/registered yet. Run preprocess stitch register first.")

    def segment(self, roi: str, codebook: str):
        return self.deconved / f"segment--{roi}+{codebook}"

    def opt(self, codebook: str):
        return OptimizePath(self.deconved / f"opt--{codebook}")
