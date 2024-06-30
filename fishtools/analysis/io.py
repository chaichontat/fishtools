import json
from pathlib import Path

import numpy as np
from tifffile import TiffFile, imread


def fishread(path: str | Path, prefix: str, i: int, *, reshape: bool = False):
    path = Path(path)
    img = imread(path / prefix / f"{prefix}-{i:04d}.tif")
    if reshape:
        return img[:-1].reshape(-1, len(path.stem.split("_")), 2048, 2048).max(axis=0), img[-1]
    return img


def metadata(path: str | Path, prefix: str, i: int):
    with TiffFile(Path(path) / prefix / f"{prefix}-{i:04d}.tif") as t:
        if t.shaped_metadata:
            return t.shaped_metadata[0]
        else:
            return t.imagej_metadata
