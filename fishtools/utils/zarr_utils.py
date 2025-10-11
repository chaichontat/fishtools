from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np


def numpy_array_to_zarr(write_path: Path | str, array: np.ndarray, chunks: tuple[int, ...]):
    import zarr

    zarr_array = zarr.create_array(
        write_path,
        shape=array.shape,
        chunks=chunks,
        dtype=array.dtype,
    )
    zarr_array[...] = array
    return zarr_array
