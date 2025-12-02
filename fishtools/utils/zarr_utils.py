from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def default_zarr_codecs() -> list[Any]:
    """Return the default Zarr v3 codecs used for all fishtools arrays.

    This centralizes our compression settings so that array creation is
    consistent across the codebase.
    """
    import zarr

    return [
        zarr.codecs.BytesCodec(),
        zarr.codecs.BloscCodec(
            cname="zstd",
            clevel=4,
            shuffle=zarr.codecs.BloscShuffle.shuffle,
        ),
    ]


def numpy_array_to_zarr(write_path: Path | str, array: np.ndarray, chunks: tuple[int, ...]):
    import zarr

    zarr_array = zarr.create_array(
        write_path,
        shape=array.shape,
        chunks=chunks,
        dtype=array.dtype,
        codecs=default_zarr_codecs(),
    )
    zarr_array[...] = array
    return zarr_array
