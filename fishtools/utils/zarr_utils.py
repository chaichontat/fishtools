from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def default_zarr_codecs(dtype: np.dtype | type[np.generic] | str) -> list[Any]:
    """Return the default Zarr v3 codecs used for all fishtools arrays.

    This centralizes our compression settings so that array creation is
    consistent across the codebase.
    """
    import zarr

    typesize = np.dtype(dtype).itemsize
    return [
        zarr.codecs.BytesCodec(),
        zarr.codecs.BloscCodec(
            cname="zstd",
            clevel=4,
            shuffle=zarr.codecs.BloscShuffle.shuffle,
            typesize=typesize,
        ),
    ]


def numpy_array_to_zarr(write_path: Path | str, array: np.ndarray, chunks: tuple[int, ...]):
    import zarr

    zarr.config.set({"array.target_shard_size_bytes": "10MB"})
    zarr_array = zarr.open(
        str(write_path),
        shape=array.shape,
        chunks=chunks,
        dtype=array.dtype,
        codecs=default_zarr_codecs(array.dtype),
        mode="w",
    )
    zarr_array[...] = array
    return zarr_array
