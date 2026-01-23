from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_TARGET_SHARD_SIZE_BYTES: int = 512 * 1024 * 1024


def default_zarr_codecs(dtype: np.dtype | type[np.generic] | str) -> list[Any]:
    """Return the default Zarr v3 codecs used for all fishtools arrays.

    This centralizes our compression settings so that array creation is
    consistent across the codebase.
    """
    import zarr

    typesize = np.dtype(dtype).itemsize
    shuffle = (
        zarr.codecs.BloscShuffle.bitshuffle if typesize == 1 else zarr.codecs.BloscShuffle.shuffle
    )
    return [
        zarr.codecs.BytesCodec(),
        zarr.codecs.BloscCodec(
            cname="zstd",
            clevel=5,
            shuffle=shuffle,
            typesize=typesize,
        ),
    ]


def label_zarr_codecs(dtype: np.dtype | type[np.generic] | str) -> list[Any]:
    """Compression tuned for segmentation label arrays (typically uint32)."""
    import zarr

    typesize = np.dtype(dtype).itemsize
    return [
        zarr.codecs.BytesCodec(),
        zarr.codecs.BloscCodec(
            cname="zstd",
            clevel=7,
            shuffle=zarr.codecs.BloscShuffle.bitshuffle,
            typesize=typesize,
        ),
    ]


def choose_shard_shape(
    *,
    shape: tuple[int, ...],
    chunks: tuple[int, ...],
    dtype: np.dtype | type[np.generic] | str,
    target_shard_size_bytes: int = DEFAULT_TARGET_SHARD_SIZE_BYTES,
) -> tuple[int, ...]:
    item_size = int(np.dtype(dtype).itemsize)
    if item_size <= 0:
        raise ValueError(f"Invalid dtype itemsize={item_size} for dtype={dtype!r}")
    if len(shape) != len(chunks):
        raise ValueError(f"shape ndim={len(shape)} does not match chunks ndim={len(chunks)}")
    if any(c <= 0 for c in chunks):
        raise ValueError(f"Invalid chunks={chunks}; all chunk dims must be > 0")
    if any(s <= 0 for s in shape):
        raise ValueError(f"Invalid shape={shape}; all dims must be > 0")

    target_elems = max(1, int(target_shard_size_bytes) // item_size)

    multipliers = [1] * len(shape)
    max_multipliers = [max(1, (s + c - 1) // c) for s, c in zip(shape, chunks, strict=True)]
    current_elems = int(np.prod(chunks))

    grow_order = sorted(range(len(shape)), key=lambda i: int(shape[i]), reverse=True)
    while current_elems < target_elems:
        best_i: int | None = None
        best_m = 1
        best_elems = current_elems
        for i in grow_order:
            mi = multipliers[i]
            if mi >= max_multipliers[i]:
                continue

            for candidate in (min(max_multipliers[i], mi * 2), mi + 1):
                if candidate <= mi:
                    continue
                new_elems = (current_elems // mi) * candidate
                if new_elems <= target_elems and new_elems > best_elems:
                    best_i = i
                    best_m = candidate
                    best_elems = new_elems

        if best_i is None:
            break

        current_elems = best_elems
        multipliers[best_i] = best_m

    return tuple(int(c * m) for c, m in zip(chunks, multipliers, strict=True))

def create_sharded_array(
    write_path: Path | str,
    *,
    shape: tuple[int, ...],
    chunks: tuple[int, ...],
    dtype: np.dtype | type[np.generic] | str,
    overwrite: bool,
    target_shard_size_bytes: int = DEFAULT_TARGET_SHARD_SIZE_BYTES,
    codecs: list[Any] | None = None,
):
    import zarr

    codecs = default_zarr_codecs(dtype) if codecs is None else codecs
    shards = choose_shard_shape(
        shape=shape,
        chunks=chunks,
        dtype=dtype,
        target_shard_size_bytes=target_shard_size_bytes,
    )
    return zarr.create_array(
        str(write_path),
        shape=shape,
        chunks=chunks,
        shards=shards,
        dtype=dtype,
        serializer=codecs[0],
        compressors=tuple(codecs[1:]),
        overwrite=overwrite,
        config={"write_empty_chunks": False},
    )


def numpy_array_to_zarr(write_path: Path | str, array: np.ndarray, chunks: tuple[int, ...]):
    zarr_array = create_sharded_array(
        write_path,
        shape=array.shape,
        chunks=chunks,
        dtype=array.dtype,
        overwrite=True,
    )
    zarr_array[...] = array
    return zarr_array
