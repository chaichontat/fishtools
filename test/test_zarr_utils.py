from __future__ import annotations

import numpy as np

from fishtools.utils.zarr_utils import create_zarr_array


def test_create_zarr_array_does_not_use_shards(tmp_path) -> None:
    arr = create_zarr_array(
        tmp_path / "a.zarr",
        shape=(1, 32, 32),
        chunks=(1, 16, 16),
        dtype=np.uint16,
        overwrite=True,
    )

    assert getattr(arr, "shards", None) is None

    data = np.arange(32 * 32, dtype=np.uint16).reshape((1, 32, 32))
    arr[...] = data
    assert np.array_equal(arr[...], data)
