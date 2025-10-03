from pathlib import Path

import numpy as np
import pytest
import tifffile
import zarr

from fishtools.segment.extract import _append_max_from, _load_and_select


def test_append_max_from_no_dir_returns_same(tmp_path: Path) -> None:
    img = np.zeros((3, 2, 8, 8), dtype=np.uint16)
    out, idx = _append_max_from(img, tmp_path / "reg-0001.tif", None)
    assert out is img
    assert idx is None


def test_append_max_from_shape_mismatch_raises(tmp_path: Path) -> None:
    file = tmp_path / "reg-0001.tif"
    file.touch()
    other_dir = tmp_path / "other"
    other_dir.mkdir()
    # Write a 4D array with mismatched YX
    tifffile.imwrite(other_dir / file.name, np.zeros((3, 2, 7, 8), dtype=np.uint16))

    img = np.zeros((3, 2, 8, 8), dtype=np.uint16)
    with pytest.raises(ValueError):
        _append_max_from(img, file, other_dir)


def test_append_max_from_appends_channel(tmp_path: Path) -> None:
    file = tmp_path / "reg-0002.tif"
    file.touch()
    other_dir = tmp_path / "other"
    other_dir.mkdir()
    # Primary image: Z=3, C=2, Y=8, X=8
    img = np.arange(3 * 2 * 8 * 8, dtype=np.uint16).reshape(3, 2, 8, 8)
    # Other image with same Z,Y,X, different channel count
    other = np.ones((3, 4, 8, 8), dtype=np.uint16)
    tifffile.imwrite(other_dir / file.name, other)

    out, idx = _append_max_from(img, file, other_dir)
    assert out.shape == (3, 3, 8, 8)  # +1 channel from max projection
    assert idx == 2  # last channel index
    # The appended channel equals max over other channels per Z
    expected_max = other.max(axis=1, keepdims=True)
    np.testing.assert_array_equal(out[:, -1:], expected_max)


def test_load_and_select_filters_and_clips(tmp_path: Path) -> None:
    # Write a small 4D stack and exercise filtering/clipping via loader
    file = tmp_path / "reg-000.tif"
    z, c, y, x = 3, 2, 16, 16
    img = np.zeros((z, c, y, x), dtype=np.uint16)
    img[:, :, 8, 8] = 60000
    tifffile.imwrite(file, img, metadata={"axes": "ZCYX"})

    # filter_first=True to mimic z-mode
    out, names = _load_and_select(file, channels="0,1", crop=0, max_from_dir=None, filter_first=True)
    assert out.shape == img.shape
    assert out.dtype == np.uint16
    assert out.min() >= 0
    # Implementation clips to 65530 before cast
    assert out.max() <= 65530
    assert names == ["0", "1"]


def test_load_and_select_supports_zarr(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = tmp_path / "fused.zarr"
    arr = zarr.open(
        store,
        mode="w",
        shape=(4, 8, 8, 3),
        chunks=(2, 8, 8, 3),
        dtype=np.uint16,
    )
    arr[...] = np.arange(arr.size, dtype=np.uint16).reshape(arr.shape)
    arr.attrs["key"] = ["polyA", "reddot", "spots"]

    # Avoid modifying data through unsharp filtering for deterministic assertion
    monkeypatch.setattr("fishtools.segment.extract.unsharp_all", lambda x, **kwargs: x)

    out, names = _load_and_select(store, channels="auto", crop=0, max_from_path=None, filter_first=False)

    assert out.shape == (4, 2, 8, 8)  # auto selects 2 channels
    assert names == ["polyA", "reddot"]
