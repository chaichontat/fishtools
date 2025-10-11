from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from tifffile import imwrite

from fishtools.preprocess.cli_stitch import combine


@pytest.fixture()
def mock_zarr_ops(monkeypatch: Any):
    store: dict[Path, Any] = {}

    class MockZarrArray:
        def __init__(self, shape: tuple[int, ...], chunks: tuple[int, ...], dtype: Any):
            self.shape = shape
            self.chunks = chunks
            self.dtype = dtype
            self.attrs: dict[str, Any] = {}

        def __setitem__(self, key: Any, value: np.ndarray) -> None:
            pass

        def __getitem__(self, key: Any) -> np.ndarray:
            # Return ones for any slice request
            if isinstance(key, tuple):
                # Build shape from slices roughly; keep dims
                shape = []
                for k, dim in zip(key, self.shape):
                    if isinstance(k, slice):
                        start = 0 if k.start is None else k.start
                        stop = dim if k.stop is None else k.stop
                        shape.append(max(0, stop - start))
                    else:
                        shape.append(1)
                return np.ones(tuple(shape), dtype=self.dtype)
            return np.ones(self.shape, dtype=self.dtype)

    def mock_create_array(path: Path, shape: tuple[int, ...], chunks: tuple[int, ...], dtype: Any, **kwargs):
        arr = MockZarrArray(shape, chunks, dtype)
        store[path] = arr
        return arr

    def mock_open_array(path: Path, mode: str = "r", **kwargs):
        return store.get(path) or MockZarrArray(kwargs.get("shape", (1, 1, 1, 1)), kwargs.get("chunks", (1, 1, 1, 1)), kwargs.get("dtype", np.uint16))

    monkeypatch.setattr("zarr.create_array", mock_create_array)
    monkeypatch.setattr("zarr.open_array", mock_open_array)
    return store


def _make_stitch_tree(base: Path, roi: str, codebook: str, z: int, c: int, size: int = 16) -> Path:
    """Create stitch--{roi}+{codebook}/ZZ/CC/fused_CC-1.tif structure with constant pixels."""
    stitch_path = base / "analysis" / "deconv" / f"stitch--{roi}+{codebook}"
    stitch_path.mkdir(parents=True, exist_ok=True)
    for zi in range(z):
        for ci in range(c):
            chdir = stitch_path / f"{zi:02d}" / f"{ci:02d}"
            chdir.mkdir(parents=True, exist_ok=True)
            imwrite(chdir / f"fused_{ci:02d}-1.tif", np.ones((size, size), dtype=np.uint16))
    return stitch_path


def _install_registered_with_names(base: Path, roi: str, codebook: str, names: list[str] | None, monkeypatch: Any) -> None:
    reg_dir = base / "analysis" / "deconv" / f"registered--{roi}+{codebook}"
    reg_dir.mkdir(parents=True, exist_ok=True)
    tif = reg_dir / "dummy.tif"
    imwrite(tif, np.ones((8, 8), dtype=np.uint16))

    # Fake TiffFile to return shaped_metadata with provided names
    class _T:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        @property
        def shaped_metadata(self):  # type: ignore[no-redef]
            if names is None:
                return None
            return [dict(key=names)]

    monkeypatch.setattr("fishtools.preprocess.cli_stitch.TiffFile", lambda p: _T())


@pytest.mark.parametrize(
    "c, meta_names, expect_names",
    [
        (1, ["ch0"], ["ch0"]),  # single channel, names propagate
        (2, None, None),  # no metadata
        (3, ["a", "b"], ["a", "b", "spots"]),  # C == len(names)+1 -> append spots
        (4, ["a", "b", "c", "d"], ["a", "b", "c", "d"]),  # equal lengths
    ],
)
def test_combine_channel_variants(tmp_path: Path, monkeypatch: Any, mock_zarr_ops, c: int, meta_names: list[str] | None, expect_names: list[str] | None) -> None:
    base = tmp_path / "ws"
    roi = "roi1"
    cb = "cb1"
    stitch_path = _make_stitch_tree(base, roi, cb, z=2, c=c, size=16)

    if meta_names is not None:
        _install_registered_with_names(base, roi, cb, meta_names, monkeypatch)

    # Run combine (direct callback due to @batch_roi)
    combine.callback(path=base / "analysis" / "deconv", roi=roi, codebook=cb, chunk_size=8)

    # Validate zarr shape
    import zarr
    z = zarr.open_array(stitch_path / "fused.zarr", mode="r")
    assert z.shape == (2, 16, 16, c)

    # Validate attrs['key'] behavior
    if expect_names is None:
        assert "key" not in z.attrs
    else:
        assert list(z.attrs.get("key")) == expect_names

    # Normalization is no longer computed during combine; file should not exist
    assert not (stitch_path / "normalization.json").exists()
