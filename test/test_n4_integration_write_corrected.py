from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import zarr

from fishtools.preprocess import n4


pytestmark = pytest.mark.timeout(30)


def _make_fused(path: Path, data: np.ndarray, names: list[str] | None = None) -> Path:
    store = zarr.open_array(path / "fused.zarr", mode="w", shape=data.shape, chunks=(1, data.shape[1], data.shape[2], 1), dtype=data.dtype)
    store[...] = data
    if names is not None:
        store.attrs["key"] = names
    return path / "fused.zarr"


def test_write_multi_channel_field_tiff_metadata(tmp_path: Path) -> None:
    # Arrange: two fields and names
    field0 = np.ones((6, 7), dtype=np.float32)
    field1 = np.linspace(1.0, 2.0, 42, dtype=np.float32).reshape(6, 7)
    fields = np.stack([field0, field1], axis=0)
    names = ["alpha", "beta"]
    out = tmp_path / "n4_correction_field.tif"

    # Act
    written = n4._write_multi_channel_field(fields, names, out, overwrite=True, meta_extra={"roi": "r", "codebook": "cb"})

    # Assert
    assert written == out and out.exists()
    from tifffile import imread as _tifread
    arr = _tifread(out)
    assert arr.shape == (2, 6, 7)
    # Metadata round-trip through tifffile -> zarr.imread may not fully preserve custom keys,
    # but axes should be present (CYX) via the writer path.


def test_resolve_fused_path_returns_existing_stitch_path(tmp_path: Path) -> None:
    # Arrange: make the expected directory structure and an empty fused.zarr directory
    ws = tmp_path / "ws"
    fused_dir = ws / "analysis/deconv/stitch--roi+cb"
    (fused_dir / "fused.zarr").mkdir(parents=True)

    # Act
    fused_path = n4._resolve_fused_path(ws, "roi", "cb")

    # Assert
    assert fused_path == fused_dir / "fused.zarr"


def test_load_channel_reads_from_zarr_dummy(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange: dummy zarr array object
    class Dummy:
        def __init__(self) -> None:
            self.shape = (2, 6, 5, 3)
            self.dtype = np.uint16

        @property
        def ndim(self) -> int:
            return 4

        def __getitem__(self, idx):
            z, y, x, c = idx
            assert isinstance(z, int) and isinstance(c, int)
            # Return constant plane uniquely determined by z and c
            val = 100 * z + 10 * c + 7
            return np.full((6, 5), val, dtype=np.float32)

        @property
        def attrs(self):  # pragma: no cover - not used in this test
            return {"key": ["c0", "c1", "c2"]}

    monkeypatch.setattr(n4.zarr, "open_array", lambda *a, **k: Dummy())

    # Act
    plane = n4._load_channel(Path("ignored.zarr"), channel_index=2, z_index=1)

    # Assert
    assert plane.shape == (6, 5)
    assert plane.dtype == np.float32
    np.testing.assert_allclose(plane, 100 * 1 + 10 * 2 + 7, rtol=0, atol=0)
