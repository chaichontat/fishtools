from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from fishtools.preprocess.illumination import (
    apply_field_tcyx_store_to_img,
    clear_field_store_cache,
    parse_tile_index_from_path,
    resolve_roi_for_field,
    tile_origin,
)
from fishtools.preprocess.tileconfig import TileConfiguration


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    clear_field_store_cache()


def _write_field_store_tcyx(path: Path, low: np.ndarray, rng: np.ndarray, channels: list[str]) -> None:
    import zarr

    low = low.astype(np.float32)
    rng = rng.astype(np.float32)
    arr = np.stack([low, rng], axis=0)
    za = zarr.open_array(str(path), mode="w", shape=arr.shape, dtype="f4")
    za[...] = arr
    za.attrs["axes"] = "TCYX"
    za.attrs["t_labels"] = ["low", "range"]
    za.attrs["model_meta"] = {
        "channels": channels,
        "downsample": 1,
        "x0": 0.0,
        "y0": 0.0,
    }


def test_resolve_roi_for_field_infers_from_path(tmp_path: Path) -> None:
    inferred = resolve_roi_for_field(tmp_path / "stitch--roiA+cb1", None)
    assert inferred == "roiA"


def test_parse_tile_index_from_path_handles_dash() -> None:
    assert parse_tile_index_from_path(Path("round-0012")) == 12
    assert parse_tile_index_from_path(Path("15")) == 15


def test_tile_origin_reads_tile_configuration(tmp_path: Path) -> None:
    ws_root = tmp_path / "workspace"
    tc_dir = ws_root / "analysis" / "deconv" / "stitch--roiA"
    tc_dir.mkdir(parents=True)
    (ws_root / "workspace.DONE").touch()
    tc_path = tc_dir / "TileConfiguration.registered.txt"
    TileConfiguration(pl.DataFrame({"index": [7], "x": [12.5], "y": [18.0]})).write(tc_path)

    origin = tile_origin(ws_root, "roiA", 7)
    assert origin == (12.5, 18.0)


def test_apply_field_tcyx_store_to_img_cyx(tmp_path: Path) -> None:
    field_store = tmp_path / "field.zarr"
    low = np.array([[1.0, 2.0], [0.0, 1.0]], dtype=np.float32)[None, ...]
    rng = np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32)[None, ...]
    _write_field_store_tcyx(field_store, low, rng, ["cy3"])

    img = np.array([[2.0, 5.0], [1.0, 4.0]], dtype=np.float32)[None, ...]
    img_original = img.copy()
    corrected = apply_field_tcyx_store_to_img(
        img,
        ["cy3"],
        tcyx_zarr=field_store,
        x0=0.0,
        y0=0.0,
        trim=0,
    )

    expected = (
        np.maximum(img_original - np.array([[1.0, 2.0], [0.0, 1.0]])[None, ...], 0.0)
        * np.array([[2.0, 3.0], [4.0, 5.0]])[None, ...]
    )
    assert np.allclose(corrected, expected.astype(np.float32))


def test_apply_field_tcyx_store_to_img_zcyx(tmp_path: Path) -> None:
    field_store = tmp_path / "field.zarr"
    low = np.zeros((1, 2, 2), dtype=np.float32)
    rng = np.ones((1, 2, 2), dtype=np.float32)
    _write_field_store_tcyx(field_store, low, rng, ["cy3"])

    img = np.arange(8, dtype=np.float32).reshape(2, 1, 2, 2)
    img_original = img.copy()
    corrected = apply_field_tcyx_store_to_img(
        img,
        ["cy3"],
        tcyx_zarr=field_store,
        x0=0.0,
        y0=0.0,
        trim=0,
    )

    assert corrected.dtype == np.float32
    assert np.allclose(corrected, img_original)
