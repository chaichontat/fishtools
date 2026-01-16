from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pytest
import zarr

from fishtools.io.workspace import Workspace
from fishtools.segment.cell_thumbnail import cell_thumbnail_from_adata


def _write_fused(path: Path, *, shape: tuple[int, int, int, int]) -> zarr.Array:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = zarr.open_array(path, mode="w", shape=shape, dtype="uint16")
    z_dim, y_dim, x_dim, c_dim = shape
    for z in range(z_dim):
        yy, xx = np.meshgrid(np.arange(y_dim, dtype=np.uint16), np.arange(x_dim, dtype=np.uint16), indexing="ij")
        base = (z * 10000 + yy * 100 + xx).astype(np.uint16, copy=False)
        for c in range(c_dim):
            arr[z, :, :, c] = base + np.uint16(c)
    return arr


def test_cell_thumbnail_from_adata_returns_centered_crop(tmp_path: Path) -> None:
    ws_root = tmp_path / "workspace"
    ws_root.mkdir(parents=True, exist_ok=True)
    (ws_root / "workspace.DONE").touch()
    ws = Workspace(ws_root)
    ws.deconved.mkdir(parents=True, exist_ok=True)

    roi = "roi"
    codebook = "cb1"
    fused_path = ws.stitch(roi, codebook) / "fused.zarr"
    fused = _write_fused(fused_path, shape=(3, 100, 100, 2))

    cell = "roi1"
    x = 60.2
    y = 40.6
    z = 2.0
    adata = ad.AnnData(
        X=np.zeros((1, 1), dtype=np.float32),
        obs={"roi": [roi], "x": [x], "y": [y], "z": [z]},
    )
    adata.obs_names = [cell]

    thumb = cell_thumbnail_from_adata(adata, cell=cell, codebook=codebook, workspace=ws)
    assert thumb.shape == (50, 50, 2)

    z_index = 2
    x_center = int(np.rint(x))
    y_center = int(np.rint(y))
    expected = np.asarray(fused[z_index, y_center, x_center, :])
    assert np.array_equal(thumb[25, 25, :], expected)


def test_cell_thumbnail_from_adata_pads_out_of_bounds(tmp_path: Path) -> None:
    ws_root = tmp_path / "workspace"
    ws_root.mkdir(parents=True, exist_ok=True)
    (ws_root / "workspace.DONE").touch()
    ws = Workspace(ws_root)
    ws.deconved.mkdir(parents=True, exist_ok=True)

    roi = "roi"
    codebook = "cb1"
    fused_path = ws.stitch(roi, codebook) / "fused.zarr"
    fused = _write_fused(fused_path, shape=(1, 10, 10, 1))

    adata = ad.AnnData(
        X=np.zeros((1, 1), dtype=np.float32),
        obs={"roi": [roi], "x": [0.0], "y": [0.0], "z": [0.0]},
    )
    adata.obs_names = ["cell0"]

    thumb = cell_thumbnail_from_adata(adata, cell="cell0", codebook=codebook, workspace=ws, size=50)
    assert thumb.shape == (50, 50, 1)

    # Center pixel maps to the (0,0) pixel.
    assert thumb[25, 25, 0] == np.asarray(fused[0, 0, 0, 0])
    # Far corners are padded with zeros.
    assert thumb[0, 0, 0] == 0
    assert thumb[-1, -1, 0] == 0


def test_cell_thumbnail_infers_workspace_from_uns(tmp_path: Path) -> None:
    ws_root = tmp_path / "workspace"
    ws_root.mkdir(parents=True, exist_ok=True)
    (ws_root / "workspace.DONE").touch()
    ws = Workspace(ws_root)
    ws.deconved.mkdir(parents=True, exist_ok=True)

    roi = "roi"
    codebook = "cb1"
    fused_path = ws.stitch(roi, codebook) / "fused.zarr"
    _write_fused(fused_path, shape=(1, 60, 60, 1))

    adata = ad.AnnData(
        X=np.zeros((1, 1), dtype=np.float32),
        obs={"roi": [roi], "x": [30.0], "y": [30.0], "z": [0.0]},
    )
    adata.obs_names = ["cell0"]
    adata.uns.setdefault("fishtools", {})["segment_export"] = {"workspace_path": str(ws.path)}

    thumb = cell_thumbnail_from_adata(adata, cell="cell0", codebook=codebook)
    assert thumb.shape == (50, 50, 1)


def test_cell_thumbnail_requires_roi_and_xyz_columns(tmp_path: Path) -> None:
    ws_root = tmp_path / "workspace"
    ws_root.mkdir(parents=True, exist_ok=True)
    (ws_root / "workspace.DONE").touch()
    ws = Workspace(ws_root)
    ws.deconved.mkdir(parents=True, exist_ok=True)

    adata = ad.AnnData(X=np.zeros((1, 1), dtype=np.float32), obs={})
    adata.obs_names = ["cell0"]

    with pytest.raises(ValueError, match="missing required columns"):
        cell_thumbnail_from_adata(adata, cell="cell0", codebook="cb1", workspace=ws)

