from pathlib import Path

import numpy as np
import pytest
import tifffile
import zarr
from starfish.core.types import Axes

from fishtools.preprocess.spots.align_prod import make_fetcher
from fishtools.utils.zarr_utils import default_zarr_codecs


def _write_fake_tif(path: Path, shape=(5, 3, 32, 48)) -> None:
    z, c, y, x = shape
    rng = np.random.default_rng(0)
    # Simulate realistic 16-bit dynamic range
    arr = (rng.uniform(0, 50000, size=shape)).astype(np.uint16)
    tifffile.imwrite(path, arr)


def test_make_fetcher_float32_unit_interval(tmp_path: Path):
    p = tmp_path / "reg-0001.tif"
    _write_fake_tif(p)

    stack = make_fetcher(p, np.s_[:])

    # Basic shape checks
    data = stack.xarray.values  # small, safe to load eagerly
    assert data.dtype == np.float32
    assert data.min() >= 0.0
    assert data.max() <= 1.0

    # Ensure axes exist and plane extraction produces finite values
    plane = stack.sel({Axes.ZPLANE: 0, Axes.CH: 0}).xarray.values.squeeze()
    assert plane.dtype == np.float32
    assert np.isfinite(plane).all()


def _write_tileconfig(root: Path, roi: str, index: int, x: float = 0.0, y: float = 0.0) -> None:
    tc_dir = root / "analysis" / "deconv" / f"stitch--{roi}"
    tc_dir.mkdir(parents=True, exist_ok=True)
    (tc_dir / "TileConfiguration.registered.txt").write_text(f"dim=2\n{index:04d}.tif; ; ({x}, {y})\n")


def _write_field_store(
    root: Path, roi: str, codebook: str, c: int, h: int, w: int, low: float, rng: float, ds: int = 1
) -> Path:
    slug = codebook.replace("-", "_").replace(" ", "_")
    store = root / "analysis" / "deconv" / f"fields+{slug}" / f"field--{roi}+{slug}.zarr"
    store.parent.mkdir(parents=True, exist_ok=True)
    za = zarr.open(
        str(store),
        mode="w",
        shape=(2, c, h // ds if ds > 1 else h, w // ds if ds > 1 else w),
        chunks=(1, 1, max(1, h // (ds * 2)), max(1, w // (ds * 2))),
        dtype=np.float32,
        codecs=default_zarr_codecs(),
    )
    za[0] = low
    za[1] = rng
    za.attrs["axes"] = "TCYX"
    za.attrs["t_labels"] = ["low", "range"]
    za.attrs["model_meta"] = {
        "downsample": int(ds),
        "x0": 0.0,
        "y0": 0.0,
        "channels": [str(i) for i in range(c)],
    }
    return store


def test_make_fetcher_with_field_zarr_applies_correction(tmp_path: Path):
    # Workspace root with sentinel
    (tmp_path / "ROOT.DONE").write_text("")
    roi, cb = "roiX", "cbX"
    reg_dir = tmp_path / "analysis" / "deconv" / f"registered--{roi}+{cb}"
    reg_dir.mkdir(parents=True, exist_ok=True)

    # Synthetic tile
    z, c, h, w = 3, 2, 32, 48
    tile = (np.random.default_rng(1).uniform(0, 65535, size=(z, c, h, w))).astype(np.uint16)
    tiff_path = reg_dir / "reg-0001.tif"
    tifffile.imwrite(tiff_path, tile)

    # Minimal TileConfiguration with index=1 at (0,0)
    _write_tileconfig(tmp_path, roi, index=1, x=0.0, y=0.0)

    # Field store with constant low=0.1 and range=0.5
    _write_field_store(tmp_path, roi, cb, c=c, h=h, w=w, low=0.1, rng=0.5, ds=1)

    # Raw stack (no correction)
    stack_raw = make_fetcher(tiff_path, np.s_[:])
    plane_raw = stack_raw.sel({Axes.ZPLANE: 0, Axes.CH: 0}).xarray.values.squeeze()

    # Corrected stack
    stack_cor = make_fetcher(tiff_path, np.s_[:], field_correct=True, workspace_root=tmp_path)
    plane_cor = stack_cor.sel({Axes.ZPLANE: 0, Axes.CH: 0}).xarray.values.squeeze()

    # Validate types and ranges
    assert plane_cor.dtype == np.float32
    assert 0.0 <= float(plane_cor.min()) <= float(plane_cor.max()) <= 1.0
    assert plane_raw.dtype == np.float32
    assert 0.0 <= float(plane_raw.min()) <= float(plane_raw.max()) <= 1.0

    # Correction should not increase intensity everywhere; expect some decrease
    diff = plane_raw - plane_cor
    assert np.isfinite(diff).all()
    assert (diff >= -1e-6).all()
    assert (diff > 1e-4).any()


def test_make_fetcher_field_missing_raises(tmp_path: Path):
    (tmp_path / "ROOT.DONE").write_text("")
    roi, cb = "roiY", "cbY"
    reg_dir = tmp_path / "analysis" / "deconv" / f"registered--{roi}+{cb}"
    reg_dir.mkdir(parents=True, exist_ok=True)
    _write_fake_tif(reg_dir / "reg-0001.tif", shape=(2, 1, 16, 16))
    with pytest.raises(FileNotFoundError):
        make_fetcher(
            reg_dir / "reg-0001.tif",
            np.s_[:],
            field_correct=True,
            workspace_root=tmp_path,
        )
