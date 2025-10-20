from __future__ import annotations

from pathlib import Path

import numpy as np
import zarr
from click.testing import CliRunner
from skimage.transform import resize

from fishtools.io.workspace import Workspace
from fishtools.preprocess.cli_correct_illum import _mask_union_of_tiles, correct_illum
from fishtools.preprocess.spots.illumination import RangeFieldPointsModel


def _ws_mark(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "READY.DONE").write_text("ok\n")


def _write_tc(stitch_dir: Path, entries: list[tuple[int, float, float]]) -> None:
    stitch_dir.mkdir(parents=True, exist_ok=True)
    txt = ["dim=2\n"]
    for idx, x, y in entries:
        txt.append(f"{idx:04d}.tif; ; ({x}, {y})\n")
    (stitch_dir / "TileConfiguration.registered.txt").write_text("".join(txt))


def test_export_field_multi_channel_cyx(tmp_path: Path) -> None:
    # Create minimal workspace and 2x2 tiling
    ws_root = tmp_path / "ws"
    roi = "roi"
    cb = "cb"
    _ws_mark(ws_root)
    w = h = 64
    entries = [(1, 0.0, 0.0), (2, float(w), 0.0), (3, 0.0, float(h)), (4, float(w), float(h))]
    _write_tc(ws_root / "analysis" / "deconv" / f"stitch--{roi}", entries)

    # Build a synthetic multi-channel NPZ directly
    # Four points roughly at tile centers
    xy = np.array(
        [
            [w * 0.5, h * 0.5],
            [w * 1.5, h * 0.5],
            [w * 0.5, h * 1.5],
            [w * 1.5, h * 1.5],
        ],
        dtype=np.float32,
    )
    # Two channels with different low/high patterns
    vlow = np.stack(
        [
            np.array([100.0, 120.0, 110.0, 115.0], dtype=np.float32),
            np.array([200.0, 180.0, 190.0, 210.0], dtype=np.float32),
        ],
        axis=1,
    )
    vhigh = np.stack(
        [
            np.array([400.0, 420.0, 410.0, 405.0], dtype=np.float32),
            np.array([500.0, 520.0, 510.0, 505.0], dtype=np.float32),
        ],
        axis=1,
    )
    meta = {
        "roi": roi,
        "codebook": cb,
        "kernel": "thin_plate_spline",
        "smoothing": 2.0,
        "neighbors": 64,
        "tile_w": int(w),
        "tile_h": int(h),
        "channels": ["ch0", "ch1"],
        "workspace": str(ws_root),
    }
    model = RangeFieldPointsModel(xy=xy, vlow=vlow, vhigh=vhigh, meta=meta)
    npz_path = (tmp_path / "illum-mc.npz").resolve()
    model.to_npz(npz_path)

    runner = CliRunner(mix_stderr=False)
    out_zarr = tmp_path / "exported_field_mc.zarr"
    res = runner.invoke(
        correct_illum,
        [
            "export-field",
            str(npz_path),
            "--workspace",
            str(ws_root),
            "--roi",
            roi,
            "--codebook",
            cb,
            "--what",
            "range",
            "--downsample",
            "2",
            "--output",
            str(out_zarr),
        ],
        catch_exceptions=False,
    )
    assert res.exit_code == 0, res.output
    assert out_zarr.exists()

    za = zarr.open(str(out_zarr), mode="r")
    arr_all = np.asarray(za).astype(np.float32)
    attrs = dict(getattr(za, "attrs", {}))
    axes = attrs.get("axes")
    assert axes == "TCYX"
    t_labels = list(attrs.get("t_labels", []))
    assert "range" in t_labels and "low" in t_labels
    t_idx = t_labels.index("range")
    # Select RANGE plane; expect C×Y×X
    arr = arr_all[t_idx]
    assert arr.ndim == 3 and arr.shape[0] == 2  # two channels

    # Means per channel should be ~1.0 after normalization (RANGE plane)
    m0 = float(np.nanmean(arr[0]))
    m1 = float(np.nanmean(arr[1]))
    assert abs(m0 - 1.0) < 5e-2 and abs(m1 - 1.0) < 5e-2

    # Recompute expected for channel 0 using the same math
    ws = Workspace(ws_root)
    idx, xs0, ys0 = ws.tileconfig(roi).df.select("index", "x", "y").to_pandas().to_numpy().T
    xs0 = xs0.astype(np.float64)
    ys0 = ys0.astype(np.float64)
    x0 = float(xs0.min())
    x1 = float(xs0.max() + w)
    y0 = float(ys0.min())
    y1 = float(ys0.max() + h)

    # Use single-channel submodel for expected reconstruction
    sub0 = RangeFieldPointsModel(xy=xy, vlow=vlow[:, 0], vhigh=vhigh[:, 0], meta=dict(meta, channel="ch0"))
    xs, ys, lf, hf = sub0.evaluate(x0, y0, x1, y1)
    mask = _mask_union_of_tiles(xs, ys, xs0, ys0, w, h)
    inv = sub0.range_correction(lf, hf, mask=mask)
    # Crop to union bbox and downsample by 2 (match CLI)
    rows_any = mask.any(axis=1)
    cols_any = mask.any(axis=0)
    j0 = int(np.argmax(rows_any))
    j1 = int(len(rows_any) - 1 - np.argmax(rows_any[::-1]))
    i0 = int(np.argmax(cols_any))
    i1 = int(len(cols_any) - 1 - np.argmax(cols_any[::-1]))
    inv_sub = inv[j0 : j1 + 1, i0 : i1 + 1]
    H = int(np.ceil(ys[j1] - ys[j0]))
    W = int(np.ceil(xs[i1] - xs[i0]))
    native = resize(inv_sub, (H, W), preserve_range=True, anti_aliasing=True).astype(np.float32)
    got0 = arr[0]
    expected0 = resize(
        native,
        (max(1, H // 2), max(1, W // 2)),
        preserve_range=True,
        anti_aliasing=True,
    ).astype(np.float32)
    assert got0.shape == expected0.shape
    assert np.allclose(np.nanmean(got0), np.nanmean(expected0), atol=5e-2)
