from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from click.testing import CliRunner
from tifffile import TiffFile

from fishtools.io.workspace import Workspace
from fishtools.preprocess.cli_correct_illum import correct_illum
from fishtools.preprocess.spots.illumination import RangeFieldPointsModel


def _write_ws_marker(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "OK.DONE").write_text("ok\n")


def _write_tileconfig(stitch_dir: Path, entries: list[tuple[int, float, float]]) -> None:
    stitch_dir.mkdir(parents=True, exist_ok=True)
    lines = ["dim=2\n"]
    for idx, x, y in entries:
        lines.append(f"{idx:04d}.tif; ; ({x}, {y})\n")
    (stitch_dir / "TileConfiguration.registered.txt").write_text("".join(lines))


def _make_synthetic_model(
    npz_path: Path, bbox: tuple[float, float, float, float], tile_w: int, tile_h: int
) -> None:
    x0, y0, x1, y1 = bbox
    xs = np.linspace(x0, x1, 9, dtype=np.float32)
    ys = np.linspace(y0, y1, 7, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    xy = np.c_[X.ravel(), Y.ravel()].astype(np.float32)

    # Smoothly varying low/high so that RANGE is not constant
    vlow = 50.0 + 0.10 * (X - X.min()) + 0.05 * (Y - Y.min())
    vhigh = vlow + 30.0 + 0.02 * (X - X.mean())
    vlow = vlow.ravel().astype(np.float32)
    vhigh = vhigh.ravel().astype(np.float32)

    meta = {
        "roi": "roi",
        "codebook": "cb",
        "workspace": str(npz_path.parent.parent),
        "tile_w": float(tile_w),
        "tile_h": float(tile_h),
        "kernel": "thin_plate_spline",
        "smoothing": 1.0,
        "neighbors": 0,
        "grid_step_suggest": 8.0,
        "range_mean": 30.0,
        "range_min": 0.5,
        "range_max": 2.0,
    }

    np.savez_compressed(
        npz_path,
        xy=xy,
        vlow=vlow,
        vhigh=vhigh,
        meta=json.dumps(meta).encode("utf-8"),
    )


def test_render_global_field_stamps_patches(tmp_path: Path) -> None:
    # Workspace and tile config
    ws_root = tmp_path / "ws"
    _write_ws_marker(ws_root)
    roi = "roi"
    cb = "cb"
    stitch_dir = ws_root / "analysis" / "deconv" / f"stitch--{roi}"
    entries = [(1, 0.0, 0.0), (2, 40.0, 0.0), (3, 0.0, 32.0)]
    _write_tileconfig(stitch_dir, entries)

    tile_w, tile_h = 40, 32
    # Bounding box covering all tiles
    xs = [e[1] for e in entries]
    ys = [e[2] for e in entries]
    bbox = (min(xs), min(ys), max(xs) + tile_w, max(ys) + tile_h)

    ws = Workspace(ws_root)

    # Synthetic model
    cb_slug = Workspace.sanitize_codebook_name(cb)
    model_dir = ws.deconved / f"fields+{cb_slug}"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_npz = model_dir / f"illum-field--{roi}+{cb_slug}--synthetic.npz"
    _make_synthetic_model(model_npz, bbox, tile_w, tile_h)

    # Run CLI to render the first 2 tiles globally
    out_tif = model_npz.with_suffix(".global-range.tif")
    runner = CliRunner(mix_stderr=False)
    res = runner.invoke(
        correct_illum,
        [
            "render-global",
            str(model_npz),
            "--workspace",
            str(ws_root),
            "--roi",
            roi,
            "-n",
            "2",
            "--mode",
            "range",
            "--neighbors",
            "0",
            "--smoothing",
            "1.0",
            "--grid-step",
            "8.0",
            "--output",
            str(out_tif),
        ],
        catch_exceptions=False,
    )
    assert res.exit_code == 0, res.output

    # Load rendered output
    with TiffFile(out_tif) as tif:
        rendered = tif.asarray().astype(np.float32)

    # Recompute expected by stamping patches for the first 2 tiles
    rng_model = RangeFieldPointsModel.from_npz(model_npz)
    # Compute extents for first 2 tiles (sorted by index)
    entries2 = sorted(entries, key=lambda t: t[0])[:2]
    xs2 = [e[1] for e in entries2]
    ys2 = [e[2] for e in entries2]
    x_min = min(xs2)
    y_min = min(ys2)
    x_max = max(xs2) + tile_w
    y_max = max(ys2) + tile_h
    W = int(np.ceil(x_max - x_min))
    H = int(np.ceil(y_max - y_min))
    expected = np.zeros((H, W), dtype=np.float32)
    origins_list = [(float(x), float(y)) for x, y in zip(xs2, ys2, strict=False)]
    for x0, y0 in zip(xs2, ys2, strict=False):
        patch = rng_model.field_patch(
            x0=float(x0),
            y0=float(y0),
            width=tile_w,
            height=tile_h,
            mode="range",
            tile_origins=origins_list,
            tile_w=float(tile_w),
            tile_h=float(tile_h),
            grid_step=8.0,
            neighbors=None,
            kernel="thin_plate_spline",
            smoothing=1.0,
        )
        xo = int(round(float(x0 - x_min)))
        yo = int(round(float(y0 - y_min)))
        expected[yo : yo + tile_h, xo : xo + tile_w] = patch.astype(np.float32)

    assert rendered.shape == expected.shape
    assert np.allclose(rendered, expected, rtol=1e-3, atol=1e-3)
