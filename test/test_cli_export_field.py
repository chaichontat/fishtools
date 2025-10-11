from __future__ import annotations

from pathlib import Path

import numpy as np
from click.testing import CliRunner
from skimage.transform import resize
from tifffile import imwrite
import zarr

from fishtools.io.workspace import Workspace
from fishtools.preprocess.cli_correct_illum import correct_illum, _mask_union_of_tiles
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


def _tile_mul_grad(
    x0: int,
    y0: int,
    w: int,
    h: int,
    base_bias: float,
    amp: float,
    mx: float,
    my: float,
) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    gx = (xx + float(x0)) / float(w * 2)
    gy = (yy + float(y0)) / float(h * 2)
    grad_mult = 1.0 + mx * gx + my * gy
    return (base_bias + amp * grad_mult).astype(np.float32)


def _write_reg_tile(path: Path, arr_yx: np.ndarray) -> None:
    zcyx = arr_yx[None, None, :, :].astype(np.float32)
    imwrite(path, zcyx)


def test_export_field_matches_plot_logic(tmp_path: Path) -> None:
    # Workspace + 2x2 layout
    ws_root = tmp_path / "ws"
    roi = "roi"
    cb = "cb"
    _ws_mark(ws_root)

    w = h = 96
    tc_dir = ws_root / "analysis" / "deconv" / f"stitch--{roi}"
    entries = [(1, 0.0, 0.0), (2, float(w), 0.0), (3, 0.0, float(h)), (4, float(w), float(h))]
    _write_tc(tc_dir, entries)

    reg_dir = ws_root / "analysis" / "deconv" / f"registered--{roi}+{cb}"
    reg_dir.mkdir(parents=True, exist_ok=True)

    base_bias, amp, mx, my = 1000.0, 300.0, 0.7, 0.5
    for idx, x0, y0 in entries:
        img = _tile_mul_grad(int(x0), int(y0), w, h, base_bias=base_bias, amp=amp, mx=mx, my=my)
        _write_reg_tile(reg_dir / f"reg-{idx:04d}.tif", img)

    runner = CliRunner(mix_stderr=False)
    # Generate subtiles percentiles
    res1 = runner.invoke(
        correct_illum,
        [
            "calculate-percentiles",
            str(ws_root),
            roi,
            "--codebook",
            cb,
            "--percentiles",
            "0.1",
            "99.9",
            "--grid",
            "4",
            "--threads",
            "1",
            "--out-suffix",
            ".subtiles.json",
            "--overwrite",
        ],
        catch_exceptions=False,
    )
    assert res1.exit_code == 0, res1.output

    # Build NPZ model
    res2 = runner.invoke(
        correct_illum,
        [
            "field-generate",
            str(ws_root),
            roi,
            cb,
            "--kernel",
            "thin_plate_spline",
            "--smoothing",
            "2.0",
            "--subtile-suffix",
            ".subtiles.json",
        ],
        catch_exceptions=False,
    )
    assert res2.exit_code == 0, res2.output

    ws = Workspace(ws_root)
    model_npz = sorted(ws.opt(cb).path.glob(f"illum-field--{roi}-*.npz"))[0]

    # Export field with ds=4
    out_zarr = tmp_path / "exported_field.zarr"
    res3 = runner.invoke(
        correct_illum,
        [
            "export-field",
            str(model_npz),
            "--workspace",
            str(ws_root),
            "--roi",
            roi,
            "--codebook",
            cb,
            "--what",
            "range",
            "--downsample",
            "4",
            "--output",
            str(out_zarr),
        ],
        catch_exceptions=False,
    )
    assert res3.exit_code == 0, res3.output
    assert out_zarr.exists()

    # Compute expected via the same plot-field math
    rng = RangeFieldPointsModel.from_npz(model_npz)
    meta = rng.meta
    tile_w = float(meta.get("tile_w", 1968.0))
    tile_h = float(meta.get("tile_h", 1968.0))

    idx, xs0, ys0 = ws.tileconfig(roi).df.select("index", "x", "y").to_pandas().to_numpy().T
    xs0 = xs0.astype(np.float64)
    ys0 = ys0.astype(np.float64)
    x0 = float(xs0.min())
    x1 = float(xs0.max() + tile_w)
    y0 = float(ys0.min())
    y1 = float(ys0.max() + tile_h)

    xs, ys, low_field, high_field = rng.evaluate(x0, y0, x1, y1)
    mask = _mask_union_of_tiles(xs, ys, xs0, ys0, tile_w, tile_h)
    inv_range = rng.range_correction(low_field, high_field, mask=mask)

    W = int(np.ceil(x1 - x0))
    H = int(np.ceil(y1 - y0))
    native = resize(inv_range, (H, W), preserve_range=True, anti_aliasing=True).astype(np.float32)
    expected = resize(native, (max(1, H // 4), max(1, W // 4)), preserve_range=True, anti_aliasing=True).astype(
        np.float32
    )

    za = zarr.open(str(out_zarr), mode="r")
    got_raw = np.asarray(za).astype(np.float32)
    axes = getattr(za, "attrs", {}).get("axes")
    if axes is not None:
        assert axes == "CYX"

    # Expect CYX with singleton channel
    assert got_raw.ndim == 3 and got_raw.shape[0] == 1
    got = got_raw[0]
    assert got.shape == expected.shape
    # Means should be ~1.0 after normalization
    assert np.isclose(float(got.mean()), 1.0, atol=5e-2)
    # Fields should match numerically (smooth surfaces; allow small tolerance)
    assert np.allclose(got, expected, rtol=1e-2, atol=1e-2)
    # Parity on variability with expected (may be uniform for some inputs)
    assert np.isclose(float(got.std()), float(expected.std()), atol=1e-6)
