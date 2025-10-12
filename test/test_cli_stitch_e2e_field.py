from __future__ import annotations

from pathlib import Path

import numpy as np
from click.testing import CliRunner
from tifffile import TiffFile, imwrite

from fishtools.io.workspace import Workspace
from fishtools.preprocess.cli_correct_illum import correct_illum
from fishtools.preprocess.cli_stitch import extract
from fishtools.preprocess.imageops import apply_low_range_zcyx
from fishtools.preprocess.spots.illumination import RangeFieldPointsModel, _slice_field_to_tile


def _write_ws_marker(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "OK.DONE").write_text("ok\n")


def _write_tileconfig(stitch_dir: Path, entries: list[tuple[int, float, float]]) -> None:
    stitch_dir.mkdir(parents=True, exist_ok=True)
    lines = ["dim=2\n"]
    for idx, x, y in entries:
        lines.append(f"{idx:04d}.tif; ; ({x}, {y})\n")
    (stitch_dir / "TileConfiguration.registered.txt").write_text("".join(lines))


def _make_reg_tile(path: Path, z: int, c: int, h: int, w: int) -> None:
    # Z,C,Y,X synthetic with mild gradients to produce nontrivial percentiles
    zyx = np.zeros((z, c, h, w), dtype=np.float32)
    for zi in range(z):
        base = 50.0 + 10.0 * zi
        yy, xx = np.mgrid[0:h, 0:w]
        zyx[zi, 0] = base + 0.2 * xx + 0.1 * yy
    imwrite(path, zyx)


def test_e2e_field_apply_after_downsample_no_mock(tmp_path: Path) -> None:
    # Workspace layout
    ws_root = tmp_path / "ws"
    _write_ws_marker(ws_root)
    roi = "roi"
    cb = "cb"

    # Tile config with a single tile at a nonzero origin
    tc_dir = ws_root / "analysis" / "deconv" / f"stitch--{roi}"
    _write_tileconfig(tc_dir, [(1, 120.0, 80.0)])

    # Registered tile
    reg_dir = ws_root / "analysis" / "deconv" / f"registered--{roi}+{cb}"
    reg_dir.mkdir(parents=True, exist_ok=True)
    reg_tile = reg_dir / "reg-0001.tif"
    _make_reg_tile(reg_tile, z=2, c=1, h=32, w=32)

    # 1) Collect percentiles (CPU) via CLI
    runner = CliRunner(mix_stderr=False)
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
            "2",
            "--threads",
            "1",
            "--out-suffix",
            ".subtiles.json",
            "--overwrite",
        ],
        catch_exceptions=False,
    )
    assert res1.exit_code == 0, res1.output

    # 2) Generate model NPZ via CLI
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

    # Resolve generated model path (opt_{cb}/illum-field--{roi}-*.npz)
    ws = Workspace(ws_root)
    opt_dir = ws.opt(cb).path
    models = sorted(opt_dir.glob(f"illum-field--{roi}-*.npz"))
    assert models, "Expected a generated field model NPZ"
    model_npz = models[0]

    # 3) Apply via extract with after-downsample path (downsample=1 to avoid GPU)
    out_dir = ws.stitch(roi, cb)
    extract(
        reg_tile,
        out_dir,
        trim=0,
        downsample=1,
        reduce_bit_depth=0,
        subsample_z=1,
        max_proj=False,
        is_2d=False,
        channels=[0],
        field_corr=model_npz,
        field_patch_downsample=1,
        field_neighbors=0,  # use model meta
        field_smoothing=None,
        sc=None,
        workspace_root=ws_root,
        roi_for_ws=roi,
        debug=False,
    )

    # Load corrected outputs (z=0 and z=1)
    out_z0 = out_dir / "00" / "00" / "0001.tif"
    out_z1 = out_dir / "01" / "00" / "0001.tif"
    with TiffFile(out_z0) as tif0, TiffFile(out_z1) as tif1:
        arr0 = tif0.asarray().astype(np.float32)
        arr1 = tif1.asarray().astype(np.float32)

    # Reconstruct expected correction from the model and tile origins
    rng_model = RangeFieldPointsModel.from_npz(model_npz)
    tc = ws.tileconfig(roi)
    row = tc.df.filter(tc.df["index"] == 1)
    x0 = float(row[0, "x"])  # type: ignore[index]
    y0 = float(row[0, "y"])  # type: ignore[index]
    tile_h = 32
    tile_w = 32

    # Evaluate global fields using the same grid step logic as the applicator
    base_step = float(rng_model.meta.get("grid_step_suggest", 192.0))
    tile_min = max(2.0, float(min(tile_w, tile_h)))
    grid_step = min(base_step * 1.0, tile_min / 6.0)
    xs, ys, low_field, high_field = rng_model.evaluate(
        x0,
        y0,
        x0 + tile_w,
        y0 + tile_h,
        grid_step=grid_step,
        neighbors=int(rng_model.meta.get("neighbors", 64)),
        kernel=str(rng_model.meta.get("kernel", "thin_plate_spline")),
        smoothing=float(rng_model.meta.get("smoothing", 2.0)),
    )

    # Global-normalized inverse RANGE multiplier
    scale = float(rng_model.meta.get("range_mean", getattr(rng_model, "range_mean", 1.0)))
    scale = 1.0 if not np.isfinite(scale) or scale <= 0.0 else scale
    inv_global = scale / np.maximum((high_field.astype(np.float64) - low_field.astype(np.float64)), 1e-6)
    inv_global = np.clip(inv_global, 1.0 / 3.0, 3.0)
    denom = float(inv_global.mean()) if inv_global.size else 1.0
    inv_global /= denom if denom > 0 else 1.0

    # Slice and upsample to tile size
    low_patch = _slice_field_to_tile(low_field, xs, ys, x0, y0, x0 + tile_w, y0 + tile_h, (tile_h, tile_w))
    rng_patch = _slice_field_to_tile(
        inv_global.astype(np.float32), xs, ys, x0, y0, x0 + tile_w, y0 + tile_h, (tile_h, tile_w)
    )

    # Apply formula to original input tile to get expected corrected
    with TiffFile(reg_tile) as tif:
        orig = tif.asarray().astype(np.float32)  # Z,C,Y,X
    low3 = np.zeros((1, tile_h, tile_w), dtype=np.float32)
    rng3 = np.ones((1, tile_h, tile_w), dtype=np.float32)
    low3[0] = low_patch
    rng3[0] = rng_patch
    expected = apply_low_range_zcyx(orig, low3, rng3)

    # Compare per-plane
    assert np.allclose(arr0, expected[0, 0], rtol=1e-3, atol=1e-2)
    assert np.allclose(arr1, expected[1, 0], rtol=1e-3, atol=1e-2)

    # Additionally, verify the correction reduces the X-gradient of the signal
    def slope_x(img: np.ndarray) -> float:
        # Mean over Y, fit linear model across X
        prof = img.mean(axis=0)
        xs = np.arange(prof.size, dtype=np.float32)
        m, b = np.polyfit(xs, prof, 1)
        return float(abs(m))

    with TiffFile(reg_tile) as tif:
        orig0 = tif.asarray()[0, 0]
        orig1 = tif.asarray()[1, 0]

    assert slope_x(arr0) <= 0.75 * slope_x(orig0)
    assert slope_x(arr1) <= 0.75 * slope_x(orig1)
