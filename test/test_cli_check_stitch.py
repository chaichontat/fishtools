from __future__ import annotations

import json
from pathlib import Path

import polars as pl
from click.testing import CliRunner

from fishtools.preprocess.cli import main as preprocess
from fishtools.preprocess.tileconfig import TileConfiguration


def _write_minimal_tileconfig(path: Path) -> None:
    tc = TileConfiguration(
        pl.DataFrame(
            {
                "index": [0, 1, 2, 3],
                "x": [0.0, 100.0, 0.0, 100.0],
                "y": [0.0, 0.0, 100.0, 100.0],
            },
            schema={"index": pl.UInt32, "x": pl.Float32, "y": pl.Float32},
        )
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    tc.write(path)


def test_check_stitch_overlays_spots_final(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    roi = "roiA"
    codebook = "cb"

    ws.mkdir(parents=True, exist_ok=True)
    (ws / "workspace.DONE").write_text("")

    tileconfig = ws / "analysis" / "deconv" / f"stitch--{roi}" / "TileConfiguration.registered.txt"
    _write_minimal_tileconfig(tileconfig)

    parquets_dir = ws / "analysis" / "output" / "parquets"
    parquets_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"x": [10.0, 50.0], "y": [20.0, 70.0]}).write_parquet(parquets_dir / f"{roi}+{codebook}.parquet")

    cfg_path = ws / "analysis" / "deconv" / "config.json"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(
        json.dumps(
            {
                "image_size": 2048,
                "pixel_size_um": 0.108,
                "registration": {"chromatic_path": str(ws), "crop": 40},
            }
        )
    )

    runner = CliRunner()
    res = runner.invoke(
        preprocess,
        ["check-stitch", str(ws), "--spots", codebook, "--per-roi"],
        prog_name="preprocess",
    )
    assert res.exit_code == 0, res.output

    out_dir = ws / "analysis" / "output" / "stitch_layout"
    assert (out_dir / "stitch_layout_all.png").exists()
    assert (out_dir / f"stitch_layout--{roi}.png").exists()


def test_check_stitch_spots_preserve_xy_columns(tmp_path: Path) -> None:
    from fishtools.io.workspace import Workspace
    from fishtools.preprocess.cli_check_stitch import _load_spots_final

    ws_path = tmp_path / "ws"
    roi = "roiA"
    codebook = "cb"

    ws_path.mkdir(parents=True, exist_ok=True)
    (ws_path / "workspace.DONE").write_text("")
    ws = Workspace(ws_path)

    parquets_dir = ws_path / "analysis" / "output" / "parquets"
    parquets_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"x": [10.0, 50.0], "y": [20.0, 70.0]}).write_parquet(parquets_dir / f"{roi}+{codebook}.parquet")

    spots_by_roi = _load_spots_final(ws, [roi], codebook)
    assert set(spots_by_roi.keys()) == {roi}

    df = spots_by_roi[roi]
    assert df.columns == ["plot_x", "plot_y"]
    assert df["plot_x"].to_list() == [10.0, 50.0]
    assert df["plot_y"].to_list() == [20.0, 70.0]


def test_check_stitch_rejects_tile_size_override_with_config(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    roi = "roiA"

    ws.mkdir(parents=True, exist_ok=True)
    (ws / "workspace.DONE").write_text("")

    tileconfig = ws / "analysis" / "deconv" / f"stitch--{roi}" / "TileConfiguration.registered.txt"
    _write_minimal_tileconfig(tileconfig)

    cfg_path = ws / "config.json"
    cfg_path.write_text(
        json.dumps(
            {
                "image_size": 2048,
                "pixel_size_um": 0.108,
                "registration": {"chromatic_path": str(ws), "crop": 40},
            }
        )
    )

    runner = CliRunner()
    res = runner.invoke(
        preprocess,
        ["check-stitch", str(ws), "--config", str(cfg_path), "--tile-size-px", "10"],
        prog_name="preprocess",
    )
    assert res.exit_code != 0
    assert "--tile-size-px cannot be used with --config" in res.output
