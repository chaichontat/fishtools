from __future__ import annotations

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
