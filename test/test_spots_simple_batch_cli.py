from pathlib import Path

import numpy as np
import polars as pl
from click.testing import CliRunner

from fishtools.io.workspace import safe_imwrite
from fishtools.preprocess.spots.align_prod import spots as spots_cli
from fishtools.preprocess.tileconfig import TileConfiguration


def _make_workspace(tmp_path: Path) -> Path:
    # Workspace root marker for Workspace(path) auto-resolution
    (tmp_path / "workspace.DONE").write_text("")
    deconv_dir = tmp_path / "analysis" / "deconv"
    deconv_dir.mkdir(parents=True, exist_ok=True)
    return deconv_dir


def test_spots_simple_batch_writes_pickles_and_stitches(tmp_path: Path) -> None:
    deconv_dir = _make_workspace(tmp_path)
    roi = "roiA"
    codebook = "cb"

    reg_dir = deconv_dir / f"registered--{roi}+{codebook}"
    reg_dir.mkdir(parents=True, exist_ok=True)

    # Synthetic ZCYX tile (size must be >= 1024 for split logic)
    size = 1100
    tile = np.zeros((1, 2, size, size), dtype=np.uint16)
    # Place one bright spot in each quadrant; alternate channels
    tile[0, 0, 50, 50] = 65535  # top-left (GeneA)
    tile[0, 1, 50, size - 50] = 65535  # top-right (GeneB)
    tile[0, 1, size - 50, 50] = 65535  # bottom-left (GeneB)
    tile[0, 0, size - 50, size - 50] = 65535  # bottom-right (GeneA)

    safe_imwrite(
        reg_dir / "reg-0000.tif",
        tile,
        metadata={"axes": "ZCYX", "key": ["GeneA", "GeneB"]},
    )

    # Minimal TileConfiguration for stitching
    tc_dir = deconv_dir / f"stitch--{roi}"
    tc_dir.mkdir(parents=True, exist_ok=True)
    TileConfiguration(
        pl.DataFrame(
            {"index": [0], "x": [0.0], "y": [0.0]},
            schema=pl.Schema({"index": pl.UInt32, "x": pl.Float32, "y": pl.Float32}),
        )
    ).write(tc_dir / "TileConfiguration.registered.txt")

    codebook_path = tmp_path / f"{codebook}.json"
    codebook_path.write_text("{}")

    runner = CliRunner()
    res = runner.invoke(
        spots_cli,
        [
            "simple-batch",
            str(deconv_dir),
            roi,
            "--codebook",
            str(codebook_path),
            "--threads",
            "1",
            "--tophat-radius",
            "0",
            "--blob-detector",
            '{"min_sigma":0.5,"max_sigma":1.5,"num_sigma":3,"threshold":0.1,"measurement_type":"mean","is_volume":true}',
        ],
    )
    assert res.exit_code == 0, res.output

    decoded_dir = reg_dir / f"decoded-{codebook}"
    for split in range(4):
        assert (decoded_dir / f"reg-0000-{split}.pkl").exists()

    res = runner.invoke(
        spots_cli,
        [
            "stitch",
            str(deconv_dir),
            roi,
            "--codebook",
            str(codebook_path),
            "--threads",
            "1",
        ],
    )
    assert res.exit_code == 0, res.output
    assert (decoded_dir / "spots.parquet").exists()
    spots = pl.read_parquet(decoded_dir / "spots.parquet")
    assert set(spots["target"].to_list()) >= {"GeneA", "GeneB"}


def test_spots_simple_runs_one_tile(tmp_path: Path) -> None:
    deconv_dir = _make_workspace(tmp_path)
    roi = "roiA"
    codebook = "cb"

    reg_dir = deconv_dir / f"registered--{roi}+{codebook}"
    reg_dir.mkdir(parents=True, exist_ok=True)

    size = 1100
    tile = np.zeros((1, 2, size, size), dtype=np.uint16)
    tile[0, 0, 50, 50] = 65535  # split 0
    tile[0, 1, size - 50, size - 50] = 65535  # split 3
    path_tile = reg_dir / "reg-0000.tif"
    safe_imwrite(
        path_tile,
        tile,
        metadata={"axes": "ZCYX", "key": ["GeneA", "GeneB"]},
    )

    codebook_path = tmp_path / f"{codebook}.json"
    codebook_path.write_text("{}")

    runner = CliRunner()
    res = runner.invoke(
        spots_cli,
        [
            "simple",
            str(path_tile),
            "--codebook",
            str(codebook_path),
            "--tophat-radius",
            "0",
            "--blob-detector",
            '{"min_sigma":0.5,"max_sigma":1.5,"num_sigma":3,"threshold":0.1,"measurement_type":"mean","is_volume":true}',
        ],
    )
    assert res.exit_code == 0, res.output

    decoded_dir = reg_dir / f"decoded-{codebook}"
    assert (decoded_dir / "reg-0000-0.pkl").exists()
    assert (decoded_dir / "reg-0000-3.pkl").exists()
