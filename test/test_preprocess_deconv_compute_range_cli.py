from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile
from click.testing import CliRunner

from fishtools.preprocess.cli import main as preprocess


def _write_deconv_tile(
    workspace: Path,
    *,
    round_name: str,
    roi: str,
    deconv_mode: str,
    deconv_min: list[float],
    deconv_scale: list[float],
) -> None:
    out_dir = Path(workspace) / "analysis" / "deconv" / f"{round_name}--{roi}"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = np.zeros((1, 8, 8), dtype=np.uint16)
    tifffile.imwrite(
        out_dir / f"{round_name}-0000.tif",
        payload,
        metadata={
            "deconv_mode": deconv_mode,
            "deconv_min": deconv_min,
            "deconv_scale": deconv_scale,
        },
    )


def test_compute_range_errors_on_non_legacy_round(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)
    (workspace / "workspace.DONE").touch()

    round_name = "r1"
    _write_deconv_tile(
        workspace,
        round_name=round_name,
        roi="roiA",
        deconv_mode="u16",
        deconv_min=[1.0],
        deconv_scale=[2.0],
    )

    runner = CliRunner()
    result = runner.invoke(preprocess, ["deconv", "compute-range", str(workspace / "analysis" / "deconv")])
    assert result.exit_code != 0

    assert not (workspace / "analysis" / "deconv_scaling" / f"{round_name}.txt").exists()


def test_compute_range_runs_on_legacy_round(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)
    (workspace / "workspace.DONE").touch()

    round_name = "r1"
    _write_deconv_tile(
        workspace,
        round_name=round_name,
        roi="roiA",
        deconv_mode="legacy",
        deconv_min=[1.0],
        deconv_scale=[2.0],
    )

    runner = CliRunner()
    result = runner.invoke(preprocess, ["deconv", "compute-range", str(workspace / "analysis" / "deconv")])
    assert result.exit_code == 0, result.output

    scaling_path = workspace / "analysis" / "deconv_scaling" / f"{round_name}.txt"
    assert scaling_path.exists()
    loaded = np.loadtxt(scaling_path).reshape(2, -1)
    assert loaded.shape == (2, 1)
    assert loaded[0, 0] == 1.0
    assert loaded[1, 0] == 2.0
