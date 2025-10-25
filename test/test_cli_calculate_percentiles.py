from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from click.testing import CliRunner
from tifffile import imwrite

from fishtools.io.workspace import Workspace
from fishtools.preprocess.cli_correct_illum import correct_illum


def _mark_workspace(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "READY.DONE").write_text("ok\n")


def _write_registered_tile(path: Path, z: int = 2, c: int = 2, h: int = 4, w: int = 4) -> None:
    arr = np.arange(z * c * h * w, dtype=np.float32).reshape(z, c, h, w)
    imwrite(path, arr)


def test_calculate_percentiles_all_rois(tmp_path: Path) -> None:
    ws_root = tmp_path / "ws"
    _mark_workspace(ws_root)
    cb = "cb_test"
    rois = ("roi_a", "roi_b")

    for roi in rois:
        reg_dir = ws_root / "analysis" / "deconv" / f"registered--{roi}+{cb}"
        reg_dir.mkdir(parents=True, exist_ok=True)
        _write_registered_tile(reg_dir / "reg-0001.tif")

    runner = CliRunner(mix_stderr=False)
    result = runner.invoke(
        correct_illum,
        [
            "calculate-percentiles",
            str(ws_root),
            "--codebook",
            cb,
            "--percentiles",
            "1",
            "99",
            "--grid",
            "2",
            "--threads",
            "1",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output

    ws = Workspace(ws_root)
    for roi in rois:
        json_path = ws.regimg(roi, cb, 1).with_name("reg-0001" + ".subtiles.json")
        assert json_path.exists(), f"Missing subtile JSON for ROI {roi}"
        payload = json.loads(json_path.read_text())
        assert payload, "Expected channel payloads"
        first_channel = next(iter(payload.values()))
        assert len(first_channel) == 4, "Grid=2 should yield 4 subtile entries"
        entry = first_channel[0]
        assert "percentiles" in entry and {"1", "99"}.issubset(entry["percentiles"].keys())

    # Second invocation without overwrite should still succeed and skip work gracefully
    result2 = runner.invoke(
        correct_illum,
        [
            "calculate-percentiles",
            str(ws_root),
            "--codebook",
            cb,
            "--grid",
            "2",
            "--threads",
            "1",
        ],
        catch_exceptions=False,
    )
    assert result2.exit_code == 0, result2.output
