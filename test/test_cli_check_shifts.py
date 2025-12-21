from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import pytest
from loguru import logger

from fishtools.io.workspace import Workspace
from fishtools.plot.diagnostics.shifts import ShiftsAdapter
from fishtools.preprocess.cli_check_shifts import _check_missing_tiles


@pytest.fixture
def log_capture() -> Generator[list[str], None, None]:
    messages: list[str] = []

    def _capture(message: str) -> None:
        messages.append(message)

    handler_id = logger.add(_capture, level="INFO")
    try:
        yield messages
    finally:
        logger.remove(handler_id)


def _write_shift_json(shift_dir: Path, tile: int, round_name: str) -> None:
    payload = ShiftsAdapter.validate_python({
        round_name: {"shifts": [0.0, 0.0], "corr": 0.95, "residual": 0.0},
    })
    shift_dir.mkdir(parents=True, exist_ok=True)
    (shift_dir / f"shifts-{tile:04d}.json").write_bytes(ShiftsAdapter.dump_json(payload))


def test_check_missing_tiles_prefers_repaired_round(log_capture: list[str], tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "ready.DONE").touch()
    deconv_dir = workspace / "analysis" / "deconv"
    roi = "roi1"
    codebook = "cb1"
    ref_round = "1_2_3"

    ref_dir = deconv_dir / f"{ref_round}--{roi}"
    repaired_dir = deconv_dir / f"{ref_round}--{roi}--repaired"
    shift_dir = deconv_dir / f"shifts--{roi}+{codebook}"

    ref_dir.mkdir(parents=True, exist_ok=True)
    repaired_dir.mkdir(parents=True, exist_ok=True)

    # Reference directory has fewer tiles than the repaired directory.
    for idx in range(2):
        (ref_dir / f"{ref_round}-{idx:04d}.tif").touch()
    for idx in range(3):
        (repaired_dir / f"{ref_round}-{idx:04d}.tif").touch()
        _write_shift_json(shift_dir, idx, ref_round)

    ws = Workspace(workspace)
    _check_missing_tiles(ws, roi, codebook, ref_round, shift_dir)

    combined_logs = "\n".join(log_capture)
    assert "Using repaired folder for reference round 1_2_3" in combined_logs
    assert "Missing shifts for tiles" not in combined_logs
