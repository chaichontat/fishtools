from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

from click.testing import CliRunner
import pytest
from loguru import logger

from fishtools.io.workspace import Workspace
from fishtools.plot.diagnostics.shifts import ShiftsAdapter
from fishtools.preprocess.cli_check_shifts import _check_missing_tiles, check_shifts


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


def test_check_shifts_defaults_output_to_workspace_analysis_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "ready.DONE").touch()
    (workspace / "analysis").mkdir(parents=True, exist_ok=True)

    roi = "roi1"
    codebook = "cb1"
    round_name = "1_2_3"

    deconv_dir = workspace / "analysis" / "deconv"
    shift_dir = deconv_dir / f"shifts--{roi}+{codebook}"
    _write_shift_json(shift_dir, 0, round_name)

    stitch_dir = deconv_dir / f"stitch--{roi}"
    stitch_dir.mkdir(parents=True, exist_ok=True)
    (stitch_dir / "TileConfiguration.registered.txt").write_text("dim=2\n0000.tif; ; (0.0, 0.0)\n")

    codebook_path = tmp_path / f"{codebook}.json"
    codebook_path.write_text("{}")

    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    def _empty_fig(*_args: object, **_kwargs: object) -> Figure:
        return plt.figure()

    monkeypatch.setattr("fishtools.preprocess.cli_check_shifts.make_shifts_scatter_figure", _empty_fig)
    monkeypatch.setattr("fishtools.preprocess.cli_check_shifts.make_corr_vs_l2_figure", _empty_fig)
    monkeypatch.setattr("fishtools.preprocess.cli_check_shifts.make_corr_hist_figure", _empty_fig)
    monkeypatch.setattr("fishtools.preprocess.cli_check_shifts.make_shifts_layout_figure", _empty_fig)
    def _tile_size_px(*_args: object, **_kwargs: object) -> float:
        return 100.0

    monkeypatch.setattr("fishtools.preprocess.cli_check_shifts._infer_tile_size_px", _tile_size_px)

    runner = CliRunner()
    result = runner.invoke(check_shifts, [str(workspace / "analysis"), roi, "--codebook", str(codebook_path)])
    assert result.exit_code == 0, result.output

    expected = (
        workspace
        / "analysis"
        / "output"
        / "shifts_layout"
        / f"shifts_layout--{roi}+{codebook}.png"
    )
    assert expected.exists()
