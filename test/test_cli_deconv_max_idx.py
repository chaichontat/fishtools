from __future__ import annotations

from pathlib import Path
import types

import numpy as np
import pytest
import tifffile
from click.testing import CliRunner
import cupy as cp

from fishtools.preprocess.cli import main as preprocess


def _install_cuda_stub() -> None:
    runtime = types.SimpleNamespace(
        getDeviceCount=lambda: 1,
        CUDARuntimeError=RuntimeError,
    )
    dummy_stream = types.SimpleNamespace(synchronize=lambda: None)

    def _event_factory():  # type: ignore[no-untyped-def]
        return types.SimpleNamespace(record=lambda *args, **kwargs: None, synchronize=lambda: None)

    cp.cuda = types.SimpleNamespace(  # type: ignore[attr-defined]
        runtime=runtime,
        get_current_stream=lambda: dummy_stream,
        Event=lambda: _event_factory(),
        get_elapsed_time=lambda start, end: 0.0,
        Device=lambda _: types.SimpleNamespace(use=lambda: None),
    )


@pytest.fixture(autouse=True)
def stub_cupy_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_cuda_stub()
    monkeypatch.setattr(cp, "cuda", cp.cuda, raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)


def _write_workspace(tmp_path: Path) -> Path:
    (tmp_path / "workspace.DONE").touch()
    round_name = "r1"
    roi = "roiA"
    tile_dir = tmp_path / f"{round_name}--{roi}"
    tile_dir.mkdir(parents=True, exist_ok=True)
    payload = np.zeros((1, 4, 4), dtype=np.uint16)
    tifffile.imwrite(tile_dir / f"{round_name}-0000.tif", payload)
    return tmp_path


def test_deconvnew_run_max_idx_forces_no_delete_origin_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = _write_workspace(tmp_path)

    captured: dict[str, object] = {}

    def _fake_plan_execute(**kwargs: object) -> list[object]:
        captured.update(kwargs)
        return []

    monkeypatch.setattr("fishtools.preprocess.cli_deconv._plan_and_execute", _fake_plan_execute)

    runner = CliRunner()
    result = runner.invoke(preprocess, ["deconvnew", "run", "--max-idx", "0", str(workspace), "r1"])
    assert result.exit_code == 0, result.output

    assert captured["max_idx"] == 0
    assert captured["delete_origin"] is False


def test_deconvnew_run_max_idx_rejects_delete_origin(tmp_path: Path) -> None:
    workspace = _write_workspace(tmp_path)

    runner = CliRunner()
    result = runner.invoke(
        preprocess,
        ["deconvnew", "run", "--max-idx", "0", "--delete-origin", str(workspace), "r1"],
    )
    assert result.exit_code != 0
    assert "Refusing to delete origin directories with --max-idx" in result.output


def test_collect_round_tiles_respects_max_idx(tmp_path: Path) -> None:
    from fishtools.preprocess.cli_deconv import _collect_round_tiles

    (tmp_path / "workspace.DONE").touch()
    tile_dir = tmp_path / "r1--roiA"
    tile_dir.mkdir(parents=True, exist_ok=True)

    payload = np.zeros((1, 4, 4), dtype=np.uint16)
    for idx in (0, 2, 5):
        tifffile.imwrite(tile_dir / f"r1-{idx:04d}.tif", payload)

    tiles = _collect_round_tiles(tmp_path, "r1", rois=["roiA"], max_idx=2)
    assert [t.name for t in tiles] == ["r1-0000.tif", "r1-0002.tif"]
