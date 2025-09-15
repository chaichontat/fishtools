from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from click.testing import CliRunner
from fishtools.preprocess.spots.align_batchoptimize import optimize as optimize_cmd


class _Call:
    def __init__(self, args: list[str]):
        self.args = args

    def __repr__(self) -> str:
        return " ".join(map(str, self.args))


@pytest.fixture()
def workspace(tmp_path: Path) -> Path:
    # Minimal structure required by optimize(): at least one registered folder
    (tmp_path / "registered--roiA").mkdir(parents=True)
    return tmp_path


def test_optimize_invokes_subcommands_without_blank(monkeypatch: Any, workspace: Path, tmp_path: Path) -> None:
    calls: list[_Call] = []

    def fake_run(args: list[Any], check: bool, capture_output: bool):  # type: ignore[no-untyped-def]
        calls.append(_Call([str(a) for a in args]))
        class _R:  # minimal stub
            returncode = 0
        return _R()

    monkeypatch.setattr("subprocess.run", fake_run)

    cb = tmp_path / "cb.json"
    cb.write_text("{}")

    runner = CliRunner()
    res = runner.invoke(
        optimize_cmd,
        [
            str(workspace),
            "roiA",
            "--codebook",
            str(cb),
            "--rounds",
            "1",
            "--threads",
            "3",
            "--batch-size",
            "50",
            # no --blank
            # max_proj False by default
            "--threshold",
            "0.008",
        ],
    )
    assert res.exit_code == 0, res.output

    # Expect exactly 3 subprocess calls: step-optimize, combine, find-threshold
    assert len(calls) == 3
    assert calls[0].args[2] == "step-optimize"
    assert any("--split=0" == a for a in calls[0].args)
    assert not any(a.startswith("--blank=") or a == "--blank" for a in calls[0].args)

    assert calls[1].args[2] == "combine"

    assert calls[2].args[2] == "find-threshold"
    assert not any(a.startswith("--blank=") or a == "--blank" for a in calls[2].args)


def test_optimize_includes_blank_when_provided(monkeypatch: Any, workspace: Path, tmp_path: Path) -> None:
    calls: list[_Call] = []

    def fake_run(args: list[Any], check: bool, capture_output: bool):  # type: ignore[no-untyped-def]
        calls.append(_Call([str(a) for a in args]))
        class _R:
            returncode = 0
        return _R()

    monkeypatch.setattr("subprocess.run", fake_run)

    cb = tmp_path / "cb.json"
    cb.write_text("{}")

    runner = CliRunner()
    res = runner.invoke(
        optimize_cmd,
        [
            str(workspace),
            "roiA",
            "--codebook",
            str(cb),
            "--rounds",
            "1",
            "--threads",
            "3",
            "--batch-size",
            "50",
            "--max-proj",
            "--blank",
            "cb_blank",
            "--threshold",
            "0.008",
        ],
    )
    assert res.exit_code == 0, res.output

    assert len(calls) == 3
    step_args = calls[0].args
    find_args = calls[2].args
    assert any(a.startswith("--blank=") or a == "--blank" for a in step_args)
    assert any(a.startswith("--blank=") or a == "--blank" for a in find_args)
    assert any(a == "--max-proj=1" for a in step_args)


def test_optimize_forwards_config(monkeypatch: Any, workspace: Path, tmp_path: Path) -> None:
    calls: list[_Call] = []

    def fake_run(args: list[Any], check: bool, capture_output: bool):  # type: ignore[no-untyped-def]
        calls.append(_Call([str(a) for a in args]))
        class _R:
            returncode = 0
        return _R()

    monkeypatch.setattr("subprocess.run", fake_run)

    cb = tmp_path / "cb.json"
    cfg = tmp_path / "project.json"
    cb.write_text("{}")
    cfg.write_text("{}")

    runner = CliRunner()
    res = runner.invoke(
        optimize_cmd,
        [
            str(workspace),
            "roiA",
            "--codebook",
            str(cb),
            "--rounds",
            "1",
            "--threads",
            "3",
            "--batch-size",
            "50",
            "--config",
            str(cfg),
        ],
    )
    assert res.exit_code == 0, res.output

    assert len(calls) == 3
    # step-optimize and find-threshold should receive --config
    assert any(a == "--config" for a in calls[0].args)
    assert any(a == str(cfg) for a in calls[0].args)
    assert any(a == "--config" for a in calls[2].args)
    assert any(a == str(cfg) for a in calls[2].args)
