from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from fishtools.preprocess.cli import main as preprocess


def test_deconvnew_easy_does_not_invoke_quantize_when_u16_pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "workspace.DONE").touch()
    round_name = "pi_edu"

    scaling_dir = tmp_path / "analysis" / "deconv_scaling"
    scaling_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        scaling_dir / f"{round_name}.txt",
        np.vstack([np.array([0.0], dtype=np.float32), np.array([1.0], dtype=np.float32)]),
    )

    calls: list[list[str]] = []

    def fake_run(cmd: list[str], *_: object, **__: object) -> subprocess.CompletedProcess[object]:
        calls.append(list(cmd))
        if cmd[:3] == ["preprocess", "deconvnew", "quantize"]:
            raise subprocess.CalledProcessError(1, cmd)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    runner = CliRunner()
    result = runner.invoke(preprocess, ["deconvnew", "easy", str(tmp_path), round_name])
    assert result.exit_code == 0, result.output

    assert any(cmd[:3] == ["preprocess", "deconvnew", "run"] for cmd in calls), calls
    assert not any(cmd[:3] == ["preprocess", "deconvnew", "quantize"] for cmd in calls), calls


def test_deconvnew_easy_invokes_quantize_when_float32_tiles_exist(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "workspace.DONE").touch()
    round_name = "pi_edu"

    scaling_dir = tmp_path / "analysis" / "deconv_scaling"
    scaling_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        scaling_dir / f"{round_name}.txt",
        np.vstack([np.array([0.0], dtype=np.float32), np.array([1.0], dtype=np.float32)]),
    )

    deconv32_dir = tmp_path / "analysis" / "deconv32" / f"{round_name}--roiA"
    deconv32_dir.mkdir(parents=True, exist_ok=True)
    (deconv32_dir / f"{round_name}-0001.tif").touch()

    calls: list[list[str]] = []

    def fake_run(cmd: list[str], *_: object, **__: object) -> subprocess.CompletedProcess[object]:
        calls.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    runner = CliRunner()
    result = runner.invoke(preprocess, ["deconvnew", "easy", str(tmp_path), round_name])
    assert result.exit_code == 0, result.output

    assert any(cmd[:3] == ["preprocess", "deconvnew", "run"] for cmd in calls), calls
    assert any(cmd[:3] == ["preprocess", "deconvnew", "quantize"] for cmd in calls), calls


def test_deconvnew_easy_p_min_p_max_recomputes_scaling(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "workspace.DONE").touch()
    round_name = "pi_edu"

    scaling_dir = tmp_path / "analysis" / "deconv_scaling"
    scaling_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        scaling_dir / f"{round_name}.txt",
        np.vstack([np.array([0.0], dtype=np.float32), np.array([1.0], dtype=np.float32)]),
    )

    calls: list[list[str]] = []

    def fake_run(cmd: list[str], *_: object, **__: object) -> subprocess.CompletedProcess[object]:
        calls.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    runner = CliRunner()
    result = runner.invoke(
        preprocess,
        ["deconvnew", "easy", "--p-min", "0.01", "--p-max", "0.99", str(tmp_path), round_name],
    )
    assert result.exit_code == 0, result.output

    assert any(cmd[:3] == ["preprocess", "deconvnew", "prepare"] for cmd in calls), calls
    assert any(
        cmd[:3] == ["preprocess", "deconvnew", "precompute"] and "--p-low" in cmd and "--p-high" in cmd
        for cmd in calls
    ), calls


def test_deconvnew_easy_requires_both_p_min_and_p_max(tmp_path: Path) -> None:
    (tmp_path / "workspace.DONE").touch()
    round_name = "pi_edu"

    scaling_dir = tmp_path / "analysis" / "deconv_scaling"
    scaling_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        scaling_dir / f"{round_name}.txt",
        np.vstack([np.array([0.0], dtype=np.float32), np.array([1.0], dtype=np.float32)]),
    )

    runner = CliRunner()
    result = runner.invoke(preprocess, ["deconvnew", "easy", "--p-min", "0.01", str(tmp_path), round_name])
    assert result.exit_code != 0
    assert "Require both --p-min and --p-max" in result.output
