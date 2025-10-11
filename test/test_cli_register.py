from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from fishtools.preprocess.cli_register import (
    DATA,
    Config,
    Fiducial,
    RegisterConfig,
    _run,
)
from fishtools.preprocess.cli_register import (
    register as register_cli,
)


def _make_codebook(tmp_path: Path) -> Path:
    cb = tmp_path / "cb.json"
    cb.write_text("{}")
    return cb


def test_cli_register_run_invokes_internal(tmp_path: Path, monkeypatch: Any) -> None:
    # Arrange: create minimal workspace and codebook
    ws = tmp_path / "ws"
    ws.mkdir()
    cb = _make_codebook(tmp_path)

    called: dict[str, Any] = {}

    # Stub heavy internal pipeline to avoid I/O
    def fake__run(
        path: Path,
        roi: str,
        idx: int,
        *,
        codebook: str | Path,
        reference: str,
        config,
        debug: bool,
        overwrite: bool,
        no_priors: bool,
    ) -> None:  # type: ignore[no-untyped-def]
        called.update({
            "path": path,
            "roi": roi,
            "idx": idx,
            "codebook": Path(codebook),
            "reference": reference,
            "config": config,
            "debug": debug,
            "overwrite": overwrite,
            "no_priors": no_priors,
        })

    monkeypatch.setattr("fishtools.preprocess.cli_register._run", fake__run)

    runner = CliRunner()
    result = runner.invoke(
        register_cli,
        [
            "run",
            str(ws),
            "42",
            "--codebook",
            str(cb),
            "--roi",
            "roiA",
            "--reference",
            "4_12_20",
            "--overwrite",
        ],
    )

    assert result.exit_code == 0, result.output
    # Verify our stub saw the right parameters
    assert called["path"] == ws
    assert called["roi"] == "roiA"
    assert called["idx"] == 42
    assert called["codebook"] == cb
    assert called["reference"] == "4_12_20"
    assert called["overwrite"] is True
    # Defaults: fwhm=4.0 (from click default), threshold=5.0
    cfg = called["config"]
    assert pytest.approx(cfg.registration.fiducial.fwhm, rel=0, abs=1e-6) == 4.0
    assert pytest.approx(cfg.registration.fiducial.threshold, rel=0, abs=1e-6) == 5.0
    # Sanity on other core defaults plumbed through
    assert cfg.registration.crop == 40
    assert cfg.registration.downsample == 1


def test_cli_register_batch_spawns_subprocess(tmp_path: Path, monkeypatch: Any) -> None:
    # Arrange workspace structure expected by batch
    base = tmp_path / "analysis" / "deconv"
    (base / "2_10_18--roiA").mkdir(parents=True)
    # create one input tif path that matches the glob, content not used
    (base / "2_10_18--roiA" / "2_10_18-0001.tif").write_text("")

    cb = _make_codebook(tmp_path)

    # Fake Workspace returned by cli_register.Workspace
    class _WS:
        def __init__(self, *_: Any, **__: Any) -> None:
            self.rois = ["roiA"]
            self.rounds = ["2_10_18"]

    monkeypatch.setattr("fishtools.preprocess.cli_register.Workspace", _WS)

    calls: list[list[str]] = []

    def fake_run(argv: list[str], *, check: bool = True):  # type: ignore[no-untyped-def]
        assert check is True
        calls.append(argv)

        class _R:  # Minimal CompletedProcess-like shim
            returncode = 0

        return _R()

    monkeypatch.setattr("fishtools.preprocess.cli_register._run_child_cli", fake_run)

    runner = CliRunner()
    result = runner.invoke(
        register_cli,
        [
            "batch",
            str(base),
            "--codebook",
            str(cb),
            "--overwrite",
            "--threads",
            "1",
        ],
    )

    assert result.exit_code == 0, result.output
    # Expect one submission for idx 1 in roiA
    assert len(calls) == 1
    argv = calls[0]
    assert argv[:3] == ["preprocess", "register", "run"]
    assert str(base) in argv
    assert f"--codebook={cb}" in argv
    assert "--reference" in argv  # ref auto-selected
    # Batch should forward default fwhm/threshold into child command
    assert any(a.startswith("--fwhm=") and float(a.split("=", 1)[1]) == 4.0 for a in argv)
    assert any(a.startswith("--threshold=") and float(a.split("=", 1)[1]) == 6.0 for a in argv)


def test_cli_register_run_respects_cli_overrides(tmp_path: Path, monkeypatch: Any) -> None:
    """Ensure CLI flags override the config passed to _run."""
    ws = tmp_path / "ws"
    ws.mkdir()
    cb = _make_codebook(tmp_path)

    seen: dict[str, Any] = {}

    def fake__run(
        path: Path,
        roi: str,
        idx: int,
        *,
        codebook: str | Path,
        reference: str,
        config,
        debug: bool,
        overwrite: bool,
        no_priors: bool,
    ) -> None:  # type: ignore[no-untyped-def]
        seen.update({"config": config, "roi": roi, "idx": idx, "reference": reference})

    monkeypatch.setattr("fishtools.preprocess.cli_register._run", fake__run)

    runner = CliRunner()
    result = runner.invoke(
        register_cli,
        [
            "run",
            str(ws),
            "7",
            "--codebook",
            str(cb),
            "--roi",
            "roiB",
            "--reference",
            "7_15_23",
            "--threshold",
            "7.5",
            "--fwhm",
            "3.5",
        ],
    )

    assert result.exit_code == 0, result.output
    cfg = seen["config"]
    assert pytest.approx(cfg.registration.fiducial.threshold, rel=0, abs=1e-6) == 7.5
    assert pytest.approx(cfg.registration.fiducial.fwhm, rel=0, abs=1e-6) == 3.5
    assert seen["roi"] == "roiB"
    assert seen["idx"] == 7
    assert seen["reference"] == "7_15_23"


def test_cli_register_run_skips_when_shifts_exist(tmp_path: Path, monkeypatch: Any) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    cb = _make_codebook(tmp_path)

    shift_dir = ws / "shifts--roiA+cb"
    shift_dir.mkdir(parents=True)
    (shift_dir / "shifts-0042.json").write_text("{}")

    called = False

    def fake__run(*_: Any, **__: Any) -> None:  # type: ignore[no-untyped-def]
        nonlocal called
        called = True

    monkeypatch.setattr("fishtools.preprocess.cli_register._run", fake__run)

    runner = CliRunner()
    result = runner.invoke(
        register_cli,
        [
            "run",
            str(ws),
            "42",
            "--codebook",
            str(cb),
            "--roi",
            "roiA",
            "--reference",
            "4_12_20",
        ],
    )

    assert result.exit_code == 0, result.output
    assert called is False


def test_run_internal_returns_early_when_shifts_exist(tmp_path: Path) -> None:
    shift_dir = tmp_path / "shifts--roiA+cb"
    shift_dir.mkdir(parents=True)
    (shift_dir / "shifts-0007.json").write_text("{}")

    cfg = Config(
        dataPath=str(DATA),
        exclude=None,
        registration=RegisterConfig(
            chromatic_shifts={
                "650": str(DATA / "560to650.txt"),
                "750": str(DATA / "560to750.txt"),
            },
            fiducial=Fiducial(
                use_fft=False,
                fwhm=4.0,
                threshold=6.0,
                priors={},
                overrides={},
                n_fids=2,
            ),
            reference="4_12_20",
            downsample=1,
            crop=40,
            slices=slice(None),
            reduce_bit_depth=0,
            discards=None,
        ),
    )

    _run(
        path=tmp_path,
        roi="roiA",
        idx=7,
        codebook=tmp_path / "cb.json",
        reference="4_12_20",
        config=cfg,
        debug=False,
        overwrite=False,
        no_priors=False,
    )

    assert not (tmp_path / "registered--roiA+cb").exists()


def test_cli_register_batch_skips_existing_shifts(tmp_path: Path, monkeypatch: Any) -> None:
    base = tmp_path / "analysis" / "deconv"
    (base / "2_10_18--roiA").mkdir(parents=True)
    (base / "2_10_18--roiA" / "2_10_18-0001.tif").write_text("")

    shift_dir = base / "shifts--roiA+cb"
    shift_dir.mkdir(parents=True)
    (shift_dir / "shifts-0001.json").write_text("{}")

    cb = _make_codebook(tmp_path)

    class _WS:
        def __init__(self, *_: Any, **__: Any) -> None:
            self.rois = ["roiA"]
            self.rounds = ["2_10_18"]

    monkeypatch.setattr("fishtools.preprocess.cli_register.Workspace", _WS)

    calls: list[list[str]] = []

    def fake_run(argv: list[str], check: bool) -> Any:  # type: ignore[no-untyped-def]
        calls.append(argv)

        class _R:
            returncode = 0

        return _R()

    monkeypatch.setattr("fishtools.preprocess.cli_register.subprocess.run", fake_run)

    runner = CliRunner()
    result = runner.invoke(
        register_cli,
        [
            "batch",
            str(base),
            "--codebook",
            str(cb),
            "--threads",
            "1",
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls == []
