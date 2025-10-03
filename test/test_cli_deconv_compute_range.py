from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from fishtools.preprocess.cli_deconv_old import (
    DEFAULT_PERCENTILE_OVERRIDES,
    _get_percentiles_for_round,
    deconv as deconv_cli,
)


def test_compute_range_requires_deconv_directory(tmp_path: Path) -> None:
    """compute-range must be invoked within a path containing 'deconv'."""
    runner = CliRunner()
    result = runner.invoke(deconv_cli, ["compute-range", str(tmp_path)])
    assert result.exit_code != 0
    assert isinstance(result.exception, ValueError)
    assert "must be run within a 'deconv' directory" in str(result.exception)


def test_compute_range_creates_scaling_files(monkeypatch: Any, tmp_path: Path) -> None:
    """compute-range calls _compute_range per round and writes deconv_scaling files.

    We stub Workspace and _compute_range to avoid heavy IO.
    """
    base = tmp_path / "analysis" / "deconv"
    base.mkdir(parents=True)

    # Fake Workspace with a single round
    class _WS:
        def __init__(self, *_: Any, **__: Any) -> None:
            self.rounds = ["1_2_3"]

    called: dict[str, Any] = {}

    def fake_get_perc(round_name: str, pm: float, ps: float, max_rna_bit: int, override: dict[str, tuple[float, float]] | None = None):  # type: ignore[no-untyped-def]
        # Return deterministic per-bit percentiles
        called["override"] = override
        return [10.0, 20.0, 30.0], [0.1, 0.2, 0.3]

    def fake_compute(path: Path, round_: str, *, perc_min, perc_scale):  # type: ignore[no-untyped-def]
        called["args"] = {
            "path": path,
            "round": round_,
            "perc_min": perc_min,
            "perc_scale": perc_scale,
        }
        # Simulate write performed by real _compute_range
        (path / "deconv_scaling").mkdir(exist_ok=True)
        (path / "deconv_scaling" / f"{round_}.txt").write_text("1 1 1\n1 1 1\n")

    monkeypatch.setattr("fishtools.preprocess.cli_deconv_old.Workspace", _WS)
    monkeypatch.setattr("fishtools.preprocess.cli_deconv_old._get_percentiles_for_round", fake_get_perc)
    monkeypatch.setattr("fishtools.preprocess.cli_deconv_old._compute_range", fake_compute)

    runner = CliRunner()
    result = runner.invoke(deconv_cli, ["compute-range", str(base), "--overwrite"])  # ensure not skipped

    assert result.exit_code == 0, result.output
    out = base / "deconv_scaling" / "1_2_3.txt"
    assert out.exists()
    # Confirm our fake received per-bit lists
    assert called["args"]["perc_min"] == [10.0, 20.0, 30.0]
    assert called["args"]["perc_scale"] == [0.1, 0.2, 0.3]
    assert called["override"] is not None
    for key, value in {"edu": (5.0, 5.0), "brdu": (5.0, 5.0), "wga": (50.0, 50.0), "pi": (1.0, 1.0)}.items():
        assert called["override"].get(key) == value


def test_compute_range_skips_rounds_without_json(monkeypatch: Any, tmp_path: Path) -> None:
    """If a round has no .deconv.json sidecars, compute-range should skip it."""
    base = tmp_path / "analysis" / "deconv"
    (base / "r1--roiA").mkdir(parents=True)
    # Write a tif but no adjacent .deconv.json
    (base / "r1--roiA" / "r1-0001.tif").write_bytes(b"TIFFPLACEHOLDER")

    class _WS:
        def __init__(self, *_: Any, **__: Any) -> None:
            self.rounds = ["r1"]

    monkeypatch.setattr("fishtools.preprocess.cli_deconv_old.Workspace", _WS)

    # If _compute_range is called, we'll record it (should not be called)
    called = {"count": 0}

    def fake_compute(*args: Any, **kwargs: Any) -> None:
        called["count"] += 1

    monkeypatch.setattr("fishtools.preprocess.cli_deconv_old._compute_range", fake_compute)

    runner = CliRunner()
    result = runner.invoke(deconv_cli, ["compute-range", str(base)])
    assert result.exit_code == 0, result.output
    assert called["count"] == 0  # skipped
    assert not (base / "deconv_scaling" / "r1.txt").exists()


def test_compute_range_override_validation(monkeypatch: Any, tmp_path: Path) -> None:
    """Invalid overrides should raise a helpful error when bit not present in rounds."""
    base = tmp_path / "analysis" / "deconv"
    base.mkdir(parents=True)

    class _WS:
        def __init__(self, *_: Any, **__: Any) -> None:
            self.rounds = ["1_2_3"]

    monkeypatch.setattr("fishtools.preprocess.cli_deconv_old.Workspace", _WS)

    runner = CliRunner()
    # bit '9' not present in round '1_2_3'
    result = runner.invoke(
        deconv_cli,
        [
            "compute-range",
            str(base),
            "--override",
            "9,5,6",
        ],
    )
    assert result.exit_code != 0
    assert isinstance(result.exception, ValueError)
    assert "Overridden bits must exist" in str(result.exception)


def test_get_percentiles_for_round_applies_default_overrides() -> None:
    """ensure hardcoded overrides take effect for matching bits."""
    mins, scales = _get_percentiles_for_round(
        "edu_5_pi",
        0.2,
        0.3,
        max_rna_bit=36,
        override=DEFAULT_PERCENTILE_OVERRIDES.copy(),
    )

    assert mins == [5.0, 0.2, 1.0]
    assert scales == [5.0, 0.3, 1.0]
