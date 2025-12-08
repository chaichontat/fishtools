from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from fishtools.preprocess.cli_check_shifts import check_shifts as check_shifts_cli


class DummyWorkspace:
    """Minimal Workspace stub recording resolve_rois input."""

    last_rois: list[str] | None

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        DummyWorkspace.last_rois = None

    @property
    def deconved(self) -> Path:
        return self.path / "analysis" / "deconv"

    def resolve_rois(self, rois: list[str] | None = None) -> list[str]:
        DummyWorkspace.last_rois = rois
        if rois is None:
            return ["roiA", "roiB"]
        return list(rois)


class DummyCodebook:
    """Lightweight Codebook stub exposing .name."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    @property
    def name(self) -> str:
        return self.path.stem


@pytest.fixture()
def cli_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[Path, Path, list[dict[str, Any]]]:
    workspace = tmp_path / "ws"
    workspace.mkdir()

    codebook_path = tmp_path / "cb.json"
    codebook_path.write_text("{}")

    log_calls: list[dict[str, Any]] = []

    def fake_setup_cli_logging(
        path: Path | None,
        *,
        component: str,
        file: str,
        idx: int | None = None,
        debug: bool = False,
        extra: dict[str, Any] | None = None,
        **_: Any,
    ) -> Path:
        log_calls.append({
            "path": Path(path) if path is not None else None,
            "component": component,
            "file": file,
            "idx": idx,
            "debug": debug,
            "extra": extra or {},
        })
        if path is None:
            return Path("analysis/logs") / f"{file}.log"
        return Path(path) / "analysis" / "logs" / f"{file}.log"

    def fake_load_shifts(_: Path) -> dict[int, dict[str, Any]]:
        # Return empty mapping to skip heavy plotting/path logic.
        return {}

    def fake_check_missing_tiles(*_: Any, **__: Any) -> None:
        return None

    monkeypatch.setattr("fishtools.preprocess.cli_check_shifts.Workspace", DummyWorkspace)
    monkeypatch.setattr("fishtools.preprocess.cli_check_shifts.Codebook", DummyCodebook)
    monkeypatch.setattr("fishtools.preprocess.cli_check_shifts._load_shifts", fake_load_shifts)
    monkeypatch.setattr("fishtools.preprocess.cli_check_shifts._check_missing_tiles", fake_check_missing_tiles)
    monkeypatch.setattr(
        "fishtools.preprocess.cli_check_shifts.setup_cli_logging",
        fake_setup_cli_logging,
    )

    return workspace, codebook_path, log_calls


def test_check_shifts_uses_all_rois_when_none_provided(cli_env: tuple[Path, Path, list[dict[str, Any]]]) -> None:
    workspace, codebook_path, log_calls = cli_env

    runner = CliRunner()
    result = runner.invoke(
        check_shifts_cli,
        [
            str(workspace),
            "--codebook",
            str(codebook_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert DummyWorkspace.last_rois is None
    assert log_calls
    assert log_calls[0]["extra"].get("roi") == "all"


def test_check_shifts_accepts_positional_roi(cli_env: tuple[Path, Path, list[dict[str, Any]]]) -> None:
    workspace, codebook_path, log_calls = cli_env

    runner = CliRunner()
    result = runner.invoke(
        check_shifts_cli,
        [
            str(workspace),
            "roiX",
            "--codebook",
            str(codebook_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert DummyWorkspace.last_rois == ["roiX"]
    assert log_calls
    assert log_calls[0]["extra"].get("roi") == "roiX"


def test_check_shifts_accepts_multiple_roi_options(cli_env: tuple[Path, Path, list[dict[str, Any]]]) -> None:
    workspace, codebook_path, log_calls = cli_env

    runner = CliRunner()
    result = runner.invoke(
        check_shifts_cli,
        [
            str(workspace),
            "--codebook",
            str(codebook_path),
            "--roi",
            "roi1",
            "--roi",
            "roi2",
        ],
    )

    assert result.exit_code == 0, result.output
    assert DummyWorkspace.last_rois == ["roi1", "roi2"]
    assert log_calls
    assert log_calls[0]["extra"].get("roi") == "roi1,roi2"


def test_check_shifts_rejects_conflicting_roi_arguments(cli_env: tuple[Path, Path, list[dict[str, Any]]]) -> None:
    workspace, codebook_path, _ = cli_env

    runner = CliRunner()
    result = runner.invoke(
        check_shifts_cli,
        [
            str(workspace),
            "roiX",
            "--codebook",
            str(codebook_path),
            "--roi",
            "roiY",
        ],
    )

    assert result.exit_code != 0
    assert "Specify ROI either as a positional argument or via --roi" in result.output

