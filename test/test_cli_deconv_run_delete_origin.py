from __future__ import annotations

from pathlib import Path
from typing import Any

from click.testing import CliRunner

from fishtools.preprocess.cli_deconv import (
    _PrefixedLogger,
    _RoundProcessingPlan,
    deconv as deconv_cli,
)


def _touch(path: Path, data: bytes = b"x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def test_run_delete_origin_removes_source_dirs(monkeypatch: Any, tmp_path: Path) -> None:
    """run with --delete-origin removes source folders once outputs exist."""

    round_name = "1_2_3"
    rois = ("roiA", "roiB")
    expected_tiles: list[Path] = []
    for roi in rois:
        for idx in (0, 1):
            tile = tmp_path / f"{round_name}--{roi}" / f"{round_name}-{idx:04d}.tif"
            _touch(tile)
            expected_tiles.append(tile)

    prepared_files: list[list[Path]] = []
    progress_calls: list[Path] = []
    progress_ids: set[int] = set()

    monkeypatch.setattr("fishtools.preprocess.cli_deconv.parse_device_spec", lambda _spec: [0])
    monkeypatch.setattr("fishtools.preprocess.cli_deconv._configure_logging", lambda *_args, **_kwargs: None)

    def fake_prepare(**kwargs: Any) -> _RoundProcessingPlan | None:  # type: ignore[override]
        files = list(kwargs["files"])
        prepared_files.append(files)
        return _RoundProcessingPlan(
            label=kwargs["label"],
            prefixed=kwargs["prefixed"],
            files=files,
            pending=list(files),
            processor_factory=lambda _device: None,
            out_dir=kwargs["out_dir"],
        )

    def fake_execute(
        plan: _RoundProcessingPlan,
        *,
        devices: Any,
        stop_on_error: bool,
        debug: bool,
        progress: Any,
    ) -> list[Any]:  # type: ignore[override]
        progress_ids.add(id(progress))
        for path in plan.pending:
            out_dir = plan.out_dir / path.parent.name
            out_dir.mkdir(parents=True, exist_ok=True)
            _touch(out_dir / path.name)
            progress()
            progress_calls.append(path)
        return []

    monkeypatch.setattr("fishtools.preprocess.cli_deconv._prepare_round_plan", fake_prepare)
    monkeypatch.setattr("fishtools.preprocess.cli_deconv._execute_round_plan", fake_execute)

    runner = CliRunner()
    result = runner.invoke(deconv_cli, ["run", str(tmp_path), "--delete-origin"])

    assert result.exit_code == 0, result.output

    # One plan per round, containing all ROI tiles.
    assert prepared_files == [expected_tiles]
    # Shared progress bar is reused for all tiles.
    assert len(progress_calls) == len(expected_tiles)
    assert len(progress_ids) == 1

    # Source folders removed while outputs retained.
    for roi in rois:
        assert not (tmp_path / f"{round_name}--{roi}").exists()
        assert (tmp_path / "analysis" / "deconv" / f"{round_name}--{roi}").exists()


def test_run_without_delete_origin_keeps_sources(monkeypatch: Any, tmp_path: Path) -> None:
    """run without --delete-origin leaves raw tiles in place."""

    round_name = "1_2_3"
    roi = "roiA"
    tile = tmp_path / f"{round_name}--{roi}" / f"{round_name}-0000.tif"
    _touch(tile)

    monkeypatch.setattr("fishtools.preprocess.cli_deconv.parse_device_spec", lambda _spec: [0])
    monkeypatch.setattr("fishtools.preprocess.cli_deconv._configure_logging", lambda *_args, **_kwargs: None)

    plan = _RoundProcessingPlan(
        label=round_name,
        prefixed=_PrefixedLogger(round_name),
        files=[tile],
        pending=[tile],
        processor_factory=lambda _device: None,
        out_dir=tmp_path / "analysis" / "deconv",
    )

    def fake_prepare(**kwargs: Any) -> _RoundProcessingPlan | None:  # type: ignore[override]
        return plan

    def fake_execute(
        plan: _RoundProcessingPlan,
        *,
        devices: Any,
        stop_on_error: bool,
        debug: bool,
        progress: Any,
    ) -> list[Any]:  # type: ignore[override]
        out_dir = plan.out_dir / tile.parent.name
        out_dir.mkdir(parents=True, exist_ok=True)
        _touch(out_dir / tile.name)
        progress()
        return []

    monkeypatch.setattr("fishtools.preprocess.cli_deconv._prepare_round_plan", fake_prepare)
    monkeypatch.setattr("fishtools.preprocess.cli_deconv._execute_round_plan", fake_execute)

    runner = CliRunner()
    result = runner.invoke(deconv_cli, ["run", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert (tmp_path / f"{round_name}--{roi}").exists()
