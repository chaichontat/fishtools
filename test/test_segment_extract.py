from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from fishtools.segment import app as segment_app
from fishtools.segment.extract import _distribute_file_budget, _expand_positions_with_context


def test_expand_positions_with_context_orders_and_bounds() -> None:
    base = [10]
    result = _expand_positions_with_context(base, crop=0, axis_len=40, context_pairs=3, step=2)
    assert result[:7] == [10, 12, 8, 14, 6, 16, 4]
    assert result[-1] == 4


def test_expand_positions_with_context_deduplicates_and_respects_crop() -> None:
    base = [4, 6]
    result = _expand_positions_with_context(base, crop=2, axis_len=15, context_pairs=3, step=2)
    # Should clamp to [2, 12] and avoid duplicates when windows overlap.
    assert result[0] == 4
    assert result[1:5] == [6, 2, 8, 10]
    assert all(2 <= pos <= 12 for pos in result)


@pytest.fixture(autouse=True)
def _mock_zarr(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure Zarr access is never exercised during tests."""

    def _fail_open_array(*_: Any, **__: Any) -> None:
        raise AssertionError("zarr.open_array should not be invoked in tests.")

    monkeypatch.setattr("fishtools.segment.extract.zarr.open_array", _fail_open_array)


def test_distribute_file_budget_proportional_rounding() -> None:
    rois = ["roi_a", "roi_b", "roi_c"]
    counts = {"roi_a": 5, "roi_b": 3, "roi_c": 2}

    result = _distribute_file_budget(rois, counts, total=5)

    assert result == {"roi_a": 3, "roi_b": 1, "roi_c": 1}
    assert sum(result.values()) == 5
    # No ROI should receive more files than it has available.
    for roi, quota in result.items():
        assert quota <= counts[roi]


def test_distribute_file_budget_handles_edge_cases() -> None:
    rois = ["roi_a", "roi_b"]
    counts = {"roi_a": 1, "roi_b": 0}

    # Budget exceeds availability → return the per-ROI availability.
    assert _distribute_file_budget(rois, counts, total=5) == {"roi_a": 1, "roi_b": 0}
    # Zero budget should yield zeros even when files exist.
    assert _distribute_file_budget(rois, counts, total=0) == {"roi_a": 0, "roi_b": 0}


def test_extract_cli_delegates_across_rois(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace = tmp_path / "ws"
    (workspace / "analysis" / "deconv").mkdir(parents=True)
    (workspace / "analysis" / "deconv" / "registered--roi_a+cb1").mkdir()
    (workspace / "analysis" / "deconv" / "registered--roi_b+cb1").mkdir()
    (workspace / "workspace.DONE").write_text("")

    available_counts = {"roi_a": 3, "roi_b": 2}

    def _fake_discover_inputs(
        ws: Any,  # noqa: ANN401 - only used for interface compatibility
        current_roi: str,
        codebook: str,
        *,
        require_zarr: bool = False,
    ) -> list[Path]:
        assert codebook == "cb1"
        assert require_zarr is False
        return [
            Path(f"/fake/{current_roi}-{idx:02d}.tif") for idx in range(available_counts[current_roi])
        ]

    calls: list[dict[str, Any]] = []

    def _fake_extract_single_roi(**kwargs: Any) -> None:
        calls.append(kwargs)

    monkeypatch.setattr("fishtools.segment.extract._discover_registered_inputs", _fake_discover_inputs)
    monkeypatch.setattr("fishtools.segment.extract._extract_single_roi", _fake_extract_single_roi)

    runner = CliRunner()
    result = runner.invoke(
        segment_app,
        [
            "extract",
            "z",
            str(workspace),
            "--codebook",
            "cb1",
            "--n",
            "3",
            "--seed",
            "11",
        ],
        prog_name="segment",
    )

    assert result.exit_code == 0, result.output
    assert len(calls) == 2

    by_roi = {call["roi"]: call for call in calls}

    assert set(by_roi) == {"roi_a", "roi_b"}

    roi_a_call = by_roi["roi_a"]
    roi_b_call = by_roi["roi_b"]

    assert roi_a_call["file_quota"] == 2
    assert roi_b_call["file_quota"] == 1
    assert roi_a_call["seed"] == 11
    assert roi_b_call["seed"] == 12
    assert roi_a_call["mode"] == "z"
    assert roi_b_call["mode"] == "z"
    assert roi_a_call["codebook"] == "cb1"
    assert roi_b_call["codebook"] == "cb1"
    assert len(roi_a_call["prefetched_inputs"]) == available_counts["roi_a"]
    assert len(roi_b_call["prefetched_inputs"]) == available_counts["roi_b"]
    assert roi_a_call["use_zarr"] is False
    assert roi_b_call["use_zarr"] is False


def test_extract_cli_single_roi_argument(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace = tmp_path / "ws"
    (workspace / "analysis" / "deconv").mkdir(parents=True)
    (workspace / "analysis" / "deconv" / "registered--roi_c+cb1").mkdir()
    (workspace / "workspace.DONE").write_text("")

    def _fake_discover_inputs(
        ws: Any,  # noqa: ANN401
        current_roi: str,
        codebook: str,
        *,
        require_zarr: bool = False,
    ) -> list[Path]:
        assert codebook == "cb1"
        assert current_roi == "roi_c"
        assert require_zarr is False
        return [Path("/fake/roi_c-00.tif"), Path("/fake/roi_c-01.tif")]

    calls: list[dict[str, Any]] = []

    def _fake_extract_single_roi(**kwargs: Any) -> None:
        calls.append(kwargs)

    monkeypatch.setattr("fishtools.segment.extract._discover_registered_inputs", _fake_discover_inputs)
    monkeypatch.setattr("fishtools.segment.extract._extract_single_roi", _fake_extract_single_roi)

    runner = CliRunner()
    result = runner.invoke(
        segment_app,
        [
            "extract",
            "z",
            str(workspace),
            "roi_c",
            "--codebook",
            "cb1",
            "--n",
            "2",
        ],
        prog_name="segment",
    )

    assert result.exit_code == 0, result.output
    assert calls and calls[-1]["roi"] == "roi_c"
    assert calls[-1]["file_quota"] is None


def test_extract_cli_zarr_forces_upscale(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace = tmp_path / "ws"
    (workspace / "analysis" / "deconv").mkdir(parents=True)
    (workspace / "analysis" / "deconv" / "registered--roi_d+cb1").mkdir()
    (workspace / "workspace.DONE").write_text("")

    def _fake_discover_inputs(
        ws: Any,  # noqa: ANN401
        current_roi: str,
        codebook: str,
        *,
        require_zarr: bool = False,
    ) -> list[Path]:
        assert current_roi == "roi_d"
        assert codebook == "cb1"
        assert require_zarr is True
        return [Path("/fake/roi_d.zarr")]

    calls: list[dict[str, Any]] = []

    def _fake_extract_single_roi(**kwargs: Any) -> None:
        calls.append(kwargs)

    monkeypatch.setattr("fishtools.segment.extract._discover_registered_inputs", _fake_discover_inputs)
    monkeypatch.setattr("fishtools.segment.extract._extract_single_roi", _fake_extract_single_roi)

    runner = CliRunner()
    result = runner.invoke(
        segment_app,
        [
            "extract",
            "z",
            str(workspace),
            "roi_d",
            "--codebook",
            "cb1",
            "--zarr",
            "--upscale",
            "5.0",
        ],
        prog_name="segment",
    )

    assert result.exit_code == 0, result.output
    assert calls and math.isclose(calls[-1]["upscale"], 2.0)
