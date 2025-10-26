from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from tifffile import imread, imwrite
from typer.testing import CliRunner

from fishtools.segment import app as segment_app
from fishtools.segment.extract import (
    _distribute_file_budget,
    _expand_positions_with_context,
    _open_mask_volume,
    _process_file_to_ortho,
    _process_file_to_z_slices,
    _squeeze_mask,
)


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


def test_extract_single_cli_delegates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    reg_file = tmp_path / "reg-00.tif"
    reg_file.write_bytes(b"")

    captured: dict[str, Any] = {}

    def _fake_execute(**kwargs: Any) -> None:
        captured.update(kwargs)

    monkeypatch.setattr("fishtools.segment.extract._execute_extraction", _fake_execute)

    runner = CliRunner()
    result = runner.invoke(
        segment_app,
        [
            "extract-single",
            "z",
            str(reg_file),
            "--n",
            "5",
        ],
        prog_name="segment",
    )

    assert result.exit_code == 0, result.output
    assert captured["label"] == "reg-00"
    assert captured["files"] == [reg_file.resolve()]
    assert captured["mode"] == "z"
    assert captured["n"] == 5
    assert captured["out_dir"] == reg_file.parent / "segment_extract"
    assert captured["max_from_path"] is None


def test_extract_single_cli_with_max_from(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    reg_file = tmp_path / "reg-01.tif"
    reg_file.write_bytes(b"")
    max_dir = tmp_path / "other_cb"
    max_dir.mkdir()
    (max_dir / reg_file.name).write_bytes(b"")

    validate_args: dict[str, Any] = {}

    def _fake_validate(source: Path, files: list[Path], *, label: str) -> None:
        validate_args["source"] = source
        validate_args["files"] = files
        validate_args["label"] = label

    monkeypatch.setattr("fishtools.segment.extract._validate_max_from_path", _fake_validate)
    monkeypatch.setattr("fishtools.segment.extract._execute_extraction", lambda **_: None)

    runner = CliRunner()
    result = runner.invoke(
        segment_app,
        [
            "extract-single",
            "ortho",
            str(reg_file),
            "--max-from",
            str(max_dir),
            "--label",
            "custom",
        ],
        prog_name="segment",
    )

    assert result.exit_code == 0, result.output
    assert validate_args["source"] == max_dir.resolve()
    assert validate_args["files"] == [reg_file.resolve()]
    assert validate_args["label"] == "custom"


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


def test_process_file_to_z_slices_includes_masks(tmp_path: Path) -> None:
    roi = "roi_a"
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    volume = np.arange(2 * 2 * 3 * 3, dtype=np.uint16).reshape(2, 2, 3, 3)
    mask = (np.arange(2 * 3 * 3, dtype=np.uint16).reshape(2, 3, 3) * 5).astype(np.uint16)

    reg_file = tmp_path / "reg-00.tif"
    mask_file = tmp_path / "reg-00_masks.tif"
    imwrite(reg_file, volume, metadata={"axes": "ZCYX"})
    imwrite(mask_file, mask, metadata={"axes": "ZYX"})

    _process_file_to_z_slices(
        reg_file,
        roi,
        out_dir,
        channels="0,1",
        crop=0,
        dz=1,
        n=2,
        max_from_path=None,
        upscale=1.0,
        mask_path=mask_file,
        progress=None,
    )

    mask_output = out_dir / f"{roi}--reg-00_z00_masks.tif"
    assert mask_output.exists()
    written_mask = imread(mask_output)
    mask_volume = _open_mask_volume(mask_file)
    np.testing.assert_array_equal(written_mask, _squeeze_mask(mask_volume[0, ...]))


def test_process_file_to_ortho_includes_masks(tmp_path: Path) -> None:
    roi = "roi_b"
    out_dir = tmp_path / "ortho"
    out_dir.mkdir()

    volume = np.arange(2 * 2 * 4 * 4, dtype=np.uint16).reshape(2, 2, 4, 4)
    mask = (np.arange(2 * 4 * 4, dtype=np.uint16).reshape(2, 4, 4) + 3).astype(np.uint16)

    reg_file = tmp_path / "reg-01.tif"
    mask_file = tmp_path / "reg-01_masks.tif"
    imwrite(reg_file, volume, metadata={"axes": "ZCYX"})
    imwrite(mask_file, mask, metadata={"axes": "ZYX"})

    _process_file_to_ortho(
        reg_file,
        roi,
        out_dir,
        channels="0,1",
        crop=0,
        n=1,
        anisotropy=2,
        max_from_path=None,
        upscale=1.0,
        progress=None,
        mask_path=mask_file,
    )

    mask_outputs = sorted(out_dir.glob(f"{roi}--reg-01_*_masks.tif"))
    assert mask_outputs, "Expected mask ortho slices to be written."
    for mask_path in mask_outputs:
        arr = imread(mask_path)
        assert arr.dtype == mask.dtype
