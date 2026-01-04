from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import tifffile

import pytest
from click.testing import CliRunner

from fishtools.segment import app as segment_app
from fishtools.segment.extract_helpers import (
    _distribute_file_budget,
    _expand_positions_with_context,
    _score_and_select_tiles,
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

    monkeypatch.setattr("fishtools.segment.extract_core.zarr.open_array", _fail_open_array)


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


def test_score_and_select_tiles_respects_score_function() -> None:
    mask = np.array(
        [
            [
                [1, 1, 1],
                [2, 2, 2],
                [3, 4, 0],
            ]
        ]
    )

    candidates = [(0, 0), (1, 1)]

    coverage_top = _score_and_select_tiles(
        candidates,
        mask,
        tile_size=2,
        count=1,
        score_fn=lambda tile: int(np.sum(tile > 0)),
    )
    # Coverage favours the densely filled (0,0) tile
    assert coverage_top == [(0, 0)]

    diversity_top = _score_and_select_tiles(
        candidates,
        mask,
        tile_size=2,
        count=1,
        score_fn=lambda tile: len(np.unique(tile)),
    )
    # Diversity favours the tile containing labels 0,2,4
    assert diversity_top == [(1, 1)]


def test_resolve_enrich_mask_defaults_and_disable(tmp_path: Path) -> None:
    from fishtools.segment.extract_helpers import _resolve_enrich_mask
    from fishtools.io.workspace import Workspace

    ws_root = tmp_path / "ws"
    stitch_dir = ws_root / "analysis" / "deconv" / "stitch--roi_a+cb1"
    stitch_dir.mkdir(parents=True, exist_ok=True)
    (stitch_dir / "output_segmentation-sam.zarr").mkdir()
    (ws_root / "workspace.DONE").write_text("")

    ws = Workspace(ws_root)

    resolved = _resolve_enrich_mask(ws, "roi_a", "cb1", enrich_boundaries=None, enable=True)
    assert resolved == stitch_dir / "output_segmentation-sam.zarr"

    disabled = _resolve_enrich_mask(ws, "roi_a", "cb1", enrich_boundaries=None, enable=False)
    assert disabled is None


def test_extract_z_slices_writes_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import fishtools.segment.extract_core as extract_core_mod
    from fishtools.segment.extract_core import _extract_z_slices

    # Avoid GPU requirement in unsharp_all during tests.
    monkeypatch.setattr(extract_core_mod, "unsharp_all", lambda img, **_: np.asarray(img))

    reg_path = tmp_path / "reg-00.tif"
    reg_data = np.zeros((2, 2, 6, 6), dtype=np.uint16)
    tifffile.imwrite(reg_path, reg_data)

    mask_path = tmp_path / "reg-00_masks.tif"
    mask_data = np.zeros((2, 6, 6), dtype=np.uint16)
    tifffile.imwrite(mask_path, mask_data)

    out_dir = tmp_path / "out"
    out_dir.mkdir()

    _extract_z_slices(
        file=reg_path,
        roi="roi_a",
        out_dir=out_dir,
        channels=None,
        dz=1,
        n_crops=1,
        upscale=1.0,
        max_from_path=None,
        mask_path=mask_path,
        enrich_boundaries=None,
        seed=0,
        progress=None,
    )

    outputs = list(out_dir.glob("*.tif"))
    assert any(p.name.endswith("_z00.tif") for p in outputs)
    assert any(p.name.endswith("_z01.tif") for p in outputs)


def test_extract_z_slices_supports_single_channel(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import fishtools.segment.extract_core as extract_core_mod
    from fishtools.segment.extract_core import _extract_z_slices

    # Avoid GPU requirement in unsharp_all during tests.
    monkeypatch.setattr(extract_core_mod, "unsharp_all", lambda img, **_: np.asarray(img))

    reg_path = tmp_path / "reg-00.tif"
    reg_data = np.zeros((2, 1, 6, 6), dtype=np.uint16)
    tifffile.imwrite(reg_path, reg_data)

    out_dir = tmp_path / "out"
    out_dir.mkdir()

    _extract_z_slices(
        file=reg_path,
        roi="roi_a",
        out_dir=out_dir,
        channels=None,
        dz=1,
        n_crops=1,
        upscale=1.0,
        max_from_path=None,
        mask_path=None,
        enrich_boundaries=None,
        seed=0,
        progress=None,
    )

    outputs = list(out_dir.glob("*.tif"))
    assert any(p.name.endswith("_z00.tif") for p in outputs)
    assert any(p.name.endswith("_z01.tif") for p in outputs)


def test_append_max_from_avoids_global_max(monkeypatch: pytest.MonkeyPatch) -> None:
    import fishtools.segment.extract_core as extract_core_mod

    class FakeVolume:
        def __init__(self, data: np.ndarray):
            self._data = data

        @property
        def shape(self) -> tuple[int, ...]:
            return self._data.shape

        @property
        def dtype(self) -> np.dtype:
            return self._data.dtype

        def __getitem__(self, key: Any) -> np.ndarray:
            return self._data[key]

        def max(self, *args: Any, **kwargs: Any) -> np.ndarray:  # noqa: D401
            raise RuntimeError("Global max should not be called on FakeVolume")

    other_data = np.stack(
        [np.full((2, 3, 3), fill_value=z, dtype=np.uint16) for z in range(130)]
    )  # (Z, C, Y, X)
    img = np.ones((130, 2, 3, 3), dtype=np.uint16)

    def _fake_load_registered_stack(path: Path):
        return FakeVolume(other_data), None, path.name

    monkeypatch.setattr(extract_core_mod, "_load_registered_stack", _fake_load_registered_stack)

    new_img, appended_idx = extract_core_mod._append_max_from(img, Path("reg-00.tif"), Path("max.tif"))

    expected = np.concatenate([img, other_data.max(axis=1, keepdims=True)], axis=1)
    np.testing.assert_array_equal(new_img, expected)
    assert appended_idx == img.shape[1]


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
        return [Path(f"/fake/{current_roi}-{idx:02d}.tif") for idx in range(available_counts[current_roi])]

    calls: list[dict[str, Any]] = []

    def _fake_extract_single_roi(**kwargs: Any) -> None:
        calls.append(kwargs)

    monkeypatch.setattr("fishtools.segment.extract_core._discover_registered_inputs", _fake_discover_inputs)
    monkeypatch.setattr("fishtools.segment.extract_core._extract_single_roi", _fake_extract_single_roi)

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
    assert roi_a_call["z_crops_per_file"] == 1
    assert roi_b_call["z_crops_per_file"] == 1


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

    monkeypatch.setattr("fishtools.segment.extract_core._discover_registered_inputs", _fake_discover_inputs)
    monkeypatch.setattr("fishtools.segment.extract_core._extract_single_roi", _fake_extract_single_roi)

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

    monkeypatch.setattr("fishtools.segment.extract_core._execute_extraction", _fake_execute)

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
    # In z mode, extract-single should always emit a single crop,
    # so the effective n passed into the core extraction is 1.
    assert captured["n"] == 1
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

    monkeypatch.setattr("fishtools.segment.extract_core._validate_max_from_path", _fake_validate)
    monkeypatch.setattr("fishtools.segment.extract_core._execute_extraction", lambda **_: None)

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


def test_extract_single_directory_respects_overwrite_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Directory-mode extract-single should skip existing outputs unless --overwrite is set."""
    reg_dir = tmp_path / "regdir"
    reg_dir.mkdir()
    reg0 = reg_dir / "reg-00.tif"
    reg1 = reg_dir / "reg-01.tif"
    reg0.write_bytes(b"0")
    reg1.write_bytes(b"1")

    # Pre-create outputs for reg-00 to exercise idempotency.
    out_dir = reg_dir / "segment_extract"
    out_dir.mkdir()
    existing = out_dir / "reg-00--reg-00_z00.tif"
    existing.write_bytes(b"x")

    calls: list[Path] = []

    def _fake_cmd_extract_single(
        mode: str,
        registered: Path,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        calls.append(registered)

    monkeypatch.setattr("fishtools.segment.extract.cmd_extract_single", _fake_cmd_extract_single)

    runner = CliRunner()

    # Without --overwrite: reg-00 should be skipped, reg-01 processed.
    result = runner.invoke(
        segment_app,
        [
            "extract-single",
            "z",
            str(reg_dir),
        ],
        prog_name="segment",
    )

    assert result.exit_code == 0, result.output
    assert "Skipping reg-00.tif" in result.output
    assert calls == [reg1]

    calls.clear()

    # With --overwrite: both files should be processed.
    result = runner.invoke(
        segment_app,
        [
            "extract-single",
            "z",
            str(reg_dir),
            "--overwrite",
        ],
        prog_name="segment",
    )

    assert result.exit_code == 0, result.output
    assert set(calls) == {reg0, reg1}


def test_extract_single_z_mode_outputs_single_crop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """extract-single z should produce exactly one crop per Z-plane."""
    import fishtools.segment.extract_core as extract_core_mod

    # Avoid GPU requirement in unsharp_all during tests.
    monkeypatch.setattr(extract_core_mod, "unsharp_all", lambda img, **_: np.asarray(img))

    # Create a registered TIFF large enough to allow multiple crop positions.
    reg_file = tmp_path / "reg-00.tif"
    z, c, y, x = 3, 2, 1025, 1025  # y/x > DEFAULT_CROP_SIZE (=1024) → multiple crops possible
    data = np.zeros((z, c, y, x), dtype=np.uint16)
    tifffile.imwrite(reg_file, data)

    runner = CliRunner()
    result = runner.invoke(
        segment_app,
        [
            "extract-single",
            "z",
            str(reg_file),
            "--n",
            "10",
        ],
        prog_name="segment",
    )

    assert result.exit_code == 0, result.output

    out_dir = reg_file.parent / "segment_extract"
    outputs = list(out_dir.glob("*.tif"))
    # We expect one output per Z plane, but no multiple crop indices.
    assert len(outputs) == z
    assert all("_crop" not in p.name for p in outputs)


def test_extract_single_rejects_zarr_input(tmp_path: Path) -> None:
    """extract-single should refuse Zarr inputs (use segment extract instead)."""
    # Use a file-suffixed .zarr path so the Typer callback runs and rejects it.
    reg_zarr = tmp_path / "reg-00.zarr"
    reg_zarr.write_bytes(b"")

    runner = CliRunner()
    result = runner.invoke(
        segment_app,
        [
            "extract-single",
            "z",
            str(reg_zarr),
        ],
        prog_name="segment",
    )

    assert result.exit_code != 0
    # The error message should indicate Zarr inputs are unsupported.
    assert result.exception is not None
    assert "Zarr input is not supported for extract-single" in str(result.exception)


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

    monkeypatch.setattr("fishtools.segment.extract_core._discover_registered_inputs", _fake_discover_inputs)
    monkeypatch.setattr("fishtools.segment.extract_core._extract_single_roi", _fake_extract_single_roi)

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


def test_extract_cli_out_dir_requires_overwrite(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """segment extract must require --overwrite when --out points to a non-empty directory."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "workspace.DONE").write_text("")

    out_dir = tmp_path / "custom_out"
    out_dir.mkdir()
    (out_dir / "dummy.txt").write_text("x")

    # Avoid hitting the Typer/segment internals; we only care about the guard.
    monkeypatch.setattr("fishtools.segment.extract.cmd_extract", lambda *args, **kwargs: None)

    runner = CliRunner()
    result = runner.invoke(
        segment_app,
        [
            "extract",
            "z",
            str(workspace),
            "--codebook",
            "cb1",
            "--out",
            str(out_dir),
        ],
        prog_name="segment",
    )

    assert result.exit_code != 0
    assert result.exception is not None
    assert "use --overwrite" in str(result.exception)

    # With --overwrite the guard should allow execution.
    result_ok = runner.invoke(
        segment_app,
        [
            "extract",
            "z",
            str(workspace),
            "--codebook",
            "cb1",
            "--out",
            str(out_dir),
            "--overwrite",
        ],
        prog_name="segment",
    )

    assert result_ok.exit_code == 0, result_ok.output


def test_ortho_context_slices_share_perpendicular_position(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Context slices must share the same perpendicular position as their base slice.

    When extracting orthozx (Y slices), each base Y position gets context slices
    (Y ± 2, 4, 6, ...). All context slices for a given base should have the
    SAME perpendicular X crop, not independently random X crops.
    """
    # Track perpendicular slices passed to _process_ortho_slice
    slice_calls: list[dict[str, Any]] = []

    def _fake_process_ortho_slice(
        *,
        vol: Any,
        mask_vol: Any,
        other_vol: Any,
        position: int,
        axis: str,
        perpendicular_slice: slice,
        selected_indices: list[int],
        out_names: list[str],
        anisotropy: int,
        upscale: float,
        out_dir: Path,
        file_stem: str,
        roi: str,
        channels: str | None,
    ) -> None:
        slice_calls.append(
            {
                "position": position,
                "axis": axis,
                "perpendicular_slice": perpendicular_slice,
            }
        )

    monkeypatch.setattr("fishtools.segment.extract_core._process_ortho_slice", _fake_process_ortho_slice)

    # Create fake volume data (Z, Y, X, C) - large enough to trigger random sampling
    # MAX_WIDTH_AFTER_UPSCALE is 4096, with upscale=1.0, max_width_pre_upscale=4096
    # Volume X=8000 with crop=100 leaves 7800 width, much larger than 4096
    # So _compute_perpendicular_slice must randomly sample within 7800
    import numpy as np

    fake_vol = np.zeros((10, 8000, 8000, 3), dtype=np.uint16)
    fake_names = ["ch1", "ch2", "ch3"]

    def _fake_open_volume(path: Path) -> tuple[Any, list[str]]:
        return fake_vol, fake_names

    monkeypatch.setattr("fishtools.segment.extract_core._open_volume", _fake_open_volume)

    # Disable mask loading
    monkeypatch.setattr("fishtools.segment.extract_core._resolve_mask_path", lambda f: None)
    monkeypatch.setattr("fishtools.segment.extract_core._open_mask_volume", lambda p: None)

    # Create a fake registered file
    reg_file = tmp_path / "reg-00.zarr"
    reg_file.mkdir()

    from fishtools.segment.extract_core import (
        _execute_zarr_ortho_extraction,
    )
    from fishtools.segment.extract_helpers import ExtractionConfig

    config = ExtractionConfig(
        mode="ortho",
        channels=None,
        crop=100,
        dz=1,
        n=1,
        anisotropy=4,
        upscale=1.0,
        seed=42,
        threads=1,
    )

    _execute_zarr_ortho_extraction(
        label="test",
        files=[reg_file],
        config=config,
        out_dir=tmp_path / "output",
        max_from_path=None,
        explicit_mask_path=None,
        enrich_boundaries=None,
    )

    # With n=1 and context_pairs=10, step=2, we expect:
    # - 1 base position expanded to ~21 positions per axis
    # - All positions for the same base should share the same perpendicular slice
    assert len(slice_calls) > 2, "Expected context slices to be generated"

    # Group Y slices by their base position
    y_slices = [c for c in slice_calls if c["axis"] == "y"]
    assert len(y_slices) > 1, f"Expected multiple Y slices (context), got {len(y_slices)}"

    # All Y slices should share the same perpendicular X slice
    # (This is the bug - currently each gets a different random X slice)
    perp_slices = [c["perpendicular_slice"] for c in y_slices]
    first_perp = perp_slices[0]
    for i, perp in enumerate(perp_slices[1:], 1):
        assert perp.start == first_perp.start and perp.stop == first_perp.stop, (
            f"Context slice {i} has different perpendicular position "
            f"({perp.start}:{perp.stop}) than base ({first_perp.start}:{first_perp.stop}). "
            "Context slices should share the same perpendicular crop as their base."
        )


def test_process_ortho_slice_creates_output_directory(tmp_path: Path) -> None:
    """Regression test: _process_ortho_slice must create out_dir if it doesn't exist."""
    import numpy as np

    from fishtools.segment.extract_core import _process_ortho_slice

    # Output directory that doesn't exist yet
    out_dir = tmp_path / "nested" / "output" / "dir"
    assert not out_dir.exists()

    # Minimal fake volume (Z, Y, X, C)
    fake_vol = np.zeros((4, 100, 100, 2), dtype=np.uint16)

    _process_ortho_slice(
        vol=fake_vol,
        mask_vol=None,
        other_vol=None,
        position=50,
        axis="y",
        perpendicular_slice=slice(10, 90),
        selected_indices=[0, 1],
        out_names=["ch1", "ch2"],
        anisotropy=2,
        upscale=1.0,
        out_dir=out_dir,
        file_stem="test_vol",
        roi="test_roi",
        channels=None,
    )

    # Directory should now exist and contain the output file
    assert out_dir.exists()
    output_files = list(out_dir.glob("*.tif"))
    assert len(output_files) == 1
    assert "orthozx" in output_files[0].name


def test_mask_slice_matches_image_dimensions_in_z_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Z-mode extraction must fail fast when mask and volume spatial shapes differ."""
    import numpy as np

    import fishtools.segment.extract_core as extract_core_mod

    # Avoid GPU requirement in unsharp_all during tests.
    monkeypatch.setattr(extract_core_mod, "unsharp_all", lambda img, **_: np.asarray(img))

    from fishtools.segment.extract_core import _extract_z_slices

    reg_path = tmp_path / "reg-00.tif"
    mask_path = tmp_path / "reg-00_masks.tif"

    # Use mismatched spatial shapes (mask lower resolution) to mirror real data.
    z, c, y, x = 4, 2, 64, 64
    vol = np.ones((z, c, y, x), dtype=np.uint16)
    mask_vol = np.ones((z, y // 2, x // 2), dtype=np.uint16)

    tifffile.imwrite(reg_path, vol)
    tifffile.imwrite(mask_path, mask_vol)

    out_dir = tmp_path / "out_z"
    out_dir.mkdir()

    with pytest.raises(ValueError, match="spatial dimensions"):
        _extract_z_slices(
            file=reg_path,
            roi="roi_a",
            out_dir=out_dir,
            channels=None,
            dz=1,
            n_crops=1,
            upscale=1.5,
            max_from_path=None,
            mask_path=mask_path,
            enrich_boundaries=None,
            seed=0,
            progress=None,
        )


def test_mask_slice_matches_image_dimensions_in_ortho_mode_tiff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ortho-mode TIFF extraction must fail fast when mask and volume spatial shapes differ."""
    import numpy as np

    import fishtools.segment.extract_core as extract_core_mod

    # Avoid GPU requirement in unsharp_all during tests.
    monkeypatch.setattr(extract_core_mod, "unsharp_all", lambda img, **_: np.asarray(img))

    from fishtools.segment.extract_core import _extract_ortho_slices

    reg_path = tmp_path / "reg-ortho.tif"
    mask_path = tmp_path / "reg-ortho_masks.tif"

    # Volume: (Z, C, Y, X) and mask: (Z, Y, X) with lower-res mask.
    z, c, y, x = 4, 2, 64, 64
    vol = np.ones((z, c, y, x), dtype=np.uint16)
    mask_vol = np.ones((z, y // 2, x // 2), dtype=np.uint16)

    tifffile.imwrite(reg_path, vol)
    tifffile.imwrite(mask_path, mask_vol)

    out_dir = tmp_path / "out_ortho"
    out_dir.mkdir()

    with pytest.raises(ValueError, match="spatial dimensions"):
        _extract_ortho_slices(
            file=reg_path,
            roi="roi_b",
            out_dir=out_dir,
            channels=None,
            crop=2,
            n=2,
            anisotropy=4,
            upscale=1.5,
            max_from_path=None,
            mask_path=mask_path,
            enrich_boundaries=None,
            seed=0,
            progress=None,
        )


def test_mask_tile_matches_image_dimensions_in_zarr_z_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mask tiles from fused Zarr in z mode must match image tile dimensions."""
    import numpy as np

    import fishtools.segment.extract_core as extract_core_mod

    # Avoid GPU requirement in unsharp_all during tests (defensive).
    monkeypatch.setattr(extract_core_mod, "unsharp_all", lambda img, **_: np.asarray(img))

    from fishtools.segment.extract_core import _extract_tiles_from_zarr
    from fishtools.segment.extract_helpers import TileJob

    # Synthetic fused volume (Z,Y,X,C) and mask (Z,Y,X).
    z, y, x, c = 3, 40, 44, 2
    vol = np.ones((z, y, x, c), dtype=np.uint16)
    mask_vol = np.ones((z, y, x), dtype=np.uint16)

    out_dir = tmp_path / "out_zarr_z"
    out_dir.mkdir()

    job = TileJob(
        file=tmp_path / "reg.zarr",
        vol=vol,
        channel_names=["ch0", "ch1"],
        mask_vol=mask_vol,
        mask_path=None,
        tile_origins=[(0, 0)],
        z_candidates=[0],
    )

    _extract_tiles_from_zarr(
        job=job,
        roi="roi_zarr",
        out_dir=out_dir,
        channels=None,
        dz=1,
        upscale=1.5,
        max_from_path=None,
        progress=None,
    )

    from tifffile import imread

    for img_path in sorted(out_dir.glob("*.tif")):
        if img_path.name.endswith("_masks.tif"):
            continue
        mask_file = out_dir / (img_path.stem + "_masks.tif")
        assert mask_file.exists(), f"Missing mask for {img_path.name}"
        img = imread(img_path)
        mask = imread(mask_file)
        # CYX vs YX – compare spatial dimensions only.
        assert img.shape[1:] == mask.shape, (
            f"Mask {mask_file.name} shape {mask.shape} != image {img.shape[1:]}"
        )


def test_mask_slice_matches_image_dimensions_in_zarr_ortho_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mask slices from fused Zarr in ortho mode must match image slice dimensions."""
    import numpy as np

    import fishtools.segment.extract_core as extract_core_mod

    # Avoid GPU requirement in unsharp_all during tests (defensive).
    monkeypatch.setattr(extract_core_mod, "unsharp_all", lambda img, **_: np.asarray(img))

    from fishtools.segment.extract_core import _execute_zarr_ortho_extraction
    from fishtools.segment.extract_helpers import ExtractionConfig

    # Synthetic fused volume (Z,Y,X,C) and mask (Z,Y,X).
    z, y, x, c = 4, 65, 67, 2
    fake_vol = np.ones((z, y, x, c), dtype=np.uint16)
    fake_mask = np.ones((z, y, x), dtype=np.uint16)
    fake_names = ["ch0", "ch1"]

    # Monkeypatch IO helpers to avoid real Zarr/TIFF reads.
    def _fake_open_volume(path: Path) -> tuple[np.ndarray, list[str]]:
        return fake_vol, fake_names

    monkeypatch.setattr("fishtools.segment.extract_core._open_volume", _fake_open_volume)
    monkeypatch.setattr(
        "fishtools.segment.extract_core.resolve_file_mask_path",
        lambda f, explicit_mask: Path("dummy_masks.zarr"),
    )
    monkeypatch.setattr(
        "fishtools.segment.extract_core.open_and_validate_mask",
        lambda mask_path, vol, *, label: fake_mask,
    )
    monkeypatch.setattr("fishtools.segment.extract_core._resolve_other_volume", lambda f, p: None)

    reg_file = tmp_path / "reg-ortho.zarr"
    reg_file.mkdir()

    out_dir = tmp_path / "out_zarr_ortho"
    out_dir.mkdir()

    config = ExtractionConfig(
        mode="ortho",
        channels=None,
        crop=2,
        dz=1,
        n=2,
        anisotropy=4,
        upscale=1.5,
        seed=0,
        threads=1,
    )

    _execute_zarr_ortho_extraction(
        label="roi_zarr",
        files=[reg_file],
        config=config,
        out_dir=out_dir,
        max_from_path=None,
        explicit_mask_path=None,
        enrich_boundaries=None,
    )

    from tifffile import imread

    for img_path in sorted(out_dir.glob("*.tif")):
        if img_path.name.endswith("_masks.tif"):
            continue
        mask_file = out_dir / (img_path.stem + "_masks.tif")
        assert mask_file.exists(), f"Missing mask for {img_path.name}"
        img = imread(img_path)
        mask = imread(mask_file)
        # Image is CZX or CZY; mask is ZX or ZY – compare spatial dims.
        assert img.shape[1:] == mask.shape, (
            f"Mask {mask_file.name} shape {mask.shape} != image {img.shape[1:]}"
        )
