from pathlib import Path

import numpy as np
import polars as pl
import pytest
from click.testing import CliRunner

from test._cellpose_stub import ensure_cellpose_stub


def _make_polygons():
    from shapely.geometry import Polygon

    return [
        (Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]), {"label": 101}),
        (Polygon([(2, 2), (3, 2), (3, 3), (2, 3)]), {"label": 202}),
    ]


def test_segment_overlay_spots_help():
    ensure_cellpose_stub()
    from fishtools.segment import app

    runner = CliRunner()
    result = runner.invoke(app, ["overlay", "spots", "--help"])
    assert result.exit_code == 0
    # Check that core options are present
    assert "--codebook" in result.output
    assert "--seg-codebook" in result.output
    assert "--segmentation-name" in result.output


def test_segment_overlay_executable_direct():
    # Ensure the consolidated command function is importable
    from fishtools.segment.overlay_spots import overlay as segment_overlay
    runner = CliRunner()
    result = runner.invoke(segment_overlay, ["--help"])
    assert result.exit_code == 0
    assert "--codebook" in result.output


def test_segment_overlay_intensity_help():
    ensure_cellpose_stub()
    from fishtools.segment import app

    runner = CliRunner()
    result = runner.invoke(app, ["overlay", "intensity", "--help"])
    assert result.exit_code == 0
    assert "--segmentation-name" in result.output
    assert "--seg-codebook" in result.output
    assert "--intensity-codebook" in result.output
    assert "--intensity-store" in result.output
    assert "--threads" in result.output


def test_segment_overlay_intensity_direct():
    ensure_cellpose_stub()
    from fishtools.segment.overlay_intensity import overlay_intensity

    runner = CliRunner()
    result = runner.invoke(overlay_intensity, ["--help"])
    assert result.exit_code == 0
    assert "--seg-codebook" in result.output
    assert "--intensity-codebook" in result.output
    assert "--intensity-store" in result.output
    assert "--threads" in result.output


@pytest.fixture
def sync_executor(monkeypatch):
    from concurrent.futures import Future
    import importlib

    ensure_cellpose_stub()
    overlay_mod = importlib.import_module("fishtools.segment.overlay_spots")

    class _ImmediateExecutor:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, *args, **kwargs):
            future = Future()
            try:
                result = fn(*args, **kwargs)
            except Exception as exc:  # pragma: no cover - surface exact exception upstream
                future.set_exception(exc)
            else:
                future.set_result(result)
            return future

    monkeypatch.setattr(overlay_mod, "ProcessPoolExecutor", _ImmediateExecutor)
    return None


def _sanitize_codebook(codebook: str) -> str:
    return codebook.replace("-", "_").replace(" ", "_")


def _make_workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "workspace"
    (ws / "analysis/deconv").mkdir(parents=True, exist_ok=True)
    (ws / "analysis/output").mkdir(parents=True, exist_ok=True)
    (ws / "workspace.DONE").touch()
    return ws


def _write_tileconfig(ws: Path, roi: str, coords: list[tuple[float, float]]) -> Path:
    tc_dir = ws / "analysis/deconv" / f"stitch--{roi}"
    tc_dir.mkdir(parents=True, exist_ok=True)
    lines = ["dim=2"]
    for idx, (x, y) in enumerate(coords):
        lines.append(f"{idx:04d}.tif; ; ({float(x)}, {float(y)})")
    tc_path = tc_dir / "TileConfiguration.registered.txt"
    tc_path.write_text("\n".join(lines) + "\n")
    return tc_path


def _write_segmentation(ws: Path, roi: str, seg_cb: str, data: np.ndarray) -> Path:
    import zarr

    seg_dir = ws / "analysis/deconv" / f"stitch--{roi}+{seg_cb}"
    seg_dir.mkdir(parents=True, exist_ok=True)
    arr = zarr.open_array(
        str(seg_dir / "output_segmentation.zarr"), mode="w", shape=data.shape, dtype=data.dtype, chunks=data.shape
    )
    arr[:] = data
    return seg_dir


def _write_spots(ws: Path, roi: str, codebook: str, rows: list[dict[str, float | str]]) -> Path:
    out_dir = ws / "analysis/output"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{roi}+{_sanitize_codebook(codebook)}.parquet"
    pl.DataFrame(rows).write_parquet(path)
    return path


def _overlay_chunks_dir(seg_dir: Path, codebook: str) -> Path:
    return seg_dir / f"chunks+{_sanitize_codebook(codebook)}"


def _invoke_overlay_spots(
    ws: Path,
    roi: str,
    codebook: str,
    seg_cb: str,
    spots_path: Path,
    extra_args: list[str] | None = None,
):
    ensure_cellpose_stub()
    from fishtools.segment import app

    runner = CliRunner()
    args = [
        "overlay",
        "spots",
        str(ws),
        roi,
        "--codebook",
        codebook,
        "--seg-codebook",
        seg_cb,
        "--segmentation-name",
        "output_segmentation.zarr",
        "--spots",
        str(spots_path),
        "--overwrite",
    ]
    if extra_args:
        args.extend(extra_args)
    return runner.invoke(app, args)


def _two_block_plane(label_a: int = 1, label_b: int = 2, size: int = 5) -> np.ndarray:
    block_a = np.full((size, size), label_a, dtype=np.int32)
    block_b = np.full((size, size), label_b, dtype=np.int32)
    zeros = np.zeros((size, size), dtype=np.int32)
    top = np.hstack([block_a, zeros])
    bottom = np.hstack([zeros, block_b])
    return np.vstack([top, bottom])


def _single_block_plane(label: int = 1, size: int = 5) -> np.ndarray:
    block = np.full((size, size), label, dtype=np.int32)
    zeros = np.zeros((size, size), dtype=np.int32)
    top = np.hstack([block, zeros])
    bottom = np.hstack([zeros, zeros])
    return np.vstack([top, bottom])


def test_assign_spots_handles_shapely_query_layout():
    ensure_cellpose_stub()
    from fishtools.segment.overlay_spots import assign_spots_to_polygons, build_spatial_index

    polygons_with_meta = _make_polygons()
    tree, tree_indices = build_spatial_index(polygons_with_meta, idx=0)

    spots_df = pl.DataFrame(
        {
            "spot_id": [1, 2],
            "x_adj": [0.5, 2.5],
            "y_adj": [0.5, 2.5],
            "z": [0.0, 0.0],
            "target": ["a", "b"],
        }
    )

    assignments = assign_spots_to_polygons(spots_df, tree, tree_indices, polygons_with_meta, idx=0)

    assert assignments.sort("spot_id").to_dicts() == [
        {"spot_id": 1, "target": "a", "label": 101},
        {"spot_id": 2, "target": "b", "label": 202},
    ]


@pytest.mark.usefixtures("sync_executor")
def test_overlay_spots_cli_generates_expected_outputs(tmp_path):
    roi = "roi"
    codebook = "cb1"
    seg_cb = "seg"
    ws = _make_workspace(tmp_path)
    seg_dir = _write_segmentation(ws, roi, seg_cb, np.array([_two_block_plane()], dtype=np.int32))
    _write_tileconfig(ws, roi, [(0.0, 0.0), (2.0, 0.0)])
    spots_path = _write_spots(
        ws,
        roi,
        codebook,
        [
            {"x": 5.0, "y": 5.0, "z": 0.0, "target": "geneA"},
            {"x": 15.0, "y": 15.0, "z": 0.0, "target": "geneB"},
        ],
    )

    result = _invoke_overlay_spots(ws, roi, codebook, seg_cb, spots_path)

    assert result.exit_code == 0, result.output
    chunk_dir = _overlay_chunks_dir(seg_dir, codebook)
    ident = pl.read_parquet(chunk_dir / "ident_0.parquet").sort("spot_id")
    assert ident.to_dicts() == [
        {"spot_id": 0, "target": "geneA", "label": 1},
        {"spot_id": 1, "target": "geneB", "label": 2},
    ]
    polygons = pl.read_parquet(chunk_dir / "polygons_0.parquet")
    assert sorted(polygons.get_column("label").to_list()) == [1, 2]


@pytest.mark.usefixtures("sync_executor")
def test_overlay_spots_cli_filters_spots_by_z(tmp_path):
    roi = "roi"
    codebook = "cb1"
    seg_cb = "seg"
    ws = _make_workspace(tmp_path)
    first_slice = _single_block_plane()
    second_slice = np.zeros_like(first_slice)
    seg_dir = _write_segmentation(ws, roi, seg_cb, np.stack([first_slice, second_slice], axis=0))
    _write_tileconfig(ws, roi, [(0.0, 0.0)])
    spots_path = _write_spots(
        ws,
        roi,
        codebook,
        [
            {"x": 5.0, "y": 5.0, "z": 0.0, "target": "geneA"},
            {"x": 5.0, "y": 5.0, "z": 10.0, "target": "geneFar"},
        ],
    )

    result = _invoke_overlay_spots(ws, roi, codebook, seg_cb, spots_path)

    assert result.exit_code == 0, result.output
    chunk_dir = _overlay_chunks_dir(seg_dir, codebook)
    ident0 = pl.read_parquet(chunk_dir / "ident_0.parquet")
    assert ident0.get_column("target").to_list() == ["geneA"]
    ident1 = pl.read_parquet(chunk_dir / "ident_1.parquet")
    assert ident1.height == 0


@pytest.mark.usefixtures("sync_executor")
def test_overlay_spots_cli_applies_tile_offsets(tmp_path):
    roi = "roi"
    codebook = "cb1"
    seg_cb = "seg"
    ws = _make_workspace(tmp_path)
    seg_dir = _write_segmentation(ws, roi, seg_cb, np.array([_single_block_plane()], dtype=np.int32))
    _write_tileconfig(ws, roi, [(100.0, 200.0), (110.0, 210.0)])
    offset_x = 100.0
    offset_y = 200.0
    desired = 2.5
    spots_path = _write_spots(
        ws,
        roi,
        codebook,
        [
            {"x": 2 * desired + offset_x, "y": 2 * desired + offset_y, "z": 0.0, "target": "geneA"}
        ],
    )

    result = _invoke_overlay_spots(ws, roi, codebook, seg_cb, spots_path)

    assert result.exit_code == 0, result.output
    ident = pl.read_parquet(_overlay_chunks_dir(seg_dir, codebook) / "ident_0.parquet")
    assert ident.height == 1
    assert ident.get_column("label").to_list() == [1]


@pytest.mark.usefixtures("sync_executor")
def test_overlay_spots_cli_handles_empty_segmentation(tmp_path):
    roi = "roi"
    codebook = "cb1"
    seg_cb = "seg"
    ws = _make_workspace(tmp_path)
    seg_dir = _write_segmentation(
        ws,
        roi,
        seg_cb,
        np.zeros((1, 2, 2), dtype=np.int32),
    )
    _write_tileconfig(ws, roi, [(0.0, 0.0)])
    spots_path = _write_spots(
        ws,
        roi,
        codebook,
        [{"x": 1.0, "y": 1.0, "z": 0.0, "target": "geneA"}],
    )

    result = _invoke_overlay_spots(ws, roi, codebook, seg_cb, spots_path)

    assert result.exit_code == 0, result.output
    chunk_dir = _overlay_chunks_dir(seg_dir, codebook)
    assert pl.read_parquet(chunk_dir / "ident_0.parquet").height == 0
    assert pl.read_parquet(chunk_dir / "polygons_0.parquet").height == 0


@pytest.mark.usefixtures("sync_executor")
def test_overlay_spots_cli_errors_on_missing_segmentation(tmp_path):
    roi = "roi"
    codebook = "cb1"
    seg_cb = "seg"
    ws = _make_workspace(tmp_path)
    _write_tileconfig(ws, roi, [(0.0, 0.0)])
    spots_path = _write_spots(
        ws,
        roi,
        codebook,
        [{"x": 1.0, "y": 1.0, "z": 0.0, "target": "geneA"}],
    )

    result = _invoke_overlay_spots(ws, roi, codebook, seg_cb, spots_path)

    assert result.exit_code != 0
    assert result.exception is not None
    assert "segmentation not found" in str(result.exception).lower()
