from pathlib import Path

import numpy as np
import polars as pl
import pytest
from click.testing import CliRunner

from _cellpose_stub import ensure_cellpose_stub
from fishtools.utils.zarr_utils import default_zarr_codecs


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


def test_segment_overlay_all_help():
    ensure_cellpose_stub()
    from fishtools.segment import app

    runner = CliRunner()
    result = runner.invoke(app, ["overlay", "all", "--help"])
    assert result.exit_code == 0
    assert "--codebook" in result.output
    assert "--seg-codebook" in result.output
    assert "--intensity-codebook" in result.output
    assert "--segmentation-name" in result.output
    assert "--fused-name" in result.output
    assert "--erode" in result.output
    assert "--threads" in result.output
    assert "--export" in result.output


def test_segment_overlay_all_uses_callbacks(tmp_path, monkeypatch):
    ensure_cellpose_stub()
    import importlib

    overlay_all_mod = importlib.import_module("fishtools.segment.overlay_all")
    overlay_intensity_mod = importlib.import_module("fishtools.segment.overlay_intensity")
    overlay_spots_mod = importlib.import_module("fishtools.segment.overlay_spots")

    ws = _make_workspace(tmp_path)
    (ws / "analysis/deconv/stitch--roi+cb1" / "output_segmentation-sam.zarr").mkdir(
        parents=True, exist_ok=True
    )
    (ws / "analysis/deconv/stitch--roi+cb_int" / "fused.zarr").mkdir(parents=True, exist_ok=True)

    calls: dict[str, dict[str, object]] = {}

    def fake_spots(
        *,
        path: Path,
        roi: str,
        codebook: str,
        spots_opt: Path | None,
        seg_codebook: str | None,
        segmentation_name: str,
        overwrite: bool,
        debug: bool,
    ) -> None:
        calls["spots"] = {
            "path": path,
            "roi": roi,
            "codebook": codebook,
            "spots_opt": spots_opt,
            "seg_codebook": seg_codebook,
            "segmentation_name": segmentation_name,
            "overwrite": overwrite,
            "debug": debug,
        }

    def fake_intensity(
        *,
        path: Path,
        roi: str,
        seg_codebook: str,
        intensity_codebook: str,
        segmentation_name: str,
        fused_name: str,
        erode: int,
        channel: str | None,
        threads: int,
        overwrite: bool,
    ) -> None:
        calls["intensity"] = {
            "path": path,
            "roi": roi,
            "seg_codebook": seg_codebook,
            "intensity_codebook": intensity_codebook,
            "segmentation_name": segmentation_name,
            "fused_name": fused_name,
            "erode": erode,
            "channel": channel,
            "threads": threads,
            "overwrite": overwrite,
        }

    def fail_main(*args, **kwargs):
        raise AssertionError("Command.main should not be called from overlay all.")

    monkeypatch.setattr(overlay_spots_mod.overlay, "main", fail_main)
    monkeypatch.setattr(overlay_intensity_mod.overlay_intensity, "main", fail_main)
    monkeypatch.setattr(overlay_spots_mod.overlay, "callback", fake_spots)
    monkeypatch.setattr(overlay_intensity_mod.overlay_intensity, "callback", fake_intensity)

    overlay_all_mod.overlay_all.callback(
        path=ws,
        roi=None,
        codebook="cb1",
        seg_codebook=None,
        intensity_codebook="cb_int",
        spots_opt=None,
        segmentation_name="output_segmentation-sam.zarr",
        fused_name="fused.zarr",
        channel=None,
        threads=2,
        erode=2,
        export_opt=False,
        overwrite=False,
        debug=False,
    )

    assert calls["spots"]["roi"] == "roi"
    assert calls["spots"]["seg_codebook"] is None
    assert calls["intensity"]["roi"] == "roi"
    assert calls["intensity"]["seg_codebook"] == "cb1"


def test_segment_overlay_all_export_runs_segment_export_direct(tmp_path, monkeypatch):
    ensure_cellpose_stub()
    import importlib

    overlay_all_mod = importlib.import_module("fishtools.segment.overlay_all")
    overlay_intensity_mod = importlib.import_module("fishtools.segment.overlay_intensity")
    overlay_spots_mod = importlib.import_module("fishtools.segment.overlay_spots")
    export_mod = importlib.import_module("fishtools.segment.export")

    ws = _make_workspace(tmp_path)
    (ws / "analysis/deconv/stitch--roi+cb1" / "output_segmentation-sam.zarr").mkdir(
        parents=True, exist_ok=True
    )
    (ws / "analysis/deconv/stitch--roi+cb_int" / "fused.zarr").mkdir(parents=True, exist_ok=True)

    def fake_spots(**_kwargs) -> None:
        return None

    def fake_intensity(**_kwargs) -> None:
        return None

    export_calls: dict[str, object] = {}

    def fake_export_cmd(
        *,
        path: Path,
        roi: str | None,
        seg_codebook: str,
        codebooks: tuple[str, ...],
        segmentation_name: str,
        channels: str,
        thumbnail_scale: float = 8.0,
        diag: bool = False,
    ) -> None:
        export_calls.update(
            {
                "path": path,
                "roi": roi,
                "seg_codebook": seg_codebook,
                "codebooks": codebooks,
                "segmentation_name": segmentation_name,
                "channels": channels,
                "thumbnail_scale": thumbnail_scale,
                "diag": diag,
            }
        )

    monkeypatch.setattr(overlay_spots_mod.overlay, "callback", fake_spots)
    monkeypatch.setattr(overlay_intensity_mod.overlay_intensity, "callback", fake_intensity)
    monkeypatch.setattr(export_mod, "export_cmd", fake_export_cmd)

    runner = CliRunner()
    result = runner.invoke(
        overlay_all_mod.overlay_all,
        [
            str(ws),
            "--codebook",
            "cb1",
            "--seg-codebook",
            "cb1",
            "--intensity-codebook",
            "cb_int",
            "--export",
        ],
    )
    assert result.exit_code == 0, result.output
    assert export_calls == {
        "path": ws,
        "roi": None,
        "seg_codebook": "cb1",
        "codebooks": ("cb1",),
        "segmentation_name": "output_segmentation-sam.zarr",
        "channels": "auto",
        "thumbnail_scale": 8.0,
        "diag": False,
    }


def test_segment_overlay_all_skips_rois_missing_segmentation_zarr(tmp_path, monkeypatch):
    ensure_cellpose_stub()
    import importlib

    overlay_all_mod = importlib.import_module("fishtools.segment.overlay_all")
    overlay_intensity_mod = importlib.import_module("fishtools.segment.overlay_intensity")
    overlay_spots_mod = importlib.import_module("fishtools.segment.overlay_spots")

    ws = _make_workspace(tmp_path)
    (ws / "analysis/deconv/stitch--2+pi").mkdir(parents=True, exist_ok=True)
    (ws / "analysis/deconv/stitch--roi1+pi" / "output_segmentation.zarr").mkdir(
        parents=True, exist_ok=True
    )
    (ws / "analysis/deconv/stitch--roi1+edu" / "fused.zarr").mkdir(parents=True, exist_ok=True)

    spots_called: list[str] = []
    intensity_called: list[str] = []

    def fake_spots(*, roi: str, **kwargs) -> None:
        spots_called.append(roi)

    def fake_intensity(*, roi: str, **kwargs) -> None:
        intensity_called.append(roi)

    monkeypatch.setattr(overlay_spots_mod.overlay, "callback", fake_spots)
    monkeypatch.setattr(overlay_intensity_mod.overlay_intensity, "callback", fake_intensity)

    runner = CliRunner()
    result = runner.invoke(
        overlay_all_mod.overlay_all,
        [
            str(ws),
            "--codebook",
            "cs_base",
            "--seg-codebook",
            "pi",
            "--intensity-codebook",
            "edu",
            "--segmentation-name",
            "output_segmentation.zarr",
            "--fused-name",
            "fused.zarr",
        ],
    )

    assert result.exit_code == 0, result.output
    assert spots_called == ["roi1"]
    assert intensity_called == ["roi1"]


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
    assert "--fused-name" in result.output
    assert "--erode" in result.output
    assert "--threads" in result.output


def test_segment_overlay_intensity_direct():
    ensure_cellpose_stub()
    from fishtools.segment.overlay_intensity import overlay_intensity

    runner = CliRunner()
    result = runner.invoke(overlay_intensity, ["--help"])
    assert result.exit_code == 0
    assert "--seg-codebook" in result.output
    assert "--intensity-codebook" in result.output
    assert "--fused-name" in result.output
    assert "--erode" in result.output
    assert "--threads" in result.output


def test_segment_overlay_intensity_all_token_triggers_batch_mode(tmp_path, monkeypatch):
    ensure_cellpose_stub()
    import importlib

    overlay_mod = importlib.import_module("fishtools.segment.overlay_intensity")

    ws = _make_workspace(tmp_path)
    (ws / "analysis/deconv/stitch--roi1+seg").mkdir(parents=True, exist_ok=True)
    (ws / "analysis/deconv/stitch--roi2+seg").mkdir(parents=True, exist_ok=True)

    called: list[str] = []

    def fake_run(
        workspace,
        roi: str,
        seg_codebook: str,
        intensity_codebook: str,
        segmentation_name: str,
        fused_name: str,
        channel: str | None,
        threads: int,
        erode: int,
        overwrite: bool,
    ) -> None:
        called.append(roi)

    monkeypatch.setattr(overlay_mod, "_run_overlay_for_roi", fake_run)

    runner = CliRunner()
    result = runner.invoke(
        overlay_mod.overlay_intensity,
        [str(ws), "all", "--seg-codebook", "seg", "--intensity-codebook", "cb_int"],
    )

    assert result.exit_code == 0, result.output
    assert called == ["roi1", "roi2"]


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
        str(seg_dir / "output_segmentation.zarr"),
        mode="w",
        shape=data.shape,
        dtype=data.dtype,
        chunks=data.shape,
        codecs=default_zarr_codecs(data.dtype),
    )
    arr[:] = data
    return seg_dir


def _write_spots(ws: Path, roi: str, codebook: str, rows: list[dict[str, float | str]]) -> Path:
    out_dir = ws / "analysis/output"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{roi}+{_sanitize_codebook(codebook)}.parquet"
    pl.DataFrame(rows).write_parquet(path)
    return path


def _overlay_chunks_dir(seg_dir: Path, codebook: str, seg_name: str = "output_segmentation.zarr") -> Path:
    # Chunks are inside the segmentation zarr folder
    return seg_dir / seg_name / f"chunks+{_sanitize_codebook(codebook)}"


def _write_intensity_store(ws: Path, roi: str, cb: str, data: np.ndarray, *, key: list[str]) -> Path:
    import zarr

    out_dir = ws / "analysis/deconv" / f"stitch--{roi}+{cb}"
    out_dir.mkdir(parents=True, exist_ok=True)
    arr = zarr.open_array(
        str(out_dir / "fused.zarr"),
        mode="w",
        shape=data.shape,
        dtype=data.dtype,
        chunks=data.shape,
        codecs=default_zarr_codecs(data.dtype),
    )
    arr[:] = data
    arr.attrs["key"] = key
    return out_dir / "fused.zarr"


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


def _invoke_overlay_spots_no_overwrite(
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
def test_overlay_spots_cli_assigns_single_pixel_region(tmp_path):
    roi = "roi"
    codebook = "cb1"
    seg_cb = "seg"
    ws = _make_workspace(tmp_path)
    plane = np.zeros((10, 10), dtype=np.int32)
    plane[2, 2] = 1
    seg_dir = _write_segmentation(ws, roi, seg_cb, np.array([plane], dtype=np.int32))
    _write_tileconfig(ws, roi, [(0.0, 0.0)])
    spots_path = _write_spots(
        ws,
        roi,
        codebook,
        [{"x": 4.0, "y": 4.0, "z": 0.0, "target": "geneA"}],
    )

    result = _invoke_overlay_spots(ws, roi, codebook, seg_cb, spots_path)

    assert result.exit_code == 0, result.output
    chunk_dir = _overlay_chunks_dir(seg_dir, codebook)
    ident = pl.read_parquet(chunk_dir / "ident_0.parquet")
    assert ident.to_dicts() == [{"spot_id": 0, "target": "geneA", "label": 1}]


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


@pytest.mark.usefixtures("sync_executor")
def test_overlay_spots_cli_skips_when_outputs_exist_and_no_overwrite(tmp_path, monkeypatch):
    ensure_cellpose_stub()
    import importlib

    overlay_mod = importlib.import_module("fishtools.segment.overlay_spots")

    roi = "roi"
    codebook = "cb1"
    seg_cb = "seg"
    ws = _make_workspace(tmp_path)
    seg_dir = _write_segmentation(ws, roi, seg_cb, np.array([_single_block_plane()], dtype=np.int32))
    _write_tileconfig(ws, roi, [(0.0, 0.0)])
    spots_path = _write_spots(ws, roi, codebook, [{"x": 1.0, "y": 1.0, "z": 0.0, "target": "geneA"}])

    chunk_dir = _overlay_chunks_dir(seg_dir, codebook)
    chunk_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"spot_id": [0], "target": ["geneA"], "label": [1]}).write_parquet(
        chunk_dir / "ident_0.parquet"
    )
    pl.DataFrame({"label": [1], "area": [1]}).write_parquet(chunk_dir / "polygons_0.parquet")

    def fail_run(*_args, **_kwargs):
        raise AssertionError("run_ should not be called when outputs already exist and --overwrite is not set.")

    monkeypatch.setattr(overlay_mod, "run_", fail_run)

    result = _invoke_overlay_spots_no_overwrite(ws, roi, codebook, seg_cb, spots_path)
    assert result.exit_code == 0, result.output


@pytest.fixture
def sync_executor_intensity(monkeypatch):
    from concurrent.futures import Future
    import importlib

    ensure_cellpose_stub()
    overlay_mod = importlib.import_module("fishtools.segment.overlay_intensity")

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


@pytest.mark.usefixtures("sync_executor_intensity")
def test_overlay_intensity_cli_skips_when_outputs_exist_and_no_overwrite(tmp_path, monkeypatch):
    ensure_cellpose_stub()
    import importlib

    overlay_mod = importlib.import_module("fishtools.segment.overlay_intensity")
    from fishtools.segment import app

    roi = "roi"
    seg_cb = "seg"
    intensity_cb = "cb_int"
    ws = _make_workspace(tmp_path)

    seg_dir = _write_segmentation(ws, roi, seg_cb, np.array([_single_block_plane()], dtype=np.int32))
    _write_intensity_store(ws, roi, intensity_cb, np.zeros((1, 10, 10), dtype=np.uint16), key=["ch0"])

    seg_zarr = seg_dir / "output_segmentation.zarr"
    out_dir = seg_zarr / "intensity_ch0"
    out_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"label": [1], "mean_intensity": [1.0]}).write_parquet(out_dir / "intensity-00.parquet")

    def fail_process(*_args, **_kwargs):
        raise AssertionError(
            "_process_slice_shared_detection should not run when outputs already exist and --overwrite is not set."
        )

    monkeypatch.setattr(overlay_mod, "_process_slice_shared_detection", fail_process)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "overlay",
            "intensity",
            str(ws),
            roi,
            "--seg-codebook",
            seg_cb,
            "--intensity-codebook",
            intensity_cb,
            "--segmentation-name",
            "output_segmentation.zarr",
            "--fused-name",
            "fused.zarr",
            "--threads",
            "1",
        ],
    )
    assert result.exit_code == 0, result.output


@pytest.mark.usefixtures("sync_executor_intensity")
def test_overlay_intensity_erode_affects_intensity_stats(tmp_path: Path) -> None:
    ensure_cellpose_stub()
    from fishtools.segment import app

    roi = "roi"
    seg_cb = "seg"
    intensity_cb = "cb_int"
    ws = _make_workspace(tmp_path)

    seg = np.zeros((1, 10, 10), dtype=np.int32)
    seg[0, 1:9, 1:9] = 1
    seg_dir = _write_segmentation(ws, roi, seg_cb, seg)

    intensity = np.zeros((1, 10, 10), dtype=np.uint16)
    intensity[:, 1:9, 1:9] = 10
    ring = np.zeros((10, 10), dtype=bool)
    ring[1, 1:9] = True
    ring[8, 1:9] = True
    ring[1:9, 1] = True
    ring[1:9, 8] = True
    intensity[0, ring] = 1000
    _write_intensity_store(ws, roi, intensity_cb, intensity, key=["ch0"])

    runner = CliRunner()
    res0 = runner.invoke(
        app,
        [
            "overlay",
            "intensity",
            str(ws),
            roi,
            "--seg-codebook",
            seg_cb,
            "--intensity-codebook",
            intensity_cb,
            "--segmentation-name",
            "output_segmentation.zarr",
            "--fused-name",
            "fused.zarr",
            "--channel",
            "ch0",
            "--erode",
            "0",
            "--threads",
            "1",
            "--overwrite",
        ],
    )
    assert res0.exit_code == 0, res0.output

    df0 = pl.read_parquet(seg_dir / "output_segmentation.zarr" / "intensity_ch0" / "intensity-00.parquet")
    mean0 = float(df0.filter(pl.col("label") == 1)["mean_intensity"][0])
    assert "median_intensity" in df0.columns
    median0 = float(df0.filter(pl.col("label") == 1)["median_intensity"][0])
    assert "intensity_std" in df0.columns
    std0 = float(df0.filter(pl.col("label") == 1)["intensity_std"][0])

    assert median0 == pytest.approx(10.0)

    res2 = runner.invoke(
        app,
        [
            "overlay",
            "intensity",
            str(ws),
            roi,
            "--seg-codebook",
            seg_cb,
            "--intensity-codebook",
            intensity_cb,
            "--segmentation-name",
            "output_segmentation.zarr",
            "--fused-name",
            "fused.zarr",
            "--channel",
            "ch0",
            "--erode",
            "2",
            "--threads",
            "1",
            "--overwrite",
        ],
    )
    assert res2.exit_code == 0, res2.output

    df2 = pl.read_parquet(seg_dir / "output_segmentation.zarr" / "intensity_ch0" / "intensity-00.parquet")
    mean2 = float(df2.filter(pl.col("label") == 1)["mean_intensity"][0])
    assert "median_intensity" in df2.columns
    median2 = float(df2.filter(pl.col("label") == 1)["median_intensity"][0])
    assert "intensity_std" in df2.columns
    std2 = float(df2.filter(pl.col("label") == 1)["intensity_std"][0])

    assert mean2 < mean0
    assert median2 == pytest.approx(10.0)
    assert std0 > 0.0
    assert std2 == pytest.approx(0.0)
