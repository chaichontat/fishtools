from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import zarr
from click.testing import CliRunner

from fishtools.segment import app as segment_app
from fishtools.segmentation.distributed import distributed_postproc as dpp


pytestmark = pytest.mark.timeout(30)


def _make_segmentation_zarr(stitch_dir: Path) -> Path:
    stitch_dir.mkdir(parents=True, exist_ok=True)
    seg_path = stitch_dir / "output_segmentation-sam.zarr"
    arr = zarr.open_array(seg_path, mode="w", shape=(1, 4, 4), chunks=(1, 4, 4), dtype=np.uint32)
    arr[...] = 1
    return seg_path


def test_segment_postproc_batch_all_rois(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ws = tmp_path / "ws"
    ws.mkdir(parents=True, exist_ok=True)
    (ws / "workspace.DONE").touch()
    _make_segmentation_zarr(ws / "analysis/deconv/stitch--roi1+cb1")
    (ws / "analysis/deconv/stitch--roi2+cb2").mkdir(parents=True, exist_ok=True)  # missing segmentation zarr

    called: list[Path] = []

    def _fake_postproc(*, input_path: Path, **_: object) -> None:
        called.append(input_path)

    monkeypatch.setattr(dpp, "distributed_postproc", _fake_postproc)

    runner = CliRunner()
    res = runner.invoke(segment_app, ["postproc-batch", str(ws)], prog_name="segment")
    assert res.exit_code == 0, res.output
    assert called == [ws / "analysis/deconv/stitch--roi1+cb1/output_segmentation-sam.zarr"]


def test_segment_postproc_batch_single_roi_runs_all_codebooks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ws = tmp_path / "ws"
    ws.mkdir(parents=True, exist_ok=True)
    (ws / "workspace.DONE").touch()
    _make_segmentation_zarr(ws / "analysis/deconv/stitch--roi1+cb1")
    _make_segmentation_zarr(ws / "analysis/deconv/stitch--roi1+cb2")

    called: list[Path] = []

    def _fake_postproc(*, input_path: Path, **_: object) -> None:
        called.append(input_path)

    monkeypatch.setattr(dpp, "distributed_postproc", _fake_postproc)

    runner = CliRunner()
    res = runner.invoke(segment_app, ["postproc-batch", str(ws), "roi1"], prog_name="segment")
    assert res.exit_code == 0, res.output
    assert sorted(called) == sorted(
        [
            ws / "analysis/deconv/stitch--roi1+cb1/output_segmentation-sam.zarr",
            ws / "analysis/deconv/stitch--roi1+cb2/output_segmentation-sam.zarr",
        ]
    )


def test_segment_postproc_batch_output_path_requires_single_input(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir(parents=True, exist_ok=True)
    (ws / "workspace.DONE").touch()
    _make_segmentation_zarr(ws / "analysis/deconv/stitch--roi1+cb1")
    _make_segmentation_zarr(ws / "analysis/deconv/stitch--roi2+cb2")

    runner = CliRunner()
    res = runner.invoke(
        segment_app,
        ["postproc-batch", str(ws), "--output-path", str(ws / "out.zarr")],
        prog_name="segment",
    )
    assert res.exit_code != 0
    assert res.exception is not None
    assert "--output-path can only be used when processing a single input zarr" in str(res.exception)


def test_segment_postproc_batch_help() -> None:
    runner = CliRunner()
    res = runner.invoke(segment_app, ["postproc-batch", "--help"], prog_name="segment")
    assert res.exit_code == 0, res.output
