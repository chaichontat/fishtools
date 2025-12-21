from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from fishtools.preprocess.cli_status import status as status_cmd


def test_preprocess_status_reports_overlay_intensity_for_intensity_codebooks(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "workspace.DONE").touch()

    roi = "roi1"
    round_ = "1"
    codebook = "cb1"

    raw_dir = workspace / f"{round_}--{roi}"
    raw_dir.mkdir()
    (raw_dir / f"{round_}-0001.tif").touch()

    deconv_dir = workspace / "analysis" / "deconv" / f"{round_}--{roi}"
    deconv_dir.mkdir(parents=True)
    (deconv_dir / f"{round_}-0001.tif").touch()

    registered_dir = workspace / "analysis" / "deconv" / f"registered--{roi}+{codebook}"
    registered_dir.mkdir(parents=True)
    (registered_dir / "reg-0001.tif").touch()

    tileconfig_path = (
        workspace / "analysis" / "deconv" / f"stitch--{roi}" / "TileConfiguration.registered.txt"
    )
    tileconfig_path.parent.mkdir(parents=True)
    tileconfig_path.write_text("dummy\n")

    stitch_dir = workspace / "analysis" / "deconv" / f"stitch--{roi}+{codebook}"
    stitch_dir.mkdir(parents=True)

    fused_zarr = stitch_dir / "fused.zarr"
    fused_zarr.mkdir()
    (fused_zarr / ".zgroup").write_text("{}\n")

    seg_zarr = stitch_dir / "output_segmentation.zarr"
    seg_zarr.mkdir()
    (seg_zarr / ".zgroup").write_text("{}\n")

    intensity_dir = seg_zarr / "intensity_dapi"
    intensity_dir.mkdir()
    (intensity_dir / "intensity-00.parquet").touch()

    runner = CliRunner()
    result = runner.invoke(status_cmd, [str(workspace), "--codebook", codebook, "--json"])
    assert result.exit_code == 0, result.output

    payload = json.loads(result.output)
    assert isinstance(payload, list)
    assert payload

    roi_payload = payload[0]["rois"][roi]
    assert roi_payload["overlay_intensity"]["complete"] is True
    assert roi_payload["overlay_intensity"]["count"] == 1


def test_preprocess_status_reports_overlay_spots(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "workspace.DONE").touch()

    roi = "roi1"
    round_ = "1"
    codebook = "cb1"

    raw_dir = workspace / f"{round_}--{roi}"
    raw_dir.mkdir()
    (raw_dir / f"{round_}-0001.tif").touch()

    stitch_dir = workspace / "analysis" / "deconv" / f"stitch--{roi}+{codebook}"
    stitch_dir.mkdir(parents=True)

    seg_zarr = stitch_dir / "output_segmentation.zarr"
    seg_zarr.mkdir()
    (seg_zarr / ".zgroup").write_text("{}\n")

    chunks_dir = seg_zarr / f"chunks+{codebook}"
    chunks_dir.mkdir()
    (chunks_dir / "ident_0.parquet").touch()
    (chunks_dir / "ident_1.parquet").touch()
    (chunks_dir / "polygons_0.parquet").touch()
    (chunks_dir / "polygons_1.parquet").touch()

    runner = CliRunner()
    result = runner.invoke(status_cmd, [str(workspace), "--codebook", codebook, "--json"])
    assert result.exit_code == 0, result.output

    payload = json.loads(result.output)
    roi_payload = payload[0]["rois"][roi]
    assert roi_payload["overlay_spots"]["complete"] is True
    assert roi_payload["overlay_spots"]["count"] == 2


def test_preprocess_status_reports_export(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "workspace.DONE").touch()

    roi = "roi1"
    round_ = "1"
    codebook = "cb1"

    raw_dir = workspace / f"{round_}--{roi}"
    raw_dir.mkdir()
    (raw_dir / f"{round_}-0001.tif").touch()

    stitch_dir = workspace / "analysis" / "deconv" / f"stitch--{roi}+{codebook}"
    stitch_dir.mkdir(parents=True)

    seg_zarr = stitch_dir / "output_segmentation.zarr"
    seg_zarr.mkdir()
    (seg_zarr / ".zgroup").write_text("{}\n")
    (seg_zarr / f"{codebook}.h5ad").touch()

    runner = CliRunner()
    result = runner.invoke(status_cmd, [str(workspace), "--codebook", codebook, "--json"])
    assert result.exit_code == 0, result.output

    payload = json.loads(result.output)
    roi_payload = payload[0]["rois"][roi]
    assert roi_payload["export"]["complete"] is True
    assert roi_payload["export"]["count"] == 1


def test_preprocess_status_reports_postproc(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "workspace.DONE").touch()

    roi = "roi1"
    round_ = "1"
    codebook = "cb1"

    raw_dir = workspace / f"{round_}--{roi}"
    raw_dir.mkdir()
    (raw_dir / f"{round_}-0001.tif").touch()

    stitch_dir = workspace / "analysis" / "deconv" / f"stitch--{roi}+{codebook}"
    stitch_dir.mkdir(parents=True)

    # Create postproc zarr
    postproc_zarr = stitch_dir / "output_segmentation-sam_postproc.zarr"
    postproc_zarr.mkdir()
    (postproc_zarr / ".zgroup").write_text("{}\n")

    runner = CliRunner()
    result = runner.invoke(status_cmd, [str(workspace), "--codebook", codebook, "--json"])
    assert result.exit_code == 0, result.output

    payload = json.loads(result.output)
    roi_payload = payload[0]["rois"][roi]
    assert roi_payload["postproc"]["complete"] is True
    assert roi_payload["postproc"]["count"] == 1


def test_preprocess_status_segmentation_prefers_stitch_outputs(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "workspace.DONE").touch()

    roi = "roi1"
    round_ = "1"
    codebook = "cb1"

    raw_dir = workspace / f"{round_}--{roi}"
    raw_dir.mkdir()
    (raw_dir / f"{round_}-0001.tif").touch()

    stitch_dir = workspace / "analysis" / "deconv" / f"stitch--{roi}+{codebook}"
    stitch_dir.mkdir(parents=True)

    seg_zarr = stitch_dir / "output_segmentation.zarr"
    seg_zarr.mkdir()
    (seg_zarr / ".zgroup").write_text("{}\n")

    seg_dir = workspace / "analysis" / "deconv" / f"segment--{roi}+{codebook}"
    seg_dir.mkdir(parents=True)
    seg_dir_zarr = seg_dir / "other_segmentation.zarr"
    seg_dir_zarr.mkdir()
    (seg_dir_zarr / ".zgroup").write_text("{}\n")

    runner = CliRunner()
    result = runner.invoke(status_cmd, [str(workspace), "--codebook", codebook, "--json"])
    assert result.exit_code == 0, result.output

    payload = json.loads(result.output)
    roi_payload = payload[0]["rois"][roi]
    assert roi_payload["segment"]["count"] == 1
