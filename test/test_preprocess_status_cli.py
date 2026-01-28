from __future__ import annotations

import json
import os
import time
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


def test_preprocess_status_ccf_tracks_imagej_pipeline(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "workspace.DONE").touch()

    roi = "roi1"

    h5ad_dir = workspace / "analysis" / "output" / "h5ads"
    h5ad_dir.mkdir(parents=True)
    (h5ad_dir / f"{roi}.h5ad").touch()

    ccf_root = workspace / "analysis" / "output" / "ccf-transforms" / roi
    ccf_root.mkdir(parents=True)
    (ccf_root / "p1_landmarks.json").write_text("{}", encoding="utf-8")
    (ccf_root / "p1_threshold.json").write_text('{"threshold": 1.0}', encoding="utf-8")

    run_dir = ccf_root / "landmark_syn_mi"
    run_dir.mkdir(parents=True)
    (run_dir / "similarity_plus_syn_summary.json").write_text("{}", encoding="utf-8")

    mask_edit = run_dir / "mask_edit"
    mask_edit.mkdir(parents=True)
    (mask_edit / "warped_moving_z0_ds8_target2um.png").touch()
    (mask_edit / "mask_z0_ds8_target2um.tif").touch()
    (mask_edit / "RoiSet.zip").touch()

    (ccf_root / f"{roi}.syn.h5ad").touch()
    (ccf_root / f"{roi}.syn.annotated.h5ad").touch()

    runner = CliRunner()
    result = runner.invoke(status_cmd, [str(workspace), "--ccf", "--json"])
    assert result.exit_code == 0, result.output

    payload = json.loads(result.output)
    assert payload["mode"] == "ccf"
    roi_payload = payload["rois"][roi]

    assert roi_payload["h5ad"]["complete"] is True
    assert roi_payload["p1"]["complete"] is True
    assert roi_payload["ants"]["complete"] is True
    assert roi_payload["mask_edit"]["complete"] is True
    assert roi_payload["imagej_roi"]["complete"] is True
    assert roi_payload["warp_h5ad"]["complete"] is True
    assert roi_payload["filter_h5ad"]["complete"] is True


def test_preprocess_status_ccf_imagej_stale_only_vs_tileconfig(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "workspace.DONE").touch()

    roi = "roi1"

    h5ad_dir = workspace / "analysis" / "output" / "h5ads"
    h5ad_dir.mkdir(parents=True)
    (h5ad_dir / f"{roi}.h5ad").touch()

    ccf_root = workspace / "analysis" / "output" / "ccf-transforms" / roi
    ccf_root.mkdir(parents=True)
    (ccf_root / "p1_landmarks.json").write_text("{}", encoding="utf-8")
    (ccf_root / "p1_threshold.json").write_text('{"threshold": 1.0}', encoding="utf-8")

    run_dir = ccf_root / "landmark_syn_mi"
    run_dir.mkdir(parents=True)
    (run_dir / "similarity_plus_syn_summary.json").write_text("{}", encoding="utf-8")

    mask_edit = run_dir / "mask_edit"
    mask_edit.mkdir(parents=True)
    (mask_edit / "warped_moving_z0_ds8_target2um.png").touch()
    (mask_edit / "mask_z0_ds8_target2um.tif").touch()
    roi_zip = mask_edit / "RoiSet.zip"
    roi_zip.touch()

    tileconfig = (
        workspace / "analysis" / "deconv" / f"stitch--{roi}" / "TileConfiguration.registered.txt"
    )
    tileconfig.parent.mkdir(parents=True, exist_ok=True)
    tileconfig.write_text("dummy\n")

    now = time.time()
    os.utime(roi_zip, (now - 100.0, now - 100.0))
    os.utime(tileconfig, (now, now))

    runner = CliRunner()
    result = runner.invoke(status_cmd, [str(workspace), "--ccf", "--json"])
    assert result.exit_code == 0, result.output

    payload = json.loads(result.output)
    roi_payload = payload["rois"][roi]
    assert roi_payload["imagej_roi"]["stale"] is True


def test_preprocess_status_ccf_filter_requires_imagej_roi(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "workspace.DONE").touch()

    roi = "roi1"

    h5ad_dir = workspace / "analysis" / "output" / "h5ads"
    h5ad_dir.mkdir(parents=True)
    (h5ad_dir / f"{roi}.h5ad").touch()

    ccf_root = workspace / "analysis" / "output" / "ccf-transforms" / roi
    ccf_root.mkdir(parents=True)

    # Filter output exists, but ImageJ ROI does not.
    (ccf_root / f"{roi}.syn.annotated.h5ad").touch()

    runner = CliRunner()
    result = runner.invoke(status_cmd, [str(workspace), "--ccf", "--json"])
    assert result.exit_code == 0, result.output

    payload = json.loads(result.output)
    roi_payload = payload["rois"][roi]
    assert roi_payload["filter_h5ad"]["complete"] is False
    assert roi_payload["filter_h5ad"]["count"] == 0
