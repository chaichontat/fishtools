from __future__ import annotations

import json
from pathlib import Path

import matplotlib.image as mpimg
import numpy as np
import pytest
import tifffile
import zarr
from click.testing import CliRunner


def _write_minimal_workspace(*, root: Path, roi: str, codebook: str, channels: list[str] | None = None) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    # Workspace root is detected by the presence of any *.DONE file.
    (root / "TEST.DONE").write_text("ok", encoding="utf-8")

    deconv = root / "analysis" / "deconv"
    deconv.mkdir(parents=True, exist_ok=True)

    # Minimal fused.zarr expected by the pipeline: (z, y, x, c).
    keys = list(channels) if channels is not None else [codebook]
    stitch_dir = deconv / f"stitch--{roi}+{codebook}"
    zarr_root = stitch_dir / "fused.zarr"
    y = 32
    x = 32
    c = len(keys)
    arr = zarr.open(str(zarr_root), mode="w", shape=(1, y, x, c), chunks=(1, y, x, c), dtype="uint16")
    arr.attrs["key"] = keys

    img0 = (np.linspace(0, 1, y * x, dtype=np.float32).reshape(y, x) * 5000).astype(np.uint16)
    arr[0, :, :, 0] = img0
    if c >= 2:
        arr[0, :, :, 1] = img0[::-1, ::-1]
    if c >= 3:
        arr[0, :, :, 2] = img0[:, ::-1]
    for idx in range(3, c):
        arr[0, :, :, idx] = img0

    out_root = root / "analysis" / "output" / "ccf-transforms" / roi
    out_root.mkdir(parents=True, exist_ok=True)

    p1_landmarks = {
        "prior_rotation_deg": 0,
        "prior_flip_x": False,
        "fixed_points_cropped_xy": [[5, 5], [20, 5], [5, 20]],
        "moving_points_fullres_xy_in_rotated_crop": [[5, 5], [20, 5], [5, 20]],
        "atlas_crop_bbox": [0, y, 0, x],
        "sample_rotated_crop_bbox": [0, y, 0, x],
        "atlas_slice_idx": 0,
        "atlas_plane": "coronal",
        "atlas_name": "dummy",
        "atlas_voxel_um": 20.0,
        "sample_channel": codebook,
        "sample_z_idx": 0,
        "sample_voxel_xy_um": 20.0,
    }
    (out_root / "p1_landmarks.json").write_text(json.dumps(p1_landmarks, indent=2), encoding="utf-8")
    (out_root / "p1_threshold.json").write_text(json.dumps({"threshold": 1.0}, indent=2), encoding="utf-8")

    return root


class _DummyAtlas:
    def __init__(self, _name: str) -> None:
        y = 32
        x = 32
        ref = (np.linspace(0, 1, y * x, dtype=np.float32).reshape(y, x) * 5000).astype(np.uint16)
        ann = np.ones((y, x), dtype=np.uint16)
        self.reference = np.stack([ref], axis=0)
        self.annotation = np.stack([ann], axis=0)
        self.structures = {
            1: {"id": 1, "acronym": "REG1", "name": "Region 1", "structure_id_path": [1]},
        }


class _DummyEmptyMaskAtlas:
    def __init__(self, _name: str) -> None:
        y = 32
        x = 32
        ref = (np.linspace(0, 1, y * x, dtype=np.float32).reshape(y, x) * 5000).astype(np.uint16)
        ann = np.zeros((y, x), dtype=np.uint16)
        self.reference = np.stack([ref], axis=0)
        self.annotation = np.stack([ann], axis=0)
        self.structures = {
            1: {"id": 1, "acronym": "REG1", "name": "Region 1", "structure_id_path": [1]},
        }


class _ExplodingAtlas:
    def __init__(self, _name: str) -> None:
        raise AssertionError("BrainGlobeAtlas should not be constructed in --reference-only mode.")


def test_export_mask_edit_pack_reference_only_does_not_require_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ccf import cli_export_mask_edit_pack as export

    monkeypatch.setattr(export, "BrainGlobeAtlas", _ExplodingAtlas)

    roi = "2"
    codebook = "pi"
    workspace = _write_minimal_workspace(root=tmp_path / "ws", roi=roi, codebook=codebook, channels=[codebook, "c1", "c2"])

    runner = CliRunner()
    result = runner.invoke(
        export.main,
        [
            str(workspace),
            roi,
            "--run-dirname",
            "test_run",
            "--reference-only",
            "--z-idx",
            "0",
            "--moving-pre-downsample",
            "8",
            "--target-spacing-um",
            "2",
        ],
    )
    assert result.exit_code == 0, result.output

    out_dir = workspace / "analysis" / "output" / "ccf-transforms" / roi / "test_run" / "mask_edit"
    assert (out_dir / "reference_slice_z0_ds8_target2um.png").exists()
    assert not (out_dir / "warped_moving_z0_ds8_target2um.png").exists()
    assert not (out_dir / "mask_z0_ds8_target2um.tif").exists()


def test_export_mask_edit_pack_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from ccf import ants_landmark_syn_init as syn
    from ccf import cli_export_mask_edit_pack as export

    monkeypatch.setattr(syn, "BrainGlobeAtlas", _DummyAtlas)
    monkeypatch.setattr(export, "BrainGlobeAtlas", _DummyAtlas)

    roi = "2"
    codebook = "pi"
    workspace = _write_minimal_workspace(root=tmp_path / "ws", roi=roi, codebook=codebook, channels=[codebook, "c1", "c2"])

    syn.run_pipeline(
        workspace=workspace,
        roi=roi,
        stitch_codebook=codebook,
        run_dirname="test_run",
        p1_threshold=None,
        landmark_linear_transform_type="affine",
        syn_type_of_transform="SyNOnly",
        syn_metric="mattes",
        syn_metric_param=8,
        syn_reg_iterations=(5, 2, 1),
        syn_grad_step=0.15,
        syn_flow_sigma=4.0,
        syn_total_sigma=1.0,
        moving_presmooth_sigma_um=0.0,
        fixed_edge_guard_um=0.0,
        moving_edge_guard_um=0.0,
        use_affine_refine=False,
        use_n4=False,
        use_feature_images=False,
        use_mask_distance_metric=True,
        use_landmark_heatmap_metric=True,
        landmark_heatmap_sigma_um=70.0,
        landmark_heatmap_weight=0.4,
        crop_fixed_to_overlap=False,
    )

    runner = CliRunner()
    result = runner.invoke(
        export.main,
        [
            str(workspace),
            "--run-dirname",
            "test_run",
            "--z-idx",
            "0",
            "--moving-pre-downsample",
            "8",
            "--target-spacing-um",
            "2",
            "--term",
            "1",
            "--kind",
            "id",
        ],
    )
    assert result.exit_code == 0, result.output

    out_dir = workspace / "analysis" / "output" / "ccf-transforms" / roi / "test_run" / "mask_edit"
    png = out_dir / "warped_moving_z0_ds8_target2um.png"
    ref_png = out_dir / "reference_slice_z0_ds8_target2um.png"
    mask = out_dir / "mask_z0_ds8_target2um.tif"
    assert png.exists()
    assert ref_png.exists()
    assert mask.exists()

    mask_yx = tifffile.imread(mask)
    assert mask_yx.shape == (320, 320)
    assert mask_yx.dtype == np.uint8

    png_yx = mpimg.imread(png)
    assert png_yx.shape[0] == mask_yx.shape[0]
    assert png_yx.shape[1] == mask_yx.shape[1]
    assert png_yx.ndim == 3
    assert png_yx.shape[2] >= 3
    rgb = png_yx[:, :, :3]

    # Compare well inside the field-of-view (robust against boundary padding).
    inside = 239
    assert rgb[0, 0, 0] < rgb[inside, inside, 0]
    assert rgb[0, 0, 1] > rgb[inside, inside, 1]
    assert float(np.abs(rgb[inside, inside, 0] - rgb[inside, inside, 2])) > 1e-3

    ref_yx = mpimg.imread(ref_png)
    assert ref_yx.shape[0] == mask_yx.shape[0]
    assert ref_yx.shape[1] == mask_yx.shape[1]


def test_export_mask_edit_pack_skips_rois_without_a_mask(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from ccf import ants_landmark_syn_init as syn
    from ccf import cli_export_mask_edit_pack as export

    monkeypatch.setattr(syn, "BrainGlobeAtlas", _DummyAtlas)
    monkeypatch.setattr(export, "BrainGlobeAtlas", _DummyEmptyMaskAtlas)

    roi = "2"
    codebook = "pi"
    workspace = _write_minimal_workspace(root=tmp_path / "ws", roi=roi, codebook=codebook, channels=[codebook, "c1", "c2"])

    syn.run_pipeline(
        workspace=workspace,
        roi=roi,
        stitch_codebook=codebook,
        run_dirname="test_run",
        p1_threshold=None,
        landmark_linear_transform_type="affine",
        syn_type_of_transform="SyNOnly",
        syn_metric="mattes",
        syn_metric_param=8,
        syn_reg_iterations=(5, 2, 1),
        syn_grad_step=0.15,
        syn_flow_sigma=4.0,
        syn_total_sigma=1.0,
        moving_presmooth_sigma_um=0.0,
        fixed_edge_guard_um=0.0,
        moving_edge_guard_um=0.0,
        use_affine_refine=False,
        use_n4=False,
        use_feature_images=False,
        use_mask_distance_metric=True,
        use_landmark_heatmap_metric=True,
        landmark_heatmap_sigma_um=70.0,
        landmark_heatmap_weight=0.4,
        crop_fixed_to_overlap=False,
    )

    runner = CliRunner()
    result = runner.invoke(
        export.main,
        [
            str(workspace),
            "--run-dirname",
            "test_run",
            "--z-idx",
            "0",
            "--moving-pre-downsample",
            "8",
            "--target-spacing-um",
            "2",
            "--term",
            "1",
            "--kind",
            "id",
        ],
    )
    assert result.exit_code == 0, result.output

    out_dir = workspace / "analysis" / "output" / "ccf-transforms" / roi / "test_run" / "mask_edit"
    png = out_dir / "warped_moving_z0_ds8_target2um.png"
    ref_png = out_dir / "reference_slice_z0_ds8_target2um.png"
    mask = out_dir / "mask_z0_ds8_target2um.tif"
    assert not png.exists()
    assert not ref_png.exists()
    assert not mask.exists()


def test_export_mask_edit_pack_all_rois_skips_missing_run_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from ccf import ants_landmark_syn_init as syn
    from ccf import cli_export_mask_edit_pack as export

    monkeypatch.setattr(syn, "BrainGlobeAtlas", _DummyAtlas)
    monkeypatch.setattr(export, "BrainGlobeAtlas", _DummyAtlas)

    codebook = "pi"
    workspace = tmp_path / "ws"
    _write_minimal_workspace(root=workspace, roi="1", codebook=codebook, channels=[codebook, "c1", "c2"])
    _write_minimal_workspace(root=workspace, roi="2", codebook=codebook, channels=[codebook, "c1", "c2"])

    syn.run_pipeline(
        workspace=workspace,
        roi="2",
        stitch_codebook=codebook,
        run_dirname="test_run",
        p1_threshold=None,
        landmark_linear_transform_type="affine",
        syn_type_of_transform="SyNOnly",
        syn_metric="mattes",
        syn_metric_param=8,
        syn_reg_iterations=(5, 2, 1),
        syn_grad_step=0.15,
        syn_flow_sigma=4.0,
        syn_total_sigma=1.0,
        moving_presmooth_sigma_um=0.0,
        fixed_edge_guard_um=0.0,
        moving_edge_guard_um=0.0,
        use_affine_refine=False,
        use_n4=False,
        use_feature_images=False,
        use_mask_distance_metric=True,
        use_landmark_heatmap_metric=True,
        landmark_heatmap_sigma_um=70.0,
        landmark_heatmap_weight=0.4,
        crop_fixed_to_overlap=False,
    )

    runner = CliRunner()
    result = runner.invoke(
        export.main,
        [
            str(workspace),
            "--run-dirname",
            "test_run",
            "--z-idx",
            "0",
            "--moving-pre-downsample",
            "8",
            "--target-spacing-um",
            "2",
            "--term",
            "1",
            "--kind",
            "id",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "skipping" in result.output

    out_dir = workspace / "analysis" / "output" / "ccf-transforms" / "2" / "test_run" / "mask_edit"
    assert (out_dir / "warped_moving_z0_ds8_target2um.png").exists()
    assert (out_dir / "reference_slice_z0_ds8_target2um.png").exists()
    assert (out_dir / "mask_z0_ds8_target2um.tif").exists()
