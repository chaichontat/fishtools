from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch
import zarr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def _load_infer_module():  # type: ignore[return-type]
    script_path = Path(__file__).parents[1] / "scripts/segmentation/brdu_edu_infer_cell_dino.py"
    spec = importlib.util.spec_from_file_location("brdu_edu_infer_cell_dino", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load inference script from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _IdentityModel(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :2, 0, 0]


class _MeanModel(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :2].mean(dim=(2, 3)) + 1.0


def _write_fake_dinov2_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "dinov2"
    transforms_path = repo / "dinov2/data/cell_dino/transforms.py"
    transforms_path.parent.mkdir(parents=True, exist_ok=True)
    for init_path in [
        repo / "dinov2/__init__.py",
        repo / "dinov2/data/__init__.py",
        repo / "dinov2/data/cell_dino/__init__.py",
    ]:
        init_path.write_text("", encoding="utf-8")
    transforms_path.write_text(
        "\n".join(
            [
                "import torch",
                "",
                "class _IdentityTransform:",
                "    def __call__(self, batch: torch.Tensor) -> torch.Tensor:",
                "        return batch",
                "",
                "def make_classification_eval_cell_transform(*, resize_size: int, crop_size: int):",
                "    return _IdentityTransform()",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return repo


def _make_group_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "cell": "roiA|11",
                "dataset": "DS1",
                "roi": "roiA",
                "x": 5.0,
                "y": 4.0,
                "z": 0.0,
                "_x_center": 5,
                "_y_center": 4,
                "_z_index": 0,
                "_label": 11,
                "_row_order": 5,
            },
            {
                "cell": "roiA|21",
                "dataset": "DS1",
                "roi": "roiA",
                "x": 3.0,
                "y": 8.0,
                "z": 1.0,
                "_x_center": 3,
                "_y_center": 8,
                "_z_index": 1,
                "_label": 21,
                "_row_order": 2,
            },
            {
                "cell": "roiA|12",
                "dataset": "DS1",
                "roi": "roiA",
                "x": 10.0,
                "y": 9.0,
                "z": 0.0,
                "_x_center": 10,
                "_y_center": 9,
                "_z_index": 0,
                "_label": 12,
                "_row_order": 9,
            },
            {
                "cell": "roiA|22",
                "dataset": "DS1",
                "roi": "roiA",
                "x": 9.0,
                "y": 2.0,
                "z": 1.0,
                "_x_center": 9,
                "_y_center": 2,
                "_z_index": 1,
                "_label": 22,
                "_row_order": 1,
            },
        ]
    )


def test_assemble_cell_dino_batch_matches_full_plane_reference() -> None:
    module = _load_infer_module()

    fused_plane = np.arange(12 * 13 * 2, dtype=np.uint16).reshape(12, 13, 2)
    seg_plane = np.zeros((12, 13), dtype=np.int32)
    seg_plane[2:6, 3:7] = 11
    seg_plane[7:10, 8:12] = 22

    rows = pd.DataFrame(
        [
            {
                "cell": "roiA|11",
                "dataset": "DS1",
                "roi": "roiA",
                "x": 5.0,
                "y": 4.0,
                "_x_center": 5,
                "_y_center": 4,
                "_label": 11,
                "_row_order": 0,
            },
            {
                "cell": "roiA|22",
                "dataset": "DS1",
                "roi": "roiA",
                "x": 10.0,
                "y": 9.0,
                "_x_center": 10,
                "_y_center": 9,
                "_label": 22,
                "_row_order": 1,
            },
        ]
    )
    crop_size = 6
    y0, y1, x0, x1 = module._slab_bounds_for_group(
        rows,
        crop_size=crop_size,
        y_dim=fused_plane.shape[0],
        x_dim=fused_plane.shape[1],
    )
    fused_slab = fused_plane[y0:y1, x0:x1, :]
    seg_slab = seg_plane[y0:y1, x0:x1]

    got = module._assemble_cell_dino_batch(
        batch_rows=rows,
        fused_slab=fused_slab,
        seg_slab=seg_slab,
        crop_size=crop_size,
        slab_x0=x0,
        slab_y0=y0,
        input_channels=5,
    )

    expected = []
    for row in rows.to_dict("records"):
        x_center = int(np.rint(float(row["x"])))
        y_center = int(np.rint(float(row["y"])))
        _, label = module._cell_id_roi_and_label(str(row["cell"]))
        thumb = module._crop_centered_yxc(fused_plane, x_center=x_center, y_center=y_center, size=crop_size)
        mask = module._resolve_segmentation_mask_from_plane(
            seg_plane,
            x_center=x_center,
            y_center=y_center,
            size=crop_size,
            label=label,
        )
        image_hwc = module._compose_cell_dino_channels(thumb, mask=mask)
        expected.append(np.moveaxis(image_hwc, -1, 0))

    expected_batch = np.stack(expected, axis=0)
    assert np.array_equal(got, expected_batch)


def test_slab_batch_loader_matches_direct_batch_assembly(tmp_path: Path) -> None:
    module = _load_infer_module()
    repo = _write_fake_dinov2_repo(tmp_path)
    group = _make_group_rows()

    fused_path = tmp_path / "fused.zarr"
    seg_path = tmp_path / "seg.zarr"
    fused_data = np.arange(2 * 12 * 13 * 2, dtype=np.uint16).reshape(2, 12, 13, 2)
    seg_data = np.zeros((2, 12, 13), dtype=np.int32)
    seg_data[0, 2:6, 3:7] = 11
    seg_data[0, 7:10, 8:12] = 12
    seg_data[1, 6:10, 1:5] = 21
    seg_data[1, 0:4, 7:11] = 22
    zarr.open_array(fused_path, mode="w", shape=fused_data.shape, dtype=fused_data.dtype)[:] = fused_data
    zarr.open_array(seg_path, mode="w", shape=seg_data.shape, dtype=seg_data.dtype)[:] = seg_data

    dataset = module._SlabBatchIterableDataset(
        group,
        fused_path=fused_path,
        seg_path=seg_path,
        channel_indices=(0, 1),
        crop_size=6,
        input_channels=5,
        resize_size=6,
        crop_eval_size=6,
        batch_size=1,
        repo=repo,
        y_dim=fused_data.shape[1],
        x_dim=fused_data.shape[2],
    )
    loader = module._build_batch_loader(dataset, num_workers=0, device=torch.device("cpu"))

    expected_specs = module._make_slab_batch_specs(
        group,
        crop_size=6,
        y_dim=fused_data.shape[1],
        x_dim=fused_data.shape[2],
        batch_size=1,
    )
    got_batches = list(loader)
    assert len(got_batches) == len(expected_specs)

    for (batch_tensor, batch_meta), spec in zip(got_batches, expected_specs, strict=True):
        batch_rows = group.iloc[list(spec.row_indices)]
        fused_slab = fused_data[spec.z_index, spec.y0 : spec.y1, spec.x0 : spec.x1, :]
        seg_slab = seg_data[spec.z_index, spec.y0 : spec.y1, spec.x0 : spec.x1]
        expected_batch = module._assemble_cell_dino_batch(
            batch_rows=batch_rows,
            fused_slab=fused_slab,
            seg_slab=seg_slab,
            crop_size=6,
            slab_x0=spec.x0,
            slab_y0=spec.y0,
            input_channels=5,
        )

        assert np.array_equal(batch_tensor.numpy(), expected_batch.astype(np.float32))
        assert batch_meta["cell"] == batch_rows["cell"].tolist()
        assert batch_meta["dataset"] == batch_rows["dataset"].tolist()
        assert batch_meta["roi"] == batch_rows["roi"].tolist()
        assert np.array_equal(batch_meta["_row_order"], batch_rows["_row_order"].to_numpy(dtype=np.int64))


def test_assemble_cell_dino_batch_keeps_zero_intensity_masked_cells() -> None:
    module = _load_infer_module()

    fused_plane = np.zeros((8, 8, 2), dtype=np.uint16)
    seg_plane = np.zeros((8, 8), dtype=np.int32)
    seg_plane[2:6, 2:6] = 7
    rows = pd.DataFrame(
        [
            {
                "cell": "roiA|7",
                "dataset": "DS1",
                "roi": "roiA",
                "x": 4.0,
                "y": 4.0,
                "_x_center": 4,
                "_y_center": 4,
                "_label": 7,
                "_row_order": 0,
            }
        ]
    )

    got = module._assemble_cell_dino_batch(
        batch_rows=rows,
        fused_slab=fused_plane,
        seg_slab=seg_plane,
        crop_size=4,
        slab_x0=0,
        slab_y0=0,
        input_channels=5,
    )

    assert got.shape == (1, 5, 4, 4)
    assert np.count_nonzero(got) == 0


def test_resolve_output_path_defaults_to_nvme_dataset_path() -> None:
    module = _load_infer_module()

    cfg = module.InferConfig(
        barrage_dir=Path("output/brdu_edu_barrage"),
        h5ad_path=Path("/tmp/input.h5ad"),
        model_path=Path("output/brdu_edu_barrage/brdu_edu_cell_dino_hpa_vitl16_logreg.joblib"),
        output_path=None,
        dinov2_repo=None,
        weights=None,
        datasets=(),
        rois=(),
        batch_size=128,
        num_workers=4,
    )
    table = pd.DataFrame({"dataset": ["20251201_JaxA6_Coro6"]})

    got = module._resolve_output_path(cfg, table)

    assert got == Path("~/nvme/dinoinfer/20251201_JaxA6_Coro6-dino-vit16.parquet").expanduser()


def test_predict_group_preserves_row_order_with_multiple_workers(tmp_path: Path, monkeypatch: Any) -> None:
    module = _load_infer_module()
    repo = _write_fake_dinov2_repo(tmp_path)
    workspace_root = tmp_path / "DS1"
    workspace_root.mkdir(parents=True, exist_ok=True)
    (workspace_root / "ready.DONE").write_text("", encoding="utf-8")
    ws = module.Workspace(workspace_root)
    group = _make_group_rows()

    fused_path = ws.stitch("roiA", "edu") / module.FUSED_ZARR_NAME
    seg_path = ws.stitch("roiA", "pi") / "segmentation.zarr"
    fused_path.parent.mkdir(parents=True, exist_ok=True)
    seg_path.parent.mkdir(parents=True, exist_ok=True)
    fused_data = np.arange(2 * 12 * 13 * 2, dtype=np.uint16).reshape(2, 12, 13, 2)
    seg_data = np.zeros((2, 12, 13), dtype=np.int32)
    seg_data[0, 2:6, 3:7] = 11
    seg_data[0, 7:10, 8:12] = 12
    seg_data[1, 6:10, 1:5] = 21
    seg_data[1, 0:4, 7:11] = 22
    fused = zarr.open_array(fused_path, mode="w", shape=fused_data.shape, dtype=fused_data.dtype)
    fused[:] = fused_data
    fused.attrs["key"] = ["brdu", "edu"]
    zarr.open_array(seg_path, mode="w", shape=seg_data.shape, dtype=seg_data.dtype)[:] = seg_data

    train_features = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 1.0],
        ],
        dtype=np.float32,
    )
    scaler = StandardScaler().fit(train_features)
    brdu_model = LogisticRegression(random_state=0, solver="lbfgs").fit(train_features, np.asarray([0, 0, 1, 1]))
    edu_model = LogisticRegression(random_state=0, solver="lbfgs").fit(train_features, np.asarray([1, 0, 1, 0]))

    monkeypatch.setattr(module, "_resolve_workspace_root", lambda dataset: workspace_root)
    out = module._predict_group(
        group=group,
        model=_MeanModel(),
        bundle={
            "feature_extractor": "cell_dino_cp_vits8",
            "codebook": "edu",
            "seg_codebook": "pi",
            "segmentation_name": "segmentation.zarr",
            "crop_size": 6,
            "resize_size": 6,
            "crop_eval_size": 6,
            "feature_scaler": scaler,
            "brdu_model": brdu_model,
            "edu_model": edu_model,
        },
        repo=repo,
        device=torch.device("cpu"),
        batch_size=1,
        num_workers=2,
    )

    assert out["_row_order"].tolist() == [1, 2, 5, 9]
    assert out["cell"].tolist() == ["roiA|22", "roiA|21", "roiA|11", "roiA|12"]
    assert ((out["brdu_prob"] >= 0.0) & (out["brdu_prob"] <= 1.0)).all()
    assert ((out["edu_prob"] >= 0.0) & (out["edu_prob"] <= 1.0)).all()


def test_predict_transformed_batch_returns_probability_columns_in_input_order() -> None:
    module = _load_infer_module()

    train_features = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 1.0],
        ],
        dtype=np.float32,
    )
    scaler = StandardScaler().fit(train_features)
    brdu_model = LogisticRegression(random_state=0, solver="lbfgs").fit(train_features, np.asarray([0, 0, 1, 1]))
    edu_model = LogisticRegression(random_state=0, solver="lbfgs").fit(train_features, np.asarray([1, 0, 1, 0]))

    batch_rows = pd.DataFrame(
        [
            {"cell": "roiA|1", "dataset": "DS1", "roi": "roiA", "_row_order": 5},
            {"cell": "roiA|2", "dataset": "DS1", "roi": "roiA", "_row_order": 6},
        ]
    )
    raw_batch = np.asarray(
        [
            [[[3, 4]], [[0, 0]], [[0, 0]], [[0, 0]], [[0, 0]]],
            [[[5, 12]], [[0, 0]], [[0, 0]], [[0, 0]], [[0, 0]]],
        ],
        dtype=np.uint8,
    )

    got = module._predict_transformed_batch(
        batch_meta={
            "cell": batch_rows["cell"].tolist(),
            "dataset": batch_rows["dataset"].tolist(),
            "roi": batch_rows["roi"].tolist(),
            "_row_order": batch_rows["_row_order"].to_numpy(dtype=np.int64),
        },
        batch_images=torch.from_numpy(raw_batch).to(dtype=torch.float32),
        model=_IdentityModel(),
        scaler=scaler,
        brdu_model=brdu_model,
        edu_model=edu_model,
        device=torch.device("cpu"),
    )

    assert got["cell"].tolist() == ["roiA|1", "roiA|2"]
    assert got["_row_order"].tolist() == [5, 6]
    assert ((got["brdu_prob"] >= 0.0) & (got["brdu_prob"] <= 1.0)).all()
    assert ((got["edu_prob"] >= 0.0) & (got["edu_prob"] <= 1.0)).all()


def test_slab_bounds_clip_to_image_edges() -> None:
    module = _load_infer_module()

    rows = pd.DataFrame(
        [
            {"x": 0.0, "y": 1.0},
            {"x": 9.0, "y": 7.0},
        ]
    )

    assert module._slab_bounds_for_group(rows, crop_size=6, y_dim=8, x_dim=10) == (0, 8, 0, 10)
