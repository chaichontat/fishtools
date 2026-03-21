from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
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


def test_assemble_cell_dino_batch_matches_full_plane_reference() -> None:
    module = _load_infer_module()

    fused_plane = np.arange(12 * 13 * 2, dtype=np.uint16).reshape(12, 13, 2)
    seg_plane = np.zeros((12, 13), dtype=np.int32)
    seg_plane[2:6, 3:7] = 11
    seg_plane[7:10, 8:12] = 22

    rows = pd.DataFrame(
        [
            {"cell": "roiA|11", "dataset": "DS1", "roi": "roiA", "x": 5.0, "y": 4.0, "_row_order": 0},
            {"cell": "roiA|22", "dataset": "DS1", "roi": "roiA", "x": 10.0, "y": 9.0, "_row_order": 1},
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


def test_predict_batch_returns_probability_columns_in_input_order() -> None:
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

    got = module._predict_batch(
        batch_rows=batch_rows,
        raw_batch=raw_batch,
        model=_IdentityModel(),
        transform=lambda x: x,
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
