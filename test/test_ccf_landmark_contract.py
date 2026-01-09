from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from fishtools.ccf.landmark import LandmarkRegistrationOutputs, P1Landmarks
from fishtools.ccf.sitk_utils import normalize_robust


def test_outputs_contract_paths(tmp_path: Path) -> None:
    out = LandmarkRegistrationOutputs(tmp_path)

    assert out.p1_landmarks_json == tmp_path / "p1_landmarks.json"
    assert out.p1_similarity_tfm == tmp_path / "p1_similarity.tfm"
    assert out.p1_result_png == tmp_path / "p1_result.png"


def test_p1_landmarks_parsing_and_rotation_loading(tmp_path: Path) -> None:
    out = LandmarkRegistrationOutputs(tmp_path)

    payload = {
        "prior_rotation_deg": 70,
        "prior_flip_x": True,
        "fixed_points_cropped_xy": [[1, 2], [3.5, 4.5], [10, 11]],
        "moving_points_fullres_xy_in_rotated_crop": [[5, 6], [7, 8], [9, 10]],
        "atlas_crop_bbox": [1, 2, 3, 4],
        "sample_rotated_crop_bbox": [5, 6, 7, 8],
        "atlas_plane": "coronal",
        "preview_downsample": 8,
        "atlas_full_shape_yx": [100, 200],
        "sample_rotated_full_shape_yx": [300, 400],
    }
    out.p1_landmarks_json.write_text(json.dumps(payload), encoding="utf-8")

    lm = P1Landmarks.from_json(out.p1_landmarks_json)
    assert lm.prior_rotation_deg == 70
    assert lm.prior_flip_x is True
    assert lm.atlas_plane == "coronal"
    assert lm.atlas_crop_bbox == (1, 2, 3, 4)
    assert lm.sample_rotated_crop_bbox == (5, 6, 7, 8)
    assert lm.preview_downsample == 8
    assert lm.atlas_full_shape_yx == (100, 200)
    assert lm.sample_rotated_full_shape_yx == (300, 400)

    assert out.try_read_prior_rotation_deg() == 70
    assert out.read_p1_landmarks().prior_rotation_deg == 70
    assert out.try_read_p1_landmarks() == lm


def test_try_read_p1_landmarks_missing_returns_none(tmp_path: Path) -> None:
    out = LandmarkRegistrationOutputs(tmp_path)
    assert out.try_read_p1_landmarks() is None


def test_p1_landmarks_missing_keys_raises(tmp_path: Path) -> None:
    path = tmp_path / "p1_landmarks.json"
    path.write_text(json.dumps({"prior_rotation_deg": 0}), encoding="utf-8")

    with pytest.raises(ValueError, match="Missing key"):
        _ = P1Landmarks.from_json(path)


def test_outputs_write_and_read_p1_landmarks(tmp_path: Path) -> None:
    out = LandmarkRegistrationOutputs(tmp_path)

    lm = P1Landmarks(
        prior_rotation_deg=12,
        prior_flip_x=True,
        fixed_points_cropped_xy=[(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)],
        moving_points_fullres_xy_in_rotated_crop=[(7.0, 8.0), (9.0, 10.0), (11.0, 12.0)],
        atlas_crop_bbox=(1, 2, 3, 4),
        sample_rotated_crop_bbox=(5, 6, 7, 8),
        atlas_plane="sagittal",
        preview_downsample=8,
        atlas_full_shape_yx=(100, 200),
        sample_rotated_full_shape_yx=(300, 400),
    )

    out.write_p1_landmarks(lm)
    assert out.read_p1_landmarks() == lm


def test_normalize_robust_scales_to_unit_interval() -> None:
    arr = normalize_robust(np.array([0.0, 1.0, 2.0, 3.0]), lo=0.0, hi=100.0)
    assert np.allclose(arr, np.array([0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0]))
