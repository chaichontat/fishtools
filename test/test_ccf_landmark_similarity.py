from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

from ccf.landmark import LandmarkRegistrationOutputs, P1Landmarks
from ccf.sitk_utils import compute_similarity2d_from_landmarks


@pytest.mark.unit
def test_compute_similarity2d_from_landmarks_maps_fixed_to_moving() -> None:
    fixed = [(5.0, 7.0), (12.0, 3.0), (9.0, 18.0), (20.0, 11.0), (2.0, 15.0)]

    gt = sitk.Similarity2DTransform()
    gt.SetCenter((10.0, 10.0))
    gt.SetAngle(float(np.deg2rad(30.0)))
    gt.SetScale(1.2)
    gt.SetTranslation((5.0, -3.0))

    moving = [gt.TransformPoint(p) for p in fixed]

    est = compute_similarity2d_from_landmarks(fixed, moving)

    for fp, mp in zip(fixed, moving, strict=True):
        assert np.allclose(est.TransformPoint(fp), mp, atol=1e-6)


@pytest.mark.unit
def test_p1_landmarks_roundtrip(tmp_path: Path) -> None:
    out = LandmarkRegistrationOutputs(tmp_path)
    p1 = P1Landmarks(
        prior_rotation_deg=12,
        atlas_slice_idx=321,
        fixed_points_cropped_xy=[(1.25, 2.5), (3.0, 4.0), (5.0, 6.0)],
        moving_points_fullres_xy_in_rotated_crop=[(10.0, 20.0), (30.0, 40.0), (50.0, 60.0)],
        atlas_crop_bbox=(1, 2, 3, 4),
        sample_rotated_crop_bbox=(5, 6, 7, 8),
        preview_downsample=8,
        atlas_full_shape_yx=(100, 200),
        sample_rotated_full_shape_yx=(300, 400),
    )

    out.write_p1_landmarks(p1)
    assert out.p1_landmarks_json.exists()

    p2 = out.read_p1_landmarks()
    assert p2 == p1
