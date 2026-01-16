import numpy as np

from fishtools.preprocess.chromatic import apply_dense_xy_displacement_field


def test_apply_dense_xy_displacement_field_matches_pull_convention() -> None:
    # One-slice image with a single bright pixel.
    img = np.zeros((1, 6, 7), dtype=np.float32)
    img[0, 2, 3] = 1.0

    dx = np.ones((6, 7), dtype=np.float32)  # sample input at x+1 => content shifts left by 1
    dy = np.zeros((6, 7), dtype=np.float32)

    warped = apply_dense_xy_displacement_field(img, dx_dense=dx, dy_dense=dy, default_pixel_value=0.0)

    assert warped.shape == img.shape
    assert float(warped[0, 2, 2]) == 1.0
    assert float(np.sum(warped)) == 1.0


def test_apply_dense_xy_displacement_field_applies_to_each_z_slice() -> None:
    img = np.zeros((2, 6, 7), dtype=np.float32)
    img[0, 2, 3] = 1.0
    img[1, 4, 5] = 2.0

    dx = np.ones((6, 7), dtype=np.float32)
    dy = np.zeros((6, 7), dtype=np.float32)

    warped = apply_dense_xy_displacement_field(img, dx_dense=dx, dy_dense=dy, default_pixel_value=0.0)

    assert warped.shape == img.shape
    assert float(warped[0, 2, 2]) == 1.0
    assert float(warped[1, 4, 4]) == 2.0
