from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import special


@dataclass(frozen=True)
class NdimageRotateParams:
    rot_matrix_yx: NDArray[np.float64]
    offset_yx: NDArray[np.float64]
    out_shape_yx: tuple[int, int]


def ndimage_rotate_params(*, in_shape_yx: tuple[int, int], angle_deg: float, reshape: bool = True) -> NdimageRotateParams:
    """
    Compute the (matrix, offset, out_shape) used by `scipy.ndimage.rotate` for 2D arrays.

    SciPy implements rotation by calling `affine_transform(input, matrix, offset, ...)` where
    the transform maps output indices -> input indices:

        [y_in, x_in]^T = matrix @ [y_out, x_out]^T + offset

    This returns the same `matrix` and `offset` for `reshape=True`, and the output shape.
    """
    if not reshape:
        raise ValueError("Only reshape=True is supported (matches our CCF pipeline).")
    iy, ix = (int(in_shape_yx[0]), int(in_shape_yx[1]))
    if iy <= 0 or ix <= 0:
        raise ValueError(f"Invalid in_shape_yx={in_shape_yx}.")

    c = float(special.cosdg(angle_deg))
    s = float(special.sindg(angle_deg))
    rot_matrix = np.array([[c, s], [-s, c]], dtype=np.float64)

    # Matches scipy.ndimage.rotate for reshape=True
    out_bounds = rot_matrix @ np.array([[0.0, 0.0, float(iy), float(iy)], [0.0, float(ix), 0.0, float(ix)]])
    out_plane_shape = (np.ptp(out_bounds, axis=1) + 0.5).astype(int)
    out_shape_yx = (int(out_plane_shape[0]), int(out_plane_shape[1]))

    out_center = rot_matrix @ ((np.array(out_shape_yx, dtype=np.float64) - 1.0) / 2.0)
    in_center = (np.array([float(iy), float(ix)], dtype=np.float64) - 1.0) / 2.0
    offset = in_center - out_center

    return NdimageRotateParams(rot_matrix_yx=rot_matrix, offset_yx=offset, out_shape_yx=out_shape_yx)


def ndimage_rotate_input_to_output_yx(
    *,
    y_in: NDArray[np.floating],
    x_in: NDArray[np.floating],
    in_shape_yx: tuple[int, int],
    angle_deg: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], tuple[int, int]]:
    params = ndimage_rotate_params(in_shape_yx=in_shape_yx, angle_deg=angle_deg, reshape=True)
    y_in_f = np.asarray(y_in, dtype=np.float64)
    x_in_f = np.asarray(x_in, dtype=np.float64)
    if y_in_f.shape != x_in_f.shape:
        raise ValueError(f"x/y shape mismatch: {x_in_f.shape} vs {y_in_f.shape}.")

    in_yx = np.stack([y_in_f, x_in_f], axis=0)
    out_yx = params.rot_matrix_yx.T @ (in_yx - params.offset_yx[:, None])
    return out_yx[0], out_yx[1], params.out_shape_yx


def ndimage_rotate_output_to_input_yx(
    *,
    y_out: NDArray[np.floating],
    x_out: NDArray[np.floating],
    in_shape_yx: tuple[int, int],
    angle_deg: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], tuple[int, int]]:
    params = ndimage_rotate_params(in_shape_yx=in_shape_yx, angle_deg=angle_deg, reshape=True)
    y_out_f = np.asarray(y_out, dtype=np.float64)
    x_out_f = np.asarray(x_out, dtype=np.float64)
    if y_out_f.shape != x_out_f.shape:
        raise ValueError(f"x/y shape mismatch: {x_out_f.shape} vs {y_out_f.shape}.")

    out_yx = np.stack([y_out_f, x_out_f], axis=0)
    in_yx = params.rot_matrix_yx @ out_yx + params.offset_yx[:, None]
    return in_yx[0], in_yx[1], params.out_shape_yx


def fused_xy_to_rotated_full_xy(
    *,
    x_fused: NDArray[np.floating],
    y_fused: NDArray[np.floating],
    fused_shape_yx: tuple[int, int],
    prior_flip_x: bool,
    prior_rotation_deg: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], tuple[int, int]]:
    x = np.asarray(x_fused, dtype=np.float64)
    y = np.asarray(y_fused, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError(f"x/y shape mismatch: {x.shape} vs {y.shape}.")

    h, w = (int(fused_shape_yx[0]), int(fused_shape_yx[1]))
    if prior_flip_x:
        x = (float(w) - 1.0) - x

    if float(prior_rotation_deg) == 0.0:
        return x, y, (h, w)

    y_rot, x_rot, out_shape_yx = ndimage_rotate_input_to_output_yx(
        y_in=y,
        x_in=x,
        in_shape_yx=(h, w),
        angle_deg=float(prior_rotation_deg),
    )
    return x_rot, y_rot, out_shape_yx


def fused_xy_to_rotated_crop_xy(
    *,
    x_fused: NDArray[np.floating],
    y_fused: NDArray[np.floating],
    fused_shape_yx: tuple[int, int],
    prior_flip_x: bool,
    prior_rotation_deg: float,
    rotated_crop_bbox: tuple[int, int, int, int],
) -> tuple[NDArray[np.float64], NDArray[np.float64], tuple[int, int]]:
    x_full, y_full, rotated_shape_yx = fused_xy_to_rotated_full_xy(
        x_fused=x_fused,
        y_fused=y_fused,
        fused_shape_yx=fused_shape_yx,
        prior_flip_x=prior_flip_x,
        prior_rotation_deg=prior_rotation_deg,
    )
    sr0, sr1, sc0, sc1 = (int(rotated_crop_bbox[0]), int(rotated_crop_bbox[1]), int(rotated_crop_bbox[2]), int(rotated_crop_bbox[3]))
    if sr1 <= sr0 or sc1 <= sc0:
        raise ValueError(f"Invalid rotated_crop_bbox={rotated_crop_bbox}.")
    return x_full - float(sc0), y_full - float(sr0), rotated_shape_yx

