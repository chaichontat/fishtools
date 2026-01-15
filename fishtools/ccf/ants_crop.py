from __future__ import annotations

import numpy as np


def fixed_crop_bounds_from_masks_xy(
    *,
    fixed_mask_xy: np.ndarray,
    moving_mask_in_fixed_xy: np.ndarray,
    pad_vox: int,
) -> tuple[list[int], list[int]] | None:
    """Compute crop bounds for a fixed-space image, in (x,y) index order.

    Parameters
    ----------
    fixed_mask_xy:
        Fixed-space mask array in ANTs-style (x,y) axis order.
    moving_mask_in_fixed_xy:
        Moving mask warped into fixed space, in the same (x,y) axis order and shape.
    pad_vox:
        Padding in voxels around the computed bounding box.

    Returns
    -------
    (lower, upper) indices for `ants.crop_indices`, or None if the mask is empty.
    """

    pad = int(pad_vox)
    if pad < 0:
        raise ValueError(f"pad_vox must be >= 0, got {pad_vox}.")
    if fixed_mask_xy.shape != moving_mask_in_fixed_xy.shape:
        raise ValueError(
            f"Mask shapes must match, got fixed_mask_xy.shape={fixed_mask_xy.shape} vs moving_mask_in_fixed_xy.shape={moving_mask_in_fixed_xy.shape}."
        )

    moving_xy = np.asarray(moving_mask_in_fixed_xy) > 0
    if not np.any(moving_xy):
        return None

    idx = np.argwhere(moving_xy)
    lo_x, lo_y = idx.min(axis=0).tolist()
    hi_x, hi_y = idx.max(axis=0).tolist()

    lower = [max(0, int(lo_x - pad)), max(0, int(lo_y - pad))]
    upper = [min(fixed_mask_xy.shape[0], int(hi_x + pad + 1)), min(fixed_mask_xy.shape[1], int(hi_y + pad + 1))]
    return lower, upper
