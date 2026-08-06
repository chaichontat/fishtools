from __future__ import annotations

import numpy as np

from ccf.ants_crop import fixed_crop_bounds_from_masks_xy


def test_fixed_crop_bounds_from_masks_xy_uses_warped_moving_mask_not_overlap() -> None:
    fixed_mask = np.zeros((100, 100), dtype=bool)
    fixed_mask[20:80, 20:80] = True

    # Warped moving mask extends outside fixed mask (top-left corner) but still
    # contains an in-mask region too.
    moving_mask_in_fixed = np.zeros((100, 100), dtype=bool)
    moving_mask_in_fixed[0:10, 0:10] = True
    moving_mask_in_fixed[25:30, 25:30] = True

    lower, upper = fixed_crop_bounds_from_masks_xy(
        fixed_mask_xy=fixed_mask,
        moving_mask_in_fixed_xy=moving_mask_in_fixed,
        pad_vox=0,
    )

    assert lower == [0, 0]
    assert upper == [30, 30]

