from __future__ import annotations

import numpy as np


def assign_temporal_order_from_brdu_edu(*, brdu_pos: np.ndarray, edu_pos: np.ndarray) -> np.ndarray:
    """Assign the dual-pulse temporal order EdU-only -> dual-positive -> BrdU-only.

    Cells that are negative for both pulses are left as ``-1`` so callers can
    decide whether to exclude them or treat them separately.
    """

    if brdu_pos.shape != edu_pos.shape:
        raise ValueError(f"`brdu_pos` and `edu_pos` must have the same shape, got {brdu_pos.shape} vs {edu_pos.shape}.")

    brdu = np.asarray(brdu_pos, dtype=bool)
    edu = np.asarray(edu_pos, dtype=bool)

    time = np.full(brdu.shape, fill_value=-1, dtype=int)
    time[(~brdu) & edu] = 0
    time[brdu & edu] = 1
    time[brdu & (~edu)] = 2
    return time
