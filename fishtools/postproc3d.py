"""Compatibility shim for legacy imports.

postproc3d lives under ``fishtools.segment.postproc3d``.
This module re-exports its public API to avoid breakage.
"""

from __future__ import annotations

import sys
from types import ModuleType

from fishtools.segment.postproc3d import *  # noqa: F403
from fishtools.segment.postproc3d import (  # noqa: F401
    compute_metadata_and_adjacency,
    donate_small_cells,
    gaussian_erosion_to_margin_and_scale,
    gaussian_smooth_labels,
    gaussian_smooth_labels_cupy,
    relabel_connected_components,
    smooth_masks_3d,
)

# Ensure third-party examples importing "cellpose.postproc3d" resolve to us as well.
mod: ModuleType = sys.modules[__name__]
sys.modules.setdefault("cellpose.postproc3d", mod)
