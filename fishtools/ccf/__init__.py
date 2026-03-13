"""CCF (Common Coordinate Framework) utilities.

This package contains small, typed helpers used by notebook-style CCF workflows
in `ccf/` so those workflows can share a stable, testable implementation.
"""

from fishtools.ccf.landmark import LandmarkRegistrationOutputs, P1Landmarks
from fishtools.ccf.native_surface_plotting import (
    ApMlNativeSurfaceProjectionContext,
    build_apml_native_surface_projection_context,
    plot_coronal_surface_projection,
    write_apml_native_proj_montage,
)
from fishtools.ccf.ontology import CCFTermKind, filter_ccf_subtree, mask_ccf_subtree

__all__ = [
    "CCFTermKind",
    "P1Landmarks",
    "LandmarkRegistrationOutputs",
    "ApMlNativeSurfaceProjectionContext",
    "build_apml_native_surface_projection_context",
    "plot_coronal_surface_projection",
    "write_apml_native_proj_montage",
    "filter_ccf_subtree",
    "mask_ccf_subtree",
]
