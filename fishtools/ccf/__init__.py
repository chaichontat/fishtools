"""CCF (Common Coordinate Framework) utilities.

This package contains small, typed helpers used by notebook-style CCF workflows
in `ccf/` so those workflows can share a stable, testable implementation.
"""

from fishtools.ccf.landmark import LandmarkRegistrationOutputs, P1Landmarks

__all__ = [
    "P1Landmarks",
    "LandmarkRegistrationOutputs",
]
