"""Z-slice selection and projection helpers.

This module provides a small, well-typed surface for selecting Z ranges from
volumetric stacks and optionally collapsing them via maximum projection. It is
intentionally minimal to keep testability high and avoid heavyweight GPU
dependencies in the core logic. A private ``_max_project_gpu`` hook exists for
tests to monkeypatch when GPU execution should be asserted.
"""

from __future__ import annotations

import os
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

SliceRanges = Sequence[tuple[int, int]]


def _as_bool_env(name: str, default: bool = False) -> bool:
    """Parse a boolean environment variable in a forgiving way.

    Accepts 1/0, true/false, yes/no, on/off (case-insensitive). Falls back to
    ``default`` when unset or unrecognized.
    """

    raw = os.environ.get(name)
    if raw is None:
        return default
    val = raw.strip().lower()
    if val in {"1", "true", "yes", "on"}:
        return True
    if val in {"0", "false", "no", "off"}:
        return False
    return default


def _max_project_cpu(volume: NDArray[np.generic], s: slice) -> NDArray[np.generic]:
    """CPU implementation of a max projection across Z for the given slice."""

    # ``axis=0`` is the Z dimension in our stacks
    return np.asarray(volume[s]).max(axis=0)


def _max_project_gpu(volume: NDArray[np.generic], s: slice) -> NDArray[np.generic]:
    """GPU path placeholder for max projection over Z.

    The default implementation uses the CPU for portability. Tests can
    monkeypatch this function to verify that the GPU pathway is exercised
    based on flags/environment.
    """

    return _max_project_cpu(volume, s)


def collapse_slices(
    volume: NDArray[np.generic],
    spec: slice | SliceRanges | None,
    *,
    use_gpu: bool | None = None,
) -> NDArray[np.generic]:
    """Select and optionally collapse Z-slices from ``volume``.

    Behavior by ``spec`` type:
    - ``None``: return ``volume`` unchanged (view).
    - ``slice``: return ``volume[spec]`` (view; no projection).
    - list of ``(start, end)`` tuples: for each tuple, compute a max
      projection over ``axis=0`` within that Z-range and stack results along a
      new leading dimension.

    The GPU path is activated when ``use_gpu is True`` or when the environment
    variable ``GPU`` parses as truthy via :func:`_as_bool_env`.
    """

    if spec is None:
        # Identity: retain original view
        return volume

    if isinstance(spec, slice):
        # Simple range selection; do not collapse
        return volume[spec]

    # Otherwise a sequence of (start, end) pairs → project each range
    if use_gpu is None:
        use_gpu = _as_bool_env("GPU", default=False)

    projections: list[NDArray[np.generic]] = []
    for start, end in spec:
        s = slice(int(start), int(end))
        proj = _max_project_gpu(volume, s) if use_gpu else _max_project_cpu(volume, s)
        projections.append(proj)

    # Stack along a new first dimension to match tests' expectations
    return np.stack(projections, axis=0)


def collapse_z(
    volume: NDArray[np.generic],
    spec: slice | SliceRanges | None,
    *,
    use_gpu: bool | None = None,
) -> NDArray[np.generic]:
    """Compatibility wrapper used by ``cli_register``.

    Delegates to :func:`collapse_slices`.
    """

    return collapse_slices(volume, spec, use_gpu=use_gpu)


__all__ = ["collapse_slices", "collapse_z"]
