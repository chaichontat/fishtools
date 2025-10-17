"""Shared illumination field helpers for preprocessing pipelines.

This module centralizes the logic for loading LOW/RANGE correction fields
exported by ``correct-illum`` and slicing tile-aligned patches. The helpers
are intentionally lightweight so that CLI entrypoints and programmatic users
can apply field corrections without re-implementing the bookkeeping around
channel labels, workspace ROI resolution, or tile origins.

Design goals:
    - Keep IO concerns (Zarr loading, Workspace access) contained here.
    - Provide shape-agnostic application that works for CYX and ZCYX arrays.
    - Fail fast with contextual error messages when metadata/contracts break.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from fishtools.io.workspace import Workspace

# Cache for pre-exported field stores (LOW/RANGE), Zarr-backed.
_FIELD_STORE_CACHE: dict[str, tuple[np.ndarray, dict[str, Any]]] = {}

__all__ = [
    "apply_field_tcyx_store_to_img",
    "clear_field_store_cache",
    "load_field_t_plane_as_cyx",
    "parse_tile_index_from_path",
    "resolve_roi_for_field",
    "slice_field_patch",
    "tile_origin",
]


def clear_field_store_cache() -> None:
    """Clear the module-level Zarr field cache (mainly for tests)."""

    _FIELD_STORE_CACHE.clear()


# Legacy CYX loader removed; TCYX path is now the default.


def load_field_t_plane_as_cyx(path: Path, kind: str) -> tuple[np.ndarray, dict[str, Any]]:
    """Load a CYX field for a specific kind ("low" or "range") from a TCYX or CYX store.

    - TCYX: selects the T-plane matching ``kind`` using ``t_labels`` attribute
      (defaults to ["low", "range"] if missing), returns the resulting CYX array.
    - CYX: returns the array directly (caller should ensure it matches ``kind``).
    """
    if kind not in {"low", "range"}:
        raise ValueError(f"Unknown field kind: {kind}")

    try:
        import zarr

        za = zarr.open_array(str(path), mode="r")
        arr = np.asarray(za, dtype=np.float32)
        attrs = getattr(za, "attrs", {})
        axes = (attrs.get("axes") or "").upper() if hasattr(attrs, "get") else ""
        model_meta = dict(attrs.get("model_meta", {})) if hasattr(attrs, "get") else {}
    except Exception as exc:  # pragma: no cover - precise zarr errors vary
        raise ValueError(f"Failed to read field Zarr array at {path}: {exc}") from exc

    if arr.ndim == 4 or axes == "TCYX":
        # Determine T index by labels; fallback to canonical order
        t_labels = attrs.get("t_labels") if hasattr(attrs, "get") else None
        if not isinstance(t_labels, (list, tuple)):
            t_labels = ["low", "range"]
        try:
            t_index = int(t_labels.index(kind))
        except ValueError as exc:  # pragma: no cover - defensive
            raise ValueError(f"kind '{kind}' not present in field t_labels={t_labels}") from exc
        cyx = np.asarray(arr[t_index], dtype=np.float32)
        if cyx.ndim == 2:
            cyx = cyx[None, ...]
        elif cyx.ndim != 3:
            raise ValueError(f"Expected CYX slice from TCYX, got shape {cyx.shape}")
        return cyx, model_meta

    # Legacy CYX or YX store
    if arr.ndim == 2:
        arr = arr[None, ...]
    elif arr.ndim != 3:
        raise ValueError(f"Expected CYX or YX field array, got shape {arr.shape}")
    return arr, model_meta


def slice_field_patch(
    field_cyx: np.ndarray,
    field_meta: dict[str, Any],
    channel_name: str,
    *,
    tile_x0: float,
    tile_y0: float,
    out_h: int,
    out_w: int,
    trim: int,
) -> np.ndarray:
    """Slice a YX patch for the given channel and tile from a CYX field."""

    channels = field_meta.get("channels")
    if isinstance(channels, (list, tuple)):
        try:
            cidx = int(channels.index(channel_name))
        except ValueError as exc:  # pragma: no cover - defensive, hard to trigger deterministically
            raise ValueError(f"Channel '{channel_name}' not found in field channels: {channels}") from exc
    else:
        cidx = 0

    downsample = int(field_meta.get("downsample", 1) or 1)
    if downsample != 1:
        raise ValueError(f"Field store downsample={downsample} is unsupported; export with --downsample 1.")

    fx0 = float(field_meta.get("x0", 0.0))
    fy0 = float(field_meta.get("y0", 0.0))

    x_start = int(round((tile_x0 + float(trim)) - fx0))
    y_start = int(round((tile_y0 + float(trim)) - fy0))
    if x_start < 0 or y_start < 0:
        raise ValueError(
            f"Requested field patch starts outside field bounds (x_start={x_start}, y_start={y_start})."
        )

    y_end = y_start + int(out_h)
    x_end = x_start + int(out_w)
    if y_end > field_cyx.shape[1] or x_end > field_cyx.shape[2]:
        raise ValueError(
            "Requested field patch exceeds field bounds: "
            f"({y_end},{x_end}) > ({field_cyx.shape[1]},{field_cyx.shape[2]})."
        )

    patch = field_cyx[cidx, y_start:y_end, x_start:x_end]
    return patch.astype(np.float32, copy=False)


def resolve_roi_for_field(out_path: Path, roi_for_ws: str | None) -> str:
    """Infer the ROI associated with a stitch output directory."""

    if roi_for_ws:
        return str(roi_for_ws)

    name = out_path.name
    if not name.startswith("stitch--"):
        raise ValueError("Unable to infer ROI from output path for field correction")
    roi_cb = name.split("--", 1)[1]
    roi = roi_cb.split("+", 1)[0]
    return roi


def parse_tile_index_from_path(tile_path: Path) -> int:
    """Extract the tile index from a registered tile filename."""

    stem = tile_path.stem
    try:
        return int(stem.split("-")[1])
    except Exception:
        return int(stem)


def tile_origin(workspace: Workspace | Path, roi: str, tile_index: int) -> tuple[float, float]:
    """Return the (x0, y0) origin for a tile index within a workspace ROI."""

    ws = workspace if isinstance(workspace, Workspace) else Workspace(workspace)
    tc = ws.tileconfig(str(roi))

    import polars as pl

    df = tc.df
    if not isinstance(df, pl.DataFrame):  # pragma: no cover - defensive
        raise TypeError("TileConfiguration.df must be a Polars DataFrame")

    row = df.filter(pl.col("index") == int(tile_index))
    if row.height == 0:
        raise ValueError(f"Tile index {tile_index} not found for ROI '{roi}'")

    x0 = float(row[0, "x"])  # type: ignore[index]
    y0 = float(row[0, "y"])  # type: ignore[index]
    return x0, y0


# Legacy CYX two-store application removed; use apply_field_tcyx_store_to_img instead.


def apply_field_tcyx_store_to_img(
    img: np.ndarray,
    channel_labels_selected: Sequence[str],
    *,
    tcyx_zarr: Path,
    x0: float,
    y0: float,
    trim: int,
) -> np.ndarray:
    """Apply field correction using a single TCYX field store.

    Selects T-planes 'low' and 'range' and applies the same correction rule as
    ``apply_field_stores_to_img``. Returns float32 array aligned with ``img``.
    """
    low_cyx, low_meta = load_field_t_plane_as_cyx(tcyx_zarr, "low")
    rng_cyx, rng_meta = load_field_t_plane_as_cyx(tcyx_zarr, "range")

    def _patch(channel: str) -> tuple[np.ndarray, np.ndarray]:
        low_patch = slice_field_patch(
            low_cyx,
            low_meta,
            channel,
            tile_x0=x0,
            tile_y0=y0,
            out_h=int(img.shape[-2]),
            out_w=int(img.shape[-1]),
            trim=int(trim),
        )
        range_patch = slice_field_patch(
            rng_cyx,
            rng_meta,
            channel,
            tile_x0=x0,
            tile_y0=y0,
            out_h=int(img.shape[-2]),
            out_w=int(img.shape[-1]),
            trim=int(trim),
        )
        return low_patch, range_patch

    if img.ndim == 3:
        channels, _, _ = img.shape
        out = img.astype(np.float32, copy=False)
        for c_index, channel_label in enumerate(channel_labels_selected):
            if c_index >= channels:
                break
            low_patch, range_patch = _patch(channel_label)
            out[c_index] = np.maximum(out[c_index] - low_patch, 0.0) * range_patch
        return out

    if img.ndim == 4:
        z, channels, _, _ = img.shape
        out = img.astype(np.float32, copy=False)
        cached_patches: list[tuple[np.ndarray, np.ndarray]] = []
        for c_index, channel_label in enumerate(channel_labels_selected):
            if c_index >= channels:
                break
            cached_patches.append(_patch(channel_label))
        for c_index, (low_patch, range_patch) in enumerate(cached_patches):
            for z_index in range(z):
                out[z_index, c_index] = np.maximum(out[z_index, c_index] - low_patch, 0.0) * range_patch
        return out

    raise ValueError(f"Unsupported image shape for field application: {img.shape}")
