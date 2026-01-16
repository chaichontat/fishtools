from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Literal, overload

import numpy as np
import zarr

from fishtools.io.workspace import Workspace

if TYPE_CHECKING:
    import anndata


def _infer_workspace_from_adata(adata: "anndata.AnnData") -> Workspace | None:
    meta = adata.uns.get("fishtools", {}).get("segment_export")
    if not isinstance(meta, dict):
        return None
    workspace_path = meta.get("workspace_path")
    if not isinstance(workspace_path, str) or not workspace_path:
        return None
    return Workspace(Path(workspace_path))


def _resolve_fused_store(*, ws: Workspace, roi: str, codebook: str, store_name: str) -> Path:
    if not codebook:
        raise ValueError("codebook must be provided to resolve fused.zarr")

    search_labels = [codebook]
    sanitized = ws.sanitize_codebook_name(codebook)
    if sanitized not in search_labels:
        search_labels.append(sanitized)

    searched: list[Path] = []
    for label in search_labels:
        candidate = ws.stitch(roi, label) / store_name
        searched.append(candidate)
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Could not find {store_name} for roi={roi!r}, codebook={codebook!r}. "
        + "Searched: "
        + ", ".join(str(p) for p in searched)
    )


def _round_index(value: float, *, mode: Literal["round", "floor", "ceil"]) -> int:
    if mode == "round":
        return int(np.rint(value))
    if mode == "floor":
        return int(np.floor(value))
    if mode == "ceil":
        return int(np.ceil(value))
    raise ValueError(f"Unsupported rounding mode: {mode!r}")


def _crop_centered(
    arr_yxc: np.ndarray,
    *,
    x_center: int,
    y_center: int,
    size: int,
) -> np.ndarray:
    if size <= 0:
        raise ValueError("size must be positive")
    if arr_yxc.ndim != 3:
        raise ValueError(f"Expected YXC array, got shape={arr_yxc.shape}")

    y_dim, x_dim, c_dim = arr_yxc.shape
    half = size // 2
    y0 = y_center - half
    y1 = y0 + size
    x0 = x_center - half
    x1 = x0 + size

    out = np.zeros((size, size, c_dim), dtype=arr_yxc.dtype)

    src_y0 = max(0, y0)
    src_y1 = min(y_dim, y1)
    src_x0 = max(0, x0)
    src_x1 = min(x_dim, x1)
    if src_y0 >= src_y1 or src_x0 >= src_x1:
        return out

    dst_y0 = src_y0 - y0
    dst_y1 = dst_y0 + (src_y1 - src_y0)
    dst_x0 = src_x0 - x0
    dst_x1 = dst_x0 + (src_x1 - src_x0)

    out[dst_y0:dst_y1, dst_x0:dst_x1, :] = arr_yxc[src_y0:src_y1, src_x0:src_x1, :]
    return out


@overload
def cell_thumbnail_from_fused(
    fused: zarr.Array,
    *,
    z_index: int,
    x_center: int,
    y_center: int,
    size: int = 50,
    channels: None = None,
) -> np.ndarray: ...


@overload
def cell_thumbnail_from_fused(
    fused: zarr.Array,
    *,
    z_index: int,
    x_center: int,
    y_center: int,
    size: int = 50,
    channels: Sequence[int],
) -> np.ndarray: ...


def cell_thumbnail_from_fused(
    fused: zarr.Array,
    *,
    z_index: int,
    x_center: int,
    y_center: int,
    size: int = 50,
    channels: Sequence[int] | None = None,
) -> np.ndarray:
    """Extract a centered thumbnail from a stitched fused.zarr (ZYXC).

    Returns a NumPy array with shape (size, size, C_selected).
    """

    if fused.ndim != 4:
        raise ValueError(f"Expected fused.zarr to be 4D (ZYXC), got shape={fused.shape}")

    z_dim, y_dim, x_dim, c_dim = fused.shape
    if z_index < 0 or z_index >= z_dim:
        raise IndexError(f"z_index={z_index} out of range for Z dimension size={z_dim}")
    if c_dim <= 0:
        raise ValueError("fused.zarr has zero channels")

    if channels is None:
        channel_idx = np.arange(c_dim, dtype=np.int64)
    else:
        channel_idx = np.asarray(list(channels), dtype=np.int64)
        if channel_idx.size == 0:
            raise ValueError("channels must contain at least one channel index")
        if np.any(channel_idx < 0) or np.any(channel_idx >= c_dim):
            raise IndexError(f"channels={channel_idx.tolist()} out of range for C dimension size={c_dim}")

    y0 = max(0, y_center - (size // 2))
    y1 = min(y_dim, y0 + size)
    x0 = max(0, x_center - (size // 2))
    x1 = min(x_dim, x0 + size)

    # Read a small crop first (still YXC); pad/center to exact size afterwards.
    crop = np.asarray(fused[z_index, y0:y1, x0:x1, :])
    crop = _crop_centered(crop, x_center=x_center - x0, y_center=y_center - y0, size=size)
    return crop[:, :, channel_idx]


def cell_thumbnail_from_h5ad(
    h5ad_path: Path,
    *,
    cell: str | int,
    codebook: str,
    workspace: Path | Workspace | None = None,
    store_name: str = "fused.zarr",
    size: int = 50,
    channels: Sequence[int] | None = None,
    z_rounding: Literal["round", "floor", "ceil"] = "round",
) -> np.ndarray:
    """Load an exported `.h5ad` and return a (size,size,C) thumbnail for one cell."""

    import anndata as ad

    adata = ad.read_h5ad(h5ad_path)
    return cell_thumbnail_from_adata(
        adata,
        cell=cell,
        codebook=codebook,
        workspace=workspace,
        store_name=store_name,
        size=size,
        channels=channels,
        z_rounding=z_rounding,
    )


def cell_thumbnail_from_adata(
    adata: "anndata.AnnData",
    *,
    cell: str | int,
    codebook: str,
    workspace: Path | Workspace | None = None,
    store_name: str = "fused.zarr",
    size: int = 50,
    channels: Sequence[int] | None = None,
    z_rounding: Literal["round", "floor", "ceil"] = "round",
) -> np.ndarray:
    """Return a (size,size,C) thumbnail from fused.zarr centered at the cell centroid.

    The centroid is read from `adata.obs[['x','y','z']]` and the ROI from `adata.obs['roi']`.
    """

    ws: Workspace | None
    if workspace is None:
        ws = _infer_workspace_from_adata(adata)
    elif isinstance(workspace, Workspace):
        ws = workspace
    else:
        ws = Workspace(workspace)
    if ws is None:
        raise ValueError(
            "workspace must be provided (or embedded in adata.uns['fishtools']['segment_export']['workspace_path'])."
        )

    if isinstance(cell, int):
        if cell < 0 or cell >= adata.n_obs:
            raise IndexError(f"cell index {cell} out of range for adata.n_obs={adata.n_obs}")
        cell_name = str(adata.obs_names[cell])
    else:
        cell_name = str(cell)

    if cell_name not in adata.obs_names:
        raise KeyError(f"Cell {cell_name!r} not found in adata.obs_names")

    required = {"roi", "x", "y", "z"}
    missing = sorted(required - set(adata.obs.columns))
    if missing:
        raise ValueError(f"adata.obs is missing required columns: {missing}")

    row = adata.obs.loc[cell_name]
    roi = str(row["roi"])
    if not roi:
        raise ValueError(f"Cell {cell_name!r} has empty roi; cannot resolve fused.zarr")

    x = float(row["x"])
    y = float(row["y"])
    z = float(row["z"])

    z_index = _round_index(z, mode=z_rounding)
    x_center = _round_index(x, mode="round")
    y_center = _round_index(y, mode="round")

    fused_path = _resolve_fused_store(ws=ws, roi=roi, codebook=codebook, store_name=store_name)
    fused = zarr.open_array(fused_path, mode="r")
    return cell_thumbnail_from_fused(
        fused,
        z_index=z_index,
        x_center=x_center,
        y_center=y_center,
        size=size,
        channels=channels,
    )
