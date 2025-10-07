from __future__ import annotations

import time
from pathlib import Path
from typing import Sequence

import numpy as np
import polars as pl
from loguru import logger
from skimage.transform import resize as _resize

from fishtools.io.workspace import Workspace
from fishtools.preprocess.imageops import apply_low_range_zcyx
from fishtools.preprocess.spots.illumination import RangeFieldPointsModel, _slice_field_to_tile

_FIELD_CACHE: dict[
    tuple[float, float, float, float, float, float, float, float, str, float, float],
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
] = {}


def apply_field_corr_to_tile_zcyx(
    img: np.ndarray,
    *,
    model: Path | RangeFieldPointsModel,
    workspace: Path | Workspace,
    roi: str,
    tile_index: int,
    downsample: int = 1,
    patch_downsample: int = 2,
    return_field: bool = False,
    channels_to_apply: Sequence[int] | None = None,
    neighbors_override: int | None = None,
    smoothing_override: float | None = None,
    # New parameters to support application after downsample/crop
    pretrim: int = 0,
    coords_in_downsampled_space: bool = False,
) -> np.ndarray:
    """Apply LOW/RANGE field correction to a sanitized Z×C×Y×X tile.

    This computes per-tile LOW and RANGE patches at the current spatial resolution
    and applies: max(img − LOW, 0) × clip(RANGE, 1/3, 3). Returns float32.

    Args:
        img: Array with shape (Z, C, Y, X).
        model: Illumination field NPZ path or an already loaded RangeFieldPointsModel.
        workspace: Workspace or path to the workspace root.
        roi: ROI name for which to look up tile positions.
        tile_index: Tile index as in TileConfiguration.registered.txt.

    Returns:
        Float32 array of the same shape as ``img``.

    Notes on alignment when applying after downsample/crop
    -----------------------------------------------
    - Tile origins (x0,y0) from TileConfiguration are in full-resolution
      mosaic pixels at the tile's top-left corner.
    - When correction is applied AFTER trimming/downsampling, the field
      slice bounds must be computed in the same full-resolution frame:
        x0_use = x0 + pretrim
        y0_use = y0 + pretrim
        x1     = x0_use + (x_out * downsample)
        y1     = y0_use + (y_out * downsample)
      where (y_out,x_out) are the post-downsample spatial dimensions of
      ``img``. This preserves alignment with the global field grid.
    """
    if img.ndim != 4:
        raise ValueError(f"Expected Z×C×Y×X input, got shape {img.shape!r}")

    ws = workspace if isinstance(workspace, Workspace) else Workspace(workspace)
    tc = ws.tileconfig(roi)

    # Resolve tile origin in global coordinates
    df = tc.df
    if not isinstance(df, pl.DataFrame):  # defensive: project uses Polars
        raise TypeError("TileConfiguration.df must be a Polars DataFrame")
    row = df.filter(pl.col("index") == int(tile_index))
    if row.height == 0:
        raise ValueError(f"Tile index {tile_index} not found for ROI '{roi}'")
    x0 = float(row[0, "x"])  # type: ignore[index]
    y0 = float(row[0, "y"])  # type: ignore[index]

    # Load model if a path was provided
    t0 = time.perf_counter()
    rng_model = model if isinstance(model, RangeFieldPointsModel) else RangeFieldPointsModel.from_npz(model)
    logger.info(
        f"FieldCorr: loaded model for ROI={roi} tile={tile_index} in {time.perf_counter() - t0:.3f}s; img={img.shape}"
    )

    # Build field patches sized to current tile resolution
    # Note: when applying AFTER downsample/crop, (y, x) are reduced dimensions.
    _, _, y, x = img.shape
    # Origins are used implicitly via the global bbox below

    # Compute global field once (cache) at coarser step; slice per tile to output size
    t1 = time.perf_counter()
    tile_w_meta = float(rng_model.meta.get("tile_w", float(x)))
    tile_h_meta = float(rng_model.meta.get("tile_h", float(y)))
    base_step = float(rng_model.meta.get("grid_step_suggest", 192.0))
    # Use a finer global grid to reduce anchor/tiling artifacts. Bound by a fraction
    # of the tile size so there are multiple samples within each tile.
    tile_min = max(2.0, min(tile_w_meta, tile_h_meta))
    grid_step = min(base_step * max(1, int(downsample)), tile_min / 6.0)

    xs0 = df.select("x").to_numpy().reshape(-1).astype(float)
    ys0 = df.select("y").to_numpy().reshape(-1).astype(float)
    x_min = float(xs0.min())
    y_min = float(ys0.min())
    x_max = float(xs0.max() + tile_w_meta)
    y_max = float(ys0.max() + tile_h_meta)

    kernel = str(rng_model.meta.get("kernel", "thin_plate_spline"))
    smoothing_meta = float(rng_model.meta.get("smoothing", 1.0))
    neighbors_meta = int(rng_model.meta.get("neighbors", 64))
    neighbors_eff = int(neighbors_override) if neighbors_override is not None else neighbors_meta
    smoothing_eff = float(smoothing_override) if smoothing_override is not None else float(smoothing_meta)

    # Key includes bounds, step, kernel params to ensure correctness
    cache_key = (
        x_min,
        y_min,
        x_max,
        y_max,
        float(grid_step),
        tile_w_meta,
        tile_h_meta,
        neighbors_eff,
        kernel,
        smoothing_eff,
        float(downsample),
    )

    if cache_key in _FIELD_CACHE:
        xs, ys, low_field, high_field = _FIELD_CACHE[cache_key]
        logger.info(f"FieldCorr: cache hit — global field xs={xs.size}, ys={ys.size}, step={grid_step:.1f}")
    else:
        xs, ys, low_field, high_field = rng_model.evaluate(
            x_min,
            y_min,
            x_max,
            y_max,
            grid_step=grid_step,
            neighbors=None if neighbors_eff <= 0 else int(neighbors_eff),
            kernel=kernel,
            smoothing=smoothing_eff,
        )
        _FIELD_CACHE[cache_key] = (xs, ys, low_field, high_field)
        logger.info(
            f"FieldCorr: cache miss — evaluated global field in {time.perf_counter() - t1:.3f}s (step={grid_step:.1f})"
        )

    # Build a global union-of-tiles mask (parity with plot-field) and normalize RANGE globally
    def _mask_union_of_tiles(
        xs: np.ndarray, ys: np.ndarray, xs0: np.ndarray, ys0: np.ndarray, tw: float, th: float
    ) -> np.ndarray:
        xs = np.asarray(xs, dtype=np.float64)
        ys = np.asarray(ys, dtype=np.float64)
        xs0 = np.asarray(xs0, dtype=np.float64)
        ys0 = np.asarray(ys0, dtype=np.float64)
        if xs.size == 0 or ys.size == 0 or xs0.size == 0:
            return np.ones((ys.size, xs.size), dtype=bool)
        x1 = xs0 + float(tw)
        y1 = ys0 + float(th)
        dx = float(np.min(np.diff(xs))) if xs.size > 1 else 0.0
        dy = float(np.min(np.diff(ys))) if ys.size > 1 else 0.0
        tol = max(dx, dy, 1.0) * 1e-6
        X = (xs[None, :] >= (xs0[:, None] - tol)) & (xs[None, :] <= (x1[:, None] + tol))
        Y = (ys[None, :] >= (ys0[:, None] - tol)) & (ys[None, :] <= (y1[:, None] + tol))
        counts = Y.astype(np.uint16).T @ X.astype(np.uint16)
        return counts > 0

    mask = _mask_union_of_tiles(xs, ys, xs0, ys0, tile_w_meta, tile_h_meta)
    inv_global = rng_model.range_correction(low_field, high_field, mask=mask)

    # Compute the slice bounds in the global (full-resolution) coordinate frame.
    # If correction is applied AFTER downsample/crop, adjust origin by `pretrim` and
    # scale the width/height by the downsample factor.
    if coords_in_downsampled_space:
        x0_use = float(x0) + float(pretrim)
        y0_use = float(y0) + float(pretrim)
        x1 = x0_use + float(x) * float(max(1, int(downsample)))
        y1 = y0_use + float(y) * float(max(1, int(downsample)))
    else:
        x0_use = float(x0)
        y0_use = float(y0)
        x1 = x0_use + float(x)
        y1 = y0_use + float(y)
    # Slice at reduced resolution for speed, then upsample quickly to the output shape
    pd = max(1, int(patch_downsample))
    y2 = max(2, (int(y) + pd - 1) // pd)
    x2 = max(2, (int(x) + pd - 1) // pd)
    t2 = time.perf_counter()
    low_half = _slice_field_to_tile(
        low_field, xs, ys, float(x0_use), float(y0_use), x1, y1, (int(y2), int(x2))
    )
    rng_half = _slice_field_to_tile(
        inv_global.astype(np.float32, copy=False),
        xs,
        ys,
        float(x0_use),
        float(y0_use),
        x1,
        y1,
        (int(y2), int(x2)),
    )
    t_up = time.perf_counter()
    rng_patch = _resize(
        rng_half,
        (int(y), int(x)),
        order=1,
        preserve_range=True,
        anti_aliasing=False,
    ).astype(np.float32, copy=False)
    low_patch = _resize(
        low_half,
        (int(y), int(x)),
        order=1,
        preserve_range=True,
        anti_aliasing=False,
    ).astype(np.float32, copy=False)
    logger.info(
        f"FieldCorr: sliced {y2}x{x2} (patch_ds={pd}) and upsampled to {int(y)}x{int(x)} in "
        f"{(time.perf_counter() - t2):.3f}s (slice) + {(time.perf_counter() - t_up):.3f}s (upsample)"
    )

    c = img.shape[1]
    # Build per-channel fields; enforce channel-specific application
    if not channels_to_apply:
        raise ValueError(
            "Channel-specific correction required: channels_to_apply must be a non-empty sequence of indices."
        )
    low3 = np.zeros((c, y, x), dtype=np.float32)
    rng3 = np.ones((c, y, x), dtype=np.float32)
    for ci in channels_to_apply:
        idx = int(ci)
        if 0 <= idx < c:
            low3[idx] = low_patch
            rng3[idx] = rng_patch
    t3 = time.perf_counter()
    out = apply_low_range_zcyx(img, low3, rng3)
    logger.info(
        f"FieldCorr: applied correction in {time.perf_counter() - t3:.3f}s (tile {tile_index}); out={out.shape}"
    )
    if return_field:
        return out, rng_patch
    return out


__all__ = ["apply_field_corr_to_tile_zcyx"]
