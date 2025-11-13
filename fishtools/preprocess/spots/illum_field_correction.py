from __future__ import annotations

import time

# TCYX Illumination Field Correction Helpers.
#
# Overview
# --------
# Small, testable helpers to apply a precomputed illumination correction field
# to tile images prior to any downstream processing (e.g., spot calling or
# percentile calculations).
#
# Inputs are TCYX Zarr stores exported by:
#     preprocess correct-illum export-field <model.npz> --what both --downsample 1
#
# The store contains two planes over the T axis (t_labels = ["low", "range"]):
#   - low:   additive floor per pixel
#   - range: multiplicative normalization per pixel (identity=1.0)
#
# For a given registered tile at global origin (tile_x0, tile_y0), we:
#   1) Slice a downsampled (ds) patch from the TCYX store aligned by
#      model_meta.x0/y0 and the tile origin; pad outside with identities
#      low=0.0 (additive identity) and range=1.0 (multiplicative identity).
#   2) Upsample the (low, range) ds patch to the tile size on GPU (CuPy)
#      and apply the correction in one block:
#          corrected = max(plane - low, 0) * range
#      Then we transfer the corrected plane back to CPU and free GPU memory.
#
# These helpers do not perform file IO or workspace lookups; the calling site
# is responsible for loading the TCYX array and providing coordinates.
#
# Coordinate model and contracts
# ------------------------------
# - All (x, y) coordinates are in the native registered mosaic pixel frame.
# - model_meta.x0/y0 in the TCYX store denote the top-left of the cropped
#   global field canvas at native resolution.
# - ds is the integer downsample factor used when exporting the field store.
# - Padding outside the union-of-tiles area uses identities: low=0.0, range=1.0.
#
# Performance notes
# -----------------
# - To minimize transfers, the GPU path upsamples and corrects in the same block
#   and returns only the corrected plane as float32.
# - Callers may slice per use; this module keeps no long-lived caches to
#   minimize peak memory when processing large tiles.
from dataclasses import dataclass
from typing import Optional  # noqa: F401

import numpy as np
import numpy.typing as npt
from loguru import logger
from numpy.typing import NDArray


@dataclass
class FieldContext:
    """Typed context capturing field geometry for a tile.

    Attributes
    ----------
    enabled:
        Gate to enable/disable correction without altering call sites.
    field_arr:
        TCYX array (float32) holding [T=low, T=range], channels, Y, X.
    t_low, t_range:
        Indices into the T axis for 'low' and 'range' planes.
    ds:
        Integer downsample factor used by the field store.
    fx0, fy0:
        Field-native top-left crop origin at native (non-downsampled) pixels.
    tile_x0, tile_y0:
        Registered tile origin (native pixels) before any local slicing.
    x_off, y_off:
        Additional offsets introduced by caller-side slicing (e.g., quadrant).
    tile_h, tile_w:
        Output Y, X sizes of the corrected tile plane.
    c_index_map:
        Maps caller channel indices to field channel indices.
    use_gpu:
        Whether to prefer the GPU path (CuPy available and enabled).
    """

    enabled: bool
    field_arr: npt.ArrayLike  # TCYX
    t_low: int
    t_range: int
    ds: int
    fx0: float
    fy0: float
    tile_x0: float
    tile_y0: float
    x_off: int
    y_off: int
    tile_h: int
    tile_w: int
    c_index_map: dict[int, int]
    use_gpu: bool
    low_scale: float = 1.0


def slice_field_ds_for_tile(
    arr_tcyx: NDArray[np.float32],
    t_low: int,
    t_range: int,
    ds: int,
    tile_w: int,
    tile_h: int,
    x0_native: float,
    y0_native: float,
    fx0: float,
    fy0: float,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Slice downsampled field planes (no upsampling) for a tile.

    Returns (low_ds, range_ds) as float32 arrays of shape (C, hd, wd) where
    hd=ceil(tile_h/ds), wd=ceil(tile_w/ds). Regions outside the source field
    are padded with identities: low=0.0, range=1.0.
    """

    ds_int = int(ds)
    if ds_int <= 0:
        raise ValueError("ds must be a positive integer")

    # Desired downsampled patch shape for this tile
    wd = int(np.ceil(float(tile_w) / float(ds_int)))
    hd = int(np.ceil(float(tile_h) / float(ds_int)))

    # Top-left of this tile in downsampled coordinates
    x0d = int(np.floor((float(x0_native) - float(fx0)) / float(ds_int)))
    y0d = int(np.floor((float(y0_native) - float(fy0)) / float(ds_int)))

    # Bounds of the source field planes
    Hf = int(arr_tcyx.shape[-2])
    Wf = int(arr_tcyx.shape[-1])

    # Intersection in source
    xs_src = max(0, x0d)
    ys_src = max(0, y0d)
    xe_src = min(Wf, x0d + wd)
    ye_src = min(Hf, y0d + hd)

    # Offsets inside the destination (hd, wd) window
    xs_dst = int(xs_src - x0d)
    ys_dst = int(ys_src - y0d)
    w_src = max(0, int(xe_src - xs_src))
    h_src = max(0, int(ye_src - ys_src))

    C = int(arr_tcyx.shape[1])
    # Identity padding: low=0 (additive), range=1 (multiplicative)
    low_ds = np.zeros((C, hd, wd), dtype=np.float32)
    rng_ds = np.ones((C, hd, wd), dtype=np.float32)

    if w_src > 0 and h_src > 0:
        low_ds[:, ys_dst : ys_dst + h_src, xs_dst : xs_dst + w_src] = arr_tcyx[
            t_low, :, ys_src:ye_src, xs_src:xe_src
        ]
        rng_ds[:, ys_dst : ys_dst + h_src, xs_dst : xs_dst + w_src] = arr_tcyx[
            t_range, :, ys_src:ye_src, xs_src:xe_src
        ]

    return low_ds, rng_ds


def rescale_slice(s: slice, factor: float) -> slice:
    """
    Scale a half-open slice by a factor.
    For downsampling by ds use factor=1/ds.

    start' = floor(start * factor); stop' = ceil(stop * factor).
    step' is scaled only when divisible; None is preserved.
    """

    def _scale_start(v: int | None) -> int | None:
        if v is None:
            return None
        return int(np.floor(v * factor))

    def _scale_stop(v: int | None) -> int | None:
        if v is None:
            return None
        return int(np.ceil(v * factor))

    def _scale_step(v: int | None) -> int | None:
        if v is None:
            return None
        scaled = v * factor
        if abs(scaled - round(scaled)) > 1e-9:
            raise ValueError(f"slice step {v} is not divisible by 1/factor={1 / factor}")
        return int(round(scaled))

    start = _scale_start(s.start)
    stop = _scale_stop(s.stop)
    step = _scale_step(s.step)

    if step == 0:  # pragma: no cover - defensive
        raise ValueError("Scaled step cannot be 0")

    return slice(start, stop, step)


def _resize2d_cpu(arr: NDArray[np.float32], out_h: int, out_w: int, order: int) -> NDArray[np.float32]:
    from skimage.transform import resize as _resize

    return _resize(
        arr,
        (int(out_h), int(out_w)),
        order=order,
        preserve_range=True,
        anti_aliasing=False,
    ).astype(np.float32, copy=False)


def _resize2d_gpu(arr_dev, out_h: int, out_w: int, order: int):
    from cucim.skimage.transform import resize as _resize  # type: ignore

    return _resize(
        arr_dev,
        (int(out_h), int(out_w)),
        order=order,
        preserve_range=True,
        anti_aliasing=False,
    )


def correct_channel_with_field(
    stack: NDArray[np.float32],
    low_ds: NDArray[np.float32],
    rng_ds: NDArray[np.float32],
    ds: int,
    *,
    use_gpu: bool = True,
    normalize: bool = True,
    sl: slice = np.s_[:, :, :, :],
) -> NDArray[np.float32]:
    """Upsample fields and apply correction on an entire Z stack (Z,Y,X).

    Optimization for large downsampled stores: when ``ds > 4`` and divisible by 4,
    perform a two‑stage upsample — quadratic resize to 4× the store resolution,
    then stripe‑copy (nearest repeat) by ``ds/4`` to native. This caps the costly
    interpolation at 4× while preserving alignment, and is significantly faster
    for very coarse TCYX stores.
    """
    if stack.ndim != 4:
        raise ValueError("correct_plane_with_field expects a 4D array shaped (Z, C, Y, X)")
    if low_ds.ndim != 3 or rng_ds.ndim != 3:
        raise ValueError("Field planes must be 3D arrays shaped (C, Y, X)")
    if ds <= 0:
        raise ValueError("Downsample factor ds must be a positive integer")

    t_start = time.perf_counter()
    strategy = "two-stage-4x-repeat" if (ds > 4 and ds % 4 == 0) else "single-resize"
    Z, C, H, W = map(int, stack.shape)
    logger.info(
        f"FieldApply: backend={'GPU' if use_gpu else 'CPU'}, ds={int(ds)}, strategy={strategy}, Z={Z}, C={C}, out={H}x{W}"
    )

    if use_gpu:
        import cupy as cp  # type: ignore
        # resize to exact (H, W) to avoid rounding drift

        try:
            low_dev = cp.asarray(low_ds, dtype=cp.float32)
            rng_dev = cp.asarray(rng_ds, dtype=cp.float32)
            out_h = int(stack.shape[-2])
            out_w = int(stack.shape[-1])
            low_up = cp.empty((low_ds.shape[0], out_h, out_w), dtype=cp.float32)
            rng_up = cp.empty((rng_ds.shape[0], out_h, out_w), dtype=cp.float32)

            # Crop ds planes according to the provided slice, then resize to (H, W)
            sl_scale = [rescale_slice(sl[-2], 1 / ds), rescale_slice(sl[-1], 1 / ds)]
            for i in range(low_ds.shape[0]):
                src = low_dev[i, *sl_scale]
                src_r = rng_dev[i, *sl_scale]
                ch_t0 = time.perf_counter()
                if ds > 4 and ds % 4 == 0:
                    # Two‑stage: 4× quadratic zoom, then stripe‑copy by (ds/4)
                    inter_h = min(out_h, int(src.shape[-2]) * 4)
                    inter_w = min(out_w, int(src.shape[-1]) * 4)
                    tmp_low = _resize2d_gpu(src, inter_h, inter_w, order=2)
                    tmp_rng = _resize2d_gpu(src_r, inter_h, inter_w, order=2)
                    ry = int(ds // 4)
                    rx = int(ds // 4)
                    rep_low = cp.repeat(cp.repeat(tmp_low, ry, axis=0), rx, axis=1)
                    rep_rng = cp.repeat(cp.repeat(tmp_rng, ry, axis=0), rx, axis=1)
                    low_up[i] = rep_low[:out_h, :out_w]
                    rng_up[i] = rep_rng[:out_h, :out_w]
                    logger.debug(
                        f"FieldApply[GPU]: ch={i} stage1=resize4x {int(inter_h)}x{int(inter_w)}; stage2=repeat ry=rx={ry}; out={out_h}x{out_w}; dt={(time.perf_counter() - ch_t0):.3f}s"
                    )
                else:
                    low_up[i] = _resize2d_gpu(src, out_h, out_w, order=2)
                    rng_up[i] = _resize2d_gpu(src_r, out_h, out_w, order=2)
                    logger.debug(
                        f"FieldApply[GPU]: ch={i} single-resize to {out_h}x{out_w}; dt={(time.perf_counter() - ch_t0):.3f}s"
                    )
            del low_dev, rng_dev

            img_dev = cp.asarray(stack, dtype=cp.float32)
            # Avoid double-slicing: stack already reflects `sl`; fields match (H, W)
            t_apply = time.perf_counter()
            out = cp.maximum(img_dev - low_up[None, ...], 0.0) * rng_up[None, ...]
            if normalize:
                if np.issubdtype(stack.dtype, np.integer):
                    max_val = float(np.iinfo(stack.dtype).max)
                else:
                    max_val = 1.0
                out = cp.clip(out / max_val, 0.0, 1.0)
            logger.debug(f"FieldApply[GPU]: combine+normalize dt={(time.perf_counter() - t_apply):.3f}s")
            out_np = out.get()
            del out, img_dev, low_up, rng_up
            logger.info(f"FieldApply[GPU]: total dt={(time.perf_counter() - t_start):.3f}s")
            return out_np
        finally:
            ...
            # cp.get_default_memory_pool().free_all_blocks()  # type: ignore[attr-defined]

    # CPU path: mirror GPU semantics using resize to exact (H, W)

    out_h = int(stack.shape[-2])
    out_w = int(stack.shape[-1])
    low_up = np.empty((int(low_ds.shape[0]), out_h, out_w), dtype=np.float32)
    rng_up = np.empty((int(rng_ds.shape[0]), out_h, out_w), dtype=np.float32)

    # Mirror GPU slice semantics: scale the Y/X slice used on ds arrays
    sl_scale = [rescale_slice(sl[-2], 1 / ds), rescale_slice(sl[-1], 1 / ds)]
    for i in range(int(low_ds.shape[0])):
        src = low_ds[i, *sl_scale]
        src_r = rng_ds[i, *sl_scale]
        ch_t0 = time.perf_counter()
        if ds > 4 and ds % 4 == 0:
            inter_h = min(out_h, int(src.shape[-2]) * 4)
            inter_w = min(out_w, int(src.shape[-1]) * 4)
            tmp_low = _resize2d_cpu(src, inter_h, inter_w, order=2)
            tmp_rng = _resize2d_cpu(src_r, inter_h, inter_w, order=2)
            ry = int(ds // 4)
            rx = int(ds // 4)
            rep_low = np.repeat(np.repeat(tmp_low, ry, axis=0), rx, axis=1)
            rep_rng = np.repeat(np.repeat(tmp_rng, ry, axis=0), rx, axis=1)
            low_up[i] = rep_low[:out_h, :out_w]
            rng_up[i] = rep_rng[:out_h, :out_w]
            logger.debug(
                f"FieldApply[CPU]: ch={i} stage1=resize4x {int(inter_h)}x{int(inter_w)}; stage2=repeat ry=rx={ry}; out={out_h}x{out_w}; dt={(time.perf_counter() - ch_t0):.3f}s"
            )
        else:
            low_up[i] = _resize2d_cpu(src, out_h, out_w, order=2)
            rng_up[i] = _resize2d_cpu(src_r, out_h, out_w, order=2)
            logger.debug(
                f"FieldApply[CPU]: ch={i} single-resize to {out_h}x{out_w}; dt={(time.perf_counter() - ch_t0):.3f}s"
            )

    # Avoid double-slicing: `stack` already carries the caller's `sl`
    out = np.maximum(stack.astype(np.float32) - low_up[None, ...], 0.0) * rng_up[None, ...]
    if normalize:
        if np.issubdtype(stack.dtype, np.integer):
            max_val = float(np.iinfo(stack.dtype).max)
        else:
            max_val = 1.0
        np.clip(out / max_val, 0.0, 1.0, out=out)
    logger.info(f"FieldApply[CPU]: total dt={(time.perf_counter() - t_start):.3f}s")
    return out


__all__ = [
    "FieldContext",
    "slice_field_ds_for_tile",
    "correct_channel_with_field",
    "remap_field_channels",
]


def remap_field_channels(
    low_ds: NDArray[np.float32],
    rng_ds: NDArray[np.float32],
    sel_positions: list[int],
    c_index_map: dict[int, int],
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Reorder field planes to match a selected image channel order.

    Parameters
    ----------
    low_ds, rng_ds:
        Downsampled field planes shaped (C, Hd, Wd).
    sel_positions:
        Image channel positions selected in the stack (relative to TIFF order).
    c_index_map:
        Mapping from image channel position → field channel index built from
        attrs['channel_names'] vs TIFF metadata['key'].

    Returns
    -------
    (low_remap, rng_remap):
        Field planes reordered so channel i aligns with image channel i.
    """
    if low_ds.ndim != 3 or rng_ds.ndim != 3:
        raise ValueError("low_ds and rng_ds must be (C, Hd, Wd)")
    if low_ds.shape != rng_ds.shape:
        raise ValueError("low_ds and rng_ds must have the same shape")

    sel_positions = [int(p) for p in sel_positions]
    mapped = [int(c_index_map.get(p, p)) for p in sel_positions]
    C = int(low_ds.shape[0])
    for mi in mapped:
        if mi < 0 or mi >= C:
            raise ValueError(f"Mapped field channel index out of range: {mi} not in [0,{C})")
    remap_idx = np.asarray(mapped, dtype=np.intp)
    return low_ds[remap_idx], rng_ds[remap_idx]
