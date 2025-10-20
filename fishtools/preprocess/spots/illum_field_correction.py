from __future__ import annotations

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
#      model_meta.x0/y0 and the tile origin; pad outside with 1.0.
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
# - Padding outside the union-of-tiles area uses identity=1.0 for both planes.
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

    Returns (low_ds, range_ds) as float32 arrays sized roughly tile/ ds with padding=1.0.
    """

    x0d = int(np.floor((float(x0_native) - float(fx0)) / float(max(1, int(ds)))))
    y0d = int(np.floor((float(y0_native) - float(fy0)) / float(max(1, int(ds)))))
    wd = int(np.ceil(float(tile_w) / float(max(1, int(ds)))))
    hd = int(np.ceil(float(tile_h) / float(max(1, int(ds)))))

    Hf = int(arr_tcyx.shape[-2])
    Wf = int(arr_tcyx.shape[-1])
    xs = max(0, x0d)
    ys = max(0, y0d)
    xe = min(Wf, x0d + wd)
    ye = min(Hf, y0d + hd)

    low_ds = arr_tcyx[t_low, :, ys:ye, xs:xe]
    rng_ds = arr_tcyx[t_range, :, ys:ye, xs:xe]

    return low_ds, rng_ds


def rescale_slice(s, factor):
    """
    Rescale a slice by multiplying by a factor and rounding to nearest integer.
    Raises ValueError if results aren't close to integers.
    """

    def scale(value):
        if value is None:
            return None
        scaled = value * factor
        rounded = round(scaled)
        if abs(scaled - rounded) > 1e-9:
            raise ValueError(f"{value} * {factor} = {scaled} is not close to integer")
        return rounded

    start = scale(s.start)
    stop = scale(s.stop)
    step = scale(s.step)

    if step == 0:
        raise ValueError("Scaled step cannot be 0")

    return slice(start, stop, step)


def _resize_cpu(
    low_ds: NDArray[np.float32], rng_ds: NDArray[np.float32], tile_h: int, tile_w: int
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    from skimage.transform import resize as _resize

    low_up = _resize(
        low_ds,
        (int(tile_h), int(tile_w)),
        order=1,
        preserve_range=True,
        anti_aliasing=False,
    ).astype(
        np.float32,
        copy=False,
    )
    rng_up = _resize(
        rng_ds,
        (int(tile_h), int(tile_w)),
        order=1,
        preserve_range=True,
        anti_aliasing=False,
    ).astype(
        np.float32,
        copy=False,
    )
    return low_up, rng_up


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
    """Upsample fields and apply correction on an entire Z stack (Z,Y,X)."""
    if stack.ndim != 4:
        raise ValueError("correct_plane_with_field expects a 4D array shaped (Z, C, Y, X)")
    if low_ds.ndim != 3 or rng_ds.ndim != 3:
        raise ValueError("Field planes must be 3D arrays shaped (C, Y, X)")
    if ds <= 0:
        raise ValueError("Downsample factor ds must be a positive integer")

    scale = float(ds)

    if use_gpu:
        import cupy as cp  # type: ignore
        from cucim.skimage.transform import rescale as _cucim_rescale  # type: ignore

        try:
            low_dev = cp.asarray(low_ds, dtype=cp.float32)
            rng_dev = cp.asarray(rng_ds, dtype=cp.float32)
            low_up = cp.empty(
                (low_ds.shape[0], int(stack.shape[-2]), int(stack.shape[-1])),
                dtype=cp.float32,
            )
            rng_up = cp.empty(
                (rng_ds.shape[0], int(stack.shape[-2]), int(stack.shape[-1])),
                dtype=cp.float32,
            )
            sl_scale = [rescale_slice(sl[-2], 1 / ds), rescale_slice(sl[-1], 1 / ds)]

            for i in range(low_ds.shape[0]):
                low_up[i] = _cucim_rescale(
                    low_dev[i, *sl_scale],
                    scale=scale,
                    order=2,
                    preserve_range=True,
                    anti_aliasing=False,
                )
                rng_up[i] = _cucim_rescale(
                    rng_dev[i, *sl_scale],
                    scale=scale,
                    order=2,
                    preserve_range=True,
                    anti_aliasing=False,
                )
            del low_dev, rng_dev

            img_dev = cp.asarray(stack, dtype=cp.float32)
            out = cp.maximum(img_dev - low_up[None, :, sl[-2], sl[-1]], 0.0) * rng_up[None, :, sl[-2], sl[-1]]
            if normalize:
                out *= 1.0 / (65535.0 * 2)
                out = cp.clip(out, 0.0, 1.0)
            out_np = out.get()
            del out, img_dev, low_up, rng_up
            return out_np
        finally:
            ...
            # cp.get_default_memory_pool().free_all_blocks()  # type: ignore[attr-defined]

    # CPU path: mirror GPU semantics using skimage.transform.rescale
    from skimage.transform import rescale as _sk_rescale  # type: ignore

    out_h = int(stack.shape[-2])
    out_w = int(stack.shape[-1])
    low_up = np.empty((int(low_ds.shape[0]), out_h, out_w), dtype=np.float32)
    rng_up = np.empty((int(rng_ds.shape[0]), out_h, out_w), dtype=np.float32)

    # Mirror GPU slice semantics: scale the Y/X slice used on ds arrays
    sl_scale = [rescale_slice(sl[-2], 1 / ds), rescale_slice(sl[-1], 1 / ds)]
    for i in range(int(low_ds.shape[0])):
        low_up[i] = _sk_rescale(
            low_ds[i, *sl_scale],
            scale=scale,
            order=2,
            preserve_range=True,
            anti_aliasing=False,
        ).astype(np.float32, copy=False)
        rng_up[i] = _sk_rescale(
            rng_ds[i, *sl_scale],
            scale=scale,
            order=2,
            preserve_range=True,
            anti_aliasing=False,
        ).astype(np.float32, copy=False)

    out = (
        np.maximum(stack.astype(np.float32) - low_up[None, :, sl[-2], sl[-1]], 0.0)
        * rng_up[None, :, sl[-2], sl[-1]]
    )
    if normalize:
        out *= 1.0 / (65535.0 * 2)
        np.clip(out, 0.0, 1.0, out=out)
    return out


__all__ = [
    "FieldContext",
    "slice_field_ds_for_tile",
    "correct_channel_with_field",
]
