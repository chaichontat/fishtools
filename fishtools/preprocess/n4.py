"""
N4 bias-field correction for stitched mosaics.

- Input: fused.zarr (ZYXC) under analysis/deconv/stitch--{roi}+{codebook}/.
- Estimate a 2D per-channel field at --z-index (N4), normalized to mean 1 on
  foreground (>0), then apply to one plane or all Z.
- Writes: correction field TIFF (CYX, float32) plus per‑channel PNGs; corrected
  fused.zarr (ZYXC, uint16). Add --debug to also write a float32 corrected Zarr.
- Quantization: per-channel 0.01–99.99 percentiles via deterministic random
  subsampling (≤50k samples/plane) with a 99.9 fallback when total samples < 2e5;
  metadata records bounds, scale, percentiles, and sampling stats.
- Channels are scaled independently; no cross‑channel normalization.

Example:
    python -m fishtools.preprocess.n4 /workspace roi --codebook cb --z-index 0 --debug
"""

from __future__ import annotations

import math
import os
import shutil
from concurrent.futures import as_completed
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any, Callable, Iterable, Sequence

import cupy as cp
import numpy as np
import SimpleITK as sitk
import tifffile
import zarr
from cucim.skimage import filters as cucim_filters
from loguru import logger
from PIL import Image
from skimage import filters

if TYPE_CHECKING:  # pragma: no cover - typing aid
    from cupy import ndarray as cupy_ndarray  # type: ignore[import-not-found]
else:
    cupy_ndarray = Any

from fishtools import IMWRITE_KWARGS
from fishtools.io.workspace import Workspace, safe_imwrite
from fishtools.utils.logging import setup_logging
from fishtools.utils.pretty_print import progress_bar
from fishtools.utils.threading import shared_thread_pool

setup_logging()

DEFAULT_ITERATIONS: tuple[int, ...] = (50, 50, 30, 20)

QUANT_LOWER_PERCENTILE = 0.01
QUANT_UPPER_PERCENTILE = 99.99
QUANT_FALLBACK_UPPER_PERCENTILE = 99.9
QUANT_MIN_TOTAL_SAMPLES_FOR_HIGH_PERCENTILE = 200_000
QUANT_MIN_RANGE = 1e-6
# Add a small headroom above the chosen upper percentile so that
# a tiny fraction of very bright pixels do not saturate to 0xFFFF.
# This does not change the recorded percentile but widens the scale used.
QUANT_HEADROOM_FRACTION = 0.02  # 2% extra width beyond (upper - lower)

# Custom CuPy kernels removed for simplicity. We perform per‑plane division and
# optional unsharp masking on GPU, then quantize on CPU.

# --- Utilities ---------------------------------------------------------------


def _zarr_attrs_write(attrs: Any, mapping: dict[str, Any]) -> None:
    """
    Robustly write Zarr attributes across versions by trying update, item assignment,
    and put(k, v) in that order.
    """
    try:
        # Zarr v2 Attributes is MutableMapping and supports update()
        attrs.update(mapping)  # type: ignore[attr-defined]
        return
    except Exception:
        pass
    for k, v in mapping.items():
        try:
            attrs[k] = v  # type: ignore[index]
        except Exception:
            put = getattr(attrs, "put", None)
            if callable(put):
                put(k, v)
            else:
                raise


def _normalize_to_uint8_by_percentile(img: np.ndarray, p_lo: float = 1.0, p_hi: float = 99.0) -> np.ndarray:
    """Normalize to uint8 using approximate percentiles from a simple grid sample.

    We compute p_lo/p_hi percentiles on a uniform subsample to keep this fast and
    deterministic, then map the full image with those bounds.
    """
    f = np.asarray(img, dtype=np.float32)
    f = np.where(np.isfinite(f), f, 0.0)
    # Uniform grid sampling to ~50k points for large images
    total = f.size
    if total > 50_000:
        h, w = f.shape
        stride = int(math.ceil(math.sqrt(total / 50_000)))
        sample = f[::stride, ::stride].ravel()
    else:
        sample = f.ravel()
    p1, p99 = np.percentile(sample, [p_lo, p_hi])
    if not np.isfinite(p1):
        p1 = float(np.nanmin(sample)) if np.isfinite(np.nanmin(sample)) else 0.0
    if not np.isfinite(p99) or p99 <= p1:
        p99 = float(np.nanmax(sample)) if np.isfinite(np.nanmax(sample)) else (p1 + 1.0)
        if p99 <= p1:
            p99 = p1 + 1.0
    fn = np.clip((f - p1) / (p99 - p1), 0.0, 1.0)
    return (fn * 255.0 + 0.5).astype(np.uint8)


def _write_png_normalized(field: np.ndarray, png_path: Path, long_edge_max: int = 1024) -> None:
    """Subsample array first, then normalize and write a small PNG.

    We avoid interpolation: take an integer stride so the long edge ≤ long_edge_max,
    then run percentile normalization on that subsampled array and write it.
    """
    f = np.asarray(field, dtype=np.float32)
    h, w = f.shape
    long_edge = max(h, w)
    stride = max(1, int(math.ceil(long_edge / float(long_edge_max))))
    f_small = f[::stride, ::stride]
    img8 = _normalize_to_uint8_by_percentile(f_small, 1.0, 99.0)
    Image.fromarray(img8, mode="L").save(png_path)


def _ensure_float32(arr: np.ndarray) -> np.ndarray:
    if isinstance(arr, np.ndarray) and arr.dtype == np.float32:
        return arr if arr.flags.c_contiguous else np.ascontiguousarray(arr)
    return np.asarray(arr, dtype=np.float32)


def _resolve_threshold_function(name: str) -> tuple[Callable[[np.ndarray], Any], str]:
    """Return a skimage.filters threshold function and its resolved name."""

    attr: Callable[[np.ndarray], Any] | None = None
    resolved_name = name.strip()
    if resolved_name:
        attr = getattr(filters, resolved_name, None)
        if attr is None and not resolved_name.startswith("threshold_"):
            candidate = f"threshold_{resolved_name}"
            attr = getattr(filters, candidate, None)
            if attr is not None:
                resolved_name = candidate
    if attr is None or not callable(attr):
        raise ValueError(
            f"Unknown threshold method '{name}'. Available methods include: "
            f"{', '.join(sorted(fn for fn in dir(filters) if fn.startswith('threshold_')))}"
        )
    return attr, resolved_name


def _empty_mask_message(threshold: float | str | None, *, resolved_name: str | None = None) -> str:
    if threshold is None:
        return "Mask is empty after thresholding >0; check the selected channel."
    if isinstance(threshold, (int, float)):
        return f"Mask is empty after thresholding > {float(threshold)}; check the selected channel."
    name = resolved_name if resolved_name is not None else str(threshold)
    return f"Mask is empty after applying skimage.filters.{name}; check the selected channel."


def _compute_threshold_mask(
    image: np.ndarray,
    threshold: float | str | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Generate a boolean foreground mask plus metadata for the chosen threshold."""

    data = _ensure_float32(image)
    data = np.where(np.isfinite(data), data, 0.0)

    if threshold is None:
        return data > 0.0, {"kind": "default_gt_zero", "value": 0.0}

    if isinstance(threshold, (int, float)):
        value = float(threshold)
        return data > value, {"kind": "numeric", "value": value}

    func, resolved_name = _resolve_threshold_function(str(threshold).strip())
    threshold_result = func(data)
    threshold_array = np.asarray(threshold_result, dtype=np.float32)

    if threshold_array.shape not in [(), data.shape]:
        raise ValueError(
            f"Threshold function '{resolved_name}' returned shape {threshold_array.shape}, "
            f"expected scalar or {data.shape}."
        )

    if threshold_array.shape == ():
        scalar_value = float(threshold_array)
        meta: dict[str, Any] = {
            "kind": "method",
            "function": resolved_name,
            "result": {"type": "scalar", "value": scalar_value},
        }
        mask = data > scalar_value
    else:
        finite = np.isfinite(threshold_array)
        if not np.any(finite):
            raise ValueError(
                f"Threshold function '{resolved_name}' produced no finite values for the mask decision."
            )
        summary = {
            "type": "array",
            "min": float(np.nanmin(threshold_array)),
            "max": float(np.nanmax(threshold_array)),
        }
        meta = {
            "kind": "method",
            "function": resolved_name,
            "result": summary,
        }
        mask = data > threshold_array

    return mask.astype(bool, copy=False), meta


def _summarize_threshold_spec(threshold: float | str | None) -> dict[str, Any]:
    if threshold is None:
        return {"kind": "default_gt_zero", "value": 0.0}
    if isinstance(threshold, (int, float)):
        return {"kind": "numeric", "value": float(threshold)}
    _, resolved_name = _resolve_threshold_function(str(threshold).strip())
    return {"kind": "method", "function": resolved_name}


QUANT_MAX_SAMPLES = 50_000


def _float_store_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}_float32{path.suffix}")


def _prepare_gpu_field(field: np.ndarray) -> cupy_ndarray:
    """Transfer a correction field to GPU memory when CuPy is available."""
    return cp.asarray(_ensure_float32(field), dtype=cp.float32)


def _cupy_to_numpy(array: cupy_ndarray) -> np.ndarray:
    return np.asarray(cp.asnumpy(array), dtype=np.float32)


def _apply_unsharp_mask_if_enabled(image: np.ndarray, *, mask: np.ndarray, enabled: bool) -> np.ndarray:
    """Apply CuCIM unsharp mask to `image` within `mask` when enabled."""

    if not enabled:
        return _ensure_float32(image)

    mask_bool = np.asarray(mask, dtype=bool)
    if not np.any(mask_bool):
        return _ensure_float32(image)

    image_gpu = cp.asarray(image, dtype=cp.float32)
    mask_gpu = cp.asarray(mask_bool, dtype=cp.bool_)
    # Explicit radius for consistency across pipeline components
    sharpened_gpu = cucim_filters.unsharp_mask(image_gpu, radius=3, preserve_range=True)
    result_gpu = cp.where(mask_gpu, sharpened_gpu, image_gpu)
    result = _cupy_to_numpy(result_gpu)
    # Explicitly drop GPU references before forcing pool cleanup
    del result_gpu
    del sharpened_gpu
    del mask_gpu
    del image_gpu
    # Force clear VRAM after the initial unsharp mask: free CuPy pools
    # Memory pool cleanup omitted for simplicity.
    return result


def _correct_plane_gpu(
    plane: np.ndarray,
    *,
    field_gpu: cupy_ndarray,
    use_unsharp_mask: bool,
    mask_cpu: np.ndarray | None = None,
) -> np.ndarray:
    """Return corrected plane as float32 (CPU): (plane / field) with optional unsharp mask.

    The field must already be on GPU. This function copies the plane to GPU,
    performs division (+optional unsharp), sanitizes non-finite values, and returns CPU float32.
    """
    img_gpu = cp.asarray(np.asarray(plane, dtype=np.float32), dtype=cp.float32)
    img_gpu /= field_gpu
    if use_unsharp_mask:
        # Apply explicit radius to match segmentation and normalization defaults
        sharpened_gpu = cucim_filters.unsharp_mask(img_gpu, radius=3, preserve_range=True)
        if mask_cpu is not None:
            if mask_cpu.shape != plane.shape:
                raise ValueError(f"Mask shape {mask_cpu.shape} does not match plane shape {plane.shape}")
            mask_gpu = cp.asarray(mask_cpu, dtype=cp.bool_)
        else:
            # Fallback to >0 mask on corrected plane
            mask_gpu = img_gpu > 0.0
        img_gpu = cp.where(mask_gpu, sharpened_gpu, img_gpu)
        del sharpened_gpu, mask_gpu
    # Replace non-finite with zeros for safety
    cp.nan_to_num(img_gpu, copy=False)
    corrected = cp.asnumpy(img_gpu).astype(np.float32, copy=False)
    del img_gpu
    return corrected


def _compute_channel_quant_from_single_plane(
    corrected_plane: np.ndarray,
    mask: np.ndarray,
) -> QuantizationParams:
    """Compute channel quantization from a single corrected plane using fixed percentiles."""
    arr = np.asarray(corrected_plane, dtype=np.float32)
    arr = np.where(np.isfinite(arr), arr, 0.0)
    maskb = np.asarray(mask, dtype=bool)
    if maskb.shape != arr.shape:
        raise ValueError(f"Quantization mask shape {maskb.shape} does not match corrected plane {arr.shape}")
    data = arr[maskb]
    if data.size == 0:
        # Fallback to the whole plane if mask is empty
        data = arr.ravel()
    lower = float(np.percentile(data, QUANT_LOWER_PERCENTILE))
    # When very few finite pixels exist, fall back to 99.9 for stability
    upper_pct = QUANT_UPPER_PERCENTILE
    if data.size < QUANT_MIN_TOTAL_SAMPLES_FOR_HIGH_PERCENTILE:
        upper_pct = QUANT_FALLBACK_UPPER_PERCENTILE
    upper = float(np.percentile(data, upper_pct))
    if not math.isfinite(lower):
        lower = float(np.nanmin(data)) if np.isfinite(np.nanmin(data)) else 0.0
    if not math.isfinite(upper) or upper <= lower:
        u = float(np.nanmax(data)) if np.isfinite(np.nanmax(data)) else (lower + 1.0)
        upper = max(u, lower + QUANT_MIN_RANGE)
    if upper - lower < QUANT_MIN_RANGE:
        center = (upper + lower) * 0.5
        lower = center - QUANT_MIN_RANGE * 0.5
        upper = center + QUANT_MIN_RANGE * 0.5
    return QuantizationParams(
        lower=lower,
        upper=upper,
        observed_min=float(np.nanmin(data)),
        observed_max=float(np.nanmax(data)),
        lower_percentile=QUANT_LOWER_PERCENTILE,
        upper_percentile=upper_pct,
        sample_count=int(data.size),
    )


def _build_global_mask_from_fused(src: zarr.Array, qz: int) -> np.ndarray:
    """Compute a single boolean YX mask once from fused.zarr at reference z-plane.

    Uses union across channels: mask[y,x] = any(src[qz,y,x,c] > 0).
    """
    slab = np.asarray(src[qz, :, :, :])
    if slab.ndim == 2:
        return slab > 0
    return np.any(slab > 0, axis=-1)


@dataclass(slots=True)
class N4RuntimeConfig:
    workspace: Path
    roi: str
    codebook: str
    channel: int
    shrink: int
    spline_lowres_px: float
    z_index: int | None
    iterations: Sequence[int] = DEFAULT_ITERATIONS
    field_output: Path | None = None
    corrected_output: Path | None = None
    apply_correction: bool = True
    overwrite: bool = False
    correct_single_plane: bool = False
    # Optional list of channels to process in batch. If provided, these take
    # precedence over the single `channel` when using the multi-channel API
    # or CLI option. Existing single-channel behavior remains unchanged.
    channels: tuple[int, ...] | None = None
    debug: bool = False
    threshold: float | str | None = None
    use_unsharp_mask: bool = False


@dataclass(slots=True)
class N4Result:
    field_path: Path
    corrected_path: Path | None


@dataclass(slots=True)
class QuantizationParams:
    lower: float
    upper: float
    observed_min: float
    observed_max: float
    lower_percentile: float
    upper_percentile: float
    sample_count: int


# CPU quantization-parameter sampler removed; we now sample on GPU in
# _compute_quant_params_gpu_for_channel.


# CPU quantization path removed; quantization is now performed on GPU in
# _correct_and_quantize_to_u16_gpu to minimize transfers.


def _resolve_fused_path(workspace_path: Path, roi: str, codebook: str) -> Path:
    ws = Workspace(workspace_path)
    stitch_root = ws.stitch(roi, codebook)
    logger.info(
        "Resolved stitch directory for ROI '{roi}' and codebook '{codebook}' to {path}",
        roi=roi,
        codebook=codebook,
        path=stitch_root,
    )
    fused_path = stitch_root / "fused.zarr"
    if not fused_path.exists():
        raise FileNotFoundError(f"Could not find fused.zarr under {stitch_root}")
    logger.info("Found fused.zarr at {path}", path=fused_path)
    return fused_path


def _load_channel(fused_path: Path, *, channel_index: int, z_index: int | None) -> np.ndarray:
    fused = zarr.open_array(fused_path, mode="r")

    logger.info(
        "Opening fused store {path} with shape {shape} and dtype {dtype}",
        path=fused_path,
        shape=fused.shape,
        dtype=str(fused.dtype),
    )

    if fused.ndim != 4:
        raise ValueError(f"Expected fused.zarr to have 4 dimensions (Z, Y, X, C); received {fused.ndim}")

    z_dim, y_dim, x_dim, c_dim = fused.shape

    if z_index is None:
        raise ValueError("fused.zarr has a Z dimension; provide --z-index.")

    zi = int(z_index)
    if zi < 0 or zi >= z_dim:
        raise ValueError(f"Requested z-index {zi} but fused.zarr only has {z_dim} planes")

    if channel_index < 0 or channel_index >= c_dim:
        raise ValueError(f"Requested channel {channel_index} but fused.zarr only has {c_dim} channels")

    selected_channel = np.asarray(fused[zi, :, :, channel_index], dtype=np.float32)
    if selected_channel.shape != (y_dim, x_dim):  # pragma: no cover - defensive check
        raise ValueError(
            f"Resolved channel plane has unexpected shape {selected_channel.shape}; expected {(y_dim, x_dim)}"
        )

    logger.info(
        "Loaded channel {channel} at z={z} with array shape {shape}",
        channel=channel_index,
        z=zi,
        shape=selected_channel.shape,
    )
    return selected_channel


def _read_channel_names_from_zarr(store: Path) -> list[str] | None:
    try:
        arr = zarr.open_array(store, mode="r")
        # Prefer common metadata keys in order.
        raw = (
            arr.attrs.get("key")
            or arr.attrs.get("channel_names")
            or arr.attrs.get("channels")
            or arr.attrs.get("labels")
        )
        if isinstance(raw, (list, tuple)):
            return [str(x) for x in raw]
        if isinstance(raw, str):
            return [raw]
    except Exception:
        ...
    return None


def _slugify(name: str) -> str:
    # Safe filename-ish: lowercase, replace non-alnum with hyphen, collapse repeats
    import re

    s = name.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s or "unnamed"


def _write_fused_corrected_zyxc(
    fused_path: Path,
    *,
    channels: Sequence[int],
    fields: Sequence[np.ndarray],
    names: Sequence[str],
    z_index: int | None,
    single_plane: bool,
    overwrite: bool,
    roi: str,
    codebook: str,
    n4_params: dict[str, Any],
    output_path: Path | None,
    debug: bool,
    use_unsharp_mask: bool,
) -> Path:
    """Write corrected imagery to Zarr with axes ordered as ZYXC."""
    if len(channels) != len(fields) or len(channels) != len(names):
        raise ValueError("channels, fields, and names must be the same length")

    src = zarr.open_array(fused_path, mode="r")
    if src.ndim != 4:
        raise ValueError("Expected fused.zarr to be 4D (Z, Y, X, C)")

    z_dim, y_dim, x_dim, c_dim = src.shape
    for ch in channels:
        if ch < 0 or ch >= c_dim:
            raise ValueError(f"Requested channel {ch} but fused.zarr only has {c_dim} channels")

    if single_plane and z_index is not None:
        z_sel = [int(z_index)]
    else:
        z_sel = list(range(z_dim))

    logger.info(
        f"Preparing fused correction with channels={channels} over Z planes={z_sel} (debug={debug})",
    )
    logger.debug(
        "Opening fused source Zarr {path} (shape={shape}, dtype={dtype})",
        path=fused_path,
        shape=src.shape,
        dtype=str(src.dtype),
    )

    dest_path = output_path if output_path is not None else fused_path.parent / "fused_n4.zarr"
    # Use .partial directory for atomic write, then rename to final path
    partial_dest_path = dest_path.with_name(f"{dest_path.name}.partial")
    if dest_path.exists() and not overwrite:
        raise FileExistsError(f"{dest_path} already exists. Use --overwrite to replace it.")
    # Clean up any stale partial from previous failed runs
    if partial_dest_path.exists():
        shutil.rmtree(partial_dest_path, ignore_errors=True)
    float_dest_path: Path | None = None
    float_partial_path: Path | None = None
    if debug:
        float_dest_path = _float_store_path(dest_path)
        float_partial_path = float_dest_path.with_name(f"{float_dest_path.name}.partial")
        if float_dest_path.exists() and not overwrite:
            raise FileExistsError(f"{float_dest_path} already exists. Use --overwrite to replace it.")
        if float_partial_path.exists():
            shutil.rmtree(float_partial_path, ignore_errors=True)

    if src.chunks is not None and len(src.chunks) == 4:
        zc, yc, xc, _cc = src.chunks
        # Keep source chunking for Z/Y/X, stream in C with chunk=1
        dest_chunks = (min(zc, max(1, len(z_sel))), yc, xc, 1)
    else:
        dest_chunks = None

    t0_open = perf_counter()
    dest = zarr.open_array(
        partial_dest_path,
        mode="w",
        shape=(len(z_sel), y_dim, x_dim, len(channels)),
        chunks=dest_chunks,
        dtype=np.uint16,
    )
    logger.info(
        "Opened destination Zarr (partial create) at {path} with shape {shape} (took {dt:.2f}s)",
        path=dest_path,
        shape=dest.shape,
        dt=perf_counter() - t0_open,
    )
    float_dest = None
    if debug and float_dest_path is not None and float_partial_path is not None:
        t0_fopen = perf_counter()
        float_dest = zarr.open_array(
            float_partial_path,
            mode="w",
            shape=(len(z_sel), y_dim, x_dim, len(channels)),
            chunks=dest_chunks,
            dtype=np.float32,
        )
        logger.info(
            "Opened float32 debug Zarr at {path} with shape {shape} (took {dt:.2f}s)",
            path=float_dest_path,
            shape=float_dest.shape,
            dt=perf_counter() - t0_fopen,
        )

    field_arrays = [_ensure_float32(field) for field in fields]
    channel_quant: list[QuantizationParams] = []

    # Validate field shapes
    for field_arr in field_arrays:
        _validate_correction_field(field_arr)
        if field_arr.shape != (y_dim, x_dim):
            raise ValueError(
                f"Correction field shape {field_arr.shape} does not match fused YX {(y_dim, x_dim)}"
            )

    uint16_max = np.iinfo(np.uint16).max
    with progress_bar(max(1, len(channels) * len(z_sel))) as advance:
        for ci, (ch, field_arr) in enumerate(zip(channels, field_arrays)):
            # Upload field once per channel
            field_gpu = _prepare_gpu_field(field_arr)

            # Determine quantization from the z-index plane (or first selected plane)
            qz = int(z_index) if z_index is not None else int(z_sel[0])
            # Build global mask once for all channels/planes from union across channels
            global_mask = _build_global_mask_from_fused(src, qz)
            plane0 = np.asarray(src[qz, :, :, ch], dtype=np.float32)
            corrected0 = _correct_plane_gpu(
                plane0,
                field_gpu=field_gpu,
                use_unsharp_mask=use_unsharp_mask,
                mask_cpu=global_mask,
            )
            params = _compute_channel_quant_from_single_plane(corrected0, global_mask)
            channel_quant.append(params)
            logger.info(
                "Quant params for channel {ch} from z={z}: lower={lo:.4f}, upper={hi:.4f}",
                ch=ch,
                z=qz,
                lo=float(params.lower),
                hi=float(params.upper),
            )

            # Precompute scale with headroom
            width = max(params.upper - params.lower, QUANT_MIN_RANGE)
            scale = uint16_max / (width * (1.0 + QUANT_HEADROOM_FRACTION))

            for zi, z_slot in enumerate(z_sel):
                plane = np.asarray(src[z_slot, :, :, ch], dtype=np.float32)
                corrected = _correct_plane_gpu(
                    plane,
                    field_gpu=field_gpu,
                    use_unsharp_mask=use_unsharp_mask,
                    mask_cpu=global_mask,
                )
                # Quantize on CPU
                y = (corrected - params.lower) * scale
                np.clip(y, 0.0, float(uint16_max), out=y)
                u16 = np.rint(y).astype(np.uint16)
                if float_dest is not None:
                    float_dest[zi, :, :, ci] = corrected
                dest[zi, :, :, ci] = u16
                advance()

    src_attrs: dict[str, Any] = {}
    try:
        src_attrs = dict(src.attrs)
    except Exception:
        pass
    dest_attrs: dict[str, Any] = dict(src_attrs)
    dest_attrs["axes"] = "ZYXC"
    dest_attrs["key"] = [str(n) for n in names]
    dest_attrs["roi"] = roi
    dest_attrs["codebook"] = codebook
    dest_attrs["n4"] = n4_params
    quant_channels: list[dict[str, Any]] = []
    uint16_range = np.iinfo(np.uint16).max
    for idx, name, params in zip(channels, names, channel_quant):
        width = max(params.upper - params.lower, QUANT_MIN_RANGE)
        scale = uint16_range / (width * (1.0 + QUANT_HEADROOM_FRACTION))
        quant_channels.append({
            "index": int(idx),
            "name": str(name),
            "lower": float(params.lower),
            "upper": float(params.upper),
            "observed_min": float(params.observed_min),
            "observed_max": float(params.observed_max),
            "scale": float(scale),
            "lower_percentile": float(params.lower_percentile),
            "upper_percentile": float(params.upper_percentile),
            "samples": int(params.sample_count),
        })

    default_quant = channel_quant[0] if channel_quant else None
    dest_attrs["quantization"] = {
        "lower_percentile": (
            float(default_quant.lower_percentile) if default_quant is not None else QUANT_LOWER_PERCENTILE
        ),
        "upper_percentile": (
            float(default_quant.upper_percentile) if default_quant is not None else QUANT_UPPER_PERCENTILE
        ),
        "channels": quant_channels,
        "dtype": "uint16",
        "sampling": {
            "strategy": "single_plane_percentile",
            "plane_index": int(z_index) if z_index is not None else int(z_sel[0]),
        },
    }
    if float_dest_path is not None:
        dest_attrs["quantization"]["float_store"] = str(float_dest_path.name)
    _zarr_attrs_write(dest.attrs, dest_attrs)

    if float_dest is not None and float_dest_path is not None:
        try:
            float_attrs = {
                "axes": "ZYXC",
                "key": [str(n) for n in names],
                "roi": roi,
                "codebook": codebook,
                "n4": n4_params,
            }
            _zarr_attrs_write(float_dest.attrs, float_attrs)
        except Exception:
            logger.exception("Failed to write metadata for float32 corrected store")

    # Finalize: remove any existing destination stores, then move partial → final.
    # This avoids "Directory not empty" errors when a previous run left a non-empty
    # destination directory behind.
    try:
        # Proactively remove existing debug float store if present
        if float_dest_path is not None and Path(float_dest_path).exists():
            logger.warning(
                "Removing existing float32 corrected store before rename: {path}",
                path=float_dest_path,
            )
            if Path(float_dest_path).is_dir():
                shutil.rmtree(float_dest_path)
            else:
                Path(float_dest_path).unlink()

        # Proactively remove existing destination store if present
        if Path(dest_path).exists():
            logger.warning(
                "Removing existing corrected store before rename: {path}",
                path=dest_path,
            )
            if Path(dest_path).is_dir():
                shutil.rmtree(dest_path)
            else:
                Path(dest_path).unlink()

        # Atomically move partial directories to final paths
        if float_dest_path is not None and float_partial_path is not None and float_dest is not None:
            Path(float_partial_path).replace(float_dest_path)
        Path(partial_dest_path).replace(dest_path)
    except Exception:
        # On failure, try to clean up partials to avoid clutter
        try:
            if Path(partial_dest_path).exists():
                shutil.rmtree(partial_dest_path, ignore_errors=True)
            if float_partial_path is not None and Path(float_partial_path).exists():
                shutil.rmtree(float_partial_path, ignore_errors=True)
        finally:
            raise

    logger.info(
        "Corrected fused store written to {path} with shape {shape}",
        path=dest_path,
        shape=dest.shape,
    )
    if float_dest_path is not None:
        logger.info(
            "Float32 corrected store written to {path}",
            path=float_dest_path,
        )
    return dest_path


def compute_correction_field(
    image_yx: np.ndarray,
    *,
    shrink: int,
    spline_lowres_px: float,
    iterations: Iterable[int] = DEFAULT_ITERATIONS,
    threshold: float | str | None = None,
) -> np.ndarray:
    if image_yx.ndim != 2:
        raise ValueError(f"N4 expects a 2D YX image; received shape {image_yx.shape}")
    if shrink < 1:
        raise ValueError("shrink must be >= 1")
    if spline_lowres_px <= 0:
        raise ValueError("spline_lowres_px must be > 0")

    start_ts = perf_counter()
    iters = list(iterations)
    logger.info(
        "Starting N4 bias-field estimation (shape={shape}, shrink={shrink}, spline={spline}, iterations={iters})",
        shape=image_yx.shape,
        shrink=shrink,
        spline=spline_lowres_px,
        iters=iters,
    )
    img_full = sitk.GetImageFromArray(np.asarray(image_yx, dtype=np.float32))
    # If you know XY spacing, set it here instead of (1,1):
    img_full.SetSpacing((1.0, 1.0))

    # Downsample for fitting
    shrink_factors = [shrink] * img_full.GetDimension()
    img_small = sitk.Shrink(img_full, shrink_factors)

    # Foreground/tissue mask on the small image using the configured threshold.
    mask_small_array, mask_small_meta = _compute_threshold_mask(sitk.GetArrayFromImage(img_small), threshold)
    if not np.any(mask_small_array):
        raise ValueError(_empty_mask_message(threshold, resolved_name=mask_small_meta.get("function")))
    sm_total = mask_small_array.size
    sm_fg = int(np.count_nonzero(mask_small_array))
    logger.info(f"Small-image mask coverage: {sm_fg}/{sm_total} ({(sm_fg / max(sm_total, 1)):.2%})")
    mask_small_image = sitk.GetImageFromArray(mask_small_array.astype(np.uint8, copy=False))
    mask_small_image.CopyInformation(img_small)
    mask_small = sitk.Cast(mask_small_image, sitk.sitkUInt8)

    n4 = sitk.N4BiasFieldCorrectionImageFilter()
    n4.SetMaximumNumberOfIterations(iters)

    # Configure B-spline control point grid: spacing in *small-image pixels*
    # Convert desired pixel spacing (small grid) to physical spacing.
    spline_order = n4.GetSplineOrder()  # default 3
    cps: list[int] = []
    for d in range(img_small.GetDimension()):
        spacing_phys = img_small.GetSpacing()[d]  # ~shrink
        desired_phys = float(spline_lowres_px) * spacing_phys
        physical_len = spacing_phys * img_small.GetSize()[d]
        mesh = max(1, int(round(physical_len / max(desired_phys, 1e-6))))
        n_ctrl_dim = mesh + spline_order  # ensures mesh >= 1
        cps.append(n_ctrl_dim)
    n4.SetNumberOfControlPoints(cps)  # mesh size = cps - order

    # Run N4 on the downsampled image and reconstruct the log-bias field at full-res
    _ = n4.Execute(img_small, mask_small)
    log_bias_full = sitk.Cast(n4.GetLogBiasFieldAsImage(img_full), sitk.sitkFloat32)
    # GetArrayFromImage returns a writable numpy array; avoids read-only view issues.
    field = sitk.GetArrayFromImage(sitk.Exp(log_bias_full)).astype(np.float32, copy=False)

    # Normalize field to ~1.0 on foreground
    fg_mask, mask_meta = _compute_threshold_mask(image_yx, threshold)
    if not np.any(fg_mask):
        raise ValueError(_empty_mask_message(threshold, resolved_name=mask_meta.get("function")))
    scale = float(np.nanmedian(field[fg_mask]))  # robust center
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    field /= scale
    fg_total = fg_mask.size
    fg_fg = int(np.count_nonzero(fg_mask))
    logger.info(
        f"Foreground mask coverage: {fg_fg}/{fg_total} ({(fg_fg / max(fg_total, 1)):.2%}); normalization scale={scale:.6f}"
    )
    # Strict positivity for division stability
    np.maximum(field, np.finfo(np.float32).tiny, out=field)
    duration = perf_counter() - start_ts
    fmin = float(np.min(field))
    fmax = float(np.max(field))
    fmean = float(np.mean(field))
    logger.info(
        f"Finished N4 bias-field estimation in {duration:.2f}s (shape={image_yx.shape}); "
        f"field stats min={fmin:.6f}, max={fmax:.6f}, mean={fmean:.6f}"
    )
    return field


def _validate_correction_field(field: np.ndarray) -> None:
    if not isinstance(field, np.ndarray):
        raise TypeError("Field must be a numpy.ndarray")
    if field.ndim != 2:
        raise ValueError("Field must be 2D (Y, X)")
    if not np.all(np.isfinite(field)):
        raise ValueError("Field must be finite")
    if np.any(field <= 0):
        raise ValueError("Field must be strictly positive")


def _write_multi_channel_field(
    fields_cyx: np.ndarray,  # C, Y, X
    names: Sequence[str],
    output_path: Path,
    *,
    overwrite: bool,
    meta_extra: dict[str, Any] | None = None,
) -> Path:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} already exists. Use --overwrite to replace it.")
    fields_cyx = np.asarray(fields_cyx, dtype=np.float32)
    if fields_cyx.ndim != 3:
        raise ValueError("fields_cyx must be a 3D array with shape (C, Y, X)")
    meta = {
        "axes": "CYX",
        "key": [str(n) for n in names],
    }
    if meta_extra:
        meta.update(meta_extra)
    # Use safe_imwrite but route through the module-level imwrite so tests can patch
    safe_imwrite(
        output_path,
        fields_cyx,
        dtype=np.float32,
        **IMWRITE_KWARGS,
        metadata=meta,
    )
    logger.info(
        "Wrote correction field stack to {path} with shape {shape}", path=output_path, shape=fields_cyx.shape
    )
    return output_path


def _load_multi_channel_field(path: Path) -> tuple[np.ndarray, list[str] | None]:
    """Load a previously written correction field stack (CYX) along with channel names."""

    with tifffile.TiffFile(path) as tif:
        data = tif.asarray()
        meta_names: list[str] | None = None
        shaped = getattr(tif, "shaped_metadata", None)
        if shaped:
            first_meta = shaped[0]
            if isinstance(first_meta, dict):
                raw = first_meta.get("key") or first_meta.get("channel_names") or first_meta.get("channels")
                if isinstance(raw, (list, tuple)):
                    meta_names = [str(v) for v in raw]
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim == 2:  # single channel stored without explicit C axis
        arr = arr[np.newaxis, ...]
    return arr, meta_names


def compute_fields_from_workspace(config: N4RuntimeConfig) -> list[N4Result]:
    """Compute (and optionally apply) N4 correction for multiple channels.

    Produces ONE multi-channel correction field (CYX) and ONE corrected fused Zarr (ZYXC).
    `config.channels` selects which C-indices to process; defaults to the single `channel`.
    """
    fused_path = _resolve_fused_path(config.workspace, config.roi, config.codebook)
    if config.z_index is None:
        raise ValueError("This workflow computes a 2D N4 field from a single Z plane; please set --z-index.")

    # Determine channels to process
    if config.channels and len(config.channels) > 0:
        channels: list[int] = list(dict.fromkeys(int(ch) for ch in config.channels))
    else:
        channels = [int(config.channel)]

    # Resolve channel names from fused.zarr metadata when available
    names = _read_channel_names_from_zarr(fused_path)

    # Ensure destination folder for fields
    fields_dir = fused_path.parent / "n4-fields"
    fields_dir.mkdir(parents=True, exist_ok=True)
    field_output = config.field_output if config.field_output else fields_dir / "n4_correction_field.tif"

    field_list: list[np.ndarray] = []
    name_list: list[str] = []
    reuse_existing = False

    if field_output.exists() and not config.overwrite:
        try:
            existing_stack, stored_names = _load_multi_channel_field(field_output)
            if existing_stack.shape[0] == len(channels):
                logger.info(
                    "Found existing correction field at {path} and --overwrite not set; reusing without recomputation.",
                    path=field_output,
                )
                reuse_existing = True
                for idx, ch in enumerate(channels):
                    field_arr = np.asarray(existing_stack[idx], dtype=np.float32)
                    _validate_correction_field(field_arr)
                    field_list.append(field_arr.copy())
                    if stored_names and len(stored_names) == existing_stack.shape[0]:
                        ch_name = stored_names[idx]
                    elif names and 0 <= ch < len(names):
                        ch_name = names[ch]
                    else:
                        ch_name = f"channel_{ch}"
                    name_list.append(ch_name)
            else:
                logger.warning(
                    "Existing correction field at {path} has {found} channel(s), expected {expected}; recomputing.",
                    path=field_output,
                    found=existing_stack.shape[0],
                    expected=len(channels),
                )
        except Exception:
            logger.opt(exception=True).warning(
                "Failed to load existing correction field from {path}; recomputing.", path=field_output
            )

    # Compute per-channel fields in parallel to reduce wall-clock time.
    # Preserve result order by channel index.
    def _compute_one(ch: int) -> tuple[int, str, np.ndarray]:
        logger.info(
            f"Processing channel {ch} (workspace={config.workspace}, roi={config.roi}, codebook={config.codebook})"
        )
        logger.info(
            f"N4 params for channel {ch}: z_index={config.z_index}, shrink={config.shrink}, "
            f"spline_lowres_px={config.spline_lowres_px}, iterations={list(config.iterations)}, "
            f"threshold={_summarize_threshold_spec(config.threshold)}, unsharp={config.use_unsharp_mask}"
        )

        image_yx = _load_channel(
            fused_path,
            channel_index=ch,
            z_index=config.z_index,
        )

        if config.use_unsharp_mask:
            logger.info(f"Applying unsharp mask to channel {ch} plane prior to N4")
            mask = image_yx > 0.0
            image_yx = _apply_unsharp_mask_if_enabled(image_yx, mask=mask, enabled=True)

        field = compute_correction_field(
            image_yx,
            shrink=config.shrink,
            spline_lowres_px=config.spline_lowres_px,
            iterations=config.iterations,
            threshold=config.threshold,
        )

        # Resolve channel tag from metadata (fallback to index)
        ch_name = names[ch] if names and 0 <= ch < len(names) else f"channel_{ch}"

        return ch, ch_name, field

    if not reuse_existing:
        max_workers = min(len(channels), max(1, (os.cpu_count() or 1) // 2))
        logger.info(
            "Submitting N4 field computations for {n} channel(s) using {w} worker thread(s)...",
            n=len(channels),
            w=max_workers,
        )
        results: dict[int, tuple[str, np.ndarray]] = {}
        if len(channels) == 1:
            ch = channels[0]
            cidx, cname, cfield = _compute_one(ch)
            results[cidx] = (cname, cfield)
        else:
            # Small launch delay to avoid CPU/memory spikes when N4 spawns threads internally.
            futmap: dict[Any, int] = {}
            try:
                with shared_thread_pool(
                    max_workers=max_workers, launch_delay=0.05, thread_name_prefix="n4-field"
                ) as pool:
                    logger.info("Waiting for {n} field-future(s) to complete...", n=len(channels))
                    futmap = {pool.submit(_compute_one, ch): ch for ch in channels}
                    for fut in as_completed(futmap):
                        ch = futmap[fut]
                        cidx, cname, cfield = fut.result()  # propagate exceptions immediately
                        results[cidx] = (cname, cfield)
                        logger.debug("Completed field for channel {ch}", ch=ch)
            except KeyboardInterrupt:
                for fut in futmap:
                    fut.cancel()
                logger.warning("KeyboardInterrupt received; cancelled pending N4 field computations.")
                raise

        # Accumulate in the original channel order
        for ch in channels:
            cname, cfield = results[ch]
            name_list.append(cname)
            field_list.append(cfield)

    # Write stacked multi-channel field once at the end
    meta_extra_all: dict[str, Any] = {
        "roi": config.roi,
        "codebook": config.codebook,
        "n4": {
            "shrink": int(config.shrink),
            "spline_lowres_px": float(config.spline_lowres_px),
            "iterations": list(config.iterations),
            "z_index": (int(config.z_index) if config.z_index is not None else None),
            "threshold": _summarize_threshold_spec(config.threshold),
            "use_unsharp_mask": bool(config.use_unsharp_mask),
        },
    }
    if reuse_existing:
        written_field = field_output
    else:
        stacked = np.stack(field_list, axis=0).astype(np.float32)
        written_field = _write_multi_channel_field(
            stacked,
            name_list,
            field_output,
            overwrite=config.overwrite,
            meta_extra=meta_extra_all,
        )

    # Emit per-channel diagnostic PNGs
    if not reuse_existing:
        for ch_name, arr in zip(name_list, field_list):
            try:
                tag = _slugify(ch_name)
                png_path = written_field.with_name(f"{written_field.stem}_{tag}.png")
                _write_png_normalized(arr, png_path, long_edge_max=1024)
            except Exception:
                logger.exception("Failed to write diagnostic PNG for channel {name}", name=ch_name)

    corrected_path: Path | None = None
    if config.apply_correction:
        corrected_output = config.corrected_output or (fused_path.parent / "fused_n4.zarr")
        corrected_path = _write_fused_corrected_zyxc(
            fused_path,
            channels=channels,
            fields=field_list,
            names=name_list,
            z_index=config.z_index,
            single_plane=config.correct_single_plane,
            overwrite=config.overwrite,
            roi=config.roi,
            codebook=config.codebook,
            n4_params={
                "shrink": int(config.shrink),
                "spline_lowres_px": float(config.spline_lowres_px),
                "iterations": list(config.iterations),
                "z_index": (int(config.z_index) if config.z_index is not None else None),
                "threshold": _summarize_threshold_spec(config.threshold),
                "use_unsharp_mask": bool(config.use_unsharp_mask),
            },
            output_path=corrected_output,
            debug=config.debug,
            use_unsharp_mask=config.use_unsharp_mask,
        )

    logger.info("Finished N4 workflow for {n} channel(s)", n=len(channels))
    return [N4Result(field_path=written_field, corrected_path=corrected_path)]


def run_cli_workflow(
    *,
    workspace: Path,
    roi: str,
    codebook: str,
    channels: str | None,
    shrink: int,
    spline_lowres_px: float,
    z_index: int | None,
    threshold: str | None,
    field_output: Path | None,
    corrected_output: Path | None,
    apply_correction: bool,
    overwrite: bool,
    single_plane: bool,
    debug: bool,
    use_unsharp_mask: bool,
) -> list[N4Result]:
    """Shared CLI workflow for Click- and Typer-based entrypoints."""

    if z_index is None:
        raise ValueError("--z-index is required (field is computed from one Z plane).")

    logger.info(
        "Invoked CLI with workspace={workspace}, roi={roi}, codebook={codebook}, channels={channels}, "
        "use_unsharp_mask={use_unsharp_mask}",
        workspace=workspace,
        roi=roi,
        codebook=codebook,
        channels=channels,
        use_unsharp_mask=use_unsharp_mask,
    )

    fused_path = _resolve_fused_path(workspace, roi, codebook)
    arr = zarr.open_array(fused_path, mode="r")
    if arr.ndim != 4:
        raise ValueError(
            f"Expected fused.zarr to be 4D with axes ZYXC; found ndim={arr.ndim} and shape={arr.shape}"
        )

    c_count = int(arr.shape[-1])
    if c_count <= 0:
        raise ValueError("fused.zarr reports zero channels; cannot compute N4 correction field.")

    name_list = _read_channel_names_from_zarr(fused_path) or [f"channel_{i}" for i in range(c_count)]

    if channels is None or not channels.strip():
        selected_indices: list[int] = list(range(c_count))
    else:
        tokens = [token.strip() for token in channels.split(",") if token.strip()]
        name_to_idx = {name.lower(): idx for idx, name in enumerate(name_list)}
        selected_indices = []
        for token in tokens:
            idx = name_to_idx.get(token.lower())
            if idx is None:
                try:
                    idx = int(token)
                except ValueError as exc:
                    available = (
                        ", ".join(name_list) if name_list else ", ".join(str(i) for i in range(c_count))
                    )
                    raise ValueError(f"Unknown channel '{token}'. Available: {available}") from exc
            if idx < 0 or idx >= c_count:
                raise ValueError(f"Channel index out of range: {idx}")
            if idx not in selected_indices:
                selected_indices.append(idx)

    if not selected_indices:
        raise ValueError("No channels selected; specify --channels or ensure fused.zarr has channels.")

    threshold_value: float | str | None = None
    if threshold is not None and threshold.strip():
        token = threshold.strip()
        try:
            numeric_value = float(token)
        except ValueError:
            try:
                _resolve_threshold_function(token)
            except ValueError as exc:
                raise ValueError(str(exc)) from exc
            threshold_value = token
        else:
            if not math.isfinite(numeric_value):
                raise ValueError("Threshold must be a finite numeric value.")
            threshold_value = numeric_value

    config = N4RuntimeConfig(
        workspace=workspace,
        roi=roi,
        codebook=codebook,
        channel=selected_indices[0],
        shrink=shrink,
        spline_lowres_px=spline_lowres_px,
        z_index=z_index,
        iterations=DEFAULT_ITERATIONS,
        field_output=field_output,
        corrected_output=corrected_output,
        apply_correction=apply_correction,
        overwrite=overwrite,
        correct_single_plane=single_plane,
        channels=tuple(selected_indices),
        debug=debug,
        threshold=threshold_value,
        use_unsharp_mask=use_unsharp_mask,
    )

    return compute_fields_from_workspace(config)


__all__ = [
    "N4RuntimeConfig",
    "N4Result",
    "DEFAULT_ITERATIONS",
    "compute_correction_field",
    "compute_fields_from_workspace",
    "run_cli_workflow",
]
