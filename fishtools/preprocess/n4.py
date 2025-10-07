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
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any, Callable, Iterable, Iterator, Sequence

import numpy as np
import SimpleITK as sitk
import zarr
from loguru import logger
from PIL import Image
from skimage import filters
from tifffile import imwrite

try:  # pragma: no cover - exercised via monkeypatch in tests
    import cupy as cp
except Exception:  # pragma: no cover - runtime fallback when CuPy unavailable
    cp = None  # type: ignore[assignment]

if TYPE_CHECKING:  # pragma: no cover - typing aid
    from cupy import ndarray as cupy_ndarray  # type: ignore[import-not-found]
else:
    cupy_ndarray = Any

from fishtools import IMWRITE_KWARGS
from fishtools.io.workspace import Workspace
from fishtools.utils.logging import setup_logging
from fishtools.utils.pretty_print import progress_bar

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
    f = np.asarray(img, dtype=np.float32)
    f = np.where(np.isfinite(f), f, 0.0)
    p1, p99 = np.percentile(f, [p_lo, p_hi])
    if not np.isfinite(p1):
        p1 = float(np.nanmin(f)) if np.isfinite(np.nanmin(f)) else 0.0
    if not np.isfinite(p99) or p99 <= p1:
        p99 = float(np.nanmax(f)) if np.isfinite(np.nanmax(f)) else (p1 + 1.0)
        if p99 <= p1:
            p99 = p1 + 1.0
    fn = np.clip((f - p1) / (p99 - p1), 0.0, 1.0)
    return (fn * 255.0 + 0.5).astype(np.uint8)


def _write_png_normalized(field: np.ndarray, png_path: Path, long_edge_max: int = 1024) -> None:
    img8 = _normalize_to_uint8_by_percentile(field, 1.0, 99.0)
    h, w = img8.shape
    long_edge = max(h, w)
    if long_edge > long_edge_max:
        scale = long_edge_max / float(long_edge)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        Image.fromarray(img8, mode="L").resize((new_w, new_h), resample=Image.BILINEAR).save(png_path)
    else:
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
            "Threshold function '{name}' returned shape {returned}, expected scalar or {expected}.".format(
                name=resolved_name,
                returned=threshold_array.shape,
                expected=data.shape,
            )
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


@lru_cache(maxsize=1)
def _cupy_available() -> bool:
    """Return True when CuPy is importable and at least one device is accessible."""

    if cp is None:
        return False
    try:
        cuda = getattr(cp, "cuda", None)
        runtime = getattr(cuda, "runtime", None) if cuda is not None else None
        if runtime is not None:
            device_count = runtime.getDeviceCount()  # type: ignore[attr-defined]
            if isinstance(device_count, int) and device_count <= 0:
                return False
        # Validate we can allocate on the selected backend (GPU or NumPy stub).
        _ = cp.asarray([0.0], dtype=cp.float32)
    except Exception:
        return False
    return True


def _prepare_gpu_field(field: np.ndarray) -> cupy_ndarray | None:
    """Transfer a correction field to GPU memory when CuPy is available."""

    if not _cupy_available():
        return None
    # Let any transfer error surface to the caller; no silent CPU fallback here.
    return cp.asarray(_ensure_float32(field), dtype=cp.float32)


def _cupy_to_numpy(array: cupy_ndarray) -> np.ndarray:
    if cp is None:
        return np.asarray(array, dtype=np.float32)
    # Propagate errors instead of converting via a CPU fallback path.
    return np.asarray(cp.asnumpy(array), dtype=np.float32)


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


def _compute_quantization_params(
    plane_supplier: Callable[[], Iterator[np.ndarray]],
    *,
    lower_percentile: float = QUANT_LOWER_PERCENTILE,
    upper_percentile: float = QUANT_UPPER_PERCENTILE,
    min_range: float = QUANT_MIN_RANGE,
) -> QuantizationParams:
    """Derive robust min/max for quantization using random subsampling and percentiles."""

    sampled_values: list[np.ndarray] = []
    observed_min = math.inf
    observed_max = -math.inf
    rng = np.random.default_rng(0)

    for plane in plane_supplier():
        arr = np.asarray(plane, dtype=np.float32)
        if arr.size == 0:
            continue
        finite_mask = np.isfinite(arr)
        if not np.any(finite_mask):
            continue
        finite_values = arr[finite_mask]
        plane_min = float(np.min(finite_values))
        plane_max = float(np.max(finite_values))
        if plane_min < observed_min:
            observed_min = plane_min
        if plane_max > observed_max:
            observed_max = plane_max

        flat = finite_values.ravel()
        if flat.size <= QUANT_MAX_SAMPLES:
            sampled = flat
        else:
            idx = rng.choice(flat.size, size=QUANT_MAX_SAMPLES, replace=False)
            sampled = flat[idx]
        # Random subsampling provides coverage without relying on a fixed stride.
        if sampled.size:
            sampled_values.append(sampled.astype(np.float32, copy=False))

    if not sampled_values:
        if not math.isfinite(observed_min):
            observed_min = 0.0
        if not math.isfinite(observed_max):
            observed_max = observed_min + 1.0
        upper = max(observed_max, observed_min + min_range)
        return QuantizationParams(
            lower=float(observed_min),
            upper=float(upper),
            observed_min=float(observed_min),
            observed_max=float(observed_max),
            lower_percentile=lower_percentile,
            upper_percentile=upper_percentile,
            sample_count=0,
        )

    samples = np.concatenate(sampled_values)
    total_samples = int(samples.size)
    effective_lower = float(lower_percentile)
    effective_upper = float(upper_percentile)
    if total_samples < QUANT_MIN_TOTAL_SAMPLES_FOR_HIGH_PERCENTILE and math.isclose(
        upper_percentile, QUANT_UPPER_PERCENTILE
    ):
        effective_upper = QUANT_FALLBACK_UPPER_PERCENTILE
        logger.info(
            "Falling back to upper percentile {upper} due to limited samples ({samples})",
            upper=effective_upper,
            samples=total_samples,
        )

    lower = float(np.percentile(samples, effective_lower))
    upper = float(np.percentile(samples, effective_upper))

    if not math.isfinite(lower):
        lower = float(observed_min if math.isfinite(observed_min) else 0.0)
    if not math.isfinite(upper):
        upper = float(observed_max if math.isfinite(observed_max) else lower + min_range)

    if upper - lower < min_range:
        center = (upper + lower) / 2.0
        lower = center - (min_range / 2.0)
        upper = center + (min_range / 2.0)

    return QuantizationParams(
        lower=lower,
        upper=upper,
        observed_min=float(observed_min if math.isfinite(observed_min) else lower),
        observed_max=float(observed_max if math.isfinite(observed_max) else upper),
        lower_percentile=effective_lower,
        upper_percentile=effective_upper,
        sample_count=total_samples,
    )


def _quantize_to_uint16(data: np.ndarray, params: QuantizationParams) -> np.ndarray:
    dtype_info = np.iinfo(np.uint16)
    lower = params.lower
    upper = params.upper
    width = max(upper - lower, QUANT_MIN_RANGE)
    # Widen the mapping range by a small headroom to reduce saturation.
    scale = dtype_info.max / (width * (1.0 + QUANT_HEADROOM_FRACTION))
    scaled = np.clip((_ensure_float32(data) - lower) * scale, 0.0, dtype_info.max)
    return np.round(scaled).astype(np.uint16)


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

    dest_path = output_path if output_path is not None else fused_path.parent / "fused_n4.zarr"
    if dest_path.exists() and not overwrite:
        raise FileExistsError(f"{dest_path} already exists. Use --overwrite to replace it.")
    float_dest_path: Path | None = None
    if debug:
        float_dest_path = _float_store_path(dest_path)
        if float_dest_path.exists() and not overwrite:
            raise FileExistsError(f"{float_dest_path} already exists. Use --overwrite to replace it.")

    if src.chunks is not None and len(src.chunks) == 4:
        zc, yc, xc, _cc = src.chunks
        # Keep source chunking for Z/Y/X, stream in C with chunk=1
        dest_chunks = (min(zc, max(1, len(z_sel))), yc, xc, 1)
    else:
        dest_chunks = None

    dest = zarr.open_array(
        dest_path,
        mode="w",
        shape=(len(z_sel), y_dim, x_dim, len(channels)),
        chunks=dest_chunks,
        dtype=np.uint16,
    )
    float_dest = None
    if debug and float_dest_path is not None:
        float_dest = zarr.open_array(
            float_dest_path,
            mode="w",
            shape=(len(z_sel), y_dim, x_dim, len(channels)),
            chunks=dest_chunks,
            dtype=np.float32,
        )

    field_arrays = [_ensure_float32(field) for field in fields]
    for field_arr in field_arrays:
        _validate_correction_field(field_arr)
    channel_quant: list[QuantizationParams] = []
    gpu_fields: list[cupy_ndarray | None] = []

    for channel_idx, field_arr in zip(channels, field_arrays):
        field_gpu = _prepare_gpu_field(field_arr)
        gpu_fields.append(field_gpu)

        def _supplier() -> Iterator[np.ndarray]:
            for zi in z_sel:
                yield _apply_correction_field_prepared(
                    np.asarray(src[zi, :, :, channel_idx], dtype=np.float32),
                    field_arr,
                    field_gpu=field_gpu,
                )

        params = _compute_quantization_params(_supplier)
        channel_quant.append(params)

    total_steps = max(1, len(channels) * len(z_sel))
    with progress_bar(total_steps) as advance:
        for ci, (ch, field_arr, params) in enumerate(zip(channels, field_arrays, channel_quant)):
            field_gpu = gpu_fields[ci]
            for zi, z_slot in enumerate(z_sel):
                corrected = _apply_correction_field_prepared(
                    np.asarray(src[z_slot, :, :, ch], dtype=np.float32),
                    field_arr,
                    field_gpu=field_gpu,
                )
                if float_dest is not None:
                    float_dest[zi, :, :, ci] = corrected
                dest[zi, :, :, ci] = _quantize_to_uint16(corrected, params)
                advance()

            if field_gpu is not None and _cupy_available():
                memory_pool_getter = getattr(cp, "get_default_memory_pool", None)
                if callable(memory_pool_getter):
                    try:
                        memory_pool_getter().free_all_blocks()
                    except Exception:
                        logger.opt(exception=True).debug("Unable to release CuPy memory pool; continuing.")
            gpu_fields[ci] = None

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
        scale = uint16_range / max(params.upper - params.lower, QUANT_MIN_RANGE)
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
            "strategy": "random_without_replacement",
            "max_samples": int(QUANT_MAX_SAMPLES),
            "total_samples": (int(default_quant.sample_count) if default_quant is not None else 0),
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

    logger.info(
        "Corrected fused store written to {path} with shape {shape}",
        path=dest_path,
        shape=dest.shape,
    )
    if float_dest_path is not None and float_dest is not None:
        logger.info(
            "Float32 corrected store written to {path} with shape {shape}",
            path=float_dest_path,
            shape=float_dest.shape,
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
    # Strict positivity for division stability
    np.maximum(field, np.finfo(np.float32).tiny, out=field)
    duration = perf_counter() - start_ts
    logger.info(
        "Finished N4 bias-field estimation in {duration:.2f}s (shape={shape})",
        duration=duration,
        shape=image_yx.shape,
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


def apply_correction_field(image_yx: np.ndarray, field_yx: np.ndarray) -> np.ndarray:
    field_f32 = _ensure_float32(field_yx)
    _validate_correction_field(field_f32)
    field_gpu = _prepare_gpu_field(field_f32)
    return _apply_correction_field_prepared(image_yx, field_f32, field_gpu=field_gpu)


def _apply_correction_field_prepared(
    image_yx: np.ndarray,
    field_f32: np.ndarray,
    *,
    field_gpu: cupy_ndarray | None = None,
) -> np.ndarray:
    # Caller guarantees validation and float32 casting of field
    img = _ensure_float32(image_yx)
    if field_gpu is not None and _cupy_available():
        img_gpu = cp.asarray(img, dtype=cp.float32)
        corrected_gpu = img_gpu / field_gpu
        return _cupy_to_numpy(corrected_gpu)
    return img / field_f32


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
    imwrite(output_path, fields_cyx, dtype=np.float32, **IMWRITE_KWARGS, metadata=meta)
    logger.info(
        "Wrote correction field stack to {path} with shape {shape}", path=output_path, shape=fields_cyx.shape
    )
    return output_path


def apply_correction_to_store(
    fused_path: Path,
    *,
    channel_index: int,
    field: np.ndarray,
    output_path: Path,
    overwrite: bool,
    z_index: int | None = None,
    single_plane: bool = False,
    debug: bool = False,
) -> Path:
    fused = zarr.open_array(fused_path, mode="r")
    if fused.ndim != 4:
        raise ValueError("fused.zarr is expected to be ZYXC")
    z_dim, y_dim, x_dim, c_dim = fused.shape
    if channel_index < 0 or channel_index >= c_dim:
        raise ValueError(f"Invalid channel index {channel_index} for fused.zarr with {c_dim} channels")

    if single_plane and z_index is None:
        raise ValueError("single_plane=True requires z_index to be provided")

    chunks = getattr(fused, "chunks", None)
    single_plane_mode = bool(single_plane)
    if single_plane_mode:
        zi = int(z_index)
        plane_mappings: Sequence[tuple[int, int]] = [(0, zi)]
        output_shape: Sequence[int] = (1, y_dim, x_dim, 1)
        if chunks is not None:
            chunk_spec = (1, chunks[1], chunks[2], 1)
        else:
            chunk_spec = None
    else:
        plane_mappings = [(idx, idx) for idx in range(z_dim)]
        output_shape = (z_dim, y_dim, x_dim, 1)
        if chunks is not None:
            chunk_spec = (chunks[0], chunks[1], chunks[2], 1)
        else:
            chunk_spec = None

    logger.info(
        "Quantizing channel {channel} using Z plane(s) {planes} (debug={debug})",
        channel=channel_index,
        planes=[src for _, src in plane_mappings],
        debug=debug,
    )

    target_dtype = np.uint16
    dtype_info = np.iinfo(target_dtype)

    def _load_plane(index: int) -> np.ndarray:
        return np.asarray(fused[index, :, :, channel_index], dtype=np.float32)

    field_f = _ensure_float32(field)
    _validate_correction_field(field_f)
    field_gpu = _prepare_gpu_field(field_f)

    def _supplier() -> Iterator[np.ndarray]:
        for _, src_idx in plane_mappings:
            yield _apply_correction_field_prepared(
                _load_plane(src_idx),
                field_f,
                field_gpu=field_gpu,
            )

    quant_params = _compute_quantization_params(_supplier)
    dtype_info = np.iinfo(np.uint16)
    scale = dtype_info.max / max(quant_params.upper - quant_params.lower, QUANT_MIN_RANGE)
    logger.info(
        "Scaling corrected output with range [{lower:.4f}, {upper:.4f}] (scale={scale:.4f})",
        lower=quant_params.lower,
        upper=quant_params.upper,
        scale=scale,
    )

    corrected_store = zarr.open_array(
        output_path,
        mode="w",
        shape=output_shape,
        chunks=chunk_spec,
        dtype=target_dtype,
    )
    float_output_path: Path | None = None
    float_store = None
    if debug:
        float_output_path = _float_store_path(output_path)
        if float_output_path.exists() and not overwrite:
            raise FileExistsError(f"{float_output_path} already exists. Use --overwrite to replace it.")
        float_store = zarr.open_array(
            float_output_path,
            mode="w",
            shape=output_shape,
            chunks=chunk_spec,
            dtype=np.float32,
        )

    for dest_idx, src_idx in plane_mappings:
        corrected = _apply_correction_field_prepared(
            _load_plane(src_idx),
            field_f,
            field_gpu=field_gpu,
        )
        if float_store is not None:
            float_store[dest_idx, :, :, 0] = corrected
        corrected_store[dest_idx, :, :, 0] = _quantize_to_uint16(corrected, quant_params)

    if field_gpu is not None and _cupy_available():
        memory_pool_getter = getattr(cp, "get_default_memory_pool", None)
        if callable(memory_pool_getter):
            try:
                memory_pool_getter().free_all_blocks()
            except Exception:
                logger.opt(exception=True).debug("Unable to release CuPy memory pool; continuing.")

    logger.info("Corrected imagery written to {path}", path=output_path)
    if float_store is not None and float_output_path is not None:
        logger.info("Float32 corrected imagery written to {path}", path=float_output_path)
    quant_meta: dict[str, Any] = {
        "lower_percentile": float(quant_params.lower_percentile),
        "upper_percentile": float(quant_params.upper_percentile),
        "lower": float(quant_params.lower),
        "upper": float(quant_params.upper),
        "observed_min": float(quant_params.observed_min),
        "observed_max": float(quant_params.observed_max),
        "scale": float(scale),
        "dtype": "uint16",
        "sampling": {
            "strategy": "random_without_replacement",
            "max_samples": int(QUANT_MAX_SAMPLES),
            "total_samples": int(quant_params.sample_count),
        },
    }
    if float_output_path is not None:
        quant_meta["float_store"] = str(float_output_path.name)
    _zarr_attrs_write(
        corrected_store.attrs,
        {"axes": "ZYXC", "channel_index": int(channel_index), "quantization": quant_meta},
    )
    if float_store is not None:
        _zarr_attrs_write(float_store.attrs, {"axes": "ZYXC", "channel_index": int(channel_index)})
    return output_path


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

    field_list: list[np.ndarray] = []
    name_list: list[str] = []
    for ch in channels:
        logger.info(
            "Processing channel {channel} (workspace={workspace}, roi={roi}, codebook={codebook})",
            channel=ch,
            workspace=config.workspace,
            roi=config.roi,
            codebook=config.codebook,
        )

        image_yx = _load_channel(
            fused_path,
            channel_index=ch,
            z_index=config.z_index,
        )

        field = compute_correction_field(
            image_yx,
            shrink=config.shrink,
            spline_lowres_px=config.spline_lowres_px,
            iterations=config.iterations,
            threshold=config.threshold,
        )

        # Resolve channel tag from metadata (fallback to index)
        ch_name = names[ch] if names and 0 <= ch < len(names) else f"channel_{ch}"

        # Accumulate fields and names for multi-channel output
        field_list.append(field)
        name_list.append(ch_name)

    # Write stacked multi-channel field once at the end
    stacked = np.stack(field_list, axis=0).astype(np.float32)
    field_output = config.field_output if config.field_output else fields_dir / "n4_correction_field.tif"
    meta_extra_all: dict[str, Any] = {
        "roi": config.roi,
        "codebook": config.codebook,
        "n4": {
            "shrink": int(config.shrink),
            "spline_lowres_px": float(config.spline_lowres_px),
            "iterations": list(config.iterations),
            "z_index": (int(config.z_index) if config.z_index is not None else None),
            "threshold": _summarize_threshold_spec(config.threshold),
        },
    }
    written_field = _write_multi_channel_field(
        stacked,
        name_list,
        field_output,
        overwrite=config.overwrite,
        meta_extra=meta_extra_all,
    )

    # Emit per-channel diagnostic PNGs
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
            },
            output_path=corrected_output,
            debug=config.debug,
        )

    logger.info("Finished N4 workflow for {n} channel(s)", n=len(channels))
    return [N4Result(field_path=written_field, corrected_path=corrected_path)]


def compute_field_from_workspace(config: N4RuntimeConfig) -> N4Result:
    logger.info(
        "Starting N4 correction (workspace={workspace}, roi={roi}, codebook={codebook}, channel={channel})",
        workspace=config.workspace,
        roi=config.roi,
        codebook=config.codebook,
        channel=config.channel,
    )
    fused_path = _resolve_fused_path(config.workspace, config.roi, config.codebook)
    channel_index = config.channel
    image_yx = _load_channel(
        fused_path,
        channel_index=channel_index,
        z_index=config.z_index,
    )

    field = compute_correction_field(
        image_yx,
        shrink=config.shrink,
        spline_lowres_px=config.spline_lowres_px,
        iterations=config.iterations,
        threshold=config.threshold,
    )

    # Write as multi-channel field (C=1) with consistent naming
    names_single = _read_channel_names_from_zarr(fused_path) or [f"channel_{channel_index}"]
    ch_name = names_single[channel_index] if channel_index < len(names_single) else f"channel_{channel_index}"
    fields_dir_single = fused_path.parent / "n4-fields"
    fields_dir_single.mkdir(parents=True, exist_ok=True)
    field_output = config.field_output or (fields_dir_single / "n4_correction_field.tif")
    meta_extra_single: dict[str, Any] = {
        "roi": config.roi,
        "codebook": config.codebook,
        "n4": {
            "shrink": int(config.shrink),
            "spline_lowres_px": float(config.spline_lowres_px),
            "iterations": list(config.iterations),
            "z_index": (int(config.z_index) if config.z_index is not None else None),
            "threshold": _summarize_threshold_spec(config.threshold),
        },
    }
    written_field = _write_multi_channel_field(
        np.expand_dims(field, 0),
        [ch_name],
        field_output,
        overwrite=config.overwrite,
        meta_extra=meta_extra_single,
    )

    corrected_path: Path | None = None
    if config.apply_correction:
        corrected_output = config.corrected_output or (fused_path.parent / "fused_n4.zarr")
        corrected_path = _write_fused_corrected_zyxc(
            fused_path,
            channels=[channel_index],
            fields=[field],
            names=[ch_name],
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
            },
            output_path=corrected_output,
            debug=config.debug,
        )

    return N4Result(field_path=written_field, corrected_path=corrected_path)


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
) -> list[N4Result]:
    """Shared CLI workflow for Click- and Typer-based entrypoints."""

    if z_index is None:
        raise ValueError("--z-index is required (field is computed from one Z plane).")

    logger.info(
        "Invoked CLI with workspace={workspace}, roi={roi}, codebook={codebook}, channels={channels}",
        workspace=workspace,
        roi=roi,
        codebook=codebook,
        channels=channels,
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
    )

    return compute_fields_from_workspace(config)


__all__ = [
    "N4RuntimeConfig",
    "N4Result",
    "DEFAULT_ITERATIONS",
    "compute_correction_field",
    "compute_field_from_workspace",
    "compute_fields_from_workspace",
    "apply_correction_field",
    "apply_correction_to_store",
    "run_cli_workflow",
]
