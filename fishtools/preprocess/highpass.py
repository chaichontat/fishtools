from __future__ import annotations

import math
import shutil
from pathlib import Path

import cupy as cp
import numpy as np
import zarr
from loguru import logger

from fishtools.utils.pretty_print import progress_bar
from fishtools.utils.zarr_utils import create_zarr_array


def _cupy_available() -> bool:
    cuda = getattr(cp, "cuda", None)
    runtime = getattr(cuda, "runtime", None)
    get_device_count = getattr(runtime, "getDeviceCount", None)
    if not callable(get_device_count):
        return False
    return int(get_device_count()) > 0


def _quantize_to_uint16(data: np.ndarray) -> np.ndarray:
    """Round-to-nearest and clamp float data into uint16."""
    arr = np.asarray(data, dtype=np.float32)
    np.nan_to_num(arr, copy=False)
    np.clip(arr, 0.0, float(np.iinfo(np.uint16).max), out=arr)
    return np.floor(arr + 0.5).astype(np.uint16)


def _quantize_to_uint16_gpu(data: cp.ndarray) -> cp.ndarray:
    arr = cp.asarray(data, dtype=cp.float32)
    cp.nan_to_num(arr, copy=False)
    arr = cp.clip(arr, 0.0, float(np.iinfo(np.uint16).max))
    return cp.floor(arr + 0.5).astype(cp.uint16)


def _pick_input_zarr(stitch_root: Path) -> Path:
    for candidate in (stitch_root / "fused_n4.zarr", stitch_root / "fused.zarr"):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find fused zarr under {stitch_root} (expected fused_n4.zarr or fused.zarr)."
    )


def _copy_zarr_attrs(
    dst_attrs: zarr.core.attributes.Attributes,
    src_attrs: zarr.core.attributes.Attributes,
) -> None:
    for key in src_attrs:
        try:
            dst_attrs[key] = src_attrs[key]
        except TypeError:
            logger.warning(f"Skipping non-JSON zarr attribute {key!r}")


def _choose_xy_step(
    *,
    dim_y: int,
    dim_x: int,
    dim_z: int,
    base_y: int,
    base_x: int,
    pad_xy: int,
) -> tuple[int, int]:
    # Aim for larger blocks (GPU throughput) while keeping temporary float32 working-set bounded.
    # Working-set is roughly Z * (Y+2P) * (X+2P) * 4 bytes (ignoring intermediates).
    target_bytes = 256 * 1024 * 1024
    denom = max(1, int(dim_z) * 4)
    max_side_total = int(math.sqrt(target_bytes / denom))
    max_center = max(64, max_side_total - 2 * int(pad_xy))

    y = min(dim_y, max(base_y, max_center))
    x = min(dim_x, max(base_x, max_center))
    return int(y), int(x)


def run_highpass_workflow(
    *,
    stitch_root: Path,
    sigma_px: float = 20.0,
    anisotropy: float = 2.0,
    overwrite: bool = False,
) -> Path:
    """Compute a Gaussian high-pass filtered fused zarr and write `fused_highpassed.zarr` as uint16."""
    fused_path = _pick_input_zarr(stitch_root)
    logger.info(f"Using fused source zarr at {fused_path}")

    dest_path = fused_path.parent / "fused_highpassed.zarr"
    partial_dest_path = dest_path.with_name(f"{dest_path.name}.partial")

    if dest_path.exists() and not overwrite:
        logger.info(f"Skipping; output already exists at {dest_path} (use --overwrite to replace).")
        return dest_path

    if partial_dest_path.exists():
        shutil.rmtree(partial_dest_path, ignore_errors=True)

    src = zarr.open_array(fused_path, mode="r")
    if src.ndim != 4:
        raise ValueError(f"Expected fused zarr to be 4D (ZYXC); got ndim={src.ndim}, shape={src.shape}")

    z_dim, y_dim, x_dim, c_dim = src.shape
    if c_dim <= 0:
        raise ValueError("Input fused zarr reports zero channels.")

    if src.chunks is not None and len(src.chunks) == 4:
        zc, yc, xc, _cc = src.chunks
        dest_chunks = (zc, yc, xc, 1)
        base_y = int(yc)
        base_x = int(xc)
    else:
        dest_chunks = (1, min(2048, y_dim), min(2048, x_dim), 1)
        base_y = min(1024, y_dim)
        base_x = min(1024, x_dim)

    sigma_px_f = float(sigma_px)
    anisotropy_f = float(anisotropy)
    pad_xy = int(math.ceil(4.0 * sigma_px_f))

    dest = create_zarr_array(
        partial_dest_path,
        shape=(z_dim, y_dim, x_dim, c_dim),
        chunks=dest_chunks,
        dtype=np.uint16,
        overwrite=True,
    )

    _copy_zarr_attrs(dest.attrs, src.attrs)
    dest.attrs["axes"] = "ZYXC"
    dest.attrs["highpass"] = {
        "sigma_px": sigma_px_f,
        "anisotropy": anisotropy_f,
        "quantization": {
            "mode": "round_uint16",
            "clamp": [0, int(np.iinfo(np.uint16).max)],
        },
        "input": fused_path.name,
        "output": dest_path.name,
        "mode": "gaussian_highpass",
    }

    use_gpu = _cupy_available()
    y_step, x_step = base_y, base_x
    if use_gpu:
        y_step, x_step = _choose_xy_step(
            dim_y=y_dim,
            dim_x=x_dim,
            dim_z=z_dim,
            base_y=base_y,
            base_x=base_x,
            pad_xy=pad_xy,
        )
        logger.info(f"Highpass: using CuCIM gaussian (tile={y_step}x{x_step}, pad={pad_xy})")
    else:
        logger.info(f"Highpass: using SciPy gaussian_filter (tile={y_step}x{x_step}, pad={pad_xy})")

    logger.info(
        f"Computing highpass (sigma_px={sigma_px_f}, anisotropy={anisotropy_f}, pad_xy={pad_xy}) "
        f"for shape={src.shape}"
    )

    if use_gpu:
        from cucim.skimage import filters as cucim_filters  # type: ignore[import-not-found]
    else:
        from scipy.ndimage import gaussian_filter

    sigma_z = sigma_px_f / anisotropy_f if z_dim > 1 else 0.0
    sigma = (sigma_z, sigma_px_f, sigma_px_f)

    total_tiles = c_dim * max(1, math.ceil(y_dim / y_step) * math.ceil(x_dim / x_step))
    with progress_bar(total_tiles) as advance:
        for ch in range(c_dim):
            for y0 in range(0, y_dim, y_step):
                y1 = min(y_dim, y0 + y_step)
                y0p = max(0, y0 - pad_xy)
                y1p = min(y_dim, y1 + pad_xy)
                yin0 = y0 - y0p
                yin1 = yin0 + (y1 - y0)

                for x0 in range(0, x_dim, x_step):
                    x1 = min(x_dim, x0 + x_step)
                    x0p = max(0, x0 - pad_xy)
                    x1p = min(x_dim, x1 + pad_xy)
                    xin0 = x0 - x0p
                    xin1 = xin0 + (x1 - x0)

                    block = np.asarray(src[:, y0p:y1p, x0p:x1p, ch], dtype=np.float32)
                    if use_gpu:
                        block_gpu = cp.asarray(block, dtype=cp.float32)
                        low_gpu = cucim_filters.gaussian(block_gpu, sigma=sigma, mode="reflect", preserve_range=True)
                        hp_gpu = block_gpu - low_gpu
                        hp_gpu = cp.maximum(hp_gpu, 0.0)
                        center_gpu = hp_gpu[:, yin0:yin1, xin0:xin1]
                        u16_gpu = _quantize_to_uint16_gpu(center_gpu)
                        dest[:, y0:y1, x0:x1, ch] = cp.asnumpy(u16_gpu)
                        del block_gpu, low_gpu, hp_gpu, center_gpu, u16_gpu
                    else:
                        low = gaussian_filter(block, sigma=sigma, mode="reflect")
                        hp = block - low
                        np.maximum(hp, 0.0, out=hp)
                        center = hp[:, yin0:yin1, xin0:xin1]
                        u16 = _quantize_to_uint16(center)
                        dest[:, y0:y1, x0:x1, ch] = u16
                    advance()

    try:
        if dest_path.exists():
            if dest_path.is_dir():
                shutil.rmtree(dest_path)
            else:
                dest_path.unlink()
        partial_dest_path.replace(dest_path)
    except Exception:
        shutil.rmtree(partial_dest_path, ignore_errors=True)
        raise

    thumbnail_dir = dest_path.parent / "thumbnails"
    thumbnail_dir.mkdir(exist_ok=True)
    from fishtools.utils.thumbnails import ThumbnailOptions, save_thumbnail_png

    thumb_options = ThumbnailOptions()
    preview_c = min(3, c_dim)
    readback = zarr.open_array(dest_path, mode="r")
    write_options = ThumbnailOptions(
        z_stride=thumb_options.z_stride,
        xy_downsample=1,
        shift_bits=thumb_options.shift_bits,
        percentiles=thumb_options.percentiles,
    )
    for zi in range(0, z_dim, thumb_options.z_stride):
        ds = thumb_options.xy_downsample
        thumb = np.asarray(readback[zi, ::ds, ::ds, :preview_c], dtype=np.uint16)
        thumb_path = thumbnail_dir / f"thumbnail_highpass_z{zi:03d}.png"
        save_thumbnail_png(thumb, thumb_path, options=write_options)

    logger.info(f"Wrote highpassed fused store to {dest_path} (shape={dest.shape}, dtype=uint16)")
    return dest_path
