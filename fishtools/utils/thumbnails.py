from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


@dataclass(frozen=True, slots=True)
class ThumbnailOptions:
    """Configuration for thumbnail downsampling/normalization."""

    z_stride: int = 8
    xy_downsample: int = 8
    shift_bits: int = 10
    percentiles: tuple[float, float] | None = None

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "ThumbnailOptions":
        z_stride = int(payload.get("z_stride", cls.z_stride))
        xy_downsample = int(payload.get("xy_downsample", cls.xy_downsample))
        shift_bits = int(payload.get("shift_bits", cls.shift_bits))
        percentiles = payload.get("percentiles")
        percs: tuple[float, float] | None = None
        if percentiles is not None:
            if not isinstance(percentiles, (list, tuple)) or len(percentiles) != 2:
                raise ValueError("percentiles must be a 2-item list/tuple")
            percs = (float(percentiles[0]), float(percentiles[1]))
        if z_stride <= 0:
            raise ValueError("z_stride must be a positive integer")
        if xy_downsample <= 0:
            raise ValueError("xy_downsample must be a positive integer")
        if shift_bits < 0:
            raise ValueError("shift_bits must be >= 0")
        if percs is not None:
            lo, hi = percs
            if not (0.0 <= lo < hi <= 100.0):
                raise ValueError("percentiles must satisfy 0 <= low < high <= 100")
        return cls(
            z_stride=z_stride,
            xy_downsample=xy_downsample,
            shift_bits=shift_bits,
            percentiles=percs,
        )


def load_thumbnail_options(path: Path | None) -> ThumbnailOptions:
    if path is None:
        return ThumbnailOptions()
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("options JSON must be an object")
    return ThumbnailOptions.from_mapping(payload)


def _normalize_to_uint8(data: np.ndarray, options: ThumbnailOptions) -> np.ndarray:
    if options.percentiles is not None:
        lo, hi = options.percentiles
        p_lo, p_hi = np.percentile(data, [lo, hi])
        if not np.isfinite(p_lo) or not np.isfinite(p_hi) or p_hi <= p_lo:
            p_lo, p_hi = 0.0, 1.0
        scaled = (data.astype(np.float32) - float(p_lo)) / (float(p_hi) - float(p_lo))
        return (np.clip(scaled, 0.0, 1.0) * 255.0).astype(np.uint8)
    if not np.issubdtype(data.dtype, np.integer):
        data = np.clip(data, 0, 255).astype(np.uint8)
        return data
    return (data >> options.shift_bits).astype(np.uint8)


def save_thumbnail_png(
    thumbnail_data: np.ndarray,
    output_path: Path,
    *,
    options: ThumbnailOptions | None = None,
    lowhigh: np.ndarray | None = None,
) -> None:
    """Save a downsampled RGB thumbnail."""
    rgb = thumbnail_rgb(thumbnail_data, options=options, lowhigh=lowhigh)
    thumbnail_img = Image.fromarray(rgb, mode="RGB")
    thumbnail_img.save(output_path)


def thumbnail_rgb(
    thumbnail_data: np.ndarray,
    *,
    options: ThumbnailOptions | None = None,
    lowhigh: np.ndarray | None = None,
) -> np.ndarray:
    """Render a thumbnail as an RGB uint8 array."""
    opts = options or ThumbnailOptions()
    td = thumbnail_data[:: opts.xy_downsample, :: opts.xy_downsample]
    if lowhigh is not None:
        td = _normalize_to_uint8_by_lowhigh(td, lowhigh)
    else:
        td = _normalize_to_uint8(td, opts)
    if td.dtype != np.uint8:
        td = td.astype(np.uint8)
    if td.ndim == 2:
        td = np.repeat(td[:, :, None], 3, axis=2)
    elif td.shape[2] == 1:
        td = np.repeat(td, 3, axis=2)
    elif td.shape[2] == 2:
        td = np.concatenate([td, np.zeros_like(td[:, :, :1])], axis=2)
    return td


def _normalize_to_uint8_by_lowhigh(data: np.ndarray, lowhigh: np.ndarray) -> np.ndarray:
    if data.ndim == 2:
        data = data[:, :, None]
    if data.ndim != 3:
        raise ValueError(f"Expected YX or YXC array, got shape={data.shape}")

    if lowhigh.ndim != 2 or lowhigh.shape[1] != 2 or lowhigh.shape[0] < data.shape[2]:
        raise ValueError(f"lowhigh must be shaped (C,2) with C>=data channels; got shape={lowhigh.shape}")

    out = np.empty(data.shape, dtype=np.uint8)
    data_f = data.astype(np.float32, copy=False)
    for ch in range(data.shape[2]):
        lo = float(lowhigh[ch, 0])
        hi = float(lowhigh[ch, 1])
        if hi <= lo:
            out[:, :, ch] = 0
            continue
        scaled = (data_f[:, :, ch] - lo) / (hi - lo)
        out[:, :, ch] = (np.clip(scaled, 0.0, 1.0) * 255.0).astype(np.uint8)
    return out
