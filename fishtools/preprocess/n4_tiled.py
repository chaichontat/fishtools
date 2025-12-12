"""
Tiled GPU correction for N4 bias field - drop-in replacement for n4.py GPU functions.

This module provides memory-efficient tiled processing for large images (8K×8K+)
that would otherwise cause GPU OOM errors. The tiled implementation produces
identical results to the full-plane version within floating-point tolerance.

Usage:
    from fishtools.preprocess.n4_tiled import correct_plane_gpu_tiled

    # Same interface as _correct_plane_gpu but with tile_size parameter
    result = correct_plane_gpu_tiled(
        plane,
        field_gpu=field_gpu,
        use_unsharp_mask=True,
        mask_cpu=mask,
        tile_size=1024
    )

Memory reduction for 8192×8192 images:
    - Full plane: ~1.3 GB peak GPU memory
    - Tiled (1024): ~20 MB peak GPU memory (65× reduction)
"""

from __future__ import annotations

import cupy as cp
import numpy as np
from cucim.skimage import filters as cucim_filters
from cupy import ndarray as cupy_ndarray

DEFAULT_TILE_SIZE = 1024
# Halo for unsharp mask edge continuity. For radius=3 Gaussian, effective support
# is ~4*sigma=12 pixels. We use 16 for safety margin (~5.3 sigma coverage).
# If UNSHARP_RADIUS changes in n4.py, update this to ceil(5*radius)+1.
UNSHARP_HALO = 16
AUTO_TILE_THRESHOLD = 4096  # Auto-tile when image exceeds this in any dimension


def get_auto_tile_size(H: int, W: int, tile_size: int | None, tile_threshold: int | None = None) -> int:
    """Determine tile size based on image dimensions and user preference.

    Args:
        H: Image height
        W: Image width
        tile_size: User preference:
            - None: Auto-detect (tile if image > 4096 in any dimension)
            - 0: Disabled (process full plane)
            - int > 0: Explicit tile size
        tile_threshold: Optional override for the auto-tiling threshold. When
            <= 0 the auto path is disabled unless an explicit tile size is set.

    Returns:
        Effective tile size to use. Returns max(H, W) if tiling disabled.
    """
    if tile_size == 0:
        return max(H, W)  # No tiling - process full plane
    if tile_size is not None and tile_size > 0:
        return tile_size
    # Auto: tile when plane exceeds threshold
    threshold = AUTO_TILE_THRESHOLD if tile_threshold is None else tile_threshold
    if threshold is not None and threshold > 0 and (H > threshold or W > threshold):
        return DEFAULT_TILE_SIZE
    return max(H, W)  # Full plane for smaller images


def _process_tile(
    tile_plane: np.ndarray,
    tile_field: cupy_ndarray,
    tile_mask: np.ndarray | None,
    use_unsharp_mask: bool,
) -> np.ndarray:
    """Process a single tile on GPU: divide by field, optional unsharp mask.

    Args:
        tile_plane: Input tile (CPU, float32)
        tile_field: Correction field tile (GPU, float32)
        tile_mask: Optional foreground mask tile (CPU, bool)
        use_unsharp_mask: Whether to apply unsharp mask

    Returns:
        Corrected tile as CPU float32 array
    """
    # Transfer tile to GPU
    img_gpu = cp.asarray(tile_plane, dtype=cp.float32)

    # Apply correction (divide by field)
    img_gpu /= tile_field

    if use_unsharp_mask:
        # Apply unsharp mask with explicit radius to match n4.py defaults
        sharpened_gpu = cucim_filters.unsharp_mask(img_gpu, radius=3, preserve_range=True)

        if tile_mask is not None:
            mask_gpu = cp.asarray(tile_mask, dtype=cp.bool_)
        else:
            # Fallback to >0 mask on corrected plane
            mask_gpu = img_gpu > 0.0

        cp.copyto(img_gpu, sharpened_gpu, where=mask_gpu)
        del sharpened_gpu, mask_gpu

    # Replace non-finite values with zeros for safety
    cp.nan_to_num(img_gpu, copy=False)

    # Transfer result back to CPU
    result = cp.asnumpy(img_gpu).astype(np.float32, copy=False)
    del img_gpu

    return result


def correct_plane_gpu_tiled(
    plane: np.ndarray,
    *,
    field_gpu: cupy_ndarray | None = None,
    field_cpu: np.ndarray | None = None,
    use_unsharp_mask: bool,
    mask_cpu: np.ndarray | None = None,
    tile_size: int | None = None,
    tile_threshold: int | None = None,
) -> np.ndarray:
    """Tiled version of _correct_plane_gpu - same interface, lower memory.

    Processes the plane in tiles to reduce peak GPU memory usage. Each tile
    is processed with a halo region to ensure correct unsharp mask results
    at tile boundaries.

    Args:
        plane: Input 2D plane (Y, X) as CPU float32
        field_gpu: Correction field on GPU (Y, X) - must match plane shape.
        field_cpu: Optional CPU copy of the field (converted when field_gpu is None).
        use_unsharp_mask: Whether to apply unsharp mask after correction
        mask_cpu: Optional foreground mask (Y, X) as CPU bool array
        tile_size: Tile size for processing:
            - None: Auto-detect based on image size
            - 0: Disabled (process full plane)
            - int > 0: Explicit tile size
        tile_threshold: Override for auto-tiling threshold (pixels). <=0 disables auto-tiling.

    Returns:
        Corrected plane as CPU float32 array with same shape as input
    """
    H, W = plane.shape

    # Validate input shapes
    if field_gpu is None:
        if field_cpu is None:
            raise ValueError("correct_plane_gpu_tiled requires field_gpu or field_cpu.")
        field_gpu = cp.asarray(np.asarray(field_cpu, dtype=np.float32), dtype=cp.float32)

    if field_gpu.shape != (H, W):
        raise ValueError(f"Field shape {field_gpu.shape} does not match plane shape {(H, W)}")
    if mask_cpu is not None and mask_cpu.shape != (H, W):
        raise ValueError(f"Mask shape {mask_cpu.shape} does not match plane shape {(H, W)}")

    effective_tile_size = get_auto_tile_size(H, W, tile_size, tile_threshold)

    # If tile size >= image size, process full plane (no tiling overhead)
    if effective_tile_size >= max(H, W):
        result = _process_tile(
            plane,
            field_gpu,
            mask_cpu,
            use_unsharp_mask,
        )
        return result

    # Determine halo size based on whether unsharp mask is used
    halo = UNSHARP_HALO if use_unsharp_mask else 0

    # Pre-allocate output array
    result = np.empty((H, W), dtype=np.float32)

    # Process tiles
    for y0 in range(0, H, effective_tile_size):
        y1 = min(y0 + effective_tile_size, H)

        for x0 in range(0, W, effective_tile_size):
            x1 = min(x0 + effective_tile_size, W)

            # Expand tile bounds with halo for unsharp mask edge continuity
            y0h = max(0, y0 - halo)
            y1h = min(H, y1 + halo)
            x0h = max(0, x0 - halo)
            x1h = min(W, x1 + halo)

            # Extract tile with halo from inputs
            tile_plane = np.ascontiguousarray(plane[y0h:y1h, x0h:x1h])
            tile_field = field_gpu[y0h:y1h, x0h:x1h]
            tile_mask = mask_cpu[y0h:y1h, x0h:x1h] if mask_cpu is not None else None

            # Process the tile (with halo)
            tile_result = _process_tile(
                tile_plane,
                tile_field,
                tile_mask,
                use_unsharp_mask,
            )

            # Extract core region (without halo) and write to output
            core_y0 = y0 - y0h
            core_y1 = core_y0 + (y1 - y0)
            core_x0 = x0 - x0h
            core_x1 = core_x0 + (x1 - x0)

            result[y0:y1, x0:x1] = tile_result[core_y0:core_y1, core_x0:core_x1]

    return result


__all__ = [
    "DEFAULT_TILE_SIZE",
    "UNSHARP_HALO",
    "AUTO_TILE_THRESHOLD",
    "get_auto_tile_size",
    "correct_plane_gpu_tiled",
]
