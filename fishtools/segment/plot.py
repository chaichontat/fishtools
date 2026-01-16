from __future__ import annotations

from pathlib import Path


def export_zslices_with_boundaries(
    image_path: Path,
    mask_path: Path,
    output_dir: Path,
    *,
    channel: int = 0,
    subsample: int = 2,
    boundary_color: tuple[int, int, int] = (255, 255, 255),
    z_step: int = 8,
    z_start: int = 0,
    z_end: int | None = None,
    cmap_name: str = "magma",
    p_low: float = 1.0,
    p_high: float = 99.99,
    overwrite: bool = False,
) -> None:
    """Export PNGs for selected z-slices with and without segmentation boundaries.

    Writes two images per z-slice:
      - <output_dir>/z###_nomask.png
      - <output_dir>/z###_mask.png

    Output dimensions follow strided slicing (`::subsample`).
    """
    import numpy as np
    import zarr
    from matplotlib import colormaps
    from PIL import Image
    from skimage.segmentation import find_boundaries

    if subsample < 1:
        raise ValueError(f"subsample must be >= 1, got {subsample}.")
    if z_step < 1:
        raise ValueError(f"z_step must be >= 1, got {z_step}.")
    if channel < 0:
        raise ValueError(f"channel must be >= 0, got {channel}.")
    if p_high <= p_low:
        raise ValueError(f"p_high must be greater than p_low (got p_low={p_low}, p_high={p_high}).")

    img_arr = zarr.open_array(str(image_path), mode="r")
    mask_arr = zarr.open_array(str(mask_path), mode="r")

    if img_arr.ndim not in (3, 4):
        raise ValueError(f"Expected image zarr to be 3D (Z,Y,X) or 4D (Z,Y,X,C), got {img_arr.shape}")
    if mask_arr.ndim != 3:
        raise ValueError(f"Expected mask zarr to be 3D (Z,Y,X), got {mask_arr.shape}")

    if img_arr.shape[0] != mask_arr.shape[0]:
        raise ValueError(
            f"Z mismatch between image ({img_arr.shape[0]}) and mask ({mask_arr.shape[0]}): "
            f"{image_path} vs {mask_path}"
        )
    if img_arr.shape[1] != mask_arr.shape[1] or img_arr.shape[2] != mask_arr.shape[2]:
        raise ValueError(
            f"XY mismatch between image ({img_arr.shape[1:3]}) and mask ({mask_arr.shape[1:3]}): "
            f"{image_path} vs {mask_path}"
        )

    if img_arr.ndim == 3:
        if channel != 0:
            raise ValueError("Image zarr has no channel axis (Z,Y,X); channel must be 0.")
    else:
        num_channels = img_arr.shape[3]
        if channel >= num_channels:
            raise ValueError(f"channel {channel} out of bounds for image with {num_channels} channels.")

    num_slices = img_arr.shape[0]
    start = max(0, z_start)
    end = min(num_slices, z_end if z_end is not None else num_slices)
    if start >= end:
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    cmap = colormaps[cmap_name]

    from fishtools.utils.pretty_print import progress_bar

    n_slices = (end - start + z_step - 1) // z_step
    with progress_bar(n_slices) as progress:
        for z in range(start, end, z_step):
            raw_path = output_dir / f"z{z:03d}_nomask.png"
            overlay_path = output_dir / f"z{z:03d}_mask.png"
            need_raw = overwrite or not raw_path.exists()
            need_overlay = overwrite or not overlay_path.exists()
            if not (need_raw or need_overlay):
                progress()
                continue

            if img_arr.ndim == 3:
                img_slice = np.asarray(img_arr[z, ::subsample, ::subsample])
            else:
                img_slice = np.asarray(img_arr[z, ::subsample, ::subsample, channel])

            p1, p99 = np.percentile(img_slice, [p_low, p_high])
            if p99 > p1:
                img_norm = np.clip((img_slice - p1) / (p99 - p1), 0, 1)
            else:
                img_norm = np.zeros_like(img_slice, dtype=np.float32)

            rgb = (cmap(img_norm)[:, :, :3] * 255).astype(np.uint8)

            if need_raw:
                Image.fromarray(rgb).save(raw_path)

            if need_overlay:
                mask_slice = np.asarray(mask_arr[z, ::subsample, ::subsample])
                boundaries = find_boundaries(mask_slice, mode="outer")
                rgb_overlay = rgb.copy()
                rgb_overlay[boundaries] = boundary_color
                Image.fromarray(rgb_overlay).save(overlay_path)

            progress()
