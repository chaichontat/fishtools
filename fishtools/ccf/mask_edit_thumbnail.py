from __future__ import annotations

from pathlib import Path

import numpy as np
import SimpleITK as sitk
import zarr
from scipy.ndimage import rotate as ndimage_rotate

from fishtools.ccf.landmark import P1Landmarks
from fishtools.ccf.sitk_utils import normalize_robust, resample_sitk_to_spacing, sitk_from_numpy_2d
from fishtools.io.workspace import Workspace


def load_moving_crop_yxc(
    *,
    ws: Workspace,
    roi: str,
    stitch_codebook: str,
    p1: P1Landmarks,
    z_idx: int,
    fused_name: str = "fused.zarr",
) -> tuple[np.ndarray, list[str]]:
    fused_zarr = ws.stitch(roi, stitch_codebook) / str(fused_name)
    if not fused_zarr.exists():
        raise FileNotFoundError(f"Missing {fused_name} at {fused_zarr}")
    arr = zarr.open(str(fused_zarr), mode="r")

    keys = list(arr.attrs.get("key", []))
    keys = [str(k) for k in keys] if keys else [stitch_codebook]

    if z_idx < 0 or z_idx >= int(arr.shape[0]):
        raise ValueError(f"z_idx out of range for fused.zarr: z_idx={z_idx}, shape[0]={int(arr.shape[0])}.")

    sample_slice_full_raw = np.asarray(arr[z_idx, :, :, :], dtype=np.float32)
    if sample_slice_full_raw.ndim != 3:
        raise ValueError(f"Expected fused slice shape (y,x,c), got {sample_slice_full_raw.shape}.")

    if p1.prior_flip_x:
        sample_slice_full_raw = sample_slice_full_raw[:, ::-1, :]
    if p1.prior_rotation_deg != 0:
        rotated: list[np.ndarray] = []
        for c in range(int(sample_slice_full_raw.shape[2])):
            rotated.append(ndimage_rotate(sample_slice_full_raw[:, :, c], int(p1.prior_rotation_deg), reshape=True, order=1))
        sample_slice_full_raw = np.stack(rotated, axis=2)

    sr0, sr1, sc0, sc1 = p1.sample_rotated_crop_bbox
    moving_yxc = sample_slice_full_raw[int(sr0) : int(sr1), int(sc0) : int(sc1), :]
    if moving_yxc.ndim != 3:
        raise ValueError(f"Expected moving crop to be 3D (y,x,c), got shape={moving_yxc.shape}.")

    if len(keys) != int(moving_yxc.shape[2]):
        keys = [f"ch{c}" for c in range(int(moving_yxc.shape[2]))]
    return moving_yxc, keys


def render_mask_edit_thumbnail_rgb01(
    *,
    moving_crop_yxc: np.ndarray,
    sample_voxel_xy_um: float,
    moving_pre_downsample: int,
    target_spacing_um: float,
    channel_indices: tuple[int, ...] | None = None,
    normalize_percentiles: tuple[float, float] = (1.0, 99.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Render the same moving thumbnail used by `ccf export-mask-edit-pack`.

    Returns:
      - rgb01: (Y,X,3) float32 in [0,1] suitable for `matplotlib.pyplot.imsave(..., vmin=0, vmax=1)`
      - ref_yx: (Y,X) float32 resampled channel-0 image (used as the reference grid for mask warping)
    """

    ds = int(moving_pre_downsample)
    if ds <= 0:
        raise ValueError(f"moving_pre_downsample must be > 0, got {ds}.")
    target_um = float(target_spacing_um)
    if not target_um > 0:
        raise ValueError(f"target_spacing_um must be > 0, got {target_um}.")

    img = np.asarray(moving_crop_yxc, dtype=np.float32)
    if img.ndim != 3:
        raise ValueError(f"Expected moving_crop_yxc to be 3D (y,x,c), got shape={img.shape}.")

    if channel_indices is not None:
        if not channel_indices:
            raise ValueError("channel_indices must not be empty when provided.")
        max_idx = int(img.shape[2]) - 1
        for idx in channel_indices:
            if idx < 0 or idx > max_idx:
                raise ValueError(f"channel index {idx} out of range for C={int(img.shape[2])}.")
        img = img[:, :, list(channel_indices)]

    img_ds = img[::ds, ::ds, :].astype(np.float32, copy=False)
    moving_spacing_um = float(sample_voxel_xy_um) * float(ds)

    n_in = int(img_ds.shape[2])
    n_rgb = min(3, n_in)
    if n_rgb <= 0:
        raise ValueError("Fused image has no channels.")

    outs_yx: list[np.ndarray] = []
    norms_yx: list[np.ndarray] = []
    lo, hi = (float(normalize_percentiles[0]), float(normalize_percentiles[1]))
    for c in range(n_rgb):
        moving_sitk = sitk_from_numpy_2d(img_ds[:, :, c], spacing_um=moving_spacing_um)
        moving_out_sitk = resample_sitk_to_spacing(moving_sitk, target_spacing_um=target_um, interp=sitk.sitkLinear)
        out_yx = sitk.GetArrayFromImage(moving_out_sitk).astype(np.float32, copy=False)
        outs_yx.append(out_yx)
        norms_yx.append(normalize_robust(out_yx, lo, hi))

    ref_yx = outs_yx[0]

    if len(norms_yx) == 1:
        rgb01 = np.repeat(norms_yx[0][:, :, None], 3, axis=2)
    elif len(norms_yx) == 2:
        rgb01 = np.stack([norms_yx[0], norms_yx[1], np.zeros_like(norms_yx[0])], axis=2)
    else:
        rgb01 = np.stack(norms_yx[:3], axis=2)

    if rgb01.ndim != 3 or rgb01.shape[2] != 3:
        raise ValueError(f"Expected RGB output, got shape={rgb01.shape}.")
    if ref_yx.shape != rgb01.shape[:2]:
        raise ValueError(f"Output shape mismatch: ref={ref_yx.shape} vs rgb={rgb01.shape[:2]}.")
    return (rgb01.astype(np.float32, copy=False), ref_yx.astype(np.float32, copy=False))


def save_mask_edit_thumbnail_png(*, path: Path, rgb01: np.ndarray) -> None:
    # Use matplotlib's imsave (same behavior as the current CLI) without relying on an interactive backend.
    import matplotlib.image as mpimg

    path.parent.mkdir(parents=True, exist_ok=True)
    mpimg.imsave(path.as_posix(), rgb01, vmin=0.0, vmax=1.0)
