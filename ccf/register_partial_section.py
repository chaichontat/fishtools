# %% [markdown]
# # Partial Section Registration to DevCCF
#
# Hybrid pipeline for registering partial LSFM sections to atlas:
#
# | Phase | Method | Purpose |
# |-------|--------|---------|
# | 1 | Landmark Similarity2D | Solve orientation ambiguity |
# | 2 | B-spline (optional) | Gentle local deformation |
#
# Run cells sequentially. Each phase saves outputs for inspection.

# %%
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import zarr
from brainglobe_atlasapi import BrainGlobeAtlas
from IPython import get_ipython
from scipy.ndimage import rotate as ndimage_rotate

from fishtools.ccf.landmark import LandmarkRegistrationOutputs, P1Landmarks
from fishtools.ccf.landmark_ui import pick_atlas_slice_idx, pick_paired_landmarks, pick_rotation_deg
from fishtools.ccf.sitk_utils import compute_similarity2d_from_landmarks, pixels_to_physical_um
from fishtools.io.workspace import Workspace
from fishtools.preprocess.config import NumpyEncoder

# Use widget backend for VS Code interactive mode
ip = get_ipython()
if ip is not None:
    ip.run_line_magic("matplotlib", "widget")

# %%
# === EDIT THESE ===

# Workspace configuration
WORKSPACE = Path("/working/20250929_JaxA3_Coro4")
ROI = "3"
STITCH_CODEBOOK = "pi"  # analysis/deconv/stitch--{ROI}+{STITCH_CODEBOOK}/fused.zarr

# Atlas configuration
ATLAS_NAME = "kim_dev_mouse_e15-5_lsfm_20um"

# Sample configuration (resolved via Workspace)
ws = Workspace(WORKSPACE)
SAMPLE_ZARR = ws.stitch(ROI, STITCH_CODEBOOK) / "fused.zarr"
SAMPLE_Z_IDX = 5
SAMPLE_CHANNEL = "pi"

# Voxel sizes (µm)
SAMPLE_VOXEL_XY = 0.216
ATLAS_VOXEL = 20.0

# Output directory (analysis/output/ccf-transforms/{ROI}/)
OUTDIR = ws.ccf_transforms(ROI)
OUTDIR.mkdir(parents=True, exist_ok=True)

OUT = LandmarkRegistrationOutputs(OUTDIR)
P1_TFM_PATH = OUT.p1_similarity_tfm

# Preview downsample factor for interactive elements (sample only)
PREVIEW_DOWNSAMPLE = 8

# %% [markdown]
# ## Phase 0: Load Data

# %%
atlas = BrainGlobeAtlas(ATLAS_NAME)

arr = zarr.open(str(SAMPLE_ZARR), mode="r")
keys = list(arr.attrs.get("key", []))
ch_idx = keys.index(SAMPLE_CHANNEL) if SAMPLE_CHANNEL in keys else 0
sample_slice_full_raw = np.asarray(arr[SAMPLE_Z_IDX, :, :, ch_idx])

# Mask atlas to brain only (non-zero annotations) and crop
def crop_to_content(img: np.ndarray, mask: np.ndarray, pad: int = 10) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Crop image to bounding box of mask with padding."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        raise ValueError("Mask is empty; pick a different atlas slice (annotation is all zeros).")
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    # Add padding
    rmin = max(0, rmin - pad)
    rmax = min(img.shape[0], rmax + pad + 1)
    cmin = max(0, cmin - pad)
    cmax = min(img.shape[1], cmax + pad + 1)

    return img[rmin:rmax, cmin:cmax], (rmin, rmax, cmin, cmax)


# Crop sample to non-zero content (preview crop for rotation slider only)
sample_mask_raw = sample_slice_full_raw > np.percentile(sample_slice_full_raw, 5)  # threshold above noise
sample_slice, _sample_crop_bbox = crop_to_content(sample_slice_full_raw, sample_mask_raw, pad=50)


# %% [markdown]
# ### Interactive Rotation (find approximate angle first)
#
# Use the slider to rotate the sample to a sensible orientation (upright-ish),
# then run the next cell.

# %%
# Downsample sample only for fast rotation preview (rotation UI is always shown).
ROTATION_PREVIEW_DOWNSAMPLE = PREVIEW_DOWNSAMPLE * 2
sample_preview = sample_slice[::ROTATION_PREVIEW_DOWNSAMPLE, ::ROTATION_PREVIEW_DOWNSAMPLE]

existing_p1 = OUT.try_read_p1_landmarks()
initial_rotation_deg = existing_p1.prior_rotation_deg if existing_p1 is not None else 0

rotation_picker = pick_rotation_deg(
    moving_image_yx=sample_preview,
    initial_deg=initial_rotation_deg,
)

# %%
PRIOR_ROTATION_DEG = rotation_picker.deg
print(f"Selected rotation: {PRIOR_ROTATION_DEG}°")

# Rotate the FULL sample (raw), then crop for landmark selection.
if PRIOR_ROTATION_DEG != 0:
    sample_slice_full = ndimage_rotate(sample_slice_full_raw, PRIOR_ROTATION_DEG, reshape=True, order=1)
else:
    sample_slice_full = sample_slice_full_raw

sample_crop_bbox_rot: tuple[int, int, int, int]
if (
    existing_p1 is not None
    and existing_p1.prior_rotation_deg == PRIOR_ROTATION_DEG
    and existing_p1.sample_rotated_full_shape_yx == tuple(int(x) for x in sample_slice_full.shape)
):
    sample_crop_bbox_rot = existing_p1.sample_rotated_crop_bbox
    sr0, sr1, sc0, sc1 = sample_crop_bbox_rot
    sample_slice_rotated = sample_slice_full[sr0:sr1, sc0:sc1]
else:
    sample_mask = sample_slice_full > np.percentile(sample_slice_full, 5)  # threshold above noise
    sample_slice_rotated, sample_crop_bbox_rot = crop_to_content(sample_slice_full, sample_mask, pad=50)

# Crop offset in the ROTATED full-image coordinate system (x_offset, y_offset)
SAMPLE_CROP_OFFSET = (sample_crop_bbox_rot[2], sample_crop_bbox_rot[0])

# Pick atlas Z after rotation (use rotated sample as the fixed reference).
ATLAS_Z_PREVIEW_DOWNSAMPLE = PREVIEW_DOWNSAMPLE * 2
atlas_z_sample_preview = sample_slice_rotated[::ATLAS_Z_PREVIEW_DOWNSAMPLE, ::ATLAS_Z_PREVIEW_DOWNSAMPLE]

atlas_slice_picker = pick_atlas_slice_idx(
    atlas_reference_zyx=atlas.reference,
    moving_image_yx=atlas_z_sample_preview,
    initial_idx=(existing_p1.atlas_slice_idx if existing_p1 is not None and existing_p1.atlas_slice_idx is not None else 0),
    z_min_idx=0,
    z_max_idx=320,
)

# %%
atlas_slice_idx = atlas_slice_picker.idx
print(f"Selected atlas_slice_idx={atlas_slice_idx}")

atlas_slice_full = atlas.reference[atlas_slice_idx, :, :]
atlas_annotation_full = atlas.annotation[atlas_slice_idx, :, :]

atlas_brain_mask = atlas_annotation_full > 0

atlas_slice_masked = atlas_slice_full.copy()
atlas_slice_masked[~atlas_brain_mask] = 0

atlas_slice, atlas_crop_bbox = crop_to_content(atlas_slice_masked, atlas_brain_mask, pad=5)

ATLAS_CROP_OFFSET = (atlas_crop_bbox[2], atlas_crop_bbox[0])  # (x_offset, y_offset)

# %% [markdown]
# ### Interactive Landmark Selection
#
# Click paired landmarks (atlas first, then sample). The UI enforces pairing and saves
# `p1_landmarks.json` as you go.

# %%
def _save_p1_landmarks(
    *,
    fixed_points_cropped_xy: list[tuple[float, float]],
    moving_points_fullres_xy_in_rotated_crop: list[tuple[float, float]],
) -> None:
    OUT.write_p1_landmarks(
        P1Landmarks(
            prior_rotation_deg=PRIOR_ROTATION_DEG,
            atlas_slice_idx=atlas_slice_idx,
            fixed_points_cropped_xy=fixed_points_cropped_xy,
            moving_points_fullres_xy_in_rotated_crop=moving_points_fullres_xy_in_rotated_crop,
            atlas_crop_bbox=atlas_crop_bbox,
            sample_rotated_crop_bbox=sample_crop_bbox_rot,
            preview_downsample=PREVIEW_DOWNSAMPLE,
            atlas_full_shape_yx=tuple(int(x) for x in atlas_slice_full.shape),
            sample_rotated_full_shape_yx=tuple(int(x) for x in sample_slice_full.shape),
        ),
        encoder=NumpyEncoder,
    )

sample_preview_rotated = sample_slice_rotated[::PREVIEW_DOWNSAMPLE, ::PREVIEW_DOWNSAMPLE]

initial_fixed: list[tuple[float, float]] = []
initial_moving: list[tuple[float, float]] = []
if (
    existing_p1 is not None
    and existing_p1.prior_rotation_deg == PRIOR_ROTATION_DEG
    and existing_p1.atlas_crop_bbox == atlas_crop_bbox
    and existing_p1.sample_rotated_crop_bbox == sample_crop_bbox_rot
    and existing_p1.preview_downsample == PREVIEW_DOWNSAMPLE
):
    initial_fixed = existing_p1.fixed_points_cropped_xy
    initial_moving = existing_p1.moving_points_fullres_xy_in_rotated_crop
    print(
        f"Loaded existing landmarks from {OUT.p1_landmarks_json}: "
        f"pairs={len(initial_fixed)} (rotation={existing_p1.prior_rotation_deg}°)"
    )


def _on_landmarks_change(
    fixed_points_cropped_xy: list[tuple[float, float]],
    moving_points_fullres_xy_in_rotated_crop: list[tuple[float, float]],
) -> None:
    _save_p1_landmarks(
        fixed_points_cropped_xy=fixed_points_cropped_xy,
        moving_points_fullres_xy_in_rotated_crop=moving_points_fullres_xy_in_rotated_crop,
    )


landmark_picker = pick_paired_landmarks(
    fixed_image_yx=atlas_slice,
    moving_image_yx_preview=sample_preview_rotated,
    moving_downsample=PREVIEW_DOWNSAMPLE,
    initial_fixed_points_cropped_xy=initial_fixed,
    initial_moving_points_fullres_xy_in_rotated_crop=initial_moving,
    min_pairs=3,
    on_change=_on_landmarks_change,
    fixed_title="FIXED (atlas)",
    moving_title=f"MOVING (sample, {PREVIEW_DOWNSAMPLE}x ds, rot {PRIOR_ROTATION_DEG}°)",
)

# %%
fixed_points, moving_points_fullres = landmark_picker.get_points()
_save_p1_landmarks(
    fixed_points_cropped_xy=fixed_points,
    moving_points_fullres_xy_in_rotated_crop=moving_points_fullres,
)

print(f"Landmark pairs: {len(fixed_points)}")

# %% [markdown]
# ## Phase 1: Landmark-based Similarity2D
#
# Solves: rotation, scale, translation from user-provided landmarks.

# %%
# Convert cropped pixel landmarks to physical coordinates in full-image space.

fixed_points_full = [(x + ATLAS_CROP_OFFSET[0], y + ATLAS_CROP_OFFSET[1]) for x, y in fixed_points]
moving_points_full = [(x + SAMPLE_CROP_OFFSET[0], y + SAMPLE_CROP_OFFSET[1]) for x, y in moving_points_fullres]

fixed_pts_phys = pixels_to_physical_um(fixed_points_full, ATLAS_VOXEL)
moving_pts_phys = pixels_to_physical_um(moving_points_full, SAMPLE_VOXEL_XY)

# Compute transform
transform_similarity = compute_similarity2d_from_landmarks(fixed_pts_phys, moving_pts_phys)

print("Similarity2D transform:")
print(f"  Center: {transform_similarity.GetCenter()}")
print(f"  Angle: {np.degrees(transform_similarity.GetAngle()):.2f}°")
print(f"  Scale: {transform_similarity.GetScale():.4f}")
print(f"  Translation: {transform_similarity.GetTranslation()}")

# Verify landmark errors
errors = []
for fp, mp in zip(fixed_pts_phys, moving_pts_phys):
    transformed = transform_similarity.TransformPoint(fp)
    error = np.linalg.norm(np.array(transformed) - np.array(mp))
    errors.append(error)

max_error = max(errors)
mean_error = np.mean(errors)
print(f"Landmark error: max={max_error:.2f} µm, mean={mean_error:.2f} µm")

if max_error > 100:
    print("WARNING: High landmark error - check landmark correspondences")

# Save Phase 1 transform
sitk.WriteTransform(transform_similarity, str(P1_TFM_PATH))
print(f"Saved: {P1_TFM_PATH}")

# Visualize Phase 1 results: warp sample to atlas space
# Using FULL images so the transform applies to the moving image definition (rotated full sample).
def array_to_sitk(arr: np.ndarray, spacing: tuple[float, float]) -> sitk.Image:
    """Convert 2D numpy array to SimpleITK image."""
    img = sitk.GetImageFromArray(arr.astype(np.float32))
    img.SetSpacing(spacing)
    return img

atlas_slice_full_masked = atlas_slice_full.copy().astype(np.float32)
atlas_slice_full_masked[~atlas_brain_mask] = 0.0

# Create SimpleITK images from FULL versions
atlas_sitk = array_to_sitk(atlas_slice_full_masked, (ATLAS_VOXEL, ATLAS_VOXEL))
sample_sitk = array_to_sitk(sample_slice_full, (SAMPLE_VOXEL_XY, SAMPLE_VOXEL_XY))

# Resample sample to atlas space using the transform
sample_warped = sitk.Resample(
    sample_sitk,
    atlas_sitk,
    transform_similarity,
    sitk.sitkLinear,
    0.0,
    sample_sitk.GetPixelID()
)

sample_warped_arr = sitk.GetArrayFromImage(sample_warped)

# Show Phase 1 results (preview uses CROPPED reference)
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

r0, r1, c0, c1 = atlas_crop_bbox
sample_warped_arr_preview = sample_warped_arr[r0:r1, c0:c1]

axes[0].imshow(atlas_slice, cmap="gray")
axes[0].set_title("Atlas cropped (FIXED)")
axes[0].axis("off")

axes[1].imshow(sample_warped_arr_preview, cmap="gray")
axes[1].set_title("Sample warped to atlas space")
axes[1].axis("off")

# Overlay: atlas in magenta, sample in green
overlay = np.zeros((*atlas_slice.shape, 3), dtype=np.float32)
atlas_norm = atlas_slice / (atlas_slice.max() + 1e-8)
sample_norm = sample_warped_arr_preview / (sample_warped_arr_preview.max() + 1e-8)
overlay[..., 0] = atlas_norm  # R
overlay[..., 1] = sample_norm  # G
overlay[..., 2] = atlas_norm  # B (magenta = R+B)

axes[2].imshow(overlay)
axes[2].set_title("Overlay (magenta=atlas, green=sample)")
axes[2].axis("off")

plt.tight_layout()
plt.savefig(OUT.p1_result_png, dpi=150)
plt.show()

print("Phase 1 complete.")

# %% [markdown]
# ## Summary
#
# | Phase | Transform | File |
# |-------|-----------|------|
# | 1 | Landmark Similarity2D | `p1_similarity.tfm` |
#
# To apply to full-resolution data or spots:
# ```python
# transform = sitk.ReadTransform("transform_final.tfm")
# warped = sitk.Resample(moving, fixed, transform, sitk.sitkLinear)
# ```

# %%
