# %% [markdown]
# # Landmark → Similarity + B-spline refinement (ANTsPy)
#
# 1) Fits a landmark-driven Similarity transform (fixed→moving) using
#    `ants.fit_transform_to_paired_points(..., transform_type="similarity")`.
# 2) Fits a smooth B-spline displacement field to the *residual* displacements using
#    `ants.fit_bspline_object_to_scattered_data(...)`, defined over the moving domain.
#
# The final mapping is: `fixed --(similarity)--> moving --(residual warp)--> moving`.
#
# Run cells sequentially. Outputs are written under:
# `<workspace>/analysis/output/ccf-transforms/<roi>/`.

# %% [markdown]
# ## Configuration

# %%
from __future__ import annotations

from pathlib import Path

import ants
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk
import zarr
from brainglobe_atlasapi import BrainGlobeAtlas
from scipy.ndimage import rotate as ndimage_rotate

from fishtools.ccf.landmark import LandmarkRegistrationOutputs
from fishtools.ccf.sitk_utils import UM_TO_MM, normalize_robust, sitk_from_numpy_2d
from fishtools.io.workspace import Workspace

# %%
# === EDIT THESE ===

WORKSPACE = Path("/working/20251228_JaxA4_Sag4")
ROI = "3"
STITCH_CODEBOOK = "pi"  # analysis/deconv/stitch--{ROI}+{STITCH_CODEBOOK}/fused.zarr

# B-spline fitting knobs (start gentle; increase mesh for more flexibility)
BSPLINE_FITTING_LEVELS = 5
BSPLINE_MESH_SIZE = (1, 1)
BSPLINE_SPLINE_ORDER = 3
BSPLINE_ENFORCE_STATIONARY_BOUNDARY = True

ws = Workspace(WORKSPACE)
OUTDIR = ws.ccf_transforms(ROI)
OUTDIR.mkdir(parents=True, exist_ok=True)

OUT = LandmarkRegistrationOutputs(OUTDIR)
p1 = OUT.read_p1_landmarks()

ATLAS_VOXEL_UM = float(p1.atlas_voxel_um) if p1.atlas_voxel_um is not None else 20.0
SAMPLE_VOXEL_XY_UM = float(p1.sample_voxel_xy_um) if p1.sample_voxel_xy_um is not None else 0.216

ATLAS_NAME = p1.atlas_name or "kim_dev_mouse_e15-5_lsfm_20um"
ATLAS_PLANE = p1.atlas_plane or ("sagittal" if "Sag" in Path(WORKSPACE).name else "coronal")
if p1.atlas_slice_idx is None:
    raise ValueError(f"p1_landmarks.json at {OUT.p1_landmarks_json} is missing atlas_slice_idx.")
ATLAS_SLICE_IDX = int(p1.atlas_slice_idx)

SAMPLE_ZARR = ws.stitch(ROI, STITCH_CODEBOOK) / "fused.zarr"
SAMPLE_Z_IDX = int(p1.sample_z_idx) if p1.sample_z_idx is not None else 5
SAMPLE_CHANNEL = p1.sample_channel or STITCH_CODEBOOK

DOMAIN_NIFTI = OUTDIR / "fixed_domain_crop.nii.gz"
SIMILARITY_MAT_PATH = OUTDIR / "init_similarity_fixed2moving_from_p1.mat"
RESIDUAL_BSPLINE_WARP_PATH = OUTDIR / "init_residual_bspline_movingdomain_warp.nii.gz"
FIXED_NIFTI = OUTDIR / "fixed_atlas_crop.nii.gz"
MOVING_NIFTI = OUTDIR / "moving_sample_crop.nii.gz"
WARPED_NIFTI = OUTDIR / "moving_warped_similarity_plus_residual_bspline.nii.gz"
QC_PNG = OUTDIR / "similarity_plus_residual_bspline_qc.png"
BEFORE_AFTER_DIFF_PNG = OUTDIR / "similarity_plus_residual_bspline_before_after_diff.png"
DEFORMATION_FIELD_PNG = OUTDIR / "init_residual_bspline_deformation_field.png"
DEFORMATION_OVERLAY_UNWARPED_PNG = OUTDIR / "init_residual_bspline_deformation_on_sample_unwarped.png"
DEFORMATION_OVERLAY_WARPED_PNG = OUTDIR / "init_residual_bspline_deformation_on_sample_warped.png"
DEFORMATION_MAX_DIM = 512
DEFORMATION_QUIVER_STEP = 12
WARPED_VIS_SPACING_UM = 2.0
WARPED_VIS_CROP_PAD_PX = 40

# %% [markdown]
# ## Phase 0: Load landmarks + create a fixed-domain image
#
# ANTs needs `domain_image` for nonlinear landmark fits (e.g., B-spline). We create a
# dummy 2D image matching the atlas crop size, with spacing in **mm**.

# %%
fixed_points_cropped_xy = p1.fixed_points_cropped_xy
moving_points_fullres_xy_in_rot_crop = p1.moving_points_fullres_xy_in_rotated_crop

if len(fixed_points_cropped_xy) < 3:
    raise ValueError(f"Need at least 3 landmarks, got {len(fixed_points_cropped_xy)}.")
if len(fixed_points_cropped_xy) != len(moving_points_fullres_xy_in_rot_crop):
    raise ValueError(
        f"Landmark counts must match, got fixed={len(fixed_points_cropped_xy)} vs moving={len(moving_points_fullres_xy_in_rot_crop)}."
    )

ar0, ar1, ac0, ac1 = p1.atlas_crop_bbox
fixed_shape_yx = (int(ar1 - ar0), int(ac1 - ac0))
if fixed_shape_yx[0] <= 0 or fixed_shape_yx[1] <= 0:
    raise ValueError(f"Invalid atlas_crop_bbox={p1.atlas_crop_bbox} -> fixed_shape_yx={fixed_shape_yx}.")

fixed_domain_sitk = sitk_from_numpy_2d(np.zeros(fixed_shape_yx, dtype=np.float32), spacing_um=ATLAS_VOXEL_UM)
sitk.WriteImage(fixed_domain_sitk, str(DOMAIN_NIFTI))
fixed_domain_ants = ants.image_read(str(DOMAIN_NIFTI))

fixed_pts_mm = np.array(
    [(x * ATLAS_VOXEL_UM * UM_TO_MM, y * ATLAS_VOXEL_UM * UM_TO_MM) for x, y in fixed_points_cropped_xy],
    dtype=np.float64,
)
moving_pts_mm = np.array(
    [
        (x * SAMPLE_VOXEL_XY_UM * UM_TO_MM, y * SAMPLE_VOXEL_XY_UM * UM_TO_MM)
        for x, y in moving_points_fullres_xy_in_rot_crop
    ],
    dtype=np.float64,
)

print(f"OUTDIR={OUTDIR}")
print(f"Landmarks={fixed_pts_mm.shape[0]}")
print(f"Fixed crop shape={fixed_shape_yx}, spacing(mm)={fixed_domain_ants.spacing}")
print(f"ATLAS_VOXEL_UM={ATLAS_VOXEL_UM}, SAMPLE_VOXEL_XY_UM={SAMPLE_VOXEL_XY_UM}")

# %% [markdown]
# ## Phase 1: Fit similarity (fixed→moving) + B-spline residual warp + print landmark RMSE

# %%
similarity_tx = ants.fit_transform_to_paired_points(
    moving_points=moving_pts_mm,
    fixed_points=fixed_pts_mm,
    transform_type="similarity",
)

ants.write_transform(similarity_tx, str(SIMILARITY_MAT_PATH))
print(f"Wrote: {SIMILARITY_MAT_PATH}")

pred_moving = np.vstack([np.asarray(similarity_tx.apply_to_point(tuple(p))) for p in fixed_pts_mm])
errs_sim_mm = np.linalg.norm(pred_moving - moving_pts_mm, axis=1)
rmse_sim_mm = float(np.sqrt(np.mean(errs_sim_mm**2)))
max_sim_mm = float(errs_sim_mm.max())
print(f"Similarity landmark RMSE: {rmse_sim_mm * 1e3:.2f} µm (max {max_sim_mm * 1e3:.2f} µm)")

# %% [markdown]
# ## Phase 2: Warp moving crop into fixed crop (QC)

# %%
def ants_numpy_yx(img: ants.ANTsImage) -> np.ndarray:
    arr_xy = np.asarray(img.numpy())
    if arr_xy.ndim != 2:
        raise ValueError(f"Expected 2D ANTsImage, got shape={arr_xy.shape}.")
    return arr_xy.T


atlas = BrainGlobeAtlas(ATLAS_NAME)
atlas_reference_slices = atlas.reference if ATLAS_PLANE == "coronal" else atlas.reference.transpose(2, 1, 0)
atlas_annotation_slices = atlas.annotation if ATLAS_PLANE == "coronal" else atlas.annotation.transpose(2, 1, 0)

atlas_slice_full = atlas_reference_slices[ATLAS_SLICE_IDX, :, :]
atlas_annotation_full = atlas_annotation_slices[ATLAS_SLICE_IDX, :, :]
atlas_brain_mask = atlas_annotation_full > 0
atlas_slice_full_masked = atlas_slice_full.copy().astype(np.float32)
atlas_slice_full_masked[~atlas_brain_mask] = 0.0

ar0, ar1, ac0, ac1 = p1.atlas_crop_bbox
fixed_np = atlas_slice_full_masked[ar0:ar1, ac0:ac1].astype(np.float32)

arr = zarr.open(str(SAMPLE_ZARR), mode="r")
keys = list(arr.attrs.get("key", []))
ch_idx = keys.index(SAMPLE_CHANNEL) if SAMPLE_CHANNEL in keys else 0
sample_slice_full_raw = np.asarray(arr[SAMPLE_Z_IDX, :, :, ch_idx])
sample_slice_full_raw_pose = sample_slice_full_raw[:, ::-1] if p1.prior_flip_x else sample_slice_full_raw
if p1.prior_rotation_deg != 0:
    sample_slice_full = ndimage_rotate(sample_slice_full_raw_pose, p1.prior_rotation_deg, reshape=True, order=1)
else:
    sample_slice_full = sample_slice_full_raw_pose

sr0, sr1, sc0, sc1 = p1.sample_rotated_crop_bbox
moving_np = sample_slice_full[sr0:sr1, sc0:sc1].astype(np.float32)

fixed_sitk = sitk_from_numpy_2d(normalize_robust(fixed_np), spacing_um=ATLAS_VOXEL_UM)
moving_sitk = sitk_from_numpy_2d(normalize_robust(moving_np), spacing_um=SAMPLE_VOXEL_XY_UM)
sitk.WriteImage(fixed_sitk, str(FIXED_NIFTI))
sitk.WriteImage(moving_sitk, str(MOVING_NIFTI))

fixed_ants = ants.image_read(str(FIXED_NIFTI))
moving_ants = ants.image_read(str(MOVING_NIFTI))

if RESIDUAL_BSPLINE_WARP_PATH.exists():
    residual_bspline_field = ants.image_read(str(RESIDUAL_BSPLINE_WARP_PATH))
    print(f"Loaded: {RESIDUAL_BSPLINE_WARP_PATH}")
else:
    residual_mm = (moving_pts_mm - pred_moving).astype(np.float64)
    residual_bspline_field = ants.fit_bspline_object_to_scattered_data(
        scattered_data=residual_mm,
        parametric_data=pred_moving,
        parametric_domain_origin=moving_ants.origin,
        parametric_domain_spacing=moving_ants.spacing,
        parametric_domain_size=moving_ants.shape,
        number_of_fitting_levels=BSPLINE_FITTING_LEVELS,
        mesh_size=BSPLINE_MESH_SIZE,
        spline_order=BSPLINE_SPLINE_ORDER,
    )
    ants.image_write(residual_bspline_field, str(RESIDUAL_BSPLINE_WARP_PATH))
    print(f"Wrote: {RESIDUAL_BSPLINE_WARP_PATH}")

residual_tx = ants.transform_from_displacement_field(residual_bspline_field)
pred_refined = np.vstack([np.asarray(residual_tx.apply_to_point(tuple(p))) for p in pred_moving])
errs_refined_mm = np.linalg.norm(pred_refined - moving_pts_mm, axis=1)
rmse_refined_mm = float(np.sqrt(np.mean(errs_refined_mm**2)))
max_refined_mm = float(errs_refined_mm.max())
print(f"Refined landmark RMSE: {rmse_refined_mm * 1e3:.2f} µm (max {max_refined_mm * 1e3:.2f} µm)")

warped = ants.apply_transforms(
    fixed=fixed_ants,
    moving=moving_ants,
    transformlist=[str(RESIDUAL_BSPLINE_WARP_PATH), str(SIMILARITY_MAT_PATH)],
    interpolator="linear",
)
ants.image_write(warped, str(WARPED_NIFTI))

f = normalize_robust(ants_numpy_yx(fixed_ants).astype(np.float32))
w = normalize_robust(ants_numpy_yx(warped).astype(np.float32))
overlay = np.stack([f, w, f], axis=-1)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(f, cmap="gray")
axes[0].set_title("Fixed (atlas crop)")
axes[0].axis("off")
axes[1].imshow(w, cmap="gray")
axes[1].set_title("Warped moving (B-spline init)")
axes[1].axis("off")
axes[2].imshow(overlay)
axes[2].set_title("Overlay (magenta=fixed, green=warped moving)")
axes[2].axis("off")
plt.tight_layout()
plt.savefig(QC_PNG, dpi=160)
plt.show()

print(f"Wrote: {WARPED_NIFTI}")
print(f"Wrote: {QC_PNG}")

# %% [markdown]
# ## Phase 2b: Before/after diff (similarity-only vs final warp)
#
# Mirrors the `before_after_diff.png` QC in `ccf/ant.py`.

# %%
before_warped = ants.apply_transforms(
    fixed=fixed_ants,
    moving=moving_ants,
    transformlist=[str(SIMILARITY_MAT_PATH)],
    interpolator="linear",
)

fixed_n = normalize_robust(ants_numpy_yx(fixed_ants).astype(np.float32))
before_n = normalize_robust(ants_numpy_yx(before_warped).astype(np.float32))
after_n = normalize_robust(ants_numpy_yx(warped).astype(np.float32))

diff_before = np.abs(fixed_n - before_n)
diff_after = np.abs(fixed_n - after_n)

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes[0, 0].imshow(fixed_n, cmap="gray")
axes[0, 0].set_title("Fixed (atlas)")
axes[0, 0].axis("off")

axes[0, 1].imshow(before_n, cmap="gray")
axes[0, 1].set_title("Before (similarity only)")
axes[0, 1].axis("off")

axes[0, 2].imshow(after_n, cmap="gray")
axes[0, 2].set_title("After (similarity + residual B-spline)")
axes[0, 2].axis("off")

axes[1, 0].imshow(diff_before, cmap="magma")
axes[1, 0].set_title("|fixed - before|")
axes[1, 0].axis("off")

axes[1, 1].imshow(diff_after, cmap="magma")
axes[1, 1].set_title("|fixed - after|")
axes[1, 1].axis("off")

axes[1, 2].imshow(diff_before - diff_after, cmap="bwr")
axes[1, 2].set_title("(|f-b| - |f-a|) (red=improved)")
axes[1, 2].axis("off")

plt.tight_layout()
plt.savefig(BEFORE_AFTER_DIFF_PNG, dpi=160)
plt.show()
print(f"Wrote: {BEFORE_AFTER_DIFF_PNG}")

# %% [markdown]
# ## Phase 3: Plot deformation field (residual B-spline)

# %%
plot_xy = residual_bspline_field.shape
max_dim = max(plot_xy)
if max_dim <= 0:
    raise ValueError(f"Invalid residual_bspline_field.shape={plot_xy}.")

scale = max_dim / float(DEFORMATION_MAX_DIM)
plot_xy_ds = (max(32, int(round(plot_xy[0] / scale))), max(32, int(round(plot_xy[1] / scale))))
residual_ds = ants.resample_image(residual_bspline_field, plot_xy_ds, use_voxels=True, interp_type=0)

disp_xyc = np.asarray(residual_ds.numpy())
if disp_xyc.ndim != 3 or disp_xyc.shape[2] != 2:
    raise ValueError(f"Expected displacement field array with shape (x,y,2), got {disp_xyc.shape}.")
disp_yxc = np.transpose(disp_xyc, (1, 0, 2))

dx_mm = disp_yxc[..., 0]
dy_mm = disp_yxc[..., 1]
mag_um = np.sqrt(dx_mm**2 + dy_mm**2) / UM_TO_MM

spacing_x_mm, spacing_y_mm = residual_ds.spacing
dx_px = dx_mm / float(spacing_x_mm)
dy_px = dy_mm / float(spacing_y_mm)

step = int(DEFORMATION_QUIVER_STEP)
yy, xx = np.mgrid[0 : mag_um.shape[0] : step, 0 : mag_um.shape[1] : step]
u = dx_px[::step, ::step]
v = dy_px[::step, ::step]
max_disp_px = float(np.sqrt(u**2 + v**2).max())
desired_max_len = step * 0.8
quiver_scale = max_disp_px / desired_max_len if max_disp_px > 0 else 1.0

fig, ax = plt.subplots(figsize=(10, 7))
im = ax.imshow(mag_um, cmap="magma")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="|displacement| (µm)")
ax.quiver(
    xx,
    yy,
    u,
    v,
    color="cyan",
    angles="xy",
    scale_units="xy",
    scale=quiver_scale,
    width=0.0022,
    alpha=0.75,
)
ax.set_title("Residual B-spline displacement field (moving domain, downsampled)")
ax.axis("off")
plt.tight_layout()
plt.savefig(DEFORMATION_FIELD_PNG, dpi=200)
plt.show()

print(f"Wrote: {DEFORMATION_FIELD_PNG}")

# %% [markdown]
# ## Phase 3b: Overlay deformation field on moving sample (alignment check)

# %%
moving_ds = ants.resample_image(moving_ants, residual_ds.shape, use_voxels=True, interp_type=0)
bg = normalize_robust(ants_numpy_yx(moving_ds).astype(np.float32))
if bg.shape != mag_um.shape:
    raise ValueError(f"Unexpected bg shape {bg.shape} vs mag_um {mag_um.shape}.")

origin_x_mm, origin_y_mm = moving_ds.origin
spacing_x_mm, spacing_y_mm = moving_ds.spacing

pred_x = (pred_moving[:, 0] - origin_x_mm) / float(spacing_x_mm)
pred_y = (pred_moving[:, 1] - origin_y_mm) / float(spacing_y_mm)
target_x = (moving_pts_mm[:, 0] - origin_x_mm) / float(spacing_x_mm)
target_y = (moving_pts_mm[:, 1] - origin_y_mm) / float(spacing_y_mm)

res_u = (moving_pts_mm[:, 0] - pred_moving[:, 0]) / float(spacing_x_mm)
res_v = (moving_pts_mm[:, 1] - pred_moving[:, 1]) / float(spacing_y_mm)

sign = -1.0
fig, ax = plt.subplots(figsize=(10, 7))
ax.imshow(bg, cmap="gray", interpolation="bilinear")
ax.imshow(mag_um, cmap="magma", alpha=0.25, interpolation="bilinear")
mask_u = bg[::step, ::step] > 0.03
u_plot = np.ma.masked_where(~mask_u, sign * u)
v_plot = np.ma.masked_where(~mask_u, sign * v)
ax.quiver(
    xx,
    yy,
    u_plot,
    v_plot,
    color="cyan",
    angles="xy",
    scale_units="xy",
    scale=quiver_scale,
    width=0.0022,
    alpha=0.45,
)
ax.scatter(pred_x, pred_y, s=35, c="white", marker="o", linewidths=0.0, label="Similarity(pred)")
ax.scatter(target_x, target_y, s=35, c="yellow", marker="x", linewidths=1.3, label="Target(moving)")
ax.quiver(
    target_x,
    target_y,
    -res_u,
    -res_v,
    color="lime",
    angles="xy",
    scale_units="xy",
    scale=quiver_scale,
    width=0.004,
    alpha=0.9,
)
ax.set_title("Residual deformation over moving sample (moving→fixed)")
ax.axis("off")
ax.legend(loc="lower right", frameon=True)

plt.tight_layout()
plt.savefig(DEFORMATION_OVERLAY_UNWARPED_PNG, dpi=200)
plt.show()

print(f"Wrote: {DEFORMATION_OVERLAY_UNWARPED_PNG}")

# %% [markdown]
# ## Phase 3c: Overlay deformation field on warped sample (alignment check)
#
# This overlays the *same* residual deformation field, but resampled into the FIXED
# (atlas-crop) domain, on top of the final warped sample result.

# %%
size_x, size_y = fixed_ants.shape
origin_fixed_x_mm, origin_fixed_y_mm = fixed_ants.origin
spacing_fixed_x_mm, spacing_fixed_y_mm = fixed_ants.spacing

xs_mm = origin_fixed_x_mm + np.arange(size_x, dtype=np.float64) * float(spacing_fixed_x_mm)
ys_mm = origin_fixed_y_mm + np.arange(size_y, dtype=np.float64) * float(spacing_fixed_y_mm)
grid_x_mm, grid_y_mm = np.meshgrid(xs_mm, ys_mm, indexing="xy")

df_fixed_grid = pd.DataFrame(
    {
        "x": grid_x_mm.ravel(),
        "y": grid_y_mm.ravel(),
        "z": 0.0,
        "t": 0.0,
    }
)

grid_pred_moving = ants.apply_transforms_to_points(
    dim=2,
    points=df_fixed_grid,
    transformlist=[str(SIMILARITY_MAT_PATH)],
)
df_pred = grid_pred_moving[["x", "y"]].copy()
df_pred["z"] = 0.0
df_pred["t"] = 0.0
grid_refined_moving = ants.apply_transforms_to_points(
    dim=2,
    points=df_pred,
    transformlist=[str(RESIDUAL_BSPLINE_WARP_PATH)],
)

dx_fixed_mm = (grid_refined_moving["x"] - grid_pred_moving["x"]).to_numpy(dtype=np.float64).reshape((size_y, size_x))
dy_fixed_mm = (grid_refined_moving["y"] - grid_pred_moving["y"]).to_numpy(dtype=np.float64).reshape((size_y, size_x))
disp_fixed_xyc = np.stack([dx_fixed_mm.T, dy_fixed_mm.T], axis=-1).astype(np.float32)
disp_fixed_field = ants.from_numpy(
    disp_fixed_xyc,
    origin=fixed_ants.origin,
    spacing=fixed_ants.spacing,
    direction=fixed_ants.direction,
    has_components=True,
)

vis_spacing_mm = float(WARPED_VIS_SPACING_UM) * UM_TO_MM
extent_x_mm = float(spacing_fixed_x_mm) * float(size_x)
extent_y_mm = float(spacing_fixed_y_mm) * float(size_y)
vis_size_x = max(16, int(round(extent_x_mm / vis_spacing_mm)))
vis_size_y = max(16, int(round(extent_y_mm / vis_spacing_mm)))
fixed_vis = ants.resample_image(fixed_ants, (vis_size_x, vis_size_y), use_voxels=True, interp_type=0)
warped_vis = ants.apply_transforms(
    fixed=fixed_vis,
    moving=moving_ants,
    transformlist=[str(RESIDUAL_BSPLINE_WARP_PATH), str(SIMILARITY_MAT_PATH)],
    interpolator="linear",
)
after_vis_n = normalize_robust(ants_numpy_yx(warped_vis).astype(np.float32))

disp_vis = ants.resample_image(disp_fixed_field, fixed_vis.shape, use_voxels=True, interp_type=1)
disp_vis_xyc = np.asarray(disp_vis.numpy())
disp_vis_yxc = np.transpose(disp_vis_xyc, (1, 0, 2))
dx_vis_mm = disp_vis_yxc[..., 0]
dy_vis_mm = disp_vis_yxc[..., 1]
mag_vis_um = np.sqrt(dx_vis_mm**2 + dy_vis_mm**2) / UM_TO_MM

spacing_vis_x_mm, spacing_vis_y_mm = fixed_vis.spacing
dx_vis_px = dx_vis_mm / float(spacing_vis_x_mm)
dy_vis_px = dy_vis_mm / float(spacing_vis_y_mm)

origin_vis_x_mm, origin_vis_y_mm = fixed_vis.origin
mask_vis = after_vis_n > 0.03
fixed_x_full = (fixed_pts_mm[:, 0] - origin_vis_x_mm) / float(spacing_vis_x_mm)
fixed_y_full = (fixed_pts_mm[:, 1] - origin_vis_y_mm) / float(spacing_vis_y_mm)
if np.any(mask_vis):
    rows = np.any(mask_vis, axis=1)
    cols = np.any(mask_vis, axis=0)
    r0 = int(max(0, np.where(rows)[0][0] - WARPED_VIS_CROP_PAD_PX))
    r1 = int(min(after_vis_n.shape[0], np.where(rows)[0][-1] + WARPED_VIS_CROP_PAD_PX + 1))
    c0 = int(max(0, np.where(cols)[0][0] - WARPED_VIS_CROP_PAD_PX))
    c1 = int(min(after_vis_n.shape[1], np.where(cols)[0][-1] + WARPED_VIS_CROP_PAD_PX + 1))
else:
    r0, r1, c0, c1 = 0, after_vis_n.shape[0], 0, after_vis_n.shape[1]

in_bounds = (
    (fixed_x_full >= 0)
    & (fixed_x_full < after_vis_n.shape[1])
    & (fixed_y_full >= 0)
    & (fixed_y_full < after_vis_n.shape[0])
)
if np.any(in_bounds):
    r0 = int(max(0, min(r0, np.floor(fixed_y_full[in_bounds].min()) - WARPED_VIS_CROP_PAD_PX)))
    r1 = int(
        min(after_vis_n.shape[0], max(r1, np.ceil(fixed_y_full[in_bounds].max()) + WARPED_VIS_CROP_PAD_PX + 1))
    )
    c0 = int(max(0, min(c0, np.floor(fixed_x_full[in_bounds].min()) - WARPED_VIS_CROP_PAD_PX)))
    c1 = int(
        min(after_vis_n.shape[1], max(c1, np.ceil(fixed_x_full[in_bounds].max()) + WARPED_VIS_CROP_PAD_PX + 1))
    )

after_crop = after_vis_n[r0:r1, c0:c1]
mag_crop = mag_vis_um[r0:r1, c0:c1]
dx_crop = dx_vis_px[r0:r1, c0:c1]
dy_crop = dy_vis_px[r0:r1, c0:c1]
mask_crop = mask_vis[r0:r1, c0:c1]

step_fixed = int(DEFORMATION_QUIVER_STEP)
yy_f, xx_f = np.mgrid[0 : mag_crop.shape[0] : step_fixed, 0 : mag_crop.shape[1] : step_fixed]
u_f = dx_crop[::step_fixed, ::step_fixed]
v_f = dy_crop[::step_fixed, ::step_fixed]
max_disp_fixed_px = float(np.sqrt(u_f**2 + v_f**2).max())
desired_max_len_fixed = step_fixed * 0.8
quiver_scale_fixed = max_disp_fixed_px / desired_max_len_fixed if max_disp_fixed_px > 0 else 1.0

fixed_x = fixed_x_full - float(c0)
fixed_y = fixed_y_full - float(r0)

res_lm_u = (pred_refined[:, 0] - pred_moving[:, 0]) / float(spacing_vis_x_mm)
res_lm_v = (pred_refined[:, 1] - pred_moving[:, 1]) / float(spacing_vis_y_mm)

sign = -1.0
fig, ax = plt.subplots(figsize=(10, 7))
ax.imshow(after_crop, cmap="gray", interpolation="bicubic")
ax.imshow(mag_crop, cmap="magma", alpha=0.25, interpolation="bilinear")
mask_f = mask_crop[::step_fixed, ::step_fixed]
u_f_plot = np.ma.masked_where(~mask_f, sign * u_f)
v_f_plot = np.ma.masked_where(~mask_f, sign * v_f)
ax.quiver(
    xx_f,
    yy_f,
    u_f_plot,
    v_f_plot,
    color="cyan",
    angles="xy",
    scale_units="xy",
    scale=quiver_scale_fixed,
    width=0.0022,
    alpha=0.45,
)
ax.scatter(fixed_x, fixed_y, s=35, c="white", marker="o", linewidths=0.0, label="Fixed landmarks")
ax.quiver(
    fixed_x,
    fixed_y,
    sign * res_lm_u,
    sign * res_lm_v,
    color="lime",
    angles="xy",
    scale_units="xy",
    scale=quiver_scale_fixed,
    width=0.004,
    alpha=0.9,
)
ax.set_title("Residual deformation over warped sample (moving→fixed)")
ax.axis("off")
ax.legend(loc="lower right", frameon=True)

plt.tight_layout()
plt.savefig(DEFORMATION_OVERLAY_WARPED_PNG, dpi=200)
plt.show()
print(f"Wrote: {DEFORMATION_OVERLAY_WARPED_PNG}")


# %%
