# %% [markdown]
# # Landmark → linear init + diffeomorphic SyN refinement (MI, ANTsPy)
#
# Replaces the residual B-spline displacement (non-diffeomorphic) with an ANTs SyN-family
# registration (diffeomorphic non-rigid warp) driven by Mattes mutual information.
#
# High level:
# 1) Fit a landmark-driven linear transform (moving→fixed) using
#    `ants.fit_transform_to_paired_points(...)` (swap fixed/moving points).
# 2) Run diffeomorphic registration (SyN) initialized by that similarity, using MI.
#
# Outputs are written under:
# `<workspace>/analysis/output/ccf-transforms/<roi>/`.

# %%
from __future__ import annotations

import json
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
from fishtools.ccf.sitk_utils import (
    UM_TO_MM,
    erode_by_um,
    gradmag_feature,
    make_moving_mask_sitk,
    normalize_robust,
    resample_sitk_to_spacing,
    signed_distance,
    sitk_from_numpy_2d,
)
from fishtools.io.workspace import Workspace

# %% [markdown]
# ## Configuration

# %%
# === EDIT THESE ===
WORKSPACE = Path("/working/20251228_JaxA4_Sag4")
ROI = "3"
STITCH_CODEBOOK = "pi"  # analysis/deconv/stitch--{ROI}+{STITCH_CODEBOOK}/fused.zarr

# Diffeomorphic registration knobs
# Use "SyNOnly" so `syn_metric="mattes"` (MI) is honored; some antsRegistrationSyN* presets
# hard-code CC as the deformable metric.
SYN_TYPE_OF_TRANSFORM = "SyNOnly"  # diffeomorphic non-rigid
SYN_METRIC = "mattes"  # MI
SYN_SAMPLING = 64
SYN_REG_ITERATIONS = (300, 200, 100, 50)
EDGE_GUARD_UM = 100.0  # erode masks to avoid edge-driven warps
SYN_GRAD_STEP = 0.5
SYN_FLOW_SIGMA = 1.0
SYN_TOTAL_SIGMA = 0.0

# Landmark-driven linear initialization.
# NOTE: For ANTs image resampling (`ants.apply_transforms`) and `ants.registration(initial_transform=...)`,
# the transform should map **fixed→moving** (the resampler internally pulls from the moving image).
# "affine" often helps when the ROI is a partial/warped view relative to atlas space.
LANDMARK_LINEAR_TRANSFORM_TYPE = "affine"  # "similarity" | "affine"

# Bias-field correction (helps MI for uneven illumination / shading)
USE_N4 = True
N4_APPLY_TO_FIXED = False
N4_MAX_ITERATIONS = (50, 50, 30, 20)

# Use feature images (gradient magnitude) for cross-modality stability.
USE_FEATURE_IMAGES = True
FEATURE_SIGMA_UM = 60.0
FEATURE_WEIGHT = 0.25

# Inject coarse shape via signed distance maps of masks.
USE_MASK_DISTANCE_METRIC = True
MASK_DISTANCE_WEIGHT = 0.5

# Crop the fixed image to where the warped moving mask lands (helps partial-overlap registration).
CROP_FIXED_TO_OVERLAP = True
CROP_PAD_VOX = 24

# Landmark injection (as an extra image metric via Gaussian heatmaps)
USE_LANDMARK_HEATMAP_METRIC = True
LANDMARK_HEATMAP_SIGMA_UM = 250.0
LANDMARK_HEATMAP_WEIGHT = 0.25

ws = Workspace(WORKSPACE)
WS_OUTDIR = ws.ccf_transforms(ROI)
OUTDIR = WS_OUTDIR / "landmark_syn_mi"
OUTDIR.mkdir(parents=True, exist_ok=True)

OUT = LandmarkRegistrationOutputs(WS_OUTDIR)
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

LINEAR_INIT_MAT_PATH = OUTDIR / f"init_{LANDMARK_LINEAR_TRANSFORM_TYPE}_fixed2moving_from_p1.mat"
FIXED_NIFTI = OUTDIR / "fixed_atlas_crop.nii.gz"
MOVING_NIFTI = OUTDIR / "moving_sample_crop.nii.gz"
FIXED_MASK_NIFTI = OUTDIR / "fixed_mask_crop.nii.gz"
MOVING_MASK_NIFTI = OUTDIR / "moving_mask_crop.nii.gz"
OVERLAP_MASK_NIFTI = OUTDIR / "overlap_mask_eroded.nii.gz"
MOVING_REG_RAW_NIFTI = OUTDIR / "moving_sample_crop_reg_raw.nii.gz"
FIXED_REG_RAW_NIFTI = OUTDIR / "fixed_reg_raw.nii.gz"
MOVING_REG_NIFTI = OUTDIR / "moving_sample_crop_reg.nii.gz"
FIXED_REG_NIFTI = OUTDIR / "fixed_reg.nii.gz"
MOVING_MASK_REG_NIFTI = OUTDIR / "moving_mask_crop_reg.nii.gz"
FIXED_FEAT_NIFTI = OUTDIR / "fixed_feature_gradmag.nii.gz"
MOVING_FEAT_NIFTI = OUTDIR / "moving_feature_gradmag_reg.nii.gz"
FIXED_REG_CROP_NIFTI = OUTDIR / "fixed_reg_crop_for_syn.nii.gz"
FIXED_MASK_CROP_NIFTI = OUTDIR / "fixed_mask_crop_for_syn.nii.gz"
FIXED_DT_NIFTI = OUTDIR / "fixed_mask_signed_distance.nii.gz"
MOVING_DT_NIFTI = OUTDIR / "moving_mask_signed_distance_reg.nii.gz"
FIXED_LM_NIFTI = OUTDIR / "fixed_landmark_heatmap.nii.gz"
MOVING_LM_NIFTI = OUTDIR / "moving_landmark_heatmap_reg.nii.gz"

SYN_PREFIX = OUTDIR / "final_syn_mi_"
WARPED_BEFORE_NIFTI = OUTDIR / "moving_warped_init.nii.gz"
WARPED_AFTER_NIFTI = OUTDIR / "moving_warped_similarity_plus_syn.nii.gz"
QC_PNG = OUTDIR / "similarity_plus_syn_qc.png"
QC_MASKED_PNG = OUTDIR / "similarity_plus_syn_qc_masked.png"
QC_ZOOM_PNG = OUTDIR / "similarity_plus_syn_qc_zoom.png"
QC_ZOOM_MASKED_PNG = OUTDIR / "similarity_plus_syn_qc_zoom_masked.png"
SUMMARY_JSON = OUTDIR / "similarity_plus_syn_summary.json"
MOVING_MASK_WARPED_FINAL_NIFTI = OUTDIR / "moving_mask_warped_final.nii.gz"
OVERLAP_MASK_FINAL_NIFTI = OUTDIR / "overlap_mask_final.nii.gz"

# %% [markdown]
# ## Helpers

# %%
def ants_numpy_yx(img: ants.ANTsImage) -> np.ndarray:
    arr_xy = np.asarray(img.numpy())
    if arr_xy.ndim != 2:
        raise ValueError(f"Expected 2D ANTsImage, got shape={arr_xy.shape}.")
    return arr_xy.T


def qc_overlay_png(
    *,
    fixed_img: ants.ANTsImage,
    moving_warped: ants.ANTsImage,
    out_png: Path,
    title: str,
    moving_mask_fixed: ants.ANTsImage | None = None,
) -> None:
    f = normalize_robust(ants_numpy_yx(fixed_img).astype(np.float32))
    m = normalize_robust(ants_numpy_yx(moving_warped).astype(np.float32))
    if moving_mask_fixed is not None:
        mask = ants_numpy_yx(moving_mask_fixed) > 0
        m = m * mask.astype(np.float32)
    overlay = np.stack([f, m, f], axis=-1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(f, cmap="gray")
    axes[0].set_title("Fixed (atlas crop)")
    axes[0].axis("off")
    axes[1].imshow(m, cmap="gray")
    axes[1].set_title("Warped moving")
    axes[1].axis("off")
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay (magenta=fixed, green=warped moving)")
    axes[2].axis("off")
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.show()


def landmark_error_after_registration(
    *,
    moving_pts_mm: np.ndarray,
    fixed_pts_mm: np.ndarray,
    invtransforms: list[str] | str,
) -> dict[str, float]:
    inv_list = [invtransforms] if isinstance(invtransforms, str) else list(invtransforms)
    df = pd.DataFrame(moving_pts_mm, columns=["x", "y"])
    df["z"] = 0.0
    df["t"] = 0.0
    # Note: ANTs "forward transforms" are fixed→moving (point mapping), used for resampling moving into fixed.
    # To map moving-space points into fixed space, use inverse transforms.
    df_out = ants.apply_transforms_to_points(dim=2, points=df, transformlist=inv_list)
    pred_fixed = df_out[["x", "y"]].to_numpy(dtype=np.float64)
    err = np.linalg.norm(pred_fixed - fixed_pts_mm, axis=1)
    return {
        "rmse_mm": float(np.sqrt(np.mean(err**2))),
        "max_mm": float(err.max()),
        "mean_mm": float(err.mean()),
    }


def landmark_heatmap_from_points_mm(
    *,
    domain_img: ants.ANTsImage,
    points_mm: np.ndarray,
    sigma_um: float,
) -> ants.ANTsImage:
    """Create a smooth landmark heatmap on `domain_img` grid.

    We rasterize landmarks in *physical* coordinates (mm) into the image's index space and
    apply Gaussian smoothing. This lets us inject landmarks as an additional image metric
    in diffeomorphic registration (SyN) via `multivariate_extras`.
    """

    origin_x_mm, origin_y_mm = domain_img.origin
    spacing_x_mm, spacing_y_mm = domain_img.spacing
    size_x, size_y = domain_img.shape

    base_xy = np.zeros((size_x, size_y), dtype=np.float32)
    for x_mm, y_mm in np.asarray(points_mm, dtype=np.float64):
        ix = int(np.round((float(x_mm) - float(origin_x_mm)) / float(spacing_x_mm)))
        iy = int(np.round((float(y_mm) - float(origin_y_mm)) / float(spacing_y_mm)))
        if 0 <= ix < size_x and 0 <= iy < size_y:
            base_xy[ix, iy] = 1.0

    img = ants.from_numpy(
        base_xy,
        origin=domain_img.origin,
        spacing=domain_img.spacing,
        direction=domain_img.direction,
    )
    sigma_mm = float(sigma_um) * UM_TO_MM
    return ants.smooth_image(img, sigma_mm, sigma_in_physical_coordinates=True)


def n4_correct_sitk(*, img: sitk.Image, mask: sitk.Image, max_iters: tuple[int, ...]) -> sitk.Image:
    img_f = sitk.Cast(img, sitk.sitkFloat32)
    mask_u8 = sitk.Cast(mask, sitk.sitkUInt8)

    n4 = sitk.N4BiasFieldCorrectionImageFilter()
    n4.SetMaximumNumberOfIterations(list(max_iters))
    return n4.Execute(img_f, mask_u8)


def normalize_sitk_intensity(img: sitk.Image) -> sitk.Image:
    a = normalize_robust(sitk.GetArrayFromImage(img))
    out = sitk.GetImageFromArray(a.astype(np.float32))
    out.CopyInformation(img)
    return out


# %% [markdown]
# ## Phase 0: Load landmarks + fit linear init (fixed→moving)

# %%
fixed_points_cropped_xy = p1.fixed_points_cropped_xy
moving_points_fullres_xy_in_rot_crop = p1.moving_points_fullres_xy_in_rotated_crop

if len(fixed_points_cropped_xy) < 3:
    raise ValueError(f"Need at least 3 landmarks, got {len(fixed_points_cropped_xy)}.")
if len(fixed_points_cropped_xy) != len(moving_points_fullres_xy_in_rot_crop):
    raise ValueError(
        f"Landmark counts must match, got fixed={len(fixed_points_cropped_xy)} vs moving={len(moving_points_fullres_xy_in_rot_crop)}."
    )

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

if LANDMARK_LINEAR_TRANSFORM_TYPE not in {"similarity", "affine"}:
    raise ValueError(
        f"LANDMARK_LINEAR_TRANSFORM_TYPE must be 'similarity' or 'affine', got {LANDMARK_LINEAR_TRANSFORM_TYPE!r}."
    )

linear_init_tx = ants.fit_transform_to_paired_points(
    moving_points=moving_pts_mm,
    fixed_points=fixed_pts_mm,
    transform_type=LANDMARK_LINEAR_TRANSFORM_TYPE,
)
ants.write_transform(linear_init_tx, str(LINEAR_INIT_MAT_PATH))
print(f"Wrote: {LINEAR_INIT_MAT_PATH}")

pred_moving = np.vstack([np.asarray(linear_init_tx.apply_to_point(tuple(p))) for p in fixed_pts_mm])
errs_lin_mm = np.linalg.norm(pred_moving - moving_pts_mm, axis=1)
rmse_lin_mm = float(np.sqrt(np.mean(errs_lin_mm**2)))
max_lin_mm = float(errs_lin_mm.max())
print(
    f"{LANDMARK_LINEAR_TRANSFORM_TYPE} landmark RMSE: {rmse_lin_mm * 1e3:.2f} µm (max {max_lin_mm * 1e3:.2f} µm)"
)

# %% [markdown]
# ## Phase 1: Load fixed/moving crops as ANTs images (+ masks)

# %%
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
fixed_mask_np = atlas_brain_mask[ar0:ar1, ac0:ac1].astype(np.uint8)

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
fixed_mask_sitk = sitk_from_numpy_2d(fixed_mask_np.astype(np.float32), spacing_um=ATLAS_VOXEL_UM) > 0
moving_reg_sitk = resample_sitk_to_spacing(moving_sitk, target_spacing_um=ATLAS_VOXEL_UM, interp=sitk.sitkLinear)
moving_mask_reg_seed_sitk = make_moving_mask_sitk(moving_reg_sitk)
moving_mask_reg_sitk = moving_mask_reg_seed_sitk

fixed_reg_sitk = fixed_sitk
moving_reg_intensity_sitk = moving_reg_sitk

sitk.WriteImage(fixed_reg_sitk, str(FIXED_REG_RAW_NIFTI))
sitk.WriteImage(moving_reg_intensity_sitk, str(MOVING_REG_RAW_NIFTI))

if USE_N4:
    if N4_APPLY_TO_FIXED:
        fixed_reg_sitk = n4_correct_sitk(img=fixed_reg_sitk, mask=fixed_mask_sitk, max_iters=N4_MAX_ITERATIONS)
    moving_reg_intensity_sitk = n4_correct_sitk(
        img=moving_reg_intensity_sitk, mask=moving_mask_reg_seed_sitk, max_iters=N4_MAX_ITERATIONS
    )
    moving_mask_reg_sitk = make_moving_mask_sitk(moving_reg_intensity_sitk)

fixed_reg_sitk = normalize_sitk_intensity(fixed_reg_sitk)
moving_reg_intensity_sitk = normalize_sitk_intensity(moving_reg_intensity_sitk)

fixed_mask_sitk = erode_by_um(fixed_mask_sitk, EDGE_GUARD_UM)
moving_mask_reg_sitk = erode_by_um(moving_mask_reg_sitk, EDGE_GUARD_UM)

if USE_FEATURE_IMAGES:
    fixed_feat_sitk = gradmag_feature(fixed_reg_sitk, FEATURE_SIGMA_UM)
    moving_feat_sitk = gradmag_feature(moving_reg_intensity_sitk, FEATURE_SIGMA_UM)
else:
    fixed_feat_sitk = None
    moving_feat_sitk = None

sitk.WriteImage(fixed_reg_sitk, str(FIXED_REG_NIFTI))
sitk.WriteImage(moving_reg_intensity_sitk, str(MOVING_REG_NIFTI))

sitk.WriteImage(fixed_sitk, str(FIXED_NIFTI))
sitk.WriteImage(moving_sitk, str(MOVING_NIFTI))
sitk.WriteImage(sitk.Cast(fixed_mask_sitk, sitk.sitkUInt8), str(FIXED_MASK_NIFTI))
sitk.WriteImage(sitk.Cast(moving_mask_reg_sitk, sitk.sitkUInt8), str(MOVING_MASK_REG_NIFTI))

if USE_FEATURE_IMAGES:
    assert fixed_feat_sitk is not None
    assert moving_feat_sitk is not None
    sitk.WriteImage(fixed_feat_sitk, str(FIXED_FEAT_NIFTI))
    sitk.WriteImage(moving_feat_sitk, str(MOVING_FEAT_NIFTI))

fixed_ants = ants.image_read(str(FIXED_NIFTI))
moving_ants = ants.image_read(str(MOVING_NIFTI))
fixed_mask_ants = ants.image_read(str(FIXED_MASK_NIFTI))
moving_mask_reg_ants = ants.image_read(str(MOVING_MASK_REG_NIFTI))
fixed_reg_ants = ants.image_read(str(FIXED_REG_NIFTI))
moving_reg_ants = ants.image_read(str(MOVING_REG_NIFTI))
fixed_feat_ants = ants.image_read(str(FIXED_FEAT_NIFTI)) if USE_FEATURE_IMAGES else None
moving_feat_ants = ants.image_read(str(MOVING_FEAT_NIFTI)) if USE_FEATURE_IMAGES else None

initial_transform_for_syn: list[str] = [str(LINEAR_INIT_MAT_PATH)]

# Overlap mask in FIXED space (stabilizes MI on partial tissue)
moving_mask_in_fixed = ants.apply_transforms(
    fixed=fixed_mask_ants,
    moving=moving_mask_reg_ants,
    transformlist=initial_transform_for_syn,
    interpolator="nearestNeighbor",
)
overlap_mask_ants = fixed_mask_ants * moving_mask_in_fixed
ants.image_write(overlap_mask_ants, str(OVERLAP_MASK_NIFTI))
print(f"Wrote: {OVERLAP_MASK_NIFTI}")
print("Overlap frac (fixed grid):", float((overlap_mask_ants.numpy() > 0).mean()))

# Optionally crop the fixed registration target down to the overlap region to avoid
# optimizer getting "stuck" because most of the fixed frame has no corresponding tissue.
fixed_crop_lower = None
fixed_crop_upper = None
if CROP_FIXED_TO_OVERLAP:
    overlap_arr = overlap_mask_ants.numpy() > 0
    if np.any(overlap_arr):
        idx = np.argwhere(overlap_arr)
        lo_x, lo_y = idx.min(axis=0).tolist()
        hi_x, hi_y = idx.max(axis=0).tolist()
        pad = int(CROP_PAD_VOX)
        lo = [max(0, int(lo_x - pad)), max(0, int(lo_y - pad))]
        hi = [min(fixed_reg_ants.shape[0] - 1, int(hi_x + pad)), min(fixed_reg_ants.shape[1] - 1, int(hi_y + pad))]
        fixed_crop_lower = lo
        fixed_crop_upper = [hi[0] + 1, hi[1] + 1]

        fixed_reg_crop = ants.crop_indices(fixed_reg_ants, fixed_crop_lower, fixed_crop_upper)
        fixed_mask_crop = ants.crop_indices(overlap_mask_ants, fixed_crop_lower, fixed_crop_upper)
        ants.image_write(fixed_reg_crop, str(FIXED_REG_CROP_NIFTI))
        ants.image_write(fixed_mask_crop, str(FIXED_MASK_CROP_NIFTI))
        print(f"Wrote: {FIXED_REG_CROP_NIFTI}")
        print(f"Wrote: {FIXED_MASK_CROP_NIFTI}")

# IMPORTANT: do NOT run registration in the cropped fixed domain.
# Cropping changes the fixed image origin, which makes the resulting transform misaligned
# when later applied on the full fixed crop. Use a fixed-space overlap mask instead.
fixed_reg_syn = fixed_reg_ants
fixed_mask_syn = overlap_mask_ants

# %% [markdown]
# ## Phase 2: MI-based diffeomorphic refinement (SyN)

# %%
before_warped = ants.apply_transforms(
    fixed=fixed_ants,
    moving=moving_ants,
    transformlist=initial_transform_for_syn,
    interpolator="linear",
)
ants.image_write(before_warped, str(WARPED_BEFORE_NIFTI))

mi_before = float(ants.image_mutual_information(fixed_ants, before_warped))
print(f"MI before (init): {mi_before:.6f}")

kwargs: dict[str, object] = dict(
    fixed=fixed_reg_syn,
    moving=moving_reg_ants,
    type_of_transform=SYN_TYPE_OF_TRANSFORM,
    initial_transform=initial_transform_for_syn,
    grad_step=SYN_GRAD_STEP,
    flow_sigma=SYN_FLOW_SIGMA,
    total_sigma=SYN_TOTAL_SIGMA,
    syn_metric=SYN_METRIC,
    syn_sampling=SYN_SAMPLING,
    reg_iterations=SYN_REG_ITERATIONS,
    mask=fixed_mask_syn,
    moving_mask=moving_mask_reg_ants,
    mask_all_stages=True,
    random_seed=0,
    write_composite_transform=True,
    verbose=True,
    outprefix=str(SYN_PREFIX),
)

extras: list[tuple[str, ants.ANTsImage, ants.ANTsImage, float, int]] = []

if USE_FEATURE_IMAGES:
    assert fixed_feat_ants is not None
    assert moving_feat_ants is not None
    extras.append(("MeanSquares", fixed_feat_ants, moving_feat_ants, float(FEATURE_WEIGHT), 0))

if USE_MASK_DISTANCE_METRIC:
    fixed_dt_sitk = signed_distance(fixed_mask_sitk)
    moving_dt_sitk = signed_distance(moving_mask_reg_sitk)
    sitk.WriteImage(fixed_dt_sitk, str(FIXED_DT_NIFTI))
    sitk.WriteImage(moving_dt_sitk, str(MOVING_DT_NIFTI))
    fixed_dt_ants = ants.image_read(str(FIXED_DT_NIFTI))
    moving_dt_ants = ants.image_read(str(MOVING_DT_NIFTI))
    extras.append(("MeanSquares", fixed_dt_ants, moving_dt_ants, float(MASK_DISTANCE_WEIGHT), 0))

if USE_LANDMARK_HEATMAP_METRIC:
    fixed_lm = landmark_heatmap_from_points_mm(
        domain_img=fixed_reg_ants,
        points_mm=fixed_pts_mm,
        sigma_um=LANDMARK_HEATMAP_SIGMA_UM,
    )
    moving_lm = landmark_heatmap_from_points_mm(
        domain_img=moving_reg_ants,
        points_mm=moving_pts_mm,
        sigma_um=LANDMARK_HEATMAP_SIGMA_UM,
    )
    ants.image_write(fixed_lm, str(FIXED_LM_NIFTI))
    ants.image_write(moving_lm, str(MOVING_LM_NIFTI))
    extras.append(("MeanSquares", fixed_lm, moving_lm, float(LANDMARK_HEATMAP_WEIGHT), 0))

if extras:
    kwargs["multivariate_extras"] = extras

tx = ants.registration(**kwargs)

warped_after = ants.apply_transforms(
    fixed=fixed_ants,
    moving=moving_ants,
    transformlist=tx["fwdtransforms"],
    interpolator="linear",
)
ants.image_write(warped_after, str(WARPED_AFTER_NIFTI))

moving_mask_final_in_fixed = ants.apply_transforms(
    fixed=fixed_mask_ants,
    moving=moving_mask_reg_ants,
    transformlist=tx["fwdtransforms"],
    interpolator="nearestNeighbor",
)
ants.image_write(moving_mask_final_in_fixed, str(MOVING_MASK_WARPED_FINAL_NIFTI))
overlap_mask_final = fixed_mask_ants * moving_mask_final_in_fixed
ants.image_write(overlap_mask_final, str(OVERLAP_MASK_FINAL_NIFTI))
print(f"Wrote: {MOVING_MASK_WARPED_FINAL_NIFTI}")
print(f"Wrote: {OVERLAP_MASK_FINAL_NIFTI}")
print("Overlap frac after (fixed grid):", float((overlap_mask_final.numpy() > 0).mean()))

mi_after = float(ants.image_mutual_information(fixed_ants, warped_after))
print(f"MI after (init + SyN): {mi_after:.6f}")

lm_err = landmark_error_after_registration(
    moving_pts_mm=moving_pts_mm,
    fixed_pts_mm=fixed_pts_mm,
    invtransforms=tx["invtransforms"],
)
print(f"Post-SyN landmark RMSE: {lm_err['rmse_mm'] * 1e3:.2f} µm (max {lm_err['max_mm'] * 1e3:.2f} µm)")

qc_overlay_png(
    fixed_img=fixed_ants,
    moving_warped=warped_after,
    out_png=QC_PNG,
    title=f"SyN(MI) | MI {mi_before:.3f}→{mi_after:.3f} | lm_rmse={lm_err['rmse_mm']*1e3:.1f} µm",
)
qc_overlay_png(
    fixed_img=fixed_ants,
    moving_warped=warped_after,
    moving_mask_fixed=moving_mask_final_in_fixed,
    out_png=QC_MASKED_PNG,
    title=f"SyN(MI) (masked) | MI {mi_before:.3f}→{mi_after:.3f} | lm_rmse={lm_err['rmse_mm']*1e3:.1f} µm",
)

if fixed_crop_lower is not None and fixed_crop_upper is not None:
    fixed_zoom = ants.crop_indices(fixed_ants, fixed_crop_lower, fixed_crop_upper)
    moving_zoom = ants.crop_indices(warped_after, fixed_crop_lower, fixed_crop_upper)
    mask_zoom = ants.crop_indices(moving_mask_final_in_fixed, fixed_crop_lower, fixed_crop_upper)
    qc_overlay_png(
        fixed_img=fixed_zoom,
        moving_warped=moving_zoom,
        out_png=QC_ZOOM_PNG,
        title=f"SyN(MI) zoom | MI {mi_before:.3f}→{mi_after:.3f} | lm_rmse={lm_err['rmse_mm']*1e3:.1f} µm",
    )
    qc_overlay_png(
        fixed_img=fixed_zoom,
        moving_warped=moving_zoom,
        moving_mask_fixed=mask_zoom,
        out_png=QC_ZOOM_MASKED_PNG,
        title=f"SyN(MI) zoom (masked) | MI {mi_before:.3f}→{mi_after:.3f} | lm_rmse={lm_err['rmse_mm']*1e3:.1f} µm",
    )

summary = {
    "use_n4": USE_N4,
    "n4_apply_to_fixed": N4_APPLY_TO_FIXED if USE_N4 else None,
    "n4_max_iterations": list(N4_MAX_ITERATIONS) if USE_N4 else None,
    "landmark_linear_transform_type": LANDMARK_LINEAR_TRANSFORM_TYPE,
    "syn_type_of_transform": SYN_TYPE_OF_TRANSFORM,
    "syn_grad_step": SYN_GRAD_STEP,
    "syn_flow_sigma": SYN_FLOW_SIGMA,
    "syn_total_sigma": SYN_TOTAL_SIGMA,
    "syn_metric": SYN_METRIC,
    "syn_sampling": SYN_SAMPLING,
    "syn_reg_iterations": SYN_REG_ITERATIONS,
    "use_feature_images": USE_FEATURE_IMAGES,
    "feature_sigma_um": FEATURE_SIGMA_UM if USE_FEATURE_IMAGES else None,
    "feature_weight": FEATURE_WEIGHT if USE_FEATURE_IMAGES else None,
    "use_mask_distance_metric": USE_MASK_DISTANCE_METRIC,
    "mask_distance_weight": MASK_DISTANCE_WEIGHT if USE_MASK_DISTANCE_METRIC else None,
    "use_landmark_heatmap_metric": USE_LANDMARK_HEATMAP_METRIC,
    "landmark_heatmap_sigma_um": LANDMARK_HEATMAP_SIGMA_UM if USE_LANDMARK_HEATMAP_METRIC else None,
    "landmark_heatmap_weight": LANDMARK_HEATMAP_WEIGHT if USE_LANDMARK_HEATMAP_METRIC else None,
    "mi_before": mi_before,
    "mi_after": mi_after,
    "landmark_error_mm": lm_err,
    "fwdtransforms": tx["fwdtransforms"],
    "invtransforms": tx["invtransforms"],
    "initial_transform_for_syn": initial_transform_for_syn,
    "paths": {
        "linear_init_mat": str(LINEAR_INIT_MAT_PATH),
        "fixed_nifti": str(FIXED_NIFTI),
        "moving_nifti": str(MOVING_NIFTI),
        "overlap_mask_nifti": str(OVERLAP_MASK_NIFTI),
        "fixed_reg_nifti": str(FIXED_REG_NIFTI),
        "moving_reg_nifti": str(MOVING_REG_NIFTI),
        "fixed_reg_raw_nifti": str(FIXED_REG_RAW_NIFTI),
        "moving_reg_raw_nifti": str(MOVING_REG_RAW_NIFTI),
        "fixed_feature_nifti": str(FIXED_FEAT_NIFTI) if USE_FEATURE_IMAGES else None,
        "moving_feature_nifti": str(MOVING_FEAT_NIFTI) if USE_FEATURE_IMAGES else None,
        "moving_mask_reg_nifti": str(MOVING_MASK_REG_NIFTI),
        "fixed_mask_signed_distance_nifti": str(FIXED_DT_NIFTI) if USE_MASK_DISTANCE_METRIC else None,
        "moving_mask_signed_distance_nifti": str(MOVING_DT_NIFTI) if USE_MASK_DISTANCE_METRIC else None,
        "fixed_landmark_heatmap_nifti": str(FIXED_LM_NIFTI) if USE_LANDMARK_HEATMAP_METRIC else None,
        "moving_landmark_heatmap_nifti": str(MOVING_LM_NIFTI) if USE_LANDMARK_HEATMAP_METRIC else None,
        "warped_before_nifti": str(WARPED_BEFORE_NIFTI),
        "warped_after_nifti": str(WARPED_AFTER_NIFTI),
        "moving_mask_warped_final_nifti": str(MOVING_MASK_WARPED_FINAL_NIFTI),
        "overlap_mask_final_nifti": str(OVERLAP_MASK_FINAL_NIFTI),
        "qc_png": str(QC_PNG),
        "qc_masked_png": str(QC_MASKED_PNG),
        "qc_zoom_png": str(QC_ZOOM_PNG) if fixed_crop_lower is not None and fixed_crop_upper is not None else None,
        "qc_zoom_masked_png": str(QC_ZOOM_MASKED_PNG) if fixed_crop_lower is not None and fixed_crop_upper is not None else None,
    },
}
SUMMARY_JSON.write_text(json.dumps(summary, indent=2))
print(f"Wrote: {WARPED_BEFORE_NIFTI}")
print(f"Wrote: {WARPED_AFTER_NIFTI}")
print(f"Wrote: {MOVING_MASK_WARPED_FINAL_NIFTI}")
print(f"Wrote: {OVERLAP_MASK_FINAL_NIFTI}")
print(f"Wrote: {QC_PNG}")
print(f"Wrote: {QC_MASKED_PNG}")
if fixed_crop_lower is not None and fixed_crop_upper is not None:
    print(f"Wrote: {QC_ZOOM_PNG}")
    print(f"Wrote: {QC_ZOOM_MASKED_PNG}")
print(f"Wrote: {SUMMARY_JSON}")


# %%
