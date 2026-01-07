
# %% [markdown]
# # Draft: ANTs (ANTsPy) workflow for partial 2D section → (slice of) 3D atlas
#
# **Goal:** test an ANTs-based deformable refinement that is robust to:
# - **partial tissue** (unknown missing parts / torn edge / cropped section)
# - **cross-modality mismatch** (DAPI vs LSFM autofluorescence)
#
# **Reference:** DevCCF paper uses **ANTsPy** and reports using the `antsRegistrationSyNQuick` transform type with the **mutual information** similarity metric (except one mask-driven substep using mean squares).
#
# This notebook implements a practical, testable version:
# 1) load your atlas slice + sample section
# 2) load your saved landmarks JSON (from your Phase 1) and build an **initial similarity** in ANTs
# 3) compute an **overlap mask** (fixed-space) to prevent edge-driven warp
# 4) run ANTs deformable refinement:
#    - **Option A:** `antsRegistrationSyNQuick[so]` (SyN-only quick preset) on feature images (MI)
#    - **Option B:** `antsRegistrationSyN[so]` (SyN-only preset) + **multi-metric** (image MI + mask distance MeanSquares)
# 5) QC overlays + Dice + landmark error
#
# Notes:
# - ANTs/NIfTI spacing is conventionally in **mm**. This notebook converts µm→mm before handing data to ANTs.
# - In ANTs, “forward transforms” are defined as mapping **fixed→moving** (point mapping), and are used when resampling **moving into fixed**.
#   Therefore:
#   - to warp *moving image* into fixed space: use `tx['fwdtransforms']`
#   - to map *moving-space points* into fixed space: use `tx['invtransforms']`

# %% [markdown]
# ## Output Spec (Contract)
#
# This notebook writes all artifacts under the ROI-scoped output directory:
#
# - `OUTDIR == ws.ccf_transforms(ROI) == <workspace>/analysis/output/ccf-transforms/<roi>/`
#
# **Inputs (required)**
# - Landmarks JSON from `register_partial_section.py`:
#   - preferred: `p1_landmarks.json`
# - Required keys in the landmarks JSON:
#   - `prior_rotation_deg`
#   - `atlas_crop_bbox`, `sample_rotated_crop_bbox`
#   - `fixed_points_cropped_xy`, `moving_points_fullres_xy_in_rotated_crop`
#
# **Outputs (written)**
#
# Stable filenames:
# - `init_similarity_fixed2moving.mat` (ANTs similarity initialization, fixed→moving)
# - `fixed_coarse.nii.gz`, `moving_coarse.nii.gz`
# - `fixed_mask_eroded.nii.gz`, `moving_mask_eroded.nii.gz`
# - `overlap_mask_eroded.nii.gz`
# - `fixed_feat_gradmag.nii.gz`, `moving_feat_gradmag.nii.gz`
# - `fixed_dt.nii.gz`, `moving_dt.nii.gz`
#
# Per registration run `RunSpec(name=...)` the notebook writes:
# - `{name}_warped_raw.nii.gz`, `{name}_warped_mask.nii.gz`
# - `{name}_qc.png`
# - `{name}_summary.json` (includes `fwdtransforms` and `invtransforms` lists)
# - plus ANTs registration sidecar outputs written via `outprefix` (composite transform `.h5` etc.)
#
# After selecting the best run, it also writes:
# - `{best_name}_before_after_diff.png`

# %% [markdown]
# ## 0) Install / imports

# %%
# If needed (restart kernel after install):
# !pip install -U antspyx

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import ants  # antspyx, imported as `ants`
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
# ## 1) Configuration
#
# This notebook uses the same inputs/paths as `register_partial_section.py`.
#
# Prereq: run `register_partial_section.py` at least through p1 so
# `OUTDIR/p1_landmarks.json` exists (for `prior_rotation_deg` + crop bboxes).
#
# Edit the same config block below in both notebooks.

# %%
# === EDIT THESE (mirrors register_partial_section.py) ===

# Workspace configuration
WORKSPACE = Path("/home/chaichontat/fishtools2/working/20250929_JaxA3_Coro4")
ROI = "3"
STITCH_CODEBOOK = "pi"  # analysis/deconv/stitch--{ROI}+{STITCH_CODEBOOK}/fused.zarr

# Atlas configuration
ATLAS_NAME = "kim_dev_mouse_e15-5_lsfm_20um"
ATLAS_SLICE_IDX = 600 - 382  # Coronal slice index

# Sample configuration (resolved via Workspace)
ws = Workspace(WORKSPACE)
SAMPLE_ZARR = ws.stitch(ROI, STITCH_CODEBOOK) / "fused.zarr"
SAMPLE_Z_IDX = 5
SAMPLE_CHANNEL = "pi"

# Voxel sizes (µm)
SAMPLE_VOXEL_XY = 0.2
ATLAS_VOXEL = 20.0

# Output directory (analysis/output/ccf-transforms/{ROI}/)
OUTDIR = ws.ccf_transforms(ROI)
OUTDIR.mkdir(parents=True, exist_ok=True)

OUT = LandmarkRegistrationOutputs(OUTDIR)

# Common spacing for ANTs refinement (DevCCF used 20 µm for mapping compute/size reasons).
TARGET_SPACING_UM = 10.0

# Feature scale for cross-modality robustness (tune 50–200 µm depending on age + staining)
FEATURE_SIGMA_UM = 80.0

# Mask edge guard to avoid boundary-driven deformations (partial tissue!)
EDGE_GUARD_UM = 200.0

# Pick which ANTs flavors to test
RUN_QSYN_SO = True         # antsRegistrationSyNQuick[so] : quick SyN-only
RUN_SYN_SO_MULTIMETRIC = True  # antsRegistrationSyN[so] (or SyNOnly) with multivariate_extras

RANDOM_SEED = 1

# %% [markdown]
# ## 2) Bring in your images

# %%
print("Loading p1 landmarks...")
if not OUT.p1_landmarks_json.exists():
    raise FileNotFoundError(
        f"Missing {OUT.p1_landmarks_json}. Run register_partial_section.py through p1 first."
    )
p1 = OUT.read_p1_landmarks()
PRIOR_ROTATION_DEG = p1.prior_rotation_deg
atlas_crop_bbox = p1.atlas_crop_bbox
sample_rotated_crop_bbox = p1.sample_rotated_crop_bbox

print(f"  prior_rotation_deg={PRIOR_ROTATION_DEG}")
print(f"  atlas_crop_bbox={atlas_crop_bbox}")
print(f"  sample_rotated_crop_bbox={sample_rotated_crop_bbox}")

print("Loading atlas...")
atlas = BrainGlobeAtlas(ATLAS_NAME)
print(f"  Atlas shape: {atlas.reference.shape}, resolution: {atlas.resolution} µm")

atlas_slice_full = atlas.reference[ATLAS_SLICE_IDX, :, :]
atlas_annotation_full = atlas.annotation[ATLAS_SLICE_IDX, :, :]
atlas_brain_mask = atlas_annotation_full > 0

atlas_slice_full_masked = atlas_slice_full.copy().astype(np.float32)
atlas_slice_full_masked[~atlas_brain_mask] = 0.0

ar0, ar1, ac0, ac1 = atlas_crop_bbox
atlas_slice = atlas_slice_full_masked[ar0:ar1, ac0:ac1]
atlas_mask = atlas_brain_mask[ar0:ar1, ac0:ac1]

print("Loading sample...")
arr = zarr.open(str(SAMPLE_ZARR), mode="r")
keys = list(arr.attrs.get("key", []))
if SAMPLE_CHANNEL in keys:
    ch_idx = keys.index(SAMPLE_CHANNEL)
else:
    ch_idx = 0
    print(f"  WARNING: channel {SAMPLE_CHANNEL!r} not found in attrs.key={keys}; using ch_idx=0")

sample_slice_full_raw = np.asarray(arr[SAMPLE_Z_IDX, :, :, ch_idx])
if PRIOR_ROTATION_DEG != 0:
    sample_slice_full = ndimage_rotate(sample_slice_full_raw, PRIOR_ROTATION_DEG, reshape=True, order=1)
    print(f"  Rotated sample by {PRIOR_ROTATION_DEG}°: {sample_slice_full_raw.shape} -> {sample_slice_full.shape}")
else:
    sample_slice_full = sample_slice_full_raw

sr0, sr1, sc0, sc1 = sample_rotated_crop_bbox
sample_slice_rotated = sample_slice_full[sr0:sr1, sc0:sc1]

# Work entirely in CROPPED coordinates (same frame as landmark clicks).
fixed_mask_np = atlas_mask.astype(np.uint8)
fixed_np = atlas_slice.astype(np.float32)
moving_np = sample_slice_rotated.astype(np.float32)

print(f"Fixed shape: {fixed_np.shape} (voxel={ATLAS_VOXEL} µm), Moving shape: {moving_np.shape} (voxel={SAMPLE_VOXEL_XY} µm)")

# %% [markdown]
# ## 3) Helpers (µm→mm, SimpleITK resampling, masks, features)

# %%
def ants_numpy_yx(img: ants.ANTsImage) -> np.ndarray:
    """Convert ANTsPy image to a numpy array in (Y, X) order for matplotlib/SimpleITK."""

    arr_xy = np.asarray(img.numpy())
    if arr_xy.ndim != 2:
        raise ValueError(f"Expected 2D ANTsImage, got shape={arr_xy.shape}.")
    return arr_xy.T

# %% [markdown]
# ## 4) Load landmarks JSON and create an ANTs initial similarity transform
#
# DevCCF mentions landmark-assisted refinements in their multimodal 3D alignment (manual identification then additional mask-driven steps).
#
# Here we build an ANTs similarity from your clicked landmark pairs and save it as a `.mat` so it can be used as `initial_transform` in `ants.registration`.

# %%
fixed_points_cropped_xy = p1.fixed_points_cropped_xy
moving_points_fullres_xy_in_rot_crop = p1.moving_points_fullres_xy_in_rotated_crop

# (r0,r1,c0,c1) in full atlas slice / full rotated sample
sample_crop_bbox = sample_rotated_crop_bbox

# We intentionally work in CROPPED coordinates (same frame as landmark clicks),
# so we do NOT apply crop offsets here.
expected_fixed_shape = (int(atlas_crop_bbox[1] - atlas_crop_bbox[0]), int(atlas_crop_bbox[3] - atlas_crop_bbox[2]))
expected_moving_shape = (int(sample_crop_bbox[1] - sample_crop_bbox[0]), int(sample_crop_bbox[3] - sample_crop_bbox[2]))
if fixed_np.shape != expected_fixed_shape:
    raise ValueError(
        f"fixed_np shape {fixed_np.shape} does not match atlas_crop_bbox {atlas_crop_bbox} -> {expected_fixed_shape}. "
        "Re-run section 2."
    )
if moving_np.shape != expected_moving_shape:
    raise ValueError(
        f"moving_np shape {moving_np.shape} does not match sample_rotated_crop_bbox {sample_crop_bbox} -> {expected_moving_shape}. "
        "Re-run section 2."
    )

# Convert to physical coordinates IN mm (important)
fixed_pts_mm = np.array(
    [(x * ATLAS_VOXEL * UM_TO_MM, y * ATLAS_VOXEL * UM_TO_MM) for x, y in fixed_points_cropped_xy], dtype=np.float64
)
moving_pts_mm = np.array(
    [(x * SAMPLE_VOXEL_XY * UM_TO_MM, y * SAMPLE_VOXEL_XY * UM_TO_MM) for x, y in moving_points_fullres_xy_in_rot_crop],
    dtype=np.float64,
)

print("Landmarks:", fixed_pts_mm.shape[0])
print("Fixed pts (mm) first:", fixed_pts_mm[0], "Moving pts (mm) first:", moving_pts_mm[0])

# Build initial similarity (fixed -> moving) using ANTs landmarks
init_tx = ants.fit_transform_to_paired_points(
    moving_points=moving_pts_mm,
    fixed_points=fixed_pts_mm,
    transform_type="similarity",
)

init_mat_path = OUTDIR / "init_similarity_fixed2moving.mat"
ants.write_transform(init_tx, str(init_mat_path))
print("Wrote initial transform:", init_mat_path)

# Validate landmark fit quickly (apply to fixed pts; should land near moving pts)
pred = np.vstack([np.array(init_tx.apply_to_point(tuple(p))) for p in fixed_pts_mm])
errs = np.linalg.norm(pred - moving_pts_mm, axis=1)
print("Init similarity landmark RMSE (mm):", float(np.sqrt(np.mean(errs**2))), "max:", float(errs.max()))

# %% [markdown]
# ## 5) Build coarse images, masks, overlap mask, and feature images

# %%
fixed_img = sitk_from_numpy_2d(normalize_robust(fixed_np), spacing_um=ATLAS_VOXEL)
moving_img = sitk_from_numpy_2d(normalize_robust(moving_np), spacing_um=SAMPLE_VOXEL_XY)

fixed_coarse = resample_sitk_to_spacing(fixed_img, TARGET_SPACING_UM, sitk.sitkLinear)
moving_coarse = resample_sitk_to_spacing(moving_img, TARGET_SPACING_UM, sitk.sitkLinear)

# Masks at coarse scale
fixed_mask_img = sitk_from_numpy_2d(fixed_mask_np.astype(np.float32), spacing_um=ATLAS_VOXEL)
fixed_mask_coarse = resample_sitk_to_spacing(fixed_mask_img, TARGET_SPACING_UM, sitk.sitkNearestNeighbor)
fixed_mask_coarse = sitk.Cast(fixed_mask_coarse > 0, sitk.sitkUInt8)

moving_mask_coarse = make_moving_mask_sitk(moving_coarse)

# Erode masks (guard band against edge-driven warps)
fixed_mask_eroded = erode_by_um(fixed_mask_coarse, EDGE_GUARD_UM)
moving_mask_eroded = erode_by_um(moving_mask_coarse, EDGE_GUARD_UM)

# Write to disk as NIfTI then read via ANTs (avoids numpy axis-order pitfalls)
fixed_nifti = OUTDIR / "fixed_coarse.nii.gz"
moving_nifti = OUTDIR / "moving_coarse.nii.gz"
fixed_mask_nifti = OUTDIR / "fixed_mask_eroded.nii.gz"
moving_mask_nifti = OUTDIR / "moving_mask_eroded.nii.gz"

sitk.WriteImage(fixed_coarse, str(fixed_nifti))
sitk.WriteImage(moving_coarse, str(moving_nifti))
sitk.WriteImage(fixed_mask_eroded, str(fixed_mask_nifti))
sitk.WriteImage(moving_mask_eroded, str(moving_mask_nifti))

fixed_ants = ants.image_read(str(fixed_nifti))
moving_ants = ants.image_read(str(moving_nifti))
fixed_mask_ants = ants.image_read(str(fixed_mask_nifti))
moving_mask_ants = ants.image_read(str(moving_mask_nifti))

# Build overlap mask in FIXED space by warping moving mask into fixed using initial similarity
moving_mask_in_fixed = ants.apply_transforms(
    fixed=fixed_mask_ants,
    moving=moving_mask_ants,
    transformlist=[str(init_mat_path)],
    interpolator="nearestNeighbor",
)
overlap = fixed_mask_ants * moving_mask_in_fixed
# One more erosion on overlap (often helps with partial tissue)
# Convert overlap->sitk for erosion, then back
overlap_sitk = sitk.ReadImage(str(OUTDIR / "fixed_mask_eroded.nii.gz"))  # spacing/origin
overlap_sitk = sitk.Cast(overlap_sitk > 0, sitk.sitkUInt8)
# Replace with true overlap content
overlap_arr = (ants_numpy_yx(overlap) > 0).astype(np.uint8)
overlap_ref_arr = sitk.GetArrayFromImage(overlap_sitk)
if overlap_arr.shape != overlap_ref_arr.shape:
    if overlap_arr.T.shape == overlap_ref_arr.shape:
        overlap_arr = overlap_arr.T
    else:
        raise ValueError(
            f"Unexpected overlap array shape {overlap_arr.shape} vs reference {overlap_ref_arr.shape}."
        )
overlap_sitk2 = sitk.GetImageFromArray(overlap_arr)
overlap_sitk2.CopyInformation(overlap_sitk)

overlap_sitk2 = erode_by_um(overlap_sitk2, EDGE_GUARD_UM)
overlap_mask_nifti = OUTDIR / "overlap_mask_eroded.nii.gz"
sitk.WriteImage(overlap_sitk2, str(overlap_mask_nifti))
overlap_mask_ants = ants.image_read(str(overlap_mask_nifti))

print("Overlap frac (fixed grid):", float((overlap_mask_ants.numpy() > 0).mean()))

# Feature images (cross-modality stabilization)
fixed_feat = gradmag_feature(fixed_coarse, FEATURE_SIGMA_UM)
moving_feat = gradmag_feature(moving_coarse, FEATURE_SIGMA_UM)

fixed_feat_nifti = OUTDIR / "fixed_feat_gradmag.nii.gz"
moving_feat_nifti = OUTDIR / "moving_feat_gradmag.nii.gz"
sitk.WriteImage(fixed_feat, str(fixed_feat_nifti))
sitk.WriteImage(moving_feat, str(moving_feat_nifti))

fixed_feat_ants = ants.image_read(str(fixed_feat_nifti))
moving_feat_ants = ants.image_read(str(moving_feat_nifti))

# Distance transforms for optional multi-metric term
fixed_dt = signed_distance(fixed_mask_eroded)
moving_dt = signed_distance(moving_mask_eroded)

fixed_dt_nifti = OUTDIR / "fixed_dt.nii.gz"
moving_dt_nifti = OUTDIR / "moving_dt.nii.gz"
sitk.WriteImage(fixed_dt, str(fixed_dt_nifti))
sitk.WriteImage(moving_dt, str(moving_dt_nifti))

fixed_dt_ants = ants.image_read(str(fixed_dt_nifti))
moving_dt_ants = ants.image_read(str(moving_dt_nifti))

# %% [markdown]
# ## 6) Registration runner + QC
#
# - `ants.registration` supports `mask` (fixed-space) and `moving_mask` (moving-space); `mask_all_stages=True` applies masks to all stages.
# - `type_of_transform` supports presets including `antsRegistrationSyNQuick[x]` and `antsRegistrationSyN[x]`.
# - Multi-metric deformable stages can be driven by `multivariate_extras` (only compatible with SyNOnly / antsRegistrationSyN* per docs).

# %%
@dataclass
class RunSpec:
    name: str
    type_of_transform: str
    use_features: bool
    use_multimetric: bool

def dice(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    denom = a.sum() + b.sum()
    return float(2 * inter / denom) if denom > 0 else 0.0

def qc_plot(fixed_img: ants.ANTsImage, moving_warped: ants.ANTsImage, title: str, out_png: Path) -> None:
    f = ants_numpy_yx(fixed_img).astype(np.float32)
    m = ants_numpy_yx(moving_warped).astype(np.float32)
    f = normalize_robust(f)
    m = normalize_robust(m)

    overlay = np.zeros((f.shape[0], f.shape[1], 3), dtype=np.float32)
    overlay[..., 0] = f
    overlay[..., 1] = m
    overlay[..., 2] = f

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(f, cmap="gray")
    axes[0].set_title("Fixed")
    axes[0].axis("off")
    axes[1].imshow(m, cmap="gray")
    axes[1].set_title("Warped moving")
    axes[1].axis("off")
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay (magenta=fixed, green=moving)")
    axes[2].axis("off")
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.show()

def landmark_error_after_registration(tx: dict[str, Any]) -> dict[str, float]:
    # Map moving points -> fixed points using INV transforms (point mapping is opposite of image resampling).
    df = pd.DataFrame(moving_pts_mm, columns=["x", "y"])
    # Some ANTs functions want z/t even in 2D dataframes; keep harmless columns.
    df["z"] = 0.0
    df["t"] = 0.0
    df_out = ants.apply_transforms_to_points(
        dim=2,
        points=df,
        transformlist=tx["invtransforms"],
    )
    pred_fixed = df_out[["x", "y"]].to_numpy(dtype=np.float64)
    err = np.linalg.norm(pred_fixed - fixed_pts_mm, axis=1)
    return {
        "rmse_mm": float(np.sqrt(np.mean(err**2))),
        "max_mm": float(err.max()),
        "mean_mm": float(err.mean()),
    }

def run_registration(spec: RunSpec) -> dict[str, Any]:
    prefix = str(OUTDIR / f"{spec.name}_")
    if spec.use_features:
        fixed_reg = fixed_feat_ants
        moving_reg = moving_feat_ants
    else:
        fixed_reg = fixed_ants
        moving_reg = moving_ants

    # Mask: use fixed-space overlap erosion to stabilize partial tissue alignment
    mask = overlap_mask_ants

    kwargs: dict[str, Any] = dict(
        fixed=fixed_reg,
        moving=moving_reg,
        type_of_transform=spec.type_of_transform,
        initial_transform=[str(init_mat_path)],  # prepend similarity
        outprefix=prefix,
        mask=mask,
        mask_all_stages=True,
        random_seed=RANDOM_SEED,  # improves reproducibility
        write_composite_transform=True,  # writes composite .h5
        verbose=True,
    )

    # Multi-metric term:
    # DevCCF describes multi-metric steps in their 2D mapping pipeline and mask-driven steps.
    # ANTsPy `multivariate_extras` expects (metricName, fixedImg2, movingImg2, weight, metricParam)
    # and is documented as compatible with SyNOnly or antsRegistrationSyN* transforms.
    if spec.use_multimetric:
        kwargs["multivariate_extras"] = [
            ("MeanSquares", fixed_dt_ants, moving_dt_ants, 0.5, 0),
        ]

    tx = ants.registration(**kwargs)

    # Warp raw moving into raw fixed space for QC
    warped_raw = ants.apply_transforms(
        fixed=fixed_ants,
        moving=moving_ants,
        transformlist=tx["fwdtransforms"],
        interpolator="linear",
    )
    warped_mask = ants.apply_transforms(
        fixed=fixed_mask_ants,
        moving=moving_mask_ants,
        transformlist=tx["fwdtransforms"],
        interpolator="nearestNeighbor",
    )

    # Metrics
    d = dice((fixed_mask_ants.numpy() > 0), (warped_mask.numpy() > 0))
    lm_err = landmark_error_after_registration(tx)

    # Save warped outputs
    ants.image_write(warped_raw, prefix + "warped_raw.nii.gz")
    ants.image_write(warped_mask, prefix + "warped_mask.nii.gz")

    # QC plots
    qc_plot(
        fixed_img=fixed_ants,
        moving_warped=warped_raw,
        title=f"{spec.name} | dice={d:.3f} | lm_rmse={lm_err['rmse_mm']*1e3:.1f} µm",
        out_png=OUTDIR / f"{spec.name}_qc.png",
    )

    summary = {
        "spec": asdict(spec),
        "dice_mask": d,
        "landmark_error_mm": lm_err,
        "fwdtransforms": tx["fwdtransforms"],
        "invtransforms": tx["invtransforms"],
    }
    (OUTDIR / f"{spec.name}_summary.json").write_text(json.dumps(summary, indent=2))
    return summary

# %% [markdown]
# ## 7) Run experiments

# %%
summaries = []

if RUN_QSYN_SO:
    # Quick SyN-only preset, initialized by your similarity transform
    # This mirrors the DevCCF usage of antsRegistrationSyNQuick + MI.
    summaries.append(
        run_registration(
            RunSpec(
                name="A_qsynQuick_so_features_MI_mask",
                type_of_transform="antsRegistrationSyNQuick[so]",
                use_features=True,
                use_multimetric=False,
            )
        )
    )

if RUN_SYN_SO_MULTIMETRIC:
    # Full SyN-only preset (not "Quick"), to enable multivariate_extras as documented.
    summaries.append(
        run_registration(
            RunSpec(
                name="B_syn_so_features_MI_plus_maskDT_MSQ",
                type_of_transform="antsRegistrationSyN[so]",
                use_features=True,
                use_multimetric=True,
            )
        )
    )

print("Done. Wrote outputs to:", OUTDIR.resolve())

# %% [markdown]
# ## 8) QC: difference before vs after ANTs registration
#
# Compare:
# - **before**: initial landmark-based similarity only
# - **after**: best ANTs run (by Dice) or a manually selected spec
#
# Outputs a PNG under `OUTDIR`.

# %%
if not summaries:
    raise RuntimeError("No summaries found; run section 7 first.")

best = max(summaries, key=lambda s: s["dice_mask"])
AFTER_SPEC_NAME = best["spec"]["name"]
print(f"Using AFTER_SPEC_NAME={AFTER_SPEC_NAME!r} (best dice={best['dice_mask']:.3f})")

before_warped_raw = ants.apply_transforms(
    fixed=fixed_ants,
    moving=moving_ants,
    transformlist=[str(init_mat_path)],
    interpolator="linear",
)

after_warped_path = OUTDIR / f"{AFTER_SPEC_NAME}_warped_raw.nii.gz"
after_warped_raw = ants.image_read(str(after_warped_path))

fixed_n = normalize_robust(ants_numpy_yx(fixed_ants).astype(np.float32))
before_n = normalize_robust(ants_numpy_yx(before_warped_raw).astype(np.float32))
after_n = normalize_robust(ants_numpy_yx(after_warped_raw).astype(np.float32))

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
axes[0, 2].set_title(f"After ({AFTER_SPEC_NAME})")
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
out_png = OUTDIR / f"{AFTER_SPEC_NAME}_before_after_diff.png"
plt.savefig(out_png, dpi=160)
plt.show()
print("Wrote:", out_png)

# %% [markdown]
# ## 9) (Optional) Mapping detected cell coordinates (moving → fixed)
#
# ANTs has a key convention:
# - forward transforms map **fixed→moving** points and are used to resample moving→fixed images
# - inverse transforms map **moving→fixed** points
#
# So to map (x,y) coordinates detected in your **moving** section into the atlas slice (**fixed**),
# use `invtransforms`.

# %%
# Example usage: replace with your detected cells (in moving physical units, mm)
# cells_moving_mm = np.array([[1.23, 4.56], [2.34, 5.67]], dtype=np.float64)

def map_moving_points_to_fixed(cells_moving_mm: np.ndarray, invtransforms: list[str]) -> np.ndarray:
    df = pd.DataFrame(cells_moving_mm, columns=["x", "y"])
    df["z"] = 0.0
    df["t"] = 0.0
    out = ants.apply_transforms_to_points(dim=2, points=df, transformlist=invtransforms)
    return out[["x", "y"]].to_numpy(dtype=np.float64)

# Pick best run by Dice (or by your visual QC)
if summaries:
    best = max(summaries, key=lambda s: s["dice_mask"])
    print("Best:", best["spec"]["name"], "Dice:", best["dice_mask"])
    # cells_fixed_mm = map_moving_points_to_fixed(cells_moving_mm, best["invtransforms"])
    # print(cells_fixed_mm[:5])

# %% [markdown]
# ## 10) Notes / pitfalls specific to partial tissue + ANTs
#
# 1) **Spacing units:** write NIfTI with spacing in **mm**, not µm, or ANTs will interpret 20 µm as 20 mm.
# 2) **Masking:** use an overlap-style mask (warp moving mask into fixed with init transform, AND, then erode).
# 3) **Transform direction confusion:** forward transforms are fixed→moving (point mapping); inverse transforms are moving→fixed.
# 4) **Multi-metric:** `multivariate_extras` is documented as compatible with SyNOnly / antsRegistrationSyN* (not necessarily SyNQuick).
# 5) **Cross-modality:** if raw MI is unstable, register feature images (e.g., gradient magnitude at 50–200 µm scale).
#
# If the deformation is still “wrong but smooth”, the most common fixes are:
# - increase `EDGE_GUARD_UM`
# - increase `FEATURE_SIGMA_UM`
# - reduce deformation aggressiveness (prefer `[so]` SyN-only initialized by similarity; avoid extra affine stages)
# - test `antsRegistrationSyNQuickRepro[so]` for better reproducibility/debugging
