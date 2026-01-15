# %% [markdown]
# # Landmark → linear init + diffeomorphic SyN refinement (MI, ANTsPy)
#
# Replaces the residual B-spline displacement (non-diffeomorphic) with an ANTs SyN-family
# registration (diffeomorphic non-rigid warp) driven by Mattes mutual information.
#
# High level:
# 1) Fit a landmark-driven linear transform (fixed→moving) using
#    `ants.fit_transform_to_paired_points(...)`.
# 2) Run diffeomorphic registration (SyN) initialized by that similarity, using MI.
#
# Outputs are written under:
# `<workspace>/analysis/output/ccf-transforms/<roi>/`.

# %%
from __future__ import annotations

import json
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import ants
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk
import zarr
from brainglobe_atlasapi import BrainGlobeAtlas
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import rotate as ndimage_rotate

from fishtools.ccf.landmark import LandmarkRegistrationOutputs
from fishtools.ccf.sitk_utils import (
    UM_TO_MM,
    dilate_by_um,
    erode_by_um,
    gradmag_feature,
    largest_cc,
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
WORKSPACE = Path("/home/chaichontat/nvme/20251005_JaxA3_Coro2")
ws = Workspace(WORKSPACE)
ROI = ws.rois[1]

STITCH_CODEBOOK = "pi"  # analysis/deconv/stitch--{ROI}+{STITCH_CODEBOOK}/fused.zarr

# Diffeomorphic registration knobs
# Use "SyNOnly" so `syn_metric="mattes"` (MI) is honored; some antsRegistrationSyN* presets
# hard-code CC as the deformable metric.
SYN_TYPE_OF_TRANSFORM = "SyNOnly"  # diffeomorphic non-rigid
SYN_METRIC = "mattes"  # MI
# Metric parameter (per ANTsPy `ants.registration`): for Mattes MI this is `nbins`, for CC this is `radius`.
SYN_METRIC_PARAM = 48
SYN_REG_ITERATIONS = (300, 150, 70, 30)
# Mask erosion guards (in µm). Fixed can stay conservative; moving should be looser for partial tissue.
FIXED_EDGE_GUARD_UM = 0.0
# User preference: do not erode the moving mask; use the full nonzero support.
MOVING_EDGE_GUARD_UM = 0.0
# Regularize SyN to avoid over-warping on partial tissue / cross-modality mismatch.
SYN_GRAD_STEP = 0.2
SYN_FLOW_SIGMA = 3
SYN_TOTAL_SIGMA = 1

# Optional moving-image pre-smoothing before normalization/feature construction (helps noisy partial tissue).
MOVING_PRESMOOTH_SIGMA_UM = 30.0

# Optional affine refinement (image-driven) after landmark init and before SyN.
# This helps reduce global mismatches so the diffeomorphic stage doesn't need to shear internal anatomy.
USE_AFFINE_REFINE = False
AFFINE_REFINE_TYPE_OF_TRANSFORM = "Rigid"
AFFINE_REFINE_METRIC = "mattes"
# For Mattes MI in ANTsPy this is the number of histogram bins (not the number of sampled points).
AFFINE_REFINE_MI_BINS = 64
# Fraction of points used to estimate the affine metric (controls stochastic sampling density).
AFFINE_REFINE_RANDOM_SAMPLING_RATE = 0.2
AFFINE_REFINE_ITERATIONS = (200, 100, 50, 20)
AFFINE_REFINE_SHRINK_FACTORS = (4, 2, 1, 1)
AFFINE_REFINE_SMOOTHING_SIGMAS = (2, 1, 0, 0)
AFFINE_REFINE_GRAD_STEP = 0.1

# Moving mask construction.
# Otsu can be overly conservative on low-signal edges; for partial slices we prefer
# using the full nonzero support (after robust normalization) as the mask domain.
MOVING_MASK_NONZERO = True

# Landmark-driven linear initialization.
# NOTE: For ANTs image resampling (`ants.apply_transforms`) and `ants.registration(initial_transform=...)`,
# the transform should map **fixed→moving** (the resampler internally pulls from the moving image).
# "affine" often helps when the ROI is a partial/warped view relative to atlas space.
LANDMARK_LINEAR_TRANSFORM_TYPE = "affine"  # "similarity" | "affine"

# Bias-field correction (helps MI for uneven illumination / shading)
USE_N4 = True
N4_APPLY_TO_FIXED = True
N4_MAX_ITERATIONS = (100, 100, 50, 20)

# Use feature images (gradient magnitude) for cross-modality stability.
USE_FEATURE_IMAGES = False
FEATURE_SIGMA_UM = 60.0
FEATURE_WEIGHT = 0.
# For multivariate_extras: use local CC on the feature channel (more tolerant than MeanSquares).
# In ANTsPy/ANTs, for CC the "sampling" field is the neighborhood radius (in voxels).
FEATURE_CC_RADIUS = 2

# Inject coarse shape via signed distance maps of masks.
USE_MASK_DISTANCE_METRIC = True
MASK_DISTANCE_WEIGHT = 0.1
# Optional smoothing of the binary masks *before* computing signed distance. This can reduce
# jagged boundary gradients that sometimes drive localized over-warping.
MASK_DISTANCE_SMOOTH_SIGMA_UM = 0.0
# For partial tissue, compute the fixed signed-distance map from the *overlap* mask
# (fixed brain ∩ moving wedge after linear init) so boundary gradients match what exists in the moving image.
MASK_DISTANCE_USE_OVERLAP = False

# Crop the fixed image to where the warped moving mask lands (helps partial-overlap registration).
CROP_FIXED_TO_OVERLAP = True
CROP_PAD_VOX = 24
# IMPORTANT: Cropping the fixed domain changes the spatial support of the resulting warp field
# (the displacement field is only defined on the cropped grid). If you need a globally smooth
# transform on the full fixed grid, prefer keeping `CROP_FIXED_DOMAIN_FOR_SYN=False` and relying
# on `mask` + `moving_mask` to restrict the metric.
CROP_FIXED_DOMAIN_FOR_SYN = False

# Dilate the fixed mask into the background so MI penalizes moving tissue spilling into fixed "void".
FIXED_METRIC_MASK_DILATE_UM = 75.0
# Dilate the moving mask into the background so MI penalizes fixed tissue "void" mismatch too.
MOVING_METRIC_MASK_DILATE_UM = 75.0

# Landmark injection (as an extra image metric via Gaussian heatmaps)
USE_LANDMARK_HEATMAP_METRIC = True
LANDMARK_HEATMAP_SIGMA_UM = 100
LANDMARK_HEATMAP_WEIGHT = 0.5

ws = Workspace(WORKSPACE)
WS_OUTDIR = ws.ccf_transforms(ROI)
OUTDIR = WS_OUTDIR / "landmark_syn_mi"
OUTDIR.mkdir(parents=True, exist_ok=True)

OUT = LandmarkRegistrationOutputs(WS_OUTDIR)
p1 = OUT.read_p1_landmarks()

P1_THRESHOLD_JSON = OUT.root / "p1_threshold.json"
if not P1_THRESHOLD_JSON.exists():
    raise FileNotFoundError(
        f"Missing {P1_THRESHOLD_JSON}. Run ccf/register_partial_section.py through the threshold step "
        "to write p1_threshold.json."
    )
payload = json.loads(P1_THRESHOLD_JSON.read_text(encoding="utf-8"))
if not isinstance(payload, dict):
    raise ValueError(f"Invalid p1_threshold.json at {P1_THRESHOLD_JSON}: expected a JSON object.")
threshold = payload.get("threshold")
if threshold is None:
    raise ValueError(f"Invalid p1_threshold.json at {P1_THRESHOLD_JSON}: missing 'threshold'.")
P1_THRESHOLD = float(threshold)
if "prior_rotation_deg" in payload and int(payload["prior_rotation_deg"]) != int(p1.prior_rotation_deg):
    raise ValueError(
        f"p1_threshold.json prior_rotation_deg={payload['prior_rotation_deg']!r} does not match "
        f"p1_landmarks.json prior_rotation_deg={p1.prior_rotation_deg!r}."
    )
if "prior_flip_x" in payload and bool(payload["prior_flip_x"]) != bool(p1.prior_flip_x):
    raise ValueError(
        f"p1_threshold.json prior_flip_x={payload['prior_flip_x']!r} does not match "
        f"p1_landmarks.json prior_flip_x={p1.prior_flip_x!r}."
    )
print(f"Loaded p1 threshold: {P1_THRESHOLD:g} ({P1_THRESHOLD_JSON})")

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
FIXED_MASK_ORIG_NIFTI = OUTDIR / "fixed_mask_crop_orig.nii.gz"
FIXED_METRIC_MASK_NIFTI = OUTDIR / "fixed_mask_metric_dilated.nii.gz"
MOVING_MASK_REG_FULL_NIFTI = OUTDIR / "moving_mask_reg_full.nii.gz"
MOVING_METRIC_MASK_NIFTI = OUTDIR / "moving_mask_metric_dilated.nii.gz"
OVERLAP_MASK_NIFTI = OUTDIR / "overlap_mask_eroded.nii.gz"
MOVING_REG_RAW_NIFTI = OUTDIR / "moving_sample_crop_reg_raw.nii.gz"
FIXED_REG_RAW_NIFTI = OUTDIR / "fixed_reg_raw.nii.gz"
MOVING_REG_NIFTI = OUTDIR / "moving_sample_crop_reg.nii.gz"
FIXED_REG_NIFTI = OUTDIR / "fixed_reg.nii.gz"
MOVING_MASK_REG_NIFTI = OUTDIR / "moving_mask_crop_reg.nii.gz"
FIXED_FEAT_NIFTI = OUTDIR / "fixed_feature_gradmag.nii.gz"
MOVING_FEAT_NIFTI = OUTDIR / "moving_feature_gradmag_reg.nii.gz"
FIXED_DT_NIFTI = OUTDIR / "fixed_mask_signed_distance.nii.gz"
MOVING_DT_NIFTI = OUTDIR / "moving_mask_signed_distance_reg.nii.gz"
FIXED_LM_NIFTI = OUTDIR / "fixed_landmark_heatmap.nii.gz"
MOVING_LM_NIFTI = OUTDIR / "moving_landmark_heatmap_reg.nii.gz"

SYN_PREFIX = OUTDIR / "final_syn_mi_"
AFFINE_PREFIX = OUTDIR / "affine_refine_"
WARPED_BEFORE_NIFTI = OUTDIR / "moving_warped_init.nii.gz"
WARPED_AFTER_NIFTI = OUTDIR / "moving_warped_similarity_plus_syn.nii.gz"
QC_PNG = OUTDIR / "similarity_plus_syn_qc.png"
QC_MASKED_PNG = OUTDIR / "similarity_plus_syn_qc_masked.png"
QC_ZOOM_PNG = OUTDIR / "similarity_plus_syn_qc_zoom.png"
QC_ZOOM_MASKED_PNG = OUTDIR / "similarity_plus_syn_qc_zoom_masked.png"
QC_MOVING_BBOX_PNG = OUTDIR / "moving_before_after_bbox.png"
DEFORMATION_FIELD_PNG = OUTDIR / "final_syn_mi_deformation_field.png"
DEBUG_MOVING_METRICS_OVERLAY_PNG = OUTDIR / "debug_moving_metrics_overlay.png"
DEBUG_MOVING_LANDMARK_HEATMAP_OVERLAY_PNG = OUTDIR / "debug_moving_landmark_heatmap_overlay.png"
# Deformation field plotting controls.
# Constraint: do not downsample the moving crop more than 8x relative to fused.zarr.
DEFORMATION_BG_MAX_DIM = 4096
DEFORMATION_FIELD_MAX_DIM = 1024
DEFORMATION_MAX_DOWNSAMPLE = 16.0
DEFORMATION_QUIVER_STEP = 12
DEFORMATION_SAVE_DPI = 300
DEFORMATION_MAX_FIG_IN = 18.0
QC_SAVE_DPI = 300
QC_VIS_SPACING_UM = 2.0
SUMMARY_JSON = OUTDIR / "similarity_plus_syn_summary.json"
MOVING_MASK_WARPED_FINAL_NIFTI = OUTDIR / "moving_mask_warped_final.nii.gz"
MOVING_MASK_METRIC_WARPED_FINAL_NIFTI = OUTDIR / "moving_mask_metric_warped_final.nii.gz"
OVERLAP_MASK_FINAL_NIFTI = OUTDIR / "overlap_mask_final.nii.gz"
MOVING_BBOX_PAD_VOX = 16

# %% [markdown]
# ## Helpers

# %%
def _parse_ants_diagnostic_metric_values(path: Path) -> tuple[list[int], list[float]]:
    iters: list[int] = []
    metric_values: list[float] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if "DIAGNOSTIC" not in line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3 or parts[1] in {"Iteration", ""}:
                continue
            try:
                it = int(parts[1])
                mv = float(parts[2])
            except ValueError:
                continue
            iters.append(it)
            metric_values.append(mv)
    return iters, metric_values


def _smooth_mask_for_signed_distance(*, mask: sitk.Image, sigma_um: float) -> sitk.Image:
    sigma_um = float(sigma_um)
    if sigma_um <= 0:
        return sitk.Cast(mask > 0, sitk.sitkUInt8)

    sigma_mm = sigma_um * UM_TO_MM
    sm = sitk.SmoothingRecursiveGaussian(sitk.Cast(mask > 0, sitk.sitkFloat32), sigma=sigma_mm)
    return sitk.Cast(sm >= 0.5, sitk.sitkUInt8)


def _masked_mutual_information_hist2d(
    *,
    fixed: np.ndarray,
    moving: np.ndarray,
    mask: np.ndarray,
    bins: int = 64,
) -> float:
    if fixed.shape != moving.shape or fixed.shape != mask.shape:
        raise ValueError(f"MI shape mismatch: fixed={fixed.shape}, moving={moving.shape}, mask={mask.shape}.")

    m = np.asarray(mask, dtype=bool)
    if int(m.sum()) < 64:
        return float("nan")

    f = np.asarray(fixed, dtype=np.float32)[m]
    g = np.asarray(moving, dtype=np.float32)[m]
    ok = np.isfinite(f) & np.isfinite(g)
    f = f[ok]
    g = g[ok]
    if f.size < 64:
        return float("nan")

    f_lo, f_hi = np.quantile(f, (0.005, 0.995)).tolist()
    g_lo, g_hi = np.quantile(g, (0.005, 0.995)).tolist()
    if not (np.isfinite(f_lo) and np.isfinite(f_hi) and np.isfinite(g_lo) and np.isfinite(g_hi)):
        return float("nan")
    if f_hi <= f_lo or g_hi <= g_lo:
        return float("nan")

    h, _, _ = np.histogram2d(
        f,
        g,
        bins=int(bins),
        range=((float(f_lo), float(f_hi)), (float(g_lo), float(g_hi))),
    )
    if float(h.sum()) <= 0:
        return float("nan")

    pxy = h / float(h.sum())
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)
    denom = px * py
    nz = pxy > 0
    mi = float(np.sum(pxy[nz] * np.log(pxy[nz] / denom[nz])))
    return mi


@contextmanager
def _redirect_fds(*, stdout_path: Path, stderr_path: Path | None = None) -> Iterator[None]:
    sys.stdout.flush()
    sys.stderr.flush()
    stdout_fd = os.open(str(stdout_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    if stderr_path is None:
        stderr_fd = stdout_fd
    else:
        stderr_fd = os.open(str(stderr_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    saved_stdout = os.dup(1)
    saved_stderr = os.dup(2)
    try:
        os.dup2(stdout_fd, 1)
        os.dup2(stderr_fd, 2)
        yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved_stdout, 1)
        os.dup2(saved_stderr, 2)
        os.close(saved_stdout)
        os.close(saved_stderr)
        if stderr_path is not None:
            os.close(stderr_fd)
        os.close(stdout_fd)


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

    height_px = int(max(f.shape[0], m.shape[0]))
    width_px = int(max(f.shape[1], m.shape[1]))
    fig_w_in = max(15.0, (3.0 * float(width_px)) / float(QC_SAVE_DPI))
    fig_h_in = max(5.0, float(height_px) / float(QC_SAVE_DPI))

    fig, axes = plt.subplots(1, 3, figsize=(fig_w_in, fig_h_in))
    axes[0].imshow(f, cmap="gray", interpolation="nearest", resample=False)
    axes[0].set_title("Fixed (atlas crop)")
    axes[0].axis("off")
    axes[1].imshow(m, cmap="gray", interpolation="nearest", resample=False)
    axes[1].set_title("Warped moving")
    axes[1].axis("off")
    axes[2].imshow(overlay, interpolation="nearest", resample=False)
    axes[2].set_title("Overlay (magenta=fixed, green=warped moving)")
    axes[2].axis("off")
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=int(QC_SAVE_DPI))
    plt.show()


def bbox_indices_from_mask(*, mask_xy: np.ndarray, pad_vox: int = 0) -> tuple[list[int], list[int]] | None:
    mask_xy = np.asarray(mask_xy, dtype=bool)
    if mask_xy.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape={mask_xy.shape}.")

    if not np.any(mask_xy):
        return None

    idx = np.argwhere(mask_xy)
    lo = idx.min(axis=0)
    hi = idx.max(axis=0)

    pad = int(pad_vox)
    lo = np.maximum(lo - pad, 0)
    hi = np.minimum(hi + pad + 1, np.asarray(mask_xy.shape, dtype=int))
    return lo.astype(int).tolist(), hi.astype(int).tolist()


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


def landmark_heatmaps_per_point_mm(
    *,
    domain_img: ants.ANTsImage,
    points_mm: np.ndarray,
    sigma_um: float,
) -> list[ants.ANTsImage]:
    """Create one Gaussian heatmap per landmark on the `domain_img` grid.

    This is more informative than a single combined heatmap because each landmark has a
    dedicated channel, avoiding ambiguous correspondence.
    """

    origin_x_mm, origin_y_mm = domain_img.origin
    spacing_x_mm, spacing_y_mm = domain_img.spacing
    size_x, size_y = domain_img.shape
    sigma_mm = float(sigma_um) * UM_TO_MM

    out: list[ants.ANTsImage] = []
    for x_mm, y_mm in np.asarray(points_mm, dtype=np.float64):
        base_xy = np.zeros((size_x, size_y), dtype=np.float32)
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
        out.append(ants.smooth_image(img, sigma_mm, sigma_in_physical_coordinates=True))
    return out


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


def moving_mask_from_nonzero(img: sitk.Image, threshold: float = 100.0) -> sitk.Image:
    """Create a tissue mask by thresholding intensities + light cleanup.

    Note: in this pipeline the threshold is sourced from `p1_threshold.json` (picked on the
    rotated+cropped moving image) and threaded through to all mask constructions, so it is
    expected to be in the same raw intensity scale (pre-N4).
    """
    mask = sitk.Cast(img > threshold, sitk.sitkUInt8)
    mask = sitk.BinaryMorphologicalClosing(mask, [3, 3])
    mask = sitk.BinaryFillhole(mask)
    return largest_cc(mask)


def _pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size == 0 or b.size == 0:
        return float("nan")
    a0 = a - float(a.mean())
    b0 = b - float(b.mean())
    denom = float(np.sqrt(np.sum(a0 * a0) * np.sum(b0 * b0)))
    if denom == 0.0:
        return float("nan")
    return float(np.sum(a0 * b0) / denom)


def _resample_for_plot(
    img: ants.ANTsImage, *, max_dim: int, max_downsample: float, interp_type: int
) -> ants.ANTsImage:
    if max_dim <= 0:
        raise ValueError(f"max_dim must be > 0, got {max_dim}.")
    if max_downsample < 1:
        raise ValueError(f"max_downsample must be >= 1, got {max_downsample}.")
    if interp_type not in (0, 1):
        raise ValueError(f"interp_type must be 0(linear) or 1(nearest), got {interp_type}.")

    in_max_dim = max(img.shape[0], img.shape[1])
    scale = min(float(max_downsample), max(1.0, float(in_max_dim) / float(max_dim)))
    if scale <= 1.0:
        return img

    out_shape = (
        max(32, int(round(img.shape[0] / scale))),
        max(32, int(round(img.shape[1] / scale))),
    )
    return ants.resample_image(img, out_shape, use_voxels=True, interp_type=interp_type)


def _resample_ants_to_spacing_um(img: ants.ANTsImage, *, spacing_um: float, interp_type: int) -> ants.ANTsImage:
    if spacing_um <= 0:
        raise ValueError(f"spacing_um must be > 0, got {spacing_um}.")
    if interp_type not in (0, 1):
        raise ValueError(f"interp_type must be 0(linear) or 1(nearest), got {interp_type}.")

    spacing_mm = float(spacing_um) * UM_TO_MM
    curr_x_mm, curr_y_mm = img.spacing
    if spacing_mm >= float(curr_x_mm) and spacing_mm >= float(curr_y_mm):
        return img

    return ants.resample_image(img, (spacing_mm, spacing_mm), use_voxels=False, interp_type=interp_type)


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

# Note: run N4 on the raw-ish intensities before robust normalization; normalizing first can
# suppress the low-frequency bias field N4 is trying to estimate.
fixed_sitk = sitk_from_numpy_2d(fixed_np, spacing_um=ATLAS_VOXEL_UM)
moving_sitk = sitk_from_numpy_2d(moving_np, spacing_um=SAMPLE_VOXEL_XY_UM)
fixed_mask_sitk = sitk_from_numpy_2d(fixed_mask_np.astype(np.float32), spacing_um=ATLAS_VOXEL_UM) > 0
fixed_mask_orig_sitk = fixed_mask_sitk
# Moving tissue support mask:
# - threshold in the original moving grid (raw sample intensities)
# - resample the mask to atlas spacing with nearest-neighbor
moving_mask_seed_sitk = (
    moving_mask_from_nonzero(moving_sitk, threshold=P1_THRESHOLD)
    if MOVING_MASK_NONZERO
    else make_moving_mask_sitk(moving_sitk)
)
moving_reg_sitk = resample_sitk_to_spacing(moving_sitk, target_spacing_um=ATLAS_VOXEL_UM, interp=sitk.sitkLinear)
moving_mask_reg_seed_sitk = resample_sitk_to_spacing(
    sitk.Cast(moving_mask_seed_sitk, sitk.sitkUInt8),
    target_spacing_um=ATLAS_VOXEL_UM,
    interp=sitk.sitkNearestNeighbor,
)
moving_mask_reg_seed_sitk = sitk.Cast(moving_mask_reg_seed_sitk > 0, sitk.sitkUInt8)
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
    # Do not re-threshold after N4: the moving mask is defined solely by the user-picked
    # Phase-1 threshold (`p1_threshold.json`) on the raw sample intensities.
    moving_mask_reg_sitk = moving_mask_reg_seed_sitk

if MOVING_PRESMOOTH_SIGMA_UM > 0:
    moving_reg_intensity_sitk = sitk.SmoothingRecursiveGaussian(
        sitk.Cast(moving_reg_intensity_sitk, sitk.sitkFloat32),
        sigma=float(MOVING_PRESMOOTH_SIGMA_UM) * UM_TO_MM,
    )
    # Likewise, do not re-threshold after pre-smoothing.
    moving_mask_reg_sitk = moving_mask_reg_seed_sitk

fixed_reg_sitk = normalize_sitk_intensity(fixed_reg_sitk)
moving_reg_intensity_sitk = normalize_sitk_intensity(moving_reg_intensity_sitk)

def save_debug_thumbnail(
    arr: np.ndarray,
    path: Path,
    title: str,
    *,
    cmap: str = "gray",
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=int(QC_SAVE_DPI))
    plt.close(fig)


def save_debug_pair(
    *,
    fixed_arr: np.ndarray,
    moving_arr: np.ndarray,
    path: Path,
    suptitle: str,
    fixed_title: str = "Fixed",
    moving_title: str = "Moving",
    cmap: str = "gray",
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(fixed_arr, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    axes[0].set_title(fixed_title)
    axes[0].axis("off")
    axes[1].imshow(moving_arr, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    axes[1].set_title(moving_title)
    axes[1].axis("off")
    fig.suptitle(suptitle)
    plt.tight_layout()
    plt.savefig(path, dpi=int(QC_SAVE_DPI))
    plt.close(fig)


def _points_mm_to_vox_xy(points_mm: np.ndarray, domain_img: ants.ANTsImage) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pts = np.asarray(points_mm, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"Expected (N,2) points_mm, got shape={pts.shape}.")

    origin_x_mm, origin_y_mm = domain_img.origin
    spacing_x_mm, spacing_y_mm = domain_img.spacing
    xs = np.round((pts[:, 0] - float(origin_x_mm)) / float(spacing_x_mm)).astype(np.int32, copy=False)
    ys = np.round((pts[:, 1] - float(origin_y_mm)) / float(spacing_y_mm)).astype(np.int32, copy=False)
    valid = (xs >= 0) & (xs < int(domain_img.shape[0])) & (ys >= 0) & (ys < int(domain_img.shape[1]))
    return xs, ys, valid


def save_debug_moving_metric_overlays(
    *,
    moving_base_yx: np.ndarray,
    moving_mask_yx: np.ndarray,
    moving_domain: ants.ANTsImage,
    moving_pts_mm: np.ndarray,
    fixed_pts_mm: np.ndarray,
    fixed_to_moving_tx: ants.ANTsTransform,
    feature_yx: np.ndarray | None,
    signed_distance_yx: np.ndarray | None,
    out_png: Path,
) -> None:
    base = np.asarray(moving_base_yx, dtype=np.float32)
    mask = np.asarray(moving_mask_yx, dtype=bool)
    if base.shape != mask.shape:
        raise ValueError(f"Base/mask shape mismatch: base={base.shape}, mask={mask.shape}.")
    expected_shape = (int(moving_domain.shape[1]), int(moving_domain.shape[0]))
    if base.shape != expected_shape:
        raise ValueError(
            "Expected moving_base_yx to match the moving-domain grid "
            f"(base_yx={base.shape} vs moving_domain.shape={moving_domain.shape})."
        )

    base_vis = base.copy()
    base_vis[~mask] *= 0.3

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: landmarks (moving points vs fixed points mapped into moving).
    ax = axes[0]
    ax.imshow(base_vis, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
    pred_moving_mm = np.vstack([np.asarray(fixed_to_moving_tx.apply_to_point(tuple(p))) for p in fixed_pts_mm])

    mx, my, m_ok = _points_mm_to_vox_xy(moving_pts_mm, moving_domain)
    px, py, p_ok = _points_mm_to_vox_xy(pred_moving_mm, moving_domain)
    ok = m_ok & p_ok

    ax.scatter(mx[m_ok], my[m_ok], s=36, facecolors="none", edgecolors="lime", linewidths=1.2, label="moving")
    ax.scatter(px[p_ok], py[p_ok], s=34, c="magenta", marker="x", linewidths=1.2, label="fixed→moving")
    for x0, y0, x1, y1 in zip(px[ok], py[ok], mx[ok], my[ok], strict=False):
        ax.plot([x0, x1], [y0, y1], color="yellow", lw=0.9, alpha=0.85)
    ax.set_title("Landmarks (on moving)")
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    ax.axis("off")

    # Panel 2: feature image (gradmag).
    ax = axes[1]
    ax.imshow(base_vis, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
    if feature_yx is not None:
        feat = np.asarray(feature_yx, dtype=np.float32)
        if feat.shape != base.shape:
            raise ValueError(f"Feature/base shape mismatch: feature={feat.shape}, base={base.shape}.")
        ax.imshow(feat, cmap="magma", vmin=0.0, vmax=1.0, alpha=0.45, interpolation="nearest")
    ax.set_title("Feature (gradmag)")
    ax.axis("off")

    # Panel 3: signed distance map (mask).
    ax = axes[2]
    ax.imshow(base_vis, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
    if signed_distance_yx is not None:
        dt = np.asarray(signed_distance_yx, dtype=np.float32)
        if dt.shape != base.shape:
            raise ValueError(f"Signed-distance/base shape mismatch: dt={dt.shape}, base={base.shape}.")
        ax.imshow(dt, cmap="bwr", vmin=-1.0, vmax=1.0, alpha=0.35, interpolation="nearest")
        ax.contour(dt, levels=[0.0], colors="cyan", linewidths=1.0)
    ax.set_title("Signed distance (mask)")
    ax.axis("off")

    fig.suptitle("Unwarped moving: registration metric overlays")
    plt.tight_layout()
    plt.savefig(out_png, dpi=int(QC_SAVE_DPI))
    plt.close(fig)


def save_debug_landmark_heatmap_overlay(
    *,
    moving_base_yx: np.ndarray,
    moving_mask_yx: np.ndarray,
    landmark_heatmap_yx: np.ndarray,
    out_png: Path,
    title: str,
) -> None:
    base = np.asarray(moving_base_yx, dtype=np.float32)
    mask = np.asarray(moving_mask_yx, dtype=bool)
    hm = np.asarray(landmark_heatmap_yx, dtype=np.float32)
    if base.shape != mask.shape:
        raise ValueError(f"Base/mask shape mismatch: base={base.shape}, mask={mask.shape}.")
    if hm.shape != base.shape:
        raise ValueError(f"Heatmap/base shape mismatch: heatmap={hm.shape}, base={base.shape}.")

    base_vis = base.copy()
    base_vis[~mask] *= 0.3

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(base_vis, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
    ax.imshow(hm, cmap="viridis", vmin=0.0, vmax=1.0, alpha=0.55, interpolation="nearest")
    ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=int(QC_SAVE_DPI))
    plt.close(fig)

moving_arr_normalized = sitk.GetArrayFromImage(moving_reg_intensity_sitk)
mask_arr = sitk.GetArrayFromImage(moving_mask_reg_sitk).astype(bool)
save_debug_thumbnail(moving_arr_normalized, OUTDIR / "debug_moving_normalized.png", "Moving after normalize_robust")
save_debug_thumbnail(mask_arr.astype(float), OUTDIR / "debug_moving_mask_pre_erode.png", "Moving mask (pre-erosion)")
# Overlay: show mask boundary on normalized image
overlay = moving_arr_normalized.copy()
overlay[~mask_arr] *= 0.3  # dim outside mask
save_debug_thumbnail(overlay, OUTDIR / "debug_moving_mask_overlay.png", "Mask overlay on normalized")

fixed_mask_sitk = erode_by_um(fixed_mask_sitk, FIXED_EDGE_GUARD_UM)
moving_mask_reg_full_sitk = moving_mask_reg_sitk
moving_mask_reg_sitk = erode_by_um(moving_mask_reg_full_sitk, MOVING_EDGE_GUARD_UM)
moving_metric_mask_sitk = (
    dilate_by_um(moving_mask_reg_full_sitk, MOVING_METRIC_MASK_DILATE_UM)
    if MOVING_METRIC_MASK_DILATE_UM > 0
    else moving_mask_reg_full_sitk
)

if USE_FEATURE_IMAGES:
    fixed_feat_sitk = gradmag_feature(fixed_reg_sitk, FEATURE_SIGMA_UM)
    moving_feat_sitk = gradmag_feature(moving_reg_intensity_sitk, FEATURE_SIGMA_UM)
    fixed_gradmag = normalize_robust(sitk.GetArrayFromImage(fixed_feat_sitk)).astype(np.float32)
    moving_gradmag = normalize_robust(sitk.GetArrayFromImage(moving_feat_sitk)).astype(np.float32)
    moving_feature_arr = moving_gradmag
    save_debug_pair(
        fixed_arr=fixed_gradmag,
        moving_arr=moving_gradmag,
        path=OUTDIR / "debug_gradmag_pair.png",
        suptitle="Gradient magnitude feature (normalized)",
        fixed_title="Fixed gradmag",
        moving_title="Moving gradmag",
        cmap="gray",
        vmin=0.0,
        vmax=1.0,
    )
else:
    fixed_feat_sitk = None
    moving_feat_sitk = None
    moving_feature_arr = None

sitk.WriteImage(fixed_reg_sitk, str(FIXED_REG_NIFTI))
sitk.WriteImage(moving_reg_intensity_sitk, str(MOVING_REG_NIFTI))

sitk.WriteImage(fixed_sitk, str(FIXED_NIFTI))
sitk.WriteImage(moving_sitk, str(MOVING_NIFTI))
sitk.WriteImage(sitk.Cast(fixed_mask_orig_sitk, sitk.sitkUInt8), str(FIXED_MASK_ORIG_NIFTI))
sitk.WriteImage(sitk.Cast(fixed_mask_sitk, sitk.sitkUInt8), str(FIXED_MASK_NIFTI))
fixed_metric_mask_sitk = (
    dilate_by_um(fixed_mask_orig_sitk, FIXED_METRIC_MASK_DILATE_UM)
    if FIXED_METRIC_MASK_DILATE_UM > 0
    else fixed_mask_orig_sitk
)
sitk.WriteImage(sitk.Cast(fixed_metric_mask_sitk, sitk.sitkUInt8), str(FIXED_METRIC_MASK_NIFTI))
save_debug_thumbnail(
    sitk.GetArrayFromImage(fixed_metric_mask_sitk).astype(np.float32),
    OUTDIR / "debug_fixed_metric_mask.png",
    f"Fixed metric mask (dilate_um={FIXED_METRIC_MASK_DILATE_UM:g})",
    vmin=0.0,
    vmax=1.0,
)
sitk.WriteImage(sitk.Cast(moving_mask_reg_full_sitk, sitk.sitkUInt8), str(MOVING_MASK_REG_FULL_NIFTI))
sitk.WriteImage(sitk.Cast(moving_metric_mask_sitk, sitk.sitkUInt8), str(MOVING_METRIC_MASK_NIFTI))
save_debug_thumbnail(
    sitk.GetArrayFromImage(moving_metric_mask_sitk).astype(np.float32),
    OUTDIR / "debug_moving_metric_mask.png",
    f"Moving metric mask (dilate_um={MOVING_METRIC_MASK_DILATE_UM:g})",
    vmin=0.0,
    vmax=1.0,
)
sitk.WriteImage(sitk.Cast(moving_mask_reg_sitk, sitk.sitkUInt8), str(MOVING_MASK_REG_NIFTI))

if USE_FEATURE_IMAGES:
    sitk.WriteImage(fixed_feat_sitk, str(FIXED_FEAT_NIFTI))
    sitk.WriteImage(moving_feat_sitk, str(MOVING_FEAT_NIFTI))

fixed_ants = ants.image_read(str(FIXED_NIFTI))
moving_ants = ants.image_read(str(MOVING_NIFTI))
fixed_mask_ants = ants.image_read(str(FIXED_MASK_NIFTI))
fixed_mask_orig_ants = ants.image_read(str(FIXED_MASK_ORIG_NIFTI))
fixed_metric_mask_ants = ants.image_read(str(FIXED_METRIC_MASK_NIFTI))
moving_mask_reg_ants = ants.image_read(str(MOVING_MASK_REG_NIFTI))
moving_mask_reg_full_ants = ants.image_read(str(MOVING_MASK_REG_FULL_NIFTI))
moving_metric_mask_ants = ants.image_read(str(MOVING_METRIC_MASK_NIFTI))
fixed_reg_ants = ants.image_read(str(FIXED_REG_NIFTI))
moving_reg_ants = ants.image_read(str(MOVING_REG_NIFTI))
fixed_feat_ants = ants.image_read(str(FIXED_FEAT_NIFTI)) if USE_FEATURE_IMAGES else None
moving_feat_ants = ants.image_read(str(MOVING_FEAT_NIFTI)) if USE_FEATURE_IMAGES else None

initial_transform_for_syn: list[str] = [str(LINEAR_INIT_MAT_PATH)]

# Overlap mask in FIXED space (for QC + optional cropping).
# NOTE: do NOT use a *static* overlap mask as the registration mask: it can prevent the optimizer from
# moving tissue outside the initial overlap region. Instead, provide `mask` (fixed) + `moving_mask`
# and let ANTs compute the overlap dynamically as the transform updates.
moving_mask_in_fixed = ants.apply_transforms(
    fixed=fixed_mask_orig_ants,
    moving=moving_metric_mask_ants if MOVING_METRIC_MASK_DILATE_UM > 0 else moving_mask_reg_full_ants,
    transformlist=initial_transform_for_syn,
    interpolator="nearestNeighbor",
)
overlap_mask_ants = fixed_mask_orig_ants * moving_mask_in_fixed
ants.image_write(overlap_mask_ants, str(OVERLAP_MASK_NIFTI))
print(f"Wrote: {OVERLAP_MASK_NIFTI}")
print("Overlap frac (fixed grid):", float((overlap_mask_ants.numpy() > 0).mean()))

tx_affine_refine: dict[str, object] | None = None
affine_refine_applied = False
if USE_AFFINE_REFINE:
    initial_before_affine = list(initial_transform_for_syn)
    overlap_frac_before_affine = float((overlap_mask_ants.numpy() > 0).mean())
    AFFINE_REG_STDOUT = OUTDIR / "affine_refine_antsRegistration_stdout.txt"
    # For partial tissue, restricting the fixed mask to the current overlap avoids Mattes MI
    # failures where most sampled points map outside the moving buffer.
    affine_fixed_mask = overlap_mask_ants
    with _redirect_fds(stdout_path=AFFINE_REG_STDOUT):
        tx_affine_refine = ants.registration(
            fixed=fixed_reg_ants,
            moving=moving_reg_ants,
            type_of_transform=AFFINE_REFINE_TYPE_OF_TRANSFORM,
            initial_transform=initial_before_affine,
            grad_step=AFFINE_REFINE_GRAD_STEP,
            aff_metric=AFFINE_REFINE_METRIC,
            aff_sampling=AFFINE_REFINE_MI_BINS,
            aff_random_sampling_rate=AFFINE_REFINE_RANDOM_SAMPLING_RATE,
            aff_iterations=AFFINE_REFINE_ITERATIONS,
            aff_shrink_factors=AFFINE_REFINE_SHRINK_FACTORS,
            aff_smoothing_sigmas=AFFINE_REFINE_SMOOTHING_SIGMAS,
            mask=affine_fixed_mask,
            moving_mask=moving_mask_reg_full_ants,
            mask_all_stages=False,
            random_seed=0,
            write_composite_transform=True,
            verbose=True,
            outprefix=str(AFFINE_PREFIX),
        )
    print(f"Wrote: {AFFINE_REG_STDOUT}")
    fwd = tx_affine_refine.get("fwdtransforms")
    if isinstance(fwd, str):
        initial_transform_for_syn = [fwd]
    elif isinstance(fwd, (list, tuple)) and all(isinstance(p, str) for p in fwd):
        initial_transform_for_syn = list(fwd)
    else:
        raise TypeError(f"Unexpected affine refine fwdtransforms type: {type(fwd)!r}")

    # Refresh overlap mask using the refined affine.
    moving_mask_in_fixed = ants.apply_transforms(
        fixed=fixed_mask_orig_ants,
        moving=moving_mask_reg_full_ants,
        transformlist=initial_transform_for_syn,
        interpolator="nearestNeighbor",
    )
    overlap_mask_after_affine = fixed_mask_orig_ants * moving_mask_in_fixed
    overlap_frac_after_affine = float((overlap_mask_after_affine.numpy() > 0).mean())

    if overlap_frac_after_affine < 0.5 * overlap_frac_before_affine:
        print(
            "Affine refine rejected: "
            f"overlap frac {overlap_frac_after_affine:.4f} < {0.5 * overlap_frac_before_affine:.4f}"
        )
        initial_transform_for_syn = initial_before_affine
        tx_affine_refine = None
        overlap_mask_ants = overlap_mask_ants  # keep original
    else:
        overlap_mask_ants = overlap_mask_after_affine
        affine_refine_applied = True

    ants.image_write(overlap_mask_ants, str(OVERLAP_MASK_NIFTI))
    print(f"Wrote: {OVERLAP_MASK_NIFTI}")
    print("Overlap frac after affine refine (fixed grid):", float((overlap_mask_ants.numpy() > 0).mean()))

fixed_crop_lower = None
fixed_crop_upper = None
if CROP_FIXED_TO_OVERLAP:
    overlap_arr = overlap_mask_ants.numpy() > 0
    if np.any(overlap_arr):
        idx = np.argwhere(overlap_arr)
        lo_x, lo_y = idx.min(axis=0).tolist()
        hi_x, hi_y = idx.max(axis=0).tolist()
        # Ensure the crop includes all fixed landmarks (we still want the landmark metric to contribute).
        origin_x_mm, origin_y_mm = fixed_reg_ants.origin
        spacing_x_mm, spacing_y_mm = fixed_reg_ants.spacing
        fixed_pts_vox_x = np.round((fixed_pts_mm[:, 0] - float(origin_x_mm)) / float(spacing_x_mm)).astype(np.int32)
        fixed_pts_vox_y = np.round((fixed_pts_mm[:, 1] - float(origin_y_mm)) / float(spacing_y_mm)).astype(np.int32)
        lo_x = int(min(lo_x, int(fixed_pts_vox_x.min(initial=lo_x))))
        lo_y = int(min(lo_y, int(fixed_pts_vox_y.min(initial=lo_y))))
        hi_x = int(max(hi_x, int(fixed_pts_vox_x.max(initial=hi_x))))
        hi_y = int(max(hi_y, int(fixed_pts_vox_y.max(initial=hi_y))))
        pad = int(CROP_PAD_VOX)
        lo = [max(0, int(lo_x - pad)), max(0, int(lo_y - pad))]
        hi = [min(fixed_reg_ants.shape[0] - 1, int(hi_x + pad)), min(fixed_reg_ants.shape[1] - 1, int(hi_y + pad))]
        fixed_crop_lower = lo
        fixed_crop_upper = [hi[0] + 1, hi[1] + 1]

# Registration can optionally run in a cropped fixed domain (speed / focus on overlap).
# NOTE: this changes the warp field support (the displacement is only defined on the cropped grid).
fixed_reg_syn = fixed_reg_ants
fixed_mask_syn = fixed_metric_mask_ants
fixed_feat_syn = fixed_feat_ants
crop_syn = bool(
    CROP_FIXED_DOMAIN_FOR_SYN and fixed_crop_lower is not None and fixed_crop_upper is not None
)
if crop_syn:
    fixed_reg_syn = ants.crop_indices(fixed_reg_ants, fixed_crop_lower, fixed_crop_upper)
    fixed_mask_syn = ants.crop_indices(fixed_metric_mask_ants, fixed_crop_lower, fixed_crop_upper)
    if USE_FEATURE_IMAGES and fixed_feat_syn is not None:
        fixed_feat_syn = ants.crop_indices(fixed_feat_syn, fixed_crop_lower, fixed_crop_upper)

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

mi_before_raw = float(ants.image_mutual_information(fixed_ants, before_warped))
print(f"MI before (raw images, unmasked): {mi_before_raw:.6f}")

before_warped_reg = ants.apply_transforms(
    fixed=fixed_reg_syn,
    moving=moving_reg_ants,
    transformlist=initial_transform_for_syn,
    interpolator="linear",
)
moving_metric_mask_init_in_fixed_syn = ants.apply_transforms(
    fixed=fixed_mask_syn,
    moving=moving_metric_mask_ants,
    transformlist=initial_transform_for_syn,
    interpolator="nearestNeighbor",
)
overlap_init_syn = (np.asarray(fixed_mask_syn.numpy()) > 0) & (
    np.asarray(moving_metric_mask_init_in_fixed_syn.numpy()) > 0
)
mi_before = _masked_mutual_information_hist2d(
    fixed=np.asarray(fixed_reg_syn.numpy()),
    moving=np.asarray(before_warped_reg.numpy()),
    mask=overlap_init_syn,
)
print(f"MI before (preprocessed images, overlap-masked): {mi_before:.6f}")

kwargs: dict[str, object] = dict(
    fixed=fixed_reg_syn,
    moving=moving_reg_ants,
    type_of_transform=SYN_TYPE_OF_TRANSFORM,
    initial_transform=initial_transform_for_syn,
    grad_step=SYN_GRAD_STEP,
    flow_sigma=SYN_FLOW_SIGMA,
    total_sigma=SYN_TOTAL_SIGMA,
    syn_metric=SYN_METRIC,
    syn_sampling=SYN_METRIC_PARAM,
    reg_iterations=SYN_REG_ITERATIONS,
    mask=fixed_mask_syn,
    moving_mask=moving_metric_mask_ants,
    mask_all_stages=True,
    random_seed=0,
    write_composite_transform=True,
    verbose=True,
    outprefix=str(SYN_PREFIX),
)

extras: list[tuple[str, ants.ANTsImage, ants.ANTsImage, float, int]] = []

if USE_FEATURE_IMAGES:
    extras.append(("CC", fixed_feat_syn, moving_feat_ants, float(FEATURE_WEIGHT), int(FEATURE_CC_RADIUS)))

if USE_MASK_DISTANCE_METRIC:
    if MASK_DISTANCE_USE_OVERLAP:
        # Compute signed distance on the overlap mask (fixed brain ∩ moving wedge after linear init)
        # to avoid gradients dominated by non-overlapping fixed tissue.
        overlap_mask_sitk = sitk.ReadImage(str(OVERLAP_MASK_NIFTI))
        overlap_mask_sitk = sitk.Cast(overlap_mask_sitk > 0, sitk.sitkUInt8)
        fixed_dt_mask_sitk = overlap_mask_sitk
    else:
        # Signed distance boundaries should reflect the original tissue/brain boundary, not the
        # dilated registration metric masks (which intentionally include background margin).
        fixed_dt_mask_sitk = fixed_mask_orig_sitk

    moving_dt_mask_sitk = moving_mask_reg_full_sitk
    fixed_dt_mask_sitk = _smooth_mask_for_signed_distance(
        mask=fixed_dt_mask_sitk, sigma_um=MASK_DISTANCE_SMOOTH_SIGMA_UM
    )
    moving_dt_mask_sitk = _smooth_mask_for_signed_distance(
        mask=moving_dt_mask_sitk, sigma_um=MASK_DISTANCE_SMOOTH_SIGMA_UM
    )

    fixed_dt_sitk = signed_distance(fixed_dt_mask_sitk)
    moving_dt_sitk = signed_distance(moving_dt_mask_sitk)
    sitk.WriteImage(fixed_dt_sitk, str(FIXED_DT_NIFTI))
    sitk.WriteImage(moving_dt_sitk, str(MOVING_DT_NIFTI))
    fixed_dt_arr = sitk.GetArrayFromImage(fixed_dt_sitk).astype(np.float32)
    moving_dt_arr = sitk.GetArrayFromImage(moving_dt_sitk).astype(np.float32)

    save_debug_moving_metric_overlays(
        moving_base_yx=moving_arr_normalized,
        moving_mask_yx=sitk.GetArrayFromImage(moving_mask_reg_full_sitk).astype(bool),
        moving_domain=moving_reg_ants,
        moving_pts_mm=moving_pts_mm,
        fixed_pts_mm=fixed_pts_mm,
        fixed_to_moving_tx=linear_init_tx,
        feature_yx=moving_feature_arr,
        signed_distance_yx=moving_dt_arr,
        out_png=DEBUG_MOVING_METRICS_OVERLAY_PNG,
    )
    print(f"Wrote: {DEBUG_MOVING_METRICS_OVERLAY_PNG}")

    fixed_dt_max = float(np.quantile(np.abs(fixed_dt_arr), 0.99)) if fixed_dt_arr.size else 1.0
    moving_dt_max = float(np.quantile(np.abs(moving_dt_arr), 0.99)) if moving_dt_arr.size else 1.0
    dt_max = max(fixed_dt_max, moving_dt_max, 1.0)
    save_debug_pair(
        fixed_arr=fixed_dt_arr,
        moving_arr=moving_dt_arr,
        path=OUTDIR / "debug_signed_distance_pair.png",
        suptitle="Mask signed distance (normalized, clipped)",
        fixed_title="Fixed signed distance",
        moving_title="Moving signed distance",
        cmap="bwr",
        vmin=-dt_max,
        vmax=dt_max,
    )
    fixed_dt_ants = ants.image_read(str(FIXED_DT_NIFTI))
    moving_dt_ants = ants.image_read(str(MOVING_DT_NIFTI))
    if crop_syn:
        fixed_dt_ants = ants.crop_indices(fixed_dt_ants, fixed_crop_lower, fixed_crop_upper)
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
    fixed_lm_norm = normalize_robust(ants_numpy_yx(fixed_lm).astype(np.float32))
    moving_lm_norm = normalize_robust(ants_numpy_yx(moving_lm).astype(np.float32))
    save_debug_pair(
        fixed_arr=fixed_lm_norm,
        moving_arr=moving_lm_norm,
        path=OUTDIR / "debug_landmark_heatmap_pair.png",
        suptitle=f"Landmark heatmap (sigma={LANDMARK_HEATMAP_SIGMA_UM:.0f} µm, normalized)",
        fixed_title="Fixed landmark heatmap",
        moving_title="Moving landmark heatmap",
        cmap="gray",
        vmin=0.0,
        vmax=1.0,
    )
    save_debug_landmark_heatmap_overlay(
        moving_base_yx=moving_arr_normalized,
        moving_mask_yx=sitk.GetArrayFromImage(moving_mask_reg_full_sitk).astype(bool),
        landmark_heatmap_yx=moving_lm_norm,
        out_png=DEBUG_MOVING_LANDMARK_HEATMAP_OVERLAY_PNG,
        title=f"Landmark heatmap overlay (sigma={LANDMARK_HEATMAP_SIGMA_UM:.0f} µm)",
    )
    print(f"Wrote: {DEBUG_MOVING_LANDMARK_HEATMAP_OVERLAY_PNG}")
    fixed_lm_per = landmark_heatmaps_per_point_mm(
        domain_img=fixed_reg_ants,
        points_mm=fixed_pts_mm,
        sigma_um=LANDMARK_HEATMAP_SIGMA_UM,
    )
    moving_lm_per = landmark_heatmaps_per_point_mm(
        domain_img=moving_reg_ants,
        points_mm=moving_pts_mm,
        sigma_um=LANDMARK_HEATMAP_SIGMA_UM,
    )
    if len(fixed_lm_per) != len(moving_lm_per):
        raise ValueError(
            f"Landmark heatmap channel count mismatch: fixed={len(fixed_lm_per)} vs moving={len(moving_lm_per)}."
        )

    n_lm = len(fixed_lm_per)
    if n_lm > 0:
        w = float(LANDMARK_HEATMAP_WEIGHT) / float(n_lm)
        for f_lm_i, m_lm_i in zip(fixed_lm_per, moving_lm_per, strict=True):
            if crop_syn:
                f_lm_i = ants.crop_indices(f_lm_i, fixed_crop_lower, fixed_crop_upper)
            extras.append(("MeanSquares", f_lm_i, m_lm_i, w, 0))

if extras:
    kwargs["multivariate_extras"] = extras

ANTS_REG_STDOUT = OUTDIR / "final_syn_mi_antsRegistration_stdout.txt"
LOSS_PNG = OUTDIR / "final_syn_mi_loss.png"
with _redirect_fds(stdout_path=ANTS_REG_STDOUT):
    tx = ants.registration(**kwargs)
print(f"Wrote: {ANTS_REG_STDOUT}")

_, metric_values = _parse_ants_diagnostic_metric_values(ANTS_REG_STDOUT)
if metric_values:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(np.arange(len(metric_values), dtype=np.int32), metric_values, lw=1)
    ax.set_xlabel("Iteration (all levels, sequential)")
    ax.set_ylabel("Metric value")
    ax.set_title(f"antsRegistration diagnostics ({SYN_TYPE_OF_TRANSFORM}, syn_metric={SYN_METRIC})")
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(LOSS_PNG, dpi=160)
    plt.close(fig)
    print(f"Wrote: {LOSS_PNG}")
else:
    print(f"No DIAGNOSTIC metric values found in {ANTS_REG_STDOUT}; skipping loss plot.")

# --- Deformation field plot (SyN residual vs init), shown in MOVING domain ---
# User request: use the moving image as background and plot the inverse-direction field.
# For visualization, use the native-res moving crop (no more than 16x downsample vs fused.zarr).
moving_plot = moving_ants
moving_bg_ds = _resample_for_plot(
    moving_plot,
    max_dim=DEFORMATION_BG_MAX_DIM,
    max_downsample=DEFORMATION_MAX_DOWNSAMPLE,
    interp_type=0,
)
moving_field_ds = _resample_for_plot(
    moving_plot,
    max_dim=DEFORMATION_FIELD_MAX_DIM,
    max_downsample=DEFORMATION_MAX_DOWNSAMPLE,
    interp_type=0,
)

size_x, size_y = moving_field_ds.shape
origin_x_mm, origin_y_mm = moving_field_ds.origin
spacing_x_mm, spacing_y_mm = moving_field_ds.spacing
xs_mm = origin_x_mm + np.arange(size_x, dtype=np.float64) * float(spacing_x_mm)
ys_mm = origin_y_mm + np.arange(size_y, dtype=np.float64) * float(spacing_y_mm)
grid_x_mm, grid_y_mm = np.meshgrid(xs_mm, ys_mm, indexing="xy")
df_grid = pd.DataFrame({"x": grid_x_mm.ravel(), "y": grid_y_mm.ravel(), "z": 0.0, "t": 0.0})

# Inverse mapping: moving → fixed.
inv = tx["invtransforms"]
inv_list = [inv] if isinstance(inv, str) else list(inv)
df_final_inv = ants.apply_transforms_to_points(dim=2, points=df_grid, transformlist=inv_list)

# Baseline inverse mapping corresponding to the init (linear, or optional affine-refine if enabled).
if USE_AFFINE_REFINE and tx_affine_refine is not None and affine_refine_applied:
    init_inv = tx_affine_refine["invtransforms"]
    init_inv_list = [init_inv] if isinstance(init_inv, str) else list(init_inv)
    df_init_inv = ants.apply_transforms_to_points(dim=2, points=df_grid, transformlist=init_inv_list)
else:
    init_list = list(initial_transform_for_syn)
    whichtoinvert: list[bool] = []
    for path in init_list:
        suffix = Path(path).suffix.lower()
        if suffix == ".mat":
            whichtoinvert.append(True)
        else:
            raise ValueError(
                f"Cannot invert init transform {path!r} (suffix {suffix!r}). "
                "Expected a matrix transform ('.mat') or enable affine refine."
            )
    df_init_inv = ants.apply_transforms_to_points(
        dim=2,
        points=df_grid,
        transformlist=init_list,
        whichtoinvert=whichtoinvert,
    )

dx_mm = (df_final_inv["x"].to_numpy(dtype=np.float64) - df_init_inv["x"].to_numpy(dtype=np.float64)).reshape(
    (size_y, size_x)
)
dy_mm = (df_final_inv["y"].to_numpy(dtype=np.float64) - df_init_inv["y"].to_numpy(dtype=np.float64)).reshape(
    (size_y, size_x)
)
mag_um = np.sqrt(dx_mm**2 + dy_mm**2) / UM_TO_MM

# Resample displacement components + magnitude to the background grid for display.
dx_img = ants.from_numpy(
    dx_mm.T.astype(np.float32, copy=False),
    origin=moving_field_ds.origin,
    spacing=moving_field_ds.spacing,
    direction=moving_field_ds.direction,
)
dy_img = ants.from_numpy(
    dy_mm.T.astype(np.float32, copy=False),
    origin=moving_field_ds.origin,
    spacing=moving_field_ds.spacing,
    direction=moving_field_ds.direction,
)
mag_img = ants.from_numpy(
    mag_um.T.astype(np.float32, copy=False),
    origin=moving_field_ds.origin,
    spacing=moving_field_ds.spacing,
    direction=moving_field_ds.direction,
)
dx_bg = ants.resample_image_to_target(dx_img, moving_bg_ds, interp_type=0)
dy_bg = ants.resample_image_to_target(dy_img, moving_bg_ds, interp_type=0)
mag_bg = ants.resample_image_to_target(mag_img, moving_bg_ds, interp_type=0)

dx_bg_mm = np.asarray(dx_bg.numpy(), dtype=np.float32).T
dy_bg_mm = np.asarray(dy_bg.numpy(), dtype=np.float32).T
mag_bg_um = np.asarray(mag_bg.numpy(), dtype=np.float32).T

bg_spacing_x_mm, bg_spacing_y_mm = moving_bg_ds.spacing
dx_bg_px = dx_bg_mm / float(bg_spacing_x_mm)
dy_bg_px = dy_bg_mm / float(bg_spacing_y_mm)

scale_ratio = float(moving_bg_ds.shape[0]) / float(moving_field_ds.shape[0])
step_bg = max(1, int(round(float(DEFORMATION_QUIVER_STEP) * scale_ratio)))
yy, xx = np.mgrid[0 : mag_bg_um.shape[0] : step_bg, 0 : mag_bg_um.shape[1] : step_bg]
u = dx_bg_px[::step_bg, ::step_bg]
v = dy_bg_px[::step_bg, ::step_bg]
max_disp_px = float(np.sqrt(u**2 + v**2).max())
desired_max_len = step_bg * 0.8
quiver_scale = max_disp_px / desired_max_len if max_disp_px > 0 else 1.0

fig_w_in = min(float(DEFORMATION_MAX_FIG_IN), max(10.0, float(moving_bg_ds.shape[0]) / float(DEFORMATION_SAVE_DPI)))
fig_h_in = fig_w_in * (float(moving_bg_ds.shape[1]) / float(moving_bg_ds.shape[0]))
fig, ax = plt.subplots(figsize=(fig_w_in, fig_h_in))
bg = normalize_robust(ants_numpy_yx(moving_bg_ds).astype(np.float32))
if bg.shape != mag_bg_um.shape:
    raise ValueError(f"Unexpected bg shape {bg.shape} vs mag {mag_bg_um.shape}.")
ax.imshow(bg, cmap="gray", interpolation="bilinear")
im = ax.imshow(mag_bg_um, cmap="magma", alpha=0.35, interpolation="bilinear")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="|displacement| (µm)")
mask_u = bg[::step_bg, ::step_bg] > 0.03
u_plot = np.ma.masked_where(~mask_u, u)
v_plot = np.ma.masked_where(~mask_u, v)
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
    alpha=0.75,
)
ax.set_title("SyN inverse displacement field (residual vs init, moving domain)")
ax.axis("off")
plt.tight_layout()
plt.savefig(DEFORMATION_FIELD_PNG, dpi=int(DEFORMATION_SAVE_DPI))
plt.close(fig)
print(f"Wrote: {DEFORMATION_FIELD_PNG}")

warped_after = ants.apply_transforms(
    fixed=fixed_ants,
    moving=moving_ants,
    transformlist=tx["fwdtransforms"],
    interpolator="linear",
)
ants.image_write(warped_after, str(WARPED_AFTER_NIFTI))

# Warp both the full moving tissue mask (for QC/rims) and the metric mask (for debugging).
moving_mask_metric_final_in_fixed = ants.apply_transforms(
    fixed=fixed_mask_ants,
    moving=moving_metric_mask_ants,
    transformlist=tx["fwdtransforms"],
    interpolator="nearestNeighbor",
)
ants.image_write(moving_mask_metric_final_in_fixed, str(MOVING_MASK_METRIC_WARPED_FINAL_NIFTI))

moving_mask_final_in_fixed = ants.apply_transforms(
    fixed=fixed_mask_ants,
    moving=moving_mask_reg_full_ants,
    transformlist=tx["fwdtransforms"],
    interpolator="nearestNeighbor",
)
ants.image_write(moving_mask_final_in_fixed, str(MOVING_MASK_WARPED_FINAL_NIFTI))
overlap_mask_final = fixed_mask_ants * moving_mask_final_in_fixed
ants.image_write(overlap_mask_final, str(OVERLAP_MASK_FINAL_NIFTI))
print(f"Wrote: {MOVING_MASK_WARPED_FINAL_NIFTI}")
print(f"Wrote: {MOVING_MASK_METRIC_WARPED_FINAL_NIFTI}")
print(f"Wrote: {OVERLAP_MASK_FINAL_NIFTI}")
print("Overlap frac after (fixed grid):", float((overlap_mask_final.numpy() > 0).mean()))

# Moving-only QC: before/after crops around moving tissue bounding box (in fixed grid).
moving_mask_init_in_fixed = ants.apply_transforms(
    fixed=fixed_mask_ants,
    moving=moving_mask_reg_full_ants,
    transformlist=initial_transform_for_syn,
    interpolator="nearestNeighbor",
)

mi_after_raw = float(ants.image_mutual_information(fixed_ants, warped_after))
print(f"MI after (raw images, unmasked): {mi_after_raw:.6f}")

warped_after_reg = ants.apply_transforms(
    fixed=fixed_reg_syn,
    moving=moving_reg_ants,
    transformlist=tx["fwdtransforms"],
    interpolator="linear",
)
moving_metric_mask_final_in_fixed_syn = ants.apply_transforms(
    fixed=fixed_mask_syn,
    moving=moving_metric_mask_ants,
    transformlist=tx["fwdtransforms"],
    interpolator="nearestNeighbor",
)
overlap_final_syn = (np.asarray(fixed_mask_syn.numpy()) > 0) & (
    np.asarray(moving_metric_mask_final_in_fixed_syn.numpy()) > 0
)
mi_after = _masked_mutual_information_hist2d(
    fixed=np.asarray(fixed_reg_syn.numpy()),
    moving=np.asarray(warped_after_reg.numpy()),
    mask=overlap_final_syn,
)
print(f"MI after (preprocessed images, overlap-masked): {mi_after:.6f}")

lm_err = landmark_error_after_registration(
    moving_pts_mm=moving_pts_mm,
    fixed_pts_mm=fixed_pts_mm,
    invtransforms=tx["invtransforms"],
)
print(f"Post-SyN landmark RMSE: {lm_err['rmse_mm'] * 1e3:.2f} µm (max {lm_err['max_mm'] * 1e3:.2f} µm)")

# --- QC metrics focused on internal band alignment and rim overlap ---
fixed_mask_reg_arr = np.asarray(fixed_mask_ants.numpy()) > 0
fixed_mask_rim_arr = np.asarray(fixed_mask_orig_ants.numpy()) > 0
moving_mask_final_arr = np.asarray(moving_mask_final_in_fixed.numpy()) > 0
outside_arr = moving_mask_final_arr & (~fixed_mask_rim_arr)
moving_area_px = int(moving_mask_final_arr.sum())
outside_area_px = int(outside_arr.sum())
moving_outside_frac = float(outside_area_px / moving_area_px) if moving_area_px else float("nan")
moving_inside_frac = float(1.0 - moving_outside_frac) if np.isfinite(moving_outside_frac) else float("nan")

outside_dist_mean_um = float("nan")
outside_dist_max_um = float("nan")
if outside_area_px:
    outside_dist_um = distance_transform_edt(~fixed_mask_rim_arr).astype(np.float32) * float(ATLAS_VOXEL_UM)
    outside_dist_mean_um = float(outside_dist_um[outside_arr].mean())
    outside_dist_max_um = float(outside_dist_um[outside_arr].max())

band_corr_all = float("nan")
band_corr_hi = float("nan")
band_mse_all = float("nan")
band_mse_hi = float("nan")
band_hi_quantile = 0.90
band_edge_quantile = 0.95
band_edge_dice = float("nan")
if USE_FEATURE_IMAGES:
    moving_feat_warped = ants.apply_transforms(
        fixed=fixed_feat_ants,
        moving=moving_feat_ants,
        transformlist=tx["fwdtransforms"],
        interpolator="linear",
    )
    fixed_feat_arr = np.asarray(fixed_feat_ants.numpy(), dtype=np.float32)
    moving_feat_warped_arr = np.asarray(moving_feat_warped.numpy(), dtype=np.float32)
    overlap_eval = np.asarray(overlap_mask_final.numpy()) > 0
    if np.any(overlap_eval):
        vf = fixed_feat_arr[overlap_eval]
        vm = moving_feat_warped_arr[overlap_eval]
        band_corr_all = _pearson_corr(vf, vm)
        band_mse_all = float(np.mean((vf - vm) ** 2))

        thr = float(np.quantile(vf, band_hi_quantile))
        overlap_hi = overlap_eval & (fixed_feat_arr > thr)
        if np.any(overlap_hi):
            vf_hi = fixed_feat_arr[overlap_hi]
            vm_hi = moving_feat_warped_arr[overlap_hi]
            band_corr_hi = _pearson_corr(vf_hi, vm_hi)
            band_mse_hi = float(np.mean((vf_hi - vm_hi) ** 2))

        thr_f = float(np.quantile(vf, band_edge_quantile))
        thr_m = float(np.quantile(vm, band_edge_quantile))
        edge_f = overlap_eval & (fixed_feat_arr > thr_f)
        edge_m = overlap_eval & (moving_feat_warped_arr > thr_m)
        denom = int(edge_f.sum() + edge_m.sum())
        if denom:
            band_edge_dice = float(2.0 * (edge_f & edge_m).sum() / denom)

print(
    "QC metrics: "
    f"band_corr_all={band_corr_all:.4f}, band_corr_hi={band_corr_hi:.4f} (q={band_hi_quantile}), "
    f"band_edge_dice={band_edge_dice:.4f} (q={band_edge_quantile}), "
    f"moving_inside_frac={moving_inside_frac:.4f}, moving_outside_frac={moving_outside_frac:.4f}, "
    f"outside_dist_mean_um={outside_dist_mean_um:.1f}, outside_dist_max_um={outside_dist_max_um:.1f}"
)

fixed_qc = _resample_ants_to_spacing_um(fixed_ants, spacing_um=float(QC_VIS_SPACING_UM), interp_type=0)
warped_after_qc = ants.apply_transforms(
    fixed=fixed_qc,
    moving=moving_ants,
    transformlist=tx["fwdtransforms"],
    interpolator="linear",
)
moving_mask_final_in_fixed_qc = ants.apply_transforms(
    fixed=fixed_qc,
    moving=moving_mask_reg_full_ants,
    transformlist=tx["fwdtransforms"],
    interpolator="nearestNeighbor",
)

qc_overlay_png(
    fixed_img=fixed_qc,
    moving_warped=warped_after_qc,
    out_png=QC_PNG,
    title=(
        f"SyN(MI) | MI(opt,masked) {mi_before:.3f}→{mi_after:.3f} | "
        f"MI(raw) {mi_before_raw:.3f}→{mi_after_raw:.3f} | lm_rmse={lm_err['rmse_mm']*1e3:.1f} µm"
    ),
)
qc_overlay_png(
    fixed_img=fixed_qc,
    moving_warped=warped_after_qc,
    moving_mask_fixed=moving_mask_final_in_fixed_qc,
    out_png=QC_MASKED_PNG,
    title=(
        f"SyN(MI) (masked) | MI(opt,masked) {mi_before:.3f}→{mi_after:.3f} | "
        f"MI(raw) {mi_before_raw:.3f}→{mi_after_raw:.3f} | lm_rmse={lm_err['rmse_mm']*1e3:.1f} µm"
    ),
)

union_bbox = bbox_indices_from_mask(
    mask_xy=(np.asarray(moving_mask_init_in_fixed.numpy()) > 0) | (np.asarray(moving_mask_final_in_fixed.numpy()) > 0),
    pad_vox=int(MOVING_BBOX_PAD_VOX),
)
if union_bbox is not None:
    bb_lo, bb_hi = union_bbox
    before_bbox = ants.crop_indices(before_warped, bb_lo, bb_hi)
    after_bbox = ants.crop_indices(warped_after, bb_lo, bb_hi)

    b = normalize_robust(ants_numpy_yx(before_bbox).astype(np.float32))
    a = normalize_robust(ants_numpy_yx(after_bbox).astype(np.float32))
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(b, cmap="gray", interpolation="nearest")
    axes[0].set_title("Moving after init (before SyN)")
    axes[0].axis("off")
    axes[1].imshow(a, cmap="gray", interpolation="nearest")
    axes[1].set_title("Moving after SyN")
    axes[1].axis("off")
    fig.suptitle(
        f"Moving bbox crop | MI(opt,masked) {mi_before:.3f}→{mi_after:.3f} | "
        f"MI(raw) {mi_before_raw:.3f}→{mi_after_raw:.3f} | lm_rmse={lm_err['rmse_mm']*1e3:.1f} µm"
    )
    plt.tight_layout()
    plt.savefig(QC_MOVING_BBOX_PNG, dpi=int(QC_SAVE_DPI))
    plt.show()
else:
    print("Skipping moving bbox QC: moving mask bbox is empty.")

if fixed_crop_lower is not None and fixed_crop_upper is not None:
    fixed_zoom = ants.crop_indices(fixed_ants, fixed_crop_lower, fixed_crop_upper)
    fixed_zoom_vis = _resample_ants_to_spacing_um(fixed_zoom, spacing_um=float(QC_VIS_SPACING_UM), interp_type=0)
    moving_zoom = ants.apply_transforms(
        fixed=fixed_zoom_vis,
        moving=moving_ants,
        transformlist=tx["fwdtransforms"],
        interpolator="linear",
    )
    mask_zoom = ants.apply_transforms(
        fixed=fixed_zoom_vis,
        moving=moving_mask_reg_full_ants,
        transformlist=tx["fwdtransforms"],
        interpolator="nearestNeighbor",
    )
    qc_overlay_png(
        fixed_img=fixed_zoom_vis,
        moving_warped=moving_zoom,
        out_png=QC_ZOOM_PNG,
        title=(
            f"SyN(MI) zoom | MI(opt,masked) {mi_before:.3f}→{mi_after:.3f} | "
            f"MI(raw) {mi_before_raw:.3f}→{mi_after_raw:.3f} | lm_rmse={lm_err['rmse_mm']*1e3:.1f} µm"
        ),
    )
    qc_overlay_png(
        fixed_img=fixed_zoom_vis,
        moving_warped=moving_zoom,
        moving_mask_fixed=mask_zoom,
        out_png=QC_ZOOM_MASKED_PNG,
        title=(
            f"SyN(MI) zoom (masked) | MI(opt,masked) {mi_before:.3f}→{mi_after:.3f} | "
            f"MI(raw) {mi_before_raw:.3f}→{mi_after_raw:.3f} | lm_rmse={lm_err['rmse_mm']*1e3:.1f} µm"
        ),
    )

summary = {
    "crop_fixed_to_overlap": CROP_FIXED_TO_OVERLAP,
    "crop_pad_vox": CROP_PAD_VOX if CROP_FIXED_TO_OVERLAP else None,
    "crop_fixed_domain_for_syn": CROP_FIXED_DOMAIN_FOR_SYN if CROP_FIXED_TO_OVERLAP else None,
    "fixed_edge_guard_um": FIXED_EDGE_GUARD_UM,
    "moving_edge_guard_um": MOVING_EDGE_GUARD_UM,
    "use_n4": USE_N4,
    "n4_apply_to_fixed": N4_APPLY_TO_FIXED if USE_N4 else None,
    "n4_max_iterations": list(N4_MAX_ITERATIONS) if USE_N4 else None,
    "landmark_linear_transform_type": LANDMARK_LINEAR_TRANSFORM_TYPE,
    "syn_type_of_transform": SYN_TYPE_OF_TRANSFORM,
    "syn_grad_step": SYN_GRAD_STEP,
    "syn_flow_sigma": SYN_FLOW_SIGMA,
    "syn_total_sigma": SYN_TOTAL_SIGMA,
    "syn_metric": SYN_METRIC,
    "syn_metric_param": SYN_METRIC_PARAM,
    "syn_reg_iterations": SYN_REG_ITERATIONS,
    "use_affine_refine": USE_AFFINE_REFINE,
    "affine_refine_type_of_transform": AFFINE_REFINE_TYPE_OF_TRANSFORM if USE_AFFINE_REFINE else None,
    "affine_refine_metric": AFFINE_REFINE_METRIC if USE_AFFINE_REFINE else None,
    "affine_refine_mi_bins": AFFINE_REFINE_MI_BINS if USE_AFFINE_REFINE else None,
    "affine_refine_random_sampling_rate": (
        AFFINE_REFINE_RANDOM_SAMPLING_RATE if USE_AFFINE_REFINE else None
    ),
    "affine_refine_iterations": list(AFFINE_REFINE_ITERATIONS) if USE_AFFINE_REFINE else None,
    "affine_refine_shrink_factors": list(AFFINE_REFINE_SHRINK_FACTORS) if USE_AFFINE_REFINE else None,
    "affine_refine_smoothing_sigmas": list(AFFINE_REFINE_SMOOTHING_SIGMAS) if USE_AFFINE_REFINE else None,
    "affine_refine_grad_step": AFFINE_REFINE_GRAD_STEP if USE_AFFINE_REFINE else None,
    "affine_refine_applied": affine_refine_applied if USE_AFFINE_REFINE else None,
    "affine_refine_fwdtransforms": tx_affine_refine.get("fwdtransforms") if tx_affine_refine is not None else None,
    "affine_refine_invtransforms": tx_affine_refine.get("invtransforms") if tx_affine_refine is not None else None,
    "use_feature_images": USE_FEATURE_IMAGES,
    "feature_metric": "CC" if USE_FEATURE_IMAGES else None,
    "feature_sigma_um": FEATURE_SIGMA_UM if USE_FEATURE_IMAGES else None,
    "feature_weight": FEATURE_WEIGHT if USE_FEATURE_IMAGES else None,
    "feature_cc_radius": FEATURE_CC_RADIUS if USE_FEATURE_IMAGES else None,
    "use_mask_distance_metric": USE_MASK_DISTANCE_METRIC,
    "mask_distance_weight": MASK_DISTANCE_WEIGHT if USE_MASK_DISTANCE_METRIC else None,
    "mask_distance_smooth_sigma_um": MASK_DISTANCE_SMOOTH_SIGMA_UM if USE_MASK_DISTANCE_METRIC else None,
    "mask_distance_use_overlap": MASK_DISTANCE_USE_OVERLAP if USE_MASK_DISTANCE_METRIC else None,
    "fixed_metric_mask_dilate_um": FIXED_METRIC_MASK_DILATE_UM,
    "moving_metric_mask_dilate_um": MOVING_METRIC_MASK_DILATE_UM,
    "deformation_field_png": str(DEFORMATION_FIELD_PNG),
    "use_landmark_heatmap_metric": USE_LANDMARK_HEATMAP_METRIC,
    "landmark_heatmap_sigma_um": LANDMARK_HEATMAP_SIGMA_UM if USE_LANDMARK_HEATMAP_METRIC else None,
    "landmark_heatmap_weight": LANDMARK_HEATMAP_WEIGHT if USE_LANDMARK_HEATMAP_METRIC else None,
    "mi_before": mi_before,
    "mi_after": mi_after,
    "mi_raw_before": mi_before_raw,
    "mi_raw_after": mi_after_raw,
    "landmark_error_mm": lm_err,
    "qc_band_corr_all": band_corr_all if USE_FEATURE_IMAGES else None,
    "qc_band_corr_hi": band_corr_hi if USE_FEATURE_IMAGES else None,
    "qc_band_mse_all": band_mse_all if USE_FEATURE_IMAGES else None,
    "qc_band_mse_hi": band_mse_hi if USE_FEATURE_IMAGES else None,
    "qc_band_hi_quantile": band_hi_quantile if USE_FEATURE_IMAGES else None,
    "qc_band_edge_dice": band_edge_dice if USE_FEATURE_IMAGES else None,
    "qc_band_edge_quantile": band_edge_quantile if USE_FEATURE_IMAGES else None,
    "qc_moving_inside_frac": moving_inside_frac,
    "qc_moving_outside_frac": moving_outside_frac,
    "qc_outside_dist_mean_um": outside_dist_mean_um,
    "qc_outside_dist_max_um": outside_dist_max_um,
    "fwdtransforms": tx["fwdtransforms"],
    "invtransforms": tx["invtransforms"],
    "initial_transform_for_syn": initial_transform_for_syn,
    "paths": {
        "linear_init_mat": str(LINEAR_INIT_MAT_PATH),
        "fixed_nifti": str(FIXED_NIFTI),
        "moving_nifti": str(MOVING_NIFTI),
        "fixed_mask_orig_nifti": str(FIXED_MASK_ORIG_NIFTI),
        "fixed_metric_mask_nifti": str(FIXED_METRIC_MASK_NIFTI),
        "moving_metric_mask_nifti": str(MOVING_METRIC_MASK_NIFTI),
        "overlap_mask_nifti": str(OVERLAP_MASK_NIFTI),
        "deformation_field_png": str(DEFORMATION_FIELD_PNG),
        "fixed_reg_nifti": str(FIXED_REG_NIFTI),
        "moving_reg_nifti": str(MOVING_REG_NIFTI),
        "fixed_reg_raw_nifti": str(FIXED_REG_RAW_NIFTI),
        "moving_reg_raw_nifti": str(MOVING_REG_RAW_NIFTI),
        "fixed_feature_nifti": str(FIXED_FEAT_NIFTI) if USE_FEATURE_IMAGES else None,
        "moving_feature_nifti": str(MOVING_FEAT_NIFTI) if USE_FEATURE_IMAGES else None,
        "moving_mask_reg_full_nifti": str(MOVING_MASK_REG_FULL_NIFTI),
        "moving_mask_reg_nifti": str(MOVING_MASK_REG_NIFTI),
        "fixed_mask_signed_distance_nifti": str(FIXED_DT_NIFTI) if USE_MASK_DISTANCE_METRIC else None,
        "moving_mask_signed_distance_nifti": str(MOVING_DT_NIFTI) if USE_MASK_DISTANCE_METRIC else None,
        "fixed_landmark_heatmap_nifti": str(FIXED_LM_NIFTI) if USE_LANDMARK_HEATMAP_METRIC else None,
        "moving_landmark_heatmap_nifti": str(MOVING_LM_NIFTI) if USE_LANDMARK_HEATMAP_METRIC else None,
        "warped_before_nifti": str(WARPED_BEFORE_NIFTI),
        "warped_after_nifti": str(WARPED_AFTER_NIFTI),
        "moving_mask_warped_final_nifti": str(MOVING_MASK_WARPED_FINAL_NIFTI),
        "moving_mask_metric_warped_final_nifti": str(MOVING_MASK_METRIC_WARPED_FINAL_NIFTI),
        "overlap_mask_final_nifti": str(OVERLAP_MASK_FINAL_NIFTI),
        "qc_png": str(QC_PNG),
        "qc_masked_png": str(QC_MASKED_PNG),
        "qc_moving_bbox_png": str(QC_MOVING_BBOX_PNG),
        "qc_zoom_png": str(QC_ZOOM_PNG) if fixed_crop_lower is not None and fixed_crop_upper is not None else None,
        "qc_zoom_masked_png": str(QC_ZOOM_MASKED_PNG) if fixed_crop_lower is not None and fixed_crop_upper is not None else None,
    },
}
SUMMARY_JSON.write_text(json.dumps(summary, indent=2))
print(f"Wrote: {WARPED_BEFORE_NIFTI}")
print(f"Wrote: {WARPED_AFTER_NIFTI}")
print(f"Wrote: {MOVING_MASK_WARPED_FINAL_NIFTI}")
print(f"Wrote: {MOVING_MASK_METRIC_WARPED_FINAL_NIFTI}")
print(f"Wrote: {OVERLAP_MASK_FINAL_NIFTI}")
print(f"Wrote: {QC_PNG}")
print(f"Wrote: {QC_MASKED_PNG}")
if union_bbox is not None:
    print(f"Wrote: {QC_MOVING_BBOX_PNG}")
if fixed_crop_lower is not None and fixed_crop_upper is not None:
    print(f"Wrote: {QC_ZOOM_PNG}")
    print(f"Wrote: {QC_ZOOM_MASKED_PNG}")
print(f"Wrote: {SUMMARY_JSON}")


# %%
