# %% [markdown]
# # Partial Section Registration to DevCCF (frontloaded defs)
#
# Same workflow as `ccf/register_partial_section.py`, but with *all helper function definitions grouped near the top*,
# so you can step through the later cells without scrolling around to find `def ...` blocks.
#
# Run cells sequentially. Each phase saves outputs for inspection.

# %%
from __future__ import annotations

import json
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import zarr
from brainglobe_atlasapi import BrainGlobeAtlas
from IPython import get_ipython
from matplotlib.colors import ListedColormap
from matplotlib.widgets import Button, Slider
from scipy.ndimage import gaussian_filter, sobel
from scipy.ndimage import rotate as ndimage_rotate

from fishtools.ccf.landmark import LandmarkRegistrationOutputs, P1Landmarks
from fishtools.ccf.landmark_ui import (
    pick_atlas_slice_idx,
    pick_paired_landmarks,
    pick_paired_landmarks_overlay,
    pick_rotation_deg,
)
from fishtools.ccf.sitk_utils import compute_similarity2d_from_landmarks, pixels_to_physical_um
from fishtools.io.workspace import Workspace
from fishtools.preprocess.config import NumpyEncoder

# Use widget backend for VS Code interactive mode
ip = get_ipython()
if ip is not None:
    ip.run_line_magic("matplotlib", "widget")


# === EDIT THESE ===

# Workspace configuration
WORKSPACE = Path("/working/20251001_JaxA3_Coro11")
ws = Workspace(WORKSPACE)
ROI = "1whole" #ws.rois[4]

STITCH_CODEBOOK = "pi"  # analysis/deconv/stitch--{ROI}+{STITCH_CODEBOOK}/fused.zarr

# Atlas configuration
ATLAS_NAME = "kim_dev_mouse_e15-5_lsfm_20um"
ATLAS_PLANE = "sagittal" if "Sag" in WORKSPACE.name else "coronal"

# Sample configuration (resolved via Workspace)
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
PREVIEW_DOWNSAMPLE = 16


def crop_to_content(img: np.ndarray, mask: np.ndarray, pad: int = 10) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Crop image to bounding box of mask with padding."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        raise ValueError("Mask is empty; pick a different atlas slice (annotation is all zeros).")
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    rmin = max(0, rmin - pad)
    rmax = min(img.shape[0], rmax + pad + 1)
    cmin = max(0, cmin - pad)
    cmax = min(img.shape[1], cmax + pad + 1)

    return img[rmin:rmax, cmin:cmax], (rmin, rmax, cmin, cmax)


def _pallium_subtree_ids(atlas: BrainGlobeAtlas) -> set[int]:
    df = atlas.lookup_df
    if "name" not in df.columns or "acronym" not in df.columns or "id" not in df.columns:
        raise ValueError(f"Unexpected atlas.lookup_df schema: columns={list(df.columns)}")

    candidates = df[
        (df["name"].astype(str).str.lower() == "pallium")
        | (df["acronym"].astype(str).str.lower().isin({"pal", "pallium"}))
    ]
    if candidates.empty:
        candidates = df[df["name"].astype(str).str.lower().str.contains("pallium", na=False)]
    if candidates.empty:
        raise ValueError("Could not find a 'pallium' structure in atlas lookup table.")

    candidate_ids = [int(v) for v in candidates["id"].tolist()]
    best_id = min(candidate_ids, key=lambda rid: len(atlas.structures[rid]["structure_id_path"]))

    subtree_ids: set[int] = set()
    for struct in atlas.structures_list:
        path = struct.get("structure_id_path", [])
        if isinstance(path, list) and best_id in {int(v) for v in path}:
            subtree_ids.add(int(struct["id"]))
    return subtree_ids


def _ventricles_subtree_ids(atlas: BrainGlobeAtlas) -> set[int]:
    df = atlas.lookup_df
    if "name" not in df.columns or "acronym" not in df.columns or "id" not in df.columns:
        raise ValueError(f"Unexpected atlas.lookup_df schema: columns={list(df.columns)}")

    candidates = df[
        (df["name"].astype(str).str.lower() == "ventricles")
        | (df["acronym"].astype(str).str.lower().isin({"ventricles"}))
    ]
    if candidates.empty:
        candidates = df[df["name"].astype(str).str.lower().str.contains("ventric", na=False)]
    if candidates.empty:
        raise ValueError("Could not find a 'ventricles' structure in atlas lookup table.")

    candidate_ids = [int(v) for v in candidates["id"].tolist()]
    best_id = min(candidate_ids, key=lambda rid: len(atlas.structures[rid]["structure_id_path"]))

    subtree_ids: set[int] = set()
    for struct in atlas.structures_list:
        path = struct.get("structure_id_path", [])
        if isinstance(path, list) and best_id in {int(v) for v in path}:
            subtree_ids.add(int(struct["id"]))
    return subtree_ids


def _array_to_sitk(arr: np.ndarray, spacing: tuple[float, float]) -> sitk.Image:
    img = sitk.GetImageFromArray(arr.astype(np.float32))
    img.SetSpacing(spacing)
    return img


def _save_p1_landmarks(
    *,
    fixed_points_cropped_xy: list[tuple[float, float]],
    moving_points_fullres_xy_in_rotated_crop: list[tuple[float, float]],
) -> None:
    OUT.write_p1_landmarks(
        P1Landmarks(
            prior_rotation_deg=PRIOR_ROTATION_DEG,
            prior_flip_x=PRIOR_FLIP_X,
            atlas_slice_idx=atlas_slice_idx,
            atlas_plane=ATLAS_PLANE,
            atlas_name=ATLAS_NAME,
            atlas_voxel_um=ATLAS_VOXEL,
            sample_channel=SAMPLE_CHANNEL,
            sample_z_idx=SAMPLE_Z_IDX,
            sample_voxel_xy_um=SAMPLE_VOXEL_XY,
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


def _run_p1_similarity2d(
    *,
    fixed_points_cropped_xy: list[tuple[float, float]],
    moving_points_fullres_xy_in_rotated_crop: list[tuple[float, float]],
    show: bool,
) -> tuple[sitk.Similarity2DTransform, list[tuple[float, float]], list[tuple[float, float]], np.ndarray]:
    if len(fixed_points_cropped_xy) != len(moving_points_fullres_xy_in_rotated_crop):
        raise ValueError(
            "Mismatched landmark pair counts: "
            f"fixed={len(fixed_points_cropped_xy)} vs moving={len(moving_points_fullres_xy_in_rotated_crop)}"
        )

    _save_p1_landmarks(
        fixed_points_cropped_xy=fixed_points_cropped_xy,
        moving_points_fullres_xy_in_rotated_crop=moving_points_fullres_xy_in_rotated_crop,
    )

    print(f"Landmark pairs: {len(fixed_points_cropped_xy)}")

    fixed_points_full = [(x + ATLAS_CROP_OFFSET[0], y + ATLAS_CROP_OFFSET[1]) for x, y in fixed_points_cropped_xy]
    moving_points_full = [
        (x + SAMPLE_CROP_OFFSET[0], y + SAMPLE_CROP_OFFSET[1]) for x, y in moving_points_fullres_xy_in_rotated_crop
    ]

    fixed_pts_phys = pixels_to_physical_um(fixed_points_full, ATLAS_VOXEL)
    moving_pts_phys = pixels_to_physical_um(moving_points_full, SAMPLE_VOXEL_XY)

    transform_similarity = compute_similarity2d_from_landmarks(fixed_pts_phys, moving_pts_phys)

    print("Similarity2D transform:")
    print(f"  Center: {transform_similarity.GetCenter()}")
    print(f"  Angle: {np.degrees(transform_similarity.GetAngle()):.2f}°")
    print(f"  Scale: {transform_similarity.GetScale():.4f}")
    print(f"  Translation: {transform_similarity.GetTranslation()}")

    errors: list[float] = []
    for fp, mp in zip(fixed_pts_phys, moving_pts_phys):
        transformed = transform_similarity.TransformPoint(fp)
        error = float(np.linalg.norm(np.array(transformed) - np.array(mp)))
        errors.append(error)

    max_error = max(errors)
    mean_error = float(np.mean(errors))
    print(f"Landmark error: max={max_error:.2f} µm, mean={mean_error:.2f} µm")
    if max_error > 100:
        print("WARNING: High landmark error - check landmark correspondences")

    sitk.WriteTransform(transform_similarity, str(P1_TFM_PATH))
    print(f"Saved: {P1_TFM_PATH}")

    atlas_slice_full_masked = atlas_slice_full.copy().astype(np.float32)
    atlas_slice_full_masked[~atlas_brain_mask] = 0.0

    atlas_sitk = _array_to_sitk(atlas_slice_full_masked, (ATLAS_VOXEL, ATLAS_VOXEL))
    sample_sitk = _array_to_sitk(sample_slice_full, (SAMPLE_VOXEL_XY, SAMPLE_VOXEL_XY))

    sample_warped = sitk.Resample(
        sample_sitk,
        atlas_sitk,
        transform_similarity,
        sitk.sitkLinear,
        0.0,
        sample_sitk.GetPixelID(),
    )
    sample_warped_arr = sitk.GetArrayFromImage(sample_warped)

    r0, r1, c0, c1 = atlas_crop_bbox
    sample_warped_arr_preview = sample_warped_arr[r0:r1, c0:c1]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(atlas_slice, cmap="gray")
    axes[0].set_title("Atlas cropped (FIXED)")
    axes[0].axis("off")

    axes[1].imshow(sample_warped_arr_preview, cmap="gray")
    axes[1].set_title("Sample warped to atlas space (raw)")
    axes[1].axis("off")

    overlay = np.zeros((*atlas_slice.shape, 3), dtype=np.float32)
    ATLAS_LO_PERCENTILE = 1.0
    ATLAS_HI_PERCENTILE = 99.8
    ATLAS_GAMMA = 0.85
    atlas_vals = atlas_slice[atlas_slice > 0] if np.any(atlas_slice > 0) else atlas_slice
    atlas_lo, atlas_hi = (float(v) for v in np.percentile(atlas_vals, [ATLAS_LO_PERCENTILE, ATLAS_HI_PERCENTILE]))
    atlas_norm = (atlas_slice - atlas_lo) / (atlas_hi - atlas_lo + 1e-8)
    atlas_norm = np.clip(atlas_norm, 0.0, 1.0) ** ATLAS_GAMMA
    SAMPLE_LO_PERCENTILE = 1.0
    SAMPLE_HI_PERCENTILE = 99.5
    SAMPLE_GAMMA = 0.7
    sample_lo, sample_hi = (
        float(v) for v in np.percentile(sample_warped_arr_preview, [SAMPLE_LO_PERCENTILE, SAMPLE_HI_PERCENTILE])
    )
    sample_norm = (sample_warped_arr_preview - sample_lo) / (sample_hi - sample_lo + 1e-8)
    sample_norm = np.clip(sample_norm, 0.0, 1.0) ** SAMPLE_GAMMA

    MOVING_ALPHA = 1.0
    SAMPLE_GAIN = 0.75
    ATLAS_ALPHA = 0.8
    EDGE_SIGMA = 0.25
    EDGE_THRESH = 0.25
    EDGE_GAMMA = 2.5

    atlas_blur = gaussian_filter(atlas_norm.astype(np.float32), sigma=EDGE_SIGMA)
    gx = sobel(atlas_blur, axis=1)
    gy = sobel(atlas_blur, axis=0)
    edges = np.hypot(gx, gy)
    edges_p99 = float(np.percentile(edges, 99))
    edges_norm = edges / (edges_p99 + 1e-8)
    edges_norm = np.clip(edges_norm, 0.0, 1.0)
    edges_norm = np.clip((edges_norm - EDGE_THRESH) / (1.0 - EDGE_THRESH + 1e-8), 0.0, 1.0)
    edges_norm = edges_norm**EDGE_GAMMA
    atlas_edge_enhanced = np.clip(0.8 * atlas_norm + 1.2 * edges_norm, 0.0, 1.0)
    overlay[..., 0] = ATLAS_ALPHA * atlas_edge_enhanced  # R
    overlay[..., 1] = MOVING_ALPHA * SAMPLE_GAIN * sample_norm  # G
    overlay[..., 2] = ATLAS_ALPHA * atlas_edge_enhanced  # B (magenta = R+B)

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay (magenta=atlas, green=sample)")
    axes[2].axis("off")

    fig.tight_layout()
    fig.savefig(OUT.p1_result_png, dpi=150)
    print(f"Saved: {OUT.p1_result_png}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    print("Phase 1 complete.")
    return transform_similarity, fixed_pts_phys, moving_pts_phys, overlay


def _on_landmarks_change(
    fixed_points_cropped_xy: list[tuple[float, float]],
    moving_points_fullres_xy_in_rotated_crop: list[tuple[float, float]],
) -> None:
    globals()["fixed_points"] = list(fixed_points_cropped_xy)
    globals()["moving_points_fullres"] = list(moving_points_fullres_xy_in_rotated_crop)

    if len(fixed_points_cropped_xy) < 3:
        _save_p1_landmarks(
            fixed_points_cropped_xy=fixed_points_cropped_xy,
            moving_points_fullres_xy_in_rotated_crop=moving_points_fullres_xy_in_rotated_crop,
        )
        print(f"Saved: {OUT.p1_landmarks_json}")
        print("Need at least 3 landmark pairs to compute Similarity2D. Keep clicking, then Save again.")
        return

    transform_similarity_, fixed_pts_phys_, moving_pts_phys_, overlay_ = _run_p1_similarity2d(
        fixed_points_cropped_xy=fixed_points_cropped_xy,
        moving_points_fullres_xy_in_rotated_crop=moving_points_fullres_xy_in_rotated_crop,
        show=False,
    )
    globals()["transform_similarity"] = transform_similarity_
    globals()["fixed_pts_phys"] = fixed_pts_phys_
    globals()["moving_pts_phys"] = moving_pts_phys_
    globals()["overlay"] = overlay_


def _save_overlay_landmarks(
    fixed_points_cropped_xy: list[tuple[float, float]],
    moving_points_in_fixed_cropped_xy: list[tuple[float, float]],
) -> None:
    moving_points_fullres_xy_in_rotated_crop: list[tuple[float, float]] = []
    roundtrip_errors_px: list[float] = []
    for x_cropped, y_cropped in moving_points_in_fixed_cropped_xy:
        fixed_full_px = (float(x_cropped) + ATLAS_CROP_OFFSET[0], float(y_cropped) + ATLAS_CROP_OFFSET[1])
        fixed_phys = (fixed_full_px[0] * ATLAS_VOXEL, fixed_full_px[1] * ATLAS_VOXEL)
        moving_phys = transform_similarity.TransformPoint(fixed_phys)
        moving_full_px = (moving_phys[0] / SAMPLE_VOXEL_XY, moving_phys[1] / SAMPLE_VOXEL_XY)
        moving_points_fullres_xy_in_rotated_crop.append(
            (moving_full_px[0] - SAMPLE_CROP_OFFSET[0], moving_full_px[1] - SAMPLE_CROP_OFFSET[1])
        )

        fixed_phys_roundtrip = inverse_similarity.TransformPoint(moving_phys)
        fixed_full_px_roundtrip = (fixed_phys_roundtrip[0] / ATLAS_VOXEL, fixed_phys_roundtrip[1] / ATLAS_VOXEL)
        fixed_cropped_px_roundtrip = (
            fixed_full_px_roundtrip[0] - ATLAS_CROP_OFFSET[0],
            fixed_full_px_roundtrip[1] - ATLAS_CROP_OFFSET[1],
        )
        roundtrip_errors_px.append(
            float(
                np.hypot(
                    fixed_cropped_px_roundtrip[0] - float(x_cropped),
                    fixed_cropped_px_roundtrip[1] - float(y_cropped),
                )
            )
        )

    _save_p1_landmarks(
        fixed_points_cropped_xy=fixed_points_cropped_xy,
        moving_points_fullres_xy_in_rotated_crop=moving_points_fullres_xy_in_rotated_crop,
    )
    overlay_picker.fig.savefig(out_path, dpi=150)
    print(f"Saved: {OUT.p1_landmarks_json}")
    print(f"Saved: {out_path}")
    if roundtrip_errors_px:
        print(f"Overlay back-transform roundtrip error: max={max(roundtrip_errors_px):.3g} px")


def _otsu_threshold(values: np.ndarray, *, nbins: int = 256) -> float:
    v = values[np.isfinite(values)].astype(np.float64, copy=False)
    if v.size == 0:
        return 0.0

    vmin, vmax = (float(x) for x in np.percentile(v, [0.5, 99.5]))
    if vmax <= vmin:
        return vmin

    hist, bin_edges = np.histogram(v, bins=int(nbins), range=(vmin, vmax))
    hist = hist.astype(np.float64, copy=False)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]
    mean1 = np.cumsum(hist * bin_centers) / np.maximum(weight1, 1e-12)
    mean2 = (np.cumsum((hist * bin_centers)[::-1]) / np.maximum(weight2[::-1], 1e-12))[::-1]

    between = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
    if between.size == 0:
        return float(np.median(v))
    idx = int(np.argmax(between))
    return float(bin_centers[idx])


def _threshold_update_overlay(
    *,
    threshold: float,
    threshold_img: np.ndarray,
    overlay_artist: object,
    hud: object,
    fig: object,
) -> None:
    mask = (threshold_img >= float(threshold)).astype(np.uint8)
    overlay_artist.set_data(mask)
    frac = float(mask.mean())
    hud.set_text(f"thr={float(threshold):.3g}  above={frac:.1%}")
    fig.canvas.draw_idle()


def _threshold_on_save_clicked(
    _: object,
    *,
    slider: Slider,
    out_json: Path,
    out_png: Path,
    preview_downsample: int,
    prior_rotation_deg: int,
    prior_flip_x: bool,
    fig: object,
) -> None:
    threshold = float(slider.val)
    payload = {
        "threshold": threshold,
        "image": "sample_preview_rotated",
        "preview_downsample": int(preview_downsample),
        "prior_rotation_deg": int(prior_rotation_deg),
        "prior_flip_x": bool(prior_flip_x),
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    fig.savefig(out_png, dpi=150)
    print(f"Saved: {out_json}")
    print(f"Saved: {out_png}")


def _threshold_debounce_flush(
    *_args: object,
    debounce_timer: object,
    pending_threshold: dict[str, float],
    threshold_img: np.ndarray,
    overlay_artist: object,
    hud: object,
    fig: object,
) -> None:
    debounce_timer.stop()
    _threshold_update_overlay(
        threshold=float(pending_threshold["value"]),
        threshold_img=threshold_img,
        overlay_artist=overlay_artist,
        hud=hud,
        fig=fig,
    )


def _threshold_on_slider_change(
    val: float,
    *,
    pending_threshold: dict[str, float],
    debounce_timer: object,
) -> None:
    pending_threshold["value"] = float(val)
    debounce_timer.stop()
    debounce_timer.start()


atlas = BrainGlobeAtlas(ATLAS_NAME)
atlas_reference_slices = atlas.reference if ATLAS_PLANE == "coronal" else atlas.reference.transpose(2, 1, 0)
atlas_annotation_slices = atlas.annotation if ATLAS_PLANE == "coronal" else atlas.annotation.transpose(2, 1, 0)

arr = zarr.open(str(SAMPLE_ZARR), mode="r")
keys = list(arr.attrs.get("key", []))
ch_idx = keys.index(SAMPLE_CHANNEL) if SAMPLE_CHANNEL in keys else 0
sample_slice_full_raw = np.asarray(arr[SAMPLE_Z_IDX, :, :, ch_idx])

# Crop sample to non-zero content (preview crop for rotation slider only)
sample_mask_raw = sample_slice_full_raw > np.percentile(sample_slice_full_raw, 5)  # threshold above noise
sample_slice, _sample_crop_bbox = crop_to_content(sample_slice_full_raw, sample_mask_raw, pad=50)


# %%
# ----------------------
# Rotate (orientation prior)
# ----------------------

ROTATION_PREVIEW_DOWNSAMPLE = PREVIEW_DOWNSAMPLE * 2
sample_preview = sample_slice[::ROTATION_PREVIEW_DOWNSAMPLE, ::ROTATION_PREVIEW_DOWNSAMPLE]

existing_p1 = OUT.try_read_p1_landmarks()
initial_rotation_deg = existing_p1.prior_rotation_deg if existing_p1 is not None else 0
initial_flip_x = existing_p1.prior_flip_x if existing_p1 is not None else False

rotation_picker = pick_rotation_deg(
    moving_image_yx=sample_preview,
    initial_deg=initial_rotation_deg,
    initial_flip_x=initial_flip_x,
)


# %%
# ----------------------
# Pick atlas slice
# ----------------------

PRIOR_ROTATION_DEG = rotation_picker.deg
PRIOR_FLIP_X = rotation_picker.flip_x
print(f"Selected rotation: {PRIOR_ROTATION_DEG}° (flip_x={PRIOR_FLIP_X})")

sample_slice_full_raw_pose = sample_slice_full_raw[:, ::-1] if PRIOR_FLIP_X else sample_slice_full_raw
if PRIOR_ROTATION_DEG != 0:
    sample_slice_full = ndimage_rotate(sample_slice_full_raw_pose, PRIOR_ROTATION_DEG, reshape=True, order=1)
else:
    sample_slice_full = sample_slice_full_raw_pose

sample_crop_bbox_rot: tuple[int, int, int, int]
if (
    existing_p1 is not None
    and existing_p1.prior_rotation_deg == PRIOR_ROTATION_DEG
    and existing_p1.prior_flip_x == PRIOR_FLIP_X
    and existing_p1.sample_rotated_full_shape_yx == tuple(int(x) for x in sample_slice_full.shape)
):
    sample_crop_bbox_rot = existing_p1.sample_rotated_crop_bbox
    sr0, sr1, sc0, sc1 = sample_crop_bbox_rot
    sample_slice_rotated = sample_slice_full[sr0:sr1, sc0:sc1]
else:
    sample_mask = sample_slice_full > np.percentile(sample_slice_full, 5)  # threshold above noise
    sample_slice_rotated, sample_crop_bbox_rot = crop_to_content(sample_slice_full, sample_mask, pad=50)

SAMPLE_CROP_OFFSET = (sample_crop_bbox_rot[2], sample_crop_bbox_rot[0])

ATLAS_Z_PREVIEW_DOWNSAMPLE = PREVIEW_DOWNSAMPLE * 2
atlas_z_sample_preview = sample_slice_rotated[::ATLAS_Z_PREVIEW_DOWNSAMPLE, ::ATLAS_Z_PREVIEW_DOWNSAMPLE]

atlas_slice_picker = pick_atlas_slice_idx(
    atlas_reference_zyx=atlas_reference_slices,
    moving_image_yx=atlas_z_sample_preview,
    initial_idx=(existing_p1.atlas_slice_idx if existing_p1 is not None and existing_p1.atlas_slice_idx is not None else 0),
    z_min_idx=120,
    z_max_idx=320,
)


# %%
# ----------------------
# Phase 1: Prepare atlas crop + overlays
# ----------------------

atlas_slice_idx = atlas_slice_picker.idx
print(f"Selected atlas_slice_idx={atlas_slice_idx}")

atlas_slice_full = atlas_reference_slices[atlas_slice_idx, :, :]
atlas_annotation_full = atlas_annotation_slices[atlas_slice_idx, :, :]

atlas_brain_mask = atlas_annotation_full > 0

atlas_slice_masked = atlas_slice_full.copy()
atlas_slice_masked[~atlas_brain_mask] = 0

atlas_slice, atlas_crop_bbox = crop_to_content(atlas_slice_masked, atlas_brain_mask, pad=5)

if ATLAS_PLANE == "sagittal":
    keep_w = int(round(atlas_slice.shape[1] * 0.6))
    keep_w = max(1, min(int(atlas_slice.shape[1]), keep_w))
    atlas_slice = atlas_slice[:, :keep_w]
    r0, r1, c0, c1 = atlas_crop_bbox
    atlas_crop_bbox = (r0, r1, c0, c0 + keep_w)

PALLIUM_OVERLAY_MASK: np.ndarray | None = None
VENTRICLES_OVERLAY_MASK: np.ndarray | None = None
try:
    pallium_ids = _pallium_subtree_ids(atlas)
    r0, r1, c0, c1 = atlas_crop_bbox
    pallium_full = np.isin(atlas_annotation_full, list(pallium_ids))
    PALLIUM_OVERLAY_MASK = pallium_full[r0:r1, c0:c1]
except Exception as exc:
    print(f"WARNING: Pallium overlay disabled: {exc}")

try:
    vent_ids = _ventricles_subtree_ids(atlas)
    r0, r1, c0, c1 = atlas_crop_bbox
    vent_full = np.isin(atlas_annotation_full, list(vent_ids))
    VENTRICLES_OVERLAY_MASK = vent_full[r0:r1, c0:c1]
except Exception as exc:
    print(f"WARNING: Ventricles overlay disabled: {exc}")

ATLAS_CROP_OFFSET = (atlas_crop_bbox[2], atlas_crop_bbox[0])  # (x_offset, y_offset)

sample_preview_rotated = sample_slice_rotated[::PREVIEW_DOWNSAMPLE, ::PREVIEW_DOWNSAMPLE]

_existing_landmark_picker = globals().get("landmark_picker")
if _existing_landmark_picker is not None:
    if plt.fignum_exists(_existing_landmark_picker.fig.number):
        choice = input(
            "Existing landmark editor is open. Save before restarting? "
            "[s]ave/[d]iscard/[c]ancel (default: cancel): "
        ).strip().lower()
        if choice in {"s", "save"}:
            fixed_prev = list(_existing_landmark_picker.fixed_points_cropped_xy)
            moving_prev = list(_existing_landmark_picker.moving_points_fullres_xy_in_rotated_crop)
            n_pairs = min(len(fixed_prev), len(moving_prev))
            if n_pairs == 0:
                print("No complete landmark pairs to save.")
            else:
                if len(fixed_prev) != len(moving_prev):
                    print(
                        f"Dropping incomplete pair: fixed={len(fixed_prev)} moving={len(moving_prev)}; saving {n_pairs} pair(s)."
                    )
                _save_p1_landmarks(
                    fixed_points_cropped_xy=fixed_prev[:n_pairs],
                    moving_points_fullres_xy_in_rotated_crop=moving_prev[:n_pairs],
                )
                print(f"Saved: {OUT.p1_landmarks_json}")
        elif choice in {"d", "discard"}:
            print("Discarding unsaved landmark edits.")
        else:
            raise RuntimeError("Canceled: keeping existing landmark editor session.")

    _existing_landmark_picker.close()
    globals()["landmark_picker"] = None

initial_fixed: list[tuple[float, float]] = []
initial_moving: list[tuple[float, float]] = []
p1_for_init = OUT.try_read_p1_landmarks()
if p1_for_init is not None:
    if p1_for_init.prior_rotation_deg != PRIOR_ROTATION_DEG or p1_for_init.prior_flip_x != PRIOR_FLIP_X:
        print(
            "Found existing landmarks, but rotation/flip differs from the current selection; not loading them. "
            f"(saved rot={p1_for_init.prior_rotation_deg}°, flip_x={p1_for_init.prior_flip_x})"
        )
    elif p1_for_init.atlas_slice_idx is not None and p1_for_init.atlas_slice_idx != atlas_slice_idx:
        print(
            "Found existing landmarks, but atlas_slice_idx differs from the current selection; not loading them. "
            f"(saved slice_idx={p1_for_init.atlas_slice_idx})"
        )
    else:
        initial_fixed = p1_for_init.fixed_points_cropped_xy
        initial_moving = p1_for_init.moving_points_fullres_xy_in_rotated_crop
        print(f"Loaded existing landmarks from {OUT.p1_landmarks_json}: pairs={len(initial_fixed)}")

fixed_overlays: list[tuple[np.ndarray, tuple[float, float, float, float]]] = []
if PALLIUM_OVERLAY_MASK is not None:
    fixed_overlays.append((PALLIUM_OVERLAY_MASK, (0.1, 0.4, 1.0, 0.30)))  # blue
if VENTRICLES_OVERLAY_MASK is not None:
    fixed_overlays.append((VENTRICLES_OVERLAY_MASK, (0.5, 1.0, 0.5, 0.25)))  # light green

landmark_picker = pick_paired_landmarks(
    fixed_image_yx=atlas_slice,
    fixed_overlays=fixed_overlays,
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
# ----------------------
# Phase 1: Compute Similarity2D
# ----------------------

fixed_points, moving_points_fullres = landmark_picker.get_points()
transform_similarity, fixed_pts_phys, moving_pts_phys, overlay = _run_p1_similarity2d(
    fixed_points_cropped_xy=fixed_points,
    moving_points_fullres_xy_in_rotated_crop=moving_points_fullres,
    show=True,
)

# %% Threshold
if "sample_preview_rotated" not in globals():
    raise RuntimeError("Missing sample_preview_rotated; run the earlier cells through Phase 1 first.")

P1_THRESHOLD_JSON = OUT.root / "p1_threshold.json"
P1_THRESHOLD_PREVIEW_PNG = OUT.root / "p1_threshold_preview.png"

threshold_img = sample_preview_rotated.astype(np.float32, copy=False)
threshold_vals = threshold_img[np.isfinite(threshold_img)]
if threshold_vals.size == 0:
    raise ValueError("sample_preview_rotated contains no finite values.")

existing_threshold: float | None = None
if P1_THRESHOLD_JSON.exists():
    payload = json.loads(P1_THRESHOLD_JSON.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "threshold" in payload:
        existing_threshold = float(payload["threshold"])
        print(f"Loaded existing threshold: {existing_threshold:g} ({P1_THRESHOLD_JSON})")

thr_init = existing_threshold if existing_threshold is not None else _otsu_threshold(threshold_vals)
thr_min, thr_max = (float(x) for x in np.percentile(threshold_vals, [0.5, 99.5]))
if thr_max <= thr_min:
    thr_max = thr_min + 1.0
thr_step = max((thr_max - thr_min) / 200.0, 1e-6)

fig, ax = plt.subplots(figsize=(7, 7), constrained_layout=False)
fig.subplots_adjust(bottom=0.16)
disp_lo, disp_hi = (float(x) for x in np.percentile(threshold_vals, [0.5, 99.5]))
ax.imshow(threshold_img, cmap="gray", vmin=disp_lo, vmax=disp_hi)

overlay_cmap = ListedColormap([(1.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.35)])
mask_init = (threshold_img >= float(np.clip(thr_init, thr_min, thr_max))).astype(np.uint8)
overlay_artist = ax.imshow(mask_init, cmap=overlay_cmap, vmin=0, vmax=1, interpolation="nearest")
ax.set_title(f"sample_preview_rotated (downsample={PREVIEW_DOWNSAMPLE}x)")
ax.axis("off")

hud = ax.text(
    0.01,
    0.99,
    "",
    transform=ax.transAxes,
    ha="left",
    va="top",
    color="white",
    fontsize=10,
    bbox={"facecolor": "black", "alpha": 0.5, "pad": 3},
)

slider_ax = fig.add_axes([0.14, 0.06, 0.62, 0.03])
slider = Slider(
    ax=slider_ax,
    label="threshold",
    valmin=thr_min,
    valmax=thr_max,
    valinit=float(np.clip(thr_init, thr_min, thr_max)),
    valfmt="%.3g",
    valstep=thr_step,
)

button_ax = fig.add_axes([0.80, 0.045, 0.16, 0.06])
save_btn = Button(button_ax, "Save", color="0.85", hovercolor="0.95")

DEBOUNCE_MS = 75
pending_threshold: dict[str, float] = {"value": float(slider.val)}
debounce_timer = fig.canvas.new_timer(interval=DEBOUNCE_MS)

debounce_timer.add_callback(
    partial(
        _threshold_debounce_flush,
        debounce_timer=debounce_timer,
        pending_threshold=pending_threshold,
        threshold_img=threshold_img,
        overlay_artist=overlay_artist,
        hud=hud,
        fig=fig,
    )
)
slider.on_changed(partial(_threshold_on_slider_change, pending_threshold=pending_threshold, debounce_timer=debounce_timer))
save_btn.on_clicked(
    partial(
        _threshold_on_save_clicked,
        slider=slider,
        out_json=P1_THRESHOLD_JSON,
        out_png=P1_THRESHOLD_PREVIEW_PNG,
        preview_downsample=int(PREVIEW_DOWNSAMPLE),
        prior_rotation_deg=int(PRIOR_ROTATION_DEG),
        prior_flip_x=bool(PRIOR_FLIP_X),
        fig=fig,
    )
)

_threshold_update_overlay(
    threshold=float(slider.val),
    threshold_img=threshold_img,
    overlay_artist=overlay_artist,
    hud=hud,
    fig=fig,
)

# %% [markdown]
# ### Overlay editor (optional)
#
# Edit landmark pairs on the final overlay:
# - Click order: atlas point (magenta structures) → sample point (green structures)
# - Ctrl+click near an existing marker deletes that pair
# - Use "Save landmarks" to write `p1_landmarks.json` (and re-save `p1_overlay_landmarks.png`)
# %%
overlay_with_landmarks = overlay.copy()
inverse_similarity = transform_similarity.GetInverse()
moving_pts_in_fixed_phys = [inverse_similarity.TransformPoint(mp) for mp in moving_pts_phys]
moving_pts_in_fixed_full_px = [(x / ATLAS_VOXEL, y / ATLAS_VOXEL) for x, y in moving_pts_in_fixed_phys]
moving_pts_in_fixed_cropped_px = [
    (x - ATLAS_CROP_OFFSET[0], y - ATLAS_CROP_OFFSET[1]) for x, y in moving_pts_in_fixed_full_px
]

out_path = OUT.root / "p1_overlay_landmarks.png"

overlay_picker = pick_paired_landmarks_overlay(
    overlay_image_yx_rgb=overlay_with_landmarks,
    initial_fixed_points_cropped_xy=list(fixed_points),
    initial_moving_points_in_fixed_cropped_xy=list(moving_pts_in_fixed_cropped_px),
    min_pairs=3,
    on_save=_save_overlay_landmarks,
    title="Overlay + landmarks (magenta=atlas, green=sample)",
    fixed_label="atlas point",
    moving_label="sample point",
)

# %%
