#%%
# # 3D cortex mid-surface extraction (DevCCF E15.5)
#
# Goal: reduce a folded cortical ribbon mask (neocortex+mesocortex+allocortex) to a 2D manifold.
# This script computes fold-preserving scalar fields using Euclidean distance transforms:
#   u = d(pial) / (d(pial) + d(inner))
#   r_um = d(inner)_um - d(pial)_um   (so r=0 is the midsurface, r>0 toward pial)
# and extracts a thin u≈0.5 voxel band as the midsurface.
#
# Artifacts are written under `OUTDIR`.

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from skimage.measure import find_contours, marching_cubes
from skimage.morphology import (
    ball,
    binary_closing,
    binary_erosion,
    binary_opening,
    remove_small_holes,
    remove_small_objects,
)

from fishtools.ccf.cli_export_mask_edit_pack import _term_mask_from_annotation_yx
from fishtools.segmentation.mesh import Mesh, write_ply_binary_little_endian

# Fixed atlas (DevCCF E15.5 reference used elsewhere in `ccf/`).
ATLAS_NAME = "kim_dev_mouse_e15-5_lsfm_20um"


# === EDIT THESE ===

TERMS: tuple[str, ...] = ("neocortex", "mesocortex", "allocortex")
TERM_KIND: str = "auto"  # "auto" | "id" | "acronym" | "name"
MIDLINE_TERMS: tuple[str, ...] = ("neocortex", "mesocortex", "allocortex")
# Keep all connected components so disconnected allocortex is not dropped.
KEEP_LARGEST_COMPONENT_ONLY = False

KEEP_LEFT_HEMISPHERE_ONLY = True

# Downsample factor for all computations.
# Uses nearest-neighbor slicing (annotation[::DS,::DS,::DS]).
DS = 1

# Morphological cleanup on the downsampled mask.
FILL_HOLES = False
REMOVE_SMALL_OBJECTS_VOX = 10_000
FILL_SMALL_HOLES_VOX = 5_000
# Slightly stronger closing helps eliminate thin pinches/holes that can cause u=0.5 level-set tangles.
CLOSE_RADIUS_VOX = 1
OPEN_RADIUS_VOX = 0

# Erode cortex mask before computing the EDT field (voxels in the DS grid).
# Note: r_um is always computed to the boundary of the non-eroded `cortex_clean_3d`.
# A bit more erosion makes the fitted field avoid near-boundary slivers that create self-intersections.
ERODE_RADIUS_VOX_BEFORE_EDT = 1
MIDSURF_EPS = 0.03  # u-band around 0.5 (smaller => thinner but sparser)
BBOX_PAD_VOX = 4  # compute EDT on a padded bounding box around the fit mask
# Manual one-off overrides: connect split u=0.5 contours on these coronal slices.
MANUAL_CONNECT_CORONAL_SLICE_IS: tuple[int, ...] = (183, 185, 233, 234, 235, 236, 237)
MANUAL_CONNECT_CORONAL_PATH_TEMPLATE = "manual_coronal_midcurve_override_slice{slice_i}_yx.npy"
# Manual one-off overrides: connect split u=0.5 contours on these sagittal slices.
MANUAL_CONNECT_SAGITTAL_SLICE_KS: tuple[int, ...] = (212, 213, 214, 215, 216, 217, 218, 219)
MANUAL_CONNECT_SAGITTAL_PATH_TEMPLATE = "manual_sagittal_midcurve_override_slice{slice_k}_yx.npy"

# Visualization sampling.
PLOT_MAX_MASK_POINTS = 80_000
PLOT_MAX_RESULT_POINTS = 100_000

OUTDIR = Path("ccf/out/refextract/midsurface_neocortex_mesocortex_allocortex_3d")
OUTDIR.mkdir(parents=True, exist_ok=True)

BRAINGLOBE_CONFIG_DIR = Path("ccf/out/atlases/.brainglobe_config")


def _resolution_ijk_um(res: object) -> tuple[float, float, float]:
    if isinstance(res, (int, float)):
        r = float(res)
        return (r, r, r)
    if isinstance(res, (tuple, list)) and len(res) == 3 and all(isinstance(v, (int, float)) for v in res):
        return (float(res[0]), float(res[1]), float(res[2]))
    raise ValueError(f"Unsupported atlas.resolution: {res!r}")


def _keep_largest_component(mask: np.ndarray) -> np.ndarray:
    lab, n = ndi.label(mask)
    if n <= 1:
        return mask
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    return lab == int(np.argmax(sizes))


def _maybe_keep_largest_component(mask: np.ndarray) -> np.ndarray:
    if KEEP_LARGEST_COMPONENT_ONLY:
        return _keep_largest_component(mask)
    return mask


def _subsample_points(xyz: np.ndarray, *, max_points: int, seed: int = 0) -> np.ndarray:
    if xyz.shape[0] <= max_points:
        return xyz
    rng = np.random.default_rng(seed)
    idx = rng.choice(xyz.shape[0], size=int(max_points), replace=False)
    return xyz[idx]


def _polyline_length(path_yx: np.ndarray) -> float:
    if path_yx.ndim != 2 or path_yx.shape[0] < 2 or path_yx.shape[1] != 2:
        return 0.0
    d = np.diff(path_yx.astype(np.float64, copy=False), axis=0)
    return float(np.sum(np.sqrt(np.sum(d * d, axis=1))))


def _connect_two_largest_u05_contours_in_slice(
    *,
    u_yx: np.ndarray,
    mask_yx: np.ndarray,
    include_yx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    valid = (mask_yx & include_yx).astype(bool, copy=False)
    if not np.any(valid):
        empty = np.zeros((0,), dtype=np.int64)
        return u_yx, empty, empty, None

    contours = find_contours(u_yx.astype(np.float64, copy=False), level=0.5, mask=valid)
    contours = [np.asarray(c, dtype=np.float64) for c in contours if np.asarray(c).shape[0] >= 2]
    if len(contours) < 2:
        empty = np.zeros((0,), dtype=np.int64)
        return u_yx, empty, empty, None

    contours.sort(key=_polyline_length, reverse=True)
    c1 = contours[0]
    c2 = contours[1]

    # Connect the two curves at their closest approach, then choose the branch combination
    # that stays most interior (largest mean distance to boundary).
    diff = c1[:, None, :] - c2[None, :, :]
    d2 = np.sum(diff * diff, axis=2)
    i1, i2 = np.unravel_index(int(np.argmin(d2)), d2.shape)

    seg1_options = [c1[: i1 + 1], c1[i1:][::-1]]
    seg2_options = [c2[i2:], c2[: i2 + 1][::-1]]
    interior_dist = ndi.distance_transform_edt(valid).astype(np.float64, copy=False)

    best_path: np.ndarray | None = None
    best_center_score = -np.inf
    best_len = -np.inf
    for seg1 in seg1_options:
        if seg1.shape[0] < 2:
            continue
        for seg2 in seg2_options:
            if seg2.shape[0] < 2:
                continue
            a = seg1[-1]
            b = seg2[0]
            dist = float(np.linalg.norm(a - b))
            n_steps = max(2, int(np.ceil(dist)) + 1)
            t = np.linspace(0.0, 1.0, n_steps, dtype=np.float64)
            bridge = np.column_stack([(1.0 - t) * a[0] + t * b[0], (1.0 - t) * a[1] + t * b[1]])
            merged = np.vstack([seg1, bridge[1:-1], seg2]).astype(np.float64, copy=False)

            yi = np.clip(np.rint(merged[:, 0]).astype(np.int64, copy=False), 0, valid.shape[0] - 1)
            xi = np.clip(np.rint(merged[:, 1]).astype(np.int64, copy=False), 0, valid.shape[1] - 1)
            center_score = float(np.mean(interior_dist[yi, xi]))
            path_len = _polyline_length(merged)

            if center_score > best_center_score or (
                np.isclose(center_score, best_center_score) and path_len > best_len
            ):
                best_center_score = center_score
                best_len = path_len
                best_path = merged

    if best_path is None:
        empty = np.zeros((0,), dtype=np.int64)
        return u_yx, empty, empty, None
    merged_path = best_path
    y = np.rint(merged_path[:, 0]).astype(np.int64, copy=False)
    x = np.rint(merged_path[:, 1]).astype(np.int64, copy=False)

    out = u_yx.copy()
    h, w = out.shape
    y_idx: list[int] = []
    x_idx: list[int] = []
    for yi, xi in zip(y, x, strict=False):
        yi_clamped = int(np.clip(yi, 0, h - 1))
        xi_clamped = int(np.clip(xi, 0, w - 1))
        out[yi_clamped, xi_clamped] = np.float32(0.5)
        y_idx.append(yi_clamped)
        x_idx.append(xi_clamped)
    if not y_idx:
        empty = np.zeros((0,), dtype=np.int64)
        return out, empty, empty, merged_path
    y_arr = np.asarray(y_idx, dtype=np.int64)
    x_arr = np.asarray(x_idx, dtype=np.int64)
    uniq = np.unique(np.column_stack([y_arr, x_arr]), axis=0)
    return out, uniq[:, 0].astype(np.int64, copy=False), uniq[:, 1].astype(np.int64, copy=False), merged_path


def _mask_to_xyz_um(
    mask_ijk: np.ndarray,
    *,
    res_ijk_um: tuple[float, float, float],
) -> np.ndarray:
    pts_ijk = np.argwhere(mask_ijk)
    if pts_ijk.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    res_i, res_j, res_k = (float(res_ijk_um[0]), float(res_ijk_um[1]), float(res_ijk_um[2]))
    # Plot in x=k, y=j, z=i coordinates.
    xyz_um = np.empty((pts_ijk.shape[0], 3), dtype=np.float64)
    xyz_um[:, 0] = pts_ijk[:, 2].astype(np.float64) * res_k
    xyz_um[:, 1] = pts_ijk[:, 1].astype(np.float64) * res_j
    xyz_um[:, 2] = pts_ijk[:, 0].astype(np.float64) * res_i
    return xyz_um


def _set_axes_equal_3d(ax: object, xyz_um: np.ndarray) -> None:
    if xyz_um.shape[0] == 0:
        return
    mins = np.min(xyz_um, axis=0)
    maxs = np.max(xyz_um, axis=0)
    ctr = (mins + maxs) / 2.0
    half = (maxs - mins) / 2.0
    radius = float(np.max(half))
    if not np.isfinite(radius) or radius <= 0:
        return
    ax.set_xlim(ctr[0] - radius, ctr[0] + radius)
    ax.set_ylim(ctr[1] - radius, ctr[1] + radius)
    ax.set_zlim(ctr[2] - radius, ctr[2] + radius)
    set_box_aspect = getattr(ax, "set_box_aspect", None)
    if callable(set_box_aspect):
        ax.set_box_aspect((1, 1, 1))


def _save_scatter_3d(
    *,
    out_png: Path,
    mask_xyz_um: np.ndarray,
    result_xyz_um: np.ndarray,
    title: str,
    mask_alpha: float = 0.04,
    result_alpha: float = 0.8,
) -> None:
    fig = plt.figure(figsize=(8.5, 7.5), layout="constrained")
    ax = fig.add_subplot(111, projection="3d")
    if mask_xyz_um.shape[0] > 0:
        ax.scatter(
            mask_xyz_um[:, 0],
            mask_xyz_um[:, 1],
            mask_xyz_um[:, 2],
            s=0.15,
            c="#808080",
            alpha=float(mask_alpha),
            linewidths=0,
        )
    if result_xyz_um.shape[0] > 0:
        ax.scatter(
            result_xyz_um[:, 0],
            result_xyz_um[:, 1],
            result_xyz_um[:, 2],
            s=0.35,
            c="#d62728",
            alpha=float(result_alpha),
            linewidths=0,
        )
    ax.set_title(title)
    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um)")
    ax.set_zlabel("z (um)")
    _set_axes_equal_3d(ax, mask_xyz_um if mask_xyz_um.shape[0] else result_xyz_um)
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def _boundary_partition_pial_inner(
    *,
    cortex_mask: np.ndarray,
    brain_mask: np.ndarray,
    cortex_reference_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    structure6 = ndi.generate_binary_structure(3, 1)
    cortex_mask = cortex_mask.astype(bool, copy=False)
    brain_mask = brain_mask.astype(bool, copy=False)
    if cortex_mask.shape != brain_mask.shape:
        raise ValueError(f"Shape mismatch: cortex_mask={cortex_mask.shape} brain_mask={brain_mask.shape}")

    if cortex_reference_mask is None:
        cortex_reference_mask = cortex_mask
    cortex_reference_mask = cortex_reference_mask.astype(bool, copy=False)
    if cortex_reference_mask.shape != cortex_mask.shape:
        raise ValueError(
            f"Shape mismatch: cortex_reference_mask={cortex_reference_mask.shape} cortex_mask={cortex_mask.shape}"
        )

    interior6 = binary_erosion(cortex_mask, footprint=structure6)
    boundary = cortex_mask & ~interior6

    outside_brain = ~brain_mask
    other_brain = brain_mask & ~cortex_reference_mask

    pial = cortex_mask & ndi.binary_dilation(outside_brain, structure=structure6)
    inner = cortex_mask & ndi.binary_dilation(other_brain, structure=structure6)
    pial &= boundary
    inner &= boundary

    overlap = pial & inner
    if np.any(overlap):
        inner = inner & ~overlap

    if not np.any(pial) or not np.any(inner):
        # If we erode the cortex_mask before fitting, adjacency-to-outside can become empty.
        # Fall back to a distance-based partitioning of boundary voxels:
        # pial boundary is closer to outside-brain; inner boundary is closer to other-brain (non-cortex in brain).
        if not np.any(boundary):
            raise ValueError("cortex boundary was empty.")

        d_out = ndi.distance_transform_edt(brain_mask).astype(np.float32, copy=False)
        d_other = ndi.distance_transform_edt(~other_brain).astype(np.float32, copy=False)
        is_pial = d_out <= d_other
        pial = boundary & is_pial
        inner = boundary & ~is_pial
    else:
        remaining = boundary & ~(pial | inner)
        if np.any(remaining):
            d_pial = ndi.distance_transform_edt(~pial).astype(np.float32, copy=False)
            d_inner = ndi.distance_transform_edt(~inner).astype(np.float32, copy=False)
            assign_pial = d_pial <= d_inner
            pial = pial | (remaining & assign_pial)
            inner = inner | (remaining & ~assign_pial)

    if not np.any(pial):
        raise ValueError("pial boundary was empty (check brain_mask and cortex_reference_mask).")
    if not np.any(inner):
        raise ValueError("inner boundary was empty (check brain_mask and cortex_reference_mask).")
    if np.any(pial & inner):
        raise ValueError("pial and inner boundaries overlapped after assignment.")
    if not np.all((pial | inner) == boundary):
        missing = int(np.count_nonzero(boundary & ~(pial | inner)))
        raise ValueError(f"Boundary partition was incomplete (missing={missing} voxels).")

    return pial, inner


def _bbox_slices(mask: np.ndarray, *, pad: int) -> tuple[slice, slice, slice]:
    pts = np.argwhere(mask)
    if pts.size == 0:
        raise ValueError("Mask was empty; cannot compute bounding box.")
    mins = pts.min(axis=0).astype(np.int64)
    maxs = pts.max(axis=0).astype(np.int64) + 1
    pad = int(max(0, int(pad)))
    z0 = max(int(mins[0]) - pad, 0)
    y0 = max(int(mins[1]) - pad, 0)
    x0 = max(int(mins[2]) - pad, 0)
    z1 = min(int(maxs[0]) + pad, int(mask.shape[0]))
    y1 = min(int(maxs[1]) + pad, int(mask.shape[1]))
    x1 = min(int(maxs[2]) + pad, int(mask.shape[2]))
    return (slice(z0, z1), slice(y0, y1), slice(x0, x1))


# ## Phase 0: Load atlas annotation volume

if BRAINGLOBE_CONFIG_DIR.exists():
    os.environ["BRAINGLOBE_CONFIG_DIR"] = str(BRAINGLOBE_CONFIG_DIR.resolve())

from brainglobe_atlasapi import BrainGlobeAtlas  # noqa: E402

atlas = BrainGlobeAtlas(ATLAS_NAME)
annotation_3d = np.asarray(atlas.annotation)
res_i_um, res_j_um, res_k_um = _resolution_ijk_um(atlas.resolution)
print(f"ATLAS_NAME={ATLAS_NAME!r}")
print(f"annotation_3d shape={annotation_3d.shape}, dtype={annotation_3d.dtype}")
print(f"atlas.resolution={atlas.resolution!r} (um)")

if DS <= 0:
    raise ValueError(f"DS must be >= 1, got {DS}")
annotation_3d = annotation_3d[::DS, ::DS, ::DS]
brain_mask_3d = annotation_3d != 0

res_ds_ijk_um = (res_i_um * DS, res_j_um * DS, res_k_um * DS)
np.save(OUTDIR / "annotation_3d_ds.npy", annotation_3d)
np.save(OUTDIR / "brain_mask_3d_ds.npy", brain_mask_3d.astype(np.bool_))
print(f"Downsample DS={DS} => shape={annotation_3d.shape} res_ds_ijk_um={res_ds_ijk_um}")


# ## Phase 1: Build 3D cortex mask (term + descendants)

ann_flat = annotation_3d.reshape(annotation_3d.shape[0], -1)
mask_flat = _term_mask_from_annotation_yx(
    annotation_yx=ann_flat,
    terms=TERMS,
    kind=TERM_KIND,  # type: ignore[arg-type]
    combine="any",
    invert=False,
    atlas=atlas,
)
cortex_3d = mask_flat.reshape(annotation_3d.shape).astype(bool)
if KEEP_LEFT_HEMISPHERE_ONLY:
    cortex_3d = cortex_3d[:, :, : cortex_3d.shape[2] // 2]
    brain_mask_3d = brain_mask_3d[:, :, : cortex_3d.shape[2]]
    annotation_3d = annotation_3d[:, :, : cortex_3d.shape[2]]

print(f"Cortex voxels (raw)={int(np.count_nonzero(cortex_3d))}")
np.save(OUTDIR / "cortex_mask_3d_ds.npy", cortex_3d.astype(np.bool_))

# Include-mask for clipping midsurface calculations.
ann_flat_mid = annotation_3d.reshape(annotation_3d.shape[0], -1)
mask_flat_mid = _term_mask_from_annotation_yx(
    annotation_yx=ann_flat_mid,
    terms=MIDLINE_TERMS,
    kind=TERM_KIND,  # type: ignore[arg-type]
    combine="any",
    invert=False,
    atlas=atlas,
)
midline_include_3d = mask_flat_mid.reshape(annotation_3d.shape).astype(bool)
midline_include_3d = midline_include_3d[:, :, : cortex_3d.shape[2]] & cortex_3d
np.save(OUTDIR / "midline_include_neo_meso_3d_ds.npy", midline_include_3d.astype(np.bool_))


# ## Phase 2: Cleanup mask (for both methods)

cortex_clean_3d = cortex_3d.copy()
cortex_clean_3d = _maybe_keep_largest_component(cortex_clean_3d)
if FILL_HOLES:
    cortex_clean_3d = ndi.binary_fill_holes(cortex_clean_3d)
if CLOSE_RADIUS_VOX > 0:
    cortex_clean_3d = binary_closing(cortex_clean_3d, footprint=ball(int(CLOSE_RADIUS_VOX)))
if OPEN_RADIUS_VOX > 0:
    cortex_clean_3d = binary_opening(cortex_clean_3d, footprint=ball(int(OPEN_RADIUS_VOX)))
if REMOVE_SMALL_OBJECTS_VOX > 0:
    cortex_clean_3d = remove_small_objects(cortex_clean_3d, min_size=int(REMOVE_SMALL_OBJECTS_VOX))
if FILL_SMALL_HOLES_VOX > 0:
    cortex_clean_3d = remove_small_holes(cortex_clean_3d, area_threshold=int(FILL_SMALL_HOLES_VOX))
cortex_clean_3d = _maybe_keep_largest_component(cortex_clean_3d)

print(f"Cortex voxels (clean)={int(np.count_nonzero(cortex_clean_3d))}")
np.save(OUTDIR / "cortex_mask_clean_3d_ds.npy", cortex_clean_3d.astype(np.bool_))


# ## Phase 3: EDT halfway mid-surface (fit on eroded mask)

cortex_fit_3d = cortex_clean_3d.copy()
if ERODE_RADIUS_VOX_BEFORE_EDT > 0:
    cortex_fit_3d = binary_erosion(cortex_fit_3d, footprint=ball(int(ERODE_RADIUS_VOX_BEFORE_EDT)))
cortex_fit_3d = _maybe_keep_largest_component(cortex_fit_3d)
if not np.any(cortex_fit_3d):
    raise ValueError(
        f"Erosion emptied cortex mask (ERODE_RADIUS_VOX_BEFORE_EDT={ERODE_RADIUS_VOX_BEFORE_EDT}, DS={DS})."
    )
print(f"Cortex voxels (fit)={int(np.count_nonzero(cortex_fit_3d))}")
np.save(OUTDIR / "cortex_mask_fit_3d_ds.npy", cortex_fit_3d.astype(np.bool_))

pial_b0, inner_b1 = _boundary_partition_pial_inner(
    cortex_mask=cortex_fit_3d, brain_mask=brain_mask_3d, cortex_reference_mask=cortex_clean_3d
)
print(f"[edt] b0(pial)={int(np.count_nonzero(pial_b0))} b1(inner)={int(np.count_nonzero(inner_b1))}")
np.save(OUTDIR / "laplace_b0_pial_3d_ds.npy", pial_b0.astype(np.bool_))
np.save(OUTDIR / "laplace_b1_inner_3d_ds.npy", inner_b1.astype(np.bool_))

crop = _bbox_slices(cortex_fit_3d, pad=int(BBOX_PAD_VOX))
z0, y0, x0 = (int(crop[0].start), int(crop[1].start), int(crop[2].start))

mask_crop = cortex_fit_3d[crop]
b0_crop = pial_b0[crop]
b1_crop = inner_b1[crop]
include_crop = midline_include_3d[crop]

d_pial_um = ndi.distance_transform_edt(~b0_crop, sampling=res_ds_ijk_um).astype(np.float32, copy=False)
d_inner_um = ndi.distance_transform_edt(~b1_crop, sampling=res_ds_ijk_um).astype(np.float32, copy=False)
den_um = d_pial_um + d_inner_um
u_crop = np.zeros(mask_crop.shape, dtype=np.float32)
ok = mask_crop & (den_um > 0.0)
u_crop[ok] = (d_pial_um[ok] / den_um[ok]).astype(np.float32, copy=False)
u_crop[b0_crop] = 0.0
u_crop[b1_crop] = 1.0

for manual_slice_i in MANUAL_CONNECT_CORONAL_SLICE_IS:
    local_override_i = int(manual_slice_i) - int(z0)
    if 0 <= local_override_i < int(u_crop.shape[0]):
        u_slice, y_bridge, x_bridge, merged_path = _connect_two_largest_u05_contours_in_slice(
            u_yx=u_crop[local_override_i, :, :],
            mask_yx=mask_crop[local_override_i, :, :],
            include_yx=include_crop[local_override_i, :, :],
        )
        u_crop[local_override_i, :, :] = u_slice
        if y_bridge.size > 0:
            mask_crop[local_override_i, y_bridge, x_bridge] = True
            include_crop[local_override_i, y_bridge, x_bridge] = True
            # Persist the manual bridge in masks used by midsurface_coords contour extraction.
            np.save(OUTDIR / "cortex_mask_fit_3d_ds.npy", cortex_fit_3d.astype(np.bool_))
            np.save(OUTDIR / "midline_include_neo_meso_3d_ds.npy", midline_include_3d.astype(np.bool_))
        if merged_path is not None and merged_path.shape[0] >= 2:
            # Save full-size (y, x) polyline so midsurface_coords can explicitly apply this manual override.
            merged_path_full = merged_path.copy()
            merged_path_full[:, 0] += float(y0)
            merged_path_full[:, 1] += float(x0)
            out_name = MANUAL_CONNECT_CORONAL_PATH_TEMPLATE.format(slice_i=int(manual_slice_i))
            np.save(OUTDIR / out_name, merged_path_full.astype(np.float32, copy=False))
        print(
            f"[manual] coronal slice {int(manual_slice_i)}: "
            f"connected two largest u=0.5 contours with {int(y_bridge.size)} bridge voxels"
        )
    else:
        print(
            f"[manual] coronal slice {int(manual_slice_i)} outside crop "
            f"[{z0}, {z0 + u_crop.shape[0] - 1}]; skipping manual contour connection."
        )

for manual_slice_k in MANUAL_CONNECT_SAGITTAL_SLICE_KS:
    local_override_k = int(manual_slice_k) - int(x0)
    if 0 <= local_override_k < int(u_crop.shape[2]):
        u_slice, y_bridge, x_bridge, merged_path = _connect_two_largest_u05_contours_in_slice(
            u_yx=u_crop[:, :, local_override_k],
            mask_yx=mask_crop[:, :, local_override_k],
            include_yx=include_crop[:, :, local_override_k],
        )
        u_crop[:, :, local_override_k] = u_slice
        if y_bridge.size > 0:
            mask_crop[y_bridge, x_bridge, local_override_k] = True
            include_crop[y_bridge, x_bridge, local_override_k] = True
            # Persist the manual bridge in masks used by midsurface_coords contour extraction.
            np.save(OUTDIR / "cortex_mask_fit_3d_ds.npy", cortex_fit_3d.astype(np.bool_))
            np.save(OUTDIR / "midline_include_neo_meso_3d_ds.npy", midline_include_3d.astype(np.bool_))
        if merged_path is not None and merged_path.shape[0] >= 2:
            # Save full-size (y, x) polyline in sagittal-plane coordinates (i, j).
            merged_path_full = merged_path.copy()
            merged_path_full[:, 0] += float(z0)
            merged_path_full[:, 1] += float(y0)
            out_name = MANUAL_CONNECT_SAGITTAL_PATH_TEMPLATE.format(slice_k=int(manual_slice_k))
            np.save(OUTDIR / out_name, merged_path_full.astype(np.float32, copy=False))
        print(
            f"[manual] sagittal slice {int(manual_slice_k)}: "
            f"connected two largest u=0.5 contours with {int(y_bridge.size)} bridge voxels"
        )
    else:
        print(
            f"[manual] sagittal slice {int(manual_slice_k)} outside crop "
            f"[{x0}, {x0 + u_crop.shape[2] - 1}]; skipping manual contour connection."
        )

u = np.zeros(cortex_fit_3d.shape, dtype=np.float32)
u[crop] = u_crop
np.save(OUTDIR / "halfway_u_3d_ds.npy", u.astype(np.float32, copy=False))

crop_origin_ijk = np.asarray([z0, y0, x0], dtype=np.int32)
np.save(OUTDIR / "halfway_crop_origin_ijk.npy", crop_origin_ijk)

# Signed radial coordinate (microns): r=0 midsurface, r>0 toward pial.
# Compute r to the boundary of the non-eroded mask by using distances to boundary seeds and
# masking by `cortex_clean_3d` (not the eroded fit mask).
mask_clean_crop = cortex_clean_3d[crop]
r_um_crop = np.full(mask_crop.shape, np.nan, dtype=np.float32)
t_um_crop = np.full(mask_crop.shape, np.nan, dtype=np.float32)
ok_clean = mask_clean_crop & (den_um > 0.0)
r_um_crop[ok_clean] = (d_inner_um[ok_clean] - d_pial_um[ok_clean]).astype(np.float32, copy=False)
t_um_crop[ok_clean] = den_um[ok_clean].astype(np.float32, copy=False)
np.save(OUTDIR / "halfway_r_um_crop.npy", r_um_crop)
np.save(OUTDIR / "halfway_thickness_um_crop.npy", t_um_crop)

mid_band_crop = mask_crop & include_crop & (np.abs(u_crop - 0.5) <= float(MIDSURF_EPS))
mid_band = np.zeros(cortex_fit_3d.shape, dtype=bool)
mid_band[crop] = _maybe_keep_largest_component(mid_band_crop)
print(f"[edt] mid_band voxels={int(np.count_nonzero(mid_band))}")
np.save(OUTDIR / "midsurface_halfway_midband_3d_ds.npy", mid_band.astype(np.bool_))

# Mesh the full MIDLINE_TERMS portion of the u=0.5 surface.
verts_ijk_um, faces, _, _ = marching_cubes(
    u_crop,
    level=0.5,
    spacing=res_ds_ijk_um,
    mask=mask_crop & include_crop,
)
offset_ijk_um = np.asarray(
    [z0 * res_ds_ijk_um[0], y0 * res_ds_ijk_um[1], x0 * res_ds_ijk_um[2]],
    dtype=np.float32,
)
verts_ijk_um = verts_ijk_um.astype(np.float32, copy=False) + offset_ijk_um[None, :]
faces = faces.astype(np.int32, copy=False)
np.save(OUTDIR / "midsurface_halfway_mc_verts_ijk_um.npy", verts_ijk_um.astype(np.float32, copy=False))
np.save(OUTDIR / "midsurface_halfway_mc_faces.npy", faces.astype(np.int32, copy=False))
print(f"[edt] marching_cubes verts={verts_ijk_um.shape[0]} faces={faces.shape[0]}")

mesh = Mesh(vertices_xyz=verts_ijk_um[:, [2, 1, 0]].astype(np.float32, copy=False), faces=faces)
write_ply_binary_little_endian(OUTDIR / "midsurface_halfway_u0p5.ply", mesh)


# ## Phase 4: 3D visualization (Matplotlib)

mask_xyz_um = _mask_to_xyz_um(cortex_fit_3d, res_ijk_um=res_ds_ijk_um)
mask_xyz_um = _subsample_points(mask_xyz_um, max_points=int(PLOT_MAX_MASK_POINTS), seed=0)

mid_xyz_um = _mask_to_xyz_um(mid_band, res_ijk_um=res_ds_ijk_um)
mid_xyz_um = _subsample_points(mid_xyz_um, max_points=int(PLOT_MAX_RESULT_POINTS), seed=2)

_save_scatter_3d(
    out_png=OUTDIR / "mpl3d_midsurface_halfway_edt.png",
    mask_xyz_um=mask_xyz_um,
    result_xyz_um=mid_xyz_um,
    title=f"EDT halfway midsurface voxels (u≈0.5, DS={DS}, erode={ERODE_RADIUS_VOX_BEFORE_EDT})",
)

print(f"Wrote artifacts to {OUTDIR}")
