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
OVERLAY_TERMS_NO_ALLOCORTEX: tuple[str, ...] = ("neocortex", "mesocortex")
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
# Manual one-off overrides: connect split u=0.5 contours on affected coronal slices.
MANUAL_CONNECT_CORONAL_SLICE_IS: tuple[int, ...] = (
    183,
    184,
    185,
    230,
    231,
    233,
    234,
    235,
    236,
    237,
    238,
    239,
    240,
    241,
    242,
    246,
    247,
    248,
    284,
    289,
    290,
    291,
    292,
    293,
    294,
    295,
    296,
    297,
    298,
)
MANUAL_CONNECT_CORONAL_PATH_TEMPLATE = "manual_coronal_midcurve_override_slice{slice_i}_yx.npy"
# Connect overrides are only applied when the second contour is non-trivial and not too far.
MANUAL_CONNECT_MIN_SECOND_CONTOUR_RATIO = 0.08
MANUAL_CONNECT_MAX_CLOSEST_DISTANCE_VOX = 12.0
# Continuity guard: when alternatives exist, prefer lower adjacent-slice p95 displacement.
MANUAL_CONNECT_MAX_ADJACENT_P95_VOX = 8.0
# Apply adaptive largest-vs-connected selection only where jumps were observed.
MANUAL_ADAPTIVE_CONNECT_CORONAL_SLICE_IS: tuple[int, ...] = (242, 289, 290, 297, 298)
# Manual one-off overrides: use only largest contour path on these coronal slices
# to avoid snapping to tiny wall-adjacent fragments.
MANUAL_KEEP_LARGEST_CORONAL_SLICE_IS: tuple[int, ...] = (
    230,
    231,
    232,
    234,
    235,
    236,
    237,
    238,
    239,
    240,
    241,
    289,
    290,
    291,
    292,
    293,
    297,
    298,
)
# Slice-specific smoothing override in output coordinates:
# rewrite target slice as midpoint interpolation between two neighboring manual paths.
MANUAL_BLEND_CORONAL_NEIGHBOR_SLICES: dict[int, tuple[int, int]] = {233: (232, 234)}
# Slice-specific direction correction: enforce the target path orientation to match the previous slice.
MANUAL_ALIGN_CORONAL_DIRECTION_WITH_PREV_SLICE_IS: tuple[int, ...] = (242, 246, 247, 248, 249)
# Manual one-off overrides: connect split u=0.5 contours on these sagittal slices.
MANUAL_CONNECT_SAGITTAL_SLICE_KS: tuple[int, ...] = (212, 213, 214, 215, 216, 217, 218, 219)
MANUAL_CONNECT_SAGITTAL_PATH_TEMPLATE = "manual_sagittal_midcurve_override_slice{slice_k}_yx.npy"
# Manual one-off overrides: keep only the largest sagittal contour path (ignore short disconnected segments).
MANUAL_KEEP_LARGEST_SAGITTAL_SLICE_KS: tuple[int, ...] = (
    176,
    177,
    178,
    179,
    180,
    181,
    182,
    183,
    184,
    185,
    186,
    187,
    188,
)
# Slice-local sagittal smoothing override:
# target := (1 - w_next) * prev + w_next * next after arc-length alignment.
MANUAL_BLEND_SAGITTAL_SLICE_K_WEIGHTS: dict[int, tuple[int, int, float]] = {221: (220, 222, 0.25)}
# Local sagittal kink softening: blend target with mean(prev,next) only in a narrow t-window.
# value tuple = (prev_k, next_k, t_start, t_peak, t_end, w_peak)
MANUAL_SOFTEN_SAGITTAL_LOCAL_KINKS: dict[int, tuple[int, int, float, float, float, float]] = {
    220: (219, 221, 0.12, 0.26, 0.30, 0.45)
}
# Visualization sampling.
PLOT_MAX_MASK_POINTS = 80_000
PLOT_MAX_RESULT_POINTS = 100_000

# Performance toggles for iterative midline regeneration.
SAVE_INTERMEDIATE_NPY = True
WRITE_MESH = True
WRITE_PLOT = True

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


def _resample_polyline(path_yx: np.ndarray, *, n_points: int) -> np.ndarray:
    if path_yx.ndim != 2 or path_yx.shape[1] != 2 or path_yx.shape[0] < 2:
        raise ValueError(f"Expected path shape (N,2) with N>=2, got {path_yx.shape}.")
    if int(n_points) < 2:
        raise ValueError("n_points must be >=2.")
    pts = path_yx.astype(np.float64, copy=False)
    seg = np.sqrt(np.sum(np.diff(pts, axis=0) ** 2, axis=1))
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(s[-1])
    if not np.isfinite(total) or total <= 0.0:
        return np.repeat(pts[:1], int(n_points), axis=0).astype(np.float64, copy=False)
    q = np.linspace(0.0, total, int(n_points), dtype=np.float64)
    y = np.interp(q, s, pts[:, 0])
    x = np.interp(q, s, pts[:, 1])
    return np.column_stack([y, x]).astype(np.float64, copy=False)


def _connect_two_largest_u05_contours_in_slice(
    *,
    u_yx: np.ndarray,
    mask_yx: np.ndarray,
    include_yx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, tuple[float, float, float, int] | None]:
    valid = (mask_yx & include_yx).astype(bool, copy=False)
    if not np.any(valid):
        empty = np.zeros((0,), dtype=np.int64)
        return u_yx, empty, empty, None, None

    contours = find_contours(u_yx.astype(np.float64, copy=False), level=0.5, mask=valid)
    contours = [np.asarray(c, dtype=np.float64) for c in contours if np.asarray(c).shape[0] >= 2]
    if len(contours) < 2:
        empty = np.zeros((0,), dtype=np.int64)
        if not contours:
            return u_yx, empty, empty, None, (0.0, 0.0, np.inf, 0)
        largest_len = _polyline_length(contours[0])
        return u_yx, empty, empty, None, (largest_len, 0.0, np.inf, len(contours))

    contours.sort(key=_polyline_length, reverse=True)
    c1 = contours[0]
    c2 = contours[1]
    len1 = _polyline_length(c1)
    len2 = _polyline_length(c2)

    # Connect the two curves at their closest approach, then choose the branch combination
    # that stays most interior (largest mean distance to boundary).
    diff = c1[:, None, :] - c2[None, :, :]
    d2 = np.sum(diff * diff, axis=2)
    i1, i2 = np.unravel_index(int(np.argmin(d2)), d2.shape)
    closest_distance = float(np.sqrt(d2[i1, i2]))

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
        return u_yx, empty, empty, None, (len1, len2, closest_distance, len(contours))
    merged_path = best_path
    y = np.rint(merged_path[:, 0]).astype(np.int64, copy=False)
    x = np.rint(merged_path[:, 1]).astype(np.int64, copy=False)

    out = u_yx.copy()
    if y.size == 0:
        empty = np.zeros((0,), dtype=np.int64)
        return out, empty, empty, merged_path, (len1, len2, closest_distance, len(contours))

    h, w = out.shape
    y_arr = np.clip(y, 0, h - 1).astype(np.int64, copy=False)
    x_arr = np.clip(x, 0, w - 1).astype(np.int64, copy=False)
    out[y_arr, x_arr] = np.float32(0.5)

    uniq = np.unique(np.column_stack([y_arr, x_arr]), axis=0)
    return (
        out,
        uniq[:, 0].astype(np.int64, copy=False),
        uniq[:, 1].astype(np.int64, copy=False),
        merged_path,
        (len1, len2, closest_distance, len(contours)),
    )


def _extract_largest_u05_contour_in_slice(
    *,
    u_yx: np.ndarray,
    mask_yx: np.ndarray,
    include_yx: np.ndarray,
) -> np.ndarray | None:
    valid = (mask_yx & include_yx).astype(bool, copy=False)
    if not np.any(valid):
        return None
    vals = u_yx[valid]
    finite = np.isfinite(vals)
    if not np.any(finite):
        return None
    min_v = float(np.min(vals[finite]))
    max_v = float(np.max(vals[finite]))
    if not (min_v <= 0.5 <= max_v):
        return None
    contours = find_contours(u_yx.astype(np.float64, copy=False), level=0.5, mask=valid)
    contours = [np.asarray(c, dtype=np.float64) for c in contours if np.asarray(c).shape[0] >= 2]
    if not contours:
        return None
    return max(contours, key=_polyline_length)


def _load_saved_manual_path(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    arr = np.load(path).astype(np.float64, copy=False)
    if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] < 2:
        return None
    if not np.isfinite(arr).all():
        return None
    return arr


def _resample_for_pairwise(path_yx: np.ndarray, *, n_points: int) -> np.ndarray:
    return _resample_polyline(path_yx, n_points=max(64, int(n_points)))


def _adjacent_path_displacement_p95(path_a_yx: np.ndarray, path_b_yx: np.ndarray) -> float:
    n_points = max(64, min(int(path_a_yx.shape[0]), int(path_b_yx.shape[0])))
    a_rs = _resample_for_pairwise(path_a_yx, n_points=n_points)
    b_rs = _resample_for_pairwise(path_b_yx, n_points=n_points)
    d = np.sqrt(np.sum((b_rs - a_rs) ** 2, axis=1))
    return float(np.percentile(d, 95.0))


def _orientation_cost_against_neighbors(
    *,
    candidate_path_yx: np.ndarray,
    prev_path_yx: np.ndarray | None,
    next_path_yx: np.ndarray | None,
) -> tuple[float, np.ndarray]:
    if prev_path_yx is None and next_path_yx is None:
        return float("nan"), candidate_path_yx

    def _cost_for(path: np.ndarray) -> float:
        vals: list[float] = []
        if prev_path_yx is not None:
            vals.append(_adjacent_path_displacement_p95(prev_path_yx, path))
        if next_path_yx is not None:
            vals.append(_adjacent_path_displacement_p95(path, next_path_yx))
        if not vals:
            return float("nan")
        return float(np.mean(np.asarray(vals, dtype=np.float64)))

    fwd_cost = _cost_for(candidate_path_yx)
    rev_path = candidate_path_yx[::-1]
    rev_cost = _cost_for(rev_path)
    if np.isfinite(rev_cost) and (not np.isfinite(fwd_cost) or rev_cost < fwd_cost):
        return rev_cost, rev_path
    return fwd_cost, candidate_path_yx


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


def _curve_t_parameter(path_yx: np.ndarray) -> np.ndarray:
    """Slice-local normalized arc-length parameter along a u=0.5 path.

    This `t` is `t_all`: a per-slice full-path coordinate on the representative curve.
    """
    path = np.asarray(path_yx, dtype=np.float64)
    if path.ndim != 2 or path.shape[1] != 2 or path.shape[0] < 2:
        return np.zeros((0,), dtype=np.float64)
    seg = np.sqrt(np.sum(np.diff(path, axis=0) ** 2, axis=1))
    s = np.concatenate([np.zeros((1,), dtype=np.float64), np.cumsum(seg, dtype=np.float64)])
    total = float(s[-1])
    if not np.isfinite(total) or total <= 0.0:
        return np.linspace(0.0, 1.0, int(path.shape[0]), dtype=np.float64)
    return (s / total).astype(np.float64, copy=False)


def _curve_overlap_t_ranges(path_yx: np.ndarray, overlap_mask_yx: np.ndarray) -> list[tuple[float, float, int, int]]:
    """Return overlap intervals in slice-local `_curve_t_parameter` coordinates.

    Output `t_start/t_end` are in `t_all` (slice-local normalized full-path arc length).
    These are written to `*_neocortex_mesocortex_overlap_t_ranges.csv`.
    """
    path = np.asarray(path_yx, dtype=np.float64)
    overlap_mask = np.asarray(overlap_mask_yx, dtype=bool)
    if path.ndim != 2 or path.shape[1] != 2 or path.shape[0] < 2:
        return []
    if overlap_mask.ndim != 2:
        raise ValueError(f"Expected 2D overlap mask, got shape={overlap_mask.shape}.")

    t = _curve_t_parameter(path)
    if t.shape[0] != path.shape[0]:
        return []

    h, w = overlap_mask.shape
    yi = np.clip(np.rint(path[:, 0]).astype(np.int64, copy=False), 0, h - 1)
    xi = np.clip(np.rint(path[:, 1]).astype(np.int64, copy=False), 0, w - 1)
    inside = overlap_mask[yi, xi]
    if not np.any(inside):
        return []

    idx = np.flatnonzero(inside).astype(np.int64, copy=False)
    cuts = np.where(np.diff(idx) > 1)[0] + 1
    runs = np.split(idx, cuts)
    out: list[tuple[float, float, int, int]] = []
    for run in runs:
        if run.size == 0:
            continue
        i0 = int(run[0])
        i1 = int(run[-1])
        t0 = float(t[i0])
        t1 = float(t[i1])
        if not np.isfinite(t0) or not np.isfinite(t1):
            continue
        out.append((t0, t1, int(run.size), int(path.shape[0])))
    return out


def _write_overlap_t_ranges_csv(
    out_csv: Path,
    *,
    slice_label: str,
    rows: list[tuple[int, float, float, int, int, str]],
) -> None:
    with out_csv.open("w", encoding="utf-8") as f:
        f.write(f"{slice_label},t_start_all,t_end_all,n_overlap_points,n_path_points,path_source\n")
        for slice_idx, t0, t1, n_overlap, n_path, source in rows:
            f.write(
                f"{int(slice_idx)},{float(t0):.8f},{float(t1):.8f},{int(n_overlap)},{int(n_path)},{source}\n"
            )


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
if SAVE_INTERMEDIATE_NPY:
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
if SAVE_INTERMEDIATE_NPY:
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
if SAVE_INTERMEDIATE_NPY:
    np.save(OUTDIR / "midline_include_neo_meso_3d_ds.npy", midline_include_3d.astype(np.bool_))

mask_flat_overlay = _term_mask_from_annotation_yx(
    annotation_yx=ann_flat_mid,
    terms=OVERLAY_TERMS_NO_ALLOCORTEX,
    kind=TERM_KIND,  # type: ignore[arg-type]
    combine="any",
    invert=False,
    atlas=atlas,
)
overlay_neo_meso_no_allocortex_3d = mask_flat_overlay.reshape(annotation_3d.shape).astype(bool)
overlay_neo_meso_no_allocortex_3d = overlay_neo_meso_no_allocortex_3d[:, :, : cortex_3d.shape[2]]
if SAVE_INTERMEDIATE_NPY:
    np.save(
        OUTDIR / "overlay_neocortex_mesocortex_no_allocortex_3d_ds.npy",
        overlay_neo_meso_no_allocortex_3d.astype(np.bool_),
    )


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
if SAVE_INTERMEDIATE_NPY:
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
if SAVE_INTERMEDIATE_NPY:
    np.save(OUTDIR / "cortex_mask_fit_3d_ds.npy", cortex_fit_3d.astype(np.bool_))

pial_b0, inner_b1 = _boundary_partition_pial_inner(
    cortex_mask=cortex_fit_3d, brain_mask=brain_mask_3d, cortex_reference_mask=cortex_clean_3d
)
print(f"[edt] b0(pial)={int(np.count_nonzero(pial_b0))} b1(inner)={int(np.count_nonzero(inner_b1))}")
if SAVE_INTERMEDIATE_NPY:
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

bridge_masks_dirty = False
coronal_largest_cache: dict[int, np.ndarray | None] = {}
sagittal_largest_cache: dict[int, np.ndarray | None] = {}


def _largest_coronal_contour(local_i: int) -> np.ndarray | None:
    local_i = int(local_i)
    if local_i in coronal_largest_cache:
        return coronal_largest_cache[local_i]
    path = _extract_largest_u05_contour_in_slice(
        u_yx=u_crop[local_i, :, :],
        mask_yx=mask_crop[local_i, :, :],
        include_yx=include_crop[local_i, :, :],
    )
    coronal_largest_cache[local_i] = path
    return path


def _largest_sagittal_contour(local_k: int) -> np.ndarray | None:
    local_k = int(local_k)
    if local_k in sagittal_largest_cache:
        return sagittal_largest_cache[local_k]
    path = _extract_largest_u05_contour_in_slice(
        u_yx=u_crop[:, :, local_k],
        mask_yx=mask_crop[:, :, local_k],
        include_yx=include_crop[:, :, local_k],
    )
    sagittal_largest_cache[local_k] = path
    return path


def _invalidate_coronal_contour_cache(local_i: int) -> None:
    coronal_largest_cache.pop(int(local_i), None)


def _invalidate_sagittal_contour_cache(local_k: int) -> None:
    sagittal_largest_cache.pop(int(local_k), None)

for manual_slice_i in MANUAL_CONNECT_CORONAL_SLICE_IS:
    local_override_i = int(manual_slice_i) - int(z0)
    if 0 <= local_override_i < int(u_crop.shape[0]):
        largest_path_local = _largest_coronal_contour(local_override_i)
        if largest_path_local is not None and largest_path_local.shape[0] >= 2:
            largest_path_full = largest_path_local.copy()
            largest_path_full[:, 0] += float(y0)
            largest_path_full[:, 1] += float(x0)
        else:
            largest_path_full = None

        u_slice, y_bridge, x_bridge, merged_path, connect_diag = _connect_two_largest_u05_contours_in_slice(
            u_yx=u_crop[local_override_i, :, :],
            mask_yx=mask_crop[local_override_i, :, :],
            include_yx=include_crop[local_override_i, :, :],
        )
        connected_path_full: np.ndarray | None = None
        second_ratio = 0.0
        closest_dist = float("inf")
        if connect_diag is not None:
            len1, len2, closest_dist, _ = connect_diag
            second_ratio = float(len2 / len1) if len1 > 0.0 else 0.0
        connect_candidate_ok = (
            merged_path is not None
            and merged_path.shape[0] >= 2
            and second_ratio >= float(MANUAL_CONNECT_MIN_SECOND_CONTOUR_RATIO)
            and closest_dist <= float(MANUAL_CONNECT_MAX_CLOSEST_DISTANCE_VOX)
        )
        if connect_candidate_ok:
            connected_path_full = merged_path.copy()
            connected_path_full[:, 0] += float(y0)
            connected_path_full[:, 1] += float(x0)

        if int(manual_slice_i) not in MANUAL_ADAPTIVE_CONNECT_CORONAL_SLICE_IS:
            u_crop[local_override_i, :, :] = u_slice
            _invalidate_coronal_contour_cache(local_override_i)
            if y_bridge.size > 0:
                mask_crop[local_override_i, y_bridge, x_bridge] = True
                include_crop[local_override_i, y_bridge, x_bridge] = True
                bridge_masks_dirty = True
                for local_k in np.unique(x_bridge):
                    _invalidate_sagittal_contour_cache(int(local_k))
            if merged_path is not None and merged_path.shape[0] >= 2:
                merged_path_full = merged_path.copy()
                merged_path_full[:, 0] += float(y0)
                merged_path_full[:, 1] += float(x0)
                out_name = MANUAL_CONNECT_CORONAL_PATH_TEMPLATE.format(slice_i=int(manual_slice_i))
                np.save(OUTDIR / out_name, merged_path_full.astype(np.float32, copy=False))
            print(
                f"[manual] coronal slice {int(manual_slice_i)}: "
                f"connected two largest u=0.5 contours with {int(y_bridge.size)} bridge voxels"
            )
            continue

        prev_slice_i = int(manual_slice_i) - 1
        prev_saved_path = _load_saved_manual_path(
            OUTDIR / MANUAL_CONNECT_CORONAL_PATH_TEMPLATE.format(slice_i=int(prev_slice_i))
        )
        if prev_saved_path is None:
            prev_local_i = int(prev_slice_i) - int(z0)
            if 0 <= prev_local_i < int(u_crop.shape[0]):
                prev_local_path = _largest_coronal_contour(prev_local_i)
                if prev_local_path is not None and prev_local_path.shape[0] >= 2:
                    prev_saved_path = prev_local_path.copy()
                    prev_saved_path[:, 0] += float(y0)
                    prev_saved_path[:, 1] += float(x0)

        next_slice_i = int(manual_slice_i) + 1
        next_saved_path: np.ndarray | None = None
        next_local_i = int(next_slice_i) - int(z0)
        if 0 <= next_local_i < int(u_crop.shape[0]):
            next_local_path = _largest_coronal_contour(next_local_i)
            if next_local_path is not None and next_local_path.shape[0] >= 2:
                next_saved_path = next_local_path.copy()
                next_saved_path[:, 0] += float(y0)
                next_saved_path[:, 1] += float(x0)

        chosen_source = "none"
        chosen_cost = float("nan")
        chosen_full: np.ndarray | None = None
        chosen_from_connected = False

        candidates: list[tuple[str, np.ndarray, bool, float]] = []
        if largest_path_full is not None:
            cost, oriented = _orientation_cost_against_neighbors(
                candidate_path_yx=largest_path_full,
                prev_path_yx=prev_saved_path,
                next_path_yx=next_saved_path,
            )
            candidates.append(("largest", oriented, False, cost))
        if connected_path_full is not None:
            cost, oriented = _orientation_cost_against_neighbors(
                candidate_path_yx=connected_path_full,
                prev_path_yx=prev_saved_path,
                next_path_yx=next_saved_path,
            )
            candidates.append(("connected", oriented, True, cost))

        if candidates:
            finite = [c for c in candidates if np.isfinite(c[3])]
            if finite:
                finite.sort(key=lambda c: c[3])
                chosen_source, chosen_full, chosen_from_connected, chosen_cost = finite[0]
            else:
                # If no neighbor references exist, prefer connected only when it passed quality gates.
                for source, path_full, is_connected, cost in candidates:
                    if is_connected:
                        chosen_source, chosen_full, chosen_from_connected, chosen_cost = source, path_full, is_connected, cost
                        break
                if chosen_full is None:
                    chosen_source, chosen_full, chosen_from_connected, chosen_cost = candidates[0]

        if (
            chosen_from_connected
            and np.isfinite(chosen_cost)
            and chosen_cost > float(MANUAL_CONNECT_MAX_ADJACENT_P95_VOX)
            and largest_path_full is not None
        ):
            largest_cost, largest_oriented = _orientation_cost_against_neighbors(
                candidate_path_yx=largest_path_full,
                prev_path_yx=prev_saved_path,
                next_path_yx=next_saved_path,
            )
            if np.isfinite(largest_cost) and largest_cost < chosen_cost:
                chosen_source = "largest_fallback"
                chosen_full = largest_oriented
                chosen_from_connected = False
                chosen_cost = largest_cost

        if chosen_full is None:
            print(
                f"[manual] coronal slice {int(manual_slice_i)}: no valid candidate path "
                f"(second_ratio={second_ratio:.3f}, closest_dist={closest_dist:.3f}); skipping."
            )
            continue

        out_name = MANUAL_CONNECT_CORONAL_PATH_TEMPLATE.format(slice_i=int(manual_slice_i))
        np.save(OUTDIR / out_name, chosen_full.astype(np.float32, copy=False))

        if chosen_from_connected:
            u_crop[local_override_i, :, :] = u_slice
            _invalidate_coronal_contour_cache(local_override_i)
            if y_bridge.size > 0:
                mask_crop[local_override_i, y_bridge, x_bridge] = True
                include_crop[local_override_i, y_bridge, x_bridge] = True
                bridge_masks_dirty = True
                for local_k in np.unique(x_bridge):
                    _invalidate_sagittal_contour_cache(int(local_k))
            print(
                f"[manual] coronal slice {int(manual_slice_i)}: chose connected path "
                f"(ratio={second_ratio:.3f}, closest_dist={closest_dist:.3f}, p95_cost={chosen_cost:.3f}, "
                f"bridge_vox={int(y_bridge.size)})"
            )
        else:
            print(
                f"[manual] coronal slice {int(manual_slice_i)}: chose largest path "
                f"(ratio={second_ratio:.3f}, closest_dist={closest_dist:.3f}, p95_cost={chosen_cost:.3f}, "
                f"selection={chosen_source})"
            )
    else:
        print(
            f"[manual] coronal slice {int(manual_slice_i)} outside crop "
            f"[{z0}, {z0 + u_crop.shape[0] - 1}]; skipping manual contour connection."
        )

for manual_slice_i in MANUAL_KEEP_LARGEST_CORONAL_SLICE_IS:
    local_override_i = int(manual_slice_i) - int(z0)
    if 0 <= local_override_i < int(u_crop.shape[0]):
        largest_path = _largest_coronal_contour(local_override_i)
        if largest_path is None or largest_path.shape[0] < 2:
            print(
                f"[manual] coronal slice {int(manual_slice_i)}: "
                "no valid largest u=0.5 contour for manual override."
            )
            continue
        largest_path_full = largest_path.copy()
        largest_path_full[:, 0] += float(y0)
        largest_path_full[:, 1] += float(x0)
        out_name = MANUAL_CONNECT_CORONAL_PATH_TEMPLATE.format(slice_i=int(manual_slice_i))
        np.save(OUTDIR / out_name, largest_path_full.astype(np.float32, copy=False))
        print(
            f"[manual] coronal slice {int(manual_slice_i)}: "
            f"saved largest u=0.5 contour override with {int(largest_path.shape[0])} points"
        )
    else:
        print(
            f"[manual] coronal slice {int(manual_slice_i)} outside crop "
            f"[{z0}, {z0 + u_crop.shape[0] - 1}]; skipping largest-contour override."
        )

for target_slice_i, (prev_slice_i, next_slice_i) in MANUAL_BLEND_CORONAL_NEIGHBOR_SLICES.items():
    prev_path = OUTDIR / MANUAL_CONNECT_CORONAL_PATH_TEMPLATE.format(slice_i=int(prev_slice_i))
    next_path = OUTDIR / MANUAL_CONNECT_CORONAL_PATH_TEMPLATE.format(slice_i=int(next_slice_i))
    if not prev_path.exists() or not next_path.exists():
        print(
            f"[manual] coronal slice {int(target_slice_i)}: missing neighbor manual paths "
            f"({prev_path.name}, {next_path.name}); skipping blend override."
        )
        continue
    path_prev = np.load(prev_path).astype(np.float64, copy=False)
    path_next = np.load(next_path).astype(np.float64, copy=False)
    if path_prev.ndim != 2 or path_prev.shape[1] != 2 or path_prev.shape[0] < 2:
        print(f"[manual] coronal slice {int(target_slice_i)}: invalid prev path; skipping blend override.")
        continue
    if path_next.ndim != 2 or path_next.shape[1] != 2 or path_next.shape[0] < 2:
        print(f"[manual] coronal slice {int(target_slice_i)}: invalid next path; skipping blend override.")
        continue
    n_points = max(64, int((path_prev.shape[0] + path_next.shape[0]) // 2))
    prev_rs = _resample_polyline(path_prev, n_points=int(n_points))
    next_rs = _resample_polyline(path_next, n_points=int(n_points))
    fwd = float(np.linalg.norm(prev_rs[0] - next_rs[0]) + np.linalg.norm(prev_rs[-1] - next_rs[-1]))
    rev = float(np.linalg.norm(prev_rs[0] - next_rs[-1]) + np.linalg.norm(prev_rs[-1] - next_rs[0]))
    if rev < fwd:
        next_rs = next_rs[::-1]
    blended = 0.5 * (prev_rs + next_rs)
    out_name = MANUAL_CONNECT_CORONAL_PATH_TEMPLATE.format(slice_i=int(target_slice_i))
    np.save(OUTDIR / out_name, blended.astype(np.float32, copy=False))
    print(
        f"[manual] coronal slice {int(target_slice_i)}: "
        f"saved blended override from slices {int(prev_slice_i)} and {int(next_slice_i)} "
        f"with {int(blended.shape[0])} points"
    )

for target_slice_i in MANUAL_ALIGN_CORONAL_DIRECTION_WITH_PREV_SLICE_IS:
    prev_slice_i = int(target_slice_i) - 1
    prev_path_file = OUTDIR / MANUAL_CONNECT_CORONAL_PATH_TEMPLATE.format(slice_i=int(prev_slice_i))
    target_path_file = OUTDIR / MANUAL_CONNECT_CORONAL_PATH_TEMPLATE.format(slice_i=int(target_slice_i))
    for slice_i_for_path, path_file in ((int(prev_slice_i), prev_path_file), (int(target_slice_i), target_path_file)):
        if path_file.exists():
            continue
        local_i = int(slice_i_for_path) - int(z0)
        if not (0 <= local_i < int(u_crop.shape[0])):
            print(
                f"[manual] coronal slice {int(target_slice_i)}: needed slice {int(slice_i_for_path)} is outside crop "
                f"[{z0}, {z0 + u_crop.shape[0] - 1}]; skipping direction alignment."
            )
            break
        largest_path = _largest_coronal_contour(local_i)
        if largest_path is None or largest_path.shape[0] < 2:
            print(
                f"[manual] coronal slice {int(target_slice_i)}: could not build largest contour for "
                f"slice {int(slice_i_for_path)}; skipping direction alignment."
            )
            break
        largest_full = largest_path.copy()
        largest_full[:, 0] += float(y0)
        largest_full[:, 1] += float(x0)
        np.save(path_file, largest_full.astype(np.float32, copy=False))
        print(
            f"[manual] coronal slice {int(target_slice_i)}: "
            f"saved largest u=0.5 contour for slice {int(slice_i_for_path)} "
            f"to support direction alignment ({int(largest_path.shape[0])} points)"
        )
    else:
        prev_path = np.load(prev_path_file).astype(np.float64, copy=False)
        target_path = np.load(target_path_file).astype(np.float64, copy=False)
        if prev_path.ndim != 2 or prev_path.shape[1] != 2 or prev_path.shape[0] < 2:
            print(f"[manual] coronal slice {int(target_slice_i)}: invalid prev path; skipping direction alignment.")
            continue
        if target_path.ndim != 2 or target_path.shape[1] != 2 or target_path.shape[0] < 2:
            print(f"[manual] coronal slice {int(target_slice_i)}: invalid target path; skipping direction alignment.")
            continue
        n_points = max(64, min(int(prev_path.shape[0]), int(target_path.shape[0])))
        prev_rs = _resample_polyline(prev_path, n_points=int(n_points))
        target_rs = _resample_polyline(target_path, n_points=int(n_points))
        fwd = float(np.linalg.norm(prev_rs[0] - target_rs[0]) + np.linalg.norm(prev_rs[-1] - target_rs[-1]))
        rev = float(np.linalg.norm(prev_rs[0] - target_rs[-1]) + np.linalg.norm(prev_rs[-1] - target_rs[0]))
        if rev < fwd:
            np.save(target_path_file, target_path[::-1].astype(np.float32, copy=False))
            print(
                f"[manual] coronal slice {int(target_slice_i)}: reversed path direction to match "
                f"slice {int(prev_slice_i)} (forward={fwd:.3f}, reversed={rev:.3f})"
            )
        else:
            print(
                f"[manual] coronal slice {int(target_slice_i)}: kept path direction "
                f"(forward={fwd:.3f}, reversed={rev:.3f})"
            )
        continue

for manual_slice_k in MANUAL_CONNECT_SAGITTAL_SLICE_KS:
    local_override_k = int(manual_slice_k) - int(x0)
    if 0 <= local_override_k < int(u_crop.shape[2]):
        u_slice, y_bridge, x_bridge, merged_path, _ = _connect_two_largest_u05_contours_in_slice(
            u_yx=u_crop[:, :, local_override_k],
            mask_yx=mask_crop[:, :, local_override_k],
            include_yx=include_crop[:, :, local_override_k],
        )
        u_crop[:, :, local_override_k] = u_slice
        _invalidate_sagittal_contour_cache(local_override_k)
        if y_bridge.size > 0:
            mask_crop[y_bridge, x_bridge, local_override_k] = True
            include_crop[y_bridge, x_bridge, local_override_k] = True
            bridge_masks_dirty = True
            for local_i in np.unique(y_bridge):
                _invalidate_coronal_contour_cache(int(local_i))
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

for manual_slice_k in MANUAL_KEEP_LARGEST_SAGITTAL_SLICE_KS:
    local_override_k = int(manual_slice_k) - int(x0)
    if 0 <= local_override_k < int(u_crop.shape[2]):
        largest_path = _largest_sagittal_contour(local_override_k)
        if largest_path is None or largest_path.shape[0] < 2:
            print(
                f"[manual] sagittal slice {int(manual_slice_k)}: "
                "no valid largest u=0.5 contour for manual override."
            )
            continue
        largest_path_full = largest_path.copy()
        largest_path_full[:, 0] += float(z0)
        largest_path_full[:, 1] += float(y0)
        out_name = MANUAL_CONNECT_SAGITTAL_PATH_TEMPLATE.format(slice_k=int(manual_slice_k))
        np.save(OUTDIR / out_name, largest_path_full.astype(np.float32, copy=False))
        print(
            f"[manual] sagittal slice {int(manual_slice_k)}: "
            f"saved largest u=0.5 contour override with {int(largest_path.shape[0])} points"
        )
    else:
        print(
            f"[manual] sagittal slice {int(manual_slice_k)} outside crop "
            f"[{x0}, {x0 + u_crop.shape[2] - 1}]; skipping largest-contour override."
        )


def _sagittal_manual_or_largest_path_full(slice_k: int) -> np.ndarray | None:
    path_file = OUTDIR / MANUAL_CONNECT_SAGITTAL_PATH_TEMPLATE.format(slice_k=int(slice_k))
    saved = _load_saved_manual_path(path_file)
    if saved is not None:
        return saved
    local_k = int(slice_k) - int(x0)
    if not (0 <= local_k < int(u_crop.shape[2])):
        return None
    largest_path = _largest_sagittal_contour(local_k)
    if largest_path is None or largest_path.shape[0] < 2:
        return None
    path_full = largest_path.copy()
    path_full[:, 0] += float(z0)
    path_full[:, 1] += float(y0)
    return path_full


for target_slice_k, (prev_slice_k, next_slice_k, w_next) in MANUAL_BLEND_SAGITTAL_SLICE_K_WEIGHTS.items():
    if not (0.0 <= float(w_next) <= 1.0):
        raise ValueError(
            f"MANUAL_BLEND_SAGITTAL weight must be in [0,1], got {float(w_next):.3f} "
            f"for target slice {int(target_slice_k)}."
        )
    prev_path = _sagittal_manual_or_largest_path_full(int(prev_slice_k))
    next_path = _sagittal_manual_or_largest_path_full(int(next_slice_k))
    if prev_path is None or next_path is None:
        print(
            f"[manual] sagittal slice {int(target_slice_k)}: missing blend neighbors "
            f"({int(prev_slice_k)}, {int(next_slice_k)}); skipping blend override."
        )
        continue
    n_points = max(64, int((prev_path.shape[0] + next_path.shape[0]) // 2))
    prev_rs = _resample_polyline(prev_path, n_points=int(n_points))
    next_rs = _resample_polyline(next_path, n_points=int(n_points))
    fwd = float(np.linalg.norm(prev_rs[0] - next_rs[0]) + np.linalg.norm(prev_rs[-1] - next_rs[-1]))
    rev = float(np.linalg.norm(prev_rs[0] - next_rs[-1]) + np.linalg.norm(prev_rs[-1] - next_rs[0]))
    if rev < fwd:
        next_rs = next_rs[::-1]
    blended = (1.0 - float(w_next)) * prev_rs + float(w_next) * next_rs
    out_name = MANUAL_CONNECT_SAGITTAL_PATH_TEMPLATE.format(slice_k=int(target_slice_k))
    np.save(OUTDIR / out_name, blended.astype(np.float32, copy=False))
    print(
        f"[manual] sagittal slice {int(target_slice_k)}: "
        f"saved weighted blend override from slices {int(prev_slice_k)} and {int(next_slice_k)} "
        f"(w_next={float(w_next):.2f}, points={int(blended.shape[0])})"
    )

for target_slice_k, (prev_slice_k, next_slice_k, t_start, t_peak, t_end, w_peak) in MANUAL_SOFTEN_SAGITTAL_LOCAL_KINKS.items():
    if not (0.0 <= float(t_start) < float(t_peak) < float(t_end) <= 1.0):
        raise ValueError(
            "Expected MANUAL_SOFTEN_SAGITTAL_LOCAL_KINKS t-window to satisfy "
            f"0<=t_start<t_peak<t_end<=1, got {(float(t_start), float(t_peak), float(t_end))} "
            f"for target slice {int(target_slice_k)}."
        )
    if not (0.0 <= float(w_peak) <= 1.0):
        raise ValueError(
            f"Expected MANUAL_SOFTEN_SAGITTAL_LOCAL_KINKS w_peak in [0,1], got {float(w_peak):.3f} "
            f"for target slice {int(target_slice_k)}."
        )
    prev_path = _sagittal_manual_or_largest_path_full(int(prev_slice_k))
    target_path = _sagittal_manual_or_largest_path_full(int(target_slice_k))
    next_path = _sagittal_manual_or_largest_path_full(int(next_slice_k))
    if prev_path is None or target_path is None or next_path is None:
        print(
            f"[manual] sagittal slice {int(target_slice_k)}: missing local-soften neighbors "
            f"({int(prev_slice_k)}, {int(next_slice_k)}) or target; skipping local soften."
        )
        continue
    n_points = max(64, int(target_path.shape[0]))
    target_rs = _resample_polyline(target_path, n_points=int(n_points))
    prev_rs = _resample_polyline(prev_path, n_points=int(n_points))
    next_rs = _resample_polyline(next_path, n_points=int(n_points))

    # Orient neighbors to target for stable local blending.
    fwd_prev = float(np.linalg.norm(target_rs[0] - prev_rs[0]) + np.linalg.norm(target_rs[-1] - prev_rs[-1]))
    rev_prev = float(np.linalg.norm(target_rs[0] - prev_rs[-1]) + np.linalg.norm(target_rs[-1] - prev_rs[0]))
    if rev_prev < fwd_prev:
        prev_rs = prev_rs[::-1]
    fwd_next = float(np.linalg.norm(target_rs[0] - next_rs[0]) + np.linalg.norm(target_rs[-1] - next_rs[-1]))
    rev_next = float(np.linalg.norm(target_rs[0] - next_rs[-1]) + np.linalg.norm(target_rs[-1] - next_rs[0]))
    if rev_next < fwd_next:
        next_rs = next_rs[::-1]

    ref_rs = 0.5 * (prev_rs + next_rs)
    t_vals = np.linspace(0.0, 1.0, int(n_points), dtype=np.float64)
    w = np.zeros_like(t_vals)
    left = (t_vals >= float(t_start)) & (t_vals < float(t_peak))
    if np.any(left):
        w[left] = float(w_peak) * ((t_vals[left] - float(t_start)) / (float(t_peak) - float(t_start)))
    right = (t_vals >= float(t_peak)) & (t_vals <= float(t_end))
    if np.any(right):
        w[right] = float(w_peak) * (1.0 - ((t_vals[right] - float(t_peak)) / (float(t_end) - float(t_peak))))
    softened = (1.0 - w)[:, None] * target_rs + w[:, None] * ref_rs

    out_name = MANUAL_CONNECT_SAGITTAL_PATH_TEMPLATE.format(slice_k=int(target_slice_k))
    np.save(OUTDIR / out_name, softened.astype(np.float32, copy=False))
    print(
        f"[manual] sagittal slice {int(target_slice_k)}: "
        f"saved local-kink soften override using slices {int(prev_slice_k)}/{int(next_slice_k)} "
        f"(t=[{float(t_start):.2f},{float(t_peak):.2f},{float(t_end):.2f}], w_peak={float(w_peak):.2f})"
    )


if bridge_masks_dirty and SAVE_INTERMEDIATE_NPY:
    # Persist bridge-adjusted masks once after all manual edits to avoid repeated full-volume writes.
    np.save(OUTDIR / "cortex_mask_fit_3d_ds.npy", cortex_fit_3d.astype(np.bool_))
    np.save(OUTDIR / "midline_include_neo_meso_3d_ds.npy", midline_include_3d.astype(np.bool_))

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

coronal_overlap_rows: list[tuple[int, float, float, int, int, str]] = []
for slice_i in range(int(u.shape[0])):
    overlap_mask_yx = overlay_neo_meso_no_allocortex_3d[slice_i, :, :]
    if not np.any(overlap_mask_yx):
        continue
    path = _load_saved_manual_path(OUTDIR / MANUAL_CONNECT_CORONAL_PATH_TEMPLATE.format(slice_i=int(slice_i)))
    path_source = "manual"
    if path is None:
        path = _extract_largest_u05_contour_in_slice(
            u_yx=u[slice_i, :, :],
            mask_yx=cortex_fit_3d[slice_i, :, :],
            include_yx=midline_include_3d[slice_i, :, :],
        )
        path_source = "largest"
    if path is None:
        continue
    for t0, t1, n_overlap, n_path in _curve_overlap_t_ranges(path, overlap_mask_yx):
        coronal_overlap_rows.append((int(slice_i), t0, t1, int(n_overlap), int(n_path), path_source))

sagittal_overlap_rows: list[tuple[int, float, float, int, int, str]] = []
for slice_k in range(int(u.shape[2])):
    overlap_mask_yx = overlay_neo_meso_no_allocortex_3d[:, :, slice_k]
    if not np.any(overlap_mask_yx):
        continue
    path = _load_saved_manual_path(OUTDIR / MANUAL_CONNECT_SAGITTAL_PATH_TEMPLATE.format(slice_k=int(slice_k)))
    path_source = "manual"
    if path is None:
        path = _extract_largest_u05_contour_in_slice(
            u_yx=u[:, :, slice_k],
            mask_yx=cortex_fit_3d[:, :, slice_k],
            include_yx=midline_include_3d[:, :, slice_k],
        )
        path_source = "largest"
    if path is None:
        continue
    for t0, t1, n_overlap, n_path in _curve_overlap_t_ranges(path, overlap_mask_yx):
        sagittal_overlap_rows.append((int(slice_k), t0, t1, int(n_overlap), int(n_path), path_source))

_write_overlap_t_ranges_csv(
    OUTDIR / "coronal_neocortex_mesocortex_overlap_t_ranges.csv",
    slice_label="slice_i",
    rows=coronal_overlap_rows,
)
_write_overlap_t_ranges_csv(
    OUTDIR / "sagittal_neocortex_mesocortex_overlap_t_ranges.csv",
    slice_label="slice_k",
    rows=sagittal_overlap_rows,
)
print(
    "[overlap] wrote neocortex+mesocortex t-ranges: "
    f"coronal_rows={len(coronal_overlap_rows)} sagittal_rows={len(sagittal_overlap_rows)}"
)

mid_band_crop = mask_crop & include_crop & (np.abs(u_crop - 0.5) <= float(MIDSURF_EPS))
mid_band = np.zeros(cortex_fit_3d.shape, dtype=bool)
mid_band[crop] = _maybe_keep_largest_component(mid_band_crop)
print(f"[edt] mid_band voxels={int(np.count_nonzero(mid_band))}")
np.save(OUTDIR / "midsurface_halfway_midband_3d_ds.npy", mid_band.astype(np.bool_))

if WRITE_MESH:
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
else:
    print("[edt] skipping marching_cubes/PLY because WRITE_MESH=False")


if WRITE_PLOT:
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
else:
    print("[edt] skipping plot generation because WRITE_PLOT=False")

print(f"Wrote artifacts to {OUTDIR}")
