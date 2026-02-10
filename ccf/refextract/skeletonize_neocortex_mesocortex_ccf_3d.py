#%%
# # 3D cortex skeleton (per-coronal slice lines) (E15.5 DevCCF)
#
# This workflow builds a 3D cortex mask (term + descendants), skeletonizes it into a 2D sheet-like
# medial scaffold, then extracts a single polyline per coronal slice and visualizes them in 3D.
#
# Run top-to-bottom as a script. Each phase writes artifacts under `OUTDIR`.

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from scipy.interpolate import splprep, splev
from scipy.ndimage import distance_transform_edt, map_coordinates
from skimage.morphology import (
    ball,
    binary_closing,
    binary_erosion,
    remove_small_holes,
    remove_small_objects,
    skeletonize,
)

from fishtools.ccf.cli_export_mask_edit_pack import _term_mask_from_annotation_yx

# Fixed atlas (DevCCF E15.5 reference used elsewhere in `ccf/`).
ATLAS_NAME = "kim_dev_mouse_e15-5_lsfm_20um"


# === EDIT THESE ===

# Terms to include (each term includes all descendants).
TERMS: tuple[str, ...] = ("neocortex", "mesocortex", "allocortex")
TERM_KIND: Literal["auto", "id", "acronym", "name"] = "auto"

# The term mask spans both hemispheres and is often connected across the midline.
# Keep only the left half in x (axis 2) so the skeleton doesn't bridge hemispheres.
KEEP_LEFT_HEMISPHERE_ONLY = True

# 3D cleanup before skeletonization.
ERODE_RADIUS_VOX = 2
REMOVE_SMALL_OBJECTS_VOX = 2_000
FILL_SMALL_HOLES_VOX = 2_000
CLOSE_RADIUS_VOX = 1

# Plot downsampling for large skeletons.
PLOT_MAX_SKELETON_POINTS = 60_000

# Coronal slice processing (axis 0). We skeletonize each slice independently and extract a single line.
SLICE_STRIDE = 1
MIN_MASK_PX_PER_SLICE = 500
MIN_SKELETON_PX_PER_SLICE = 50
MAX_SLICES_TO_PLOT = 150

# Hybrid chart-source rules (avoid unstable per-slice extraction near discontinuities):
# - For coronal slices i < 177, derive the coronal line from sagittal-extracted points.
# - For sagittal slices k < 148 or k > 225, derive the sagittal line from coronal-extracted points.
CORONAL_USE_SAGITTAL_BELOW_I = 177
SAGITTAL_USE_CORONAL_BELOW_K = 148
SAGITTAL_USE_CORONAL_ABOVE_K = 225

# Spline parametrization of the extracted coronal line (in pixel space).
SPLINE_ORDER = 3
SPLINE_SMOOTHING = 25.0
SPLINE_N_SAMPLES = 120

OUTDIR = Path("ccf/out/refextract/skeletonize_neocortex_mesocortex_3d")
OUTDIR.mkdir(parents=True, exist_ok=True)

BRAINGLOBE_CONFIG_DIR = Path("ccf/out/atlases/.brainglobe_config")


# ## Phase 0: Load atlas annotation volume

if BRAINGLOBE_CONFIG_DIR.exists():
    os.environ["BRAINGLOBE_CONFIG_DIR"] = str(BRAINGLOBE_CONFIG_DIR.resolve())

from brainglobe_atlasapi import BrainGlobeAtlas  # noqa: E402

atlas = BrainGlobeAtlas(ATLAS_NAME)
annotation_3d = np.asarray(atlas.annotation)
print(f"ATLAS_NAME={ATLAS_NAME!r}")
print(f"annotation_3d shape={annotation_3d.shape}, dtype={annotation_3d.dtype}")
print(f"atlas.resolution={atlas.resolution!r} (um)")

np.save(OUTDIR / "annotation_3d.npy", annotation_3d)


def _resolution_ijk_um(res: object) -> tuple[float, float, float]:
    if isinstance(res, (int, float)):
        r = float(res)
        return (r, r, r)
    if isinstance(res, (tuple, list)) and len(res) == 3 and all(isinstance(v, (int, float)) for v in res):
        return (float(res[0]), float(res[1]), float(res[2]))
    raise ValueError(f"Unsupported atlas.resolution: {res!r}")


RES_I_UM, RES_J_UM, RES_K_UM = _resolution_ijk_um(atlas.resolution)


# ## Phase 1: Build 3D mask (term + descendants)

annotation_3d = np.load(OUTDIR / "annotation_3d.npy")

# Reuse shared term subtree logic by flattening spatial dims; restore to 3D afterward.
ann_flat = annotation_3d.reshape(annotation_3d.shape[0], -1)
mask_flat = _term_mask_from_annotation_yx(
    annotation_yx=ann_flat,
    terms=TERMS,
    kind=TERM_KIND,
    combine="any",
    invert=False,
    atlas=atlas,
)
mask_3d = mask_flat.reshape(annotation_3d.shape)
if KEEP_LEFT_HEMISPHERE_ONLY:
    mask_3d = mask_3d[:, :, : mask_3d.shape[2] // 2]

print(f"Mask voxels (raw)={int(mask_3d.sum())}")
np.save(OUTDIR / "mask_3d.npy", mask_3d.astype(np.bool_))


# ## Phase 2: 3D cleanup + skeletonize

mask_3d = np.load(OUTDIR / "mask_3d.npy").astype(bool)

mask_clean_3d = mask_3d.copy()
if ERODE_RADIUS_VOX > 0:
    mask_clean_3d = binary_erosion(mask_clean_3d, footprint=ball(int(ERODE_RADIUS_VOX)))
if REMOVE_SMALL_OBJECTS_VOX > 0:
    mask_clean_3d = remove_small_objects(mask_clean_3d, min_size=int(REMOVE_SMALL_OBJECTS_VOX))
if FILL_SMALL_HOLES_VOX > 0:
    mask_clean_3d = remove_small_holes(mask_clean_3d, area_threshold=int(FILL_SMALL_HOLES_VOX))
if CLOSE_RADIUS_VOX > 0:
    mask_clean_3d = binary_closing(mask_clean_3d, footprint=ball(int(CLOSE_RADIUS_VOX)))

skeleton_3d = skeletonize(mask_clean_3d)

# Keep only skeleton voxels that fall inside neocortex+mesocortex.
# (This masks out skeleton bits that extend into allocortex without changing any preprocessing.)
if "atlas" not in globals():
    from brainglobe_atlasapi import BrainGlobeAtlas  # noqa: E402

    atlas = BrainGlobeAtlas(ATLAS_NAME)

annotation_3d = np.load(OUTDIR / "annotation_3d.npy")
ann_flat = annotation_3d.reshape(annotation_3d.shape[0], -1)
neo_meso_flat = _term_mask_from_annotation_yx(
    annotation_yx=ann_flat,
    terms=("neocortex", "mesocortex"),
    kind=TERM_KIND,
    combine="any",
    invert=False,
    atlas=atlas,
)
neo_meso_3d = neo_meso_flat.reshape(annotation_3d.shape)[:, :, : mask_clean_3d.shape[2]]
np.save(OUTDIR / "neo_meso_3d.npy", neo_meso_3d.astype(np.bool_))

skeleton_3d = skeleton_3d & neo_meso_3d
print(f"Mask voxels (clean)={int(mask_clean_3d.sum())}")
print(f"Skeleton voxels={int(skeleton_3d.sum())}")

np.save(OUTDIR / "mask_clean_3d.npy", mask_clean_3d.astype(np.bool_))
np.save(OUTDIR / "skeleton_3d.npy", skeleton_3d.astype(np.bool_))


# ## Phase 3: Coronal slice skeletons → 3D polylines

mask_3d = np.load(OUTDIR / "mask_3d.npy").astype(bool)
mask_clean_3d = np.load(OUTDIR / "mask_clean_3d.npy").astype(bool)
neo_meso_path = OUTDIR / "neo_meso_3d.npy"
if neo_meso_path.exists():
    neo_meso_3d = np.load(neo_meso_path).astype(bool)
else:
    # Backfill if Phase 2 wasn't run in this session.
    annotation_3d = np.load(OUTDIR / "annotation_3d.npy")
    ann_flat = annotation_3d.reshape(annotation_3d.shape[0], -1)
    neo_meso_flat = _term_mask_from_annotation_yx(
        annotation_yx=ann_flat,
        terms=("neocortex", "mesocortex"),
        kind=TERM_KIND,
        combine="any",
        invert=False,
        atlas=atlas,
    )
    neo_meso_3d = neo_meso_flat.reshape(annotation_3d.shape)
    neo_meso_3d = neo_meso_3d[:, :, : mask_clean_3d.shape[2]]
    np.save(neo_meso_path, neo_meso_3d.astype(np.bool_))
neo_meso_3d = neo_meso_3d[:, :, : mask_clean_3d.shape[2]]


def _neighbors8(y: int, x: int) -> list[tuple[int, int]]:
    return [
        (y - 1, x - 1),
        (y - 1, x),
        (y - 1, x + 1),
        (y, x - 1),
        (y, x + 1),
        (y + 1, x - 1),
        (y + 1, x),
        (y + 1, x + 1),
    ]


def _longest_shortest_path_yx(skel: np.ndarray) -> np.ndarray:
    coords = np.column_stack(np.nonzero(skel))
    if coords.size == 0:
        raise ValueError("Skeleton is empty.")

    coord_to_i = {(int(y), int(x)): int(i) for i, (y, x) in enumerate(coords.tolist())}
    adj: list[list[int]] = [[] for _ in range(coords.shape[0])]
    h, w = skel.shape
    for i, (y, x) in enumerate(coords.tolist()):
        y_i = int(y)
        x_i = int(x)
        for ny, nx in _neighbors8(y_i, x_i):
            if 0 <= ny < h and 0 <= nx < w:
                j = coord_to_i.get((ny, nx))
                if j is not None:
                    adj[i].append(j)

    # Keep only the largest connected component (skeletonization can yield multiple islands).
    comp = np.full((coords.shape[0],), -1, dtype=np.int32)
    cid = 0
    best_cid = -1
    best_size = 0
    for start in range(coords.shape[0]):
        if comp[start] != -1:
            continue
        q: list[int] = [start]
        comp[start] = cid
        qi = 0
        size = 0
        while qi < len(q):
            u = q[qi]
            qi += 1
            size += 1
            for v in adj[u]:
                if comp[v] == -1:
                    comp[v] = cid
                    q.append(v)
        if size > best_size:
            best_size = size
            best_cid = cid
        cid += 1

    keep = np.where(comp == best_cid)[0]
    if keep.size != coords.shape[0]:
        reindex = {int(old): int(new) for new, old in enumerate(keep.tolist())}
        coords = coords[keep]
        adj2: list[list[int]] = [[] for _ in range(coords.shape[0])]
        for old_u in keep.tolist():
            new_u = reindex[int(old_u)]
            for old_v in adj[int(old_u)]:
                if old_v in reindex:
                    adj2[new_u].append(reindex[int(old_v)])
        adj = adj2

    deg = np.array([len(v) for v in adj], dtype=np.int32)
    endpoints = np.where(deg == 1)[0].tolist()

    def bfs(start: int) -> tuple[np.ndarray, np.ndarray, int]:
        parent = np.full((coords.shape[0],), -1, dtype=np.int32)
        dist = np.full((coords.shape[0],), -1, dtype=np.int32)
        q: list[int] = [start]
        dist[start] = 0
        qi = 0
        while qi < len(q):
            u = q[qi]
            qi += 1
            for v in adj[u]:
                if dist[v] != -1:
                    continue
                dist[v] = dist[u] + 1
                parent[v] = u
                q.append(v)
        far = int(np.argmax(dist))
        return parent, dist, far

    start = int(endpoints[0]) if endpoints else 0
    _, _, a = bfs(start)
    parent_b, _, b = bfs(a)

    path_idx: list[int] = []
    cur = int(b)
    while cur != -1:
        path_idx.append(cur)
        if cur == a:
            break
        cur = int(parent_b[cur])
    path_idx.reverse()
    return coords[np.asarray(path_idx, dtype=np.int64)]


def _resample_polyline_yx_spline(
    yx: np.ndarray, *, n_samples: int, smoothing: float, order: int
) -> tuple[np.ndarray, np.ndarray]:
    if yx.ndim != 2 or yx.shape[1] != 2:
        raise ValueError(f"Expected yx shape (N,2), got {yx.shape}")
    if yx.shape[0] < 2:
        raise ValueError("Need at least 2 points for spline parametrization.")

    # Remove consecutive duplicates (can break arc-length parametrization).
    diffs = np.diff(yx.astype(np.float64, copy=False), axis=0)
    keep = np.ones((yx.shape[0],), dtype=bool)
    keep[1:] = np.any(diffs != 0, axis=1)
    yx = yx[keep]
    if yx.shape[0] < 2:
        raise ValueError("All points were duplicates after filtering.")

    dy = np.diff(yx[:, 0].astype(np.float64, copy=False))
    dx = np.diff(yx[:, 1].astype(np.float64, copy=False))
    ds = np.sqrt(dy * dy + dx * dx)
    s = np.concatenate([[0.0], np.cumsum(ds)])
    total = float(s[-1])
    if not np.isfinite(total) or total <= 0:
        u = np.linspace(0.0, 1.0, yx.shape[0])
    else:
        u = s / total

    k = int(min(int(order), int(yx.shape[0] - 1), 3))
    tck, _u = splprep([yx[:, 0].astype(np.float64), yx[:, 1].astype(np.float64)], u=u, s=float(smoothing), k=k)
    u_new = np.linspace(0.0, 1.0, int(n_samples))
    y_new, x_new = splev(u_new, tck)
    yx_new = np.column_stack([np.asarray(y_new, dtype=np.float64), np.asarray(x_new, dtype=np.float64)])
    return yx_new, u_new.astype(np.float64)


def _unique_rows_preserve_order(points: np.ndarray) -> np.ndarray:
    if points.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape={points.shape}")
    if points.shape[0] == 0:
        return points
    packed = np.ascontiguousarray(points).view(np.dtype((np.void, points.dtype.itemsize * points.shape[1])))
    _, first_idx = np.unique(packed, return_index=True)
    return points[np.sort(first_idx)]


def _embed_slice_path_in_ijk(path_yx: np.ndarray, *, slice_idx: int, axis: Literal[0, 2]) -> np.ndarray:
    if axis == 0:
        return np.column_stack([np.full((path_yx.shape[0],), slice_idx), path_yx])
    if axis == 2:
        return np.column_stack([path_yx, np.full((path_yx.shape[0],), slice_idx)])
    raise ValueError(f"Unsupported axis={axis}. Expected 0 or 2.")


def _slice_plane_path_from_ijk(path_ijk: np.ndarray, *, axis: Literal[0, 2]) -> np.ndarray:
    if axis == 0:
        return path_ijk[:, 1:3]
    if axis == 2:
        return path_ijk[:, 0:2]
    raise ValueError(f"Unsupported axis={axis}. Expected 0 or 2.")


def _extract_axis_lines_raw(
    *,
    mask_3d: np.ndarray,
    axis: Literal[0, 2],
) -> tuple[list[np.ndarray], list[int]]:
    n_slices_axis = int(mask_3d.shape[int(axis)])
    lines_ijk_axis: list[np.ndarray] = []
    slice_indices_axis: list[int] = []

    for slice_idx in range(0, n_slices_axis, int(SLICE_STRIDE)):
        if axis == 0:
            mask_2d = mask_3d[slice_idx, :, :]
        else:
            mask_2d = mask_3d[:, :, slice_idx]
        if int(mask_2d.sum()) < int(MIN_MASK_PX_PER_SLICE):
            continue
        skel_2d = skeletonize(mask_2d)
        if int(skel_2d.sum()) < int(MIN_SKELETON_PX_PER_SLICE):
            continue

        path_yx = _longest_shortest_path_yx(skel_2d)
        path_ijk = _embed_slice_path_in_ijk(path_yx.astype(np.int32), slice_idx=int(slice_idx), axis=axis).astype(
            np.int32, copy=False
        )
        if path_ijk.shape[0] < 2:
            continue

        lines_ijk_axis.append(path_ijk)
        slice_indices_axis.append(int(slice_idx))

    return lines_ijk_axis, slice_indices_axis


def _cross_section_path_from_other_axis(
    *,
    slice_idx: int,
    axis: Literal[0, 2],
    other_lines_ijk: list[np.ndarray],
    mask_shape_ijk: tuple[int, int, int],
) -> np.ndarray | None:
    # axis=0: build coronal (j,k) path from sagittal (i,j,k) lines at i==slice_idx
    # axis=2: build sagittal (i,j) path from coronal (i,j,k) lines at k==slice_idx
    if axis == 0:
        h, w = int(mask_shape_ijk[1]), int(mask_shape_ijk[2])
        pts: list[np.ndarray] = []
        for p in other_lines_ijk:
            sel = p[:, 0] == int(slice_idx)
            if np.any(sel):
                pts.append(p[sel][:, 1:3])
        if not pts:
            return None
        yx = np.vstack(pts).astype(np.int32, copy=False)
        yx = _unique_rows_preserve_order(yx)
        yx = yx[(yx[:, 0] >= 0) & (yx[:, 0] < h) & (yx[:, 1] >= 0) & (yx[:, 1] < w)]
        if yx.shape[0] < 2:
            return None
        skel_2d = np.zeros((h, w), dtype=bool)
        skel_2d[yx[:, 0], yx[:, 1]] = True
        path_yx = _longest_shortest_path_yx(skel_2d)
        return _embed_slice_path_in_ijk(path_yx.astype(np.int32), slice_idx=int(slice_idx), axis=axis).astype(
            np.int32, copy=False
        )

    h, w = int(mask_shape_ijk[0]), int(mask_shape_ijk[1])
    pts: list[np.ndarray] = []
    for p in other_lines_ijk:
        sel = p[:, 2] == int(slice_idx)
        if np.any(sel):
            pts.append(p[sel][:, 0:2])
    if not pts:
        return None
    yx = np.vstack(pts).astype(np.int32, copy=False)
    yx = _unique_rows_preserve_order(yx)
    yx = yx[(yx[:, 0] >= 0) & (yx[:, 0] < h) & (yx[:, 1] >= 0) & (yx[:, 1] < w)]
    if yx.shape[0] < 2:
        return None
    skel_2d = np.zeros((h, w), dtype=bool)
    skel_2d[yx[:, 0], yx[:, 1]] = True
    path_yx = _longest_shortest_path_yx(skel_2d)
    return _embed_slice_path_in_ijk(path_yx.astype(np.int32), slice_idx=int(slice_idx), axis=axis).astype(
        np.int32, copy=False
    )


def _build_axis_lines_with_spline_hybrid(
    *,
    axis: Literal[0, 2],
    lines_ijk_raw: list[np.ndarray],
    slice_indices_raw: list[int],
    other_lines_ijk_raw: list[np.ndarray],
    include_mask_3d: np.ndarray,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[int]]:
    lines_ijk_axis: list[np.ndarray] = []
    lines_spline_ijk_axis: list[np.ndarray] = []
    lines_spline_u_axis: list[np.ndarray] = []
    slice_indices_axis: list[int] = []

    raw_by_slice = {int(s): p for s, p in zip(slice_indices_raw, lines_ijk_raw, strict=True)}
    mask_shape_ijk = (int(include_mask_3d.shape[0]), int(include_mask_3d.shape[1]), int(include_mask_3d.shape[2]))

    for slice_idx in sorted(raw_by_slice.keys()):
        path_ijk = raw_by_slice[int(slice_idx)]

        if axis == 0 and int(slice_idx) < int(CORONAL_USE_SAGITTAL_BELOW_I):
            alt = _cross_section_path_from_other_axis(
                slice_idx=int(slice_idx),
                axis=axis,
                other_lines_ijk=other_lines_ijk_raw,
                mask_shape_ijk=mask_shape_ijk,
            )
            if alt is not None and alt.shape[0] >= 2:
                path_ijk = alt
        elif axis == 2 and (
            int(slice_idx) < int(SAGITTAL_USE_CORONAL_BELOW_K) or int(slice_idx) > int(SAGITTAL_USE_CORONAL_ABOVE_K)
        ):
            alt = _cross_section_path_from_other_axis(
                slice_idx=int(slice_idx),
                axis=axis,
                other_lines_ijk=other_lines_ijk_raw,
                mask_shape_ijk=mask_shape_ijk,
            )
            if alt is not None and alt.shape[0] >= 2:
                path_ijk = alt

        # Allocortex exclusion (neo+meso include mask) happens after hybrid sourcing.
        keep = include_mask_3d[path_ijk[:, 0], path_ijk[:, 1], path_ijk[:, 2]]
        path_ijk = path_ijk[keep]
        if path_ijk.shape[0] < 2:
            continue

        yx_spline, u_spline = _resample_polyline_yx_spline(
            _slice_plane_path_from_ijk(path_ijk, axis=axis).astype(np.float64, copy=False),
            n_samples=int(SPLINE_N_SAMPLES),
            smoothing=float(SPLINE_SMOOTHING),
            order=int(SPLINE_ORDER),
        )

        lines_ijk_axis.append(path_ijk.astype(np.int32, copy=False))
        lines_spline_ijk_axis.append(_embed_slice_path_in_ijk(yx_spline, slice_idx=int(slice_idx), axis=axis))
        lines_spline_u_axis.append(u_spline)
        slice_indices_axis.append(int(slice_idx))

    return lines_ijk_axis, lines_spline_ijk_axis, lines_spline_u_axis, slice_indices_axis


def _extract_axis_lines_with_spline(
    *,
    mask_3d: np.ndarray,
    include_mask_3d: np.ndarray,
    axis: Literal[0, 2],
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[int]]:
    n_slices_axis = int(mask_3d.shape[int(axis)])
    lines_ijk_axis: list[np.ndarray] = []
    lines_spline_ijk_axis: list[np.ndarray] = []
    lines_spline_u_axis: list[np.ndarray] = []
    slice_indices_axis: list[int] = []

    for slice_idx in range(0, n_slices_axis, int(SLICE_STRIDE)):
        if axis == 0:
            mask_2d = mask_3d[slice_idx, :, :]
        else:
            mask_2d = mask_3d[:, :, slice_idx]
        if int(mask_2d.sum()) < int(MIN_MASK_PX_PER_SLICE):
            continue
        skel_2d = skeletonize(mask_2d)
        if int(skel_2d.sum()) < int(MIN_SKELETON_PX_PER_SLICE):
            continue

        path_yx = _longest_shortest_path_yx(skel_2d)
        path_ijk = _embed_slice_path_in_ijk(path_yx.astype(np.int32), slice_idx=int(slice_idx), axis=axis).astype(
            np.int32, copy=False
        )
        keep = include_mask_3d[path_ijk[:, 0], path_ijk[:, 1], path_ijk[:, 2]]
        path_ijk = path_ijk[keep]
        if path_ijk.shape[0] < 2:
            continue

        lines_ijk_axis.append(path_ijk)
        yx_spline, u_spline = _resample_polyline_yx_spline(
            _slice_plane_path_from_ijk(path_ijk, axis=axis).astype(np.float64, copy=False),
            n_samples=int(SPLINE_N_SAMPLES),
            smoothing=float(SPLINE_SMOOTHING),
            order=int(SPLINE_ORDER),
        )
        lines_spline_ijk_axis.append(_embed_slice_path_in_ijk(yx_spline, slice_idx=int(slice_idx), axis=axis))
        lines_spline_u_axis.append(u_spline)
        slice_indices_axis.append(int(slice_idx))

    return lines_ijk_axis, lines_spline_ijk_axis, lines_spline_u_axis, slice_indices_axis


def _save_lines_um_csv(
    *,
    lines_ijk: list[np.ndarray],
    slice_indices: list[int],
    out_csv: Path,
    slice_col: str,
) -> None:
    rows: list[np.ndarray] = []
    for slice_idx, path in zip(slice_indices, lines_ijk, strict=True):
        n = int(path.shape[0])
        idx = np.arange(n, dtype=np.int32)
        i_um = path[:, 0].astype(np.float64) * RES_I_UM
        j_um = path[:, 1].astype(np.float64) * RES_J_UM
        k_um = path[:, 2].astype(np.float64) * RES_K_UM
        rows.append(np.column_stack([np.full((n,), int(slice_idx), dtype=np.int32), idx, i_um, j_um, k_um]))
    table = np.vstack(rows)
    np.savetxt(
        out_csv,
        table,
        delimiter=",",
        header=f"{slice_col},point_idx,i_um,j_um,k_um",
        comments="",
    )


def _save_spline_lines_um_csv(
    *,
    lines_spline_ijk: list[np.ndarray],
    lines_spline_u: list[np.ndarray],
    slice_indices: list[int],
    out_csv: Path,
    slice_col: str,
) -> None:
    rows: list[np.ndarray] = []
    for slice_idx, path_spline, u in zip(slice_indices, lines_spline_ijk, lines_spline_u, strict=True):
        n = int(path_spline.shape[0])
        i_um = path_spline[:, 0].astype(np.float64) * RES_I_UM
        j_um = path_spline[:, 1].astype(np.float64) * RES_J_UM
        k_um = path_spline[:, 2].astype(np.float64) * RES_K_UM
        rows.append(np.column_stack([np.full((n,), int(slice_idx), dtype=np.int32), u.astype(np.float64), i_um, j_um, k_um]))
    table = np.vstack(rows)
    np.savetxt(
        out_csv,
        table,
        delimiter=",",
        header=f"{slice_col},u,i_um,j_um,k_um",
        comments="",
    )


def _extract_save_axis_lines(
    *,
    mask_3d: np.ndarray,
    include_mask_3d: np.ndarray,
    axis: Literal[0, 2],
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[int]]:
    if axis == 0:
        axis_name = "coronal"
        slice_col = "slice_i"
    elif axis == 2:
        axis_name = "sagittal"
        slice_col = "slice_k"
    else:
        raise ValueError(f"Unsupported axis={axis}. Expected 0 or 2.")

    lines_ijk_axis, lines_spline_ijk_axis, lines_spline_u_axis, slice_indices_axis = _extract_axis_lines_with_spline(
        mask_3d=mask_3d,
        include_mask_3d=include_mask_3d,
        axis=axis,
    )
    if not lines_ijk_axis:
        raise ValueError(f"No {axis_name} slice skeleton lines were extracted; lower thresholds or check terms.")

    _save_lines_um_csv(
        lines_ijk=lines_ijk_axis,
        slice_indices=slice_indices_axis,
        out_csv=OUTDIR / f"{axis_name}_slice_lines_um.csv",
        slice_col=slice_col,
    )
    _save_spline_lines_um_csv(
        lines_spline_ijk=lines_spline_ijk_axis,
        lines_spline_u=lines_spline_u_axis,
        slice_indices=slice_indices_axis,
        out_csv=OUTDIR / f"{axis_name}_slice_lines_spline_um.csv",
        slice_col=slice_col,
    )
    return lines_ijk_axis, lines_spline_ijk_axis, lines_spline_u_axis, slice_indices_axis


coronal_lines_ijk_raw, coronal_slice_indices_raw = _extract_axis_lines_raw(mask_3d=mask_clean_3d, axis=0)
sagittal_lines_ijk_raw, sagittal_slice_indices_raw = _extract_axis_lines_raw(mask_3d=mask_clean_3d, axis=2)
if not coronal_lines_ijk_raw:
    raise ValueError("No coronal slice skeleton lines were extracted; lower thresholds or check terms.")
if not sagittal_lines_ijk_raw:
    raise ValueError("No sagittal slice skeleton lines were extracted; lower thresholds or check terms.")

lines_ijk, lines_spline_ijk, lines_spline_u, slice_indices = _build_axis_lines_with_spline_hybrid(
    axis=0,
    lines_ijk_raw=coronal_lines_ijk_raw,
    slice_indices_raw=coronal_slice_indices_raw,
    other_lines_ijk_raw=sagittal_lines_ijk_raw,
    include_mask_3d=neo_meso_3d,
)
if not lines_ijk:
    raise ValueError("No coronal slice skeleton lines remained after hybrid sourcing + include-mask filtering.")

_save_lines_um_csv(
    lines_ijk=lines_ijk,
    slice_indices=slice_indices,
    out_csv=OUTDIR / "coronal_slice_lines_um.csv",
    slice_col="slice_i",
)
_save_spline_lines_um_csv(
    lines_spline_ijk=lines_spline_ijk,
    lines_spline_u=lines_spline_u,
    slice_indices=slice_indices,
    out_csv=OUTDIR / "coronal_slice_lines_spline_um.csv",
    slice_col="slice_i",
)


# ## Phase 3h: Thickness along coronal spline (orthogonal to curve)
#
# For each slice, for each spline point, cast rays along the local normal direction to find the two
# mask boundaries and measure thickness (ventricular↔pial). We keep `v=0` as ventricular convention.

def _signed_distance_field(mask: np.ndarray) -> np.ndarray:
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape={mask.shape}")
    mask = mask.astype(bool, copy=False)
    din = distance_transform_edt(mask)
    dout = distance_transform_edt(~mask)
    return din - dout


def _sample_bilinear(img: np.ndarray, yx: np.ndarray) -> np.ndarray:
    if yx.ndim != 2 or yx.shape[1] != 2:
        raise ValueError(f"Expected yx shape (N,2), got {yx.shape}")
    return map_coordinates(img, [yx[:, 0], yx[:, 1]], order=1, mode="nearest")


def _sample_mask_linear(mask: np.ndarray, yx: np.ndarray) -> np.ndarray:
    if yx.ndim != 2 or yx.shape[1] != 2:
        raise ValueError(f"Expected yx shape (N,2), got {yx.shape}")
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape={mask.shape}")
    return map_coordinates(mask.astype(np.float64, copy=False), [yx[:, 0], yx[:, 1]], order=1, mode="constant", cval=0.0)


def _march_to_boundary_along_normal(
    mask: np.ndarray,
    *,
    p_yx: np.ndarray,
    n_yx: np.ndarray,
    direction: float,
    max_t: float,
    step: float,
    bisect_iters: int = 14,
) -> float:
    if p_yx.shape != (2,) or n_yx.shape != (2,):
        raise ValueError("p_yx and n_yx must be shape (2,).")
    if direction not in (-1.0, 1.0):
        raise ValueError("direction must be -1.0 or +1.0.")
    if max_t <= 0 or step <= 0:
        raise ValueError("max_t and step must be > 0.")
    if bisect_iters < 1:
        raise ValueError("bisect_iters must be >= 1.")
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape={mask.shape}")

    h, w = mask.shape
    p0 = p_yx.astype(np.float64, copy=False)
    n = n_yx.astype(np.float64, copy=False)

    # Fast inside check with bounds: outside the image is outside the mask.
    def inside_at(t: float) -> bool:
        pt = p0 + (direction * float(t)) * n
        y = float(pt[0])
        x = float(pt[1])
        if y < 0.0 or x < 0.0 or y > float(h - 1) or x > float(w - 1):
            return False
        yi = int(np.rint(y))
        xi = int(np.rint(x))
        yi = int(np.clip(yi, 0, h - 1))
        xi = int(np.clip(xi, 0, w - 1))
        return bool(mask[yi, xi])

    if not inside_at(0.0):
        return float("nan")

    last_inside_t = 0.0
    for t in np.arange(step, max_t + step, step, dtype=np.float64):
        if inside_at(float(t)):
            last_inside_t = float(t)
            continue

        lo = float(last_inside_t)
        hi = float(t)
        # Refine boundary using linear sampling (treat off-image as 0).
        for _ in range(int(bisect_iters)):
            mid = 0.5 * (lo + hi)
            pt = (p0 + (direction * mid) * n)[None, :]
            v = float(_sample_mask_linear(mask, pt)[0])
            if v >= 0.5:
                lo = mid
            else:
                hi = mid

        return float(direction * (0.5 * (lo + hi)))

    return float("nan")


def _snap_points_to_mask(yx: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if yx.ndim != 2 or yx.shape[1] != 2:
        raise ValueError(f"Expected yx shape (N,2), got {yx.shape}")
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape={mask.shape}")
    if not np.any(mask):
        raise ValueError("Mask is empty; cannot snap points to mask.")

    h, w = mask.shape
    yi = np.rint(yx[:, 0]).astype(np.int32)
    xi = np.rint(yx[:, 1]).astype(np.int32)
    yi = np.clip(yi, 0, h - 1)
    xi = np.clip(xi, 0, w - 1)
    outside = ~mask[yi, xi]
    if not np.any(outside):
        return yx

    _, nearest_idx = distance_transform_edt(~mask, return_distances=True, return_indices=True)
    yx_out = yx.copy()
    oy = yi[outside]
    ox = xi[outside]
    yx_out[outside, 0] = nearest_idx[0, oy, ox].astype(np.float64)
    yx_out[outside, 1] = nearest_idx[1, oy, ox].astype(np.float64)
    return yx_out


def _push_points_inside_mask(
    yx: np.ndarray, signed: np.ndarray, *, radius_px: int = 2, min_signed: float = 0.5
) -> np.ndarray:
    if yx.ndim != 2 or yx.shape[1] != 2:
        raise ValueError(f"Expected yx shape (N,2), got {yx.shape}")
    if signed.ndim != 2:
        raise ValueError(f"Expected signed 2D, got shape={signed.shape}")

    h, w = signed.shape
    yx_out = yx.copy()
    y_idx = np.clip(np.rint(yx_out[:, 0]).astype(np.int32), 0, h - 1)
    x_idx = np.clip(np.rint(yx_out[:, 1]).astype(np.int32), 0, w - 1)
    needs_push = signed[y_idx, x_idx] <= float(min_signed)
    if not np.any(needs_push):
        return yx_out

    r = int(max(radius_px, 1))
    for idx in np.where(needs_push)[0].tolist():
        y0 = int(y_idx[idx])
        x0 = int(x_idx[idx])
        y1 = max(0, y0 - r)
        y2 = min(h, y0 + r + 1)
        x1 = max(0, x0 - r)
        x2 = min(w, x0 + r + 1)
        patch = signed[y1:y2, x1:x2]
        flat_best = int(np.argmax(patch))
        best_val = float(patch.ravel()[flat_best])
        if best_val > 0.0:
            py, px = np.unravel_index(flat_best, patch.shape)
            yx_out[idx, 0] = float(y1 + int(py))
            yx_out[idx, 1] = float(x1 + int(px))

    return yx_out


def _first_zero_crossing_along_normal(
    signed: np.ndarray,
    *,
    p_yx: np.ndarray,
    n_yx: np.ndarray,
    direction: float,
    max_t: float,
    step: float,
) -> float:
    if p_yx.shape != (2,) or n_yx.shape != (2,):
        raise ValueError("p_yx and n_yx must be shape (2,).")
    if direction not in (-1.0, 1.0):
        raise ValueError("direction must be -1.0 or +1.0.")
    if max_t <= 0 or step <= 0:
        raise ValueError("max_t and step must be > 0.")

    p0 = p_yx.astype(np.float64, copy=False)
    n = n_yx.astype(np.float64, copy=False)

    v0 = float(_sample_bilinear(signed, p0[None, :])[0])
    if not np.isfinite(v0):
        return float("nan")

    prev_t = 0.0
    prev_v = v0
    inside = bool(prev_v > 0.0)
    for t in np.arange(step, max_t + step, step, dtype=np.float64):
        pt = p0 + (direction * t) * n
        vt = float(_sample_bilinear(signed, pt[None, :])[0])
        if not np.isfinite(vt):
            return float("nan")

        if inside:
            if vt <= 0.0:
                # Linear interpolation in t for the inside->outside 0-crossing.
                denom = prev_v - vt
                if denom <= 0:
                    return float(direction * t)
                frac = prev_v / denom
                t0 = prev_t + frac * (t - prev_t)
                return float(direction * t0)
        elif vt > 0.0:
            # Boundary/outside start: first enter mask, then look for the exit crossing.
            inside = True

        prev_t = float(t)
        prev_v = vt

    return float("nan")


def measure_thickness_along_coronal_spline(
    *,
    slice_i: int,
    mask_yx: np.ndarray,
    spline_yx: np.ndarray,
    u: np.ndarray,
    force_vent_minus_pia_positive: bool | None = None,
    max_t: float = 250.0,
    step: float = 0.5,
) -> dict[str, np.ndarray]:
    if mask_yx.ndim != 2:
        raise ValueError(f"Expected mask_yx 2D, got shape={mask_yx.shape}")
    if spline_yx.ndim != 2 or spline_yx.shape[1] != 2:
        raise ValueError(f"Expected spline_yx shape (N,2), got {spline_yx.shape}")
    if u.ndim != 1 or u.shape[0] != spline_yx.shape[0]:
        raise ValueError(f"Expected u shape (N,), got {u.shape} for spline length {spline_yx.shape[0]}")

    spline_yx = _snap_points_to_mask(spline_yx.astype(np.float64, copy=False), mask_yx)
    y = spline_yx[:, 0].astype(np.float64, copy=False)
    x = spline_yx[:, 1].astype(np.float64, copy=False)
    du = np.gradient(u.astype(np.float64, copy=False))
    dy_du = np.gradient(y) / du
    dx_du = np.gradient(x) / du

    # Normal in (y, x).
    ny = -dx_du
    nx = dy_du
    n_norm = np.sqrt(ny * ny + nx * nx)
    ok = n_norm > 0
    ny = np.where(ok, ny / n_norm, 0.0)
    nx = np.where(ok, nx / n_norm, 0.0)
    for idx in range(1, spline_yx.shape[0]):
        if not np.isfinite(ny[idx - 1]) or not np.isfinite(nx[idx - 1]):
            continue
        if not np.isfinite(ny[idx]) or not np.isfinite(nx[idx]):
            continue
        if (ny[idx - 1] * ny[idx] + nx[idx - 1] * nx[idx]) < 0.0:
            ny[idx] = -ny[idx]
            nx[idx] = -nx[idx]

    t_neg = np.full((spline_yx.shape[0],), np.nan, dtype=np.float64)
    t_pos = np.full((spline_yx.shape[0],), np.nan, dtype=np.float64)

    for idx in range(spline_yx.shape[0]):
        p = np.array([y[idx], x[idx]], dtype=np.float64)
        n = np.array([ny[idx], nx[idx]], dtype=np.float64)
        if not np.isfinite(n).all() or float(np.linalg.norm(n)) <= 0:
            continue
        t_pos[idx] = _march_to_boundary_along_normal(
            mask_yx, p_yx=p, n_yx=n, direction=1.0, max_t=float(max_t), step=float(step)
        )
        t_neg[idx] = _march_to_boundary_along_normal(
            mask_yx, p_yx=p, n_yx=n, direction=-1.0, max_t=float(max_t), step=float(step)
        )

    # Decide a stable vent/pia assignment for initialization. Our convention assumes the ventricular side
    # is the more-medial boundary, which in this cropped hemisphere corresponds to larger x in the 2D plane.
    # Using a slice-level median avoids arbitrary flips when the first sample is nearly vertical (b0_x≈b1_x).
    midline_is_high_x = True
    init_prefer_b0_as_vent: bool | None = None
    valid = np.isfinite(t_neg) & np.isfinite(t_pos) & (t_pos > t_neg) & np.isfinite(nx)
    if np.any(valid):
        b0_x = x[valid] + t_neg[valid] * nx[valid]
        b1_x = x[valid] + t_pos[valid] * nx[valid]
        dx = b0_x - b1_x
        med_dx = float(np.nanmedian(dx))
        if np.isfinite(med_dx) and abs(med_dx) >= 0.25:
            init_prefer_b0_as_vent = bool(med_dx >= 0.0) if midline_is_high_x else bool(med_dx < 0.0)

    p_vent = np.full((spline_yx.shape[0], 2), np.nan, dtype=np.float64)
    p_pia = np.full((spline_yx.shape[0], 2), np.nan, dtype=np.float64)
    thickness_px = np.full((spline_yx.shape[0],), np.nan, dtype=np.float64)
    v_skel = np.full((spline_yx.shape[0],), np.nan, dtype=np.float64)
    prev_vent: np.ndarray | None = None
    prev_pia: np.ndarray | None = None

    for idx in range(spline_yx.shape[0]):
        if not np.isfinite(t_neg[idx]) or not np.isfinite(t_pos[idx]):
            continue
        if float(t_pos[idx]) <= float(t_neg[idx]):
            continue

        p = np.array([y[idx], x[idx]], dtype=np.float64)
        n = np.array([ny[idx], nx[idx]], dtype=np.float64)
        b0 = p + t_neg[idx] * n
        b1 = p + t_pos[idx] * n

        if prev_vent is None or prev_pia is None:
            # Initialize by ventricular convention: closer to midline -> larger x (k in coronal view).
            prefer_b0 = init_prefer_b0_as_vent
            if prefer_b0 is None:
                prefer_b0 = bool(float(b0[1]) >= float(b1[1])) if midline_is_high_x else bool(float(b0[1]) < float(b1[1]))
            if prefer_b0:
                vent, pia = b0, b1
                t_vent, t_pia = float(t_neg[idx]), float(t_pos[idx])
            else:
                vent, pia = b1, b0
                t_vent, t_pia = float(t_pos[idx]), float(t_neg[idx])
        else:
            # Preserve branch continuity to avoid sudden vent/pia swaps in complex geometry.
            c_keep = float(np.sum((b0 - prev_vent) ** 2) + np.sum((b1 - prev_pia) ** 2))
            c_swap = float(np.sum((b1 - prev_vent) ** 2) + np.sum((b0 - prev_pia) ** 2))
            if c_keep <= c_swap:
                vent, pia = b0, b1
                t_vent, t_pia = float(t_neg[idx]), float(t_pos[idx])
            else:
                vent, pia = b1, b0
                t_vent, t_pia = float(t_pos[idx]), float(t_neg[idx])

        p_vent[idx] = vent
        p_pia[idx] = pia
        thickness_px[idx] = float(abs(t_pia - t_vent))
        v_skel[idx] = float((0.0 - t_vent) / (t_pia - t_vent))
        prev_vent = vent
        prev_pia = pia

    # Slice-level sanity: if the final assignment contradicts our ventricular side convention, flip.
    vent_x = p_vent[:, 1]
    pia_x = p_pia[:, 1]
    finite = np.isfinite(vent_x) & np.isfinite(pia_x)
    finite_n = int(np.sum(finite))
    if force_vent_minus_pia_positive is not None:
        # Cross-slice continuity mode: enforce a consistent vent/pia sign (vent_x - pia_x).
        # This intentionally relaxes the per-slice thresholds that were meant to avoid flipping on noise.
        if finite_n >= 5:
            med_dx = float(np.nanmedian(vent_x[finite] - pia_x[finite]))
            if np.isfinite(med_dx):
                wants_positive = bool(force_vent_minus_pia_positive)
                if (med_dx < 0.0) == wants_positive:
                    p_vent, p_pia = p_pia, p_vent
                    v_skel = 1.0 - v_skel
    elif finite_n >= 20:
        med_dx = float(np.nanmedian(vent_x[finite] - pia_x[finite]))
        if np.isfinite(med_dx) and abs(med_dx) >= 0.25:
            wants_positive = midline_is_high_x
            if (med_dx < 0.0) == wants_positive:
                p_vent, p_pia = p_pia, p_vent
                v_skel = 1.0 - v_skel

    return {
        "slice_i": np.full((spline_yx.shape[0],), int(slice_i), dtype=np.int32),
        "u": u.astype(np.float64, copy=False),
        "y": y,
        "x": x,
        "ny": ny,
        "nx": nx,
        "t_neg": t_neg,
        "t_pos": t_pos,
        "vent_y": p_vent[:, 0],
        "vent_x": p_vent[:, 1],
        "pia_y": p_pia[:, 0],
        "pia_x": p_pia[:, 1],
        "thickness_px": thickness_px,
        "v_skel": v_skel,
    }


def _compute_thickness_by_slice_for_axis(
    *,
    axis: Literal[0, 2],
    slice_indices_axis: list[int],
    lines_spline_ijk_axis: list[np.ndarray],
    lines_spline_u_axis: list[np.ndarray],
    mask_3d_for_thickness: np.ndarray,
    include_mask_3d: np.ndarray,
) -> dict[int, dict[str, np.ndarray]]:
    if axis == 0:
        slice_col = "slice_i"
        out_csv = OUTDIR / "coronal_spline_thickness.csv"
        res_y_um = float(RES_J_UM)
        res_x_um = float(RES_K_UM)
    elif axis == 2:
        slice_col = "slice_k"
        out_csv = OUTDIR / "sagittal_spline_thickness.csv"
        res_y_um = float(RES_I_UM)
        res_x_um = float(RES_J_UM)
    else:
        raise ValueError(f"Unsupported axis={axis}. Expected 0 or 2.")

    rows: list[np.ndarray] = []
    thickness_by_slice: dict[int, dict[str, np.ndarray]] = {}
    force_vent_minus_pia_positive: bool | None = None

    for slice_idx, path_spline, u in zip(slice_indices_axis, lines_spline_ijk_axis, lines_spline_u_axis, strict=True):
        slice_index = int(slice_idx)
        if axis == 0:
            mask_yx = (mask_3d_for_thickness[slice_index, :, :] & include_mask_3d[slice_index, :, :]).astype(bool)
        else:
            mask_yx = (mask_3d_for_thickness[:, :, slice_index] & include_mask_3d[:, :, slice_index]).astype(bool)

        yx = _slice_plane_path_from_ijk(path_spline, axis=axis).astype(np.float64, copy=False)

        meas = measure_thickness_along_coronal_spline(
            slice_i=slice_index,
            mask_yx=mask_yx,
            spline_yx=yx,
            u=u,
            force_vent_minus_pia_positive=force_vent_minus_pia_positive if axis == 0 else None,
        )
        thickness_by_slice[slice_index] = meas
        if axis == 0:
            vent_x = meas["vent_x"]
            pia_x = meas["pia_x"]
            finite = np.isfinite(vent_x) & np.isfinite(pia_x)
            if int(np.sum(finite)) >= 5:
                med_dx = float(np.nanmedian(vent_x[finite] - pia_x[finite]))
                if np.isfinite(med_dx):
                    force_vent_minus_pia_positive = bool(med_dx >= 0.0)

        vent_y_um = meas["vent_y"] * res_y_um
        vent_x_um = meas["vent_x"] * res_x_um
        pia_y_um = meas["pia_y"] * res_y_um
        pia_x_um = meas["pia_x"] * res_x_um
        thickness_um = np.sqrt((vent_y_um - pia_y_um) ** 2 + (vent_x_um - pia_x_um) ** 2)

        rows.append(
            np.column_stack(
                [
                    np.full((meas["u"].shape[0],), slice_index, dtype=np.int32),
                    meas["u"],
                    meas["thickness_px"],
                    thickness_um.astype(np.float64),
                    meas["v_skel"],
                    meas["y"],
                    meas["x"],
                    meas["vent_y"],
                    meas["vent_x"],
                    meas["pia_y"],
                    meas["pia_x"],
                ]
            )
        )

    table = np.vstack(rows)
    np.savetxt(
        out_csv,
        table,
        delimiter=",",
        header=f"{slice_col},u,thickness_px,thickness_um,v_skel,y,x,vent_y,vent_x,pia_y,pia_x",
        comments="",
    )
    return thickness_by_slice


thickness_by_slice_i = _compute_thickness_by_slice_for_axis(
    axis=0,
    slice_indices_axis=slice_indices,
    lines_spline_ijk_axis=lines_spline_ijk,
    lines_spline_u_axis=lines_spline_u,
    mask_3d_for_thickness=mask_3d,
    include_mask_3d=neo_meso_3d,
)


# ## Phase 3d: Sagittal slice skeletons → 3D polylines
#
# Same idea as Phase 3, but per-sagittal slice (axis 2).

sagittal_lines_ijk, sagittal_lines_spline_ijk, sagittal_lines_spline_u, sagittal_slice_indices = _build_axis_lines_with_spline_hybrid(
    axis=2,
    lines_ijk_raw=sagittal_lines_ijk_raw,
    slice_indices_raw=sagittal_slice_indices_raw,
    other_lines_ijk_raw=coronal_lines_ijk_raw,
    include_mask_3d=neo_meso_3d,
)
if not sagittal_lines_ijk:
    raise ValueError("No sagittal slice skeleton lines remained after hybrid sourcing + include-mask filtering.")

_save_lines_um_csv(
    lines_ijk=sagittal_lines_ijk,
    slice_indices=sagittal_slice_indices,
    out_csv=OUTDIR / "sagittal_slice_lines_um.csv",
    slice_col="slice_k",
)
_save_spline_lines_um_csv(
    lines_spline_ijk=sagittal_lines_spline_ijk,
    lines_spline_u=sagittal_lines_spline_u,
    slice_indices=sagittal_slice_indices,
    out_csv=OUTDIR / "sagittal_slice_lines_spline_um.csv",
    slice_col="slice_k",
)

thickness_by_slice_k = _compute_thickness_by_slice_for_axis(
    axis=2,
    slice_indices_axis=sagittal_slice_indices,
    lines_spline_ijk_axis=sagittal_lines_spline_ijk,
    lines_spline_u_axis=sagittal_lines_spline_u,
    mask_3d_for_thickness=mask_3d,
    include_mask_3d=neo_meso_3d,
)

# ## Phase 5: Coronal↔sagittal chart transition maps (slice, u)
#
# We treat the extracted medial scaffold as a 2D manifold embedded in 3D:
# - Coronal chart:   ϕ_c(slice_i, u_c) → (i,j,k)
# - Sagittal chart:  ϕ_s(slice_k, u_s) → (i,j,k)
#
# To map between charts, we evaluate forward into 3D using the per-slice spline, then invert
# approximately by projecting onto candidate per-slice polylines in the other chart.
#
# Note: This intentionally ignores `v` / thickness and only unifies the sheet-like scaffold.

@dataclass(frozen=True)
class SlicePolyline:
    u: np.ndarray  # (N,) in [0,1], increasing
    p2_um: np.ndarray  # (N,2) in-plane coordinates (um)
    p3_vox: np.ndarray  # (N,3) voxel coordinates (float ok)

    def flipped(self) -> "SlicePolyline":
        u = self.u.astype(np.float64, copy=False)
        return SlicePolyline(
            u=(1.0 - u[::-1]).astype(np.float64, copy=False),
            p2_um=self.p2_um[::-1].astype(np.float64, copy=False),
            p3_vox=self.p3_vox[::-1].astype(np.float64, copy=False),
        )

    def eval_p3_vox(self, u_query: float) -> np.ndarray:
        uq = float(np.clip(float(u_query), 0.0, 1.0))
        u = self.u.astype(np.float64, copy=False)
        p3 = self.p3_vox.astype(np.float64, copy=False)
        out = np.empty((3,), dtype=np.float64)
        out[0] = float(np.interp(uq, u, p3[:, 0]))
        out[1] = float(np.interp(uq, u, p3[:, 1]))
        out[2] = float(np.interp(uq, u, p3[:, 2]))
        return out

    def project_u(self, q2_um: np.ndarray) -> tuple[float, float]:
        q = np.asarray(q2_um, dtype=np.float64).reshape(1, 2)
        p = self.p2_um.astype(np.float64, copy=False)
        if p.shape[0] < 2:
            raise ValueError("Need at least 2 polyline points to project.")

        p0 = p[:-1]
        p1 = p[1:]
        v = p1 - p0  # (M,2)
        w = q - p0  # (M,2) via broadcast
        vv = np.sum(v * v, axis=1)  # (M,)
        vv = np.where(vv > 0.0, vv, 1.0)
        t = np.sum(w * v, axis=1) / vv
        t = np.clip(t, 0.0, 1.0)
        proj = p0 + v * t[:, None]
        d2 = np.sum((proj - q) ** 2, axis=1)

        m = int(np.argmin(d2))
        u0 = float(self.u[m])
        u1 = float(self.u[m + 1])
        u_proj = u0 + float(t[m]) * (u1 - u0)
        return float(u_proj), float(d2[m])


def _nearest_existing(sorted_keys: np.ndarray, key: int) -> int:
    idx = int(np.searchsorted(sorted_keys, int(key)))
    if idx <= 0:
        return int(sorted_keys[0])
    if idx >= int(sorted_keys.size):
        return int(sorted_keys[-1])
    a = int(sorted_keys[idx - 1])
    b = int(sorted_keys[idx])
    return a if abs(int(key) - a) <= abs(int(key) - b) else b


def _orient_slice_family(polys: dict[int, SlicePolyline]) -> dict[int, SlicePolyline]:
    keys = sorted(polys.keys())
    if not keys:
        raise ValueError("No slice polylines to orient.")
    out: dict[int, SlicePolyline] = {}
    prev: SlicePolyline | None = None
    for k in keys:
        cur = polys[int(k)]
        if prev is None:
            out[int(k)] = cur
            prev = cur
            continue
        err_keep = float(np.mean((cur.p2_um - prev.p2_um) ** 2))
        cur_flip = cur.flipped()
        err_flip = float(np.mean((cur_flip.p2_um - prev.p2_um) ** 2))
        out[int(k)] = cur_flip if err_flip < err_keep else cur
        prev = out[int(k)]
    return out


def _build_coronal_polylines(
    *,
    lines_spline_ijk: list[np.ndarray],
    lines_spline_u: list[np.ndarray],
    slice_indices: list[int],
) -> dict[int, SlicePolyline]:
    out: dict[int, SlicePolyline] = {}
    for slice_i, p3_vox, u in zip(slice_indices, lines_spline_ijk, lines_spline_u, strict=True):
        p3 = p3_vox.astype(np.float64, copy=False)
        u = u.astype(np.float64, copy=False)
        # coronal in-plane coords: (j,k) in um
        p2_um = np.column_stack([p3[:, 1] * RES_J_UM, p3[:, 2] * RES_K_UM]).astype(np.float64, copy=False)
        out[int(slice_i)] = SlicePolyline(u=u, p2_um=p2_um, p3_vox=p3)
    return _orient_slice_family(out)


def _build_sagittal_polylines(
    *,
    sagittal_lines_spline_ijk: list[np.ndarray],
    sagittal_lines_spline_u: list[np.ndarray],
    sagittal_slice_indices: list[int],
) -> dict[int, SlicePolyline]:
    out: dict[int, SlicePolyline] = {}
    for slice_k, p3_vox, u in zip(sagittal_slice_indices, sagittal_lines_spline_ijk, sagittal_lines_spline_u, strict=True):
        p3 = p3_vox.astype(np.float64, copy=False)
        u = u.astype(np.float64, copy=False)
        # sagittal in-plane coords: (i,j) in um
        p2_um = np.column_stack([p3[:, 0] * RES_I_UM, p3[:, 1] * RES_J_UM]).astype(np.float64, copy=False)
        out[int(slice_k)] = SlicePolyline(u=u, p2_um=p2_um, p3_vox=p3)
    return _orient_slice_family(out)


def coronal_to_sagittal(
    *,
    slice_i: int,
    u_c: float,
    coronal: dict[int, SlicePolyline],
    sagittal: dict[int, SlicePolyline],
    coronal_keys: np.ndarray | None = None,
    sagittal_keys: np.ndarray | None = None,
    k_window: int = 2,
) -> tuple[int, float, float, float]:
    """
    Map canonical coronal coords (slice_i, u_c) -> (slice_k, u_s) by:
      1) forward eval into 3D, then
      2) inverse via polyline projection in candidate sagittal slices.

    Returns (slice_k, u_s, err_um, k_vox_pred).
    """
    if coronal_keys is None:
        coronal_keys = np.asarray(sorted(coronal.keys()), dtype=np.int32)
    if sagittal_keys is None:
        sagittal_keys = np.asarray(sorted(sagittal.keys()), dtype=np.int32)
    if coronal_keys.size == 0 or sagittal_keys.size == 0:
        raise ValueError("Empty coronal/sagittal polyline dict.")

    i0 = _nearest_existing(coronal_keys, int(slice_i))
    p3 = coronal[int(i0)].eval_p3_vox(float(u_c))

    return _p3_vox_to_sagittal(p3, sagittal=sagittal, sagittal_keys=sagittal_keys, k_window=k_window)


def _p3_vox_to_sagittal(
    p3_vox: np.ndarray,
    *,
    sagittal: dict[int, SlicePolyline],
    sagittal_keys: np.ndarray,
    k_window: int,
) -> tuple[int, float, float, float]:
    i_um = float(p3_vox[0]) * RES_I_UM
    j_um = float(p3_vox[1]) * RES_J_UM
    k_vox = float(p3_vox[2])
    k_center = int(np.rint(k_vox))

    cand: list[int] = []
    for dk in range(-int(k_window), int(k_window) + 1):
        cand.append(_nearest_existing(sagittal_keys, int(k_center + dk)))
    cand = sorted(set(cand))

    best_d2: float | None = None
    best_k: int = int(cand[0])
    best_u: float = 0.0
    for k_idx in cand:
        u_s, d2_inplane = sagittal[int(k_idx)].project_u(np.array([i_um, j_um], dtype=np.float64))
        dk_um = (k_vox - float(k_idx)) * RES_K_UM
        d2 = float(d2_inplane + dk_um * dk_um)
        if best_d2 is None or d2 < best_d2:
            best_d2 = d2
            best_k = int(k_idx)
            best_u = float(u_s)

    err_um = float(np.sqrt(best_d2)) if best_d2 is not None else float("nan")
    return int(best_k), float(best_u), float(err_um), float(k_vox)


def sagittal_to_coronal(
    *,
    slice_k: int,
    u_s: float,
    sagittal: dict[int, SlicePolyline],
    coronal: dict[int, SlicePolyline],
    sagittal_keys: np.ndarray | None = None,
    coronal_keys: np.ndarray | None = None,
    i_window: int = 2,
) -> tuple[int, float, float, float]:
    """
    Map sagittal coords (slice_k, u_s) -> canonical coronal (slice_i, u_c).

    Returns (slice_i, u_c, err_um, i_vox_pred).
    """
    if sagittal_keys is None:
        sagittal_keys = np.asarray(sorted(sagittal.keys()), dtype=np.int32)
    if coronal_keys is None:
        coronal_keys = np.asarray(sorted(coronal.keys()), dtype=np.int32)
    if sagittal_keys.size == 0 or coronal_keys.size == 0:
        raise ValueError("Empty sagittal/coronal polyline dict.")

    k0 = _nearest_existing(sagittal_keys, int(slice_k))
    p3 = sagittal[int(k0)].eval_p3_vox(float(u_s))

    return _p3_vox_to_coronal(p3, coronal=coronal, coronal_keys=coronal_keys, i_window=i_window)


def _p3_vox_to_coronal(
    p3_vox: np.ndarray,
    *,
    coronal: dict[int, SlicePolyline],
    coronal_keys: np.ndarray,
    i_window: int,
) -> tuple[int, float, float, float]:
    i_vox = float(p3_vox[0])
    j_um = float(p3_vox[1]) * RES_J_UM
    k_um = float(p3_vox[2]) * RES_K_UM
    i_center = int(np.rint(i_vox))

    cand: list[int] = []
    for di in range(-int(i_window), int(i_window) + 1):
        cand.append(_nearest_existing(coronal_keys, int(i_center + di)))
    cand = sorted(set(cand))

    best_d2: float | None = None
    best_i: int = int(cand[0])
    best_u: float = 0.0
    for i_idx in cand:
        u_c, d2_inplane = coronal[int(i_idx)].project_u(np.array([j_um, k_um], dtype=np.float64))
        di_um = (i_vox - float(i_idx)) * RES_I_UM
        d2 = float(d2_inplane + di_um * di_um)
        if best_d2 is None or d2 < best_d2:
            best_d2 = d2
            best_i = int(i_idx)
            best_u = float(u_c)

    err_um = float(np.sqrt(best_d2)) if best_d2 is not None else float("nan")
    return int(best_i), float(best_u), float(err_um), float(i_vox)


coronal_polys = _build_coronal_polylines(
    lines_spline_ijk=lines_spline_ijk, lines_spline_u=lines_spline_u, slice_indices=slice_indices
)
sagittal_polys = _build_sagittal_polylines(
    sagittal_lines_spline_ijk=sagittal_lines_spline_ijk,
    sagittal_lines_spline_u=sagittal_lines_spline_u,
    sagittal_slice_indices=sagittal_slice_indices,
)

coronal_keys = np.asarray(sorted(coronal_polys.keys()), dtype=np.int32)
sagittal_keys = np.asarray(sorted(sagittal_polys.keys()), dtype=np.int32)
u_grid = coronal_polys[int(coronal_keys[0])].u.astype(np.float64, copy=False)

map_c2s_k_idx = np.full((coronal_keys.size, u_grid.size), -1, dtype=np.int32)
map_c2s_u_s = np.full((coronal_keys.size, u_grid.size), np.nan, dtype=np.float64)
map_c2s_err_um = np.full((coronal_keys.size, u_grid.size), np.nan, dtype=np.float64)
map_c2s_k_vox = np.full((coronal_keys.size, u_grid.size), np.nan, dtype=np.float64)

for ii, slice_i in enumerate(coronal_keys.tolist()):
    p3_samples = coronal_polys[int(slice_i)].p3_vox.astype(np.float64, copy=False)
    for uu in range(int(u_grid.size)):
        k_idx, u_s, err_um, k_vox = _p3_vox_to_sagittal(
            p3_samples[uu],
            sagittal=sagittal_polys,
            sagittal_keys=sagittal_keys,
            k_window=2,
        )
        map_c2s_k_idx[ii, uu] = int(k_idx)
        map_c2s_u_s[ii, uu] = float(u_s)
        map_c2s_err_um[ii, uu] = float(err_um)
        map_c2s_k_vox[ii, uu] = float(k_vox)

np.savez_compressed(
    OUTDIR / "chart_map_coronal_to_sagittal.npz",
    slice_i=coronal_keys,
    u_c=u_grid,
    slice_k=map_c2s_k_idx,
    u_s=map_c2s_u_s,
    err_um=map_c2s_err_um,
    k_vox_pred=map_c2s_k_vox,
)

map_s2c_i_idx = np.full((sagittal_keys.size, u_grid.size), -1, dtype=np.int32)
map_s2c_u_c = np.full((sagittal_keys.size, u_grid.size), np.nan, dtype=np.float64)
map_s2c_err_um = np.full((sagittal_keys.size, u_grid.size), np.nan, dtype=np.float64)
map_s2c_i_vox = np.full((sagittal_keys.size, u_grid.size), np.nan, dtype=np.float64)

for kk, slice_k in enumerate(sagittal_keys.tolist()):
    p3_samples = sagittal_polys[int(slice_k)].p3_vox.astype(np.float64, copy=False)
    for uu in range(int(u_grid.size)):
        i_idx, u_c, err_um, i_vox = _p3_vox_to_coronal(
            p3_samples[uu],
            coronal=coronal_polys,
            coronal_keys=coronal_keys,
            i_window=2,
        )
        map_s2c_i_idx[kk, uu] = int(i_idx)
        map_s2c_u_c[kk, uu] = float(u_c)
        map_s2c_err_um[kk, uu] = float(err_um)
        map_s2c_i_vox[kk, uu] = float(i_vox)

np.savez_compressed(
    OUTDIR / "chart_map_sagittal_to_coronal.npz",
    slice_k=sagittal_keys,
    u_s=u_grid,
    slice_i=map_s2c_i_idx,
    u_c=map_s2c_u_c,
    err_um=map_s2c_err_um,
    i_vox_pred=map_s2c_i_vox,
)

print(
    "Chart maps saved:",
    f"coronal→sagittal={OUTDIR/'chart_map_coronal_to_sagittal.npz'}",
    f"sagittal→coronal={OUTDIR/'chart_map_sagittal_to_coronal.npz'}",
)

print("Generation complete. Run ccf/refextract/skeletonize_neocortex_mesocortex_ccf_3d_plot.py for plotting.")

# %%
