# %% [markdown]
# # 3D cortex skeleton (per-coronal slice lines) (E15.5 DevCCF)
#
# This workflow builds a 3D cortex mask (term + descendants), skeletonizes it into a 2D sheet-like
# medial scaffold, then extracts a single polyline per coronal slice and visualizes them in 3D.
#
# Run cells sequentially. Each phase writes artifacts under `OUTDIR`.

# %%
from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from IPython import get_ipython
from matplotlib.widgets import Slider
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

# Optional: VS Code interactive Matplotlib backend
ip = get_ipython()
if ip is not None:
    ip.run_line_magic("matplotlib", "widget")

# Fixed atlas (DevCCF E15.5 reference used elsewhere in `ccf/`).
ATLAS_NAME = "kim_dev_mouse_e15-5_lsfm_20um"


# %%
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

# Spline parametrization of the extracted coronal line (in pixel space).
SPLINE_ORDER = 3
SPLINE_SMOOTHING = 25.0
SPLINE_N_SAMPLES = 120

OUTDIR = Path("ccf/out/refextract/skeletonize_neocortex_mesocortex_3d")
OUTDIR.mkdir(parents=True, exist_ok=True)

BRAINGLOBE_CONFIG_DIR = Path("ccf/out/atlases/.brainglobe_config")


# %% [markdown]
# ## Phase 0: Load atlas annotation volume

# %%
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


# %% [markdown]
# ## Phase 1: Build 3D mask (term + descendants)

# %%
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


# %% [markdown]
# ## Phase 2: 3D cleanup + skeletonize

# %%
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


# %% [markdown]
# ## Phase 3: Coronal slice skeletons → 3D polylines

# %%
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


lines_ijk, lines_spline_ijk, lines_spline_u, slice_indices = _extract_axis_lines_with_spline(
    mask_3d=mask_clean_3d,
    include_mask_3d=neo_meso_3d,
    axis=0,
)
if not lines_ijk:
    raise ValueError("No coronal slice skeleton lines were extracted; lower thresholds or check terms.")

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


# %% [markdown]
# ## Phase 3h: Thickness along coronal spline (orthogonal to curve)
#
# For each slice, for each spline point, cast rays along the local normal direction to find the two
# mask boundaries and measure thickness (ventricular↔pial). We keep `v=0` as ventricular convention.

# %%
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
            if float(b0[1]) >= float(b1[1]):
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


thickness_rows: list[np.ndarray] = []
thickness_by_slice_i: dict[int, dict[str, np.ndarray]] = {}
for i, path_spline, u in zip(slice_indices, lines_spline_ijk, lines_spline_u, strict=True):
    i_int = int(i)
    # Use neo+meso only for thickness.
    mask_yx = (mask_clean_3d[i_int, :, :] & neo_meso_3d[i_int, :, :]).astype(bool)
    yx = path_spline[:, 1:].astype(np.float64, copy=False)
    meas = measure_thickness_along_coronal_spline(slice_i=i_int, mask_yx=mask_yx, spline_yx=yx, u=u)
    thickness_by_slice_i[i_int] = meas

    vent_y_um = meas["vent_y"] * RES_J_UM
    vent_x_um = meas["vent_x"] * RES_K_UM
    pia_y_um = meas["pia_y"] * RES_J_UM
    pia_x_um = meas["pia_x"] * RES_K_UM
    thickness_um = np.sqrt((vent_y_um - pia_y_um) ** 2 + (vent_x_um - pia_x_um) ** 2)

    thickness_rows.append(
        np.column_stack(
            [
                meas["slice_i"].astype(np.int32),
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

thickness_table = np.vstack(thickness_rows)
np.savetxt(
    OUTDIR / "coronal_spline_thickness.csv",
    thickness_table,
    delimiter=",",
    header="slice_i,u,thickness_px,thickness_um,v_skel,y,x,vent_y,vent_x,pia_y,pia_x",
    comments="",
)


# %% [markdown]
# ## Phase 3i: Coronal slider with v-parameter shading
#
# Visualizes the per-slice `v ∈ [0,1]` coordinate by rasterizing normal segments (ventricular→pial).

# %%
def _v_shading_image(
    *,
    shape: tuple[int, int],
    vent_yx: np.ndarray,
    pia_yx: np.ndarray,
    mask_yx: np.ndarray,
    n_v_samples: int = 64,
) -> np.ndarray:
    if vent_yx.shape != pia_yx.shape or vent_yx.ndim != 2 or vent_yx.shape[1] != 2:
        raise ValueError(f"Expected vent/pia shape (N,2), got {vent_yx.shape} and {pia_yx.shape}")
    h, w = int(shape[0]), int(shape[1])
    v_img = np.full((h, w), np.nan, dtype=np.float32)

    v_vals = np.linspace(0.0, 1.0, int(n_v_samples), dtype=np.float64)
    for v in v_vals:
        y = vent_yx[:, 0] + v * (pia_yx[:, 0] - vent_yx[:, 0])
        x = vent_yx[:, 1] + v * (pia_yx[:, 1] - vent_yx[:, 1])
        yi = np.rint(y).astype(np.int32)
        xi = np.rint(x).astype(np.int32)
        ok = (yi >= 0) & (yi < h) & (xi >= 0) & (xi < w)
        yi = yi[ok]
        xi = xi[ok]
        if yi.size == 0:
            continue
        v_img[yi, xi] = float(v)

    v_img[~mask_yx.astype(bool, copy=False)] = np.nan
    return v_img


def view_v_parameterization_coronal(
    *,
    mask_3d: np.ndarray,
    background_3d: np.ndarray,
    thickness_by_slice: dict[int, dict[str, np.ndarray]],
) -> None:
    n_i = int(mask_3d.shape[0])
    background_3d = background_3d[:, :, : mask_3d.shape[2]]
    vmin, vmax = _robust_vmin_vmax(background_3d)

    slices = sorted(thickness_by_slice.keys())
    if not slices:
        raise ValueError("thickness_by_slice is empty.")
    cur_i = int(np.clip(int(np.median(slices)), 0, n_i - 1))

    cmap = plt.get_cmap("turbo").copy()
    cmap.set_bad(alpha=0.0)

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    manager = getattr(fig.canvas, "manager", None)
    set_window_title = getattr(manager, "set_window_title", None)
    if callable(set_window_title):
        set_window_title("Coronal: v-parameter shading (ventricular=0)")
    plt.subplots_adjust(bottom=0.14)

    img = ax.imshow(
        background_3d[cur_i, :, :],
        cmap="gray",
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
    )
    (spline_line,) = ax.plot([], [], c="white", lw=1.0, alpha=0.9)
    v_im = ax.imshow(
        np.full_like(background_3d[cur_i, :, :], np.nan, dtype=np.float32),
        cmap=cmap,
        interpolation="nearest",
        vmin=0.0,
        vmax=1.0,
        alpha=0.65,
    )
    ax.set_xlabel("x (k)")
    ax.set_ylabel("y (j)")
    cbar = fig.colorbar(v_im, ax=ax, pad=0.02, fraction=0.04)
    cbar.set_label("v (ventricular=0 → pial=1)")

    slider_ax = fig.add_axes((0.15, 0.06, 0.7, 0.03))
    slider = Slider(slider_ax, "slice i", 0, n_i - 1, valinit=cur_i, valstep=1)

    cache: dict[int, np.ndarray] = {}
    cache_order: list[int] = []

    def v_for_slice(i: int) -> np.ndarray:
        if i in cache:
            return cache[i]
        meas = thickness_by_slice.get(i)
        if meas is None:
            return np.full(mask_3d.shape[1:], np.nan, dtype=np.float32)

        vent = np.column_stack([meas["vent_y"], meas["vent_x"]]).astype(np.float64)
        pia = np.column_stack([meas["pia_y"], meas["pia_x"]]).astype(np.float64)
        finite = np.isfinite(vent).all(axis=1) & np.isfinite(pia).all(axis=1)
        vent = vent[finite]
        pia = pia[finite]

        mask_yx = (mask_3d[i, :, :]).astype(bool, copy=False)
        out = _v_shading_image(shape=mask_yx.shape, vent_yx=vent, pia_yx=pia, mask_yx=mask_yx)

        cache[i] = out
        cache_order.append(i)
        if len(cache_order) > 12:
            drop = cache_order.pop(0)
            cache.pop(drop, None)
        return out

    def set_i(i: int) -> None:
        i = int(np.clip(i, 0, n_i - 1))
        img.set_data(background_3d[i, :, :])
        v_im.set_data(v_for_slice(i))

        meas = thickness_by_slice.get(i)
        if meas is None:
            spline_line.set_data([], [])
            n_pts = 0
        else:
            spline_line.set_data(meas["x"], meas["y"])
            n_pts = int(meas["x"].shape[0])

        ax.set_title(f"coronal i={i}  |  u_samples={n_pts}  |  mask_px={int(mask_3d[i].sum())}")
        fig.canvas.draw_idle()

    def on_key(event: object) -> None:
        key = getattr(event, "key", "")
        cur = int(slider.val)
        if key in {"right", "up", "d"}:
            slider.set_val(cur + 1)
        elif key in {"left", "down", "a"}:
            slider.set_val(cur - 1)

    slider.on_changed(lambda v: set_i(int(v)))
    fig.canvas.mpl_connect("key_press_event", on_key)
    set_i(cur_i)
    plt.show()


def view_v_parameterization_sagittal(
    *,
    mask_3d: np.ndarray,
    background_3d: np.ndarray,
    thickness_by_slice: dict[int, dict[str, np.ndarray]],
) -> None:
    n_k = int(mask_3d.shape[2])
    background_3d = background_3d[:, :, : mask_3d.shape[2]]
    vmin, vmax = _robust_vmin_vmax(background_3d)

    slices = sorted(thickness_by_slice.keys())
    if not slices:
        raise ValueError("thickness_by_slice is empty.")
    cur_k = int(np.clip(int(np.median(slices)), 0, n_k - 1))

    cmap = plt.get_cmap("turbo").copy()
    cmap.set_bad(alpha=0.0)

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    manager = getattr(fig.canvas, "manager", None)
    set_window_title = getattr(manager, "set_window_title", None)
    if callable(set_window_title):
        set_window_title("Sagittal: v-parameter shading (ventricular=0)")
    plt.subplots_adjust(bottom=0.14)

    img = ax.imshow(
        background_3d[:, :, cur_k],
        cmap="gray",
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
        aspect="auto",
    )
    (spline_line,) = ax.plot([], [], c="white", lw=1.0, alpha=0.9)
    v_im = ax.imshow(
        np.full_like(background_3d[:, :, cur_k], np.nan, dtype=np.float32),
        cmap=cmap,
        interpolation="nearest",
        vmin=0.0,
        vmax=1.0,
        alpha=0.65,
        aspect="auto",
    )
    ax.set_xlabel("y (j)")
    ax.set_ylabel("coronal slice (i)")
    cbar = fig.colorbar(v_im, ax=ax, pad=0.02, fraction=0.04)
    cbar.set_label("v (ventricular=0 → pial=1)")

    slider_ax = fig.add_axes((0.15, 0.06, 0.7, 0.03))
    slider = Slider(slider_ax, "x (k)", 0, n_k - 1, valinit=cur_k, valstep=1)

    cache: dict[int, np.ndarray] = {}
    cache_order: list[int] = []

    def v_for_slice(k: int) -> np.ndarray:
        if k in cache:
            return cache[k]
        meas = thickness_by_slice.get(k)
        if meas is None:
            return np.full(mask_3d.shape[:2], np.nan, dtype=np.float32)

        vent = np.column_stack([meas["vent_y"], meas["vent_x"]]).astype(np.float64)
        pia = np.column_stack([meas["pia_y"], meas["pia_x"]]).astype(np.float64)
        finite = np.isfinite(vent).all(axis=1) & np.isfinite(pia).all(axis=1)
        vent = vent[finite]
        pia = pia[finite]

        mask_ij = mask_3d[:, :, k].astype(bool, copy=False)
        out = _v_shading_image(shape=mask_ij.shape, vent_yx=vent, pia_yx=pia, mask_yx=mask_ij)

        cache[k] = out
        cache_order.append(k)
        if len(cache_order) > 12:
            drop = cache_order.pop(0)
            cache.pop(drop, None)
        return out

    def set_k(k: int) -> None:
        k = int(np.clip(k, 0, n_k - 1))
        img.set_data(background_3d[:, :, k])
        v_im.set_data(v_for_slice(k))

        meas = thickness_by_slice.get(k)
        if meas is None:
            spline_line.set_data([], [])
            n_pts = 0
        else:
            spline_line.set_data(meas["x"], meas["y"])
            n_pts = int(meas["x"].shape[0])

        ax.set_title(f"sagittal k={k}  |  u_samples={n_pts}  |  mask_px={int(mask_3d[:, :, k].sum())}")
        fig.canvas.draw_idle()

    def on_key(event: object) -> None:
        key = getattr(event, "key", "")
        cur = int(slider.val)
        if key in {"right", "up", "d"}:
            slider.set_val(cur + 1)
        elif key in {"left", "down", "a"}:
            slider.set_val(cur - 1)

    slider.on_changed(lambda v: set_k(int(v)))
    fig.canvas.mpl_connect("key_press_event", on_key)
    set_k(cur_k)
    plt.show()



# %% [markdown]
# ## Phase 3d: Sagittal slice skeletons → 3D polylines
#
# Same idea as Phase 3, but per-sagittal slice (axis 2).

# %%
sagittal_lines_ijk, sagittal_lines_spline_ijk, sagittal_lines_spline_u, sagittal_slice_indices = (
    _extract_axis_lines_with_spline(
        mask_3d=mask_clean_3d,
        include_mask_3d=neo_meso_3d,
        axis=2,
    )
)

if not sagittal_lines_ijk:
    raise ValueError("No sagittal slice skeleton lines were extracted; lower thresholds or check terms.")

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

sagittal_thickness_rows: list[np.ndarray] = []
thickness_by_slice_k: dict[int, dict[str, np.ndarray]] = {}
for k, path_spline, u in zip(sagittal_slice_indices, sagittal_lines_spline_ijk, sagittal_lines_spline_u, strict=True):
    k_int = int(k)
    mask_ij = (mask_clean_3d[:, :, k_int] & neo_meso_3d[:, :, k_int]).astype(bool)
    yx = path_spline[:, :2].astype(np.float64, copy=False)
    meas = measure_thickness_along_coronal_spline(slice_i=k_int, mask_yx=mask_ij, spline_yx=yx, u=u)
    thickness_by_slice_k[k_int] = meas

    vent_i_um = meas["vent_y"] * RES_I_UM
    vent_j_um = meas["vent_x"] * RES_J_UM
    pia_i_um = meas["pia_y"] * RES_I_UM
    pia_j_um = meas["pia_x"] * RES_J_UM
    thickness_um = np.sqrt((vent_i_um - pia_i_um) ** 2 + (vent_j_um - pia_j_um) ** 2)

    sagittal_thickness_rows.append(
        np.column_stack(
            [
                np.full((meas["u"].shape[0],), k_int, dtype=np.int32),
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

sagittal_thickness_table = np.vstack(sagittal_thickness_rows)
np.savetxt(
    OUTDIR / "sagittal_spline_thickness.csv",
    sagittal_thickness_table,
    delimiter=",",
    header="slice_k,u,thickness_px,thickness_um,v_skel,y,x,vent_y,vent_x,pia_y,pia_x",
    comments="",
)


# %% [markdown]
# ## Phase 3b: Scroll through coronal slices (mask + per-slice skeleton line)
#
# Controls:
# - Arrow keys: ←/→ or ↑/↓
#
# Notes:
# - The overlay is the *extracted per-slice line* (not the full 3D skeleton voxel set).

# %%
def _robust_vmin_vmax(vol_3d: np.ndarray) -> tuple[float, float]:
    sample = vol_3d[::4, ::4, ::4].astype(np.float32, copy=False)
    vmin, vmax = np.percentile(sample, [1, 99]).astype(np.float64)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or float(vmax) <= float(vmin):
        vmin = float(np.nanmin(sample))
        vmax = float(np.nanmax(sample))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or float(vmax) <= float(vmin):
            vmin, vmax = 0.0, 1.0
    return float(vmin), float(vmax)


def view_mask_and_line_by_slice(
    mask_3d: np.ndarray, lines_ijk_: list[np.ndarray], background_3d: np.ndarray | None = None
) -> None:
    if mask_3d.ndim != 3:
        raise ValueError(f"Expected mask_3d.ndim==3, got shape={mask_3d.shape}")
    if not lines_ijk_:
        raise ValueError("lines_ijk is empty.")

    n_slices_ = int(mask_3d.shape[0])
    if background_3d is None:
        background_3d = mask_3d.astype(np.float32)
    if background_3d.shape[0] != mask_3d.shape[0] or background_3d.shape[1] != mask_3d.shape[1]:
        raise ValueError(f"background_3d shape must match mask_3d in i/j, got {background_3d.shape} vs {mask_3d.shape}")
    if background_3d.shape[2] < mask_3d.shape[2]:
        raise ValueError(f"background_3d has fewer x-slices than mask_3d: {background_3d.shape} vs {mask_3d.shape}")
    background_3d = background_3d[:, :, : mask_3d.shape[2]]
    vmin, vmax = _robust_vmin_vmax(background_3d)

    line_by_slice_yx: dict[int, np.ndarray] = {}
    for path_ijk in lines_ijk_:
        if path_ijk.ndim != 2 or path_ijk.shape[1] != 3:
            raise ValueError(f"Expected (N,3) path_ijk, got shape={path_ijk.shape}")
        slice_i = int(path_ijk[0, 0])
        if not np.all(path_ijk[:, 0] == slice_i):
            raise ValueError("Each path_ijk must belong to a single slice.")
        line_by_slice_yx[slice_i] = path_ijk[:, 1:].astype(np.int32)

    cur_slice = int(np.clip(int(np.median(list(line_by_slice_yx.keys()))), 0, n_slices_ - 1))

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    manager = getattr(fig.canvas, "manager", None)
    set_window_title = getattr(manager, "set_window_title", None)
    if callable(set_window_title):
        set_window_title("Mask + skeleton line (coronal slices)")
    plt.subplots_adjust(bottom=0.14)

    img = ax.imshow(
        background_3d[cur_slice, :, :],
        cmap="gray",
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
    )
    mask_img = ax.imshow(
        mask_3d[cur_slice, :, :],
        cmap="Greens",
        interpolation="nearest",
        alpha=0.22,
        vmin=0.0,
        vmax=1.0,
    )
    (line_plot,) = ax.plot([], [], c="cyan", lw=1.5, alpha=0.95)
    ax.set_title(f"slice i={cur_slice}  |  mask_px={int(mask_3d[cur_slice].sum())}")
    ax.set_xlabel("x (k)")
    ax.set_ylabel("y (j)")

    slider_ax = fig.add_axes((0.15, 0.06, 0.7, 0.03))
    slider = Slider(slider_ax, "slice i", 0, n_slices_ - 1, valinit=cur_slice, valstep=1)

    def set_slice(i: int) -> None:
        nonlocal cur_slice
        i = int(np.clip(i, 0, n_slices_ - 1))
        cur_slice = i
        img.set_data(background_3d[i, :, :])
        mask_img.set_data(mask_3d[i, :, :])
        line_yx = line_by_slice_yx.get(i)
        if line_yx is None or line_yx.size == 0:
            line_plot.set_data([], [])
        else:
            # imshow uses (row=y, col=x); plot expects x then y.
            line_plot.set_data(line_yx[:, 1], line_yx[:, 0])
        ax.set_title(f"slice i={i}  |  mask_px={int(mask_3d[i].sum())}")
        fig.canvas.draw_idle()

    def on_key(event: object) -> None:
        key = getattr(event, "key", "")
        if key in {"right", "up"}:
            slider.set_val(int(cur_slice + 1))
        elif key in {"left", "down"}:
            slider.set_val(int(cur_slice - 1))

    slider.on_changed(lambda v: set_slice(int(v)))
    fig.canvas.mpl_connect("key_press_event", on_key)

    set_slice(cur_slice)
    plt.show()


if "atlas" not in globals():
    from brainglobe_atlasapi import BrainGlobeAtlas  # noqa: E402

    atlas = BrainGlobeAtlas(ATLAS_NAME)

reference_3d = np.asarray(atlas.reference)[:, :, : mask_clean_3d.shape[2]]

mask_neo_meso_3d = (mask_clean_3d & neo_meso_3d).astype(bool)
view_v_parameterization_coronal(mask_3d=mask_neo_meso_3d, background_3d=reference_3d, thickness_by_slice=thickness_by_slice_i)
view_v_parameterization_sagittal(mask_3d=mask_neo_meso_3d, background_3d=reference_3d, thickness_by_slice=thickness_by_slice_k)


def view_coronal_raw_vs_spline(
    mask_3d: np.ndarray,
    raw_lines_ijk: list[np.ndarray],
    spline_lines_ijk: list[np.ndarray],
    *,
    background_3d: np.ndarray,
) -> None:
    if mask_3d.ndim != 3:
        raise ValueError(f"Expected mask_3d.ndim==3, got shape={mask_3d.shape}")
    if not raw_lines_ijk:
        raise ValueError("raw_lines_ijk is empty.")
    if not spline_lines_ijk:
        raise ValueError("spline_lines_ijk is empty.")

    n_slices_ = int(mask_3d.shape[0])
    background_3d = background_3d[:, :, : mask_3d.shape[2]]
    vmin, vmax = _robust_vmin_vmax(background_3d)

    raw_by_slice_yx: dict[int, np.ndarray] = {int(p[0, 0]): p[:, 1:].astype(np.int32) for p in raw_lines_ijk}
    spline_by_slice_yx: dict[int, np.ndarray] = {int(p[0, 0]): p[:, 1:].astype(np.float64) for p in spline_lines_ijk}
    common_slices = sorted(set(raw_by_slice_yx) & set(spline_by_slice_yx))
    if not common_slices:
        raise ValueError("No overlapping slices between raw and spline lines.")

    cur_slice = int(np.clip(int(np.median(common_slices)), 0, n_slices_ - 1))

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    manager = getattr(fig.canvas, "manager", None)
    set_window_title = getattr(manager, "set_window_title", None)
    if callable(set_window_title):
        set_window_title("Coronal: raw points vs spline curve")
    plt.subplots_adjust(bottom=0.14)

    img = ax.imshow(
        background_3d[cur_slice, :, :],
        cmap="gray",
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
    )
    mask_img = ax.imshow(
        mask_3d[cur_slice, :, :],
        cmap="Greens",
        interpolation="nearest",
        alpha=0.22,
        vmin=0.0,
        vmax=1.0,
    )
    raw_sc = ax.scatter([], [], s=6.0, c="magenta", alpha=0.45, linewidths=0, label="raw")
    (spline_line,) = ax.plot([], [], c="cyan", lw=1.5, alpha=0.95, label="spline")
    ax.set_xlabel("x (k)")
    ax.set_ylabel("y (j)")
    ax.legend(loc="upper right")

    slider_ax = fig.add_axes((0.15, 0.06, 0.7, 0.03))
    slider = Slider(slider_ax, "slice i", 0, n_slices_ - 1, valinit=cur_slice, valstep=1)

    def set_slice(i: int) -> None:
        i = int(np.clip(i, 0, n_slices_ - 1))
        img.set_data(background_3d[i, :, :])
        mask_img.set_data(mask_3d[i, :, :])

        raw_yx = raw_by_slice_yx.get(i)
        spline_yx = spline_by_slice_yx.get(i)
        if raw_yx is None:
            raw_sc.set_offsets(np.empty((0, 2), dtype=np.float64))
            n_raw = 0
        else:
            raw_sc.set_offsets(np.column_stack([raw_yx[:, 1], raw_yx[:, 0]]).astype(np.float64))
            n_raw = int(raw_yx.shape[0])

        if spline_yx is None:
            spline_line.set_data([], [])
            n_spline = 0
        else:
            spline_line.set_data(spline_yx[:, 1], spline_yx[:, 0])
            n_spline = int(spline_yx.shape[0])

        ax.set_title(
            f"slice i={i}  |  raw_pts={n_raw}  |  spline_pts={n_spline}  |  mask_px={int(mask_3d[i].sum())}"
        )
        fig.canvas.draw_idle()

    def on_key(event: object) -> None:
        key = getattr(event, "key", "")
        cur = int(slider.val)
        if key in {"right", "up", "d"}:
            slider.set_val(cur + 1)
        elif key in {"left", "down", "a"}:
            slider.set_val(cur - 1)

    slider.on_changed(lambda v: set_slice(int(v)))
    fig.canvas.mpl_connect("key_press_event", on_key)
    set_slice(cur_slice)
    plt.show()


# %% [markdown]
# ## Phase 3c: Scroll through sagittal slices (spots from all coronal section lines)
#
# Shows a sagittal slice (axis 2) of the cleaned mask and overlays all extracted line points that lie
# on that x-slice.

# %%
def view_spots_sagittal(
    mask_3d: np.ndarray, lines_ijk_: list[np.ndarray], background_3d: np.ndarray | None = None
) -> None:
    if mask_3d.ndim != 3:
        raise ValueError(f"Expected mask_3d.ndim==3, got shape={mask_3d.shape}")
    if not lines_ijk_:
        raise ValueError("lines_ijk is empty.")

    if background_3d is None:
        background_3d = mask_3d.astype(np.float32)
    if background_3d.shape[0] != mask_3d.shape[0] or background_3d.shape[1] != mask_3d.shape[1]:
        raise ValueError(f"background_3d shape must match mask_3d in i/j, got {background_3d.shape} vs {mask_3d.shape}")
    if background_3d.shape[2] < mask_3d.shape[2]:
        raise ValueError(f"background_3d has fewer x-slices than mask_3d: {background_3d.shape} vs {mask_3d.shape}")
    background_3d = background_3d[:, :, : mask_3d.shape[2]]
    vmin, vmax = _robust_vmin_vmax(background_3d)

    points_ijk = np.vstack(lines_ijk_).astype(np.int32)
    i_vals = points_ijk[:, 0]
    j_vals = points_ijk[:, 1]
    k_vals = points_ijk[:, 2]

    n_k = int(mask_3d.shape[2])
    order = np.argsort(k_vals, kind="stable")
    k_sorted = k_vals[order]
    i_sorted = i_vals[order]
    j_sorted = j_vals[order]

    unique_k, starts = np.unique(k_sorted, return_index=True)
    ends = np.concatenate([starts[1:], np.asarray([k_sorted.size], dtype=starts.dtype)])
    k_to_slice = {int(k): (int(s), int(e)) for k, s, e in zip(unique_k.tolist(), starts.tolist(), ends.tolist(), strict=True)}

    cur_k = int(np.clip(int(np.median(unique_k)), 0, n_k - 1))

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    manager = getattr(fig.canvas, "manager", None)
    set_window_title = getattr(manager, "set_window_title", None)
    if callable(set_window_title):
        set_window_title("Sagittal view: mask + coronal-line points")
    plt.subplots_adjust(bottom=0.14)

    img = ax.imshow(
        background_3d[:, :, cur_k],
        cmap="gray",
        interpolation="nearest",
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
    )
    mask_img = ax.imshow(
        mask_3d[:, :, cur_k],
        cmap="Greens",
        interpolation="nearest",
        aspect="auto",
        alpha=0.18,
        vmin=0.0,
        vmax=1.0,
    )
    sc = ax.scatter([], [], s=2.0, c="magenta", alpha=0.9, linewidths=0)
    ax.set_xlabel("y (j)")
    ax.set_ylabel("coronal slice (i)")

    slider_ax = fig.add_axes((0.15, 0.06, 0.7, 0.03))
    slider = Slider(slider_ax, "x (k)", 0, n_k - 1, valinit=cur_k, valstep=1)

    def set_k(k: int) -> None:
        nonlocal cur_k
        k = int(np.clip(k, 0, n_k - 1))
        cur_k = k
        img.set_data(background_3d[:, :, k])
        mask_img.set_data(mask_3d[:, :, k])

        se = k_to_slice.get(k)
        if se is None:
            sc.set_offsets(np.empty((0, 2), dtype=np.float64))
            n_pts = 0
        else:
            s, e = se
            # scatter expects (x, y) = (j, i)
            offsets = np.column_stack([j_sorted[s:e], i_sorted[s:e]]).astype(np.float64)
            sc.set_offsets(offsets)
            n_pts = int(e - s)

        ax.set_title(f"sagittal k={k}  |  points={n_pts}  |  mask_px={int(mask_3d[:, :, k].sum())}")
        fig.canvas.draw_idle()

    def on_key(event: object) -> None:
        key = getattr(event, "key", "")
        if key in {"right", "up", "d"}:
            slider.set_val(int(cur_k + 1))
        elif key in {"left", "down", "a"}:
            slider.set_val(int(cur_k - 1))

    slider.on_changed(lambda v: set_k(int(v)))
    fig.canvas.mpl_connect("key_press_event", on_key)

    set_k(cur_k)
    plt.show()




# %% [markdown]
# ## Phase 3e: Slider view of sagittal per-slice line (mask + extracted line)

# %%
def view_mask_and_line_by_sagittal_slice(
    mask_3d: np.ndarray, lines_ijk_: list[np.ndarray], background_3d: np.ndarray | None = None
) -> None:
    if mask_3d.ndim != 3:
        raise ValueError(f"Expected mask_3d.ndim==3, got shape={mask_3d.shape}")
    if not lines_ijk_:
        raise ValueError("lines_ijk is empty.")

    n_k = int(mask_3d.shape[2])
    if background_3d is None:
        background_3d = mask_3d.astype(np.float32)
    if background_3d.shape[0] != mask_3d.shape[0] or background_3d.shape[1] != mask_3d.shape[1]:
        raise ValueError(
            f"background_3d shape must match mask_3d in i/j, got {background_3d.shape} vs {mask_3d.shape}"
        )
    if background_3d.shape[2] < mask_3d.shape[2]:
        raise ValueError(f"background_3d has fewer x-slices than mask_3d: {background_3d.shape} vs {mask_3d.shape}")
    background_3d = background_3d[:, :, : mask_3d.shape[2]]
    vmin, vmax = _robust_vmin_vmax(background_3d)

    line_by_k_ij: dict[int, np.ndarray] = {}
    for path_ijk in lines_ijk_:
        if path_ijk.ndim != 2 or path_ijk.shape[1] != 3:
            raise ValueError(f"Expected (N,3) path_ijk, got shape={path_ijk.shape}")
        slice_k = int(path_ijk[0, 2])
        if not np.all(path_ijk[:, 2] == slice_k):
            raise ValueError("Each path_ijk must belong to a single sagittal slice (constant k).")
        line_by_k_ij[slice_k] = path_ijk[:, :2].astype(np.int32)

    cur_k = int(np.clip(int(np.median(list(line_by_k_ij.keys()))), 0, n_k - 1))

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    manager = getattr(fig.canvas, "manager", None)
    set_window_title = getattr(manager, "set_window_title", None)
    if callable(set_window_title):
        set_window_title("Mask + skeleton line (sagittal slices)")
    plt.subplots_adjust(bottom=0.14)

    img = ax.imshow(
        background_3d[:, :, cur_k],
        cmap="gray",
        interpolation="nearest",
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
    )
    mask_img = ax.imshow(
        mask_3d[:, :, cur_k],
        cmap="Greens",
        interpolation="nearest",
        aspect="auto",
        alpha=0.22,
        vmin=0.0,
        vmax=1.0,
    )
    (line_plot,) = ax.plot([], [], c="cyan", lw=1.5, alpha=0.95)
    ax.set_xlabel("y (j)")
    ax.set_ylabel("coronal slice (i)")

    slider_ax = fig.add_axes((0.15, 0.06, 0.7, 0.03))
    slider = Slider(slider_ax, "x (k)", 0, n_k - 1, valinit=cur_k, valstep=1)

    def set_k(k: int) -> None:
        nonlocal cur_k
        k = int(np.clip(k, 0, n_k - 1))
        cur_k = k
        img.set_data(background_3d[:, :, k])
        mask_img.set_data(mask_3d[:, :, k])

        line_ij = line_by_k_ij.get(k)
        if line_ij is None or line_ij.size == 0:
            line_plot.set_data([], [])
        else:
            # plot expects x then y -> (j, i)
            line_plot.set_data(line_ij[:, 1], line_ij[:, 0])

        ax.set_title(f"sagittal k={k}  |  mask_px={int(mask_3d[:, :, k].sum())}")
        fig.canvas.draw_idle()

    def on_key(event: object) -> None:
        key = getattr(event, "key", "")
        if key in {"right", "up", "d"}:
            slider.set_val(int(cur_k + 1))
        elif key in {"left", "down", "a"}:
            slider.set_val(int(cur_k - 1))

    slider.on_changed(lambda v: set_k(int(v)))
    fig.canvas.mpl_connect("key_press_event", on_key)

    set_k(cur_k)
    plt.show()




# %% [markdown]
# ## Phase 3f: Slider view of sagittal-line points on coronal slices
#
# For each coronal slice `i`, overlay all points from the sagittal per-slice lines that fall on that `i`.

# %%
def view_sagittal_points_on_coronal(
    mask_3d: np.ndarray, sagittal_lines_ijk_: list[np.ndarray], background_3d: np.ndarray | None = None
) -> None:
    if mask_3d.ndim != 3:
        raise ValueError(f"Expected mask_3d.ndim==3, got shape={mask_3d.shape}")
    if not sagittal_lines_ijk_:
        raise ValueError("sagittal_lines_ijk is empty.")

    n_i = int(mask_3d.shape[0])
    if background_3d is None:
        background_3d = mask_3d.astype(np.float32)
    if background_3d.shape[0] != mask_3d.shape[0] or background_3d.shape[1] != mask_3d.shape[1]:
        raise ValueError(
            f"background_3d shape must match mask_3d in i/j, got {background_3d.shape} vs {mask_3d.shape}"
        )
    if background_3d.shape[2] < mask_3d.shape[2]:
        raise ValueError(f"background_3d has fewer x-slices than mask_3d: {background_3d.shape} vs {mask_3d.shape}")
    background_3d = background_3d[:, :, : mask_3d.shape[2]]
    vmin, vmax = _robust_vmin_vmax(background_3d)

    points_ijk = np.vstack(sagittal_lines_ijk_).astype(np.int32)
    i_vals = points_ijk[:, 0]
    j_vals = points_ijk[:, 1]
    k_vals = points_ijk[:, 2]

    order = np.argsort(i_vals, kind="stable")
    i_sorted = i_vals[order]
    j_sorted = j_vals[order]
    k_sorted = k_vals[order]

    unique_i, starts = np.unique(i_sorted, return_index=True)
    ends = np.concatenate([starts[1:], np.asarray([i_sorted.size], dtype=starts.dtype)])
    i_to_slice = {
        int(i): (int(s), int(e)) for i, s, e in zip(unique_i.tolist(), starts.tolist(), ends.tolist(), strict=True)
    }

    cur_i = int(np.clip(int(np.median(unique_i)), 0, n_i - 1))

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    manager = getattr(fig.canvas, "manager", None)
    set_window_title = getattr(manager, "set_window_title", None)
    if callable(set_window_title):
        set_window_title("Coronal view: sagittal-line points")
    plt.subplots_adjust(bottom=0.14)

    img = ax.imshow(
        background_3d[cur_i, :, :],
        cmap="gray",
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
    )
    mask_img = ax.imshow(
        mask_3d[cur_i, :, :],
        cmap="Greens",
        interpolation="nearest",
        alpha=0.18,
        vmin=0.0,
        vmax=1.0,
    )
    sc = ax.scatter([], [], s=2.0, c="magenta", alpha=0.9, linewidths=0)
    ax.set_xlabel("x (k)")
    ax.set_ylabel("y (j)")

    slider_ax = fig.add_axes((0.15, 0.06, 0.7, 0.03))
    slider = Slider(slider_ax, "slice i", 0, n_i - 1, valinit=cur_i, valstep=1)

    def set_i(i: int) -> None:
        nonlocal cur_i
        i = int(np.clip(i, 0, n_i - 1))
        cur_i = i
        img.set_data(background_3d[i, :, :])
        mask_img.set_data(mask_3d[i, :, :])

        se = i_to_slice.get(i)
        if se is None:
            sc.set_offsets(np.empty((0, 2), dtype=np.float64))
            n_pts = 0
        else:
            s, e = se
            # scatter expects (x, y) = (k, j)
            offsets = np.column_stack([k_sorted[s:e], j_sorted[s:e]]).astype(np.float64)
            sc.set_offsets(offsets)
            n_pts = int(e - s)

        ax.set_title(f"coronal i={i}  |  points={n_pts}  |  mask_px={int(mask_3d[i].sum())}")
        fig.canvas.draw_idle()

    def on_key(event: object) -> None:
        key = getattr(event, "key", "")
        if key in {"right", "up", "d"}:
            slider.set_val(int(cur_i + 1))
        elif key in {"left", "down", "a"}:
            slider.set_val(int(cur_i - 1))

    slider.on_changed(lambda v: set_i(int(v)))
    fig.canvas.mpl_connect("key_press_event", on_key)

    set_i(cur_i)
    plt.show()




# %% [markdown]
# ## Phase 3g: Combine coronal + sagittal points (piecewise rules) + slider viewers
#
# Rules:
# - Use sagittal points by default
# - For k < 160 and k > 224, use coronal curves

# %%
def combine_points_piecewise(
    sagittal_lines_ijk_: list[np.ndarray],
    coronal_lines_ijk_: list[np.ndarray],
    *,
    coronal_curves_k_lt: int = 160,
    coronal_curves_k_gt: int = 224,
) -> np.ndarray:
    if not sagittal_lines_ijk_:
        raise ValueError("sagittal_lines_ijk is empty.")
    if not coronal_lines_ijk_:
        raise ValueError("coronal lines_ijk is empty.")

    sag_points = np.vstack(sagittal_lines_ijk_).astype(np.int32, copy=False)
    cor_points = np.vstack(coronal_lines_ijk_).astype(np.int32, copy=False)

    # Default: sagittal points everywhere.
    k = sag_points[:, 2]
    keep_sag = (k >= int(coronal_curves_k_lt)) & (k <= int(coronal_curves_k_gt))
    sag_mid = sag_points[keep_sag]

    k_cor = cor_points[:, 2]
    cor_extremes = cor_points[(k_cor < int(coronal_curves_k_lt)) | (k_cor > int(coronal_curves_k_gt))]

    points = np.vstack([sag_mid, cor_extremes])

    # De-duplicate while preserving original geometric ordering.
    return _unique_rows_preserve_order(points)


combined_points_ijk = combine_points_piecewise(
    sagittal_lines_ijk_=sagittal_lines_ijk,
    coronal_lines_ijk_=lines_ijk,
    coronal_curves_k_lt=160,
    coronal_curves_k_gt=224,
)
combined_points_ijk = combined_points_ijk[
    neo_meso_3d[combined_points_ijk[:, 0], combined_points_ijk[:, 1], combined_points_ijk[:, 2]]
]


def view_points_coronal(
    mask_3d: np.ndarray, points_ijk: np.ndarray, background_3d: np.ndarray | None = None, *, title: str = ""
) -> None:
    if mask_3d.ndim != 3:
        raise ValueError(f"Expected mask_3d.ndim==3, got shape={mask_3d.shape}")
    if points_ijk.ndim != 2 or points_ijk.shape[1] != 3:
        raise ValueError(f"Expected points_ijk shape (N,3), got {points_ijk.shape}")

    n_i = int(mask_3d.shape[0])
    if background_3d is None:
        background_3d = mask_3d.astype(np.float32)
    background_3d = background_3d[:, :, : mask_3d.shape[2]]
    vmin, vmax = _robust_vmin_vmax(background_3d)

    i_vals = points_ijk[:, 0].astype(np.int32, copy=False)
    j_vals = points_ijk[:, 1].astype(np.int32, copy=False)
    k_vals = points_ijk[:, 2].astype(np.int32, copy=False)

    order = np.argsort(i_vals, kind="stable")
    i_sorted = i_vals[order]
    j_sorted = j_vals[order]
    k_sorted = k_vals[order]

    unique_i, starts = np.unique(i_sorted, return_index=True)
    ends = np.concatenate([starts[1:], np.asarray([i_sorted.size], dtype=starts.dtype)])
    i_to_slice = {
        int(i): (int(s), int(e)) for i, s, e in zip(unique_i.tolist(), starts.tolist(), ends.tolist(), strict=True)
    }
    cur_i = int(np.clip(int(np.median(unique_i)), 0, n_i - 1))

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    manager = getattr(fig.canvas, "manager", None)
    set_window_title = getattr(manager, "set_window_title", None)
    if callable(set_window_title):
        set_window_title("Coronal view: combined points")
    plt.subplots_adjust(bottom=0.14)

    img = ax.imshow(
        background_3d[cur_i, :, :],
        cmap="gray",
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
    )
    mask_img = ax.imshow(
        mask_3d[cur_i, :, :],
        cmap="Greens",
        interpolation="nearest",
        alpha=0.18,
        vmin=0.0,
        vmax=1.0,
    )
    sc = ax.scatter([], [], s=2.0, c="magenta", alpha=0.9, linewidths=0)
    ax.set_xlabel("x (k)")
    ax.set_ylabel("y (j)")

    slider_ax = fig.add_axes((0.15, 0.06, 0.7, 0.03))
    slider = Slider(slider_ax, "slice i", 0, n_i - 1, valinit=cur_i, valstep=1)

    def set_i(i: int) -> None:
        i = int(np.clip(i, 0, n_i - 1))
        img.set_data(background_3d[i, :, :])
        mask_img.set_data(mask_3d[i, :, :])

        se = i_to_slice.get(i)
        if se is None:
            sc.set_offsets(np.empty((0, 2), dtype=np.float64))
            n_pts = 0
        else:
            s, e = se
            offsets = np.column_stack([k_sorted[s:e], j_sorted[s:e]]).astype(np.float64)
            sc.set_offsets(offsets)
            n_pts = int(e - s)

        prefix = f"{title} | " if title else ""
        ax.set_title(f"{prefix}coronal i={i}  |  points={n_pts}  |  mask_px={int(mask_3d[i].sum())}")
        fig.canvas.draw_idle()

    def on_key(event: object) -> None:
        key = getattr(event, "key", "")
        cur = int(slider.val)
        if key in {"right", "up", "d"}:
            slider.set_val(cur + 1)
        elif key in {"left", "down", "a"}:
            slider.set_val(cur - 1)

    slider.on_changed(lambda v: set_i(int(v)))
    fig.canvas.mpl_connect("key_press_event", on_key)
    set_i(cur_i)
    plt.show()


def view_points_sagittal(
    mask_3d: np.ndarray, points_ijk: np.ndarray, background_3d: np.ndarray | None = None, *, title: str = ""
) -> None:
    if mask_3d.ndim != 3:
        raise ValueError(f"Expected mask_3d.ndim==3, got shape={mask_3d.shape}")
    if points_ijk.ndim != 2 or points_ijk.shape[1] != 3:
        raise ValueError(f"Expected points_ijk shape (N,3), got {points_ijk.shape}")

    n_k = int(mask_3d.shape[2])
    if background_3d is None:
        background_3d = mask_3d.astype(np.float32)
    background_3d = background_3d[:, :, : mask_3d.shape[2]]
    vmin, vmax = _robust_vmin_vmax(background_3d)

    i_vals = points_ijk[:, 0].astype(np.int32, copy=False)
    j_vals = points_ijk[:, 1].astype(np.int32, copy=False)
    k_vals = points_ijk[:, 2].astype(np.int32, copy=False)

    order = np.argsort(k_vals, kind="stable")
    k_sorted = k_vals[order]
    i_sorted = i_vals[order]
    j_sorted = j_vals[order]

    unique_k, starts = np.unique(k_sorted, return_index=True)
    ends = np.concatenate([starts[1:], np.asarray([k_sorted.size], dtype=starts.dtype)])
    k_to_slice = {
        int(k): (int(s), int(e)) for k, s, e in zip(unique_k.tolist(), starts.tolist(), ends.tolist(), strict=True)
    }
    cur_k = int(np.clip(int(np.median(unique_k)), 0, n_k - 1))

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    manager = getattr(fig.canvas, "manager", None)
    set_window_title = getattr(manager, "set_window_title", None)
    if callable(set_window_title):
        set_window_title("Sagittal view: combined points")
    plt.subplots_adjust(bottom=0.14)

    img = ax.imshow(
        background_3d[:, :, cur_k],
        cmap="gray",
        interpolation="nearest",
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
    )
    mask_img = ax.imshow(
        mask_3d[:, :, cur_k],
        cmap="Greens",
        interpolation="nearest",
        aspect="auto",
        alpha=0.18,
        vmin=0.0,
        vmax=1.0,
    )
    sc = ax.scatter([], [], s=2.0, c="magenta", alpha=0.9, linewidths=0)
    ax.set_xlabel("y (j)")
    ax.set_ylabel("coronal slice (i)")

    slider_ax = fig.add_axes((0.15, 0.06, 0.7, 0.03))
    slider = Slider(slider_ax, "x (k)", 0, n_k - 1, valinit=cur_k, valstep=1)

    def set_k(k: int) -> None:
        k = int(np.clip(k, 0, n_k - 1))
        img.set_data(background_3d[:, :, k])
        mask_img.set_data(mask_3d[:, :, k])

        se = k_to_slice.get(k)
        if se is None:
            sc.set_offsets(np.empty((0, 2), dtype=np.float64))
            n_pts = 0
        else:
            s, e = se
            offsets = np.column_stack([j_sorted[s:e], i_sorted[s:e]]).astype(np.float64)
            sc.set_offsets(offsets)
            n_pts = int(e - s)

        prefix = f"{title} | " if title else ""
        ax.set_title(f"{prefix}sagittal k={k}  |  points={n_pts}  |  mask_px={int(mask_3d[:, :, k].sum())}")
        fig.canvas.draw_idle()

    def on_key(event: object) -> None:
        key = getattr(event, "key", "")
        cur = int(slider.val)
        if key in {"right", "up", "d"}:
            slider.set_val(cur + 1)
        elif key in {"left", "down", "a"}:
            slider.set_val(cur - 1)

    slider.on_changed(lambda v: set_k(int(v)))
    fig.canvas.mpl_connect("key_press_event", on_key)
    set_k(cur_k)
    plt.show()


# %% [markdown]
# ## Phase 4: mpl3d visualization of per-slice lines in 3D

# %%
lines_to_plot = lines_ijk
slice_to_plot = slice_indices
if len(lines_to_plot) > int(MAX_SLICES_TO_PLOT):
    pick = np.linspace(0, len(lines_to_plot) - 1, int(MAX_SLICES_TO_PLOT)).astype(np.int64)
    lines_to_plot = [lines_to_plot[int(p)] for p in pick.tolist()]
    slice_to_plot = [slice_to_plot[int(p)] for p in pick.tolist()]

# Optional: scatter the full 3D skeleton as context (dim).
skel_ijk = np.column_stack(np.nonzero(np.load(OUTDIR / "skeleton_3d.npy").astype(bool))).astype(np.float64)
if skel_ijk.shape[0] > PLOT_MAX_SKELETON_POINTS:
    idx = np.linspace(0, skel_ijk.shape[0] - 1, PLOT_MAX_SKELETON_POINTS).astype(np.int64)
    skel_ijk = skel_ijk[idx]
skel_um_scatter = np.column_stack([skel_ijk[:, 0] * RES_I_UM, skel_ijk[:, 1] * RES_J_UM, skel_ijk[:, 2] * RES_K_UM])

fig = plt.figure(figsize=(11, 8))
ax = fig.add_subplot(111, projection="3d")

def _set_axes_equal_3d(ax_: object, xyz_um: np.ndarray) -> None:
    mins = np.min(xyz_um, axis=0)
    maxs = np.max(xyz_um, axis=0)
    ctr = (mins + maxs) / 2.0
    half = (maxs - mins) / 2.0
    radius = float(np.max(half))
    if not np.isfinite(radius) or radius <= 0:
        return
    ax_.set_xlim(ctr[0] - radius, ctr[0] + radius)
    ax_.set_ylim(ctr[1] - radius, ctr[1] + radius)
    ax_.set_zlim(ctr[2] - radius, ctr[2] + radius)
    # Matplotlib >= 3.3: enforce equal box aspect in 3D.
    set_box_aspect = getattr(ax_, "set_box_aspect", None)
    if callable(set_box_aspect):
        ax_.set_box_aspect((1, 1, 1))


ax.scatter(
    skel_um_scatter[:, 0],
    skel_um_scatter[:, 1],
    skel_um_scatter[:, 2],
    s=0.25,
    c="dimgray",
    alpha=0.18,
    linewidths=0,
)

if len(lines_to_plot) == 1:
    colors = ["cyan"]
    k_medians_um = [float(lines_to_plot[0][:, 2].astype(np.float64).mean() * RES_K_UM)]
else:
    k_medians_um = [
        float(np.median(path_ijk[:, 2].astype(np.float64) * RES_K_UM)) for path_ijk in lines_to_plot
    ]
    k0, k1 = np.percentile(np.asarray(k_medians_um, dtype=np.float64), [1, 99])
    if not np.isfinite(k0) or not np.isfinite(k1) or float(k1) <= float(k0):
        k0 = float(min(k_medians_um))
        k1 = float(max(k_medians_um))
        if float(k1) <= float(k0):
            k1 = k0 + 1.0

    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(vmin=float(k0), vmax=float(k1))
    colors = [cmap(norm(k)) for k in k_medians_um]

for path_ijk, color in zip(lines_to_plot, colors, strict=True):
    path_um = np.column_stack(
        [
            path_ijk[:, 0].astype(np.float64) * RES_I_UM,
            path_ijk[:, 1].astype(np.float64) * RES_J_UM,
            path_ijk[:, 2].astype(np.float64) * RES_K_UM,
        ]
    )
    ax.plot(path_um[:, 0], path_um[:, 1], path_um[:, 2], c=color, lw=1.0, alpha=0.9)

if len(lines_to_plot) > 1:
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.04)
    cbar.set_label("median k (um)")

ax.set_xlabel("i_um")
ax.set_ylabel("j_um")
ax.set_zlabel("k_um")
ax.set_title(
    f"{ATLAS_NAME} | TERMS={TERMS}\ncoronal lines={len(lines_ijk)} (plotted {len(lines_to_plot)})"
)
_set_axes_equal_3d(ax, skel_um_scatter)
ax.view_init(elev=23, azim=40)
fig.tight_layout()
fig.savefig(OUTDIR / "mpl3d_coronal_slice_lines.png", dpi=220)
plt.show()


# %%
