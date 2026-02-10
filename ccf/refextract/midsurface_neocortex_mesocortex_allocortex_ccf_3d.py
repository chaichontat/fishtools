#%%
# # 3D cortex mid-surface extraction (DevCCF E15.5)
#
# Goal: reduce a folded cortical ribbon mask (neocortex+mesocortex+allocortex) to a 2D manifold.
# This script tries two approaches:
#   (1) 3D thinning skeletonization (medial-axis-like; sensitive to boundary noise)
#   (2) Harmonic (Laplace) mid-surface between pial vs inner (non-cortex) boundaries (more stable)
#
# Artifacts are written under `OUTDIR`.

from __future__ import annotations

import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import cg
from skimage.measure import marching_cubes
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

KEEP_LEFT_HEMISPHERE_ONLY = True

# Downsample factor for all computations (recommended for Laplace solve).
# Uses nearest-neighbor slicing (annotation[::DS,::DS,::DS]).
DS = 2

# Morphological cleanup on the downsampled mask.
FILL_HOLES = True
REMOVE_SMALL_OBJECTS_VOX = 10_000
FILL_SMALL_HOLES_VOX = 5_000
CLOSE_RADIUS_VOX = 1
OPEN_RADIUS_VOX = 0

# Erode cortex mask before Laplace fit (voxels in the DS grid).
ERODE_RADIUS_VOX_BEFORE_LAPLACE = 1

# Laplace midsurface parameters.
LAPLACE_MAX_ITERS = 10_000
LAPLACE_CG_RTOL = 1e-4
LAPLACE_JACOBI_TOL_MAXDELTA = 5e-5
MIDSURF_EPS = 0.03  # u-band around 0.5 (smaller => thinner but sparser)

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


def _subsample_points(xyz: np.ndarray, *, max_points: int, seed: int = 0) -> np.ndarray:
    if xyz.shape[0] <= max_points:
        return xyz
    rng = np.random.default_rng(seed)
    idx = rng.choice(xyz.shape[0], size=int(max_points), replace=False)
    return xyz[idx]


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


def _solve_laplace_dirichlet_jacobi(
    *,
    mask: np.ndarray,
    b0: np.ndarray,
    b1: np.ndarray,
    max_iters: int,
    tol_maxdelta: float,
) -> np.ndarray:
    mask = mask.astype(bool, copy=False)
    b0 = (b0 & mask).astype(bool, copy=False)
    b1 = (b1 & mask).astype(bool, copy=False)
    if np.any(b0 & b1):
        raise ValueError("b0 and b1 must be disjoint.")

    structure6 = ndi.generate_binary_structure(3, 1)
    interior = binary_erosion(mask, footprint=structure6)
    fixed = b0 | b1 | (mask & ~interior)
    update = interior & ~fixed
    if not np.any(update):
        raise ValueError("No interior voxels to solve (mask too thin at this DS).")

    u = np.zeros(mask.shape, dtype=np.float32)
    u[b1] = 1.0

    core = (slice(1, -1), slice(1, -1), slice(1, -1))
    update_core = update[core]
    if not np.any(update_core):
        raise ValueError("No interior voxels in core (mask touches volume edges).")

    t0 = time.perf_counter()
    for it in range(int(max_iters)):
        u_new = u.copy()
        nbr_avg = (
            u[:-2, 1:-1, 1:-1]
            + u[2:, 1:-1, 1:-1]
            + u[1:-1, :-2, 1:-1]
            + u[1:-1, 2:, 1:-1]
            + u[1:-1, 1:-1, :-2]
            + u[1:-1, 1:-1, 2:]
        ) / 6.0

        u_core = u[core]
        u_new_core = u_new[core]
        u_new_core[update_core] = nbr_avg[update_core].astype(np.float32, copy=False)
        u_new[core] = u_new_core

        # Re-apply boundary conditions.
        u_new[~mask] = 0.0
        u_new[b0] = 0.0
        u_new[b1] = 1.0

        max_delta = float(np.max(np.abs(u_new_core[update_core] - u_core[update_core])))
        u = u_new
        if max_delta <= float(tol_maxdelta):
            dt = time.perf_counter() - t0
            print(f"[laplace] converged it={it + 1} max_delta={max_delta:.3e} time={dt:.1f}s")
            return u
        if (it + 1) % 200 == 0:
            dt = time.perf_counter() - t0
            print(f"[laplace] it={it + 1} max_delta={max_delta:.3e} time={dt:.1f}s")

    dt = time.perf_counter() - t0
    raise RuntimeError(f"[laplace] did not converge in {max_iters} iters (last max_delta={max_delta:.3e}, {dt:.1f}s)")


def _solve_laplace_dirichlet_cg(
    *,
    mask: np.ndarray,
    b0: np.ndarray,
    b1: np.ndarray,
    maxiter: int,
    rtol: float,
) -> np.ndarray:
    mask = mask.astype(bool, copy=False)
    b0 = (b0 & mask).astype(bool, copy=False)
    b1 = (b1 & mask).astype(bool, copy=False)
    if np.any(b0 & b1):
        raise ValueError("b0 and b1 must be disjoint.")
    if not np.any(b0) or not np.any(b1):
        raise ValueError("Both b0 and b1 must be non-empty.")

    idx = -np.ones(mask.shape, dtype=np.int32)
    coords = np.argwhere(mask)
    n = int(coords.shape[0])
    idx[mask] = np.arange(n, dtype=np.int32)

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    b = np.zeros(n, dtype=np.float64)

    nbrs = np.asarray(
        [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
        dtype=np.int32,
    )

    b0_flat = b0[mask]
    b1_flat = b1[mask]

    zmax, ymax, xmax = mask.shape
    for i, (z, y, x) in enumerate(coords.tolist()):
        if b0_flat[i]:
            rows.append(i)
            cols.append(i)
            data.append(1.0)
            b[i] = 0.0
            continue
        if b1_flat[i]:
            rows.append(i)
            cols.append(i)
            data.append(1.0)
            b[i] = 1.0
            continue

        rows.append(i)
        cols.append(i)
        data.append(6.0)

        for dz, dy, dx in nbrs.tolist():
            zz, yy, xx = int(z + dz), int(y + dy), int(x + dx)
            if 0 <= zz < zmax and 0 <= yy < ymax and 0 <= xx < xmax and mask[zz, yy, xx]:
                j = int(idx[zz, yy, xx])
                rows.append(i)
                cols.append(j)
                data.append(-1.0)

    a = csr_matrix((data, (rows, cols)), shape=(n, n))
    t0 = time.perf_counter()
    diag = a.diagonal().astype(np.float64, copy=False)
    inv_diag = np.zeros_like(diag)
    ok = diag != 0
    inv_diag[ok] = 1.0 / diag[ok]
    from scipy.sparse.linalg import LinearOperator

    m = LinearOperator((n, n), matvec=lambda x: inv_diag * x)

    u_vec, info = cg(a, b, maxiter=int(maxiter), atol=0.0, rtol=float(rtol), M=m)
    dt = time.perf_counter() - t0
    if info != 0:
        raise RuntimeError(f"[laplace] CG did not converge (info={info}) after {dt:.1f}s. Try DS=2/3.")
    print(f"[laplace] CG converged in {dt:.1f}s (n={n})")

    u = np.zeros(mask.shape, dtype=np.float32)
    u[mask] = u_vec.astype(np.float32, copy=False)
    return u


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


# ## Phase 2: Cleanup mask (for both methods)

cortex_clean_3d = cortex_3d.copy()
cortex_clean_3d = _keep_largest_component(cortex_clean_3d)
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
cortex_clean_3d = _keep_largest_component(cortex_clean_3d)

print(f"Cortex voxels (clean)={int(np.count_nonzero(cortex_clean_3d))}")
np.save(OUTDIR / "cortex_mask_clean_3d_ds.npy", cortex_clean_3d.astype(np.bool_))


# ## Phase 3: Laplace harmonic mid-surface (fit on eroded mask)

cortex_fit_3d = cortex_clean_3d.copy()
if ERODE_RADIUS_VOX_BEFORE_LAPLACE > 0:
    cortex_fit_3d = binary_erosion(cortex_fit_3d, footprint=ball(int(ERODE_RADIUS_VOX_BEFORE_LAPLACE)))
cortex_fit_3d = _keep_largest_component(cortex_fit_3d)
if not np.any(cortex_fit_3d):
    raise ValueError(
        f"Erosion emptied cortex mask (ERODE_RADIUS_VOX_BEFORE_LAPLACE={ERODE_RADIUS_VOX_BEFORE_LAPLACE}, DS={DS})."
    )
print(f"Cortex voxels (fit)={int(np.count_nonzero(cortex_fit_3d))}")
np.save(OUTDIR / "cortex_mask_fit_3d_ds.npy", cortex_fit_3d.astype(np.bool_))

pial_b0, inner_b1 = _boundary_partition_pial_inner(
    cortex_mask=cortex_fit_3d, brain_mask=brain_mask_3d, cortex_reference_mask=cortex_clean_3d
)
print(f"[laplace] b0(pial)={int(np.count_nonzero(pial_b0))} b1(inner)={int(np.count_nonzero(inner_b1))}")
np.save(OUTDIR / "laplace_b0_pial_3d_ds.npy", pial_b0.astype(np.bool_))
np.save(OUTDIR / "laplace_b1_inner_3d_ds.npy", inner_b1.astype(np.bool_))

try:
    u = _solve_laplace_dirichlet_cg(
        mask=cortex_fit_3d,
        b0=pial_b0,  # u=0
        b1=inner_b1,  # u=1
        maxiter=int(LAPLACE_MAX_ITERS),
        rtol=float(LAPLACE_CG_RTOL),
    )
except RuntimeError as exc:
    print(str(exc))
    print("[laplace] falling back to Jacobi relaxation (slower but robust).")
    u = _solve_laplace_dirichlet_jacobi(
        mask=cortex_fit_3d,
        b0=pial_b0,
        b1=inner_b1,
        max_iters=int(LAPLACE_MAX_ITERS),
        tol_maxdelta=float(LAPLACE_JACOBI_TOL_MAXDELTA),
    )
np.save(OUTDIR / "laplace_u_3d_ds.npy", u.astype(np.float32, copy=False))

mid_band = cortex_fit_3d & (np.abs(u - 0.5) <= float(MIDSURF_EPS))
mid_band = _keep_largest_component(mid_band)
print(f"[laplace] mid_band voxels={int(np.count_nonzero(mid_band))}")
np.save(OUTDIR / "midsurface_laplace_midband_3d_ds.npy", mid_band.astype(np.bool_))

# Optional: marching cubes mesh of the u=0.5 isosurface (in um coordinates).
u_mc = u.copy()
u_mc[~cortex_fit_3d] = -1.0
verts_ijk, faces, _, _ = marching_cubes(u_mc, level=0.5, spacing=res_ds_ijk_um)
np.save(OUTDIR / "midsurface_laplace_mc_verts_ijk_um.npy", verts_ijk.astype(np.float32, copy=False))
np.save(OUTDIR / "midsurface_laplace_mc_faces.npy", faces.astype(np.int32, copy=False))
print(f"[laplace] marching_cubes verts={verts_ijk.shape[0]} faces={faces.shape[0]}")

laplace_mesh = Mesh(vertices_xyz=verts_ijk[:, [2, 1, 0]].astype(np.float32, copy=False), faces=faces.astype(np.int32))
write_ply_binary_little_endian(OUTDIR / "midsurface_laplace_u0p5.ply", laplace_mesh)


# ## Phase 4: 3D visualization (Matplotlib)

mask_xyz_um = _mask_to_xyz_um(cortex_fit_3d, res_ijk_um=res_ds_ijk_um)
mask_xyz_um = _subsample_points(mask_xyz_um, max_points=int(PLOT_MAX_MASK_POINTS), seed=0)

laplace_mesh_xyz_um = np.stack([verts_ijk[:, 2], verts_ijk[:, 1], verts_ijk[:, 0]], axis=1)
laplace_mesh_xyz_um = _subsample_points(laplace_mesh_xyz_um, max_points=int(PLOT_MAX_RESULT_POINTS), seed=2)

_save_scatter_3d(
    out_png=OUTDIR / "mpl3d_midsurface_laplace.png",
    mask_xyz_um=mask_xyz_um,
    result_xyz_um=laplace_mesh_xyz_um,
    title=f"Laplace midsurface mesh verts (u=0.5, DS={DS}, erode={ERODE_RADIUS_VOX_BEFORE_LAPLACE})",
)

print(f"Wrote artifacts to {OUTDIR}")
