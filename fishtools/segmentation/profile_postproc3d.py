# %% [markdown]
# # 3D Postprocessing Pipeline Profiling
#
# This PyPercent notebook profiles the 4-phase pipeline:
# - Phase 1: `gaussian_smooth_labels` (Gaussian low-pass for labels)
# - Phase 2: `relabel_connected_components` (assign unique IDs to disconnected fragments)
# - Phase 3: `compute_metadata_and_adjacency` (compute volumes and adjacency)
# - Phase 4: `donate_small_cells` (absorb tiny fragments + disconnected pieces)
#
# Phase 1 (Gaussian Smooth) uses Gaussian-weighted voting:
# - For each label k, compute score_k = G * I_k (Gaussian-blurred indicator)
# - Assign each voxel to argmax_k(score_k)
# - Features smaller than ~sigma are smoothed away (protrusions, holes, jaggedness)
#
# Phase 2 ensures that fragments disconnected by smoothing get unique IDs,
# so Phase 4 can donate them to neighbors.

# %%
import time

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from fishtools.segment.postproc3d import (
    compute_metadata_and_adjacency,
    donate_small_cells,
    gaussian_erosion_to_margin_and_scale,
    gaussian_smooth_labels,
    gaussian_smooth_labels_cupy,
    relabel_connected_components,
)

# %% [markdown]
# ## Configuration
# - `MASKS_PATH`: path to a 3D masks array (`.npy` or `.npz` with key `'masks'`)
# - `V_MIN`: volume threshold (voxels) for "small" cells
# - `MIN_CONTACT_FRACTION`: minimum best_contact_area / volume(k) to donate
# - `GAUSSIAN_SIGMA`: Gaussian sigma for smoothing (features < sigma are removed)
# - `GAUSSIAN_EROSION_VOXELS` / `GAUSSIAN_EROSION_FWHM_FRAC`:
#   optional specs for Gaussian boundary erosion; used to compute and
#   apply a background scale factor inside `gaussian_smooth_labels`.

# %%
MASKS_PATH = (
    "/working/20251001_JaxA3_Coro11/analysis/deconv/segment2_5d/reg-0072_masks.tif"
)
V_MIN = 4000
MIN_CONTACT_FRACTION = 0.0
GAUSSIAN_SIGMA = 3.5  # Features smaller than ~sigma will be smoothed away
# Max expansion: limits how far labels can expand beyond original foreground.
# Prevents small labels from growing disproportionately large.
# Set to 0 to disable constraint (allow unlimited expansion).
GAUSSIAN_MAX_EXPANSION = 1
# Optional: specify desired erosion either in voxels or as fraction of FWHM.
# Set both to None to skip theoretical margin/scale reporting.
GAUSSIAN_EROSION_VOXELS = None  # e.g. 0.3 for ~0.3 voxel inward shift
GAUSSIAN_EROSION_FWHM_FRAC = -0.1  # negative => slight net dilation (~5% of FWHM)
# Backend for Gaussian smoothing: "cpu" or "cupy"
GAUSSIAN_BACKEND = "cupy"


# %% [markdown]
# ## Load masks

# %%
ext = MASKS_PATH.split(".")[-1].lower()
if ext == "npy":
    masks = np.load(MASKS_PATH)
elif ext == "npz":
    data = np.load(MASKS_PATH)
    if "masks" not in data:
        raise ValueError("npz file must contain 'masks' array")
    masks = data["masks"]
elif ext in ("tif", "tiff"):
    masks = tifffile.imread(MASKS_PATH)
else:
    raise ValueError(f"unsupported file extension: {MASKS_PATH}")

if masks.ndim == 2:
    masks = masks[None, ...]
if masks.ndim != 3:
    raise ValueError(f"expected 2D or 3D masks, got shape {masks.shape}")
if not np.issubdtype(masks.dtype, np.integer):
    masks = masks.astype(np.int32)

print(f"Loaded masks {masks.shape}, dtype={masks.dtype}")


# %% [markdown]
# ## Utilities



# %%
def random_label_cmap(max_label: int, seed: int = 0) -> mcolors.ListedColormap:
    """Create a discrete random colormap for labels 0..max_label.

    Background (0) is mapped to black; other labels get random colors.
    """
    rng = np.random.default_rng(seed)
    colors = rng.random((max_label + 1, 3))
    colors[0] = np.array([0.0, 0.0, 0.0])
    return mcolors.ListedColormap(colors)


def compute_change_breakdown(before: np.ndarray, after: np.ndarray) -> dict:
    """Compute breakdown of changes: erosion, interface flip, fill."""
    # Erosion: label → background
    eroded = (before > 0) & (after == 0)
    n_eroded = int(eroded.sum())

    # Fill: background → label
    filled = (before == 0) & (after > 0)
    n_filled = int(filled.sum())

    # Interface flip: label_i → label_j (both non-zero, different)
    flipped = (before > 0) & (after > 0) & (before != after)
    n_flipped = int(flipped.sum())

    return {
        "n_eroded": n_eroded,
        "n_filled": n_filled,
        "n_flipped": n_flipped,
    }


# %% [markdown]
# ## Run profiling (Phases 1–4)
# Timing info is printed immediately after each phase completes.

# %%
# Phase 1: Gaussian smooth
print(
    f"Phase 1: Gaussian smoothing (σ={GAUSSIAN_SIGMA}, max_expansion={GAUSSIAN_MAX_EXPANSION}, backend={GAUSSIAN_BACKEND})..."
)
bg_scale = 1.0
if GAUSSIAN_EROSION_VOXELS is not None or GAUSSIAN_EROSION_FWHM_FRAC is not None:
    margin, bg_scale = gaussian_erosion_to_margin_and_scale(
        sigma=GAUSSIAN_SIGMA,
        erosion_voxels=GAUSSIAN_EROSION_VOXELS,
        fwhm_fraction=GAUSSIAN_EROSION_FWHM_FRAC,
    )
    print(
        f"  Gaussian erosion spec -> margin={margin:.4f}, "
        f"bg_scale={bg_scale:.4f} (applied to background scores)"
    )
t0 = time.perf_counter()
backend_used = "cpu"
if GAUSSIAN_BACKEND.lower() == "cupy":
    try:
        import cupy  # type: ignore[import]

        _ = cupy.cuda.runtime.getDevice()
        masks_smooth = gaussian_smooth_labels_cupy(
            masks,
            sigma=GAUSSIAN_SIGMA,
            in_place=False,
            bg_scale=bg_scale,
            max_expansion=GAUSSIAN_MAX_EXPANSION,
        )
        backend_used = "cupy"
        print("  Using CuPy-accelerated gaussian_smooth_labels_cupy")
    except Exception:
        print("  CuPy backend requested but not available; falling back to CPU")
        masks_smooth = gaussian_smooth_labels(
            masks,
            sigma=GAUSSIAN_SIGMA,
            in_place=False,
            bg_scale=bg_scale,
            max_expansion=GAUSSIAN_MAX_EXPANSION,
        )
else:
    masks_smooth = gaussian_smooth_labels(
        masks,
        sigma=GAUSSIAN_SIGMA,
        in_place=False,
        bg_scale=bg_scale,
        max_expansion=GAUSSIAN_MAX_EXPANSION,
    )
    backend_used = "cpu"
t1 = time.perf_counter()
smooth_breakdown = compute_change_breakdown(masks, masks_smooth)
print(
    f"  ✓ Phase 1 done: {(t1 - t0) * 1000:.1f} ms "
    f"(eroded={smooth_breakdown['n_eroded']}, flipped={smooth_breakdown['n_flipped']}, filled={smooth_breakdown['n_filled']})"
)

# Phase 2: Relabel connected components (fragments get unique IDs)
print("Phase 2: Relabeling connected components...")
masks_relabeled = relabel_connected_components(masks_smooth, in_place=False)
t2 = time.perf_counter()
# num_labels_smooth = len(np.unique(masks_smooth)) - 1  # Exclude 0
# num_labels_relabeled = len(np.unique(masks_relabeled)) - 1
print(
    f"  ✓ Phase 2 done: {(t2 - t1) * 1000:.1f} ms"
)

# Phase 3: Compute metadata
print("Phase 3: Computing metadata and adjacency...")
volumes, adjacency, contact_areas = compute_metadata_and_adjacency(masks_relabeled)
t3 = time.perf_counter()
print(f"  ✓ Phase 3 done: {(t3 - t2) * 1000:.1f} ms")

# Phase 4: Donate small cells (handles fragments + originally small cells)
print("Phase 4: Donating small cells...")
masks_post = donate_small_cells(
    masks_relabeled,
    volumes=volumes,
    adjacency=adjacency,
    contact_areas=contact_areas,
    V_min=V_MIN,
    min_contact_fraction=MIN_CONTACT_FRACTION,
    in_place=False,
)
t4 = time.perf_counter()
donate_breakdown = compute_change_breakdown(masks_relabeled, masks_post)
print(
    f"  ✓ Phase 4 done: {(t4 - t3) * 1000:.1f} ms (flipped={donate_breakdown['n_flipped']})"
)
#%%
# # Final summary
print(f"\nTotal time: {(t4 - t0) * 1000:.1f} ms")
unique_before = np.unique(masks)
unique_after = np.unique(masks_post)
num_labels_before = unique_before[unique_before > 0].size
num_labels_after = unique_after[unique_after > 0].size
changed_voxels = int(np.count_nonzero(masks != masks_post))
print(f"Labels: {num_labels_before} -> {num_labels_after}")
print(f"Changed voxels: {changed_voxels}")


# %% [markdown]
# ## Visualization for a single slice
#
# Use `Z_SLICE` to choose which axial slice to visualize. This shows
# original, intermediate, and postprocessed labels side by side.

# %%
# Use intermediate masks from profiling run (no duplicate computation)
diff = masks != masks_post
changed_slices = np.where(diff.any(axis=(1, 2)))[0]
print("Changed slices (z indices):", changed_slices)

Z_SLICE = 12
z = int(np.clip(Z_SLICE, 0, masks.shape[0] - 1))
sl = np.s_[500:1500, 500:1500]
orig_slice = masks[z, *sl]
smooth_slice = masks_smooth[z, *sl]
post_slice = masks_post[z, *sl]
diff_slice = diff[z, *sl]

vmin = 0
vmax = int(max(orig_slice.max(), smooth_slice.max(), post_slice.max()))
label_cmap = random_label_cmap(vmax, seed=0)

fig, axes = plt.subplots(1, 4, figsize=(20, 4), squeeze=False)
ax_orig, ax_smooth, ax_post, ax_diff = axes[0]

ax_orig.imshow(orig_slice, vmin=vmin, vmax=vmax, cmap=label_cmap)
ax_orig.set_title(f"Original labels (z={z})")
ax_orig.axis("off")

ax_smooth.imshow(smooth_slice, vmin=vmin, vmax=vmax, cmap=label_cmap)
ax_smooth.set_title(f"After Gaussian Smooth (σ={GAUSSIAN_SIGMA})")
ax_smooth.axis("off")

ax_post.imshow(post_slice, vmin=vmin, vmax=vmax, cmap=label_cmap)
ax_post.set_title("After Relabel + Donate")
ax_post.axis("off")

ax_diff.imshow(diff_slice, cmap="gray")
ax_diff.set_title("Changed voxels (total)")
ax_diff.axis("off")

fig.tight_layout()
plt.show()

# %% [markdown]
# ## Change breakdown visualization
#
# Shows changes per phase:
# - Red: smooth eroded (Phase 1)
# - Yellow: smooth flipped (Phase 1)
# - Green: smooth filled (Phase 1)
# - Magenta: donation flips (Phase 4)

# %%
# Compute breakdowns for each phase
smooth_eroded_slice = (orig_slice > 0) & (smooth_slice == 0)
smooth_flipped_slice = (
    (orig_slice > 0) & (smooth_slice > 0) & (orig_slice != smooth_slice)
)
smooth_filled_slice = (orig_slice == 0) & (smooth_slice > 0)
# Note: relabeled_slice would show different labels, but visually same regions
# donate_flipped compares relabeled (same visual regions) to final
relabeled_slice = masks_relabeled[z, *sl]
donate_flipped_slice = (
    (relabeled_slice > 0) & (post_slice > 0) & (relabeled_slice != post_slice)
)

# Create RGB visualization
breakdown_rgb = np.zeros((*orig_slice.shape, 3), dtype=np.uint8)
breakdown_rgb[smooth_eroded_slice] = [255, 0, 0]  # Red for smooth eroded
breakdown_rgb[smooth_flipped_slice] = [255, 255, 0]  # Yellow for smooth flipped
breakdown_rgb[smooth_filled_slice] = [0, 255, 0]  # Green for smooth filled
breakdown_rgb[donate_flipped_slice] = [255, 0, 255]  # Magenta for donation flips

fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.imshow(breakdown_rgb)
ax.set_title(
    f"Change breakdown (z={z})\n"
    f"Red=erode ({smooth_eroded_slice.sum()}), "
    f"Yellow=flip ({smooth_flipped_slice.sum()}), "
    f"Green=fill ({smooth_filled_slice.sum()}), "
    f"Magenta=donate ({donate_flipped_slice.sum()})"
)
ax.axis("off")
# plt.tight_layout()
# plt.show()

# %%
tifffile.imwrite("/working/20251001_JaxA3_Coro11/analysis/deconv/segment2_5d/reg-0072_masksfixed.tif", masks_post.astype(np.uint16))
# %%
# Histogram of ROI volumes (using existing data from Phase 4)
# volumes is already computed in Phase 3, but it's for masks_relabeled
# We need volumes for masks_post, so recompute or use the fact that
# donate_small_cells only reassigns labels, not creates new ones
volumes_post, _, _ = compute_metadata_and_adjacency(masks_post)
#%%

# fig, ax = plt.subplots(figsize=(8, 5))
# ax.hist(volumes_post[1:][np.nonzero(volumes_post[1:])[0]], bins=50, edgecolor="black", alpha=0.7)
# ax.set_xscale("log")
# ax.set_xlabel("Volume (voxels)")
# ax.set_ylabel("Count")
# ax.set_title(f"ROI Volume Distribution (n={len(volumes_post) - 1} labels)")
# ax.axvline(V_MIN, color="red", linestyle="--", label=f"V_MIN={V_MIN}")
# ax.legend()
# plt.tight_layout()
# plt.show()

print(f"Volume stats: min={volumes_post[1:].min()}, max={volumes_post[1:].max()}, median={np.median(volumes_post[1:]):.0f}, mean={volumes_post[1:].mean():.0f}")
# %%
