# %%

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import scanpy as sc
import seaborn as sns
import tifffile
from roifile import ImagejRoi
from scipy.ndimage import binary_dilation
from shapely import MultiPolygon, Point, Polygon, STRtree
from skimage.filters import gaussian

plt.rcParams["image.aspect"] = "equal"
sns.set_theme()

path = Path("/working/20241119-ZNE172-Trc/baysor--all")
df = pl.read_csv(path / "spots.csv")
yx = df[["y", "x"]].to_numpy()
# adata = sc.read_loom(path / "segmentation_counts.loom")
# adata.var_names = adata.var["Name"]
# yx = adata.obs[["y", "x"]].to_numpy()

# %%


# %%
rotated_yx = yx.copy()

# Rotate each section in-place
for i in range(len(bins) - 1):
    mask = (bins[i] < yx[:, 0]) & (yx[:, 0] < bins[i + 1])
    rotated_yx[mask] = rotate_points(yx[mask], rots[i]) + 5000 * i
yx = rotated_yx
# %%
plt.scatter(*yx[::10].T, s=0.1, alpha=0.1)
df = df.with_columns(y=yx[:, 0], x=yx[:, 1]).write_csv(path / "spots.rotated.csv")

# %% Create coords image to draw ROIs
out = np.zeros([int(yx[:, 0].max() + 3), int(yx[:, 1].max() + 3)], dtype=np.uint8)
coords = np.round(yx).astype(int)

# Create offsets and extend coordinates
offsets = np.array([(i, j) for i in [-1, 0, 1] for j in [-1, 0, 1]])
extended_coords = coords[:, None] + offsets[None, :]
extended_coords = extended_coords.reshape(-1, 2)
# Vectorized bounds checking
mask = (extended_coords >= 0).all(axis=1) & (extended_coords < np.array(out.shape)[None, :]).all(axis=1)
out[extended_coords[mask, 0], extended_coords[mask, 1]] = 255

out[extended_coords[:, 0], extended_coords[:, 1]] = 255
tifffile.imwrite(path / "coords.tif", out[::2, ::2], compression="zlib")
# %%


def parse_roi(ij: ImagejRoi):
    c = ij.integer_coordinates
    c[:, 0] += ij.left
    c[:, 1] += ij.top
    return c * 2


rois = ImagejRoi.fromfile(path / "RoiSet.zip")  # type: ignore
polygons = [Polygon(parse_roi(r)) for r in rois]

# %%
tree = STRtree(polygons)
u = tree.query(list(map(lambda t: Point([t[1], t[0]]), yx)), predicate="intersects")
res = np.array(u[0])

# %%
fig, ax = plt.subplots(figsize=(4, 8), dpi=100)
ax.scatter(yx[res, 1], -yx[res, 0], s=1, alpha=0.3)
ax.set_aspect("equal")

# assert yx.shape[0] == len(adata)
# %%
# adata = adata[res]


# adata.write_h5ad(path / "segmentation_counts_rotcortex.h5ad")


# %%
def remove_gaps(yx, min_density=5, axis=1, bins=100):
    # Ensure axis is valid (0 or 1)
    if axis not in [0, 1]:
        raise ValueError("Axis must be 0 or 1")

    # Bin points along specified axis
    counts, edges = np.histogram(yx[:, axis], bins=bins)

    # Find all low-density regions
    is_gap = counts < min_density

    # Mark gaps to remove (all except first of consecutive runs)
    to_remove = np.zeros_like(is_gap)
    in_run = False
    for i in range(len(is_gap)):
        if is_gap[i]:
            if in_run:  # If we're already in a run, mark this gap for removal
                to_remove[i] = True
            in_run = True
        else:
            in_run = False

    # Remove marked gaps by shifting points
    adjusted_yx = yx.copy()
    total_shift = 0
    plt.scatter(*yx.T, s=0.1, alpha=0.1)

    for i in range(len(counts)):
        if to_remove[i]:  # Only remove gaps that are marked
            gap_size = edges[i + 1] - edges[i]
            mask = adjusted_yx[:, axis] > (edges[i + 1] - total_shift)
            adjusted_yx[mask, axis] -= gap_size
            total_shift += gap_size
    plt.scatter(*adjusted_yx.T, s=0.1, alpha=0.1)

    return adjusted_yx


# Usage:
# adjusted_yx = remove_gaps(yx, bin_size=100, min_density=5)

# %%
ry = remove_gaps(remove_gaps(yx[res], axis=1, bins=400), axis=0, bins=100)
# %%
df = df[res].with_columns(y=ry[:, 0], x=ry[:, 1])
df.write_csv(path / "spots.cortex.csv")

# %%
plt.scatter(*ry.T, s=0.1, alpha=0.1)


# %%
u = np.histogram(ry[:, 0], bins=200)

plt.plot(u[0])
# %%
