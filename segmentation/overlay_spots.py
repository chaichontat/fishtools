# %%
import logging
import pickle
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import pyparsing as pp
import rich_click as click
import seaborn as sns
import tifffile
from line_profiler import profile
from loguru import logger
from rtree import index
from scipy.ndimage import binary_closing, binary_dilation, binary_erosion, binary_fill_holes, find_objects
from shapely import MultiPolygon, Point, Polygon, contains, intersection
from shapely.geometry import Point, Polygon
from shapely.strtree import STRtree
from skimage import feature, filters, measure
from skimage.filters import gaussian
from skimage.measure import regionprops
from skimage.morphology import disk, reconstruction
from skimage.segmentation import clear_border
from tifffile import imread, imwrite

from fishtools.preprocess.tileconfig import TileConfiguration

DEBUG = False
sns.set_theme()


# %%


# %%

path = Path("/working/20250205_3t3BrduUntreated/analysis/deconv")
img = imread(path / "masks.tif")
dapi = imread(path / "stitch--left+dapieduwga/edu.tif")
props = regionprops(img, dapi)

# %%

# dapi = imread("/fast2/3t3clean/analysis/deconv/registered/dapi3/0/fused_1.tif").squeeze()
# polya = imread("/fast2/3t3clean/analysis/deconv/registered/dapi3/2/fused_1.tif").squeeze()
# intensity = scale_image_2x_optimized(img)
#
# %%


# %%
def bbox_to_slice(bbox, transpose=False):
    """
    Convert a bounding box tuple to numpy slice notation.

    Args:
        bbox: tuple of (xmin, ymin, xmax, ymax)

    Returns:
        tuple of (slice(ymin:ymax), slice(xmin:xmax)) for numpy array indexing
    """
    xmin, ymin, xmax, ymax = bbox
    # Convert to integers since array indices must be integers
    xmin, ymin = int(np.floor(xmin)), int(np.floor(ymin))
    xmax, ymax = int(np.ceil(xmax)), int(np.ceil(ymax))

    # Return in y,x order for numpy array indexing
    if not transpose:
        return np.s_[ymin:ymax, xmin:xmax]
    return np.s_[xmin:xmax, ymin:ymax]


def filter_(
    spots: pl.DataFrame,
    lim: tuple[tuple[float, float], tuple[float, float]] | tuple[slice, slice],
    *,
    shift: bool = False,
):
    """Filter spots within spatial bounds.

    Args:
        spots: DataFrame with 'x' and 'y' columns
        lim: ((x_min,x_max), (y_min,y_max)) or np.s_[y1:y2, x1:x2]

    Returns:
        Filtered DataFrame
    """
    match lim:
        case (xlim_lo, xlim_hi), (ylim_lo, ylim_hi):
            return spots.filter(
                (pl.col("x") > xlim_lo)
                & (pl.col("y") > ylim_lo)
                & (pl.col("x") < xlim_hi)
                & (pl.col("y") < ylim_hi)
            )
        case (sl_y, sl_x):
            return spots.filter(
                ((pl.col("x") > sl_x.start) if sl_x.start is not None else True)
                & ((pl.col("y") > sl_y.start) if sl_y.start is not None else True)
                & ((pl.col("x") < sl_x.stop) if sl_x.stop is not None else True)
                & ((pl.col("y") < sl_y.stop) if sl_y.stop is not None else True)
            ).with_columns(
                x=pl.col("x") - ((sl_x.start or 0) if shift else 0),
                y=pl.col("y") - ((sl_y.start or 0) if shift else 0),
            )

        case _:
            raise ValueError(f"Unknown type {type(lim)}")


def filter_imshow(
    spots: pl.DataFrame, lim: tuple[tuple[float, float], tuple[float, float]] | tuple[slice, slice]
):
    filtered = filter_(spots, lim, shift=True)
    match lim:
        case (xlim_lo, _), (ylim_lo, _):
            filtered = filtered.with_columns(
                x=pl.col("x") - (xlim_lo or 0),
                y=pl.col("y") - (ylim_lo or 0),
            )
        case (sl_y, sl_x):
            ...
        case _:
            raise ValueError(f"Unknown type {type(lim)}")
    return filtered["x"], filtered["y"]


# %%
tc = TileConfiguration.from_file(
    Path(
        path / "stitch--left/TileConfiguration.registered.txt"
    )
).downsample(2)

coords = tc.df
minimums = (coords["x"].min(), coords["y"].min())

print(f"Found {len(props)} regions.")
# %%
from datetime import datetime
print(datetime.fromtimestamp(( path / "tricycleplus--left+tricycleplus.parquet").stat().st_mtime))
spots = (
    pl.read_parquet(
      path / "tricycleplus--left+tricycleplus.parquet"
    )
    # .filter(pl.col("tile") == "1144")
    .with_row_index("ii")
    .with_columns(x=pl.col("x") / 2 - minimums[0], y=pl.col("y") / 2 - minimums[1])
    .with_columns(x_int=pl.col("x").cast(pl.Int32), y_int=pl.col("y").cast(pl.Int32))
    .sort(["y_int", "x_int"])
)
#%%


plt.imshow(img)
plt.scatter(spots["x"][::5], spots["y"][::5], s=1, alpha=0.5)
# spots = spots if DEBUG else spots.filter(pl.col("z").is_between(idx - 0.5, idx + 0.5))
#%%
df_polygons = pl.DataFrame([{"x": p.centroid[0], "y": p.centroid[1], "area": p.area, "label": p.label, "intensity": p.intensity_mean*p.area} for p in props])

# %%
out = np.zeros([len(spots), 2], dtype=np.uint32)
curr_idx = 0
area = 0
intensities = np.zeros(len(props),dtype=np.uint64)

@profile
def loop(i, region):
    global curr_idx, area
    if i and i % 100 == 0:
        logger.info(f"{i} regions processed. Mean area: {area / 1000:.2f} px²")
        area = 0

    area += region.area
    # threshold(gaussian(region.image, sigma=1))
    # cs = measure.find_contours(
    #     np.pad(binary_erosion(binary_fill_holes(region.image), iterations=1), (1, 1)), 0.5
    # )
    sigma = 0
    pad = sigma
    # cs = np.pad(region.image, (pad, pad))
    cs = region.image
    assert cs.shape[0] == region.image.shape[0] + 2 * pad and cs.shape[1] == region.image.shape[1] + 2 * pad

    # cs = binary_closing(region.image, iterations=1)  # somehow changes shape?
    # assert cs.shape[0] == region.image.shape[0] + 2 * pad and cs.shape[1] == region.image.shape[1] + 2 * pad

    # cs = gaussian(cs, sigma=sigma) > 0.2
    # cs = binary_dilation(binary_fill_holes(cs))

    assert cs.shape[0] == region.image.shape[0] + 2 * pad and cs.shape[1] == region.image.shape[1] + 2 * pad
    # contours, _ = cv2.findContours(cs.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # print(contours)
    # print(contours)
    # contours = cs ^ binary_erosion(cs, iterations=1)
    # contours = [np.argwhere(contours)]

    # contours = measure.find_contours(cs, 0.5)
    # print(contours)
    # if not len(contours):
    #     # To maintain indexing
    #     polygons.append(Polygon())
    #     continue

    # # if contours.__len__() > 1:
    # #     raise ValueError(f"Too many contours: {contours}")

    # filter_(spots, bbox_to_slice(props[20].bbox, transpose=True)), props[20].bbox

    bbx, bby, *_ = region.bbox
    # print()
    sp = (spots if not DEBUG else filter_(spots, sl, shift=True)).filter(
        pl.col("y_int").is_between(bbx, bbx + region.image.shape[0] - 1)
        & pl.col("x_int").is_between(bby, bby + region.image.shape[1] - 1)
    )
    if sp.is_empty():
        return


    # assert sp["x_int"].min() >= bbx
    # assert sp["x_int"].max() < bbx + region.image.shape[0]
    # assert sp["y_int"].max() < bby + region.image.shape[1]

    _sp = sp.select(x=pl.col("y_int") - bbx + pad, y=pl.col("x_int") - bby + pad)
    # is_ok = points_in_coords_vectorized(_sp['coded'], coded_coords)

    oks = sp[np.flatnonzero(cs[_sp["x"], _sp["y"]]), "ii"]
    # logger.info(len(oks))
    out[curr_idx : curr_idx + len(oks), 0] = oks
    out[curr_idx : curr_idx + len(oks), 1] = region.label

    curr_idx += len(oks)

#%%

for i, region in enumerate(props):
    loop(i, region)

# %%
sl = np.s_[4000:4500, 2500:3500]
fig, axs = plt.subplots(ncols=3, nrows=1, figsize=(12, 4), dpi=200)
axs = axs.flatten()

axs[0].imshow(img[sl], zorder=1, origin="lower")
axs[0].set_aspect("equal")
axs[1].scatter(*filter_imshow(spots, sl), s=2)
axs[1].set_aspect("equal")

axs[2].imshow(dapi[sl], zorder=1, origin="lower")
axs[2].set_aspect("equal")
plt.tight_layout()


# plt.scatter(spots['x'][::10], spots['y'][::10])
# %%
# %%
def plot_mask_with_points(mask, points, bbox, figsize=(10, 10)):
    """
    Plot a binary mask with points overlaid on top, using bbox limits.
    Uses object-oriented matplotlib interface.

    Args:
        mask: 2D numpy array of the binary mask
        points: Nx2 array of (x,y) coordinates in global space
        bbox: tuple of (xmin, ymin, xmax, ymax) for plot limits
        figsize: tuple of figure dimensions

    Returns:
        fig, ax: matplotlib figure and axes objects
    """
    import matplotlib.pyplot as plt

    # Create figure and axes
    fig, axs = plt.subplots(figsize=figsize,ncols=2)

    # Plot the mask in grayscale
    im = axs[0].imshow(mask, cmap="gray", extent=[bbox[0], bbox[2], bbox[1], bbox[3]], origin="lower")

    # Overlay points in red
    if points is not None and len(points) > 0:
        axs[0].scatter(points[:, "y"], points[:, "x"], c="red", s=30)

    # Set plot limits to bbox
    axs[0].set_xlim(bbox[0], bbox[2])
    axs[0].set_ylim(bbox[1], bbox[3])
    axs[0].set_aspect("equal")

    axs[0].set_title("Mask with Overlaid Points")

    axs[1].imshow(dapi, cmap="gray", extent=[bbox[0], bbox[2], bbox[1], bbox[3]], origin="lower")
    axs[1].set_xlim(bbox[0], bbox[2])
    axs[1].set_ylim(bbox[1], bbox[3])
    axs[1].set_aspect("equal")

    return fig


# Usage:
# fig, ax = plot_mask_with_points(my_mask, my_points, bbox)
# plt.show()  # if you want to display it
# fig.savefig('output.png')  # if you want to save it


plot_mask_with_points(props[450].image, filter_(spots, bbox_to_slice(props[450].bbox, transpose=True)), props[450].bbox)
# %%

# %%

    # %%
    #    # %%
    # _polygons = [] if len(contours) > 1 else None

    # for c in contours:
    # The coordinates are relative to the region bounding box,
    # so we need to shift them to the absolute position

    # c[:, 0] += bbx - pad  # - minimums[1] - 1
    # c[:, 1] += bby - pad  # - minimums[0] - 1

    # regim = region.image
    # regen[bbx : bbx + regim.shape[0], bby : bby + regim.shape[1]] = regim
    if DEBUG:
        regen[bbx : bbx + cs.shape[0] - 2 * pad, bby : bby + cs.shape[1] - 2 * pad] = cs[pad:-pad, pad:-pad]
        # x_start_corr = np.clip(0 - (bbx - pad), 0, None)
        # y_start_corr = np.clip(0 - (bby - pad), 0, None)

        # x_start = bbx - pad - x_start_corr
        # y_start = bby - pad - y_start_corr

        # x_end_corr = np.clip(x_start + cs.shape[0] - img.shape[0], 0, None)
        # y_end_corr = np.clip(y_start + cs.shape[1] - img.shape[1], 0, None)

        # regen[
        #     x_start : x_start + cs.shape[0] - x_end_corr,
        #     y_start : y_start + cs.shape[1] - y_end_corr,
        # ] = cs[x_start_corr : -x_end_corr or None, y_start_corr : -y_end_corr or None]


# %%
# sl = np.s_[3500:4000, 2000:2500]
if DEBUG:
    fig, axs = plt.subplots(figsize=(8, 4), dpi=200, ncols=3)
    axs[0].imshow(img[sl], zorder=1, origin="lower")
    axs[1].imshow(regen, zorder=1, origin="lower")
    # axs[2].imshow(cntrs[sl], zorder=1, vmax=1, origin="lower")


# %%

# plt.hexbin(spots['x'], spots['y'], gridsize=600)

# img
if DEBUG:
    # fig, axs = plt.subplots(ncols=3, nrows=1, figsize=(12, 4), dpi=200)
    # axs = axs.flatten()
    # # sl = np.s_[500:1000, 3500:4000]

    # axs[0].imshow(intensity[sl], zorder=1, origin="lower")
    # axs[0].set_aspect("equal")
    # axs[1].imshow(regen, zorder=1, vmax=1, origin="lower")
    # axs[1].set_aspect("equal")
    # axs[2].scatter(*filter_imshow(spots, sl), s=1, alpha=0.1)
    # axs[2].set_aspect("equal")
    # plt.tight_layout()

    fig, axs = plt.subplots(ncols=4, nrows=1, figsize=(12, 4), dpi=200)
    axs = axs.flatten()
    axs[0].imshow(intensity[sl], zorder=1, origin="lower")
    axs[1].imshow(img.astype(np.uint16)[sl], zorder=1, vmax=1, origin="lower")
    axs[2].imshow(regen, zorder=1, vmax=1, origin="lower")
    for i, ax in enumerate(axs):
        ax.axis("off")
        ax.scatter(*filter_imshow(spots, sl), s=1, alpha=0.4)
        ax.set_aspect("equal")
    plt.tight_layout()

    fig, axs = plt.subplots(ncols=4, nrows=1, figsize=(12, 4), dpi=200)
    axs = axs.flatten()
    axs[0].imshow(intensity[sl], zorder=1, origin="lower")
    axs[1].imshow(img.astype(np.uint16)[sl], zorder=1, vmax=1, origin="lower")
    axs[2].imshow(regen, zorder=1, vmax=1, origin="lower")
    __sp = spots.join(pl.DataFrame(out).with_row_index(), left_on="ii", right_on="index").filter(
        pl.col("column_0") > 0
    )
    for i, ax in enumerate(axs):
        ax.axis("off")
        ax.scatter(*filter_imshow(__sp, sl), s=1, alpha=0.4)
        ax.set_aspect("equal")


# %%
# Process results
point_assignments = []
for i, (point_index, polygon_index) in enumerate(out):
    point_assignments.append({"point": point_index, "label": polygon_index})

#%%
ident = (
    pl.DataFrame(point_assignments)
    .cast(pl.UInt32)
    .filter(pl.col("label") != 0)
    .join(spots, left_on="point", right_on="ii", how="left")
)
#%%


ident.write_parquet(path / "stitch--left+dapieduwga/spots_masks.parquet")




# %%
# fig, axs = plt.subplots(ncols=3, nrows=1, figsize=(12, 4), dpi=200)
# axs = axs.flatten()
# axs[0].imshow(intensity[sl], zorder=1, origin="lower")
# axs[1].imshow(img.astype(np.uint16)[sl], zorder=1, vmax=1, origin="lower")
# for i, ax in enumerate(axs):
#     # ax.axis("off")
#     ax.scatter(*filter_imshow(ident, sl), s=1, alpha=0.4)

# # %%
# unique = df.unique("label")["label"]
# final = pivoted.sort("label").to_pandas().set_index("label")

# %%
counted_df = ident.group_by(["target", "label"]).agg(
    pl.len().alias("count")
)
#%%
# Then pivot the counted data
pivoted_df = counted_df.pivot(
    on="target",
    values="count",
    index="label"
).fill_null(0).with_columns(label=pl.col("label").cast(pl.Int64)).join(df_polygons, on="label")  # fill missing combinations with 0
pivoted_df.write_parquet(path / f"registered--left+tricycleplus/cellxgene.parquet")
# %%
plt.loglog(pivoted_df["Top2a-201"]+np.random.normal(0,0.1,size=len(pivoted_df)), pivoted_df["intensity"],'.',alpha=0.5)

#%%
import json
ba = json.loads(Path("/archive/starmap/barrel/barrel_thicc1/analysis/deconv/stitch--barrel+pipolyA/barrels.json").read_text())
# %%

from shapely.geometry import shape, MultiPolygon
from shapely.affinity import scale

# Extract geometries from features and convert to MultiPolygon
polygons = []
for feature in ba['features']:
    geom = shape(feature['geometry'])
    if geom.geom_type == 'Polygon':
        polygons.append((geom, feature['label']))
    # elif geom.geom_type == 'MultiPolygon':
    #     polygons.extend(geom.geoms)
    else:
        raise ValueError(f"Unknown geometry type: {geom.geom_type}")

# Combine into a single MultiPolygon
#%%
multipolygon = MultiPolygon(polygons)
# %%

# %%
np.array(list(multipolygon.geoms[0].exterior.coords) )/0.216e-6
# %%
scaled_multipolygon = [(scale(polygon, xfact=1/(0.216e-6), yfact=-1/0.216e-6,origin=(0,0)), label) for polygon, label in polygons]

#%%

# %%
out = {}
for polygon,label in scaled_multipolygon:
    print(label)
    tree = STRtree([polygon])
    u = tree.query(list(map(lambda t: Point([t[0], t[1]]), spots[["x", "y"]].to_numpy())), predicate="intersects")
    out[label] = u[0]
# %%
labels = np.zeros(len(spots), dtype="U2")
for label, idx in out.items():
    labels[idx] = label
#%%
areas = {label: polygon.area for polygon, label in scaled_multipolygon}
deprived = "A2, A4, B1, B3, C2, C4, C6, D3, D5, D7, E3, E5".split(", ")
#%%
# %%
gb = spots.group_by("label", "target").agg(count=pl.len()).sort("label", "target").filter(pl.col("label")!="").with_columns(deprived=pl.col("label").is_in(deprived))

# Convert the area dictionary to a Polars DataFrame
area_df = pl.DataFrame({
    'label': list(areas.keys()),
    'area': list(areas.values())
})

# Join with original dataframe and calculate density
result = (gb
    .join(area_df, on='label')
    .with_columns(
        pl.col('count').truediv(pl.col('area')).alias('density')
    )
)
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
#%%
#%%
# Assuming 'result' is your polars dataframe
genes = ["Arc-201", "Fos-201", "Plcxd2-201"]

plt.figure(figsize=(12, 6))
for i, gene in enumerate(genes, 1):
    plt.subplot(1, 3, i)
    data = result.filter(pl.col('target') == gene).to_pandas()
    sns.stripplot(data=data, x='target', y='density', hue='deprived', dodge=True, alpha=1, zorder=1)
    sns.barplot(data=data, x='target', y='density', hue='deprived', alpha=.8)

    plt.title(gene)
    plt.xticks([])  # Remove x-axis labels since they're redundant
    if i > 1:
        plt.ylabel('')  # Only keep y-label for first plot

plt.tight_layout()
#%%
# First pivot the data to wide format where each gene is a column
wide_data = (result

    .pivot(
        index='label',
        on='target',
        values='density'
    )
)
# Create a scatter plot matrix using seaborn
plt.figure(figsize=(10, 10))
sns.pairplot(
    wide_data.to_pandas(),
    diag_kind='hist',  # or 'hist' if you prefer histograms
)
plt.tight_layout()

#%%
# First pivot the data as before
wide_data = (result
    .filter(pl.col('target').is_in(["Arc-201", "Fos-201", "Plcxd2-201"]))
    .pivot(
        index='label',
        columns='target',
        values='density'
    )
)

# Create figure with subplots for each pair of genes
genes = ["Arc-201", "Fos-201", "Plcxd2-201"]
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Create scatterplots for each pair of genes
pairs = [('Arc-201', 'Fos-201'), ('Arc-201', 'Plcxd2-201'), ('Fos-201', 'Plcxd2-201')]

for ax, (gene1, gene2) in zip(axes, pairs):
    data = wide_data.to_pandas()

    # Create scatter plot
    scatter = ax.scatter(data[gene1], data[gene2], c=range(len(data)), cmap='tab20')

    # Add text labels
    for idx, row in data.iterrows():
        ax.annotate(row['label'], (row[gene1], row[gene2]),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=8)

    corr = wide_data.to_pandas()[gene1].corr(wide_data.to_pandas()[gene2])
    ax.set_xlabel(gene1)
    ax.set_ylabel(gene2)
    ax.set_title(f'{gene1} vs {gene2}. Correlation: {corr:.2f}')

plt.tight_layout()
plt.show()



# %%
from matplotlib import pyplot as plt

# Assuming you have a MultiPolygon called 'multipolygon'
fig, ax = plt.subplots()
ax.imshow(img, vmax=1,zorder=1)
# Loop through each polygon in the multipolygon
for polygon,label in scaled_multipolygon:
    # Draw exterior boundary
    x, y = polygon.exterior.xy
    ax.plot(x, y, 'r-')  # 'k-' means black solid line
    ax.text(x[0], y[0], label, fontsize=12, color='k')


ax.set_aspect('equal')
plt.show()
# %%
