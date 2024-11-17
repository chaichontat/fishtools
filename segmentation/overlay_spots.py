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

# fig, ax = plt.subplots(figsize=(16, 12), dpi=200, facecolor="black")

# ax.imshow(img.astype(np.uint16)[::4, ::4], zorder=1, vmax=1)

# %%
# out = find_objects(img)
intensity = imread("/mnt/working/lai/registered--whole_embryo/stitch/05/00/fused_00-1.tif")
# %%

# dapi = imread("/fast2/3t3clean/analysis/deconv/registered/dapi3/0/fused_1.tif").squeeze()
# polya = imread("/fast2/3t3clean/analysis/deconv/registered/dapi3/2/fused_1.tif").squeeze()
# intensity = scale_image_2x_optimized(img)
#
# %%

# te = scale_image(isotropic_label_dilation(u[0], 1), 3)

# %%

# props = regionprops(te, intensity_image=scale_image(dapi, 3))

# %%
# plt.imshow(props[5].image)

# %%


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
    Path("/mnt/working/lai/registered--whole_embryo/stitch/TileConfiguration.registered.txt")
).downsample(2)

coords = tc.df
minimums = (coords["x"].min(), coords["y"].min())


# idx = 0
# %%
# @click.command()
# @click.argument("idx", type=int)
# @profile
# def run(idx: int):
idx = 0
logger.info(f"{idx}: reading image.")
with tifffile.TiffFile("chunks/combi_nobg.tif") as tif:
    img = tif.pages[idx].asarray()

sl = np.s_[:1000, 3500:4000] if DEBUG else np.s_[:, :]

# img = imread("chunks/combi_nobg.tif")[idx]
polygons = []
regen = np.zeros_like(img[sl])


logger.info(f"{idx}: finding regions.")
props = regionprops(img[sl])
area = 0
print(idx, f"Found {len(props)} regions.")
# %%
spots = (
    pl.read_parquet("/mnt/working/lai/registered--whole_embryo/genestar/spots.parquet")
    # .filter(pl.col("tile") == "1144")
    .with_row_index("ii")
    .with_columns(x=pl.col("x") / 2 - minimums[0], y=pl.col("y") / 2 - minimums[1])
    .with_columns(x_int=pl.col("x").cast(pl.Int32), y_int=pl.col("y").cast(pl.Int32))
    .sort(["y_int", "x_int"])
)

spots = spots if DEBUG else spots.filter(pl.col("z").is_between(idx - 0.5, idx + 0.5))
# %%
out = np.zeros([len(spots), 2], dtype=np.uint32)
curr_idx = 0
area = 0


@profile
def loop(i, region):
    global curr_idx, area
    if i and i % 100 == 0:
        logger.info(f"{idx}: {i} regions processed. Mean area: {area / 1000:.2f} px²")
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
    bbx, bby, *_ = region.bbox

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

    oks = sp[np.flatnonzero(cs[_sp["x"], _sp["y"]]), "ii"]
    out[curr_idx : curr_idx + len(oks), 0] = oks
    out[curr_idx : curr_idx + len(oks), 1] = region.label
    curr_idx += len(oks)


for i, region in enumerate(props):
    loop(i, region)
    # %%
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
for i, (point_index, polygon_index) in enumerate(matches):
    point_assignments.append({"point": point_index, "polygon": polygon_index})

roi_mapping = pl.DataFrame({"polygon": list(range(len(props))), "label": [p.label for p in props]}).cast(
    pl.UInt32
)

ident = (
    pl.DataFrame(point_assignments)
    .cast(pl.UInt32)
    .join(_sp, left_on="point", right_on="ii", how="left")
    .join(roi_mapping, on="polygon")
)

assert ident["polygon"].max() < len(polygons)

ident.write_parquet(f"chunks/ident_{idx}.parquet")


if __name__ == "__main__":
    run()

# %%
# fig, axs = plt.subplots(ncols=3, nrows=1, figsize=(12, 4), dpi=200)
# axs = axs.flatten()
# axs[0].imshow(intensity[sl], zorder=1, origin="lower")
# axs[1].imshow(img.astype(np.uint16)[sl], zorder=1, vmax=1, origin="lower")
# for i, ax in enumerate(axs):
#     # ax.axis("off")
#     ax.scatter(*filter_imshow(ident, sl), s=1, alpha=0.4)


# plt.ylim(-5000, 0)

# %%
