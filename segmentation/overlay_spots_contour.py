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
# intensity = imread("/mnt/working/lai/registered--whole_embryo/stitch/05/00/fused_00-1.tif")
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


def filter_(spots: pl.DataFrame, lim: tuple[tuple[float, float], tuple[float, float]] | tuple[slice, slice]):
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
            )
        case _:
            raise ValueError(f"Unknown type {type(lim)}")


def filter_imshow(
    spots: pl.DataFrame, lim: tuple[tuple[float, float], tuple[float, float]] | tuple[slice, slice]
):
    filtered = filter_(spots, lim)
    match lim:
        case (xlim_lo, _), (ylim_lo, _):
            filtered = filtered.with_columns(
                x=pl.col("x") - (xlim_lo or 0),
                y=pl.col("y") - (ylim_lo or 0),
            )
        case (sl_y, sl_x):
            filtered = filtered.with_columns(
                x=pl.col("x") - (sl_x.start or 0),
                y=pl.col("y") - (sl_y.start or 0),
            )
        case _:
            raise ValueError(f"Unknown type {type(lim)}")
    return filtered["x"], filtered["y"]


# %%
tc = TileConfiguration.from_file(
    Path("/mnt/working/lai/registered--whole_embryo/stitch/TileConfiguration.registered.txt")
).downsample(2)

coords = tc.df
minimums = (coords["x"].min(), coords["y"].min())


# # %%
@click.command()
@click.argument("idx", type=int)
def run(idx: int):
    logger.info(f"{idx}: reading image.")
    with tifffile.TiffFile("chunks/combi.tif") as tif:
        img = tif.pages[idx].asarray()
        # img = np.where(img & ((1 << 23) - 1), img, 0)
    # img = imread("chunks/combi_nobg.tif")[idx]
    polygons = []
    regen = np.zeros_like(img)
    cntrs = np.zeros_like(img)

    spots = (
        pl.read_parquet("/mnt/working/lai/registered--whole_embryo/genestar/spots.parquet")
        # .filter(pl.col("tile") == "1144")
        .with_row_index("ii")
        .with_columns(
            ii=pl.col("ii").cast(pl.Int64), x=pl.col("x") / 2 - minimums[0], y=pl.col("y") / 2 - minimums[1]
        )
    )
    spots = (
        spots.filter(pl.col("z").is_between((idx - 0.5) / 3, (idx + 0.5) / 3, closed="left"))
        if not DEBUG
        else spots
    )

    if not len(spots):
        raise Exception(f"No spots of z={idx} loaded.")
    logger.info(f"{idx}: {len(spots)} spots found.")

    logger.info(f"{idx}: finding regions.")
    props = regionprops(img)
    area = 0
    print(idx, f"Found {len(props)} regions.")

    @profile
    def loop(i, region):
        nonlocal area
        if i and i % 5000 == 0:
            logger.info(f"{idx}: {i} regions processed. Mean area: {area / 5000:.2f} px²")
            area = 0

        area += region.area
        # threshold(gaussian(region.image, sigma=1))
        # cs = measure.find_contours(
        #     np.pad(binary_erosion(binary_fill_holes(region.image), iterations=1), (1, 1)), 0.5
        # )
        sigma = 3
        pad = sigma
        cs = region.image
        cs = np.pad(region.image, (pad, pad))
        # cs = np.pad(binary_closing(region.image, iterations=1), (pad, pad))
        cs = gaussian(cs, sigma=sigma) > 0.2
        cs = binary_erosion(binary_fill_holes(cs), iterations=1)
        # contours, _ = cv2.findContours(cs.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        # print(contours)
        # print(contours)
        # contours = cs ^ binary_erosion(cs, iterations=1)
        # contours = [np.argwhere(contours)]
        if cs.shape[0] < 4 or cs.shape[1] < 4:
            polygons.append(Polygon())
            return

        contours = measure.find_contours(cs, 0.5)
        # print(contours)
        if not len(contours):
            # To maintain indexing
            polygons.append(Polygon())
            return

        # if contours.__len__() > 1:
        #     raise ValueError(f"Too many contours: {contours}")

        _polygons = []

        for c in contours:
            if len(c) < 6:
                continue
            # The coordinates are relative to the region bounding box,
            # so we need to shift them to the absolute position
            bbx, bby, *_ = region.bbox

            c[:, 0] += bbx - pad  # - minimums[1] - 1
            c[:, 1] += bby - pad  # - minimums[0] - 1

            regim = region.image
            # regen[bbx : bbx + regim.shape[0], bby : bby + regim.shape[1]] = regim
            if DEBUG:
                x_start_corr = np.clip(0 - (bbx - pad), 0, None)
                y_start_corr = np.clip(0 - (bby - pad), 0, None)

                x_start = bbx - pad - x_start_corr
                y_start = bby - pad - y_start_corr

                x_end_corr = np.clip(x_start + cs.shape[0] - img.shape[0], 0, None)
                y_end_corr = np.clip(y_start + cs.shape[1] - img.shape[1], 0, None)

                # regen[
                #     x_start : x_start + cs.shape[0] - x_end_corr,
                #     y_start : y_start + cs.shape[1] - y_end_corr,
                # ] = cs[x_start_corr : -x_end_corr or None, y_start_corr : -y_end_corr or None]

                regen[bbx : bbx + cs.shape[0] - 2 * pad, bby : bby + cs.shape[1] - 2 * pad] = cs[
                    pad:-pad, pad:-pad
                ]

                c_show = c[(c[:, 0] < img.shape[0]) & (c[:, 1] < img.shape[1])].astype(int)
                cntrs[c_show[:, 0].astype(int), c_show[:, 1].astype(int)] = 1

            _polygons.append(Polygon(c))

        if not _polygons:
            polygons.append(Polygon())
        elif len(_polygons) == 1:
            polygons.append(_polygons[0])
        else:
            polygons.append(MultiPolygon(_polygons))

    for i, region in enumerate(props):
        loop(i, region)

    logger.info(f"{idx}: building STRtree.")
    tree = STRtree(polygons)
    assert len(tree.geometries) == len(props) == len(polygons)

    # sl = np.s_[3500:4000, 2000:2500]
    if DEBUG:
        sl = np.s_[500:1000, 3500:4000]
        fig, axs = plt.subplots(figsize=(8, 4), dpi=200, ncols=3)
        axs[0].imshow(img[sl], zorder=1, origin="lower")
        axs[1].imshow(regen[sl], zorder=1, origin="lower")
        axs[2].imshow(cntrs[sl], zorder=1, vmax=1, origin="lower")
    # %%

    # %%

    # plt.hexbin(spots['x'], spots['y'], gridsize=600)

    # img
    if DEBUG:
        sl = np.s_[500:, :3000]

        # axs[0].imshow(intensity[sl], zorder=1, origin="lower")
        # axs[0].set_aspect("equal")
        # axs[1].imshow(regen[sl], zorder=1, vmax=1, origin="lower")
        # axs[1].set_aspect("equal")
        # axs[2].scatter(*filter_imshow(spots, sl), s=1, alpha=1)
        # axs[2].set_aspect("equal")
        # plt.tight_layout()

        fig, axs = plt.subplots(ncols=3, nrows=1, figsize=(12, 4), dpi=200)
        axs = axs.flatten()
        axs[0].imshow(intensity[sl], zorder=1, origin="lower")
        axs[1].imshow(img.astype(np.uint16)[sl], zorder=1, vmax=1, origin="lower")
        for i, ax in enumerate(axs):
            ax.axis("off")
            ax.scatter(*filter_imshow(spots[::10], sl), s=1, alpha=0.4)
    # %%
    points = []
    _sp = spots.drop("ii").with_row_index("ii")
    for i, (x, y) in enumerate(zip(_sp["x"], _sp["y"])):
        if i and i % 100000 == 0:
            logger.debug(f"{idx}: {i} points processed.")
        points.append(Point((y, x)))

    # Query all points at once
    matches = tree.query(points, predicate="intersects").T
    print(f"{idx}: {len(matches)} points found in {len(points)} spots.")

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

    # %%
    if DEBUG:
        fig, axs = plt.subplots(ncols=3, nrows=1, figsize=(12, 4), dpi=200)
        axs = axs.flatten()
        sl = np.s_[500:, :]
        axs[0].imshow(intensity[sl], zorder=1, origin="lower")
        axs[1].imshow(img.astype(np.uint16)[sl], zorder=1, vmax=1, origin="lower")
        for i, ax in enumerate(axs):
            # ax.axis("off")
            ax.scatter(*filter_imshow(ident, sl), s=1, alpha=0.4)
            ax.set_aspect("equal")


# plt.ylim(-5000, 0)

# %%
if __name__ == "__main__":
    run()
