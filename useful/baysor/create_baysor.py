# %%
import json
from collections.abc import Collection
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import scanpy as sc
import seaborn as sns
import tifffile
from roifile import ImagejRoi
from scipy.ndimage import binary_dilation, shift
from shapely import MultiPolygon, Point, Polygon, STRtree
from skimage.filters import gaussian
from skimage.transform import rotate

from fishtools.preprocess.tileconfig import TileConfiguration

plt.rcParams["image.aspect"] = "equal"
sns.set_theme()

# %%


path = Path("/working/20241122-ZNE172-Trc3/baysor--all")
rois = sorted({x.stem.split("--")[1] for x in path.parent.glob("*.parquet") if "mid" not in x.name})
imgs = [tifffile.imread(path.parent / f"stitch--{roi}" / "fused.tif").max(axis=0) for roi in rois]

# %%


import numpy as np


def get_new_dimensions(shape: tuple[int, int], angle_degrees: float) -> tuple[int, int]:
    height, width = shape
    # Convert degrees to radians
    angle_rad = np.abs(np.deg2rad(angle_degrees))

    # Calculate new width and height
    new_width = np.ceil(width * np.abs(np.cos(angle_rad)) + height * np.abs(np.sin(angle_rad)))
    new_height = np.ceil(width * np.abs(np.sin(angle_rad)) + height * np.abs(np.cos(angle_rad)))

    return (int(new_height), int(new_width))


def rotate_points(spots: pl.DataFrame, deg: float, center: np.ndarray, *, divide: bool = True):
    coords = spots[["x", "y"]].to_numpy()
    # if divide:
    #     coords /= 2

    θ = np.radians(deg)
    c, s = np.cos(θ), np.sin(θ)
    # Standard counterclockwise rotation matrix
    rotation_matrix = np.array([[c, -s], [s, c]])

    # Center, rotate, and uncenter points
    centered = coords - center
    print(f"before")
    print(f"x: {spots['x'].min()}, {spots['x'].max()}")
    print(f"y: {spots['y'].min()}, {spots['y'].max()}")
    print(center)
    plt.scatter(coords[:, 0], coords[:, 1], s=0.1, alpha=0.1)

    rotated = (centered @ rotation_matrix) + center
    spots_rotated = spots.with_columns(x=rotated[:, 0], y=rotated[:, 1])

    plt.scatter(spots_rotated["x"], spots_rotated["y"], s=0.1, alpha=0.1)
    print("after")
    print(f"x: {spots_rotated['x'].min()}, {spots_rotated['x'].max()}")
    print(f"y: {spots_rotated['y'].min()}, {spots_rotated['y'].max()}")
    plt.show()
    return spots_rotated


# %%
# center = np.array(preview[1].shape) / 2
# spots_roted = rotate_points(spots, rots[1], center)
# preview_roted = rotate(preview[1], rots[1], center=center[::-1], order=1, preserve_range=True, clip=False).T
# plt.imshow(preview_roted, zorder=1, origin="lower", vmax=np.percentile(preview_roted, 99.9))
# plt.scatter(*spots_roted[::10][["y", "x"]].to_numpy().T, s=0.1, alpha=0.1, c="white")


# %%
imgs_roted = []
ymax = [0]
spotss = []
centers = []
config = json.loads((path / "config.json").read_text())
rots = config["rots"]
rots = [7, 155, 122]
ori_dims = []

new_centers = []
# config["bins"] = bins

for i, roi in enumerate(rois[:3]):
    stitched_shift = (
        TileConfiguration.from_file(path.parent / f"stitch--{roi}" / "TileConfiguration.registered.txt")
        .df[["x", "y"]]
        .to_numpy()
        .min(axis=0)
    )

    center = np.array(imgs[i].shape[1:]) / 2
    centers.append(center)
    ori_dims.append(imgs[i].shape)

    imgs_roted.append(
        rotate(
            imgs[i][2],
            rots[i],
            center=center[::-1] + 2000,
            order=1,
            preserve_range=True,
            clip=False,
            resize=True,
        )
    )
    new_centers.append(np.array(imgs_roted[i].shape) / 2)
    _spots = (
        pl.scan_parquet(path.parent / f"*--{roi}.parquet")
        .collect()
        .with_columns(roi=pl.lit(i))
        .with_columns(y=(pl.col("y") - stitched_shift[1]) / 2, x=(pl.col("x") - stitched_shift[0]) / 2)
    )
    # print(_spots['y'].min())
    # spotss.append(_spots)
    spotss.append(rotate_points(_spots, rots[i], center[::-1]))

spots = pl.concat(spotss)
shifts_from_rotation = np.concatenate([[0], np.cumsum((np.array(new_centers) - np.array(centers))[:, 1])])

bins = np.concatenate([[0], np.cumsum(spots.group_by("roi").agg(pl.col("x").max()).sort("roi")["x"])])
bins += shifts_from_rotation
bins += np.arange(0, len(bins)) * 200
mins_y = spots.group_by("roi").agg(pl.col("y").min()).sort("roi")["y"]

spots = spots.with_columns(
    x=pl.col("x") + pl.col("roi").map_elements(bins.__getitem__, return_dtype=pl.Float64),
    y=pl.col("y") - pl.col("roi").map_elements(mins_y.__getitem__, return_dtype=pl.Float64),
)
plt.scatter(*spots[::10][["x", "y"]].to_numpy().T, s=0.1, alpha=0.1)
# %%


def copy_with_offsets(canvas: np.ndarray, image: np.ndarray, shifts: Collection[int]):
    """
    Copy image onto canvas with specified shift from image's origin (0,0).
    The image can be placed anywhere on the canvas.

    Args:
        canvas: The destination array to paste into
        image: The source image to copy from
        shifts: Collection containing (shift_y, shift_x) where:
               shift_y: How much to shift image up (negative) or down (positive)
               shift_x: How much to shift image left (negative) or right (positive)
    """
    print(canvas.shape, image.shape)
    shift_y, shift_x = np.round(shifts).astype(int)

    # Calculate destination coordinates on canvas
    dst_y_start = shift_y
    dst_x_start = shift_x
    dst_y_end = shift_y + image.shape[0]
    dst_x_end = shift_x + image.shape[1]

    # Calculate source coordinates (what part of the image will be visible)
    src_y_start = 0
    src_x_start = 0
    src_y_end = image.shape[0]
    src_x_end = image.shape[1]

    print(f"Before adjustment:")
    print(f"Dst coords: y({dst_y_start}:{dst_y_end}), x({dst_x_start}:{dst_x_end})")
    print(f"Src coords: y({src_y_start}:{src_y_end}), x({src_x_start}:{src_x_end})")

    # Adjust if parts of the image would fall outside the canvas
    if dst_y_start < 0:
        src_y_start = -dst_y_start
        dst_y_start = 0
    if dst_x_start < 0:
        src_x_start = -dst_x_start
        dst_x_start = 0
    if dst_y_end > canvas.shape[0]:
        src_y_end -= dst_y_end - canvas.shape[0]
        dst_y_end = canvas.shape[0]
    if dst_x_end > canvas.shape[1]:
        src_x_end -= dst_x_end - canvas.shape[1]
        dst_x_end = canvas.shape[1]

    print(f"After adjustment:")
    print(f"Dst coords: y({dst_y_start}:{dst_y_end}), x({dst_x_start}:{dst_x_end})")
    print(f"Src coords: y({src_y_start}:{src_y_end}), x({src_x_start}:{src_x_end})")

    # Only copy if there's any part of the image that would appear on the canvas
    if (
        dst_y_end > dst_y_start
        and dst_x_end > dst_x_start
        and src_y_end > src_y_start
        and src_x_end > src_x_start
    ):
        canvas[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = image[
            src_y_start:src_y_end, src_x_start:src_x_end
        ]

    return canvas


img_out = np.zeros(
    (
        spots[["y", "x"]].to_numpy().max(axis=0).astype(int)
        - spots[["y", "x"]].to_numpy().min(axis=0).astype(int)
        + 1
    ),
    dtype=np.uint16,
)
print(img_out.shape)

offsets = np.array(new_centers) - np.array(centers)
for i, img in enumerate(imgs[:3]):
    print(f"copying {i}")
    img_out = copy_with_offsets(img_out, imgs_roted[i], [-offsets[i, 0] - mins_y[i], bins[i] - offsets[i, 1]])

# img_rotated = (img_rotated * preview[1].max() / img_rotated.max()).astype(np.uint16)

# %%
# img_out = copy_with_offsets(img_out, img_rotated, [])
fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
ax.imshow(
    img_out,
    origin="lower",
    zorder=1,
    vmin=np.percentile(img_out, 30),
    vmax=np.percentile(img_out, 99.9),
)

# ax.scatter(*spots[["x", "y"]].to_numpy()[::20].T, s=0.01, alpha=0.8, c="white")

# ax.set_xlim(3000, 3500)
# ax.set_ylim(1000, 2000)

# %%
tifffile.imwrite(path / "coords.tif", img_out, compression=22610, compressionargs={"level": 0.75})

# %%
# ymax.append(float(_spots["y"].max()) + ymax[i] + 200)
# spotss.append(_spots.with_columns(y=pl.col("y") + ymax[i]))

# %% Create coords image to draw ROIs
# out = np.zeros([int(rotated_yx[:, 0].max() + 3), int(rotated_yx[:, 1].max() + 3)], dtype=np.uint8)
# coords = np.round(yx).astype(int)

# # Create offsets and extend coordinates
# offsets = np.array([(i, j) for i in [-1, 0, 1] for j in [-1, 0, 1]])
# extended_coords = coords[:, None] + offsets[None, :]
# extended_coords = extended_coords.reshape(-1, 2)
# # Vectorized bounds checking
# mask = (extended_coords >= 0).all(axis=1) & (extended_coords < np.array(out.shape)[None, :]).all(axis=1)
# out[extended_coords[mask, 0], extended_coords[mask, 1]] = 255

# out[extended_coords[:, 0], extended_coords[:, 1]] = 255

# %%spots = spots.with_columns(y=yx[:, 0], x=yx[:, 1]).write_csv(path / "spots.rotated.csv")
# %%


# %%

# img = tifffile.imread(path.parent / f"stitch--{rois[0]}/fused.tif")

# %%

# %%
import numpy as np

plt.rcParams["figure.facecolor"] = "black"
show = img[:, 0, ::2, ::2].max(axis=0)
plt.imshow(show, zorder=1, vmax=np.percentile(show, 99.9), vmin=np.percentile(show, 30))


# %%

# %%

# %%
