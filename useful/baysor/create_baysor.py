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
from loguru import logger
from roifile import ImagejRoi
from scipy.ndimage import binary_dilation, shift
from shapely import MultiPolygon, Point, Polygon, STRtree
from skimage.filters import gaussian
from skimage.transform import rotate

from fishtools.preprocess.tileconfig import TileConfiguration

plt.rcParams["image.aspect"] = "equal"
sns.set_theme()

# %%


path = Path("/working/20241203-KWLaiOrganoidTest_sample1/baysor--all")
rois = sorted({x.stem.split("--")[1] for x in path.parent.glob("*.parquet") if "mid" not in x.name})
if not rois:
    raise ValueError("No ROIs found. Check path.")
try:
    imgs = [tifffile.imread(path.parent / f"stitch--{roi}" / "fused.tif") for roi in rois]
except FileNotFoundError as e:
    logger.warning(e)
    imgs = None

# %%


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
    plt.scatter(coords[:, 0], coords[:, 1], s=0.05, alpha=0.1, label="before")

    rotated = (centered @ rotation_matrix) + center
    spots_rotated = spots.with_columns(x=rotated[:, 0], y=rotated[:, 1])

    plt.scatter(spots_rotated["x"], spots_rotated["y"], s=0.1, alpha=0.5, label="after")
    plt.gca().set_aspect("equal")
    print("after")
    print(f"x: {spots_rotated['x'].min()}, {spots_rotated['x'].max()}")
    print(f"y: {spots_rotated['y'].min()}, {spots_rotated['y'].max()}")
    plt.show()
    return spots_rotated


# %%
# %%
from collections import defaultdict

imgs_roted = defaultdict(list)
ymax = [0]
spotss = []
centers = []
# config = json.loads((path / "config.json").read_text())
# rots = config["rots"]
rots = [54]
ori_dims = []

new_centers = []
# config["bins"] = bins

for i, roi in enumerate(rois[:3]):
    print(f"Processing {roi}")
    stitched_shift = (
        TileConfiguration.from_file(path.parent / f"stitch--{roi}" / "TileConfiguration.registered.txt")
        .df[["x", "y"]]
        .to_numpy()
        .min(axis=0)
    )

    # centers.append(center)
    # ori_dims.append(imgs[i].shape)
    # for z in range(0):
    #     imgs_roted[i].append(
    #         rotate(
    #             imgs[i][int(3.5 * i) : int(3.5 * (i + 1)), 0].max(axis=0),
    #             rots[i],
    #             center=center[::-1],
    #             order=1,
    #             preserve_range=True,
    #             clip=False,
    #             resize=True,
    #         )
    #     )
    # imgs_roted[i] = np.array(imgs_roted[i])
    # new_centers.append(np.array(imgs_roted[i].shape[1:]) / 2)
    _spots = (
        pl.scan_parquet(path.parent / f"*--{roi}.parquet")
        .collect()
        .with_columns(roi=pl.lit(i))
        .with_columns(y=(pl.col("y") - stitched_shift[1]) / 2, x=(pl.col("x") - stitched_shift[0]) / 2)
    )

    if imgs is None:
        center = np.array([
            (_spots["y"].max() - _spots["y"].min()) / 2,
            (_spots["x"].max() - _spots["x"].min()) / 2,
        ])
    # print(_spots['y'].min())
    # spotss.append(_spots)
    spotss.append(rotate_points(_spots, rots[i], center[::-1]))
# %%

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
plt.scatter(*spots[::100][["x", "y"]].to_numpy().T, s=0.1, alpha=0.1)
plt.gca().set_aspect("equal")
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
    is3d = len(canvas.shape) == 3
    print(f"Is 3D: {is3d}")

    # Calculate destination coordinates on canvas
    dst_y_start = shift_y
    dst_x_start = shift_x
    dst_y_end = shift_y + image.shape[is3d + 0]
    dst_x_end = shift_x + image.shape[is3d + 1]

    # Calculate source coordinates (what part of the image will be visible)
    src_y_start = 0
    src_x_start = 0
    src_y_end = image.shape[is3d + 0]
    src_x_end = image.shape[is3d + 1]

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

    if dst_y_end > canvas.shape[is3d + 0]:
        src_y_end -= dst_y_end - canvas.shape[is3d + 0]
        dst_y_end = canvas.shape[is3d + 0]
    if dst_x_end > canvas.shape[is3d + 1]:
        src_x_end -= dst_x_end - canvas.shape[is3d + 1]
        dst_x_end = canvas.shape[is3d + 1]

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
        if len(canvas.shape) == 3:
            print("3D")
            canvas[:, dst_y_start:dst_y_end, dst_x_start:dst_x_end] = np.max(
                np.stack([
                    image[:, src_y_start:src_y_end, src_x_start:src_x_end].astype(canvas.dtype),
                    canvas[:, dst_y_start:dst_y_end, dst_x_start:dst_x_end],
                ]),
                axis=0,
            )
        elif len(canvas.shape) == 2:
            canvas[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = image[
                src_y_start:src_y_end, src_x_start:src_x_end
            ].astype(canvas.dtype)
        else:
            raise ValueError(f"Unknown canvas shape {canvas.shape}")

    return canvas


img_out = np.zeros(
    (
        10,
        *(
            spots[["y", "x"]].to_numpy().max(axis=0).astype(int)
            - np.min(np.concatenate([[0, 0], spots[["y", "x"]].to_numpy().min(axis=0).astype(int)], axis=0))
            + 1
        ),
    ),
    dtype=np.uint16,
)
print(img_out.shape)

offsets = np.array(new_centers) - np.array(centers)
for i, img in enumerate(imgs[:3]):
    print(f"copying {i}")
    img_out = copy_with_offsets(img_out, imgs_roted[i], [-offsets[i, 0] - mins_y[i], bins[i] - offsets[i, 1]])

# %%
# img_out = copy_with_offsets(img_out, img_rotated, [])
fig, axs = plt.subplots(figsize=(8, 6), ncols=2, dpi=200)
axs[0].imshow(
    img_out[0, ::4, ::4],
    origin="lower",
    zorder=1,
    vmin=np.percentile(img_out, 30),
    vmax=np.percentile(img_out, 99.9),
)
# axs[0].scatter(*spots[["x", "y"]].to_numpy()[::20].T, s=0.1, alpha=0.8, c="white")
for ax in axs:
    if not ax.has_data():
        fig.delaxes(ax)
# axs[1].imshow(
#     img_out[1],
#     origin="lower",
#     zorder=1,
#     vmin=np.percentile(img_out, 30),
#     vmax=np.percentile(img_out, 99.9),
# )


# %%
tifffile.imwrite(path / "edu.tif", img_out, compression=22610, compressionargs={"level": 0.75})
# %%
tifffile.imwrite(path / "coords.tif", img_out.max(axis=0), compression=22610, compressionargs={"level": 0.75})
# %%


def parse_roi(ij: ImagejRoi):
    c = ij.integer_coordinates
    c[:, 0] += ij.left
    c[:, 1] += ij.top
    return c


rois = ImagejRoi.fromfile(path / "RoiSet.roi")  # type: ignore
if isinstance(rois, list):
    polygons = [Polygon(parse_roi(r)) for r in rois]
else:
    polygons = [Polygon(parse_roi(rois))]

# %%
tree = STRtree(polygons)
u = tree.query(list(map(lambda t: Point([t[0], t[1]]), spots[["x", "y"]].to_numpy())), predicate="intersects")
res = np.array(u[0])

# %%
fig, ax = plt.subplots(figsize=(4, 8), dpi=100)
ax.scatter(spots[res, "x"], spots[res, "y"], s=1, alpha=0.3)
ax.set_aspect("equal")

# %%
spots.select(x="x", y="y", z="z", gene=pl.col("target").str.split("-").list.get(0)).write_csv(
    path / "spots.csv"
)


# %%
spots[res].select(x="x", y="y", z="z", gene=pl.col("target").str.split("-").list.get(0)).write_csv(
    path / "spots.cortex.csv"
)

# %%
import shapely

geo = json.loads((path / "segmentation.cortex_polygons_3d.json").read_text())
# %%
it = iter(geo)
for i in range(100):
    print(next(it))


# %%
import numpy as np
from rasterio.features import rasterize
from shapely.geometry import shape

# def geojson_to_labels(geojson_data, width, height, bounds=None):
#     # Create (geometry, id) pairs
#     shapes = [
#         (shape(feature["geometry"]), int(feature["id"].rsplit("-", 1)[1]))
#         for feature in geojson_data["features"]
#     ]


def pixel_geojson_to_labels(geojson_data, width, height):
    # Create (geometry, id) pairs, assuming coordinates are already pixels
    shapes = [
        (shape(feature["geometry"]), int(feature["id"].rsplit("-", 1)[1]))
        for feature in geojson_data["features"]
    ]

    # Rasterize directly without transform
    labels = rasterize(shapes, out_shape=(height, width), fill=0, dtype=np.uint32)

    return labels


labels = [
    pixel_geojson_to_labels(feats, width=img_out.shape[2], height=img_out.shape[1]) for feats in geo.values()
]

# %%
labels = np.array(labels)
# %%
tifffile.imwrite(path / "labels.tif", labels, compression="zlib", metadata={"axes": "ZYX"})
# %%

# %%
cfse = tifffile.imread(path / "cfse.tif")

# %%
from skimage.measure import regionprops_table

props = regionprops_table(
    labels, cfse, properties=["label", "centroid", "area", "intensity_mean", "intensity_max"]
)
# %%
import polars as pl

df = pl.DataFrame(props).with_columns(intensity_sum=pl.col("intensity_mean") * pl.col("area"))
plt.hist(df["intensity_sum"], log=True)

# %%
df.write_csv(path / "cfse_stats.csv")
# %%
