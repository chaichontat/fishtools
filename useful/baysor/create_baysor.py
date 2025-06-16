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


path = Path("/working/20250612_ebe00219_3/analysis/deconv/baysor")
rois = sorted({
    x.stem.split("+")[0]
    for x in path.parent.glob("*.parquet")
    if "old" not in x.name and x.stem.split("+").__len__() == 2
})
print()
if not rois:
    raise ValueError("No ROIs found. Check path.")
try:
    imgs = None  # [tifffile.imread(path.parent / f"stitch--{roi}" / "fused.tif") for roi in rois]
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
# rots = [54]
ori_dims = []

new_centers = []
# config["bins"] = bins
all_spots = []
# Initialize for X-axis offsetting
next_roi_start_x = 0.0
x_padding = 0.0
rots = [-68, 6]

for i, roi in enumerate(rois):
    print(f"Processing {roi}")
    stitched_shift = (
        TileConfiguration.from_file(
            path.parent / f"stitch--{roi.split('+')[0]}" / "TileConfiguration.registered.txt"
        )
        .df[["x", "y"]]
        .to_numpy()
        .min(axis=0)
    )

    # center = np.array(imgs[i].shape[-2:]) / 2

    # centers.append(center)
    # ori_dims.append(imgs[i].shape)
    # for z in range(10):
    #     imgs_roted[i].append(
    #         rotate(
    #             imgs[i][int(imgs[i].shape[0] / 10 * i) : int(imgs[i].shape[0] / 10 * (i + 1)), 0].max(axis=0),
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
        pl.scan_parquet(path.parent / f"{roi}+*.parquet")
        .collect()
        .with_columns(roi=pl.lit(i))
        .with_columns(
            y=(pl.col("y") - stitched_shift[1]) / 2 * 0.216, x=(pl.col("x") - stitched_shift[0]) / 2 * 0.216
        )
    )

    if _spots.height > 0 and "x" in _spots.columns and "y" in _spots.columns and rots[i] != 0:
        # Calculate center of the current ROI for rotation (before x-offsetting)
        # Ensure min/max return non-null values if there are spots
        min_x_roi, max_x_roi = _spots["x"].min(), _spots["x"].max()
        min_y_roi, max_y_roi = _spots["y"].min(), _spots["y"].max()

        if not (min_x_roi is None or max_x_roi is None or min_y_roi is None or max_y_roi is None):
            center_x_roi = (min_x_roi + max_x_roi) / 2
            center_y_roi = (min_y_roi + max_y_roi) / 2
            rotation_center_roi = np.array([center_x_roi, center_y_roi])

            print(f"Rotating ROI {i} by {rots[i]} degrees around {rotation_center_roi}")
            # Rotate the spots for the current ROI
            # Note: rotate_points has plt.show(), which will pause execution for each ROI.
            # You might want to modify rotate_points to not show plots in a loop.
            _spots = rotate_points(_spots, rots[i], rotation_center_roi)
        else:
            print(f"Skipping rotation for ROI {i} due to missing coordinate bounds (empty or all nulls).")

    # Offset ROI along X-axis to prevent overlap
    if _spots.height > 0 and "x" in _spots.columns and _spots["x"].is_not_null().any():
        original_min_x = _spots["x"].min()

        # Shift current ROI's X coordinates to start at next_roi_start_x
        _spots = _spots.with_columns(x=(pl.col("x") - original_min_x + next_roi_start_x))

        # Update next_roi_start_x for the subsequent ROI
        current_max_x = _spots["x"].max()
        if current_max_x is not None:
            next_roi_start_x = current_max_x + x_padding
        else:
            # Fallback if max_x is None (e.g., all x values became null)
            # Advance by padding from where this ROI was supposed to start
            next_roi_start_x = next_roi_start_x + x_padding
    else:
        # If ROI is empty or has no valid x-coordinates,
        # advance next_roi_start_x by padding to leave a gap.
        next_roi_start_x += x_padding

    spotss.append(_spots)

    if imgs is None:
        center = np.array([
            (_spots["y"].max() - _spots["y"].min()) / 2,
            (_spots["x"].max() - _spots["x"].min()) / 2,
        ])
    print(_spots["y"].min())
    # spotss.append(rotate_points(_spots, rots[i], center[::-1]))
# %%

spots = pl.concat(spotss)
# shifts_from_rotation = np.concatenate([[0], np.cumsum((np.array(new_centers) - np.array(centers))[:, 1])])

bins = np.concatenate([[0], np.cumsum(spots.group_by("roi").agg(pl.col("x").max()).sort("roi")["x"])])
# bins += shifts_from_rotation
bins += np.arange(0, len(bins)) * 200
mins_y = spots.group_by("roi").agg(pl.col("y").min()).sort("roi")["y"]

spots = spots.with_columns(
    # x=pl.col("x") + pl.col("roi").map_elements(bins.__getitem__, return_dtype=pl.Float64),
    y=pl.col("y") - pl.col("roi").map_elements(mins_y.__getitem__, return_dtype=pl.Float64),
)
plt.scatter(*spots[::100][["x", "y"]].to_numpy().T, s=0.1, alpha=0.1)
plt.gca().set_aspect("equal")
# %%
spots_merged = pl.concat([
    spots.filter(pl.col("roi") == 0),
    spots.filter(pl.col("roi") == 1).with_columns(
        y=pl.col("y") + spots.group_by("roi").agg(pl.col("y").max())[0, "y"] + 100
    ),
])

# %%


def plot(ax, spots, gene: str):
    filtered = spots.filter(pl.col("target").eq(gene))
    ax.scatter(
        filtered["x"],
        filtered["y"],
        s=5,
        alpha=1,
        label=gene,
    )


fig, ax = plt.subplots(figsize=(8, 6))
plot(ax, spots.filter(pl.col("z").is_between(7, 9)), "SLC17A7-201")
plot(ax, spots.filter(pl.col("z").is_between(7, 9)), "GAD1-202")
ax.set_xlim(700, 1200)
ax.set_ylim(2500, 3500)

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
path.mkdir(exist_ok=True, parents=True)
spots.select(x="x", y="y", z="z", gene=pl.col("target").str.split("-").list.get(0)).filter(
    pl.col("gene") != "Blank"
).write_csv(path / "spots.csv")


# %%
spots.select(x="x", y="y", z="z", gene=pl.col("target").str.split("-").list.get(0)).write_csv(
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
# %%

labels = tifffile.imread(path / "labels.tif")
# %%
from skimage.measure import regionprops_table

props = regionprops_table(labels, properties=["label", "centroid", "area", "coords"])
df_props = pl.DataFrame(props)
# props = regionprops_table(
#     labels, cfse, properties=["label", "centroid", "area", "intensity_mean", "intensity_max"]
# )
# %%

import numpy as np
from scipy.spatial import ConvexHull, QhullError

# First store all hulls
hull_vertices_by_label = {}
for row in df_props.iter_rows(named=True):
    coords = row["coords"]
    if len(coords) >= 4:
        try:
            hull = ConvexHull(coords)
            hull_vertices_by_label[row["label"]] = coords[hull.vertices]
        except QhullError:
            ...

# Save to file - choose format based on your needs
# np.save('hull_vertices.npy', hull_vertices_by_label)
# or
# Save as CSV/txt if you prefer
# for label, vertices in hull_vertices_by_label.items():
#     np.savetxt(f'hull_vertices_{label}.txt', vertices)
# %%
import polars as pl

df = pl.DataFrame(props).with_columns(intensity_sum=pl.col("intensity_mean") * pl.col("area"))
plt.hist(df["intensity_sum"], log=True)

# %%
df.write_csv(path / "cfse_stats.csv")
# %%
from json import JSONEncoder

import numpy


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


# %%
