# %%
import os

# %%
from pathlib import Path

# os.environ["JAVA_TOOL_OPTIONS"] = "-Djava.net.useSystemProxies=true"
import imagej

# # %%
# from pathlib import Path
# %%
# # %%
import pandas as pd
import polars as pl
import scyjava

# df = pl.read_csv(Path("/raid/data/star/sagittal/positions.csv"), has_header=False).to_numpy()
# df += -df.min(axis=0)
# df * 2048 / 200 * 0.9


scyjava.config.add_option("-Xmx220g")
ij = imagej.init("/home/chaichontat/Fiji.app")
assert ij
# %%
path = "/raid/data/3t3edu2"
options = "subpixel_accuracy"
fusion = "Average"
macro = f"""
run("Memory & Threads...", "parallel=64");
run("Grid/Collection stitching", "type=[Positions from file] order=[Defined by TileConfiguration] directory={path} layout_file=TileConfiguration.txt fusion_method=[{fusion}] regression_threshold=0.3 max/avg_displacement_threshold=5 absolute_displacement_threshold=10 compute_overlap {options} computation_parameters=[Save computation time (but use more RAM)] image_output=[Write to disk] output_directory={str(path)}");
"""

import pandas as pd

# ij.py.run_macro(macro)

PATH = Path("/raid/data/equiphi")
df = pd.read_csv(PATH / "4_29_30.csv", header=None)

files_idx = sorted([int(x.stem.split("_")[-2].split("-")[1]) for x in PATH.glob("4_29_30-*_max.tif")])
# df = pd.read_csv(r"D:\MERSCOPEDATA\202310101534_EBramel-Merfish-Aorta-B3348_VMSC06101\settings\positions.csv", header=None)[:50]
# # %%
actual = 2048 * 0.108
scaling = 200 / actual
adjusted = pd.DataFrame(
    dict(y=(df[0] - df[0].min()), x=(df[1] - df[1].min())),  # / 200 * 2048,
    dtype=int,
).iloc[files_idx]
adjusted["x"] -= adjusted["x"].min()
adjusted["x"] *= -(1 / 200) * 2048 * scaling
adjusted["y"] -= adjusted["y"].min()
adjusted["y"] *= -(1 / 200) * 2048 * scaling

ats = adjusted.copy()
ats["x"] -= ats["x"].min()
ats["y"] -= ats["y"].min()

from itertools import cycle, islice

with open("/raid/data/equiphi/TileConfiguration.txt", "w") as f:
    f.write("dim=2\n")
    for i, (idx, row) in enumerate(islice(cycle(ats.iterrows()), 3 * len(ats))):
        f.write(f"{i}; ; ({row['x']}, {row['y']})\n")


# %%

# for i in range(40):
#     img = image_reader(rf"D:\MERSCOPEDATA\202310101534_EBramel-Merfish-Aorta-B3348_VMSC06101\data\stack_prestain_{i:03d}.dax")
#     tifffile.imwrite(fr"D:\MERSCOPEDATA\202310101534_EBramel-Merfish-Aorta-B3348_VMSC06101\data\stack_prestain_{i:03d}.tif", img, imagej=True, metadata={"axes": "ZYX"}, compression=22610)
# %%
import numpy as np

# %%
import tifffile

for file in files_idx:
    img = tifffile.imread(PATH / f"edu-{file:03d}_max.tif")
    # assert len(img.shape) ==
    tifffile.imwrite(
        PATH / f"edu-{file:03d}_max.tif", img, compression=22610, metadata={"axes": "CYX"}, imagej=True
    )

# %%
# import matplotlib.pyplot as plt


ymax = adjusted.y.max() + 2048
c = 5
out = np.zeros((c, ymax, adjusted.x.max() + 2048), dtype=np.uint16)
# %%

for i, row in adjusted.iterrows():
    try:
        img = tifffile.imread(PATH / f"edu-{i:03d}.tif")
        fid = img[-1]
        img = img[:-1].reshape(-1, c, 2048, 2048)
    except FileNotFoundError:
        continue
    else:
        print(i)
        img = img.max(axis=0)
        tifffile.imwrite(
            PATH / f"edu-{i:03d}_max.tif", np.concatenate([img, fid[None, :]]), compression=22610
        )
        # img = image_reader(rf"D:\MERSCOPEDATA\202310101534_EBramel-Merfish-Aorta-B3348_VMSC06101\data\stack_prestain_{i:03d}.dax")
        # out[:, row.y : row.y + 2048, row.x : row.x + 2048] = np.flip(img, axis=(1, 2))
# %%
# tifffile.imwrite('mos.tif', out, imagej=True)

tifffile.imwrite(
    PATH / "big_edu.tif",
    (np.clip(out[[0, 1, 3, 4]][:, ::2, ::2], 0, 32767) // 128).astype(np.uint8),
)

# %%


# %%
# import numpy as np
# import skimage.measure

# # y = skimage.measure.block_reduce(out[1], (2, 2), np.max)

# # %%
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np


# sns.set()

# fig, ax = plt.subplots(figsize=(16, 16))
# ax.imshow(out[0, ::4, ::4])
# ax.axis("off")

# # %%
# fig, ax = plt.subplots(figsize=(10, 10))
# ax.imshow((out[1, ::4, ::4] // 64).astype(np.uint8))
# ax.axis("off")

# # %%
# small = out[:, ::4, ::4]
# # %%
# fig, axs = plt.subplots(figsize=(14, 8), ncols=2, dpi=150)

# axs[0].imshow(small[0])
# axs[1].imshow(np.minimum(small[1] - 60, small[1]))
# axs[0].axis("off")
# axs[1].axis("off")
# axs[0].set_title("Malat1")
# axs[1].set_title("Cux2")
# for i, row in adjusted.iterrows():
#     axs[1].text(row.x // 4, row.y // 4, i, color="white", fontsize=4)
# # %%
# img = tifffile.imread(rf"/raid/data/star/sagittal/4_5_rna-090.tif_max")
# fig, axs = plt.subplots(figsize=(14, 8), ncols=2, dpi=150)

# for ax, i in zip(axs, img):
#     ax.imshow(i)

# # %%
# %%

# df = pd.read_csv(r"D:\MERSCOPEDATA\202310101534_EBramel-Merfish-Aorta-B3348_VMSC06101\settings\positions.csv", header=None)[:50]
# # %%
adjusted = df.iloc[files_idx]
# %%
{
    "index": 0,
    "file": "FCF_CSMH__54383_20121206_35_C3_zb15_zt01_63X_0-0-0_R1_L086_20130108192758780.lsm.tif",
    "position": [0.0, 0.0, 0.0],
    "size": [991, 992, 880],
    "pixelResolution": [0.097, 0.097, 0.18],
    "type": "GRAY16",
},
import json


def gen_file(idx: int, file: str, pos: list[float], size: list[int], res: list[float]):
    return {
        "index": idx,
        "file": file,
        "position": pos,
        "size": size,
        "pixelResolution": res,
        "type": "GRAY16",
    }


# files_idx = sorted([int(x.stem.split("_")[-2].split("-")[1]) for x in PATH.glob("edu-*_max.tif")])
(PATH / "ch0.json").write_text(
    json.dumps(
        [
            gen_file(
                int(i),
                f"/raid/data/3t3edu2/edu-{i:03d}_max_polyA.tif",
                [float(row["x"]), float(row["y"])],
                [2048, 2048],
                [0.108, 0.108],
            )
            for (i, row) in adjusted.iterrows()
        ]
    )
)

# %%
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

plt.imshow(tifffile.imread("/raid/data/3t3edu2/ch0-flatfield/T.tif"))
import io

import numpy as np

# %%
import zarr

# %%
store = zarr.N5Store("/raid/data/3t3edu2/export.n5")

# %%
np.frombuffer(io.BytesIO(store.get("c0/s0/6/6")).getbuffer(), dtype=np.uint16)
# store.get("c0/s0/6/6")

# %%
store.get(".zgroup")


# %%
import json

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set()


t = json.loads(
    Path("/home/chaichontat/stitching-spark/startup-scripts/ch0/iter0/ch0-n5-stitched.json").read_text()
)


# %%
plt.scatter(*np.array([x["position"] for x in t]).T)

# %%
"""This module wraps the Grid/Collection stitching plugin in ImageJ (FIJI)
using the command line headless interface of ImageJ2"""

import errno
import os

# %%
import re
import stat
import subprocess as sp
from datetime import datetime

# import requests as rq
from zipfile import ZipFile

import pandas as pd
import tifffile
from skimage.io import imread

coords = (
    Path("/home/chaichontat/stitching-spark/startup-scripts/ch0/TileConfiguration.registered.txt")
    .read_text()
    .splitlines()
)

out = []
for c in coords:
    if f := re.match(r"^([\w\-_\.]+);\s;\s\(([\d\-\.]+),\s([\d\-\.]+)\)$", c):
        out.append((f.groups()))

out = pd.DataFrame(out, columns=["file", "x", "y"])
out = out.astype({"x": float, "y": float})
out = out[(out["x"] != 0) & (out["y"] != 0)]

# %%
plt.scatter(out["x"], out["y"])
# %%
from scipy.ndimage import shift

out["x"].min(), out["x"].max()
# %%
out["y"].min(), out["y"].max()
# %%


def stitch_images_subpixel(path: str | Path, df: pd.DataFrame):
    """
    Stitch multiple images together with subpixel precision and handle overlaps using linear blending.

    Args:
    images (list of np.array): list of images as numpy arrays
    positions (list of tuple): list of positions where each image should be placed

    Returns:
    np.array: stitched image
    """
    path = Path(path)
    df = df.copy()
    df["x"] -= df["x"].min()
    df["y"] -= df["y"].min()

    # Calculate the size of the output image
    size = (int(df["y"].max()) + 1 + 2048, int(df["x"].max()) + 1 + 2048)

    # Create an empty array to hold the stitched image and a mask for blending
    stitched_image = np.zeros(size, dtype=np.float32)
    blend_mask = np.zeros(size, dtype=np.float32)
    ones = np.ones((2046, 2046), dtype=np.float32)

    # Add each image to the stitched image at the specified position
    for _, row in df.iterrows():
        print(row["file"])
        img = tifffile.imread(path / row["file"])
        x, y = int(row["x"]), int(row["y"])
        dx, dy = x - row["x"], y - row["y"]
        transformed_image = shift(img, (dy, dx), order=1, mode="constant", cval=0.0)

        # Crop the transformed image by 1 pixel on all sides
        transformed_image = transformed_image[1:-1, 1:-1]
        x += 1
        y += 1

        # # Create a mask for the current image
        # current_mask = np.zeros((max_y, max_x), dtype=np.float32)
        # current_mask[
        #     int(pos[0]) + 1 : int(pos[0]) + transformed_image.shape[0] + 1,
        #     int(pos[1]) + 1 : int(pos[1]) + transformed_image.shape[1] + 1,
        # ] = 1.0

        # Add the current image and its mask to the stitched image and blend mask
        stitched_image[
            y : y + transformed_image.shape[0],
            x : x + transformed_image.shape[1],
        ] += transformed_image
        blend_mask[
            y : y + transformed_image.shape[0],
            x : x + transformed_image.shape[1],
        ] += ones

    # Divide the stitched image by the blend mask to get the final blended image
    stitched_image = np.where(blend_mask != 0, stitched_image / (blend_mask + 1e-10), stitched_image)

    return stitched_image


# %%
res = stitch_images_subpixel("/raid/data/3t3edu2", out[:5])
# %%
plt.imshow(res[::4, ::4])
# %%
tifffile.imwrite("tstich.tif", res.astype(np.uint16))
# %%
