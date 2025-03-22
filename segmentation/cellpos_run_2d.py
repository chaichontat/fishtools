# %%
import logging
import pickle
import subprocess
import time
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import rich_click as click
import tifffile
from cellpose.io import imread, load_train_test_data
from cellpose.models import CellposeModel
from cellpose.train import train_seg
from loguru import logger
from tifffile import imread

imshow = partial(plt.imshow, zorder=1)
path = Path("/working/20250205_3t3BrduUntreated/analysis/deconv")
# "/mnt/working/lai/segment--left/models/2024-11-02_16-30-54"


logging.basicConfig(level=logging.INFO)

# path = Path("/mnt/working/e155trcdeconv/segment3d--left")
custom_models = path / "segment--left+dapieduwga/models"
# %%

# name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
img = imread(path / "stitch--left+dapieduwga/fused.tif")
# %%
wga_max = img[:3, 2].max(axis=0)

# %%
valid = wga_max[wga_max > 0]
p1, p99 = np.percentile(valid, 1), np.percentile(valid, 99)
# %%
# %%

# %%


wga_max = np.clip(wga_max, p1, p99)
wga_max = (wga_max - wga_max.min()).astype(np.float32) / (wga_max.max() - wga_max.min())


# Load image (assuming you have an image array)
# highpass = original - lowpass
# def highpass_filter(image, sigma=1):
#     lowpass = ndimage.gaussian_filter(image, sigma=sigma)
#     highpass = image - lowpass
#     return highpass


# wga_max = highpass_filter(wga_max.astype(np.float32), 10)


# %%
# Yes, the typing is wrong.
# model = CellposeModel(gpu=True, pretrained_model=cast(bool, models[0].as_posix()))
if custom_models:
    models = sorted(
        (file for file in custom_models.glob("*") if "train_losses" not in file.name),
        key=lambda x: x.stat().st_mtime,
        reverse=True,
    )
    assert models

# normalize (bool, optional): if True, normalize data so 0.0=1st percentile and 1.0=99th percentile of image intensities in each channel;
#     can also pass dictionary of parameters (all keys are optional, default values shown):
#         - "lowhigh"=None : pass in normalization values for 0.0 and 1.0 as list [low, high] (if not None, all following parameters ignored)
#         - "sharpen"=0 ; sharpen image with high pass filter, recommended to be 1/4-1/8 diameter of cells in pixels
#         - "normalize"=True ; run normalization (if False, all following parameters ignored)
#         - "percentile"=None : pass in percentiles to use as list [perc_low, perc_high]
#         - "tile_norm"=0 ; compute normalization in tiles across image to brighten dark areas, to turn on set to window size in pixels (e.g. 100)
#         - "norm3D"=False ; compute normalization across entire z-stack rather than plane-by-plane in stitching mode.
#     Defaults to True.
normalize_params = {
    # "lowhigh": [np.percentile(valid, 1), np.percentile(valid, 99)],
    # "percentile": None,  # (0.1, 99.9),
    # "normalize": True,
    # "sharpen_radius": 0,
    # "smooth_radius": 0,
    # "tile_norm_blocksize": 0,
    # "tile_norm_smooth3D": 1,
    # "invert": False,
}


# %%
from scipy import ndimage
import numpy as np


def background_subtract_highpass(image, sigma=50):
    """
    Background subtraction using Gaussian highpass filter.

    Parameters:
    image : 2D numpy array
        Input image
    sigma : float
        Standard deviation for Gaussian filter
        Larger sigma removes larger background features
    """
    # Create lowpass version (background)
    background = ndimage.gaussian_filter(image, sigma=sigma)

    # Subtract background (equivalent to highpass)
    background_removed = image - background

    return background_removed


s = background_subtract_highpass(wga_max, 100)

# %%
_model = CellposeModel(
    gpu=True,
    pretrained_model=models[0].as_posix(),
)
masks, flows, styles = _model.eval(
    wga_max,
    batch_size=24,
    channels=[0, 0],  # +1 because cellpose treats channel 0 as "gray" aka everything.
    normalize=False,
    flow_threshold=0.4,
    cellprob_threshold=0,
    diameter=184.84,
    resample=False,
    do_3D=False,
    augment=True,
)
# %%
with (path / "cellpose.pkl").open("wb") as f:
    pickle.dump((masks, flows, styles), f)

tifffile.imwrite(
    path / "masks.tif",
    masks,
    compression=22610,
    metadata={"axes": "YX"},
    bigtiff=True,
)
# %%

masks = imread(path / "masks.tif")
# %%
sl = np.s_[8000:9000, 9000:11000]
fig, axs = plt.subplots(ncols=2, nrows=1, figsize=(8, 4), dpi=200)
axs = axs.flatten()
# for i, ax in enumerate(axs):
# ax.axis("off")
axs[0].imshow(wga_max[*sl], zorder=1)
axs[1].imshow(masks[sl], zorder=1)
# %%

for i in range(20):
    for j in range(5):
        print(i, j)
        tifffile.imwrite(
            f"/archive/starmap/barrel/barrel_thicc1/analysis/deconv/stitch--barrel+pipolyA/fused_max_{i}_{j}.tif",
            img[:, 3000 + i * 500, 6000 + j * 500],
            compression=22610,
        )


# %%
img = imread("/archive/starmap/barrel/barrel_thicc1/analysis/deconv/registered--barrel+alina/reg-0047.tif")

# %%

plt.imshow(img[0, ::2, ::2], vmax=15000)

# %%
