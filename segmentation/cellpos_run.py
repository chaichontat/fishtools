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
from tifffile import imread

imshow = partial(plt.imshow, zorder=1)


logging.basicConfig(level=logging.INFO)

path = Path("/mnt/working/lai/segment3d--left")
path_models = Path("/mnt/working/lai/segment--left/models")

# name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


# %%
# Yes, the typing is wrong.
# model = CellposeModel(gpu=True, pretrained_model=cast(bool, models[0].as_posix()))

models = sorted(
    (file for file in path_models.glob("*") if "train_losses" not in file.name),
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
    "lowhigh": None,
    "percentile": (0.1, 99.9),
    "normalize": True,
    "norm3D": True,
    "sharpen_radius": 0,
    "smooth_radius": 0,
    "tile_norm_blocksize": 0,
    "tile_norm_smooth3D": 1,
    "invert": False,
}
# %%
import tifffile

# img = imread("/mnt/working/lai/registered--whole_embryo/fused.tif")[:24, [0, 2]]
# tifffile.imwrite(
#     "temporary.tif",
#     img,
#     bigtiff=True,
#     compression="zlib",
# )

bsize = np.int32(224 * 2)  # 896
nchan = 2
Ly, Lx = 448 * 10, 448 * 10
# pad if image smaller than bsize


# tiles overlap by half of tile size


# def get_tile_size(length: int, bsize: int = 448):
#     """From cellpose.transforms. Most optimal when length is divisible by bsize."""
#     ny = max(2, int(np.ceil(2.0 * length / bsize)))
#     # Make (length - bsize) divisible by (ny-1) for even spacing
#     step = (length - bsize) // (ny - 1)
#     length = step * (ny - 1) + bsize
#     assert (length - bsize) % (ny - 1) == 0
#     print((length - bsize) / (ny - 1))  # Should be a round number
#     tiles = np.linspace(0, length - bsize, ny)
#     assert int(tiles[-1]) == tiles[-1]
#     return tiles.astype(int), length


# tiles, length = get_tile_size(5000, 448)
# print(tiles, length)


# %%
def get_tile_size2(length: int, bsize: int = 448, overlap=0.1):
    ny = int(np.ceil((1.0 + 2 * overlap) * length / bsize))
    step = (length - bsize) // (ny - 1)
    length = step * (ny - 1) + bsize
    assert (length - bsize) % (ny - 1) == 0
    print((length - bsize) / (ny - 1))  # Should be a round number
    tiles = np.linspace(0, length - bsize, ny)
    assert int(tiles[-1]) == tiles[-1]
    return tiles.astype(int), length


tiles, length = get_tile_size2(4800, 448)
print(tiles, length)
# %%
size = length  # 896*5
offset = tiles[-2]  #  448 * 9  # 896*4
x, y = 3, 3


# img_norm = img.astype(np.float32)
# img_norm -= lower[None, :, None, None]
# img_norm *= 1 / (upper - lower)[None, :, None, None]
# %%
# img_norm[..., c] -= lower
# img_norm[..., c] /= (upper - lower)


# 19584, 26070


# %%

# dapi = imread("/fast2/3t3clean/analysis/deconv/registered/dapi3/0/fused_1.tif").squeeze()
# polya = imread("/fast2/3t3clean/analysis/deconv/registered/dapi3/2/fused_1.tif").squeeze()
# intensity = scale_image_2x_optimized(img)


@click.group()
def cli(): ...


@cli.command()
@click.argument("x", type=int)
@click.argument("y", type=int)
def run(x: int, y: int):
    img = imread("temporary.tif")
    # lower, upper = np.percentile(img, (1, 99.9), axis=(0, 2, 3))

    img = img[:, :, y : y + size, x : x + size]
    bsize = 448
    model = CellposeModel(gpu=True, pretrained_model=cast(bool, models[0].as_posix()))
    masks, flows, styles = model.eval(
        img,
        batch_size=64,
        channels=[1, 2],
        normalize=cast(bool, {"tile_norm_blocksize": bsize, "norm3D": True, "percentile": (1, 99.9)}),
        anisotropy=4,
        flow_threshold=0.8,  # 3D segmentation ignores the flow_threshold
        cellprob_threshold=0,
        diameter=0,  # In order for rescale to not be ignored, we need to set diameter to 0.
        rescale=1.0,
        do_3D=True,
        dP_smooth=0,
        bsize=bsize,
        augment=False,
        # diameter=model.diam_labels,
    )
    Path(f"chunks").mkdir(exist_ok=True, parents=True)

    with open(f"chunks/{x:05d}_{y:05d}.pkl", "wb") as f:
        pickle.dump((masks, flows, styles), f)

    tifffile.imwrite(
        f"chunks/masks-{x:05d}_{y:05d}.tif",
        masks,
        compression=22610,
        metadata={"axes": "ZYX"},
        imagej=True,
    )


@cli.command()
def batch():
    xx, yy = np.meshgrid(np.arange(0, 26070, offset), np.arange(0, 19584, offset), indexing="ij")
    idxs = np.column_stack([xx.flat, yy.flat])
    for i, (x, y) in enumerate(idxs):
        if (Path.cwd() / f"chunks/masks-{x:05d}_{y:05d}.pkl").exists():
            continue
        print(i, x, y)
        subprocess.run(
            [
                "python",
                __file__,
                "run",
                str(x),
                str(y),
            ],
            check=True,
            capture_output=False,
        )
    # Max is 20x5000x5000px. Limit is in compute_masks. This eats 24GB of GPU memory.

    # %%


if __name__ == "__main__":
    cli()
# %%

# %%

# ysub = []
# xsub = []
# %%
# IMG = np.zeros((len(ystart), len(xstart), nchan, bsize, bsize), np.float32)


# %%
