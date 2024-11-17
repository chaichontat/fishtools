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

# "/mnt/working/lai/segment--left/models/2024-11-02_16-30-54"


logging.basicConfig(level=logging.INFO)

# path = Path("/mnt/working/e155trcdeconv/segment3d--left")
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


nchan = 2


# %%
def calc_uniform_offset(img_shape: np.ndarray, *, overlap=0.1):
    MAGIC_NUMBER = np.sqrt(24) * 4600
    n_z = img_shape[0]
    bsize = 448 if n_z <= 24 else 224

    target_length = int(MAGIC_NUMBER / np.sqrt(n_z))

    if target_length > np.max(img_shape[2:]):
        return np.max(img_shape[2:]), bsize  # when image is smaller than tile size

    # Adjust target length to ensure that the remainders aren't too small.
    n_ys, rem_y = divmod(target_length, img_shape[2])
    n_xs, rem_x = divmod(target_length, img_shape[3])
    while min(rem_y, rem_x) < target_length * 0.2:
        if rem_y < target_length * 0.2:
            target_length = (target_length * n_ys) / (n_ys + 1)
        else:
            target_length = (target_length * n_xs) / (n_xs + 1)

    ny = int(np.ceil((1.0 + 2 * overlap) * target_length / bsize))
    step = (target_length - bsize) // (ny - 1)
    target_length = step * (ny - 1) + bsize
    assert (target_length - bsize) % (ny - 1) == 0
    print((target_length - bsize) / (ny - 1))  # Should be a round number
    tiles = np.linspace(0, target_length - bsize, ny)

    offset = tiles[-1]
    assert int(offset) == offset
    return int(offset), bsize


def get_tile_list(img_shape: np.ndarray, *, offset: int):
    xx, yy = np.meshgrid(
        np.arange(0, img_shape[3], offset), np.arange(0, img_shape[2], offset), indexing="ij"
    )
    return np.column_stack([xx.flat, yy.flat]).astype(int)


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


def save_sliced(path: Path, limit_z: int | None, chans: list[int]):
    img = imread(path)
    np.savetxt(path.parent / "cellpose" / "shape.txt", img.shape)
    if path.with_name(path.stem + "_for_cellpose.tif").exists():
        return None

    if limit_z is not None:
        if np.max(np.abs(limit_z)) >= img.shape[0]:
            raise ValueError("Limit z must be smaller than image z")
        img = img[limit_z]

    if chans != img.shape[1]:
        if np.max(np.abs(chans)) >= img.shape[1]:
            raise ValueError("Channels must be smaller than image channels")
        img = img[:, chans]

    tifffile.imwrite(
        path.with_name(path.stem + "_for_cellpose.tif"),
        img,
        compression=22610,
        compressionargs={"level": 0.75},
        bigtiff=True,
    )
    logger.info(f"Saved working file to {path.with_name(path.stem + "_for_cellpose.tif")}")
    return img


@click.group()
def cli():
    ...


@cli.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=False, file_okay=True, path_type=Path))
@click.argument("idx", type=int)
@click.option("--model", type=click.Path(exists=True, dir_okay=False, file_okay=True, path_type=Path))
@click.option("--channels", "-c", type=str, default="0,1")
@click.option("--limit-z", type=int, default=None)
@click.option("--overwrite", is_flag=True)
@click.option("--anisotropy", type=float, default=4)
def run(
    path: Path,
    idx: int,
    *,
    model: Path,
    channels: str,
    limit_z: int | None = None,
    overwrite: bool = False,
    anisotropy: float = 4,
):
    chans = [int(x) for x in channels.split(",")]
    del channels
    out_path = path.parent / "cellpose"
    out_path.mkdir(exist_ok=True)

    img = save_sliced(path, limit_z, chans) if not (path.parent / "cellpose" / "shape.txt").exists() else None

    img_shape = np.loadtxt(path.parent / "cellpose" / "shape.txt").astype(int)
    offset, bsize = calc_uniform_offset(img_shape)
    tiles = get_tile_list(img_shape, offset=offset)

    try:
        x, y = tiles[idx]
    except IndexError:
        raise ValueError(f"Tile index {idx} out of bounds. Found {len(tiles)} tiles given {img_shape}.")

    if not overwrite and (out_path / f"masks-{x:05d}_{y:05d}.tif").exists():
        logger.info(f"Tile {idx} already exists. Skipping.")
        return

    del img
    img = imread(path.with_name(path.stem + "_for_cellpose.tif"))
    logger.info(
        f"Using existing image. Limit-z and channels are ignored. "
        f"To start over, delete {path.with_name(path.stem + "_for_cellpose.tif")}"
    )

    print(x, y)
    img = img[:, :, y : min(y + offset, img.shape[2]), x : min(x + offset, img.shape[3])]
    _model = CellposeModel(gpu=True, pretrained_model=cast(bool, model.as_posix()))
    masks, flows, styles = _model.eval(
        img,
        batch_size=24,
        channels=[c + 1 for c in chans],  # +1 because cellpose treats channel 0 as "gray" aka everything.
        normalize=cast(bool, {"tile_norm_blocksize": bsize, "norm3D": True, "percentile": (1, 99.9)}),
        anisotropy=anisotropy,
        flow_threshold=1,  # 3D segmentation ignores the flow_threshold
        cellprob_threshold=-1,
        diameter=0,  # In order for rescale to not be ignored, we need to set diameter to 0.
        rescale=1.0,
        do_3D=True,
        dP_smooth=0,
        bsize=bsize,
        augment=False,
        # diameter=model.diam_labels,
    )

    with (out_path / f"{x:05d}_{y:05d}.pkl").open("wb") as f:
        pickle.dump((masks, flows, styles), f)

    tifffile.imwrite(
        out_path / f"masks-{x:05d}_{y:05d}.tif",
        masks,
        compression=22610,
        metadata={"axes": "ZYX"},
        bigtiff=True,
        # imagej=True,
    )


@cli.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=False, file_okay=True, path_type=Path))
@click.option("--model", type=click.Path(exists=True, dir_okay=False, file_okay=True, path_type=Path))
@click.option("--channels", "-c", type=str, default="0,1")
@click.option("--limit-z", type=int, default=None)
@click.option("--overwrite", is_flag=True)
@click.option("--anisotropy", type=float, default=4)
@click.option("--overwrite", is_flag=True)
def batch(path: Path, model: Path, channels: str, limit_z: int | None, overwrite: bool, anisotropy: float):
    if not (path.parent / "cellpose" / "shape.txt").exists():
        subprocess.run(
            [
                "python",
                __file__,
                "run",
                path,
                "0",
                "--model",
                model,
                "--channels",
                channels,
                *(["--overwrite"] if overwrite else []),
            ],
            check=True,
            capture_output=False,
        )

    img_shape = np.loadtxt(path.parent / "cellpose" / "shape.txt")
    offset, _ = calc_uniform_offset(img_shape)
    tiles = get_tile_list(img_shape, offset=offset)

    for i, (x, y) in enumerate(tiles):
        if not overwrite and (Path.cwd() / f"chunks/masks-{x:05d}_{y:05d}.pkl").exists():
            continue
        subprocess.run(
            [
                "python",
                __file__,
                "run",
                path,
                str(i),
                "--model",
                model,
                "--channels",
                channels,
                *(["--overwrite"] if overwrite else []),
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
