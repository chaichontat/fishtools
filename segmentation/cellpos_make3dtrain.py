import glob
import os
import pathlib
import sys
import time
from pathlib import Path

import click
import numpy as np
import tifffile
from cellpose import core, io, models, transforms, utils, version_str
from natsort import natsorted
from tqdm import tqdm


@click.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=pathlib.Path))
@click.option("--z_axis", type=int, help="Axis of image which corresponds to Z dimension", default=0)
@click.option("--channel_axis", type=int, help="Axis of image which corresponds to image channels", default=1)
@click.option("--chan", default=1, type=int, help="Channel to segment; 0: GRAY, 1: RED, 2: GREEN, 3: BLUE")
@click.option(
    "--chan2",
    default=3,
    type=int,
    help="Nuclear channel (if cyto, optional); 0: NONE, 1: RED, 2: GREEN, 3: BLUE",
)
@click.option("--anisotropy", default=4.0, type=float, help="Anisotropy of volume in 3D")
@click.option("--sharpen_radius", default=0.0, type=float, help="High-pass filtering radius")
@click.option("--tile_norm", default=0, type=int, help="Tile normalization block size")
@click.option("--nimg_per_tif", default=10, type=int, help="Number of crops in XY to save per tiff")
@click.option("--crop_size", default=512, type=int, help="Size of random crop to save")
def main(
    path: Path,
    channel_axis: int,
    z_axis: int,
    chan: int,
    chan2: int,
    anisotropy: float,
    sharpen_radius: bool,
    tile_norm: bool,
    nimg_per_tif: int,
    crop_size: int,
):
    """Prepare 3D training data for cellpose."""

    image_names: list[str] = io.get_image_files(path.resolve().as_posix(), "_masks")

    rand = np.random.default_rng(0)
    path_train = path / "train"
    path_train.mkdir(exist_ok=True)

    orientations = {
        "YX": (0, 1, 2, 3),
        "ZY": (2, 0, 1, 3),
        "ZX": (1, 0, 2, 3),
    }

    for name in image_names:
        name_path = Path(name)
        img0 = transforms.convert_image(
            io.imread(str(name_path)), channels=[chan, chan2], channel_axis=channel_axis, z_axis=z_axis
        )

        for axes, order in orientations.items():
            img = img0.transpose(order).copy()
            print(axes, img[0].shape)
            Ly, Lx = img.shape[1:3]
            imgs = img[rand.permutation(img.shape[0])[:nimg_per_tif]]

            if anisotropy > 1.0:
                imgs = transforms.resize_image(imgs, Ly=int(anisotropy * Ly), Lx=Lx)

            for k, img in enumerate(imgs):
                if tile_norm:
                    img = transforms.normalize99_tile(img, blocksize=tile_norm)
                if sharpen_radius:
                    img = transforms.smooth_sharpen_img(img, sharpen_radius=sharpen_radius)

                ly = 0 if Ly - crop_size <= 0 else rand.integers(0, Ly - crop_size)
                lx = 0 if Lx - crop_size <= 0 else rand.integers(0, Lx - crop_size)

                tifffile.imwrite(
                    path_train / f"{name_path.stem}_{axes}_{k}.tif",
                    img[ly : ly + crop_size, lx : lx + crop_size].squeeze(),
                    compression=22610,
                    compressionargs={"level": 0.75},
                )


if __name__ == "__main__":
    main()
