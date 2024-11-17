# %%
import json
import os
import pickle
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import reduce
from itertools import chain, groupby
from pathlib import Path
from typing import Any, Literal, Mapping, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import rich_click as click
import starfish
import starfish.data
import xarray as xr
from loguru import logger
from pydantic import BaseModel, TypeAdapter, field_validator
from skimage.filters import gaussian
from starfish import Codebook, ImageStack, IntensityTable
from starfish.core.intensity_table.intensity_table_coordinates import (
    transfer_physical_coords_to_intensity_table,
)
from starfish.core.spots.DetectPixels.combine_adjacent_features import CombineAdjacentFeatures
from starfish.core.types import Axes, Coordinates, CoordinateValue
from starfish.experiment.builder import FetchedTile, TileFetcher
from starfish.image import Filter
from starfish.types import Axes, Features, Levels
from starfish.util.plot import imshow_plane, intensity_histogram
from tifffile import TiffFile, imread, imwrite

from fishtools.utils.pretty_print import progress_bar

os.environ["TQDM_DISABLE"] = "1"


def make_fetcher(path: Path, sl: slice | list[int] = np.s_[:]):
    img = imread(path).astype(np.float32)[sl] / 65535

    class DemoFetchedTile(FetchedTile):
        def __init__(self, z, chs, *args, **kwargs):
            self.z = z
            self.c = chs

        @property
        def shape(self) -> Mapping[Axes, int]:
            return {
                Axes.Y: img.shape[2],
                Axes.X: img.shape[3],
            }

        @property
        def coordinates(self) -> Mapping[str | Coordinates, CoordinateValue]:
            return {
                Coordinates.X: (0, 0.001),
                Coordinates.Y: (0, 0.001),
                Coordinates.Z: (0.001 * self.z, 0.001 * (self.z + 1)),
            }

        def tile_data(self) -> np.ndarray:
            return img[self.z, self.c]  # [512:1536, 512:1536]

    class DemoTileFetcher(TileFetcher):
        def get_tile(self, fov_id: int, round_label: int, ch_label: int, zplane_label: int) -> FetchedTile:
            return DemoFetchedTile(zplane_label, ch_label)

    return ImageStack.from_tilefetcher(
        DemoTileFetcher(),
        {
            Axes.X: img.shape[3],
            Axes.Y: img.shape[2],
        },
        fov=0,
        rounds=range(1),
        chs=range(img.shape[1]),
        zplanes=range(img.shape[0]),
        group_by=(Axes.CH, Axes.ZPLANE),
    )


# Define some useful functions for viewing multiple images and histograms
def imshow_3channels(stack: starfish.ImageStack, r: int):
    fig = plt.figure(dpi=150)
    ax1 = fig.add_subplot(131, title="ch: 0")
    ax2 = fig.add_subplot(132, title="ch: 1")
    ax3 = fig.add_subplot(133, title="ch: 2")
    imshow_plane(stack, sel={Axes.ROUND: 0, Axes.CH: 0}, ax=ax1)
    imshow_plane(stack, sel={Axes.ROUND: 0, Axes.CH: 1}, ax=ax2)
    imshow_plane(stack, sel={Axes.ROUND: 0, Axes.CH: 2}, ax=ax3)


def plot_intensity_histograms(stack: starfish.ImageStack, r: int):
    fig = plt.figure(dpi=150)
    ax1 = fig.add_subplot(131, title="ch: 0")
    ax2 = fig.add_subplot(132, title="ch: 1", sharex=ax1, sharey=ax1)
    ax3 = fig.add_subplot(133, title="ch: 2", sharex=ax1, sharey=ax1)
    intensity_histogram(stack, sel={Axes.ROUND: r, Axes.CH: 0}, log=True, bins=50, ax=ax1)
    intensity_histogram(stack, sel={Axes.ROUND: r, Axes.CH: 1}, log=True, bins=50, ax=ax2)
    intensity_histogram(stack, sel={Axes.ROUND: r, Axes.CH: 2}, log=True, bins=50, ax=ax3)
    fig.tight_layout()


def scale(img: ImageStack, scale: np.ndarray[np.float32, Any]):
    Filter.ElementWiseMultiply(
        xr.DataArray(
            np.nan_to_num(scale, nan=1).reshape(-1, 1, 1, 1, 1),
            dims=("c", "x", "y", "z", "r"),
        )
    ).run(img, in_place=True)


def load_codebook(path: Path, exclude: set[str] | None = None):
    cb_json: dict[str, list[int]] = json.loads(path.read_text())
    for k in exclude or set():
        try:
            cb_json.pop(k)
        except KeyError:
            logger.warning(f"Codebook does not contain {k}")

    used_bits = sorted(set(chain.from_iterable(cb_json.values())))

    # mapping from bit name to index
    bit_map = np.ones(max(used_bits) + 1, dtype=int) * 5000
    for i, bit in enumerate(used_bits):
        bit_map[bit] = i

    arr = np.zeros((len(cb_json), len(used_bits)), dtype=bool)
    for i, bits_gene in enumerate(cb_json.values()):
        for bit in bits_gene:
            assert bit > 0
            arr[i, bit_map[bit]] = 1

    names = np.array(list(cb_json.keys()))
    is_blank = np.array([n.startswith("Blank") for n in names])[:, None]  # [:-1, None]
    arr_zeroblank = arr * ~is_blank
    return (
        Codebook.from_numpy(
            names,
            n_round=1,
            n_channel=len(used_bits),
            data=arr[:, np.newaxis],
        ),
        used_bits,
        names,
        arr_zeroblank,
    )


# %%


@click.group()
def cli(): ...


def _batch(paths: list[Path], mode: str, args: list[str], *, threads: int = 13, split: bool = False):
    with progress_bar(len(paths)) as callback, ThreadPoolExecutor(threads) as exc:
        futs = []
        for path in paths:
            for s in range(4) if split else [None]:
                futs.append(
                    exc.submit(
                        subprocess.run,
                        ["python", __file__, mode, str(path), *[a for a in args if a]]
                        + (["--split", str(s)] if split else []),
                        check=True,
                        capture_output=False,
                    )
                )

        for f in as_completed(futs):
            callback()


def sample_imgs(path: Path, round_num: int, *, batch_size: int = 50):
    rand = np.random.default_rng(round_num)
    paths = sorted(path.glob("registered--*/reg*.tif"))
    return [paths[i] for i in sorted(rand.choice(range(len(paths)), size=batch_size, replace=False))]


@cli.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=Path))
@click.option("--round", "round_num", type=int)
@click.option(
    "--codebook",
    "codebook_path",
    type=click.Path(exists=True, dir_okay=False, file_okay=True, path_type=Path),
)
@click.option("--subsample-z", type=int, default=1)
@click.option("--batch-size", "-n", type=int, default=100)
@click.option("--threads", "-t", type=int, default=8)
@click.option("--overwrite", is_flag=True)
@click.option("--split", type=int, default=0)
def optimize(
    path: Path,
    round_num: int,
    codebook_path: Path,
    batch_size: int = 100,
    subsample_z: int = 1,
    threads: int = 8,
    overwrite: bool = False,
    split: int = 0,
):
    selected = sample_imgs(path, round_num, batch_size=batch_size)

    group_counts = {key: len(list(group)) for key, group in groupby(selected, key=lambda x: x.parent.name)}
    logger.info(f"Group counts: {json.dumps(group_counts, indent=2)}")

    # Copy codebook to the same folder as the images for reproducibility.
    (path / codebook_path.stem).mkdir(exist_ok=True)
    if not (new_cb_path := path / codebook_path.stem / codebook_path.name).exists():
        shutil.copy(codebook_path, new_cb_path)

    return _batch(
        selected,
        "run",
        [
            "--calc-deviations",
            "--global-scale",
            str(path / f"opt_{codebook_path.stem}" / "global_scale.txt"),
            "--codebook",
            codebook_path.as_posix(),
            f"--round={round_num}",
            "--overwrite" if overwrite else "",
            f"--subsample-z={subsample_z}",
            f"--split={split}",
        ],
        threads=threads,
    )


@cli.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=Path))
@click.option("--global-scale", type=click.Path(dir_okay=False, file_okay=True, exists=True, path_type=Path))
@click.option(
    "--codebook",
    "codebook_path",
    type=click.Path(exists=True, dir_okay=False, file_okay=True, path_type=Path),
)
@click.option("--threads", "-t", type=int, default=13)
@click.option("--overwrite", is_flag=True)
@click.option("--subsample-z", type=int, default=1)
@click.option("--limit-z", default=None)
@click.option("--split", is_flag=True)
def batch(
    path: Path,
    global_scale: Path,
    codebook_path: Path,
    threads: int = 13,
    overwrite: bool = False,
    subsample_z: int = 1,
    limit_z: int | None = None,
    split: bool = False,
):
    return _batch(
        sorted(path.glob("reg*.tif")),
        "run",
        [
            "--global-scale",
            global_scale.as_posix(),
            "--codebook",
            codebook_path.as_posix(),
            "--overwrite" if overwrite else "",
            f"--subsample-z={subsample_z}",
            f"--limit-z={limit_z}",
        ],
        threads=threads,
        split=split,
    )


def load_2d(path: Path | str, *args, **kwargs):
    return np.atleast_2d(np.loadtxt(path, *args, **kwargs))


def create_opt_path(
    *,
    codebook_path: Path,
    mode: Literal["json", "pkl", "folder"],
    round_num: int | None = None,
    path_img: Path | None = None,
    path_folder: Path | None = None,
):
    if not ((path_img is None) ^ (path_folder is None)):
        raise ValueError("Must provide only path_img or path_folder")

    if path_img is not None:
        base = path_img.parent.parent / f"opt_{codebook_path.stem}"
        name = f"{path_img.stem}--{path_img.resolve().parent.name.split('--')[1]}"
    elif path_folder is not None:
        base = path_folder / f"opt_{codebook_path.stem}"
        if mode == "folder":
            return base
        raise ValueError("Cannot use path_folder with mode other than folder")
    else:
        raise Exception("This should never happen.")

    base.mkdir(exist_ok=True)
    if mode == "json":
        return base / f"{name}.json"
    if mode == "pkl":
        return base / f"{name}_opt{round_num:02d}.pkl"
    if mode == "folder":
        return base
    raise ValueError(f"Unknown mode {mode}")


@cli.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=Path))
@click.option(
    "--codebook",
    "codebook_path",
    type=click.Path(exists=True, dir_okay=False, file_okay=True, path_type=Path),
)
@click.option("--round", "round_num", type=int)
def combine(path: Path, codebook_path: Path, round_num: int):
    """Create global scaling factors from `optimize`.

    Part of the channel optimization process.
    This is to be run after `optimize` has been run.

    Round 0: Average of all scaling factors.
    Round n:
        Balance the scaling factors such that each positive spot
        has about the same intensity in all bit channels.
        Get deviations from all images during `optimize` and average them.
        This average deviation is a vector of length n.
        Divide this vector by its mean to get a scaling factor for each bit channel.
        This gets applied to the previous round's scaling factors.
        Also calculates the variance of the deviations to track the convergence.

    Args:
        path: registered images folder
        codebook_path
        round_num: starts from 0.
    """
    selected = sample_imgs(path, round_num, batch_size=50)
    path_opt = create_opt_path(
        path_folder=path, codebook_path=codebook_path, mode="folder", round_num=round_num
    )
    paths = [
        create_opt_path(path_img=p, codebook_path=codebook_path, mode="json", round_num=round_num)
        for p in selected
    ]

    # paths = list((path / codebook_path.stem).glob("reg*.json"))
    curr = []
    n = 0
    for p in paths:
        sf = Deviations.validate_json(p.read_text())
        # if (round_num + 1) > len(sf):
        #     raise ValueError(f"Round number {round_num} exceeds what's available ({len(sf)}).")

        if round_num == 0:
            curr.append([cast(InitialScale, s).initial_scale for s in sf if s.round_num == round_num][0])
        else:
            want = cast(Deviation, [s for s in sf if s.round_num == round_num][0])
            curr.append(np.array(want.deviation) * want.n)
            n += want.n
    curr = np.array(curr)
    # pl.DataFrame(curr).write_csv(path / name)

    global_scale_file = path_opt / "global_scale.txt"
    if round_num == 0:
        curr = np.nanmean(np.array(curr), axis=0, keepdims=True)
        np.savetxt(global_scale_file, curr)
        return

    if not global_scale_file.exists():
        raise ValueError("Round > 0 requires global scale file.")

    deviation = curr.sum(axis=0) / n
    # Normalize per channel
    deviation = deviation / np.nanmean(deviation)
    cv = np.sqrt(np.mean(np.square(deviation - 1)))
    logger.info(f"Round {round_num}. CV: {cv:04f}.")

    previouses = load_2d(global_scale_file)
    old = previouses[round_num - 1]
    new = old / deviation
    # if round_num > 1:
    #     grad = new - old
    #     β = 0.1
    #     velocity = old - previouses[round_num - 2]
    #     velocity = β * velocity + (1 - β) * grad
    #     new = old + velocity

    np.savetxt(
        global_scale_file,
        np.concatenate([np.atleast_2d(previouses[:round_num]), np.atleast_2d(new)], axis=0),
    )
    (path_opt / "mse.txt").open("a").write(f"{round_num:02d}\t{cv:04f}\n")


def initial(img: ImageStack):
    """
    Create initial scaling factors.
    Returns the max intensity of each channel.
    Seems to work well enough.
    """
    maxed = img.reduce({Axes.ROUND, Axes.ZPLANE, Axes.Y, Axes.X}, func="max")
    res = np.array(maxed.xarray).squeeze()
    if np.isnan(res).any() or (res == 0).any():
        raise ValueError("NaNs or zeros found in initial scaling factor.")
    return res


class Deviation(BaseModel):
    n: int
    deviation: list[float]
    percent_blanks: float | None = None
    round_num: int

    @field_validator("round_num")
    @classmethod
    def must_be_positive(cls, v: int) -> int:
        if v > 0:
            return v
        raise ValueError("Round number must be positive for Deviation.")


class InitialScale(BaseModel):
    initial_scale: list[float]
    round_num: int

    @field_validator("round_num")
    @classmethod
    def must_be_zero(cls, v: int) -> int:
        if v == 0:
            return v
        raise ValueError("Round number must be zero for InitialScale.")


Deviations = TypeAdapter(list[InitialScale | Deviation])


def append_json(
    path: Path,
    round_num: int,
    *,
    n: int | None = None,
    deviation: np.ndarray | None = None,
    initial_scale: np.ndarray | None = None,
    percent_blanks: float | None = None,
):
    existing = Deviations.validate_json(path.read_text()) if path.exists() else Deviations.validate_json("[]")
    if initial_scale is not None:
        if round_num > 0:
            raise ValueError("Cannot set initial scale for round > 0")
        existing.append(InitialScale(initial_scale=initial_scale.tolist(), round_num=round_num))

    elif deviation is not None and n is not None:
        if round_num == 0:
            raise ValueError("Cannot set deviation for round 0")
        # if existing.__len__() < round_num - 1:
        # raise ValueError("Round number exceeds number of existing rounds.")

        existing = [e for e in existing if e.round_num < round_num]
        existing.append(
            Deviation(n=n, deviation=deviation.tolist(), round_num=round_num, percent_blanks=percent_blanks)
        )
    else:
        raise ValueError("Must provide either initial_scale or deviation and n.")

    path.write_bytes(Deviations.dump_json(existing))


@cli.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=False, file_okay=True, path_type=Path))
@click.option("--overwrite", is_flag=True)
@click.option("--global-scale", type=click.Path(dir_okay=False, file_okay=True, path_type=Path))
@click.option("--round", "round_num", type=int, default=None)
@click.option("--calc-deviations", is_flag=True)
@click.option("--subsample-z", type=int, default=1)
@click.option("--limit-z", default=None)
@click.option("--debug", is_flag=True)
@click.option("--split", default=None)
@click.option(
    "--codebook",
    "codebook_path",
    type=click.Path(exists=True, dir_okay=False, file_okay=True, path_type=Path),
)
def run(
    path: Path,
    *,
    global_scale: Path | None,
    codebook_path: Path,
    round_num: int | None = None,
    overwrite: bool = False,
    debug: bool = False,
    calc_deviations: bool = False,
    subsample_z: int = 1,
    limit_z: int | None = None,
    split: int | None = None,
):
    """
    Run spot calling.
    Used for actual spot calling and channel optimization.

    For channel optimization, start by calling with `--calc-deviations` and `--round=0`.
    This will get the initial scaling factors for each bit channel (max).
    Returns the average intensity of all non-blank spots including the total number of spots.
    These info from multiple images are used to calculate the scaling factors for each bit channel.

    Intermediate files are saved in path / codebook_path.stem since in the current design,
    different codebooks use different channels.

    For actual spot calling, call with `--global-scale`.
    This will use the latest scaling factors calculated from the previous step to decode.
    """
    if debug:
        logger.remove()
        logger.add(sys.stderr, level="INFO")

    if calc_deviations and round_num is None:
        raise ValueError("Round must be provided for calculating deviations.")

    if calc_deviations:
        path_pickle = create_opt_path(
            path_img=path, codebook_path=codebook_path, mode="pkl", round_num=round_num
        )

    else:
        (_path_out := path.parent / codebook_path.stem).mkdir(exist_ok=True)
        path_pickle = _path_out / f"{path.stem}{f'-{split}' if split is not None else ''}.pkl"

    if path_pickle.exists() and not overwrite:
        logger.info(f"Skipping {path.name}. Already exists.")
        return

    logger.info(
        f"Running {path.parent.name}/{path.name} with {limit_z} z-slices and {subsample_z}x subsampling."
    )
    codebook, used_bits, names, arr_zeroblank = load_codebook(codebook_path, exclude={"Malat1-201"})
    used_bits = list(map(str, used_bits))

    # if not path.with_suffix(".highpassed.tif").exists():
    with TiffFile(path) as tif:
        img_keys = tif.shaped_metadata[0]["key"]

    bit_mapping = {k: i for i, k in enumerate(img_keys)}
    # img = tif.asarray()[:, [bit_mapping[k] for k in used_bits]]
    # blurred = gaussian(path, sigma=8)
    # blurred = levels(blurred)  # clip negative values to 0.
    # filtered = image - blurred
    cut = 994 + 30
    split = int(split) if split is not None else None
    if split is None:
        split_slice = np.s_[:, :]
    elif split == 0:
        split_slice = np.s_[:cut, :cut]
    elif split == 1:
        split_slice = np.s_[cut:, -cut:]
    elif split == 2:
        split_slice = np.s_[-cut:, :cut]
    elif split == 3:
        split_slice = np.s_[-cut:, -cut:]
    else:
        raise ValueError(f"Unknown split {split}")

    slc = tuple(
        np.s_[
            ::subsample_z,
            [bit_mapping[k] for k in used_bits],
        ]
    ) + tuple(split_slice)
    stack = make_fetcher(
        path,
        slc,
        # if limit_z is None
        # else np.s_[
        #     :limit_z:subsample_z,
        #     [bit_mapping[k] for k in used_bits],
        # ],
    )

    # In all modes, data below 0 is set to 0.
    # We probably wouldn't need SATURATED_BY_IMAGE here since this is a subtraction operation.
    # But it's there as a reference.
    ghp = Filter.GaussianHighPass(sigma=8, is_volume=True, level_method=Levels.SCALE_SATURATED_BY_IMAGE)
    ghp.sigma = (6, 8, 8)  # z,y,x
    logger.debug(f"Running GHP with sigma {ghp.sigma}")
    imgs: ImageStack = ghp.run(stack)

    ghp = Filter.GaussianLowPass(sigma=1, level_method=Levels.SCALE_SATURATED_BY_IMAGE)
    imgs = ghp.run(imgs)

    # z_filt = Filter.ZeroByChannelMagnitude(thresh=3e-4, normalize=False)
    # imgs = z_filt.run(imgs)
    import tifffile

    tifffile.imwrite("highpassed.tif", imgs.xarray.to_numpy())
    return

    path_json = create_opt_path(path_img=path, codebook_path=codebook_path, mode="json", round_num=round_num)
    if round_num == 0 and calc_deviations:
        logger.debug("Making scale file.")
        path_json.write_bytes(
            Deviations.dump_json(
                Deviations.validate_python([{"initial_scale": (1 / initial(imgs)).tolist(), "round_num": 0}])
            )
        )
        return

    if not global_scale or not global_scale.exists():
        raise ValueError("Production run or round > 0 requires a global scale file.")

    scale_all = np.loadtxt(global_scale)
    if len(scale_all.shape) == 1:
        scale_all = scale_all[np.newaxis, :]

    scale_factor = (
        np.array(scale_all[round_num - 1], copy=True)
        if round_num is not None and calc_deviations
        else np.array(scale_all[-1], copy=True)
    )

    try:
        scale(imgs, scale_factor)
    except ValueError:
        # global_scale.unlink()
        raise ValueError("Scale factor dim mismatch. Deleted. Please rerun.")

    # Decode
    pixel_intensities = IntensityTable.from_image_stack(imgs)
    logger.info("Decoding")
    decoded_intensities = codebook.decode_metric(
        pixel_intensities,
        max_distance=0.3,
        min_intensity=0.004,
        norm_order=2,
        metric="euclidean",
        return_original_intensities=True,
    )

    feature_traces = pixel_intensities.stack(traces=(Axes.CH.value, Axes.ROUND.value))
    mags = np.linalg.norm(feature_traces.values, axis=1)

    # plt.hist(mags, bins=20)
    # sns.despine(offset=3)
    # plt.xlabel("Barcode magnitude")
    # plt.ylabel("Number of pixels")
    # plt.yscale("log")

    logger.info("Combining")
    caf = CombineAdjacentFeatures(min_area=12, max_area=200, mask_filtered_features=True)
    decoded_spots, image_decoding_results = caf.run(intensities=decoded_intensities, n_processes=8)
    transfer_physical_coords_to_intensity_table(image_stack=imgs, intensity_table=decoded_spots)

    spot_intensities = decoded_spots.loc[decoded_spots[Features.PASSES_THRESHOLDS]]
    genes, counts = np.unique(
        spot_intensities[Features.AXIS][Features.TARGET],
        return_counts=True,
    )
    gc = dict(zip(genes, counts))
    percent_blanks = sum([v for k, v in gc.items() if k.startswith("Blank")]) / counts.sum()
    logger.info(f"{percent_blanks} blank, Total: {counts.sum()}")

    # Deviations
    if calc_deviations and round_num is not None:
        # Round num already checked above.
        names_l = {n: i for i, n in enumerate(names)}
        idxs = list(map(names_l.get, spot_intensities.coords["target"].to_index().values))

        deviations = np.nanmean(spot_intensities.squeeze() * np.where(arr_zeroblank[idxs], 1, np.nan), axis=0)
        logger.debug(f"Deviations: {np.round(deviations / np.nanmean(deviations), 4)}")

        """Concatenate with previous rounds. Will overwrite rounds beyond the current one."""
        append_json(
            path_json, round_num, deviation=deviations, n=len(spot_intensities), percent_blanks=percent_blanks
        )

    morph = [
        {"area": prop.area, "centroid": prop.centroid}
        for prop in np.array(image_decoding_results.region_properties)
    ]

    with path_pickle.open("wb") as f:
        pickle.dump((decoded_spots, morph), f)

    return decoded_spots, image_decoding_results


if __name__ == "__main__":
    cli()
# %%

# u = run.callback(
#     "/fast2/3t3clean/analysis/deconv/registered/reg-0000.tif",
#     Path("/fast2/3t3clean/analysis/deconv/registered/decode_scale.json"),
# )
# %%
# rand = np.random.default_rng(0)
# t = rand.normal(10, 1, size=(18))
# t = np.zeros(14)
# t[:3] = 200

# t = t / np.linalg.norm(t)

# u = rand.normal(10, 1, size=(18))
# u = np.zeros(14)
# u[:2] = 200
# u[3:5] = 50
# u = u / np.linalg.norm(u)

# 1 - np.dot(t, u)
# %%
# n_chans = img.shape[1]
# scale_factors = [np.percentile(imgs.get_slice({Axes.CH: i})[0].squeeze(), 99.99) for i in range(n_chans)]
# for i, v in enumerate(scale_factors):
#     if v < 1e-6:
#         scale_factors[i] = 1

# %%

# %%
# %%
# fig, axs = plt.subplots(5, 6)
# axs = axs.flatten()
# for i in range(n_chans):
#     intensity_histogram(
#         imgs,
#         sel={Axes.ROUND: 0, Axes.CH: i},
#         log=True,
#         bins=50,
#         ax=axs[i],
#         title=f"ch: {i+1}",
#     )
# fig.tight_layout()

# %%


# orig_plot: xr.DataArray = stack.sel({Axes.CH: 3, Axes.ZPLANE: 0}).xarray.squeeze()
# wth_plot: xr.DataArray = imgs.sel({Axes.CH: 3, Axes.ZPLANE: 0}).xarray.squeeze()

# f, (ax1, ax2) = plt.subplots(ncols=2, dpi=200)
# ax1.imshow(orig_plot)
# ax1.set_title("original")
# ax2.imshow(wth_plot)
# ax2.set_title("with filtered")
# ax1.axis("off")
# ax2.axis("off")
# %%
# imgs = clipped_both_scaled

# %%


# %%
# cb = json.loads(Path("scripts/starmaptest/starmaptestcb.json").read_text())
# out = defaultdict(list)

# for choose in range(2, 5):
#     for name, arr in cb.items():
#         if max(arr) > 10:
#             continue
#         for a in combinations(arr, choose):
#             out[json.dumps(a)].append(name)

# arr = np.zeros((len(out), 10), dtype=np.uint8)
# names = []
# for i, (k, v) in enumerate(out.items()):
#     for idx in json.loads(k):
#         arr[i, idx - 1] = 1
#     names.append(k + "-" + ",".join(v))


# order = [5, 6, 7, 8, 13, 14, 15, 16, 21, 22, 23, 24]
# mapping = {x: i for i, x in enumerate(order)}
# %%


# order = list(chain.from_iterable([[i, i + 8, i + 16] for i in range(1, 9)])) + list(range(25, 33))
# mhd = list(map(str, np.array(list(map(order.__getitem__, np.where(mhd)[1] + 1))).reshape(-1, 3)))
# %%

# %%

# names.extend([f"Blank{i}" for i in range(len(blanks))])

# arr = np.vstack((arr, blanks))
# names = names[~(arr[:, 3] | arr[:, 0])]
# arr = arr[~(arr[:, 0] | arr[:, 3])]
# assert len(arr) == len(mhd)

# codes = pd.DataFrame(arr)
# codes['name'] = cb.keys()
# %%

# cb = pd.read_csv("scripts/working/codebook.csv").iloc[:270]
# codes = cb.iloc[:270, 3:].to_numpy()

# extendedname = []
# extendedcode = []
# for name, code in zip(cb["name"], codes):
#     extendedname.append(name)
#     extendedcode.append(code)
#     for col in range(codes.shape[1]):
#         nc = code.copy()
#         if code[col] == 1:
#             extendedname.append(name + "_10")
#             nc[col] = 0
#         else:
#             extendedname.append(name + "_01")
#             nc[col] = 1
#         extendedcode.append(nc)

# %%

# %%
# psd = DetectPixels.PixelSpotDecoder(
#     codebook=codebook,
#     metric="euclidean",  # distance metric to use for computing distance between a pixel vector and a codeword
#     norm_order=2,  # the L_n norm is taken of each pixel vector and codeword before computing the distance. this is n
#     distance_threshold=0.28,  # minimum distance between a pixel vector and a codeword for it to be called as a gene
#     magnitude_threshold=0.001,  # discard any pixel vectors below this magnitude
#     min_area=5,  # do not call a 'spot' if it's area is below this threshold (measured in pixels)
#     max_area=100,  # do not call a 'spot' if it's area is above this threshold (measured in pixels)
# )


# %%

# %%

# %%


# decoded_spots, prop_results = psd.run(sliced)
# filter spots that do not pass thresholds
# spot_intensities = decoded_spots.loc[decoded_spots[Features.PASSES_THRESHOLDS]]

# [spot_intensities[Features.DISTANCE] < 1]
# %%


# %%


# %%
# import seaborn as sns

# sns.set()
# plt.scatter(
#     decoded_spots.coords["radius"] + rand.normal(0, 0.05, size=decoded_spots.shape[0]),
#     decoded_spots.coords["distance"],
#     c=decoded_spots.coords["passes_thresholds"],
#     # c=np.linalg.norm(decoded_spots, axis=2),
#     # c=np.where(
#     #     is_blank[list(map(names_l.get, initial_spot_intensities.coords["target"].to_index().values))], 1, 0
#     # ).flatten(),
#     alpha=0.2,
#     cmap="bwr",
#     s=2,
# )


# %%


# %%
# dist = spot_intensities[Features.AXIS].to_dataframe().groupby("target").agg(dict(distance="mean"))
# dist["blank"] = dist.index.str.startswith("Blank")

# c = pd.DataFrame.from_dict(gc, orient="index").sort_values(0)
# # c = c[c[0] > 1]
# c["color"] = c.index.str.startswith("Blank")
# c["color"] = c["color"].map({True: "red", False: "blue"})

# fig, ax = plt.subplots(figsize=(8, 10))
# ax.bar(c.index, c[0], color=c["color"], width=1)
# ax.set_xticks([])
# ax.set_yscale("log")

# %%

# set dpi to 200
# fig, ax = plt.subplots(figsize=(8, 10))
# c = c[-50:]
# c_ = pd.concat((c[:30], c[-30:]))
# ax.barh(c_.index, c_[0], color=c_["color"])
# ax.set_xticks(rotation="vertical", fontsize=6)

# %%


# %%


# %%
# genes = list(cb.keys())[:-6]

# codednames = [list(map(genes.index, x.split("-")[1].split(","))) for x in c.index.tolist()]

# X = np.zeros((len(codednames), len(genes)), dtype=float)
# for i, cc in enumerate(codednames):
#     for j in cc:
#         X[i, j] = 1

# import numpy as np
# from scipy.optimize import minimize, nnls

# solved = nnls(X, c[0].values)
# ind = np.arange(len(c))
# fig, ax = plt.subplots(figsize=(8, 10))
# ax.barh(ind + 0.4, c[0].values.tolist(), 0.4, color="green", label="counts")
# ax.barh(ind, X @ solved[0], 0.4, color="magenta", alpha=0.3, label="predicted from NNLS")
# ax.set(yticks=ind + 0.4, yticklabels=c.index, ylim=[2 * 0.4 - 1, len(c)])
# ax.legend()


# def fit(X, params):
#     return X.dot(params)


# def cost_function(params, X, y):
#     return np.sum(np.abs(y - fit(X, params)))


# plt.imshow(np.array([X @ solved[0], c[0].values]).T)


# %%


# View decoded spots overlaid on max intensity projected image
# imgs: ImageStack
# single_plane_max = np.array(imgs.xarray).squeeze()[np.where(arr.sum(axis=0))].max(axis=0)
# plt.imshow(single_plane_max)


def plot_decoded(spots):
    plt.scatter(spots.coords["x"], spots.coords["y"], s=0.1, c="red")
    # plt.imshow(prop_results.label_image[0])
    plt.title("PixelSpotDecoder Labeled Image")
    plt.axis("off")


# .where(spot_intensities.target == "Gad2-201", drop=True)
# plot_decoded(spot_intensities)


# %%

# %%


def plot_bits(spots):
    reference = np.zeros((len(spots), max(bits) + 1))
    for i, arr in enumerate(list(map(cb.get, spots.target.values))):
        for a in arr:
            reference[i, a - 1] = 1

    fig, axs = plt.subplots(figsize=(2, 10), ncols=2, dpi=200)
    axs = axs.flatten()
    axs[0].imshow(spots.squeeze())
    axs[0].axis("off")
    axs[1].imshow(reference)
    axs[1].axis("off")
    return fig, axs


# plot_bits(spot_intensities.where(spot_intensities.target == "Neurod6-201", drop=True))
# plot_bits(spot_intensities[:200])
# %%


# %%

# %%


# df = pd.DataFrame(
#     {
#         "target": spot_intensities.coords["target"],
#         "x": spot_intensities.coords["x"],
#         "y": spot_intensities.coords["y"],
#     }
# )


# sns.scatterplot(data=df[df["target"] == "Neurog2-201"], x="x", y="y", hue="target", s=10, legend=True)

# %%

# # Example of how to access the spot attributes
# print(f"The area of the first spot is {prop_results.region_properties[0].area}")

# # View labeled image after connected componenet analysis
# # View decoded spots overlaid on max intensity projected image
# single_plane_max = imgs.reduce({Axes.ROUND, Axes.CH, Axes.ZPLANE}, func="max")
# fig, axs = plt.subplots(ncols=2, figsize=(10, 5), dpi=200)
# axs[1].imshow(prop_results.label_image[0], vmax=1)
# axs[1].set_title("Decoded", loc="left")


# axs[0].imshow(single_plane_max.xarray[0].squeeze(), cmap="gray")
# axs[0].set_title("Raw", loc="left")
# for ax in axs:
#     ax.axis("off")
# fig.tight_layout()
# Uncomment code below to view spots
# %gui qt
# viewer = display(stack=single_plane_max, spots=spot_intensities)


# %%
# %%
# import seaborn as sns
# from starfish import IntensityTable


# def compute_magnitudes(stack, norm_order=2):
#     pixel_intensities = IntensityTable.from_image_stack(stack)
#     feature_traces = pixel_intensities.stack(traces=(Axes.CH.value, Axes.ROUND.value))
#     norm = np.linalg.norm(feature_traces.values, ord=norm_order, axis=1)

#     return norm


# mags = compute_magnitudes(imgs)

# plt.hist(mags, bins=200)
# sns.despine(offset=3)
# plt.xlabel("Barcode magnitude")
# plt.ylabel("Number of pixels")
# plt.yscale("log")
# plt.xscale("log")
# %%
# %%
# from starfish.core.intensity_table.decoded_intensity_table import DecodedIntensityTable
# from starfish.core.intensity_table.intensity_table import IntensityTable

# pixel_intensities = IntensityTable.from_image_stack(clipped_both_scaled.reduce("z", func="max"))
