# %%
import json
import os
import pickle
import re
import shutil
import subprocess
import sys
import threading
import time
import warnings
from datetime import timedelta
from itertools import chain, groupby
from pathlib import Path
from typing import Any, Literal, Mapping, cast

import matplotlib.pyplot as plt
import numpy as np
import rich_click as click
import starfish
import starfish.data
import tifffile
import xarray as xr
from loguru import logger
from pydantic import BaseModel, TypeAdapter, field_validator
from starfish import ImageStack, IntensityTable
from starfish.core.intensity_table.decoded_intensity_table import DecodedIntensityTable
from starfish.core.intensity_table.intensity_table_coordinates import (
    transfer_physical_coords_to_intensity_table,
)
from starfish.core.spots.DetectPixels.combine_adjacent_features import CombineAdjacentFeatures
from starfish.core.types import Axes, Coordinates, CoordinateValue
from starfish.experiment.builder import FetchedTile, TileFetcher
from starfish.image import Filter
from starfish.spots import DecodeSpots, FindSpots
from starfish.types import Axes, Features, Levels
from starfish.util.plot import imshow_plane, intensity_histogram
from tifffile import TiffFile, imread

from fishtools.preprocess.addition import ElementWiseAddition
from fishtools.preprocess.spots.align_batchoptimize import optimize
from fishtools.preprocess.spots.overlay_spots import overlay
from fishtools.preprocess.spots.stitch_spot_prod import stitch
from fishtools.utils.pretty_print import progress_bar_threadpool
from fishtools.utils.utils import git_hash

GPU = os.environ.get("GPU", "1") == "1"
if GPU:
    # logger.info("Using GPU")
    from fishtools.gpu.codebook import Codebook
else:
    from starfish import Codebook
# from fishtools.gpu.codebook import Codebook


class DecodeConfig(BaseModel):
    model_config = {"frozen": True}

    max_distance: float = 0.2
    min_intensity: float = 0.001
    min_area: int = 10
    max_area: int = 200
    use_correct_direction: bool = True


OPTIMIZE_CONFIG = DecodeConfig(
    min_intensity=0.012, max_distance=0.3, min_area=20, max_area=200, use_correct_direction=True
)


os.environ["TQDM_DISABLE"] = "1"


def make_fetcher(path: Path, sl: slice | list[int] = np.s_[:], max_proj: int = False):
    try:
        with warnings.catch_warnings(action="ignore"):
            # ! Normalize to 1. Any operations done after this WILL be clipped in [0, 1].
            img = imread(path).astype(np.float32)[sl] / 65535
    except tifffile.TiffFileError:
        path.unlink()
        raise Exception(f"{path} is corrupted. Deleted. Please rerun register")

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
            if not max_proj:
                return img[self.z, self.c]
            elif max_proj == 1:
                return img[:, self.c].max(axis=0)
            else:
                start = self.z // max_proj
                end = min(start + max_proj, img.shape[0])
                return img[start:end, self.c].max(axis=0)

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
        zplanes=range(img.shape[0])
        if not max_proj
        else [0]
        if max_proj == 1
        else range(img.shape[0] // max_proj),
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


def scale(
    img: ImageStack, scale: np.ndarray[np.float32, Any], mins: np.ndarray[np.float32, Any] | None = None
):
    if mins is not None:
        ElementWiseAddition(
            xr.DataArray(np.nan_to_num(-mins, nan=1).reshape(-1, 1, 1, 1, 1), dims=("c", "x", "y", "z", "r"))
        ).run(img, in_place=True)

    Filter.ElementWiseMultiply(
        xr.DataArray(
            np.nan_to_num(scale, nan=1).reshape(-1, 1, 1, 1, 1),
            dims=("c", "x", "y", "z", "r"),
        )
    ).run(img, in_place=True)


def load_codebook(
    path: Path,
    bit_mapping: dict[str, int],
    exclude: set[str] | None = None,
    simple: bool = False,
):
    cb_json: dict[str, list[int]] = json.loads(path.read_text())
    for k in exclude or set():
        try:
            cb_json.pop(k)
        except KeyError:
            logger.warning(f"Codebook does not contain {k}")

    if simple:
        cb_json = {k: [v[0]] for k, v in cb_json.items()}
        logger.warning("Running in simple mode. Only using the first bit for each gene.")

    # Remove any genes that are not imaged.
    available_bits = sorted(bit_mapping)
    assert len(available_bits) == len(set(available_bits))
    cb_json = {k: v for k, v in cb_json.items() if all(str(bit) in available_bits for bit in v)}

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
def spots(): ...


def _batch(
    paths: list[Path],
    mode: str,
    args: list[str],
    *,
    threads: int = 13,
    split: bool | list[int | None] = False,
):
    if isinstance(split, list):
        split = split
        if any(x is None for x in split):
            raise ValueError("Cannot use None in split as list")
    elif split:
        split = list(range(4))
    else:
        split = [None]

    if not len(paths):
        raise ValueError("No files found.")

    with progress_bar_threadpool(len(paths) * len(split), threads=threads, stop_on_exception=False) as submit:
        for path in paths:
            for s in split:
                submit(
                    subprocess.run,
                    ["python", __file__, mode, str(path), *[a for a in args if a]]
                    + (["--split", str(s)] if s is not None else []),
                    check=True,
                    capture_output=False,
                )


def sample_imgs(path: Path, codebook: str, round_num: int, *, batch_size: int = 50):
    rand = np.random.default_rng(round_num)
    paths = sorted(
        (p for p in path.glob(f"registered--*+{codebook}/reg*.tif") if not p.name.endswith(".hp.tif"))
    )
    if batch_size > len(paths):
        logger.info(f"Batch size {batch_size} is larger than {len(paths)}. Returning all images.")
        return paths
    return [paths[i] for i in sorted(rand.choice(range(len(paths)), size=batch_size, replace=False))]


@spots.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=Path))
@click.option("--round", "round_num", type=int)
@click.option(
    "--codebook",
    "codebook_path",
    type=click.Path(exists=True, dir_okay=False, file_okay=True, path_type=Path),
)
@click.option("--subsample-z", type=int, default=1)
@click.option("--batch-size", "-n", type=int, default=50)
@click.option("--threads", "-t", type=int, default=8)
@click.option("--overwrite", is_flag=True)
@click.option("--split", type=int, default=0)
@click.option("--max-proj", type=int, default=0)
def step_optimize(
    path: Path,
    round_num: int,
    codebook_path: Path,
    batch_size: int = 50,
    subsample_z: int = 1,
    threads: int = 8,
    overwrite: bool = False,
    split: int = 0,
    max_proj: int = False,
):
    if round_num > 0 and not (path / f"opt_{codebook_path.stem}" / "percentiles.json").exists():
        raise Exception("Please run `fishtools find-threshold` first.")

    selected = sample_imgs(path, codebook_path.stem, round_num, batch_size=batch_size)

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
            *(["--overwrite"] if overwrite else []),
            f"--subsample-z={subsample_z}",
            f"--split={split}",
            *([f"--max-proj={max_proj}"] if max_proj else []),
        ],
        threads=threads,
    )


@spots.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=Path))
@click.option("--codebook", type=click.Path(exists=True, dir_okay=False, file_okay=True, path_type=Path))
@click.option("--roi", type=str, default="*")
@click.option("--percentile", type=float, default=40)
@click.option("--overwrite", is_flag=True)
@click.option("--round", "round_num", type=int, default=0)
@click.option("--max-proj", type=int, default=0)
def find_threshold(
    path: Path,
    roi: str,
    codebook: Path,
    percentile: float = 40,
    overwrite: bool = False,
    round_num: int = 0,
    max_proj: int = 0,
):
    SUBFOLDER = "_highpassed"
    paths = sorted(path.glob(f"registered--{roi}+{codebook.stem}/reg*.tif"))
    path_out = path / (f"opt_{codebook.stem}" + (f"--{roi}" if roi != "*" else ""))
    jsonfile = path_out / "percentiles.json"

    rand = np.random.default_rng(0)
    if len(paths) > 50:
        paths = rand.choice(paths, size=50, replace=False)  # type: ignore
    paths = sorted(
        p for p in paths if not (p.parent / SUBFOLDER / f"{p.stem}_{codebook.stem}.hp.tif").exists()
    )

    if not (path_out / "global_scale.txt").exists():
        raise ValueError("Please run align_prod optimize first.")

    if round_num > 0 and not jsonfile.exists():
        raise ValueError(
            "Round_num > 0 but no existing percentiles file found. Please run with round_num=0 first."
        )

    if paths:
        _batch(
            paths,
            "run",
            ["--codebook", str(codebook), "--highpass-only", *(["--overwrite"] if overwrite else [])],
            threads=4,
            split=[0],
        )

    highpasses = list(path.glob(f"registered--{roi}+{codebook.stem}/{SUBFOLDER}/*_{codebook.stem}.hp.tif"))
    logger.info(f"Found {len(highpasses)} images to get percentiles from.")

    norms = {}

    try:
        mins = np.loadtxt(path_out / "global_min.txt", dtype=float).reshape(1, -1, 1, 1)
        global_scale = np.atleast_3d(np.loadtxt(path_out / "global_scale.txt", dtype=float))[
            round_num
        ].reshape(1, -1, 1, 1)
        logger.info(f"Using global scale from round {round_num}")
    except IndexError:
        raise ValueError(
            f"Round {round_num} not found. Please run align_prod optimize with the round_number first."
        )

    for p in sorted(highpasses[:50]):
        logger.debug(f"Processing {p.parent.name}/{p.name}")
        # import cupy as cp

        img = tifffile.imread(p)[:, :, :1024, :1024]  # type: ignore
        if max_proj:
            img = img.max(axis=0)
        # img = cp.asarray(img)
        # img *= cp.asarray(global_scale)
        img -= mins
        img *= global_scale

        norm = np.linalg.norm(img, axis=1)
        del img
        norms[p.parent.parent.name + "-" + p.name] = float(np.percentile(norm, percentile))

    path_out.mkdir(exist_ok=True)

    logger.info(f"Writing to {jsonfile}")
    if not jsonfile.exists():
        jsonfile.write_text(json.dumps([norms], indent=2))
    else:
        prev_norms = json.loads(jsonfile.read_text())
        prev_norms.append(norms)
        jsonfile.write_text(json.dumps(prev_norms, indent=2))


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


@spots.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=Path))
@click.option(
    "--codebook",
    "codebook_path",
    type=click.Path(exists=True, dir_okay=False, file_okay=True, path_type=Path),
)
@click.option("--batch-size", "-n", type=int, default=50)
@click.option("--round", "round_num", type=int)
def combine(path: Path, codebook_path: Path, batch_size: int, round_num: int):
    """Create global scaling factors from `optimize`.

    Part of the channel optimization process.
    This is to be run after `optimize` has been run.

    Round 0: 5th percentile of all scaling factors (lower indicates brighter spots).
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
    selected = sample_imgs(
        path, codebook_path.stem, round_num, batch_size=batch_size * 2 if round_num == 0 else batch_size
    )
    path_opt = create_opt_path(
        path_folder=path, codebook_path=codebook_path, mode="folder", round_num=round_num
    )
    paths = [
        create_opt_path(path_img=p, codebook_path=codebook_path, mode="json", round_num=round_num)
        for p in selected
    ]

    # paths = list((path / codebook_path.stem).glob("reg*.json"))
    curr = []
    mins = []
    n = 0
    for p in paths:
        if not p.exists():
            logger.warning(f"Path {p} does not exist. Skipping.")
            continue
        sf = Deviations.validate_json(p.read_text())
        # if (round_num + 1) > len(sf):
        #     raise ValueError(f"Round number {round_num} exceeds what's available ({len(sf)}).")

        if round_num == 0:
            curr.append([cast(InitialScale, s).initial_scale for s in sf if s.round_num == 0][0])
            mins.append([cast(InitialScale, s).mins for s in sf if s.round_num == 0][0])
        else:
            try:
                want = cast(Deviation, [s for s in sf if s.round_num == round_num][0])
            except IndexError:
                logger.warning(f"No deviation for round {round_num} in {p}. Skipping.")
                continue
            if want.n < 200:
                logger.debug(f"Skipping {p} at round {round_num} because n={want.n} < 200.")
                continue
            curr.append(np.nan_to_num(np.array(want.deviation, dtype=float), nan=1) * want.n)
            n += want.n
    curr, mins = np.array(curr), np.array(mins)
    # pl.DataFrame(curr).write_csv(path / name)

    global_scale_file = path_opt / "global_scale.txt"
    global_min_file = path_opt / "global_min.txt"
    if round_num == 0:
        # ! The result after scaling must be in [0, 1].
        curr = np.mean(np.array(curr), axis=0, keepdims=True)
        overallmean = np.nanmean(curr)
        curr /= overallmean
        curr = np.clip(curr, 0, 2)
        curr /= np.nanmean(curr)
        curr = np.clip(curr, 0.5, None)
        curr /= np.nanmean(curr)
        np.savetxt(global_scale_file, curr)

        np.savetxt(global_min_file, np.mean(mins, axis=0))
        return

    if not global_scale_file.exists() or not global_min_file.exists():
        raise ValueError("Round > 0 requires global scale file and global min file.")

    deviation = curr.sum(axis=0) / n
    # Normalize per channel
    deviation = deviation / np.nanmean(deviation)
    cv = np.sqrt(np.mean(np.square(deviation - 1)))
    logger.info(f"Round {round_num}. CV: {cv:04f}.")

    previouses = load_2d(global_scale_file)
    old = previouses[round_num - 1]
    new = old / deviation
    # if round_num > 3:
    #     new = (old + new) / 2
    # β = 0.1
    # velocity = old - previouses[round_num - 2]
    # velocity = β * velocity + (1 - β) * grad
    # new = old + velocity

    np.savetxt(
        global_scale_file,
        np.concatenate([np.atleast_2d(previouses[:round_num]), np.atleast_2d(new)], axis=0),
    )
    (path_opt / "mse.txt").open("a").write(f"{round_num:02d}\t{cv:04f}\n")


def initial(img: ImageStack, percentiles: tuple[float, float] = (1, 99.99)):
    """
    Create initial scaling factors.
    Returns the max intensity of each channel.
    Seems to work well enough.
    """
    maxed = img.reduce({Axes.ROUND, Axes.ZPLANE}, func="max")
    res = np.percentile(np.array(maxed.xarray).squeeze(), percentiles, axis=(1, 2))
    # res = np.array(maxed.xarray).squeeze()
    if np.isnan(res).any() or (res == 0).any():
        raise ValueError("NaNs or zeros found in initial scaling factor.")
    return res


class Deviation(BaseModel):
    n: int
    deviation: list[float | None]
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
    mins: list[float]

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
    mins: np.ndarray | None = None,
    percent_blanks: float | None = None,
):
    existing = Deviations.validate_json(path.read_text()) if path.exists() else Deviations.validate_json("[]")
    if initial_scale is not None and mins is not None:
        if round_num > 0:
            raise ValueError("Cannot set initial scale for round > 0")
        existing.append(
            InitialScale(initial_scale=initial_scale.tolist(), round_num=round_num, mins=mins.tolist())
        )
    elif (initial_scale is None) ^ (mins is None):
        raise ValueError("Must provide both initial_scale and mins or not")
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


def pixel_decoding(imgs: ImageStack, config: DecodeConfig, codebook: Codebook, *, gpu: bool = False):
    pixel_intensities = IntensityTable.from_image_stack(imgs)
    logger.info("Decoding")
    decoded_intensities = codebook.decode_metric(
        pixel_intensities,
        max_distance=config.max_distance,
        min_intensity=config.min_intensity,
        norm_order=2,
        metric="euclidean",
        return_original_intensities=True,
    )

    logger.info("Combining")
    caf = CombineAdjacentFeatures(
        min_area=config.min_area, max_area=config.max_area, mask_filtered_features=True
    )
    decoded_spots, image_decoding_results = caf.run(intensities=decoded_intensities, n_processes=8)
    transfer_physical_coords_to_intensity_table(image_stack=imgs, intensity_table=decoded_spots)
    return decoded_spots, image_decoding_results


def pixel_decoding_gpu(imgs: ImageStack, config: DecodeConfig, codebook: Codebook, lock: threading.Lock):
    pixel_intensities = IntensityTable.from_image_stack(imgs)
    shape = imgs.xarray.squeeze().shape  # CZYX
    shape = (shape[1], shape[0], *shape[2:])
    logger.info("Decoding")
    decoded_intensities = codebook.decode_metric(
        pixel_intensities,
        max_distance=config.max_distance,
        min_intensity=config.min_intensity,
        norm_order=2,
        metric="euclidean",
        return_original_intensities=True,
        shape=shape,
        lock=lock,
    )

    logger.info("Combining")
    caf = CombineAdjacentFeatures(
        min_area=config.min_area, max_area=config.max_area, mask_filtered_features=True
    )
    decoded_spots, image_decoding_results = caf.run(intensities=decoded_intensities, n_processes=8)
    transfer_physical_coords_to_intensity_table(image_stack=imgs, intensity_table=decoded_spots)
    return decoded_spots, image_decoding_results


def spot_decoding(
    imgs: ImageStack,
    codebook: Codebook,
    spot_diameter: int = 7,
    min_mass: float = 0.1,
    max_size: int = 2,
    separation: int = 8,
    noise_size: float = 0.65,
    percentile: int = 0,
) -> DecodedIntensityTable:
    # z project
    # max_imgs = imgs.reduce({Axes.ZPLANE}, func="max")
    # tlmpf = starfish.spots.FindSpots.LocalMaxPeakFinder(
    #     spot_diameter=spot_diameter,
    #     min_mass=min_mass,
    #     max_size=max_size,
    #     separation=separation,
    #     noise_size=noise_size,
    #     percentile=percentile,
    #     verbose=True,
    # )

    # run LocalMaxPeakFinder on max projected image
    bd = FindSpots.BlobDetector(
        min_sigma=1,
        max_sigma=4,
        num_sigma=10,
        threshold=0.03,
        is_volume=True,
    )
    # bd = FindSpots.BlobDetector(
    #     min_sigma=2, max_sigma=5, num_sigma=10, threshold=0.01, is_volume=True, measurement_type="mean"
    # )
    spots = bd.run(image_stack=imgs)  # , reference_image=dots)
    # spots = lmp.run(max_imgs)

    decoder = DecodeSpots.SimpleLookupDecoder(codebook=codebook)
    return decoder.run(spots=spots)


@spots.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=False, file_okay=True, path_type=Path))
@click.option("--overwrite", is_flag=True)
# @click.option("--roi", type=str, default=None)
@click.option("--global-scale", type=click.Path(dir_okay=False, file_okay=True, path_type=Path))
@click.option("--round", "round_num", type=int, default=None)
@click.option("--calc-deviations", is_flag=True)
@click.option("--subsample-z", type=int, default=1)
@click.option("--limit-z", default=None)
@click.option("--debug", is_flag=True)
@click.option("--split", default=None)
@click.option("--highpass-only", is_flag=True)
@click.option("--max-proj", type=int, default=0)
@click.option(
    "--codebook",
    "codebook_path",
    type=click.Path(exists=True, dir_okay=False, file_okay=True, path_type=Path),
)
@click.option("--simple", is_flag=True)
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
    highpass_only: bool = False,
    simple: bool = False,
    max_proj: int = 0,
    config: DecodeConfig = DecodeConfig(),
    lock: None = None,
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

    # only for optimize
    path_json = create_opt_path(path_img=path, codebook_path=codebook_path, mode="json", round_num=round_num)
    if calc_deviations:
        config = OPTIMIZE_CONFIG
        logger.info("Optimizing. Using optimize config.")
        path_pickle = create_opt_path(
            path_img=path, codebook_path=codebook_path, mode="pkl", round_num=round_num
        )
        if path_json.exists() and any(
            round_num == v.round_num for v in Deviations.validate_json(path_json.read_text())
        ):
            logger.info(f"Skipping {path.name}. Already done.")
            return

    else:
        (_path_out := path.parent / ("decoded-" + codebook_path.stem)).mkdir(exist_ok=True)
        path_pickle = _path_out / f"{path.stem}{f'-{split}' if split is not None else ''}.pkl"

    if path_pickle.exists() and not overwrite:
        logger.info(f"Skipping {path.name}. Already exists.")
        return

    logger.info(
        f"Running {path.parent.name}/{path.name} {f'split {split}' if split is not None else ''} with {limit_z} z-slices and {subsample_z}x subsampling."
    )
    with TiffFile(path) as tif:
        img_keys = tif.shaped_metadata[0]["key"]

    bit_mapping = {str(k): i for i, k in enumerate(img_keys)}

    codebook, used_bits, names, arr_zeroblank = load_codebook(
        codebook_path,
        exclude={"Malat1-201"},  # , "Nfib-201", "Stmn1-201", "Ywhae-201", "Sox11-201", "Neurod6-201"},
        simple=simple,
        bit_mapping=bit_mapping,
    )

    used_bits = list(map(str, used_bits))

    # img = tif.asarray()[:, [bit_mapping[k] for k in used_bits]]
    # blurred = gaussian(path, sigma=8)
    # blurred = levels(blurred)  # clip negative values to 0.
    # filtered = image - blurred
    cut = 1024
    # 1998 - 1024 = 974
    split = int(split) if split is not None else None
    if split is None:
        split_slice = np.s_[:, :]
    elif split == 0:
        split_slice = np.s_[:cut, :cut]  # top-left
    elif split == 1:
        split_slice = np.s_[:cut, -cut:]  # top-right
    elif split == 2:
        split_slice = np.s_[-cut:, :cut]  # bottom-left
    elif split == 3:
        split_slice = np.s_[-cut:, -cut:]  # bottom-right
    else:
        raise ValueError(f"Unknown split {split}")

    slc = tuple(
        np.s_[
            ::subsample_z,
            [bit_mapping[k] for k in used_bits if k in bit_mapping],
        ]
    ) + tuple(split_slice)
    stack = make_fetcher(
        path,
        slc,
        max_proj=max_proj,
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
    ghp.sigma = (6 / (max_proj or 1), 8, 8)  # z,y,x
    logger.debug(f"Running GHP with sigma {ghp.sigma}")

    imgs: ImageStack = ghp.run(stack)

    if highpass_only:
        (path.parent / "_highpassed").mkdir(exist_ok=True)

        tifffile.imwrite(
            path.parent / "_highpassed" / f"{path.stem}_{codebook_path.stem}.hp.tif",
            imgs.xarray.to_numpy().squeeze().swapaxes(0, 1),  # ZCYX
            compression="zlib",
            metadata={"keys": img_keys},
        )
        return

    if round_num == 0 and calc_deviations:
        logger.debug("Making scale file.")
        mins, base = initial(imgs, (1, 99.9))
        path_json.write_bytes(
            Deviations.dump_json(
                Deviations.validate_python([
                    {"initial_scale": (1 / (base - mins)).tolist(), "mins": mins.tolist(), "round_num": 0}
                ])
            )
        )
        return

    if not global_scale or not global_scale.exists():
        raise ValueError("Production run or round > 0 requires a global scale file.")

    # Scale factors
    scale_all = np.loadtxt(global_scale)
    if len(scale_all.shape) == 1:
        scale_all = scale_all[np.newaxis, :]

    scale_factor = (
        np.array(scale_all[round_num - 1], copy=True)
        if round_num is not None and calc_deviations
        else np.array(scale_all[-1], copy=True)
    )

    mins = np.loadtxt(global_scale.parent / "global_min.txt")

    try:
        scale(imgs, scale_factor, mins=mins)
    except ValueError:
        # global_scale.unlink()
        raise ValueError("Scale factor dim mismatch. Deleted. Please rerun.")

    # Zero out low norm
    try:
        perc = np.mean(
            list(
                json.loads((path.parent.parent / f"opt_{codebook_path.stem}/percentiles.json").read_text())[
                    (round_num - 1) if round_num is not None and calc_deviations else -1
                ].values()
            )
        )
    except FileNotFoundError as e:
        raise Exception("Please run `fishtools find-threshold` first.") from e

    z_filt = Filter.ZeroByChannelMagnitude(perc, normalize=False)
    imgs = z_filt.run(imgs)

    # (path.parent / "_debug").mkdir(exist_ok=True)
    # tifffile.imwrite(
    #     path.parent / "_debug" / f"{path.stem}_{codebook_path.stem}-{split}.tif",
    #     imgs.xarray.to_numpy().squeeze().swapaxes(0, 1),  # ZCYX
    #     compression="zlib",
    #     metadata={"keys": img_keys},
    # )

    # Decode
    # if GPU and lock is None:
    #     raise Exception("GPU and lock are both None.")
    lock = threading.Lock()
    if not simple:
        decoded_spots, image_decoding_results = (
            pixel_decoding(imgs, config, codebook)
            if not GPU
            else pixel_decoding_gpu(imgs, config, codebook, lock)
        )
        decoded_spots = decoded_spots.loc[decoded_spots[Features.PASSES_THRESHOLDS]]
    else:
        decoded_spots, image_decoding_results = spot_decoding(imgs, codebook), None

    spot_intensities = decoded_spots  # .loc[decoded_spots[Features.PASSES_THRESHOLDS]]
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
        try:
            if not len(spot_intensities):
                logger.warning("No spots found. Skipping.")
                append_json(path_json, round_num, deviation=np.ones(len(scale_all)), n=0, percent_blanks=0)
                return

            spot_intensities = spot_intensities[
                (
                    np.linalg.norm(spot_intensities.squeeze(), axis=1)
                    * (1 - spot_intensities.coords["distance"])
                )
                > config.min_intensity
            ]
            idxs = list(map(names_l.get, spot_intensities.coords["target"].to_index().values))

            deviations = np.nanmean(
                spot_intensities.squeeze() * np.where(arr_zeroblank[idxs], 1, np.nan), axis=0
            )
            logger.debug(f"Deviations: {np.round(deviations / np.nanmean(deviations), 4)}")
        except (np.exceptions.AxisError, ValueError):
            append_json(path_json, round_num, deviation=np.ones(len(scale_factor)), n=0, percent_blanks=0)
        else:
            """Concatenate with previous rounds. Will overwrite rounds beyond the current one."""
            append_json(
                path_json,
                round_num,
                deviation=deviations,
                n=len(spot_intensities),
                percent_blanks=percent_blanks,
            )

    morph = (
        [
            {"area": prop.area, "centroid": prop.centroid}
            for prop in np.array(image_decoding_results.region_properties)
        ]
        if image_decoding_results
        else []
    )
    meta = {
        "fishtools_commit": git_hash(),
        "config": config.model_dump(),
    }

    with path_pickle.open("wb") as f:
        pickle.dump((decoded_spots, morph, meta), f)

    return decoded_spots, image_decoding_results


def parse_duration(duration_str: str) -> timedelta:
    """Parses a duration string like '30m', '2h', '5d' into a timedelta."""
    match = re.fullmatch(r"(\d+)([mhd])", duration_str)
    if not match:
        raise ValueError(f"Invalid duration format: {duration_str}. Use 'Nm', 'Nh', or 'Nd'.")
    value, unit = match.groups()
    value = int(value)
    if unit == "m":
        return timedelta(minutes=value)
    elif unit == "h":
        return timedelta(hours=value)
    elif unit == "d":
        return timedelta(days=value)
    else:
        # This should not happen due to regex matching
        raise ValueError(f"Unknown time unit: {unit}")


@spots.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=Path))
@click.argument("roi", type=str, default="*")
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
@click.option("--simple", is_flag=True)
@click.option("--max-proj", type=int, default=0)
@click.option(
    "--since",
    type=str,
    default=None,
    help="Only process files modified since this duration (e.g., '30m', '2h', '1d').",
)
@click.option("--delete-corrupted", is_flag=True)
def batch(
    path: Path,
    roi: str,
    codebook_path: Path,
    threads: int = 13,
    overwrite: bool = False,
    subsample_z: int = 1,
    limit_z: int | None = None,
    split: bool = False,
    simple: bool = False,
    max_proj: bool = False,
    since: str | None = None,
    delete_corrupted: bool = False,
):
    all_paths = {p for p in path.glob(f"registered--{roi}+{codebook_path.stem}/reg*.tif")}
    paths_in_scope = all_paths  # Start with all paths

    if delete_corrupted:
        logger.info(f"Checking all files for corruption.")
        for i, p in enumerate(sorted(all_paths)):
            if i % 100 == 0:
                logger.info(f"Checked {i}/{len(all_paths)} files.")
            try:
                imread(p)
            except Exception as e:
                logger.warning(f"Error reading {p}: {e}. Deleting.")
                p.unlink()
                continue

    # Filter by modification time if --since is provided
    if since:
        try:
            duration = parse_duration(since)
        except ValueError as e:
            logger.error(f"Error parsing --since value: {e}")
            return  # Or raise click.BadParameter

        current_time = time.time()
        cutoff_time = current_time - duration.total_seconds()
        recent_paths = {p for p in all_paths if p.stat().st_mtime > cutoff_time}
        logger.info(
            f"Found {len(recent_paths)} files modified since {since} (out of {len(all_paths)} total)."
        )
        paths_in_scope = recent_paths  # Update paths_in_scope to only include recent ones
    else:
        logger.info(f"Found {len(paths_in_scope)} files matching pattern (no --since filter).")

    already_done = set()
    # Check which of the files *in scope* are already done
    for p in paths_in_scope:
        # Check if 4 corresponding pkl files exist for this tif file if split is True
        num_expected_pkl = 4 if split else 1
        glob_pattern = f"decoded-{codebook_path.stem}/{p.stem}{'-*' if split else ''}.pkl"
        # Use parent.parent because the decoded files are one level up
        if len(list(p.parent.glob(glob_pattern))) == num_expected_pkl:
            already_done.add(p)

    if overwrite:
        # Overwrite only applies to files within the scope (all or recent)
        paths_to_process = sorted(list(paths_in_scope))
        logger.info(f"Processing {len(paths_to_process)} files in scope (overwrite enabled).")
        # Log how many of these were already done but will be overwritten
        overwritten_count = len(already_done)
        if overwritten_count > 0:
            logger.info(f"Overwriting {overwritten_count} already processed files within the scope.")
    else:
        # Process only files in scope that are not already done
        paths_to_process = sorted(list(paths_in_scope - already_done))
        skipped_count = len(paths_in_scope) - len(paths_to_process)
        if skipped_count > 0:
            logger.info(f"Skipping {skipped_count} already processed files within the scope.")
        logger.info(f"Processing {len(paths_to_process)} new files in scope.")

    if not paths_to_process:
        logger.warning("No files found to process based on current filters and overwrite status.")
        return

    return _batch(
        paths_to_process,
        "run",
        [
            "--global-scale",
            (path / f"opt_{codebook_path.stem}" / "global_scale.txt").as_posix(),
            "--codebook",
            codebook_path.as_posix(),
            "--overwrite" if overwrite else "",
            f"--subsample-z={subsample_z}",
            f"--limit-z={limit_z}",
            "--simple" if simple else "",
            f"--max-proj={max_proj}" if max_proj else "",
        ],
        threads=threads,
        split=split,
    )


spots.add_command(optimize)
spots.add_command(stitch)
spots.add_command(overlay)

if __name__ == "__main__":
    spots()
# %%


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
