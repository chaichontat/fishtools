# %%
import json
import os
import pickle
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import reduce
from itertools import chain
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import rich_click as click
import starfish
import starfish.data
import xarray as xr
from loguru import logger
from pydantic import BaseModel, TypeAdapter
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
        cb_json.pop(k)

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


# img = imread("/fast2/3t3clean/analysis/deconv/registered/reg-0056.tif").astype(np.float32)[:, :13]
# scale_factors = np.max(img, axis=(0, 2, 3), keepdims=True)
# img /= scale_factors


def _batch(paths: list[Path], mode: str, args: list[str]):
    with progress_bar(len(paths)) as callback, ThreadPoolExecutor(13) as exc:
        futs = []
        for path in paths:
            futs.append(
                exc.submit(
                    subprocess.run,
                    ["python", __file__, mode, str(path), *args],
                    check=True,
                    capture_output=False,
                )
            )

        for f in as_completed(futs):
            callback()


@cli.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=Path))
@click.option("--round", "round_num", type=int)
@click.option("--batch-size", "-n", type=int, default=100)
def optimize(path: Path, round_num: int, batch_size: int = 100):
    paths = list(path.glob("reg*.tif"))
    rand = np.random.default_rng(0)
    selected = [paths[i] for i in sorted(rand.choice(range(len(paths)), size=batch_size, replace=False))]

    return _batch(
        selected,
        "optimize",
        ["--calc-deviations", "--global-scale", str(path / "global_scale.txt"), f"--round={round_num}"],
    )


@cli.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=Path))
@click.option("--global-scale", type=click.Path(dir_okay=False, file_okay=True, path_type=Path))
def batch(path: Path, global_scale: Path):
    return _batch(
        sorted(path.glob("reg*.tif")),
        "run",
        ["--global-scale", global_scale.as_posix()],
    )


def load_2d(path: Path | str, *args, **kwargs):
    return np.atleast_2d(np.loadtxt(path, *args, **kwargs))


@cli.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=Path))
@click.option("--round", "round_num", type=int)
def combine(path: Path, round_num: int):
    # name = f"decode_{round_num:02d}.csv"

    paths = list(path.glob("reg*.json"))
    curr = []
    n = 0
    for p in paths:
        sf = Deviations.validate_json(p.read_text())
        if (round_num + 1) > len(sf):
            raise ValueError(f"Round number {round_num} exceeds what's available ({len(sf)}).")

        if round_num == 0:
            curr.append(sf[round_num].initial_scale)
        else:
            curr.append(np.array(sf[round_num].deviation) * sf[round_num].n)
            n += sf[round_num].n
    curr = np.array(curr)
    # pl.DataFrame(curr).write_csv(path / name)

    global_scale_file = path / "global_scale.txt"
    if round_num == 0:
        curr = np.nanmean(np.array(curr), axis=0, keepdims=True)
        np.savetxt(global_scale_file, curr)
        return

    if not global_scale_file.exists():
        raise ValueError("Round > 0 requires global scale file.")

    deviation = curr.sum(axis=0) / n
    # Normalize per channel
    deviation = deviation / np.nanmean(deviation)
    mae = np.mean(np.absolute(deviation - 1))
    logger.info(f"Round {round_num}. MAE: {mae:04f}.")

    previous = load_2d(global_scale_file)
    np.savetxt(
        global_scale_file,
        np.concatenate(
            [np.atleast_2d(previous[:round_num]), np.atleast_2d(previous[round_num - 1] / deviation)], axis=0
        ),
    )
    (path / "mae.txt").open("a").write(f"{round_num:02d}\t{mae:04f}\n")


def initial(img: ImageStack):
    maxed = img.reduce({Axes.ROUND, Axes.ZPLANE, Axes.Y, Axes.X}, func="max")
    res = np.array(maxed.xarray).squeeze()
    if np.isnan(res).any() or (res == 0).any():
        raise ValueError("NaNs or zeros found in initial scaling factor.")
    return res


class Deviation(BaseModel):
    n: int
    deviation: list[float]


class InitialScale(BaseModel):
    initial_scale: list[float]


Deviations = TypeAdapter(list[InitialScale | Deviation])


def append_json(
    path: Path,
    round_num: int,
    *,
    n: int | None = None,
    deviation: np.ndarray | None = None,
    initial_scale: np.ndarray | None = None,
):
    existing = Deviations.validate_json(path.read_text()) if path.exists() else Deviations.validate_json("[]")
    if initial_scale is not None:
        if round_num > 0:
            raise ValueError("Cannot set initial scale for round > 0")
        existing.append(InitialScale(initial_scale=initial_scale.tolist()))

    elif deviation is not None and n is not None:
        if round_num == 0:
            raise ValueError("Cannot set deviation for round 0")
        if existing.__len__() < round_num - 1:
            raise ValueError("Round number exceeds number of existing rounds.")

        existing = existing[:round_num]
        existing.append(Deviation(n=n, deviation=deviation.tolist()))
    else:
        raise ValueError("Must provide either initial_scale or deviation and n.")

    path.write_bytes(Deviations.dump_json(existing))


@cli.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=False, file_okay=True, path_type=Path))
@click.option("--overwrite", is_flag=True)
@click.option("--global-scale", type=click.Path(dir_okay=False, file_okay=True, path_type=Path))
@click.option("--round", "round_num", type=int, default=None)
@click.option("--calc-deviations", is_flag=True)
@click.option("--debug", is_flag=True)
def run(
    path: Path,
    global_scale: Path | None,
    round_num: int | None = None,
    overwrite: bool = False,
    debug: bool = False,
    calc_deviations: bool = False,
):
    if debug:
        logger.remove()
        logger.add(sys.stderr, level="INFO")

    if calc_deviations and round_num is None:
        raise ValueError("Round must be provided for calculating deviations.")

    path_pickle = (
        path.with_name(f"{path.stem}_{round_num:02d}.pkl") if calc_deviations else path.with_suffix(".pkl")
    )
    if path_pickle.exists() and not overwrite:
        logger.info(f"Skipping {path.name}. Already exists.")
        return

    logger.info(f"Reading {path.name}.")
    codebook, used_bits, names, arr_zeroblank = load_codebook(
        Path.home() / "fishtools/starwork3/ordered/genestar.json", exclude={"Malat1-201"}
    )
    used_bits = list(map(str, used_bits))

    # if not path.with_suffix(".highpassed.tif").exists():
    with TiffFile(path) as tif:
        img_keys = tif.shaped_metadata[0]["key"]

    bit_mapping = {k: i for i, k in enumerate(img_keys)}
    # img = tif.asarray()[:, [bit_mapping[k] for k in used_bits]]
    # blurred = gaussian(path, sigma=8)
    # blurred = levels(blurred)  # clip negative values to 0.
    # filtered = image - blurred

    stack = make_fetcher(path, np.s_[:, [bit_mapping[k] for k in used_bits]])
    # In all modes, data below 0 is set to 0.
    # We probably wouldn't need SATURATED_BY_IMAGE here since this is a subtraction operation.
    # But it's there as a reference.
    ghp = Filter.GaussianHighPass(sigma=8, is_volume=False, level_method=Levels.SCALE_SATURATED_BY_IMAGE)
    imgs: ImageStack = ghp.run(stack)

    path_out = path.with_suffix(".json")

    if round_num == 0 and calc_deviations:
        logger.debug("Making scale file.")
        path_out.write_bytes(
            Deviations.dump_json(
                Deviations.validate_python([{"initial_scale": (1 / initial(imgs)).tolist()}])
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

    logger.info("Combining")
    caf = CombineAdjacentFeatures(min_area=8, max_area=100, mask_filtered_features=True)
    decoded_spots, image_decoding_results = caf.run(intensities=decoded_intensities, n_processes=8)
    transfer_physical_coords_to_intensity_table(image_stack=imgs, intensity_table=decoded_spots)

    spot_intensities = decoded_spots.loc[decoded_spots[Features.PASSES_THRESHOLDS]]
    genes, counts = np.unique(
        spot_intensities[Features.AXIS][Features.TARGET],
        return_counts=True,
    )
    gc = dict(zip(genes, counts))
    percent = sum([v for k, v in gc.items() if k.startswith("Blank")]) / counts.sum()
    logger.debug(f"{percent} blank, Total: {counts.sum()}")

    # Deviations
    if calc_deviations and round_num is not None:
        names_l = {n: i for i, n in enumerate(names)}
        idxs = list(map(names_l.get, spot_intensities.coords["target"].to_index().values))

        deviations = np.nanmean(spot_intensities.squeeze() * np.where(arr_zeroblank[idxs], 1, np.nan), axis=0)
        logger.debug(f"Deviations: {deviations}")

        """Concatenate with previous rounds. Will overwrite rounds beyond the current one."""
        append_json(path_out, round_num, deviation=deviations, n=len(spot_intensities))

    morph = [
        {"area": prop.area, "centroid": prop.centroid}
        for prop in np.array(image_decoding_results.region_properties)[
            decoded_spots[Features.PASSES_THRESHOLDS]
        ]
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
