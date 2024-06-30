# %%
import json
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import reduce
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
from starfish import Codebook, ImageStack, IntensityTable
from starfish.core.intensity_table.intensity_table_coordinates import (
    transfer_physical_coords_to_intensity_table,
)
import subprocess
from starfish.core.spots.DetectPixels.combine_adjacent_features import CombineAdjacentFeatures
from starfish.core.types import Axes, Coordinates, CoordinateValue
from starfish.experiment.builder import FetchedTile, TileFetcher
from starfish.image import Filter
from starfish.types import Axes, Features, Levels
from starfish.util.plot import imshow_plane, intensity_histogram
from tifffile import imread, imwrite

from fishtools.utils.pretty_print import progress_bar


# %%


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
            (1 / np.nan_to_num(scale, nan=1)).reshape(-1, 1, 1, 1, 1),
            dims=("c", "x", "y", "z", "r"),
        )
    ).run(img, in_place=True)


def load_codebook(path: Path):
    cb = json.loads(path.read_text())
    bits = reduce(lambda x, y: x | set(y), cb.values(), set())
    bit_map = np.ones(max(bits) + 1, dtype=int) * 5000
    for i, bit in enumerate(bits):
        bit_map[bit] = i

    arr = np.zeros((len(cb) - 1, len(bits) - 1), dtype=bool)
    for i, v in enumerate(cb.values()):
        if 28 in v:
            continue
        for a in v:
            assert a > 0
            arr[i, bit_map[a]] = 1

    names = np.array(list(cb.keys()))
    is_blank = np.array([n.startswith("Blank") for n in names])[:-1, None]
    arr_zeroblank = arr * ~is_blank
    return (
        Codebook.from_numpy(
            names[:-1],
            n_round=1,
            n_channel=len(bits) - 1,
            data=arr[:, np.newaxis],
        ),
        names,
        arr_zeroblank,
    )


# %%


@click.group()
def cli(): ...


# img = imread("/fast2/3t3clean/analysis/deconv/registered/reg-0056.tif").astype(np.float32)[:, :13]
# scale_factors = np.max(img, axis=(0, 2, 3), keepdims=True)
# img /= scale_factors


@cli.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=Path))
def optimize(path: Path):
    paths = list(path.glob(f"reg*.tif"))
    rand = np.random.default_rng(0)
    selected = rand.choice(range(len(paths)), size=100, replace=False)

    with progress_bar(len(selected)) as callback, ThreadPoolExecutor(8) as exc:
        futs = []
        for i in selected:
            futs.append(
                exc.submit(
                    subprocess.run,
                    [
                        "python",
                        __file__,
                        "run",
                        str(paths[i]),
                        "--scale-file",
                        str(path / "decode_scale.json"),
                        # "--overwrite",
                    ],
                    check=True,
                    capture_output=False,
                )
            )

        for f in as_completed(futs):
            callback()


@cli.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=Path))
@click.option("--round", type=int)
@click.option("--overwrite", is_flag=True)
def combine(path: Path, round: int, overwrite: bool = False):
    if not overwrite and (path / f"round{round}.tif").exists():
        curr = pl.read_csv(path / f"decode-round{round}.csv").to_numpy()
    else:
        paths = list(path.glob(f"reg*.scale.txt"))
        curr = []
        for p in paths:
            curr.append(np.loadtxt(p, dtype=np.float32))
        curr = np.array(curr)
        pl.DataFrame(curr).write_csv(path / f"decode-{round}.csv")

    curr = np.nanmean(np.array(curr), axis=0, keepdims=True)
    out_file = path / "decode_scale.json"

    assert round is not None

    if not overwrite and out_file.exists():
        # Rescale
        out = np.array(json.loads(out_file.read_text()))[:round]
        curr = out[-1] * (curr / np.nanmean(curr))

        out = np.concatenate([out, curr], axis=0)
    else:
        out = curr

    out_file.write_text(json.dumps(out.tolist()))


def initial(img: ImageStack):
    maxed = img.reduce({Axes.ROUND, Axes.ZPLANE, Axes.Y, Axes.X}, func="max")
    return np.array(maxed.xarray).reshape(1, -1)


@cli.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=False, file_okay=True, path_type=Path))
@click.option("--overwrite", is_flag=True)
@click.option("--scale-file", type=click.Path(dir_okay=False, file_okay=True, path_type=Path))
def run(path: Path, scale_file: Path, iters: int = 1, overwrite: bool = False):
    logger.info("Reading files")
    path, scale_file = Path(path), Path(scale_file)
    stack = make_fetcher(path, np.s_[:, :13])
    # In all modes, data below 0 is set to 0.
    # We probably wouldn't need SATURATED_BY_IMAGE here since this is a subtraction operation.
    # But it's there as a reference.
    ghp = Filter.GaussianHighPass(sigma=8, is_volume=False, level_method=Levels.SCALE_SATURATED_BY_IMAGE)
    # dpsf = Filter.DeconvolvePSF(num_iter=2, sigma=1.7, level_method=Levels.SCALE_SATURATED_BY_CHUNK)
    # glp = Filter.GaussianLowPass(sigma=1, is_volume=False, level_method=Levels.SCALE_SATURATED_BY_IMAGE)
    # dpsf.run(imgs, in_place=True)

    # Scaling
    imgs: ImageStack = ghp.run(stack)
    # scale_file = path.with_suffix(".scale.json")
    if not scale_file.exists():
        logger.debug(f"Making scale file.")
        np.savetxt(path.with_suffix(".scale.txt"), initial(imgs))
        return

    scale_factor = np.array(json.loads(scale_file.read_text()), dtype=np.float32)[-1]

    # scale_factor = np.array(scale_all[-1], copy=True)
    # logger.debug(f"Round {len(scale_all)}")
    scale(imgs, scale_factor)

    codebook, names, arr_zeroblank = load_codebook(
        Path("/home/chaichontat/fishtools/starwork3/ordered/tricycleplus.json")
    )

    # Decode
    pixel_intensities = IntensityTable.from_image_stack(imgs)
    logger.info(f"Decoding")
    decoded_intensities = codebook.decode_metric(
        pixel_intensities,
        max_distance=0.25,
        min_intensity=0.001,
        norm_order=2,
        metric="euclidean",
        return_original_intensities=True,
    )

    logger.info(f"Combining")
    caf = CombineAdjacentFeatures(min_area=4, max_area=100, mask_filtered_features=True)
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
    if False:
        names_l = {n: i for i, n in enumerate(names)}
        idxs = list(map(names_l.get, spot_intensities.coords["target"].to_index().values))

        avgs = np.nanmean(spot_intensities.squeeze() * np.where(arr_zeroblank[idxs], 1, np.nan), axis=0)
        np.savetxt(path.with_suffix(".scale.txt"), avgs)

    morph = [
        {"area": prop.area, "centroid": prop.centroid}
        for prop in np.array(image_decoding_results.region_properties)[
            decoded_spots[Features.PASSES_THRESHOLDS]
        ]
    ]

    with open(path.with_suffix(".pkl"), "wb") as f:
        pickle.dump((decoded_spots, morph), f)

    return decoded_spots, image_decoding_results


if __name__ == "__main__":
    cli()
# %%

u = run.callback(
    "/fast2/3t3clean/analysis/deconv/registered/reg-0000.tif",
    Path("/fast2/3t3clean/analysis/deconv/registered/decode_scale.json"),
)
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
