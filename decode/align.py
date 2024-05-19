# %%
import functools
import pickle
from copy import deepcopy
from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
import starfish
from imageio import volread, volwrite
from scipy.ndimage import shift
from skimage.registration import phase_cross_correlation
from starfish import FieldOfView, ImageStack, data, display
from starfish.core.types import Axes, Coordinates, CoordinateValue
from starfish.experiment.builder import FetchedTile, TileFetcher
from starfish.image import Filter
from starfish.spots import DetectPixels
from starfish.types import Axes, Features, Levels
from tifffile import imread, imwrite

# We use this to cache images across tiles.  To avoid reopening and decoding the TIFF file, we use a
# single-element cache that maps between file_path and the array data.


# %%

img = imread("/fast2/3t3clean/down2/full_1000.tif")


class DemoFetchedTile(FetchedTile):
    def __init__(self, z, chs, *args, **kwargs):
        self.z = z
        self.c = chs

    @property
    def shape(self) -> Mapping[Axes, int]:
        return {
            Axes.Y: img.shape[1],
            Axes.X: img.shape[2],
        }

    @property
    def coordinates(self) -> Mapping[str | Coordinates, CoordinateValue]:
        return {
            Coordinates.X: (0, 0.001),
            Coordinates.Y: (0, 0.001),
            Coordinates.Z: (0.001 * self.z, 0.001 * (self.z + 1)),
        }

    def tile_data(self) -> np.ndarray:
        return img[self.c]  # [512:1536, 512:1536]


class DemoTileFetcher(TileFetcher):
    def get_tile(self, fov_id: int, round_label: int, ch_label: int, zplane_label: int) -> FetchedTile:
        return DemoFetchedTile(zplane_label, ch_label)


stack = ImageStack.from_tilefetcher(
    DemoTileFetcher(),
    {
        Axes.X: img.shape[2],
        Axes.Y: img.shape[1],
    },
    fov=0,
    rounds=range(1),
    chs=range(img.shape[0]),
    zplanes=range(1),
    group_by=(Axes.CH, Axes.ZPLANE),
)
print(repr(stack))

import json

import matplotlib.pyplot as plt
import pandas as pd
import starfish.data
from starfish import Codebook, FieldOfView
from starfish.image import Filter
from starfish.spots import DetectPixels, FindSpots
from starfish.types import Axes, Levels
from starfish.util.plot import imshow_plane, intensity_histogram


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


# %%
imgs = stack.reduce({Axes.ROUND, Axes.ZPLANE}, func="max")
# imgs.xarray - stack.reduce({Axes.X, Axes.Y}, func="min").xarray

# %%
# masking_radius = 5
# filt = Filter.WhiteTophat(masking_radius, is_volume=False)
# filtered = filt.run(imgs, verbose=True, in_place=False)

ghp = Filter.GaussianHighPass(sigma=8)
dpsf = Filter.DeconvolvePSF(num_iter=5, sigma=1, level_method=Levels.SCALE_SATURATED_BY_CHUNK)
glp = Filter.GaussianLowPass(sigma=1)
imgs = ghp.run(stack)
ghp.run(imgs, in_place=True)
dpsf.run(imgs, in_place=True)


# %%

# %%
n_chans = 24
scale_factors = [np.percentile(imgs.get_slice({Axes.CH: i})[0].squeeze(), 90) for i in range(n_chans)]
for i, v in enumerate(scale_factors):
    if v < 1e-6:
        scale_factors[i] = 1

# %%
import xarray as xr

Filter.ElementWiseMultiply(
    xr.DataArray(
        (1 / np.array(scale_factors) / 1000).reshape(n_chans, 1, 1, 1, 1),
        dims=("c", "x", "y", "z", "r"),
    )
).run(imgs, in_place=True)
imgs.xarray
# %%


# %%

cptz_2 = Filter.ClipPercentileToZero(p_min=50, p_max=99.999, level_method=Levels.SCALE_BY_CHUNK)
# cptz_2 = Filter.MatchHistograms({Axes.CH, Axes.ROUND})


clipped_both_scaled = cptz_2.run(imgs.reduce("z", "max"), in_place=False)


# %%
def fuck(img):
    return img.fillna(0)


clipped_both_scaled = clipped_both_scaled.apply(fuck)
# plot_intensity_histograms(stack=clipped_both_scaled, r=0)
# %%
# %%
fig, axs = plt.subplots(4, 6)
axs = axs.flatten()
for i in range(n_chans):
    intensity_histogram(
        clipped_both_scaled,
        sel={Axes.ROUND: 0, Axes.CH: i},
        log=True,
        bins=50,
        ax=axs[i],
        title=f"ch: {i+1}",
    )
fig.tight_layout()

# %%
import xarray as xr

orig_plot: xr.DataArray = stack.sel({Axes.CH: 3, Axes.ZPLANE: 0}).xarray.squeeze()
wth_plot: xr.DataArray = clipped_both_scaled.sel({Axes.CH: 3, Axes.ZPLANE: 0}).xarray.squeeze()

f, (ax1, ax2) = plt.subplots(ncols=2, dpi=200)
ax1.imshow(orig_plot)
ax1.set_title("original")
ax2.imshow(wth_plot)
ax2.set_title("with filtered")
ax1.axis("off")
ax2.axis("off")
# %%
imgs = clipped_both_scaled

# %%

import json
from collections import defaultdict

# %%
from itertools import chain, combinations

import polars as pl

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
cb = json.loads(Path("starwork3/genestar.json").read_text())
mhd = np.loadtxt(f"static/18bit_on3_dist2.csv", delimiter=",", dtype=bool)
from itertools import chain

order = list(chain.from_iterable([[i, i + 8, i + 16] for i in range(1, 9)])) + list(range(25, 33))
mhd = list(map(str, np.array(list(map(order.__getitem__, np.where(mhd)[1] + 1))).reshape(-1, 3)))
print(mhd[0])
# %%
arr = np.zeros((len(cb) - 1, n_chans), dtype=bool)
for i, v in enumerate(cb.values()):
    if 28 in v:
        continue
    for a in v:
        assert a > 0
        arr[i, a - 1] = 1

# %%

existing = {str(arr[i]) for i in range(len(cb) - 1)}

blanks = []
for row in mhd:
    if str(row) not in existing:  # and len(blanks) < 10:
        blanks.append(row)

names = list(cb.keys())
# names.extend([f"Blank{i}" for i in range(len(blanks))])
names = np.array(names)
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
codebook = Codebook.from_numpy(
    names[:-1],
    n_round=1,
    n_channel=n_chans,
    data=arr[:, np.newaxis],
)

# %%
psd = DetectPixels.PixelSpotDecoder(
    codebook=codebook,
    metric="euclidean",  # distance metric to use for computing distance between a pixel vector and a codeword
    norm_order=2,  # the L_n norm is taken of each pixel vector and codeword before computing the distance. this is n
    distance_threshold=0.4096,  # minimum distance between a pixel vector and a codeword for it to be called as a gene
    magnitude_threshold=0.05,  # discard any pixel vectors below this magnitude
    min_area=4,  # do not call a 'spot' if it's area is below this threshold (measured in pixels)
    max_area=100,  # do not call a 'spot' if it's area is above this threshold (measured in pixels)
)

initial_spot_intensities, prop_results = psd.run(clipped_both_scaled)
# filter spots that do not pass thresholds
spot_intensities = initial_spot_intensities.loc[initial_spot_intensities[Features.PASSES_THRESHOLDS]]

# [spot_intensities[Features.DISTANCE] < 1]

genes, counts = np.unique(
    spot_intensities[Features.AXIS][Features.TARGET],
    return_counts=True,
)
gc = dict(zip(genes, counts))
percent = sum([v for k, v in gc.items() if k.startswith("Blank")]) / counts.sum()
print(percent, counts.sum())

# %%
dist = spot_intensities[Features.AXIS].to_dataframe().groupby("target").agg(dict(distance="mean"))
dist["blank"] = dist.index.str.startswith("Blank")

c = pd.DataFrame.from_dict(gc, orient="index").sort_values(0)
c = c[c[0] > 1]
c["color"] = c.index.str.startswith("Blank")
c["color"] = c["color"].map({True: "red", False: "blue"})
# set dpi to 200
fig, ax = plt.subplots(figsize=(8, 10))
c = c[-50:]
ax.barh(c.index, c[0], color=c["color"])
# ax.set_xticks(rotation="vertical", fontsize=6)

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
single_plane_max = imgs.reduce({Axes.ROUND, Axes.CH, Axes.ZPLANE}, func="max")
plt.imshow(single_plane_max.xarray.squeeze(), vmax=0.02)


def plot_decoded(spots):
    plt.scatter(spots.coords["x"], spots.coords["y"], s=0.1, c="red")
    # plt.imshow(prop_results.label_image[0])
    plt.title("PixelSpotDecoder Labeled Image")
    plt.axis("off")


# .where(spot_intensities.target == "Gad2-201", drop=True)
plot_decoded(spot_intensities)

# %%

# %%


def plot_bits(spots):
    reference = np.zeros((len(spots), 24))
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


plot_bits(spot_intensities.where(spot_intensities.target == "Emx2-201", drop=True))
# plot_bits(spot_intensities[:200])
# %%
# View labeled image after connected componenet analysis
names_l = {n: i for i, n in enumerate(names)}

idxs = list(map(names_l.get, spot_intensities.coords["target"].to_index().values))
arr_zeroblank = arr * np.array([not n.startswith("Blank") for n in names])[:, None]

avgs = np.nanmean(spot_intensities.squeeze() * np.where(arr_zeroblank[idxs], 1, np.nan), axis=0)
deviations = avgs / np.mean(avgs)
print(deviations)
# %%
Filter.ElementWiseMultiply(
    xr.DataArray(
        (1 / deviations).reshape(n_chans, 1, 1, 1, 1),
        dims=("c", "x", "y", "z", "r"),
    )
).run(imgs, in_place=True)


# %%

# %%
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

df = pd.DataFrame(
    {
        "target": spot_intensities.coords["target"],
        "x": spot_intensities.coords["x"],
        "y": spot_intensities.coords["y"],
    }
)


sns.scatterplot(data=df, x="x", y="y", hue="target", s=10, legend=True)

# %%

# Example of how to access the spot attributes
print(f"The area of the first spot is {prop_results.region_properties[0].area}")

# View labeled image after connected componenet analysis
# View decoded spots overlaid on max intensity projected image
single_plane_max = imgs.reduce({Axes.ROUND, Axes.CH, Axes.ZPLANE}, func="max")
fig, axs = plt.subplots(ncols=2, figsize=(10, 5), dpi=200)
axs[1].imshow(prop_results.label_image[0], vmax=1)
axs[1].set_title("Decoded", loc="left")


axs[0].imshow(single_plane_max.xarray[0].squeeze(), cmap="gray")
axs[0].set_title("Raw", loc="left")
for ax in axs:
    ax.axis("off")
fig.tight_layout()
# Uncomment code below to view spots
# %gui qt
# viewer = display(stack=single_plane_max, spots=spot_intensities)


# %%
# %%
import seaborn as sns
from starfish import IntensityTable


def compute_magnitudes(stack, norm_order=2):
    pixel_intensities = IntensityTable.from_image_stack(stack)
    feature_traces = pixel_intensities.stack(traces=(Axes.CH.value, Axes.ROUND.value))
    norm = np.linalg.norm(feature_traces.values, ord=norm_order, axis=1)

    return norm


mags = compute_magnitudes(imgs)

plt.hist(mags, bins=20)
sns.despine(offset=3)
plt.xlabel("Barcode magnitude")
plt.ylabel("Number of pixels")
plt.yscale("log")
# %%
# %%
from starfish.core.intensity_table.decoded_intensity_table import DecodedIntensityTable
from starfish.core.intensity_table.intensity_table import IntensityTable

pixel_intensities = IntensityTable.from_image_stack(clipped_both_scaled.reduce("z", func="max"))
# %%
from starfish.core.spots.DetectPixels.combine_adjacent_features import (
    CombineAdjacentFeatures,
    ConnectedComponentDecodingResult,
)

decoded_intensities = codebook.decode_metric(
    pixel_intensities,
    max_distance=psd.distance_threshold,
    min_intensity=psd.magnitude_threshold,
    norm_order=psd.norm_order,
    metric=psd.metric,
)

# %%
caf = CombineAdjacentFeatures(min_area=4, max_area=psd.max_area, mask_filtered_features=True)
decoded_spots, image_decoding_results = caf.run(intensities=decoded_intensities, n_processes=32)
# %%
import pickle

# %%
x["transformations"]["650"]
# %%
x["transformations"]["650"]["750"].translation / 0.108
# %%
