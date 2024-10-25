# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from rich import inspect
from tifffile import TiffFile, imread, imwrite

sns.set_theme()


def subtract(a: np.ndarray, b: np.ndarray, offset: int = 0):
    asub = a - offset
    bsub = b - offset
    return np.where(asub >= bsub, asub - bsub, 0)


def spillover_correction(spillee: np.ndarray, spiller: np.ndarray, corr: float):
    return subtract(spillee, spiller * corr, offset=0)


with TiffFile("/mnt/archive/starmap/bleached/1_9_17--full/1_9_17-2174.tif") as tif:
    img = tif.asarray()
    meta = tif.shaped_metadata[0]
# %%

nofid = img[:-1].reshape(14, -1, 2048, 2048).max(axis=0)


with TiffFile("/mnt/archive/starmap/bleached/registered--full/reg-0080.tif") as tif:
    img = tif.asarray()
    meta = tif.shaped_metadata[0]
    name_mapping = {b: i for i, b in enumerate(meta["key"])}


CHANNELS = {
    560: {"1", "2", "3", "4", "5", "6", "7", "8", "25"},
    650: {"9", "10", "12", "13", "14", "15", "16", "26"},
    750: {"17", "18", "19", "20", "21", "22", "23", "24", "27"},
}


# fid = img[0, -1]
nofid = img[0, :].astype(np.float32)
avail_channels = {c: sorted(set(name_mapping) & CHANNELS[c], key=int) for c in CHANNELS}
avail_idxs = {c: list(map(name_mapping.get, avail_channels[c])) for c in avail_channels}

perc = np.percentile(nofid, 20, axis=(1, 2))
scale_factors = np.ones(nofid.shape[0], dtype=np.float32)
for name, idxs in avail_idxs.items():
    currmin = np.min(perc[idxs])
    for i in idxs:
        scale_factors[i] = currmin / perc[i]

nofid *= scale_factors[:, np.newaxis, np.newaxis]
nofid = nofid.astype(np.uint16)
median = {name: np.median(nofid[idxs], axis=0).astype(np.uint16) for name, idxs in avail_idxs.items()}
# %%
# Median subtraction
for name, c in avail_idxs.items():
    nofid[np.s_[c]] = subtract(nofid[np.s_[c]], median[name] * 1.5)

spill_pairs = [(650, 560, 0.95)]
for spiller, spillee, scale in spill_pairs:
    nofid[np.s_[avail_idxs[spillee]]] = spillover_correction(
        nofid[np.s_[avail_idxs[spillee]]], nofid[np.s_[avail_idxs[spiller]]], 0.95
    )


# np.corrcoef(subtract(img[0, c["1"]], bb).flatten(), subtract(img[0, c["9"]], bb650).flatten())
# plt.imshow(subtract(img[0, c["4"]], bb), zorder=1, vmax=4000)
# %%
fig, axs = plt.subplots(ncols=3, figsize=(12, 4), dpi=200, facecolor="black")
# axs[0].imshow(img[0, c["9"]], vmax=6000, zorder=1)
# axs[1].imshow(bb650, vmax=6000, zorder=1)
sl = np.s_[:, :]

axs[0].imshow(nofid[name_mapping["1"]][sl], vmax=1000, zorder=1)
axs[1].imshow(nofid[name_mapping["9"]][sl], vmax=1000, zorder=1)
axs[2].imshow(
    subtract(nofid[name_mapping["1"]], nofid[name_mapping["17"]], offset=0)[sl],
    vmax=6000,
    zorder=1,
)
# %%


# %%
plt.scatter(
    subtract(img[0, c["1"]], bb).flatten()[::1000], subtract(img[0, c["9"]], bb650).flatten()[::1000], s=0.1
)


# %%
plt.imshow(subtract(img[0, 0], img[0, 3] * perc[0] / perc[1]), zorder=1, vmax=4000)

# %%
fig, axs = plt.subplots(ncols=3, figsize=(12, 4), dpi=200, facecolor="black")
axs[0].imshow(img[0, c["4"]], vmax=6000, zorder=1)
axs[1].imshow(img[0, c["b560"]], vmax=6000, zorder=1)
axs[2].imshow(subtract(img[0, c["4"]], img[0, c["b560"]]), vmax=6000, zorder=1)

# %%
img = imread("/mnt/archive/starmap/bleached/registered--full/reg-0080.subtracted.tif")
# %%
fig, axs = plt.subplots(ncols=8, nrows=4, figsize=(12, 8), dpi=200, facecolor="black")
axs = axs.flatten()
for i, ax in zip(range(nofid.shape[0]), axs):
    ax.axis("off")
    ax.imshow(nofid[i], vmax=5000, zorder=1)
# %%
