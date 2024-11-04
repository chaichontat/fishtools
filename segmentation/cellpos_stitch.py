# %%
import pickle
from collections.abc import Collection
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
import tifffile
from skimage.measure import label, regionprops, regionprops_table

sns.set_theme()


# %%
# mask00, _, _ = pickle.loads(Path("cellpose-00-00.pkl").read_bytes())
# mask01, _, _ = pickle.loads(Path("cellpose-01-00.pkl").read_bytes())
# %%

# %%
overlap = 100
bsize = 448 * 2

x = meh(imgk[5, -bsize + 2 * overlap : -overlap * 2 - bsize, 2000:])
y = meh(img[5, overlap * 2 : 448 - overlap * 2, 2000:])
# %%
fig, axs = plt.subplots(ncols=2, nrows=1, figsize=(12, 4), dpi=200, facecolor="black")
axs = axs.flatten()
for i, ax in enumerate(axs):
    ax.axis("off")
axs[0].imshow(imgk[5, -448 + 2 * overlap : -overlap * 2, 2000:], vmin=0, vmax=1)
axs[1].imshow(img[5, overlap * 2 : 448 - overlap * 2, 2000:], vmin=0, vmax=1)
# %%
plt.imshow(imgk[5])
# %%
fig, axs = plt.subplots(ncols=1, nrows=2, figsize=(4, 4), dpi=200, facecolor="black")
axs = axs.flatten()
for i, ax in enumerate(axs):
    ax.axis("off")
axs[0].imshow(imgk[5, -896:], vmin=0, vmax=1)
axs[1].imshow(img[5, :896], vmin=0, vmax=1)
plt.tight_layout()
# %%
fig, axs = plt.subplots(ncols=1, nrows=2, figsize=(4, 4), dpi=200, facecolor="black")
axs = axs.flatten()
for i, ax in enumerate(axs):
    ax.axis("off")
# %%
e = 810
center = e // 2
margin = 100


def get_overlapping(
    img: np.ndarray,
    *,
    pos_img: Literal["top", "bottom", "left", "right"],
    total_overlap: int = 810,
    margin: int = 100,
):
    if not pos_img in {"top", "bottom", "left", "right"}:
        raise ValueError(f"Unknown pos_img {pos_img}")

    img = np.atleast_3d(img)
    overlap_slice = slice(-total_overlap, None) if pos_img in {"top", "left"} else slice(total_overlap)

    # center = total_overlap // 2
    # slice_center = slice(center-margin, center+margin)
    if pos_img in {"top", "bottom"}:
        return img[:, overlap_slice, :]  # [:, slice_center, :]
    else:  # mainly to get the alignment right
        return img[:, :, overlap_slice]  # [:, :, slice_center]


def overlap_center(mode: Literal["top", "bottom", "left", "right"], total_overlap: int = 810) -> int:
    """Get the center index of the overlapped region."""
    if not mode in {"top", "bottom", "left", "right"}:
        raise ValueError(f"Unknown pos_img {mode}")
    if mode in {"top", "left"}:
        return -total_overlap // 2
    return total_overlap // 2


def sl_overlap_center(mode: Literal["top", "bottom", "left", "right"], *, total_overlap: int = 810) -> slice:
    """Get the center index of the overlapped region."""
    if not mode in {"top", "bottom", "left", "right"}:
        raise ValueError(f"Unknown pos_img {mode}")
    if mode in {"top", "left"}:
        sl = slice(None, (overlap_center(mode, total_overlap)))
    else:
        sl = slice((overlap_center(mode, total_overlap)), None)
    if mode in {"top", "bottom"}:
        return np.s_[:, sl]
    return np.s_[:, :, sl]


def calc_remove_crossing(
    img: np.ndarray,
    mode: Literal["top", "bottom", "left", "right"],
    total_overlap: int = 810,
    edge_margin: int = 1,
):
    """Remove things that crosses the center."""
    if img.shape.__len__() != 3:
        raise ValueError("Image must be 3D")
    if edge_margin < 1:
        raise ValueError("Edge margin must be positive")

    idx = overlap_center(mode, total_overlap)
    sl = (
        np.s_[:, idx - edge_margin : idx + edge_margin]
        if mode in {"top", "bottom"}
        else np.s_[:, :, idx - edge_margin : idx + edge_margin]
    )
    out = np.unique(img[sl])
    return out[out != 0]


def splice(img1: np.ndarray, img2: np.ndarray, axis: Literal["x", "y"]):
    ax = 1 if axis == "y" else 2
    if img1.shape[2 if axis == "y" else 1] != img2.shape[2 if axis == "y" else 1]:
        raise ValueError(
            f"Image shapes must match on the axis to splice. Got {img1.shape[ax]} and {img2.shape[ax]}"
        )
    pos = ("top", "bottom") if axis == "y" else ("left", "right")
    return np.concatenate([img1[sl_overlap_center(pos[0])], img2[sl_overlap_center(pos[1])]], axis=ax)


# %%
base_dim = 19584, 26070
files = sorted(Path("chunks").glob("masks-00000_*.tif"))

idx = 1
assert len(files) > 1
img_top = tifffile.imread(files[0])[sl_overlap_center("top")]
for i in range(1, files.__len__()):
    print(files[i].name)
    img = tifffile.imread(files[i])

    to_remove_top = calc_remove_crossing(img, "top")
    to_remove_bottom = calc_remove_crossing(img, "bottom")

    print(to_remove)
    img = img * ~np.isin(img, to_remove)

    img_top = splice(img_top, img.astype(np.uint32) | idx << 16, axis="y")
    idx += 1


# %%


def reg_props(img: np.ndarray, to_remove: Collection[int] | None = None, include_coords: bool = False):
    df = pl.DataFrame(
        regionprops_table(img, properties=["label", "centroid"] + (["coords"] if include_coords else []))
    )
    if to_remove is not None:
        return df.filter(pl.col("label").is_in(to_remove))
    return df


margin = 100


def add_col(x: np.ndarray, val: int, column: int):
    x = x.copy()
    x[:, column] += val
    return x


def purge(img: np.ndarray, mode: Literal["top", "bottom", "left", "right"], total_overlap: int = 810):
    to_remove = calc_remove_crossing(img, mode)
    start = overlap_center(mode) - margin
    end = overlap_center(mode) + margin
    sl = np.s_[:, start:end] if mode in {"top", "bottom"} else np.s_[:, :, start:end]

    df = reg_props(img[sl], to_remove, include_coords=True)
    print(to_remove)
    assert len(df) == len(to_remove)

    if not df.is_empty():
        df = df.with_columns(
            coords=pl.col("coords").map_elements(
                lambda x: add_col(x, start if start >= 0 else start + img.shape[1], 1), return_dtype=pl.Object
            )
        )
        img[sl] = img[sl] * ~np.isin(img[sl], to_remove)
    assert reg_props(img[sl], to_remove).is_empty()
    return img, df


def splice_y(
    imgt: np.ndarray,
    imgb: np.ndarray,
    *,
    idxb: int | None = None,
    remove_crossing: bool = True,
    add_crossing_back: bool = True,
):
    if add_crossing_back and not remove_crossing:
        raise ValueError(
            "Cannot add crossing back without removing crossing. Set remove_crossing=True or add_crossing_back=False"
        )

    if remove_crossing:
        imgt, dft = purge(imgt.copy(), "top")
        imgb, dfb = purge(imgb.copy(), "bottom")

    if idxb is not None:
        img = splice(imgt.astype(np.uint32), imgb.astype(np.uint32) | idxb << 16, axis="y")
    else:
        img = splice(imgt, imgb, axis="y")

    if remove_crossing and add_crossing_back:
        dfb = dfb.with_columns(
            coords=pl.col("coords").map_elements(
                lambda x: add_col(x, -imgb.shape[1], column=1), return_dtype=pl.Object
            )
        )
        for row in dfb.iter_rows(named=True):
            coords, label = row["coords"], row["label"]
            img[coords[:, 0], coords[:, 1], coords[:, 2]] = label

    return img


imgt = tifffile.imread("chunks/masks-00000_00000.tif")
imgb = tifffile.imread("chunks/masks-00000_03982.tif")

t = [
    splice_y(imgt, imgb, remove_crossing=False, add_crossing_back=False),
    splice_y(imgt, imgb, add_crossing_back=False),
    splice_y(imgt, imgb),
]


# %%
fig, axs = plt.subplots(ncols=3, nrows=1, figsize=(12, 4), dpi=200)
axs = axs.flatten()
for i, ax in enumerate(axs):
    ax.axis("off")
    ax.imshow(t[i][5][4200:4600, 2000:3000])

# plt.imshow(u[5].astype(np.uint16)[4200:4600, 2500:3000])
# %%

plt.imshow((img_top[5] >> 16).astype(np.uint16)[3800:4200])

# %%

# %%
print(get_overlapping(imgk, pos_img="top")[5].shape)
m = np.clip(
    np.transpose(
        [
            get_overlapping(imgk, pos_img="top")[5],
            get_overlapping(img, pos_img="bottom")[5],
            np.zeros([e, imgk.shape[2]]),
        ],
        [1, 2, 0],
    ),
    0,
    1,
)
plt.imshow(m[:, 2500:3000])
# %%
x = get_overlapping(imgk, pos_img="top")


# %%
top = get_overlapping(imgk, pos_img="top")
bottom = get_overlapping(img, pos_img="bottom")

tt = meh(top)
tb = meh(bottom)
tt
# %%

# def calc_bbox():
# pl.DataFrame(regionprops_table(x, properties=["label", "centroid", "bbox"]))


# t=  img.astype(np.uint32) | idx << 16
splice_p


# %%

# %%


calc_remove_crossing(x, "top")

mask = x * ~np.isin(x, calc_remove_crossing(x, "top"))
