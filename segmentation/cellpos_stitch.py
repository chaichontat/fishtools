# %%
import re
from collections.abc import Collection
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
import tifffile
from skimage.measure import regionprops_table

sns.set_theme()
BIT = 23
BIT_MASK = np.array(~((1 << BIT) - 1)).astype(np.uint32)


# %%
# %%
overlap = 100
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
    if pos_img not in {"top", "bottom", "left", "right"}:
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
    if mode not in {"top", "bottom", "left", "right"}:
        raise ValueError(f"Unknown pos_img {mode}")
    if mode in {"top", "left"}:
        return -total_overlap // 2
    return total_overlap // 2


def sl_overlap_center(mode: Literal["top", "bottom", "left", "right"], *, total_overlap: int = 810):
    """Get the center index of the overlapped region."""
    if mode not in {"top", "bottom", "left", "right"}:
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
    edge_margin: int = 2,
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


def _splice(img1: np.ndarray, img2: np.ndarray, axis: Literal["x", "y"]):
    ax = 1 if axis == "y" else 2
    if img1.shape[2 if axis == "y" else 1] != img2.shape[2 if axis == "y" else 1]:
        raise ValueError(
            f"Image shapes must match on the axis to splice. Got {img1.shape[ax]} and {img2.shape[ax]}"
        )
    pos = ("top", "bottom") if axis == "y" else ("left", "right")
    return np.concatenate([img1[sl_overlap_center(pos[0])], img2[sl_overlap_center(pos[1])]], axis=ax)


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
    assert len(df) == len(to_remove)

    if not df.is_empty():
        df = df.with_columns(
            coords=pl.col("coords").map_elements(
                lambda x: add_col(
                    x, start if start >= 0 else start + img.shape[1], 1 if mode in {"top", "left"} else 2
                ),
                return_dtype=pl.Object,
            )
        )
        img[sl] = img[sl] * ~np.isin(img[sl], to_remove)
        #     highest_bit = img.flat[0] & BIT_MASK
        # print(bin(highest_bit))
        # img[sl] = np.where(np.isin(img[sl], to_remove), highest_bit, img[sl])
        # print(reg_props(img[sl], to_remove))
        # assert reg_props(img[sl], to_remove).is_empty()

        assert reg_props(img[sl], to_remove).is_empty()

    return img, df


def splice(
    imgt: np.ndarray,
    imgb: np.ndarray,
    *,
    axis: Literal["x", "y"],
    idxb: int | None = None,
    remove_crossing: bool = True,
    add_crossing_back: bool = True,
    copy: bool = False,
):
    if add_crossing_back and not remove_crossing:
        raise ValueError(
            "Cannot add crossing back without removing crossing. Set remove_crossing=True or add_crossing_back=False"
        )

    if remove_crossing:
        imgt, dft = purge(imgt.copy() if copy else imgt, "left" if axis == "x" else "top")
        imgb, dfb = purge(imgb.copy() if copy else imgb, "right" if axis == "x" else "bottom")

    if idxb is None:
        img = _splice(imgt, imgb, axis=axis)
    else:
        img = _splice(
            imgt.astype(np.uint32),
            np.where(imgb != 0, imgb.astype(np.uint32), np.uint32(idxb << BIT)),
            axis=axis,
        )

    if remove_crossing and add_crossing_back:
        ax = 1 if axis == "y" else 2
        dfb = dfb.with_columns(
            coords=pl.col("coords").map_elements(
                lambda x: add_col(x, -imgb.shape[ax], column=ax), return_dtype=pl.Object
            )
        )
        for row in dfb.iter_rows(named=True):
            coords, label = row["coords"], row["label"]
            img[coords[:, 0], coords[:, 1], coords[:, 2]] = label if idxb is None else (label | idxb << BIT)

    return img


# %%
imgt = tifffile.imread("chunks/masks-00000_03982.tif")
imgb = tifffile.imread("chunks/masks-00000_07964.tif")

t = [
    splice(imgt, imgb, axis="y", remove_crossing=False, add_crossing_back=False, copy=True),
    splice(imgt, imgb, axis="y", add_crossing_back=False, copy=True),
    splice(imgt, imgb, axis="y", copy=True),
]

fig, axs = plt.subplots(ncols=3, nrows=1, figsize=(12, 4), dpi=200)
axs = axs.flatten()
for i, ax in enumerate(axs):
    ax.axis("off")
    ax.imshow(t[i][5, 4200:4600, 2500:3000].astype(np.uint16))

# %%


# %%

# %%


base_dim = 19584, 26070
files = sorted(Path("chunks").glob("masks-*.tif"))

file_names = pl.DataFrame([
    (x.as_posix(), *re.search(r"masks-(\d+)_(\d+).tif", x.name).groups()) for x in files
]).transpose()

for i, (idx_y, ys) in enumerate(sorted(file_names.group_by("column_1"), key=lambda x: int(x[0][0]))):
    ys = ys.sort("column_1")

    imgt = tifffile.imread(ys[0, "column_0"]).astype(np.uint32) | ((i * len(ys)) << BIT)
    for j, row in enumerate(ys[1:].iter_rows(named=True), 1):
        print(row["column_0"])
        imgb = tifffile.imread(row["column_0"])
        imgt = splice(imgt, imgb, axis="y", idxb=i * len(ys) + j)
    print("writing to file")
    tifffile.imwrite(f"chunks/spliced-{idx_y[0]}.tif", imgt, compression="zlib")


# %%

idx = 1
assert len(files) > 1
img = tifffile.imread("chunks/spliced-07964.tif")
fig, ax = plt.subplots(figsize=(8, 12), dpi=200)
ax.imshow(img[5].astype(np.uint16), zorder=1, vmax=1)


# %%
import re

base_dim = 19584, 26070
out = np.zeros(base_dim)
files = sorted(Path("chunks").glob("spliced-*.tif"))

file_names = pl.DataFrame([
    (x.as_posix(), *re.search(r"spliced-(\d+).tif", x.name).groups()) for x in files
]).transpose()

imgt = tifffile.imread(file_names[0, "column_0"])

for row in sorted(file_names.iter_rows())[1:]:
    print(row[0])
    imgb = tifffile.imread(row[0])
    imgt = splice(imgt, imgb, axis="x")
print("writing to file")
tifffile.imwrite(f"chunks/combi.tif", imgt, compression="zlib")

# %%
u = regionprops_table(imgt[5], properties=["label", "centroid"])

# %%
