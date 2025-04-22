# %%
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tifffile
from loguru import logger
from scipy.ndimage import zoom
from skimage import exposure, feature, filters
from tifffile import imread, imwrite

# plt.imshow = lambda *args, **kwargs: plt.imshow(*args, zorder=1, **kwargs)
sns.set_theme()
# roi = "tl+atp"
roi = "br+atp"
path = Path(f"/working/20250407_cs3_2/analysis/deconv/registered--{roi}/")
# path = Path(f"/working/20250317_benchmark_mousecommon/analysis/deconv/registered--{roi}/")

first = next(path.glob("*.tif"))
with tifffile.TiffFile(first) as tif:
    names = tif.shaped_metadata[0]["key"]
    mapping = dict(zip(names, range(len(names))))

mapping
# %%
if "polyA" in mapping:
    idx = list(map(mapping.get, ["polyA"]))
elif "wga" in mapping:
    idx = list(map(mapping.get, ["wga", "reddot"]))
elif "reddot" in mapping:
    idx = list(map(mapping.get, ["reddot", "atp"]))
else:
    idx = list(map(mapping.get, ["pi", "atp"]))

if any(map(lambda x: x is None, idx)):
    raise ValueError(f"Index not found: {idx}")


def run_filter(img: np.ndarray, filter_: Callable[[np.ndarray], np.ndarray], *, rescale: bool = False):
    max_ = img.max()
    out = filter_(img)
    if rescale:
        return np.uint16(out * (max_ / out.max()))
    return np.uint16(np.clip(out, 0, 65535))


unsharp_mask = partial(filters.unsharp_mask, preserve_range=True, radius=3)
# %%
# %%
# img = imread(first.parent / "../segment--barrel+pipolyA/reg-0061_small01.tif")
# # %%
# plt.imshow(img[-8, 0], zorder=1, vmax=np.percentile(img[-8, 0], 99.9))


# %%


def run_2d(file: Path):
    with tifffile.TiffFile(file) as tif:
        img = tif.asarray()
        print(img.shape)
        # (file.parent.parent / "pi--left").mkdir(exist_ok=True)
        for i in range(1, img.shape[0], 6):
            out = file.parent.parent / f"segment--{roi}" / (file.stem + f"_small{i // 2 + 1:02d}.tif")
            out.parent.mkdir(exist_ok=True)
            imgout = img[:, idx, ::2, ::2].max(axis=0)
            assert imgout.shape[0] == len(idx)
            # imgout[0] = clahe(run_filter(imgout[0], unsharp_mask))
            # imgout[1] = clahe(
            #     run_filter(
            #         img[
            #             (i - 1 if i > 1 else i) : (i + 2 if i < img.shape[0] - 1 else i + 1),
            #             idx[1],
            #             500:-500,
            #             500:-500,
            #         ].max(axis=0),
            #         unsharp_mask,
            #     )
            # )

            if min(imgout.shape) == 0:
                raise ValueError("Image is all zeros.")

            # imgout[-3] = run_filter(imgout[-3], unsharp_mask)

            imwrite(
                out,
                # img.squeeze()[2:-1, 500:-500, 500:-500].reshape(-1, 1, 1048, 1048),
                imgout,
                compression=22610,
                metadata={"axes": "CYX"},
                compressionargs={"level": 0.75},
                # imagej=True,
            )
            print(out)
    # return imgout


# %%

with ThreadPoolExecutor(8) as exc:
    futs = [
        exc.submit(run_2d, file) for file in sorted(path.glob("reg-*.tif")) if not file.stem.endswith("small")
    ]
    for fut in futs:
        fut.result()


# %%
def unsharp_all(img: np.ndarray):
    return np.stack([run_filter(img[:, i], unsharp_mask) for i in range(img.shape[1])], axis=1)


def add_max_proj(path: Path, img: np.ndarray, max_from: str):
    try:
        with tifffile.TiffFile(
            path.parent.parent / f"{path.parent.name.split('+')[0]}+{max_from}" / path.name
        ) as tif:
            maxed = tif.asarray().max(axis=1, keepdims=True)
    except FileNotFoundError:
        raise FileNotFoundError(f"Registered from codebook {max_from} not found.")
    img = np.concatenate([img, maxed], axis=1)
    _idx = idx.copy()
    _idx.append(img.shape[1] - 1)
    return img, _idx


def run_3d(file: Path, max_from: str | None = None, dz: int = 1):
    # with tifffile.TiffFile(file.parent.parent / "registered--full+mousecommon" / file.name) as tif:
    #     maxed = tif.asarray().max(axis=1)
    # print(maxed.shape)
    logger.info(
        f"Processing 3D {file.name}, {f'max_from={max_from}, ' if max_from is not None else ''}dz={dz}"
    )
    with tifffile.TiffFile(file) as tif:
        img = tif.asarray()
        # (file.parent.parent / "pi--left").mkdir(exist_ok=True)
        # print(img.shape)
    if max_from is not None:
        if len(idx) > 2:
            raise ValueError("Cannot add max_proj for 3D images with more than 2 channels.")
        img, _idx = add_max_proj(file, img, max_from)
    else:
        _idx = idx

    img = unsharp_all(img)

    for i in range(0, img.shape[0], dz):
        out = file.parent.parent / f"segment--{roi}" / (file.stem + f"_small-{i}.tif")
        out.parent.mkdir(exist_ok=True)
        # imgout = np.concatenate(
        #     [img[i, idx, 200:-200:2, 200:-200:2], maxed[[i], 200:-200:2, 200:-200:2]], axis=0
        # )

        if min(img.shape) == 0:
            raise ValueError("Image is all zeros.")

        imwrite(
            out,
            img[i, _idx, 500:-500, 500:-500],
            compression=22610,
            metadata={"axes": "CYX"},
            compressionargs={"level": 0.8},
            # imagej=True,
        )


def run_ortho(file: Path, *, n: int = 200, max_from: str | None = None):
    logger.info(f"Processing {file.name}")
    with tifffile.TiffFile(file) as tif:
        img = tif.asarray()

    ortho_idxs = np.linspace(0.1 * img.shape[2], 0.9 * img.shape[2], n).astype(int)
    if max_from is not None:
        if len(idx) > 2:
            raise ValueError("Cannot add max_proj for 3D images with more than 2 channels.")
        img, _idx = add_max_proj(file, img, max_from)
    else:
        _idx = idx

    anisotropy = 6
    filtered = unsharp_all(img[:, _idx])
    for i in ortho_idxs:
        out = file.parent.parent / f"segment--{roi}" / "ortho" / (file.stem + f"_smallortho-{i}.tif")
        out.parent.mkdir(exist_ok=True)

        zx = zoom(filtered[:, :, i, :], (anisotropy, 1, 1), order=1)
        zy = zoom(filtered[:, :, :, i], (anisotropy, 1, 1), order=1)
        assert zx.shape == zy.shape, (zx.shape, zy.shape)
        for dim, _img in zip(["zx", "zy"], [zx, zy]):
            imwrite(
                out.with_name((file.stem + f"_ortho{dim}-{i}.tif")),
                np.swapaxes(_img, 0, 1),
                compression=22610,
                metadata={"axes": "C" + dim.upper()},
                compressionargs={"level": 0.8},
                # imagej=True,
            )


# %%

with ThreadPoolExecutor(8) as exc:
    futs = [
        exc.submit(run_3d, file, max_from="mousecommon", dz=1) for file in sorted(path.glob("reg-*.tif"))[::5]
    ]
    for fut in futs:
        fut.result()

# %%
with ThreadPoolExecutor(8) as exc:
    futs = [exc.submit(run_ortho, file, max_from="morris") for file in sorted(path.glob("reg-*.tif"))[::20]]
    for fut in futs:
        fut.result()

# %%
# from scipy import ndimage as ndi

# img = imread("/mnt/working/lai/registered--whole_embryo/reg-0073.tif")[4, -3, 700:-700, 700:-700]

# edges1 = filters.unsharp_mask(img, preserve_range=True, radius=3)


# # %%
# edges2 = filters.scharr(img)
# edges2 = np.uint16(edges2 / edges2.max() * 65535)

# # display results
# fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(16, 6))

# ax[0].imshow(img, zorder=1, vmax=np.percentile(img, 99.9))
# ax[0].set_title("noisy image", fontsize=20)

# ax[1].imshow(edges1, zorder=1, vmax=np.percentile(edges1, 99.9))
# ax[1].set_title(r"Canny filter, $\sigma=1$", fontsize=20)

# ax[2].imshow(edges2, zorder=1, vmax=np.percentile(edges2, 99.9))
# ax[2].set_title(r"Canny filter, $\sigma=3$", fontsize=20)

# # %%
# img = imread("/mnt/working/lai/registered--whole_embryo/reg-0073.tif")
# # %%
# %%
