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

from fishtools.preprocess.segmentation import unsharp_all
from fishtools.utils.pretty_print import progress_bar_threadpool

# plt.imshow = lambda *args, **kwargs: plt.imshow(*args, zorder=1, **kwargs)
sns.set_theme()
# roi = "tl+atp"
roi = "left+atp"
path = Path(f"/working/20250526_cs3_cooked/analysis/deconv/registered--{roi}/")
# path = Path(f"/working/20250317_benchmark_mousecommon/analysis/deconv/registered--{roi}/")

first = next(path.glob("*.tif"))
with tifffile.TiffFile(first) as tif:
    names = tif.shaped_metadata[0]["key"]
    mapping = dict(zip(names, range(len(names))))

mapping
# %%
if "polyA" in mapping:
    idx = list(map(mapping.get, ["polyA"]))
elif "wga" in mapping and "reddot" in mapping:
    idx = list(map(mapping.get, ["wga", "reddot"]))
elif "reddot" in mapping:
    idx = list(map(mapping.get, ["reddot", "atp"]))
elif "pi" in mapping:
    idx = list(map(mapping.get, ["pi", "atp"]))
else:
    idx = list(mapping.values())
# idx = [2, 1]

if any(map(lambda x: x is None, idx)):
    raise ValueError(f"Index not found: {idx}")


# %%


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


def run_2d(file: Path, max_from: str | None = None):
    with tifffile.TiffFile(file) as tif:
        logger.info(f"Processing 2D {file.name}, {f'max_from={max_from}, ' if max_from is not None else ''}")
        img = tif.asarray()
        img = np.clip(unsharp_all(img, channel_axis=1), 0, 65530).astype(np.uint16)

        if max_from is not None:
            if len(idx) > 2:
                raise ValueError("Cannot add max_proj for 3D images with more than 2 channels.")
            img, _idx = add_max_proj(file, img, max_from)
        else:
            _idx = idx
        # (file.parent.parent / "pi--left").mkdir(exist_ok=True)
        out = (
            Path("/working/cellpose-training")
            / file.parent.parent.parent.parent.name
            / f"segment--{roi}"
            / (file.stem + "_maxed.tif")
        )

        out.parent.mkdir(exist_ok=True)
        imgout = img[:, _idx].max(axis=0)
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

files = [file for file in sorted(path.glob("reg-*.tif"))[::5] if not file.stem.endswith("small")]
with progress_bar_threadpool(len(files), threads=4, stop_on_exception=True) as submit:
    futs = [submit(run_2d, file, max_from="tricycleplus") for file in files]


# %%


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
    img = np.clip(unsharp_all(img, channel_axis=1), 0, 65530).astype(np.uint16)

    if max_from is not None:
        if len(idx) > 2:
            raise ValueError("Cannot add max_proj for 3D images with more than 2 channels.")
        img, _idx = add_max_proj(file, img, max_from)
    else:
        _idx = idx

    for i in range(0, img.shape[0], dz):
        out = (
            Path("/working/cellpose-training")
            / file.parent.parent.parent.parent.name
            / f"segment--{roi}"
            / (file.stem + f"_small-{i}.tif")
        )
        out.parent.mkdir(exist_ok=True, parents=True)

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


def run_ortho(file: Path, *, n: int = 100, max_from: str | None = None):
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
    filtered = np.clip(unsharp_all(img[:, _idx], channel_axis=1), 0, 65530).astype(np.uint16)
    for i in ortho_idxs:
        out = (
            Path("/working/cellpose-training")
            / file.parent.parent.parent.parent.name
            / f"segment--{roi}"
            / "ortho"
            / (file.stem + f"_smallortho-{i}.tif")
        )
        out.parent.mkdir(exist_ok=True, parents=True)

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
files = sorted(path.glob("reg-*.tif"), key=lambda p: p.stat().st_size, reverse=True)[::10]
with progress_bar_threadpool(len(files), threads=4, stop_on_exception=True) as submit:
    futs = [submit(run_3d, file, max_from="mousecommon", dz=1) for file in files]


# %%

files = sorted(path.glob("reg-*.tif"), key=lambda p: p.stat().st_size, reverse=True)[::200]
with progress_bar_threadpool(len(files[:5]), threads=8) as submit:
    futs = [submit(run_ortho, file, max_from=None) for file in files[:5]]

# %%
