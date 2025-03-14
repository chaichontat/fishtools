# %%
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tifffile
from skimage import exposure, feature, filters
from tifffile import imread, imwrite

# plt.imshow = lambda *args, **kwargs: plt.imshow(*args, zorder=1, **kwargs)
sns.set_theme()
roi = "rightcortex"
path = Path(f"/working/20241122-ZNE172-Trc3/registered--{roi}/")

first = next(path.glob("*.tif"))
with tifffile.TiffFile(first) as tif:
    names = tif.shaped_metadata[0]["key"]
    mapping = dict(zip(names, range(len(names))))

mapping
# %%
idx = list(map(mapping.get, ["pi", "wga"]))


def run_filter(img: np.ndarray, filter_: Callable[[np.ndarray], np.ndarray], *, rescale: bool = False):
    max_ = img.max()
    out = filter_(img)
    if rescale:
        return np.uint16(out * (max_ / out.max()))
    return np.uint16(np.clip(out, 0, 65535))


unsharp_mask = partial(filters.unsharp_mask, preserve_range=True, radius=3)
# %%
# %%
img = imread(first)
plt.imshow(img[1, 30], zorder=1, vmax=np.percentile(img[1, 30], 99.9))


# %%
def clahe(img: np.ndarray):
    return (exposure.equalize_adapthist(img, clip_limit=0.015) * 65535).astype(np.uint16)


def run_2d(file: Path):
    with tifffile.TiffFile(file) as tif:
        img = tif.asarray()
        # (file.parent.parent / "pi--left").mkdir(exist_ok=True)
        for i in range(1, img.shape[0], 6):
            out = file.parent.parent.parent / f"segment--{roi}" / (file.stem + f"_small{i // 2 + 1:02d}.tif")
            out.parent.mkdir(exist_ok=True)
            imgout = img[i, idx, 500:-500, 500:-500]
            assert imgout.shape[0] == len(idx)
            imgout[0] = clahe(run_filter(imgout[0], unsharp_mask))
            imgout[1] = clahe(
                run_filter(
                    img[
                        (i - 1 if i > 1 else i) : (i + 2 if i < img.shape[0] - 1 else i + 1),
                        idx[1],
                        500:-500,
                        500:-500,
                    ].max(axis=0),
                    unsharp_mask,
                )
            )

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
def run_3d(file: Path):
    with tifffile.TiffFile(file) as tif:
        img = tif.asarray()
        # (file.parent.parent / "pi--left").mkdir(exist_ok=True)
        print(img.shape)

        out = file.parent.parent / "segment3d--left" / (file.stem + "_small_3d.tif")
        out.parent.mkdir(exist_ok=True)
        imgout = img[:, idx, 500:-500, 500:-500]
        if min(imgout.shape) == 0:
            raise ValueError("Image is all zeros.")

        imgout[:, -3] = run_filter(imgout[:, -3], unsharp_mask)
        imgout[:, -2] = run_filter(imgout[:, -2], unsharp_mask)
        imgout[:, -1] = run_filter(imgout[:, -1], unsharp_mask)

        imwrite(
            out,
            # img.squeeze()[2:-1, 500:-500, 500:-500].reshape(-1, 1, 1048, 1048),
            imgout,
            compression=22610,
            metadata={"axes": "ZCYX"},
            compressionargs={"level": 0.75},
            # imagej=True,
        )
        print(out)


# Not really helpful.
# with ThreadPoolExecutor(8) as exc:
#     futs = [
#         exc.submit(run_3d, file) for file in sorted(path.glob("reg-*.tif")) if not file.stem.endswith("small")
#     ]
#     for fut in futs:
#         fut.result()

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
