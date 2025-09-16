# %%

import json
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np
import rich_click as click
import seaborn as sns
import tifffile
from cupyx.scipy.ndimage import shift
from loguru import logger
from pydantic import BaseModel

from fishtools.preprocess.config import NumpyEncoder
from fishtools.preprocess.fiducial import run_fiducial
from fishtools.postprocess.subtraction import plot_regression, regress
from fishtools.utils.io import get_metadata

sns.set_theme()


class Image(BaseModel):
    class Config:
        arbitrary_types_allowed = True  # Allow Path type

    img: np.ndarray
    fid: np.ndarray
    meta: dict


def get_img(path: Path, nc: int, run_basic: bool = False, subsample_z: int = 8):
    path = Path(path)
    with tifffile.TiffFile(path) as tif:
        img = tif.asarray()
        meta = tif.shaped_metadata[0]
    fid = img[-2:].max(axis=0)
    img = img[:-2].reshape(-1, nc, 2048, 2048)[::subsample_z]
    if run_basic:
        out = np.zeros_like(img)
        waveform = json.loads(meta["waveform"])
        powers = waveform["params"]["powers"]

        for i, ilm in enumerate(powers):
            basic = pickle.loads(
                (path.parent.parent / "basic" / f"{path.stem.split('-')[0]}-{ilm[-3:]}.pkl").read_bytes()
            )
            out[:, i] = basic.transform(img[:, i])
        return Image(img=out, fid=fid, meta=meta)

    return Image(img=img, fid=fid, meta=meta)


def load_img(
    path: Path,
    *,
    roi: str,
    round_: str,
    round_blank: str,
    idx: int,
    nc: int = 3,
):
    # fmt: off
    return (
        get_img(path /      f"{round_}--{roi}" /      f"{round_}-{idx:04d}.tif", nc=nc, run_basic=True),
        get_img(path / f"{round_blank}--{roi}" / f"{round_blank}-{idx:04d}.tif", nc=nc, run_basic=True),
    )
    # fmt: on


def load_imgs(
    path: Path, *, roi: str, round_: str, round_blank: str, n: int = 30, nc: int = 3, seed: int = 0
):
    path = Path(path)
    rand = np.random.default_rng(seed)

    paths = sorted((path / f"{round_}--{roi}").glob("*.tif"))
    paths_blank = sorted((path / f"{round_blank}--{roi}").glob("*.tif"))

    assert all(p.stem.split("-")[1] == pb.stem.split("-")[1] for p, pb in zip(paths, paths_blank))

    selected = set(rand.choice(np.arange(len(paths)), size=n, replace=False))

    paired = [
        (get_img(im, nc=nc, run_basic=True), get_img(bl, nc=nc, run_basic=True))
        for i, (im, bl) in enumerate(zip(paths, paths_blank))
        if i in selected
    ]
    return paired


# %%

# # %%
# def plot(img, ax=None):
#     if ax is None:
#         fig, ax = plt.subplots(1, 1)
#     ax.scatter(img[:, 0].max(axis=0).flatten()[::10], img[:, 1].max(axis=0).flatten()[::10], s=1, c="k")


# %%


@click.command()
@click.argument("path", type=Path)
@click.argument("roi", type=str)
@click.option("--round", "round_", type=str)
@click.option("--round_blank", type=str, default="b560_b650_b750")
@click.option("--n", type=int, default=30)
def main(path: Path, roi: str, round_: str, round_blank: str, n: int = 30):
    def run(paired: tuple[Image, Image]):
        im, bl = paired
        f = run_fiducial(bl.fid, threshold_sigma=3, fwhm=7)
        sh, _ = f(im.fid)
        cimg = cp.asarray(im.img)
        im.img = shift(cimg, [0, 0, sh[1], sh[0]]).get()
        return im, bl

    paireds = load_imgs(path, roi=roi, round_=round_, round_blank=round_blank, n=n)
    logger.info(f"Loaded {len(paireds)} images")
    logger.info(f"Running regression on {len(paireds)} images")

    with ThreadPoolExecutor(max_workers=8) as exc:
        futs = [exc.submit(run, paired) for paired in paireds]
        for fut in as_completed(futs):
            fut.result()

    paireds = [fut.result() for fut in futs]
    shifted = np.stack([im.img for im, _ in paireds]).astype(np.float32)
    shifted_blank = np.stack([bl.img for _, bl in paireds]).astype(np.float32)
    del paireds

    for i in range(2):
        running_slope = []
        running_intercept = []
        running_corr = []

        for j in range(shifted.shape[0]):
            linreg, bin_centers, percentiles = regress(shifted[j, :, i], shifted_blank[j, :, i])
            # y_pred = linreg.predict(bin_centers.reshape(-1, 1))
            slope = linreg.estimator_.coef_[0]
            intercept = linreg.estimator_.intercept_
            logger.info(f"{i}: Slope: {slope:.3f} Intercept: {intercept:.2f}")

            running_slope.append(slope)
            running_intercept.append(intercept)
            running_corr.append(
                np.corrcoef(shifted[j, :, i].flatten(), shifted_blank[j, :, i].flatten())[0, 1]
            )

        Path(path / "subtract").mkdir(exist_ok=True)
        Path(path / "subtract" / f"{round_}-{i}.json").write_text(
            json.dumps(
                {"slopes": running_slope, "intercepts": running_intercept, "corr": running_corr},
                indent=4,
                cls=NumpyEncoder,
            )
        )


@click.command()
@click.argument("path", type=Path)
@click.argument("roi", type=str)
@click.argument("i", type=int)
@click.option("--round", "round_", type=str)
@click.option("--round_blank", type=str, default="b560_b650_b750")
def apply(path: Path, roi: str, i: int, round_: str, round_blank: str):
    def run(paired: tuple[Image, Image]):
        im, bl = paired
        f = run_fiducial(bl.fid, threshold_sigma=3, fwhm=7)
        sh, _ = f(im.fid)

        cimg = cp.asarray(im.img)
        im.img = shift(cimg, [0, 0, sh[1], sh[0]]).astype(cp.float32)
        bl.img = cp.asarray(bl.img).astype(cp.float32)
        im.fid = shift(cp.asarray(im.fid), [0, sh[1], sh[0]])
        return im, bl

    paired = load_img(path, roi=roi, round_=round_, round_blank=round_blank, idx=i)
    im, bl = run(paired)

    for i in range(2):
        res = json.loads(path / "subtract" / f"{round_}-{i}.json").read_text()
        measured = []
        for s, i in zip(res["slopes"], res["intercepts"]):
            if i > 0:
                measured.append((s, i))
        measured = np.array(measured)
        slope, intercept = np.percentile(measured, 50, axis=0)

        im.img[:, i] -= bl.img[:, i] * slope + intercept

    im.img = im.img.get()
    offsets = np.minimum(np.zeros(im.img.shape[1]), np.percentile(im.img, 1, axis=(0, 2, 3)))
    im.img -= offsets
    im.img = im.img.astype(np.uint16)
    out_path = path / "analysis" / "subtracted" / f"{round_}--{roi}" / f"{round_}-{i:04d}.tif"
    tifffile.imwrite(
        out_path,
        im.img,
        compression=22610,
        compressionargs={"level": 0.65},
        metadata=im.meta | {"subtract": {"round_blank": round_blank, "offsets": offsets.tolist()}},
    )
    out_path.with_suffix(".deconv.json").write_text(json.dumps({"offsets": offsets.tolist()}, indent=2))


# %%
path = Path("/working/20250605_2486/analysis/deconv/registered--hippo+laihuman")
img = tifffile.imread(path / "reg-0045.tif")
blank = tifffile.imread(path.parent / "registered--hippo+blank" / "reg-0045.tif")

# %%
from skimage.filters import gaussian

c = -1
g = gaussian(img[:, c], sigma=(2, 2, 2), preserve_range=True)
# %%
subtracted = img[:, c] - np.minimum(g, img[:, c])

# %%
bsubtracted = blank[:, 0] - np.minimum(
    gaussian(blank[:, 0], sigma=(2, 2, 2), preserve_range=True), blank[:, 0]
)
# %%
plt.imshow(subtracted[5], zorder=1, vmax=500)
# %%
w = bsubtracted[5].flatten(), subtracted[5].flatten()
# %%
mask = w[0] > 120
w = w[0][mask], w[1][mask]
# %%
# logged = np.log10(w[0] + 1e-6), np.log10(w[1] + 1e-6)
from sklearn.linear_model import HuberRegressor

linreg = HuberRegressor(epsilon=1.1)
linreg.fit(w[0].reshape(-1, 1), w[1])
# Plot the fit in loglog space
fig, ax = plt.subplots(figsize=(8, 6))

# Plot the data points in log-log space
ax.scatter(np.log10(w[0]), np.log10(w[1]), alpha=0.1, s=1)

# Create prediction line
x_range = np.linspace(w[0].min(), w[0].max(), 100)
y_pred = linreg.predict(x_range.reshape(-1, 1))

# Plot the fit line in log-log space
ax.plot(
    np.log10(x_range),
    np.log10(y_pred),
    "r-",
    linewidth=2,
    label=f"Fit: slope={linreg.coef_[0]:.3f}, intercept={linreg.intercept_:.1f}",
)

ax.set_xlabel("log10(blank)")
ax.set_ylabel("log10(subtracted)")
ax.legend()
ax.grid(True, alpha=0.3)
plt.title("Log-Log Plot of Regression Fit")
plt.show()
# %%
# Apply the subtraction using the fit
corrected = w[1] - linreg.predict(w[0].reshape(-1, 1))
# Ensure corrected values don't go negative
corrected = np.maximum(0, corrected)

# Plot the results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Before correction
ax1.scatter(w[0], w[1], alpha=0.1, s=1)
ax1.set_xlabel("Blank")
ax1.set_ylabel("Original")
ax1.set_title("Before Correction")
ax1.grid(True, alpha=0.3)

# After correction
ax2.scatter(w[0], corrected, alpha=0.1, s=1)
ax2.axhline(y=0, color="r", linestyle="--", alpha=0.5)
ax2.set_xlabel("Blank")
ax2.set_ylabel("Corrected")
ax2.set_title("After Correction")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"Original correlation: {np.corrcoef(w[0], w[1])[0, 1]:.3f}")
print(f"Corrected correlation: {np.corrcoef(w[0], corrected)[0, 1]:.3f}")

# %%
# Apply the learned coefficients to subtract the processed blank from processed image
tosub = np.maximum(bsubtracted[5] * linreg.coef_[0] + linreg.intercept_, 0)
corrected_image = np.maximum(0, subtracted[5].astype(np.float32) - tosub)

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(subtracted[5], vmin=0, vmax=500)
plt.title("Original Subtracted")
plt.colorbar()
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(bsubtracted[5], vmin=0, vmax=500)
plt.title("Blank Subtracted")
plt.colorbar()
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(corrected_image, vmin=0, vmax=500)
plt.title("Final Corrected")
plt.colorbar()
plt.axis("off")

plt.tight_layout()
plt.show()
# %%
from fishtools.utils.io import Workspace, get_channels, get_metadata

get_metadata(path / "reg-0045.tif")

# %%
# %%
res = json.loads(Path("/working/20250312_melphahuman/subtract/4_12_20-1.json").read_text())
# %%
measured = []
for s, i in zip(res["slopes"], res["intercepts"]):
    if i > 0:
        measured.append((s, i))
measured = np.array(measured)
slope, intercept = np.percentile(
    measured,
    50,
    axis=0,
)

fixing = img.img[0, 1]
blanking = blank.img[0, 1]

# %%

linreg, bin_centers, percentiles = regress(fixing, blanking)
# slope = linreg.estimator_.coef_[0]
# intercept = linreg.estimator_.intercept_

plot_regression(fixing, blanking, bin_centers, percentiles, slope, intercept)

# %%
maxed = fixing.astype(np.float32)
maxed -= blanking * slope + intercept
# maxed  = np.clip(maxed, fixing.min(), None)

fig, axs = plt.subplots(1, 3, figsize=(12, 4), dpi=300)
axs[0].imshow(fixing, vmin=0, vmax=5000, zorder=1)
axs[1].imshow(maxed, vmin=-1000, vmax=5000, zorder=1)
axs[2].imshow(blanking, vmin=0, vmax=5000, zorder=1)
# %%
plt.scatter(blanking[::10], maxed[::10])

# %%

# intercept = 1100
# slope = 0.0


# def subtract(path: Path):
#     print(path)
#     with tifffile.TiffFile(path) as tif:
#         img = tif.asarray()
#         meta = tif.shaped_metadata[0]
#         fid = img[-2:]
#         img = img[:-2].reshape(-1, 2, 2048, 2048)

#     maxed = img[:, 0]
#     maxed -= intercept
#     maxed -= np.minimum((img[:, 1] * 0.12).astype(np.uint16), maxed)
#     maxed += intercept
#     img[:, 0] = maxed

#     img = np.concatenate([img.reshape(-1, 2048, 2048), fid], axis=0)

#     tifffile.imwrite1(
#         path.parent.parent / "51_polyA--hippo" / path.name,
#         img,
#         compression=22610,
#         metadata=meta,
#         compressionargs={"level": 0.65},
#     )
#     del img, maxed


# with ThreadPoolExecutor(max_workers=8) as exc:
#     futs = []
#     for path in sorted(Path("/working/20250229_2242_2/51_polyA--spill").glob("*.tif")):
#         if (path.parent / "51_polyA--hippo" / path.name).exists():
#             continue
#         futs.append(exc.submit(subtract, path))
#     for fut in futs:
#         fut.result()

# # %%
if __name__ == "__main__":
    main()
# %%
