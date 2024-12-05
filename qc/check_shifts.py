# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from astropy.modeling import fitting, models
from loguru import logger
from pydantic import BaseModel, TypeAdapter
from scipy.stats import chi2

sns.set_theme()


def mahalanobis_outliers(points: np.ndarray, threshold: float = 0.999):
    mean = np.mean(points, axis=0)
    points = points.squeeze()

    if points.ndim == 1:
        variance = np.var(points)
        diff = points - mean
        distances = np.abs(diff) / np.sqrt(variance)
        cutoff = np.sqrt(chi2.ppf(threshold, df=1))
    elif points.ndim == 2:
        covar = np.cov(points.T)
        diff = points - mean
        distances = np.sqrt(np.sum((diff @ np.linalg.inv(covar)) * diff, axis=1))
        cutoff = np.sqrt(chi2.ppf(threshold, df=2))
    else:
        raise ValueError(f"Points must be 1D or 2D. Got {points.ndim}")

    return distances > cutoff


class Shift(BaseModel):
    shifts: tuple[float, float]
    corr: float
    residual: float


Shifts = TypeAdapter(dict[str, Shift])
# %%
ref = "4_12_20"
roi = "leftleft"
path = Path(f"/mnt/working/20241113-ZNE172-Zach/analysis/deconv/shifts--{roi}")
paths = sorted(path.glob("*.json"))


shifts = {int(path.stem.rsplit("-", 1)[-1]): Shifts.validate_json(path.read_text()) for path in paths}

N_COLS = 4
first = next(iter(shifts.values()))
nrows = int(np.ceil(len(first) / N_COLS))

# %% Check for missing files (registering not finished)
refs = list((path.parent / f"{ref}--{roi}").glob("*.tif"))

if len(paths) != len(refs) and len(refs) != 0:
    missing = {r.stem.rsplit("-", 1)[-1] for r in refs} - {p.stem.rsplit("-", 1)[-1] for p in paths}
    logger.warning(f"Missing {sorted(missing)}")


# %% Shifts
threshold = 0.99
fig, axs = plt.subplots(ncols=N_COLS, nrows=nrows, figsize=(12, 3 * nrows), dpi=200)
outliers_list = set()

for ax, round_ in zip(axs.flat, sorted(first.keys())):
    c1 = np.array([s[round_].shifts for s in shifts.values()])
    lim = max(5, 1.25 * np.abs(c1).max())
    if np.sum(c1) != 0:
        outliers = mahalanobis_outliers(c1, threshold=threshold)
        # estimator = EllipticEnvelope(random_state=0, contamination=0.02)
        # outliers = estimator.fit_predict(c1)
    else:
        outliers = None

    ax.scatter(*c1.T, c=[s[round_].corr for s in shifts.values()], alpha=0.5, s=2, cmap="bwr_r")
    ax.set_aspect("equal")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_title(f"{round_} σ={c1.std(axis=0).mean():.3f}")
    if outliers is None:
        continue

    for is_outlier, (filename, shift) in zip(outliers, shifts.items()):
        if is_outlier:
            ax.text(shift[round_].shifts[0], shift[round_].shifts[1], filename)
            outliers_list.add(filename)
plt.tight_layout()
# %%
# from tifffile import imread, imwrite

# img = imread(path.parent / f"fids_shifted-0067.tif")
# ref_channel = 1  # reference channel index
# sl = np.s_[500:600, 500:600]
# fig, axs = plt.subplots(ncols=N_COLS, nrows=nrows, figsize=(16, 4 * nrows), dpi=200)

# for i, (ax, round_) in enumerate(zip(axs.flat, sorted(first.keys()))):
#     # Normalize both channels to 0-1 range
#     ref_img = img[ref_channel][sl]
#     round_img = img[i][sl]
#     ref_norm = (ref_img - ref_img.min()) / (ref_img.max() - ref_img.min())
#     round_norm = (round_img - round_img.min()) / (round_img.max() - round_img.min())

#     # Create RGB image (R=round, G=reference, B=zeros)
#     rgb = np.dstack([round_norm, ref_norm, np.zeros_like(ref_norm)])

#     # Display
#     ax.imshow(rgb)
#     ax.set_title(f"Channel {round_} vs Reference")
#     ax.axis("off")

# for ax in axs.flat:
#     if not ax.has_data():
#         fig.delaxes(ax)

# plt.tight_layout()
# print(sorted(outliers_list))
# %% Correlation
fig, axs = plt.subplots(ncols=N_COLS, nrows=nrows, figsize=(12, 3 * nrows), dpi=200)

for ax, round_ in zip(axs.flat, sorted(first)):
    c1 = np.array([s[round_].corr for s in shifts.values()])
    ax.hist(c1, linewidth=0)
    ax.set_title(round_)
    ax.set_xlim(-1, 1)

for ax in axs.flat:
    if not ax.has_data():
        fig.delaxes(ax)

plt.tight_layout()

# %%
# Correlation vs L2 distance scatter plot
fig, axs = plt.subplots(ncols=N_COLS, nrows=nrows, figsize=(12, 3 * nrows), dpi=200)

for ax, round_ in zip(axs.flat, sorted(first.keys())):
    # Get correlations
    corrs = np.array([s[round_].corr for s in shifts.values()])

    # Get shifts and calculate L2 distances from mean
    shift_points = np.array([s[round_].shifts for s in shifts.values()])
    mean_shift = np.mean(shift_points, axis=0)
    l2_distances = np.linalg.norm(shift_points - mean_shift, axis=1)

    # Check for outliers using the same threshold
    if np.sum(shift_points) != 0:
        outliers = mahalanobis_outliers(shift_points, threshold=threshold)
    else:
        outliers = None

    # Create scatter plot
    ax.scatter(corrs, l2_distances, c=outliers, alpha=0.5, s=2, cmap="bwr")
    ax.set_xlabel("Correlation")
    ax.set_ylabel("L2 distance from mean")
    ax.set_title(f"{round_}")
    ax.set_xlim(min(0.5, np.min(corrs) - 0.1), 1)
    ax.set_ylim(0, max(2, np.max(l2_distances) + 1))

    # Label outliers
    if outliers is not None:
        for is_outlier, (filename, shift), l2_dist, corr in zip(
            outliers, shifts.items(), l2_distances, corrs
        ):
            if is_outlier:
                ax.text(corr, l2_dist, filename)


for ax in axs.flat:
    if not ax.has_data():
        fig.delaxes(ax)

plt.tight_layout()
# %%

# from scipy import ndimage, optimize
# from scipy.optimize import least_squares
# from scipy.stats import pearsonr


# def optimize_shift(ref: np.ndarray, fid: np.ndarray, initial_guess=(0, 0)) -> tuple[np.ndarray, float]:
#     """
#     Optimize shift parameters to maximize correlation between fixed and moving images.

#     Args:
#         fixed: Reference image (2D array)
#         moving: Image to be shifted (2D array)
#         initial_guess: Starting point for shift optimization

#     Returns:
#         optimal_shift: Optimized shift values (y,x)
#         correlation: Final correlation value
#     """

#     def objective(shifts):
#         # Negative correlation since we want to maximize (optimizer minimizes)
#         shifted = ndimage.shift(fid, shifts, mode="constant", cval=0)
#         corr = np.corrcoef(shifted[50:-50, 50:-50].flatten(), ref[50:-50, 50:-50].flatten())[0, 1]
#         print(corr)
#         return -corr

#     # Optimize using Nelder-Mead
#     result = optimize.shgo(objective, bounds=((-20, 20), (-20, 20)), options={"f_tol": 0.01})

#     # Return optimal shifts and correlation (positive)
#     return result


# %%
from tifffile import imread, imwrite

img = imread(path.parent / "fids_shifted-0067.tif")

# Create RGB comparison images
ref_idx = 1  # assuming first channel is reference, adjust as needed

sl = np.s_[100:600, 100:600]
ref_img = img[ref_idx][sl]
ref_img = np.clip(ref_img, np.percentile(ref_img, 50), None)
ref_norm = (ref_img - ref_img.min()) / (ref_img.max() - ref_img.min())

fig, axs = plt.subplots(ncols=N_COLS, nrows=nrows, figsize=(16, 4 * nrows), dpi=200)

for i, (ax, round_) in enumerate(zip(axs.flat, sorted(first.keys()))):
    round_img = img[i][sl]
    round_img = np.clip(round_img, np.percentile(round_img, 50), None)
    correction = optimize_shift(round_img, ref_img).x

    from scipy.ndimage import shift as shiftfunc

    round_img = shiftfunc(round_img, correction)
    # Read the image for this round

    # Normalize both images to 0-1 range

    round_norm = (round_img - round_img.min()) / (round_img.max() - round_img.min())
    # Create RGB image (R=round, G=reference, B=zeros)
    rgb = np.dstack([round_norm, ref_norm, np.zeros_like(ref_norm)])

    # Display
    ax.imshow(rgb[:100, :100])
    ax.set_title(f"Round {round_} vs Reference")
    ax.axis("off")

for ax in axs.flat:
    if not ax.has_data():
        fig.delaxes(ax)

plt.tight_layout()
# %%
# Create 2D Gaussian model
g2d = models.Gaussian2D(amplitude=0.5, x_mean=0, y_mean=0, x_stddev=1, y_stddev=1)
fitter = fitting.LevMarLSQFitter()

# Perform the fit
fitted = fitter(g2d, x, y, z)

# Access parameters
print(fitted.amplitude.value)  # amplitude
print(fitted.x_mean.value)  # x center
print(fitted.y_mean.value)  # y center
print(fitted.x_stddev.value)  # x width
print(fitted.y_stddev.value)  # y width
# %%

# %%
