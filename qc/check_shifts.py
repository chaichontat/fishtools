# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from astropy.modeling import fitting, models
from loguru import logger
from pydantic import BaseModel, TypeAdapter
from scipy.stats import chi2
from sklearn.covariance import EllipticEnvelope

sns.set_theme()


def mahalanobis_outliers(points, threshold=0.9999):
    # Compute mean and covariance
    mean = np.mean(points, axis=0)
    covar = np.cov(points.T)

    # Compute Mahalanobis distances
    diff = points - mean
    distances = np.sqrt(np.sum((diff @ np.linalg.inv(covar)) * diff, axis=1))

    # Chi-square threshold for 2 degrees of freedom
    cutoff = np.sqrt(chi2.ppf(threshold, df=2))

    return distances > cutoff


class Shift(BaseModel):
    shifts: tuple[float, float]
    corr: float
    residual: float


Shifts = TypeAdapter(dict[str, Shift])
# %%
ref = "4_12_20"
roi = "full"
path = Path(f"/mnt/archive/starmap/e155/e155_zach/analysis/deconv/shifts--{roi}")
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

    ax.scatter(*c1.T, c=outliers, alpha=0.5, s=2, cmap="bwr")
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

for ax in axs.flat:
    if not ax.has_data():
        fig.delaxes(ax)

plt.tight_layout()
print(sorted(outliers_list))
# %% Residuals
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
