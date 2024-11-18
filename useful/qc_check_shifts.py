# %%

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pydantic import BaseModel, TypeAdapter
from sklearn.covariance import EllipticEnvelope

sns.set_theme()


class Shift(BaseModel):
    shifts: tuple[float, float]
    corr: float


Shifts = TypeAdapter(dict[str, Shift])
# %%
paths = sorted(Path("/mnt/archive/starmap/e155/e155_zach/analysis/deconv/shifts--full").glob("*.json"))


shifts = [Shifts.validate_json(path.read_text()) for path in paths]

N_COLS = 4
nrows = int(np.ceil(len(shifts[0]) / N_COLS))

# %% Shifts

fig, axs = plt.subplots(ncols=N_COLS, nrows=nrows, figsize=(12, 3 * nrows), dpi=200)

for ax, name in zip(axs.flat, sorted(shifts[0])):
    c1 = np.array([s[name].shifts for s in shifts])
    ax.set_title(name)
    lim = 1.25 * np.abs(c1).max()
    if np.sum(c1) != 0:
        estimator = EllipticEnvelope(random_state=0)
        outlier = estimator.fit_predict(c1)
    else:
        outlier = None

    ax.scatter(*c1.T, c=outlier, alpha=0.5, s=2, cmap="bwr_r")

    ax.set_aspect("equal")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)

for ax in axs.flat:
    if not ax.has_data():
        fig.delaxes(ax)

plt.tight_layout()

# %% Correlation
fig, axs = plt.subplots(ncols=N_COLS, nrows=nrows, figsize=(12, 3 * nrows), dpi=200)

for ax, name in zip(axs.flat, sorted(shifts[0])):
    c1 = np.array([s[name].corr for s in shifts])
    ax.hist(c1, linewidth=0)
    ax.set_title(name)
    ax.set_xlim(-1, 1)

for ax in axs.flat:
    if not ax.has_data():
        fig.delaxes(ax)

plt.tight_layout()

# %%
