# %%
import pickle
from pathlib import Path

import jax
import numpy as np
from basicpy import BaSiC
from loguru import logger
from tifffile import TiffFile

channels = [405, 560, 650, 750]
# %%
prefix = "wga_2_10_18"
path = Path(f"/fast2/synaptosome")
files = list((path / f"all--{prefix}").rglob(f"{prefix}*.tif"))
n = 500
z = 2
c = len(channels)
out = np.zeros((n, c, 2048, 2048), dtype=np.uint16)
for i, file in enumerate(files[:n]):
    if i % 100 == 0:
        logger.info("Loaded {}/{}", i, n)
    with TiffFile(file) as tif:
        for j in range(c):
            out[i, j] = tif.pages[z * c + j].asarray()

    # img = BaSiC(get_darkfield=True, smoothness_flatfield=1)(img)
    # imsave(file.with_name(file.stem + "_basic.tif"), img)

# %%
# BaSiC(get_darkfield=True, smoothness_flatfield=1)
basics = []
with jax.default_device(jax.devices("cpu")[0]):
    for c in range(out.shape[1]):
        logger.info("Fitting channel {}", c)
        basic = BaSiC()
        basic.fit(out[:, c])
        basics.append(basic)
# %%
for b, c in zip(basics, channels):
    with open(path / f"basic_{c}.pkl", "wb") as f:
        pickle.dump(b, f)
# %%
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()


def plot(basic: BaSiC):
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    im = axes[0].imshow(basic.flatfield)
    fig.colorbar(im, ax=axes[0])
    axes[0].set_title("Flatfield")
    im = axes[1].imshow(basic.darkfield)
    fig.colorbar(im, ax=axes[1])
    axes[1].set_title("Darkfield")
    axes[2].plot(basic.baseline)
    axes[2].set_xlabel("Frame")
    axes[2].set_ylabel("Baseline")
    axes[0].axis("off")
    axes[1].axis("off")
    fig.tight_layout()


plot(basics[2])
# # %%
# import pickle

# with open("basic_750.pkl", "wb") as f:
#     pickle.dump(basic, f)
# # %%
# plt.imshow(basic.fit_transform(out[[50], 2])[0], vmax=500)
# # %%

# %%
