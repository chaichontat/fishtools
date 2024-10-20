# %%
import pickle
from pathlib import Path

import jax
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from basicpy import BaSiC
from loguru import logger
from tifffile import TiffFile

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


channels = [560, 650, 750]
# %%
path = Path("/mnt/archive/starmap/e155/e155_trc")
files = list((path).rglob("3_11_19*/*.tif"))
files += list((path).rglob("4_12_20*/*.tif"))

if len(files) < 500:
    logger.warning(f"Not enough files ({len(files)}). Result may be unreliable.")

n = 500
z = 8
c = len(channels)

# This is the extracted z slices from all files.
out = np.zeros((n, c, 2048, 2048), dtype=np.uint16)
for i, file in enumerate(files[:n]):
    if i % 100 == 0:
        logger.info("Loaded {}/{}", i, n)
    with TiffFile(file) as tif:
        for j in range(c):
            out[i, j] = tif.pages[z * c + j].asarray()

logger.info(f"Loaded {len(files)} files. {out.shape}")
# img = BaSiC(get_darkfield=True, smoothness_flatfield=1)(img)
# imsave(file.with_name(file.stem + "_basic.tif"), img)


# %%
basics = []
with jax.default_device(jax.devices("cpu")[0]):
    for c in range(out.shape[1]):
        logger.info("Fitting channel {}", c)
        basic = BaSiC()
        basic.fit(out[: len(files), c])
        basics.append(basic)
        plot(basics[c])
        plt.show()
# %%
for b, c in zip(basics, channels):
    with open(path / f"basic_{c}.pkl", "wb") as f:
        pickle.dump(b, f)
# %%
with open(path / f"basic_650.pkl", "rb") as f:
    x = pickle.load(f)
# %%
plot(x)
# %%
