# %%
from pathlib import Path

import numpy as np
import tifffile
from skimage.filters import gaussian

path = Path("/mnt/working/20241113-ZNE172-Zach/analysis/deconv/registered--right/reg-0078.hp.tif")
rand = np.random.default_rng(0)

percs = np.loadtxt(path.parent / "perc.txt").astype(int).mean(axis=0).reshape(1, -1, 1, 1)
# %%
norms = []
percs = []
# np.percentile(np.linalg.norm(img, axis=1), 25)
# %%
for path in sorted(path.parent.glob("*.hp.tif")):
    print(path)
    img = tifffile.imread(path).squeeze().swapaxes(0, 1)
    percs.append(np.percentile(img, 25, axis=(0, 2, 3)))
    # norms.append(np.percentile(np.linalg.norm(img, axis=1), 25))
    print(percs[-1])

# %%
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()

# %%
plt.imshow(img[30, 3], zorder=1, vmax=np.percentile(img[30, 3], 99.9))

# %%
