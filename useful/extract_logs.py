# %%
import re
from pathlib import Path

from expression.collections import Seq

path = Path("/mnt/archive/starmap/sagittal/analysis/deconv")

logs = Seq((path / "log.txt").read_text().splitlines())
bads = sorted(set(
    logs.filter(lambda x: "residual drift too large" in x or "not enough" in x)
    .map(lambda x: re.search(r"fishtools.analysis.fiducial:(\d+) (\d+)", x).group(2))
    .map(int)
    .to_list()
))


# %%
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tifffile import imread, imwrite

sns.set_theme()
fig, axs = plt.subplots(ncols=2, nrows=1, figsize=(8, 4), dpi=300)
img = imread(path / "fids_shifted-1170.tif")
img /= img.max(axis=(1,2), keepdims=True)
sl = np.s_[700:900, 150:450]
axs[0].imshow(np.swapaxes(img[:3], 0, 2)[sl], zorder=1, vmax=0.4)
axs[1].imshow(np.swapaxes(img[-3:], 0, 2)[sl], zorder=1, vmax=0.4)
# axs[1].imshow(np.swapaxes(img[:3], 0, 2)[ 1700:, :500][:,:,-1], zorder=1)

#%%

#%%

axs = axs.flatten()
for i, ax in enumerate(axs):
    ax.axis('off')
    ax.

# %%

# %%
