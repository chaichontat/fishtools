# %%
import re
from pathlib import Path

from expression.collections import Seq

from fishtools.preprocess.fiducial import find_spots

path = Path("/mnt/archive/starmap/sagittal/analysis/deconv")

logs = Seq((path / "log.txt").read_text().splitlines())
bads = sorted(set(
    logs.filter(lambda x: "residual drift too large" in x or "Could not find spots" in x)
    .map(lambda x: re.search(r"fishtools.preprocess.fiducial:(\d+) (\d+)", x).group(2))
    .map(int)
    .to_list()
))

bads
# %%
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tifffile import imread, imwrite

sns.set_theme()
idx = 383
fig, axs = plt.subplots(ncols=2, nrows=1, figsize=(8, 4), dpi=300)
img = imread(path / f"fids_shifted-{idx:04d}.tif")[:, 100:-100, 100:-100]
img /= np.percentile(img, 99.9999, axis=(1,2),keepdims=True)
# sl = np.s_[250:600, 900:1100]

peak = find_spots(img[0], threshold_sigma=2.5, fwhm=5)[0, ['xcentroid', 'ycentroid']].to_numpy().squeeze()
# x and y are flipped from np.swapaxes below.
window = 200
sl = np.s_[max(0, int(peak[0]) - window):min(img.shape[1], int(peak[0]) + window), max(0, int(peak[1]) - window):min(img.shape[2], int(peak[1]) + window)]

# Compute correlation coefficients in one go
correlations = np.corrcoef(img[0, ::4, ::4].flatten(), img[1:, ::4, ::4].reshape(img.shape[0] - 1, -1))[0, 1:]
print(correlations)

# window of 200 pixels around the max
# sl = np.s_[:, :]
axs[0].imshow(np.swapaxes(img[:3], 0, 2)[sl], zorder=1)
axs[1].imshow(np.swapaxes(img[-3:], 0, 2)[sl], zorder=1)
# axs[1].imshow(np.swapaxes(img[:3], 0, 2)[ 1700:, :500][:,:,-1], zorder=1)

#%%
fig, axs = plt.subplots(ncols=3, nrows=2, figsize=(12, 8), dpi=200)

# img = imread(path / "fids_shifted-0105.tif")
# img /= img.max(axis=(1,2), keepdims=True)
# sl = np.s_[250:600, 900:1100]
# sl = np.s_[:, :]
for ax,slic in zip(axs.flatten(),img):
    ax.imshow(slic.T[sl], zorder=1)
    ax.axis('off')
# axs[0].imshow(np.swapaxes(img[:3], 0, 2)[sl], zorder=1, vmax=0.4)
# axs[1].imshow(np.swapaxes(img[-3:], 0, 2)[sl], zorder=1, vmax=0.4)
# axs[1].imshow(np.swapaxes(img[:3], 0, 2)[ 1700:, :500][:,:,-1], zorder=1)
#%%


#%%

axs = axs.flatten()
for i, ax in enumerate(axs):
    ax.axis('off')
    ax.

# %%

# %%
