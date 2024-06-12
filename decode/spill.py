# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from skimage.registration import phase_cross_correlation
from tifffile import imread, imwrite

sns.set_theme()

name = "6_14_22"
i = 49
img = (
    imread(f"/fast2/3t3clean/analysis/deconv/{name}/{name}-{i:04d}.tif")[:-1]
    .reshape(-1, len(name.split("_")), 2048, 2048)[5:10]
    .max(axis=0)
)
# %%
fig, axs = plt.subplots(ncols=3, nrows=1, figsize=(12, 4), dpi=200)
axs = axs.flatten()
sl = np.s_[1500:1600, 400:500]
# sl = np.s_[:]
for i, ax in enumerate(axs):
    ax.axis("off")
    if i == 2:
        ax.imshow(img[0][sl] - 0.01 * img[1][sl], vmax=10000)
    else:
        ax.imshow(img[0][sl] * (img[1][sl] > 5000), vmax=10000)
# %%

# %%

plt.scatter(img[0][sl].flatten(), img[1][sl].flatten(), alpha=0.5, s=1.5)
plt.axvline(x=1000)
# %%
from sklearn.decomposition import PCA


# %%

plt.plot(img[5, 0, 300:400, 1500:1600].max(axis=0))
# %%
plt.plot(img[5, 1, 300:400, 1500:1600].max(axis=0))

# %%

plt.plot((img[5, 0, 380, 1500:1600] - 0.22 * img[5, 1, 380, 1500:1600]))

# %%
fig, axs = plt.subplots(ncols=3, nrows=2, figsize=(12, 4), dpi=200)
axs = axs.flatten()
for i, ax in enumerate(axs):
    x = 0.15 + 0.01 * i
    ax.axis("off")
    ax.set_title(f"{x=}")
    ax.imshow(np.clip(0, img[5, 0, 300:400, 1500:1600] - x * img[5, 1, 300:400, 1500:1600], None), vmax=2000)
# %%
fid1 = imread("/fast2/alinatake2/dapi_polyA_28-1233.tif")[-1, 300:500, 300:500].astype(float)
fid2 = imread("/fast2/alinatake2/5_13_21/5_13_21-1233.tif")[-1, 300:500, 300:500].astype(float)

fig, axs = plt.subplots(ncols=2, nrows=1, figsize=(8, 4), dpi=200)
axs = axs.flatten()
for i, ax in enumerate(axs):
    ax.axis("off")
    ax.imshow(fid1 if i == 0 else fid2)
# %%
# %%
phase_cross_correlation(fid1, fid2, upsample_factor=100)

# %%
