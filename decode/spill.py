# %%
from tifffile import imread, imwrite
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.registration import phase_cross_correlation

sns.set_theme()


img = imread("/fast2/alinatake2/4_12_20/4_12_20-1233.tif")[:-1].reshape(-1, 3, 2048, 2048)
# %%
fig, axs = plt.subplots(ncols=3, nrows=1, figsize=(12, 4), dpi=200)
axs = axs.flatten()
for i, ax in enumerate(axs):
    ax.axis("off")
    if i == 2:
        ax.imshow(img[5, 0, 300:400, 1500:1600] - 0.22 * img[5, 1, 300:400, 1500:1600])
    else:
        ax.imshow(img[5, i, 300:400, 1500:1600])
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
