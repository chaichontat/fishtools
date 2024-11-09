# %%
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
import tifffile


fig, ax = plt.subplots(figsize=(8, 6), dpi=200)


img = tifffile.imread("/mnt/working/e155_trc/analysis/deconv/stitch--right/fused.tif")
ax.imshow(img[54, 0])

# %%
