# # %%
# from pathlib import Path

# import numpy as np
# from basicpy import BaSiC
# from tifffile import imread, imsave, TiffFile

# # %%
# files = list(Path("/fast2/alinatake2").rglob("6_14_22*.tif"))
# out = np.zeros((500, 3, 2048, 2048), dtype=np.uint16)
# z = 3 * 10
# for i, file in enumerate(files[:500]):
#     if i % 100 == 0:
#         print(i)
#     with TiffFile(file) as tif:
#         out[i, 0] = tif.pages[z + 0].asarray()
#         out[i, 1] = tif.pages[z + 1].asarray()
#         out[i, 2] = tif.pages[z + 2].asarray()

#     # img = BaSiC(get_darkfield=True, smoothness_flatfield=1)(img)
#     # imsave(file.with_name(file.stem + "_basic.tif"), img)


# # BaSiC(get_darkfield=True, smoothness_flatfield=1)

# # %%


# # %%
# import jax

# with jax.default_device(jax.devices("cpu")[0]):
#     basic = BaSiC()
#     basic.fit(out[:, 2])

# # %%
# import matplotlib.pyplot as plt
# import seaborn as sns

# sns.set_theme()

# fig, axes = plt.subplots(1, 3, figsize=(9, 3))
# im = axes[0].imshow(basic.flatfield)
# fig.colorbar(im, ax=axes[0])
# axes[0].set_title("Flatfield")
# im = axes[1].imshow(basic.darkfield)
# fig.colorbar(im, ax=axes[1])
# axes[1].set_title("Darkfield")
# axes[2].plot(basic.baseline)
# axes[2].set_xlabel("Frame")
# axes[2].set_ylabel("Baseline")
# fig.tight_layout()
# # %%
# import pickle

# with open("basic_750.pkl", "wb") as f:
#     pickle.dump(basic, f)
# # %%
# plt.imshow(basic.fit_transform(out[[50], 2])[0], vmax=500)
# # %%
