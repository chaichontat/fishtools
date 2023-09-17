# %%
from pathlib import Path

import numpy as np
from scipy.ndimage import shift
from skimage.io import imread
from skimage.registration import phase_cross_correlation

# %%
pxs = 2048


def calculate_shift(ref: np.ndarray, img: np.ndarray):
    return phase_cross_correlation(ref, img, upsample_factor=100)[0]


# %%
from tifffile import imread, imwrite

files = sorted(Path("~/bg").expanduser().glob("*.tif"))
shifted = {}
for file in files:
    if "stack_0" in file.stem:
        ref = imread(file)
        break

for file in files:
    if "stack_0" in file.stem:
        continue

    shifted[file.name] = calculate_shift(ref[0], imread(file)[0])


# %%
shifted = {
    "stack_0_627": [0.0, 0.0],
    "stack_1_627": [2.36, 1.34],
    "stack_2_627": [3.03, 2.02],
    "stack_3_627": [3.29, 2.91],
    "stack_4_627": [3.87, 3.66],
    "stack_5_627": [3.98, 4.11],
    "stack_6_627": [4.08, 4.95],
    "stack_7_627": [4.41, 5.29],
    "stack_8_627": [4.77, 4.5],
    "stack_9_627": [2.03, -0.15],
    # "stack_prestain_627": [-7.07, -6.95],
}
# %%
z = 7
outs = []
for name, sh in sorted(shifted.items(), key=lambda x: x[0]):
    img = imread((Path("~/bg").expanduser() / name).with_suffix(".tif"))
    # rs = shift(img[1:], shift=[0, sh[0], sh[1]], order=1)
    rs = img
    # max project
    out = np.zeros((rs.shape[0] // z, rs.shape[1], rs.shape[2]))
    for i in range(rs.shape[0] // z):
        out[i] = np.max(rs[i * z : (i + 1) * z], axis=0)
    outs.append(out)
# %%
z = 7
outs = []
for name, sh in sorted(shifted.items(), key=lambda x: x[0]):
    img = imread((Path("~/bg").expanduser() / name).with_suffix(".tif"))
    rs = shift(img[0][np.newaxis, ...], shift=[0, sh[0], sh[1]], order=1)
    outs.append(rs)

imwrite(
    "/home/chaichontat/bg/fid.tif",
    np.concatenate(outs).astype(np.uint16),
    photometric="minisblack",
    compression=22610,
    imagej=True,
)

# %%
imwrite(
    "/home/chaichontat/bg/notshifted.tif",
    np.concatenate(outs).astype(np.uint16),
    photometric="minisblack",
    compression=22610,
    imagej=True,
)
# %%
