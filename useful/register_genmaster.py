# %%
# You need itk-elastix installed to run this.
from pathlib import Path

import itk
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import SimpleITK as sitk
import tifffile
from scipy.ndimage import shift

from fishtools.preprocess.chromatic import Affine, FitAffine
from fishtools.preprocess.config import NumpyEncoder

sns.set_theme()


# %%


# Actual calculation

# def calc_shift(path: Path | str):
#     global result_transform_parameters
#     cells = tifffile.imread(path)[:-2].reshape(-1, 2, 2048, 2048).astype(np.float32)
#     print(cells.shape)
#     result_image, result_transform_parameters = itk.elastix_registration_method(
#         cells[10, 0], cells[10, 1], parameter_object=parameter_object, log_to_console=True
#     )

#     params = np.array(list(map(float, result_transform_parameters.GetParameter(0, "TransformParameters"))))
#     As.append(params[:4].reshape(2, 2))
#     ts.append(params[-2:])
#     print(result_transform_parameters.GetParameter(0, "CenterOfRotationPoints"))
#     print(As[-1])


# %%
# Somehow you need to run this, stop this cell and run it again.
# The first run will stall.
target = "650"
fit = FitAffine()

As, ts = [], []

for _, file in zip(range(10), Path(f"/working/20250322_cal/cal560_cal{target}--cere/").glob("*.tif")):
    img = tifffile.imread(file)[:-2].reshape(-1, 2, 2048, 2048)
    A, t = fit.fit(img[10, 0], img[10, 1])
    As.append(A), ts.append(t)


# %%
# %%
affined = np.array([*np.median(np.stack(As), axis=0).flatten(), *np.median(np.stack(ts), axis=0)])

write = 1
if write:
    # Path(f"data/560to{target}.txt").write_text("\n".join(map(str, affined.flatten())))
    import json

    Path(f"data/560to{target}.json").write_text(
        json.dumps(
            dict(
                As=np.median(np.stack(As), axis=0),
                At=np.median(np.stack(ts), axis=0),
            ),
            indent=2,
            cls=NumpyEncoder,
        )
    )

# result_transform_parameters.WriteParameterFile(
#     result_transform_parameters.GetParameterMap(0), "Parameters2D.txt"
# )


# %%
# Below is the validation.
# %%


# %%
# result_transform_parameters.SetParameter(0, "TransformParameters", list(map(str, affine)))

# %%
img = tifffile.imread(Path(f"/working/20250322_cal/cal560_cal{target}--cere/cal560_cal{target}-0005.tif"))[
    :-2
].reshape(-1, 2, 2048, 2048)


As = {}
ats = {}
for λ in ["650", "750"]:
    a_ = np.loadtxt(f"/home/chaichontat/fishtools/data/560to{λ}.txt")
    A = np.zeros((3, 3), dtype=np.float64)
    A[:2, :2] = a_[:4].reshape(2, 2)
    t = np.zeros(3, dtype=np.float64)
    t[:2] = a_[-2:]

    A[2] = [0, 0, 1]
    A[:, 2] = [0, 0, 1]
    t[2] = 0
    As[λ] = A
    ats[λ] = t

affine = Affine(As=As, ats=ats, cg=None)

affine.ref_image = img[:, 0]

# result_image = itk.transformix_filter(
#     img[:, 1].astype(np.float32),
#     transform_parameter_object=result_transform_parameters,
# )
# imged = affine(img[:, 1], channel=target, shiftpx=[0, 0], debug=True)

# %%


# %%
fig, axs = plt.subplots(ncols=3, nrows=1, figsize=(12, 4), dpi=200)
axs = axs.flatten()
for i, ax in enumerate(axs):
    ax.axis("off")


# %%
sl = np.s_[100:400, 100:400]
fig, axs = plt.subplots(ncols=2, figsize=(8, 4), dpi=200)
for ax in axs.flat:
    ax.axis("off")

rgb_img = np.zeros((300, 300, 3))

# Set each channel and normalize independently
for i in range(2):
    channel = img[5, i, *sl]
    rgb_img[:, :, i] = (channel - channel.min()) / (channel.max() - channel.min())
axs[0].imshow(rgb_img)
imged = affine(img[:, 1], channel=target, shiftpx=[0, 0], debug=True)

# Normalize the third channel (imged) separately
channel = imged[5, *sl]
rgb_img[:, :, 1] = (channel - channel.min()) / (channel.max() - channel.min())
axs[1].imshow(rgb_img)

# %%
