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

sns.set_theme()


# %%
As = []
ts = []
parameter_object = itk.ParameterObject.New()
default_rigid_parameter_map = parameter_object.GetDefaultParameterMap("affine")
parameter_object.AddParameterMap(default_rigid_parameter_map)
result_transform_parameters = parameter_object


# Actual calculation
# As is a 2x2 matrix
# ts is a 2x1 matrix (translation)
def calc_shift(path: Path | str):
    global result_transform_parameters
    cells = tifffile.imread(path)[:-2].reshape(-1, 2, 2048, 2048)
    print(cells.shape)
    result_image, result_transform_parameters = itk.elastix_registration_method(
        cells[5, 0], cells[5, 1], parameter_object=parameter_object, log_to_console=True
    )

    params = np.array(list(map(float, result_transform_parameters.GetParameter(0, "TransformParameters"))))
    As.append(params[:4].reshape(2, 2))
    ts.append(params[-2:])
    print(As[-1])


# %%
# Input files here.
[
    calc_shift(file)
    for _, file in zip(range(10), Path("/working/20250322_cal/cal560_cal650--cere/").glob("*.tif"))
]
# %%
affine = np.array([*np.median(np.stack(As), axis=0).flatten(), *np.median(np.stack(ts), axis=0)])

Path("data/560to650.txt").write_text("\n".join(map(str, affine.flatten())))

# result_transform_parameters.WriteParameterFile(
#     result_transform_parameters.GetParameterMap(0), "Parameters2D.txt"
# )

# %%
# Below is the validation.
# %%


# %%
result_transform_parameters.SetParameter(0, "TransformParameters", list(map(str, affine)))

# %%
result_image_transformix = itk.transformix_filter(img[5:15, 1], result_transform_parameters)

tifffile.imwrite("comp.tif", np.stack([img[10, 0], result_image_transformix[5], img[10, 1]]))

# %%
compare(img[5, 2, 200:400, 200:400], result_image_transformix[0, 200:400, 200:400], link_cmap=True)
# %%
imgs = list(Path("/disk/chaichontat/2024/sv101_ACS/2_10_18--noRNAse_big").glob("*.tif"))

img = tifffile.imread(imgs[71])[:-1]
img = img.reshape(img.shape[0] // 3, 3, 2048, 2048)

imgmax = img.max(0)

fig, axs = plt.subplots(ncols=2, figsize=(8, 4), dpi=200)

one = imgmax / np.max(imgmax, axis=(1, 2), keepdims=True)
one[1] = 0
axs[0].imshow(one.swapaxes(0, 2)[1250:1350, 1600:1700], zorder=1, vmax=0.1)


shifted = one.copy()

shifted[2] = shift(one[2], [1.8, 0.3], order=1)
axs[1].imshow(shifted.swapaxes(0, 2)[1250:1350, 1600:1700], zorder=1, vmax=0.1)
plt.show()

# %%
plt.plot(one.swapaxes(0, 2)[1338, 1600:1700, 0])
plt.plot(shifted.swapaxes(0, 2)[1338, 1600:1700, 2])

# one[2] = 0
# plt.imshow(one.swapaxes(0, 2)[1200:1400, 1600:1800], zorder=1, vmax=0.2)
# %%
plt.imshow(
    (img[0, :] / np.max(img[0, :], axis=(1, 2), keepdims=True)).swapaxes(0, 2)[1500:1700, 1500:1700], zorder=1
)
# %%
