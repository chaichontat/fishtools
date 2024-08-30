# %%
from pathlib import Path

import itk
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import SimpleITK as sitk
import tifffile
from scipy.ndimage import shift

sns.set_theme()


img = tifffile.imread("/mnt/archive/starmap/sagittal-calibration/3_3--cortexRegion/3_3-0000.tif")[
    :-1
].reshape(12, 2, 2048, 2048)

# %%
registration_method = sitk.ImageRegistrationMethod()

# Similarity metric settings.
registration_method.SetMetricAsMeanSquares()
# registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
registration_method.SetMetricSamplingPercentage(0.01)

registration_method.SetInterpolator(sitk.sitkLinear)

# Optimizer settings.
# registration_method.SetOptimizerAsGradientDescent(
#     learningRate=1.0, numberOfIterations=1000, convergenceMinimumValue=1e-6, convergenceWindowSize=10
# )
registration_method.SetOptimizerAsLBFGS2()
# registration_method.SetOptimizerAsConjugateGradientLineSearch(learningRate=1.0, numberOfIterations=1000)
registration_method.SetOptimizerScalesFromPhysicalShift()

# Setup for the multi-resolution framework.
registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

img1 = sitk.Cast(sitk.GetImageFromArray(img[0, 0]), sitk.sitkFloat32)
img2 = sitk.Cast(sitk.GetImageFromArray(img[0, 1]), sitk.sitkFloat32)

# Don't optimize in-place, we would possibly like to run this cell multiple times.
registration_method.SetInitialTransform(sitk.AffineTransform(img1.GetDimension()), inPlace=False)


fids_itk = []
ts = []
# for i in range(1, 5):
# img1 = sitk.Cast(sitk.GetImageFromArray(img[i, 1]), sitk.sitkFloat32)
# img2 = sitk.Cast(sitk.GetImageFromArray(img[i, 2]), sitk.sitkFloat32)
final_transform = registration_method.Execute(img1, img2)
ts.append(final_transform.GetParameters())
print(ts[-1])

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
    cells = tifffile.imread(path)[:-1].reshape(12, 2, 2048, 2048)
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
    for _, file in zip(
        range(10), Path("/mnt/archive/starmap/sagittal-calibration/3_3--cortexRegion").glob("*.tif")
    )
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
