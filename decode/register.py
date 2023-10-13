# %%
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import itk
import numpy as np
import SimpleITK as sitk
import tifffile
from astropy.stats import sigma_clipped_stats
from astropy.table.table import QTable
from photutils.detection import DAOStarFinder, IRAFStarFinder, find_peaks
from scipy.ndimage import shift
from scipy.spatial import cKDTree
from tifffile import imread

# %%
idx = 150
imgs = {
    file.name: tifffile.imread(file)
    for file in sorted(Path("/raid/data/raw/tricycle/").glob(f"*-{idx:03d}.tif"))
}
img = imgs[f"A_1_2-{idx:03d}.tif"]
no_a = img[:-1].reshape(12, 3, 2048, 2048)[:, [1, 2], ...].reshape(-1, 2048, 2048)
imgs[f"1_2-{idx:03d}.tif"] = np.concatenate([no_a, imgs[f"A_1_2-{idx:03d}.tif"][None, -1]], axis=0)
del imgs[f"A_1_2-{idx:03d}.tif"]


dark = imread("/raid/data/analysis/dark.tif")
flat = (imread("/raid/data/analysis/flat_647.tif") - dark).astype(np.float32)
flat /= np.min(flat)  # Prevents overflow
for name in imgs:
    img = imgs[name]
    imgs[name] = ((img - dark).astype(np.float32) / flat).astype(np.uint16)

fids = {k: v[-1] for k, v in imgs.items()}
imgs = {k: v[:-1] for k, v in imgs.items()}
keys = list(imgs.keys())
# %%
import pandas as pd


def find_spots(data: np.ndarray[np.uint16, Any]) -> QTable:
    mean, median, std = sigma_clipped_stats(data, sigma=3.0)
    # return find_peaks(data, threshold=median + 10.0 * std, box_size=10)
    daofind = IRAFStarFinder(threshold=5.0 * std, fwhm=3, exclude_border=True)
    return daofind(data - median)  # [['xcentroid', 'ycentroid']].to_pandas()


def calc_shift(ref: np.ndarray, img: np.ndarray):
    return phase_cross_correlation(ref, img, upsample_factor=100)[0]


# def export_elastix_csv(filename: str, img: np.ndarray[np.uint16, Any]) -> pd.DataFrame:
#     y = find_spots(img)
#     with open(filename, 'w') as f:
#         f.write(f"point\n{len(y)}\n")
#         y.to_csv(f, index=False, sep=" ", header=False)
#     return y


# %%

registration_method = sitk.ImageRegistrationMethod()

# Similarity metric settings.
registration_method.SetMetricAsMeanSquares()
# registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
registration_method.SetMetricSamplingStrategy(registration_method.REGULAR)
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

img1 = sitk.Cast(sitk.GetImageFromArray(fids[keys[0]]), sitk.sitkFloat32)
img2 = sitk.Cast(sitk.GetImageFromArray(fids[keys[8]]), sitk.sitkFloat32)

# Don't optimize in-place, we would possibly like to run this cell multiple times.
registration_method.SetInitialTransform(sitk.TranslationTransform(img1.GetDimension()), inPlace=False)


fids_itk = []
ts = []
for i in range(1, len(keys)):
    img2 = sitk.Cast(sitk.GetImageFromArray(fids[keys[i]]), sitk.sitkFloat32)
    final_transform = registration_method.Execute(img1, img2)
    ts.append(final_transform.GetParameters())
    print(i, ts[-1])

    # resampler = sitk.ResampleImageFilter()
    # resampler.SetReferenceImage(img1)
    # resampler.SetInterpolator(sitk.sitkLinear)
    # resampler.SetDefaultPixelValue(100)
    # resampler.SetTransform(final_transform)
    # print(final_transform)

    # out = resampler.Execute(img2)
    # registered_array = sitk.GetArrayFromImage(out).astype(np.uint16)
    # fids_itk.append(registered_array)
    # assert registered_array.size

# %%

As = {}
ats = {}
for λ in [560, 750]:
    a_ = np.loadtxt(f"650to{λ}.txt")
    A, t = a_[:9].reshape(3, 3), a_[-3:]
    A[2] = [0, 0, 1]
    A[:, 2] = [0, 0, 1]
    t[2] = 0
    As[λ] = A
    ats[λ] = t


ref = sitk.Cast(sitk.GetImageFromArray(imgs[keys[0]].reshape(12, 2, 2048, 2048)[:, 0]), sitk.sitkFloat32)


def st(img: np.ndarray[np.uint16, Any], transform: sitk.Transform):
    image = sitk.Cast(sitk.GetImageFromArray(img), sitk.sitkFloat32)
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ref)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(100)
    resampler.SetTransform(transform)
    return sitk.GetArrayFromImage(resampler.Execute(image)).astype(np.uint16)


def run_image(key: str, img: np.ndarray[np.uint16, Any]):
    translate = sitk.TranslationTransform(3)
    tparam = ts[keys.index(key) - 1]
    translate.SetParameters([*tparam, 0])

    affine = sitk.AffineTransform(3)
    affine.SetMatrix(As[750].flatten())
    affine.SetTranslation(ats[750])
    affine.SetCenter([1023.5 + tparam[0], 1023.5 + tparam[1], 0])

    composite = sitk.CompositeTransform(3)
    composite.AddTransform(translate)
    # composite.AddTransform(affine)

    img = img.reshape(12, 2, 2048, 2048)
    return {
        int(key.split("-")[0].split("_")[0]): st(img[:, 0], translate),
        int(key.split("-")[0].split("_")[1]): st(img[:, 1], composite),
    }


affine = sitk.AffineTransform(3)
affine.SetMatrix(As[750].flatten())
affine.SetTranslation(ats[750])

transformed = {
    int(keys[0].split("-")[0].split("_")[0]): imgs[keys[0]][:12],
    int(keys[0].split("-")[0].split("_")[1]): st(imgs[keys[0]][12:], affine),
}

for i, (name, img) in enumerate(imgs.items()):
    if i == 0:
        continue
    transformed |= (res := run_image(name, img))
    print({k: np.mean(v) for k, v in res.items()})

# %%
tifffile.imwrite(
    "pls.tif",
    np.stack([transformed[i][5:9].max(axis=0) for i in range(1, 25)]),
    compression=22610,
    imagej=True,
)

# %%


# def show_registration(img1, img2):


fids_itk = [fids[keys[0]]]
for i in range(1, len(keys)):
    img2 = sitk.Cast(sitk.GetImageFromArray(fids[keys[i]]), sitk.sitkFloat32)
    final_transform = registration_method.Execute(img1, img2)
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ref)

    resampler.SetTransform(final_transform)
    print(final_transform)

    out = resampler.Execute(img2)
    registered_array = sitk.GetArrayFromImage(out).astype(np.uint16)
    fids_itk.append(registered_array)
    print(i)


tifffile.imwrite("fids.tif", np.stack(fids_itk), compression=22610, imagej=True)

# %%


# Convert the result back to a numpy array


# %%
tifffile.imwrite("fids.tif", np.stack([fids[keys[0]], registered_array]), compression=22610, imagej=True)

# %%
shifts = {k: calc_shift(fids[keys[0]], v) for k, v in fids.items()}
fids_shifted = {k: shift(fids[k], s, order=1) for k, s in shifts.items()}
# %%
pointss = {k: find_spots(img) for k, img in fids_shifted.items()}

# %%
tree = cKDTree(pointss[keys[0]][["xcentroid", "ycentroid"]].to_pandas())
base = set(range(len(pointss[keys[0]][["xcentroid", "ycentroid"]].to_pandas())))


def z_noinf(data: np.ndarray):
    # Calculate the mean and standard deviation of the data, ignoring infinite values
    zscore = np.empty_like(data)
    zscore[:] = np.nan
    finite_mask = np.isfinite(data)
    finite = data[finite_mask]
    mean, std = np.mean(finite), np.std(finite)
    # Calculate the z-score of the data, ignoring infinite values
    zscore[finite_mask] = (finite - mean) / std
    return zscore


# Find all shared points
for name, points in pointss.items():
    points = points.to_pandas()
    points = points[points["sharpness"] > 0.62][["xcentroid", "ycentroid"]]
    if name == keys[0]:
        continue
    distances, indices = tree.query(points, distance_upper_bound=10, workers=2)
    indices[np.abs(z_noinf(distances)) > 3] = -1
    base &= set(indices)

# %%
present_in_all = pointss[keys[0]][["xcentroid", "ycentroid"]].to_pandas().iloc[list(base)]
with open(f"working/{keys[0]}.txt", "w") as f:
    f.write(f"point\n{len(base)}\n")
    present_in_all.to_csv(f, index=False, sep=" ", header=False)

tree = cKDTree(present_in_all)
for name, points in pointss.items():
    points = points[["xcentroid", "ycentroid"]].to_pandas()
    if name == keys[0]:
        continue
    distances, indices = tree.query(points, distance_upper_bound=10, workers=2)
    indices[np.abs(z_noinf(distances)) > 3] = -1
    indices = indices.tolist()
    reordered = points.iloc[[indices.index(i) for i in range(len(base))]]
    print((reordered.to_numpy() - present_in_all.to_numpy()).mean(axis=0))
    # assert len(reordered) == len(base)
    # with open(f"working/{name}.txt", 'w') as f:
    #     f.write(f"point\n{len(base)}\n")
    #     reordered.to_csv(f, index=False, sep=" ", header=False)


# mapping = {i: idx for i, (dist, idx) in enumerate(pairing) if idx != len(points[0])}

# reordered.to_numpy()

# %%
parameter_object = itk.ParameterObject.New()
parameter_map_rigid = parameter_object.GetDefaultParameterMap("translation")
parameter_map_rigid["Registration"] = ["MultiMetricMultiResolutionRegistration"]
original_metric = parameter_map_rigid["Metric"]
parameter_map_rigid["Metric"] = [original_metric[0], "CorrespondingPointsEuclideanDistanceMetric"]
parameter_object.AddParameterMap(parameter_map_rigid)

# %%

result_image, result_transform_parameters = itk.elastix_registration_method(
    fids[keys[0]],
    fids[keys[2]],
    fixed_point_set_file_name=f"working/{keys[0]}.txt",
    moving_point_set_file_name=f"working/{keys[2]}.txt",
    log_to_console=True,
    parameter_object=parameter_object,
)

# %%

# %%
ref = list(imgs.keys())[0]

# %%imgs, fids = imgs[:, :-1].reshape(imgs.shape[0], 12,2,2048,2048), imgs[:, -1] #.reshape(20, 3, 2048, 2048)
# %%
# %%

# %%


# %%
from skimage.registration import phase_cross_correlation

param2d = itk.ParameterObject.New()
param2d.ReadParameterFile(["Parameters.txt"])
param2d.SetParameter(0, "Transform", "TranslationTransform")
param2d.SetParameter(0, "NumberOfParameters", str(2))


with ThreadPoolExecutor() as pool:
    shifts = {k: pool.submit(calc_shift, fids[ref], v) for k, v in fids.items()}

shifts = {k: v.result() for k, v in shifts.items()}
# %%


# %%

param750 = itk.ParameterObject.New()
param750.ReadParameterFile(["shift.txt", "Parameters3D.txt"])

param650 = itk.ParameterObject.New()
param650.ReadParameterFile(["shift.txt"])


# parameter_object
# %%

# default_rigid_parameter_map = parameter_object.GetDefaultParameterMap("translation")
# parameter_object.AddParameterMap(default_rigid_parameter_map)

# %%
maxed = []
fidaln = []
for name, img in imgs.items():
    for p in [param650, param750]:
        p.SetParameter(0, "TransformParameters", list(map(str, [*shifts[name], 0])))
    param2d.SetParameter(0, "TransformParameters", list(map(str, shifts[name])))

    img = img.reshape(12, 2, 2048, 2048)
    res650 = itk.transformix_filter(img[:, 0], param650)
    res750 = itk.transformix_filter(img[:, 1], param750)
    print(name)
    tifffile.imwrite(
        f"/raid/data/raw/tricycle/aligned/{name}",
        stacked := np.stack([res650, res750]),
        compression=22610,
        metadata={"axes": "CZYX"},
    )
    maxed.append(stacked.max(axis=1))
    fidaln.append(itk.transformix_filter(fids[name], param2d))
# %%
tifffile.imwrite("comp.tif", np.concatenate(maxed))
tifffile.imwrite("fids.tif", np.stack(fidaln), compression=22610, imagej=True)

# %%
compare(img[5, 2, 200:400, 200:400], res750[0, 200:400, 200:400], link_cmap=True)
# %%
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()


# %%
A
# %%
t
# %%
