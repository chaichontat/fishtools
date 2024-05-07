# %%
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import itk

# %%
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
import SimpleITK as sitk
import tifffile
from astropy.stats import sigma_clipped_stats
from astropy.table.table import QTable
from photutils.detection import IRAFStarFinder
from scipy.ndimage import shift
from scipy.spatial import cKDTree
from tifffile import imread

from fishtools.analysis.fiducial import align_fiducials, background, find_spots
from fishtools.compression.compression import dax_reader

# sns.set()

# img = dax_reader("/raid/merfish_raw_data/202310271505_BG-Org-d70-102723_VMSC06101/data/stack_9_300.dax")

# plt.imshow(img[1:].max(axis=0))

# %%

idx = 340
n_z = 16

imgs = {
    file.name: tifffile.imread(file)
    for file in sorted(Path("/fast2/braintrctake2").glob(f"*-{idx:03d}.tif"))
    if not file.name.startswith("dapi") and not "mark" in file.name
}
keys = list(imgs.keys())

# %%


nofids = {name: img[:-1].reshape(n_z, -1, 2048, 2048) for name, img in imgs.items()}
fids = {name: img[-1] for name, img in imgs.items()}
shifts = align_fiducials(fids, reference=r"5_13_21", fiducial_corr=True)
# del imgs


# %%
# tifffile.imwrite("test.tif", nofids[keys[0]], compression=22610, compressionargs={"level": 0.75})
# Path("test.tif").lstat().st_size / 1024**2

# %%
# fig, axs = plt.subplots(ncols=2)

# axs[0].imshow(nofids[keys[0]][6, 2, 1450:1500, 800:900])
# axs[1].imshow(tifffile.imread("test.tif")[6, 2, 1450:1500, 800:900])

# %%
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()


# %%
ref = fids[keys[0]]
ref = ref - background(ref)

fixed = pl.DataFrame(find_spots(ref).to_pandas()).with_row_count("idx")
kd = cKDTree(fixed[["xcentroid", "ycentroid"]])
# %%

img = fids[keys[2]]
img = img - background(img)
fixed2 = pl.DataFrame(find_spots(img).to_pandas()).with_row_count("idx")
kd.query(fixed2[["xcentroid", "ycentroid"]])

# %%
# try:
#     nofids[f"1_2-{idx:03d}.tif"] = nofids[f"3_1_2-{idx:03d}.tif"][:, [1, 2], ...]
#     del nofids[f"A_1_2-{idx:03d}.tif"]
# except KeyError:
#     ...
# shifts[f"1_2-{idx:03d}.tif"] = shifts[f"A_1_2-{idx:03d}.tif"]

try:
    nofids[f"3_1_2-{idx:03d}.tif"] = nofids[f"3_1_2-{idx:03d}.tif"][:, [1, 2, 3], ...]
    nofids[f"4_5_6-{idx:03d}.tif"] = nofids[f"4_5_6-{idx:03d}.tif"][:, [1, 2, 3], ...]
except KeyError:
    ...
# shifts[f"3_1_2-{idx:03d}.tif"] = shifts[f"3_1_2-{idx:03d}.tif"]

# %%
img = nofids[f"1_9_17-{idx:03d}.tif"].max(axis=0)
plt.scatter(img[1, ::4, ::4], img[0, ::4, ::4], alpha=0.1, s=2)


# %%
plt.scatter(img[1, ::4, ::4], img[0, ::4, ::4] - 0.073 * img[1, ::4, ::4], alpha=0.1, s=2)
# %%


# %%
fids_shifted = {k: shift(v, -shifts[k], order=1) for k, v in fids.items()}
tifffile.imwrite(
    "comp.tif",
    np.stack(list(fids_shifted.values())),
    compression=22610,
    compressionargs={"level": 98},
    imagej=True,
)
# %%
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

ns = (len(fids_shifted) // 3) + 1
fig, axs = plt.subplots(ncols=ns, nrows=1)
axs = axs.flatten()

combi = np.stack([fids_shifted[k] for k in keys]).astype(np.float32)
combi /= np.percentile(combi, 99, axis=(1, 2))[:, None, None]

for ax, i in zip(axs, range(0, len(fids_shifted), 3)):
    if len(fids_shifted) - i < 3:
        i = len(fids_shifted) - 3
    ax.imshow(np.moveaxis(combi[i : i + 3][:, 512:1536, 512:1536], 0, 2))
    ax.axis("off")
# %%


# dark = imread("/raid/data/analysis/dark.tif")
# flat = (imread("/raid/data/analysis/flat_647.tif") - dark).astype(np.float32)
# flat /= np.min(flat)  # Prevents overflow
# for name in imgs:
#     img = imgs[name]
#     imgs[name] = ((img - dark).astype(np.float32) / flat).astype(np.uint16)

# fids = {k: v[-1] for k, v in imgs.items()}
# imgs = {k: v[:-1] for k, v in imgs.items()}
# keys = list(imgs.keys())

# %%

# %%

# %%
out = [fids[keys[0]]]
for i, px in maps.items():
    out.append(shift(fids[keys[i]], px, order=1))

tifffile.imwrite("fids.tif", np.stack(out))
# find_spots(fids[keys[0]])


import matplotlib.pyplot as plt

plt.imshow(np.stack(fids_itk[:3], axis=-1)[500:1000, 500:1000] / fids_itk[0].max())

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


ref = sitk.Cast(sitk.GetImageFromArray(nofids[keys[0]][:, 0]), sitk.sitkFloat32)


def resample(image, transform):
    # Output image Origin, Spacing, Size, Direction are taken from the reference
    # image in this call to Resample
    reference_image = image
    interpolator = sitk.sitkCosineWindowedSinc
    default_value = 100.0
    return sitk.Resample(image, reference_image, transform, interpolator, default_value)


def st(img: np.ndarray[np.uint16, Any], transform: sitk.Transform):
    image = sitk.Cast(sitk.GetImageFromArray(img), sitk.sitkFloat32)
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ref)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(100)
    resampler.SetTransform(transform)

    return sitk.GetArrayFromImage(resampler.Execute(image)).astype(np.uint16)


def run_image(key: str, img: np.ndarray[np.uint16, Any], shiftpx: np.ndarray):
    if len(shiftpx) != 2:
        raise ValueError

    translate = sitk.TranslationTransform(3)
    translate.SetParameters((shiftpx[1], shiftpx[0], 0.0))

    affine = sitk.AffineTransform(3)
    affine.SetMatrix(As[750].flatten())
    affine.SetTranslation([*ats[750], 0])
    affine.SetCenter([1023.5 + shiftpx[1], 1023.5 + shiftpx[0], 0])

    comp750 = sitk.CompositeTransform(3)
    comp750.AddTransform(translate)
    comp750.AddTransform(affine)

    if img.shape[1] == 2:
        return {
            int(key.split("-")[0].split("_")[0]): st(img[:, 0], translate),
            int(key.split("-")[0].split("_")[1]): st(img[:, 1], comp750),
        }

    elif img.shape[1] == 3:
        affine560 = sitk.AffineTransform(3)
        affine560.SetMatrix(As[560].flatten())
        affine560.SetTranslation([*ats[560], 0])
        affine560.SetCenter([1023.5 + shiftpx[1], 1023.5 + shiftpx[0], 0])

        comp560 = sitk.CompositeTransform(3)
        comp560.AddTransform(translate)
        comp560.AddTransform(affine560)
        return {
            int(key.split("-")[0].split("_")[0]): st(img[:, 0], comp560),
            int(key.split("-")[0].split("_")[1]): st(img[:, 1], translate),
            int(key.split("-")[0].split("_")[2]): st(img[:, 2], comp750),
        }
    else:
        raise ValueError(f"Wrong number of channels. {img.shape[1]}")


# affine = sitk.AffineTransform(3)
# affine.SetMatrix(As[750].flatten())
# affine.SetTranslation(ats[750])

# transformed = {
#     int(keys[0].split("-")[0].split("_")[0]): imgs[keys[0]][:12],
#     int(keys[0].split("-")[0].split("_")[1]): st(imgs[keys[0]][12:], affine),
# }
transformed = {}
for i, (name, img) in enumerate(nofids.items()):
    # if name != "8_16_24-003.tif":
    #     continue
    transformed |= (res := run_image(name, img, -shifts[name]))
    print({k: np.mean(v) for k, v in res.items()})
# %%
import matplotlib.pyplot as plt

u = np.stack(list(transformed.values()))[:, 1].swapaxes(0, 2)
plt.imshow(u[400:650, 500:650, :3] / u[400:650, 500:650, :3].max(axis=(0, 1)))
# %%

# %%
out = np.zeros((24, 2048, 2048), dtype=np.uint16)
for i in range(max(transformed)):
    if (i + 1) not in transformed:
        continue
    out[i] = transformed[i + 1][3]  # .max(axis=0)

tifffile.imwrite(
    "trc.tif",
    out,
    compression=22610,
    imagej=True,
)


# %%

# img = out["8_16_24-003.tif"]
# plt.imshow(np.swapaxes(img, 0, 2)[500:650, 500:650] / img[:, 500:650, 500:650].max(axis=(1, 2)))


from sklearn.linear_model import HuberRegressor


def spill_corr(img647: np.ndarray, img568: np.ndarray):
    ok = img647[::4, ::4].flatten() > 2500
    hr = HuberRegressor().fit(
        img647[::4, ::4].flatten()[ok].reshape(-1, 1),
        img568[::4, ::4].flatten()[ok],
    )
    return hr.coef_[0]


# %%


def corr(img647: np.ndarray, img568: np.ndarray, corr: float):
    print(corr)
    return np.clip(img568.astype(float) - img647.astype(float) * corr, 0, 65535).astype(np.uint16)


transformed = {k: v.max(axis=0) for k, v in transformed.items()}
transformed[3] = corr(transformed[1], transformed[9], spill_corr(transformed[1], transformed[9]))
transformed[4] = corr(transformed[2], transformed[10], spill_corr(transformed[2], transformed[10]))
# transformed[7] = corr(transformed[10], transformed[7], spill_corr(transformed[10], transformed[7]))
# transformed[8] = corr(transformed[9], transformed[8], spill_corr(transformed[9], transformed[8]))
# %%
from itertools import chain

bits = sorted(chain.from_iterable(map(lambda x: map(int, x.split("-")[0].split("_")), keys)))

# %%
tifffile.imwrite(
    "tr.tif",
    np.stack([transformed[i].max(axis=0) for i in bits]),
    compression=22610,
    imagej=True,
)

# %%


# def show_registration(img1, img2):

# fids_itk = [fids[keys[0]]]
# fid_ref = sitk.Cast(sitk.GetImageFromArray(fids_itk[-1]), sitk.sitkFloat32)
# for i in range(1, len(keys)):
#     img2 = sitk.Cast(sitk.GetImageFromArray(fids[keys[i]]), sitk.sitkFloat32)
#     final_transform = registration_method.Execute(img1, img2)
#     resampler = sitk.ResampleImageFilter()
#     resampler.SetReferenceImage(fid_ref)

#     resampler.SetTransform(final_transform)
#     print(final_transform)

#     out = resampler.Execute(img2)
#     registered_array = sitk.GetArrayFromImage(out).astype(np.uint16)
#     fids_itk.append(registered_array)
#     print(i)


tifffile.imwrite("fids.tif", np.stack(fids_itk), compression=22610, imagej=True)

# %%


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
import polars as pl

df = pl.read_csv("/raid/data/raw/last/codebook_0_meh_meh.csv")
# %%
cb = np.where(df[:, 3:].to_numpy())[1].reshape(-1, 4)
names = df[:, 0]
# %%
import json

Path("cscbold.json").write_text(json.dumps({name: c.tolist() for name, c in zip(names, cb + 1)}))
# %%
# %%
