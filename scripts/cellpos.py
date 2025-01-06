# %%
import logging
import pickle

import matplotlib.pyplot as plt
import numpy as np
from cellpose import models
from cellpose.io import imread
from tifffile import imread

plt.imshow = lambda *args, **kwargs: plt.imshow(*args, zorder=1, **kwargs)

logging.basicConfig(level=logging.INFO)


def scale_image_2x_optimized(image: np.ndarray):
    """
    Scale a 2D image array up by two-fold using nearest-neighbor interpolation.

    Parameters:
    image (numpy.ndarray): Input 2D image array

    Returns:
    numpy.ndarray: Scaled 2D image array
    """
    scaled_image = np.repeat(np.repeat(image, 2, axis=0), 2, axis=1)

    return scaled_image


# %%
# model_type='cyto' or model_type='nuclei'
model = models.CellposeModel(gpu=True, pretrained_model="/fast2/3t3clean/CP_dapi_polyA")
# %%

dapi = imread("/fast2/3t3clean/analysis/deconv/registered/dapi3/0/fused_1.tif").squeeze()
polya = imread("/fast2/3t3clean/analysis/deconv/registered/dapi3/2/fused_1.tif").squeeze()
# intensity = scale_image_2x_optimized(img)
# %%
res = model.eval(
    np.array([polya, dapi]),
    batch_size=8,
    channels=[1, 2],
    # normalize=False,  # CANNOT TURN OFF NORMALIZATION
    # Will run into memory issues if turned off.
    flow_threshold=0.4,
    do_3D=False,
    # diameter=model.diam_labels,
)

with open("/fast2/3t3clean/analysis/deconv/registered/dapi3/cellpose_polyA_result.pkl", "wb") as f:
    f.write(pickle.dumps(res))
