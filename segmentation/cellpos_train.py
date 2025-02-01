# %%
import logging
import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from cellpose.io import imread, load_train_test_data
from cellpose.models import CellposeModel
from cellpose.train import train_seg
from tifffile import imread

plt.imshow = lambda *args, **kwargs: plt.imshow(*args, zorder=1, **kwargs)

logging.basicConfig(level=logging.INFO)

path = Path.home() / "segmentation" / "pi-wga"


name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # + "polyA"


# %%
# Yes, the typing is wrong.
# model = CellposeModel(gpu=True, pretrained_model=cast(bool, models[0].as_posix()))

models = sorted(
    (file for file in (path / "models").glob("*") if "train_losses" not in file.name),
    key=lambda x: x.stat().st_mtime,
    reverse=True,
)
assert models

LEARNING_RATES = {
    "AdamW": 0.001,
    "AdamW_fine": 0.0001,
    "SGD": 0.05,
}

# normalize (bool, optional): if True, normalize data so 0.0=1st percentile and 1.0=99th percentile of image intensities in each channel;
#     can also pass dictionary of parameters (all keys are optional, default values shown):
#         - "lowhigh"=None : pass in normalization values for 0.0 and 1.0 as list [low, high] (if not None, all following parameters ignored)
#         - "sharpen"=0 ; sharpen image with high pass filter, recommended to be 1/4-1/8 diameter of cells in pixels
#         - "normalize"=True ; run normalization (if False, all following parameters ignored)
#         - "percentile"=None : pass in percentiles to use as list [perc_low, perc_high]
#         - "tile_norm"=0 ; compute normalization in tiles across image to brighten dark areas, to turn on set to window size in pixels (e.g. 100)
#         - "norm3D"=False ; compute normalization across entire z-stack rather than plane-by-plane in stitching mode.
#     Defaults to True.
normalize_params = {
    "lowhigh": None,
    "percentile": (1, 99.9),
    "normalize": True,
    "norm3D": True,
    "sharpen_radius": 0,
    "smooth_radius": 0,
    "tile_norm_blocksize": 0,
    "tile_norm_smooth3D": 1,
    "invert": False,
}

# %%
for p in path.rglob("*.npy"):
    try:
        u = np.load(p, allow_pickle=True)
    except Exception as e:
        print(p, e)
        raise e


# %%
def train(path: Path, name: str | None = None):
    if name is None:
        name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    output = load_train_test_data(
        path.as_posix(),
        test_dir=(path.parent / "pi-wgatest").as_posix(),
        mask_filter="_seg.npy",
        look_one_level_down=True,
    )
    images, labels, image_names, test_images, test_labels, image_names_test = output
    # return images, labels, image_names, test_images, test_labels, image_names_test

    # e.g. retrain a Cellpose model
    model = CellposeModel(model_type="cyto3", gpu=True)
    # model = CellposeModel(gpu=True, pretrained_model=models[0].as_posix())
    model_path, train_losses, test_losses = train_seg(
        model.net,
        train_data=images,
        train_labels=labels,
        save_path=path,  # /models automatically appended
        channels=[1, 2],
        batch_size=16,
        channel_axis=0,
        test_data=test_images,
        test_labels=test_labels,
        weight_decay=1e-5,
        SGD=False,
        learning_rate=0.0008,
        n_epochs=2000,
        model_name=name,
        bsize=224,
        normalize=cast(bool, normalize_params),
    )


what = train(path, name=name)


# dapi = imread("/fast2/3t3clean/analysis/deconv/registered/dapi3/0/fused_1.tif").squeeze()
# polya = imread("/fast2/3t3clean/analysis/deconv/registered/dapi3/2/fused_1.tif").squeeze()
# intensity = scale_image_2x_optimized(img)

# %%
res = model.eval(
    np.array([polya, dapi]),
    batch_size=8,
    channels=[1, 3],
    # normalize=False,  # CANNOT TURN OFF NORMALIZATION
    # Will run into memory issues if turned off.
    flow_threshold=0.4,
    do_3D=False,
    # diameter=model.diam_labels,
)

with open("/fast2/3t3clean/analysis/deconv/registered/dapi3/cellpose_polyA_result.pkl", "wb") as f:
    f.write(pickle.dumps(res))

# %%
