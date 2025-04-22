# %%
import logging
import pickle
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from cellpose.io import imread, load_train_test_data
from cellpose.models import CellposeModel
from cellpose.train import train_seg
from loguru import logger
from skimage.filters import unsharp_mask
from tifffile import imread

_original_imshow = plt.imshow
plt.imshow = lambda *args, **kwargs: _original_imshow(*args, zorder=1, **kwargs)

logging.basicConfig(level=logging.INFO)

# path = Path("/working/20250407_cs3_2/analysis/deconv/stitch--tl+atp")

path = Path("/working/cellpose-training")

name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # + "polyA"


# %%
def calc_percentile(
    img: np.ndarray, block: tuple[int, int] = (512, 512), n: int = 25, low: float = 10, high: float = 99.5
):
    rand = np.random.default_rng(0)
    x_start = rand.integers(0, img.shape[2] - block[0], n)
    y_start = rand.integers(0, img.shape[3] - block[1], n)
    out = []
    for i, (x, y) in enumerate(zip(x_start, y_start)):
        logger.info(f"Percentile processing {i + 1}/{n}")
        out.append(
            np.percentile(
                unsharp_mask(
                    img[:, :, x : x + block[0], y : y + block[1]],
                    radius=3,
                    preserve_range=True,
                    channel_axis=1,
                ),
                [low, high],
                axis=(0, 2, 3),
            )
        )
    out = np.array(out)
    return np.mean(out, axis=0).T, out


# img = imread(path / "fused.tif")
# percs, _ = calc_percentile(img)

# %%

# %%
# Yes, the typing is wrong.
# model = CellposeModel(gpu=True, pretrained_model=cast(bool, models[0].as_posix()))

models = sorted(
    (file for file in (path / "models").glob("*") if "train_losses" not in file.name),
    key=lambda x: x.stat().st_mtime,
    reverse=True,
)
# assert models

LEARNING_RATES = {
    "AdamW": 0.008,
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
    "percentile": (1, 99.5),
}

# %%
# path_labeled = Path("/working/cellpose-training/embryonic/cs3")
path_labeled = path  # Path("/working/20250407_cs3_2/analysis/deconv/segment--bl+atp/ortho")

for p in path_labeled.rglob("*.npy"):
    try:
        u = np.load(p, allow_pickle=True)
    except Exception as e:
        print(p, e)
        raise e


def concat_output(samples: list[str]) -> tuple[list]:
    first: tuple = load_train_test_data(
        (path / samples[0]).as_posix(),
        # test_dir=(path.parent / "pi-wgatest").as_posix(),
        mask_filter="_seg.npy",
        look_one_level_down=True,
    )
    for sample in samples[1:]:
        _out = load_train_test_data(
            (path / sample).as_posix(),
            # test_dir=(path.parent / "pi-wgatest").as_posix(),
            mask_filter="_seg.npy",
            look_one_level_down=True,
        )
        for lis, new in zip(first, _out):
            if lis is not None and new is not None:
                lis.extend(new)
    return first


# %%
def train(path: Path, normalize_params: dict, name: str | None = None):
    if name is None:
        name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    images, labels, image_names, test_images, test_labels, image_names_test = concat_output([
        "20250407_cs3_2"
    ])
    # return images, labels, image_names, test_images, test_labels, image_names_test

    # e.g. retrain a Cellpose model
    model = CellposeModel(model_type="cyto3", gpu=True)
    # model = CellposeModel(gpu=True, pretrained_model=models[0].as_posix())
    model_path, train_losses, test_losses = train_seg(
        model.net,
        train_data=images,
        train_labels=labels,
        save_path=path,  # /models automatically appended
        channels=[2, 1],
        batch_size=24,
        channel_axis=0,
        test_data=test_images,
        test_labels=test_labels,
        weight_decay=1e-5,
        SGD=False,
        learning_rate=0.008,
        n_epochs=300,
        model_name=name,
        bsize=224,
        normalize=cast(bool, normalize_params),
        min_train_masks=1,
    )


what = train(path_labeled, normalize_params, name="embryonic2")
# %%
