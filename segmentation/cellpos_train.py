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
path = Path("/working/cellpose-training")

# name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # + "polyA"

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

LEARNING_RATES = {"AdamW": 0.008, "AdamW_fine": 0.0001, "SGD": 0.05}

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


def concat_output(
    path: Path, samples: list[str], mask_filter: list[str] | str = "_seg.npy", one_level_down: bool = False
) -> tuple[list]:
    if isinstance(mask_filter, list) and len(samples) != len(mask_filter):
        raise ValueError("Number of samples must match number of mask filters.")

    first: tuple = load_train_test_data(
        (path / samples[0]).as_posix(),
        # test_dir=(path.parent / "pi-wgatest").as_posix(),
        mask_filter=mask_filter if isinstance(mask_filter, str) else mask_filter[0],
        look_one_level_down=one_level_down,
    )
    for i, sample in enumerate(samples[1:], 1):
        _out = load_train_test_data(
            (path / sample).as_posix(),
            # test_dir=(path.parent / "pi-wgatest").as_posix(),
            mask_filter=mask_filter if isinstance(mask_filter, str) else mask_filter[i],
            look_one_level_down=one_level_down,
        )
        for lis, new in zip(first, _out):
            if lis is not None and new is not None:
                lis.extend(new)
    # images, labels, image_names, test_images, test_labels, image_names_test
    return first


# %%
def train(out: tuple[...], channels: tuple[int, int], name: str | None = None, n_epochs: int = 200):
    if name is None:
        name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # images, labels, image_names, test_images, test_labels, image_names_test = [
    #     [*a, *b] if a is not None and b is not None else None for a, b in zip(out1, out2)
    # ]

    images, labels, image_names, test_images, test_labels, image_names_test = out

    # e.g. retrain a Cellpose model
    model = CellposeModel(model_type="cyto3", gpu=True)
    # model = CellposeModel(gpu=True, pretrained_model=models[0].as_posix())
    model_path, train_losses, test_losses = train_seg(
        model.net,
        train_data=images,
        train_labels=labels,
        save_path=path,  # /models automatically appended
        channels=channels,
        batch_size=24,
        channel_axis=0,
        test_data=test_images,
        test_labels=test_labels,
        weight_decay=1e-5,
        SGD=False,
        learning_rate=0.008,
        n_epochs=n_epochs,
        model_name=name,
        bsize=224,
        normalize=cast(bool, normalize_params),
        min_train_masks=1,
    )


out2 = concat_output(
    path,
    [
        "20250407_cs3_2",
        "20250317_benchmark_mousecommon",
        # "20250409_2486/segment--hippo+polyA",
    ],
    mask_filter="_seg.npy",
    one_level_down=True,
)


out2 = concat_output(
    path,
    [
        "20250407_cs3_2/segment--bl+atp/ortho",
        "20250407_cs3_2/segment--br+atp/ortho",
        # "20250409_2486/segment--hippo+polyA",
    ],
    mask_filter="_seg.npy",
    one_level_down=True,
)

outz = concat_output(
    path,
    [
        "z-self-supervised",
        # "20250409_2486/segment--hippo+polyA",
    ],
    mask_filter="_masks",
    one_level_down=True,
)


ou = [[*a, *b] if a is not None and b is not None else None for a, b in zip(out2, outz)]


what = train(ou, channels=(1, 2), name="embryonicortho", n_epochs=500)
# %%
