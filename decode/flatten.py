# %%
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from tifffile import imread, imwrite


# flats = [flat / flat.max() for flat in flats]
# %%
def process(path: Path | str, dark, flat):
    print(path)
    path = Path(path)
    img = imread(path)
    res = ((img - dark).astype(np.float32) / flat).astype(np.uint16)
    imwrite(path.parent.parent / "data" / path.name, res, compression=22610, imagej=True)


def run(path: Path | str):
    path = Path(path)
    dark = imread("/raid/data/analysis/dark.tif")
    flat = (imread("/raid/data/analysis/flat_647.tif") - dark).astype(np.float32)
    flat /= np.min(flat)  # Prevents overflow

    imgs = sorted(list(path.glob("*.tif")))
    imgs = [file for file in imgs if not (file.parent.parent / "data" / file.name).exists()]
    print(imgs)
    with ProcessPoolExecutor(64) as exc:
        exc.map(process, imgs, [dark] * len(imgs), [flat] * len(imgs))


if __name__ == "__main__":
    run(Path("/raid/data/raw/last/rawraw"))
