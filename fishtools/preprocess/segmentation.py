import numpy as np
from loguru import logger
from skimage.filters import unsharp_mask


def unsharp_all(
    img: np.ndarray, crop: None = None, channel_axis: int = 3
):  # crop is for distributed_segmentation
    return unsharp_mask(img, preserve_range=True, radius=3, channel_axis=channel_axis)


def calc_percentile(
    img: np.ndarray,
    channels: list[int],
    block: tuple[int, int] = (512, 512),
    *,
    n: int = 25,
    low: float = 1,
    high: float = 99,
    seed: int = 0,
):
    rand = np.random.default_rng(seed)
    x_start = rand.integers(0, img.shape[1] - block[0], n * 2)
    y_start = rand.integers(0, img.shape[2] - block[1], n * 2)
    out = []
    j = 0
    channels = [c - 1 for c in channels]  # Convert to 0-based index
    for i, (x, y) in enumerate(zip(x_start, y_start)):
        if j == n:
            break
        logger.info(f"Calculating percentile {j + 1}/{n}")
        _img = img[:, x : x + block[0], y : y + block[1], channels]
        if not np.all(_img[0, :, :, 0]):  # Skip areas with 0 (stitch edge)
            continue
        unsharped = unsharp_all(_img)
        out.append(np.percentile(unsharped, [low, high], axis=(0, 1, 2)))
        logger.debug(f"{out[-1].T}")
        j += 1
    out = np.array(out)
    return np.mean(out, axis=0).T, out
