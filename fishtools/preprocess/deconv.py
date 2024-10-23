from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger
from tifffile import TiffFile

from fishtools.utils.pretty_print import progress_bar


def scale_deconv(
    img: np.ndarray,
    idx: int,
    *,
    global_deconv_scaling: np.ndarray,
    metadata: dict[str, Any],
    name: str | None = None,
    debug: bool = False,
):
    """Scale back deconvolved image using global scaling.

    Args:
        img: Original image, e.g. 1_9_17
        idx: Channel index in original image e.g. 0, 1, 2, ...
    """
    global_deconv_scaling = global_deconv_scaling.reshape((2, -1))
    m_ = global_deconv_scaling[0, idx]
    s_ = global_deconv_scaling[1, idx]

    # Same as:
    # scaled = s_ * ((img / self.metadata["deconv_scale"][idx] + self.metadata["deconv_min"][idx]) - m_)
    scale_factor = s_ / metadata["deconv_scale"][idx]
    offset = s_ * (metadata["deconv_min"][idx] - m_)
    scaled = scale_factor * img + offset

    if debug:
        logger.debug(f"Deconvolution scaling: {scale_factor}")
    if name and scaled.max() > 65535:
        logger.warning(f"Scaled image {name} has max > 65535.")

    if np.all(scaled < 0):
        print(img.min(), img.max())
        logger.warning(f"Scaled image {name} has all negative values.")
        print(scale_factor, offset, scaled.min(), scaled.max())

    return np.clip(scaled, 0, 65535)


def _compute_range(path: Path, round_: str, *, perc_min: float = 99.9, perc_scale: float = 0.1):
    """
    Find the min and scale of the deconvolution for all files in a directory.
    The reverse scaling equation is:
                 s_global
        scaled = -------- * img + s_global * (min - m_global)
                  scale
    Hence we want scale_global to be as low as possible
    and min_global to be as high as possible.
    """
    files = sorted(path.glob(f"{round_}*/*.tif"))
    n_c = len(round_.split("_"))
    n = len(files)

    deconv_min = np.zeros((n, n_c))
    deconv_scale = np.zeros((n, n_c))
    logger.info(f"Found {n} files")
    if not files:
        raise FileNotFoundError(f"No files found in {path}")

    with progress_bar(len(files)) as pbar:
        for i, f in enumerate(files):
            with TiffFile(f) as tif:
                try:
                    deconv_min[i, :] = tif.shaped_metadata[0]["deconv_min"]
                    deconv_scale[i, :] = tif.shaped_metadata[0]["deconv_scale"]
                except KeyError:
                    raise AttributeError("No deconv metadata found.")
            pbar()

    logger.info("Calculating percentiles")
    m_ = np.percentile(deconv_min, perc_min, axis=0)
    s_ = np.percentile(deconv_scale, perc_scale, axis=0)

    (path / "deconv_scaling").mkdir(exist_ok=True)
    np.savetxt(path / "deconv_scaling" / f"{round_}.txt", np.vstack([m_, s_]))
    logger.info(f"Saved to {path / 'deconv_scaling' / f'{round_}.txt'}")
