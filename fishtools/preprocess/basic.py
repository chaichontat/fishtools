import logging
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from basicpy import BaSiC
from loguru import logger


def fit_basic(imgs: np.ndarray[np.float32, Any], c: int):
    """Fit a BaSiC model to a single channel.

    Args:
        imgs: Image stack in (z, c, y, x) format.
        c: index of the channel to fit.

    Returns:
        BaSiC model.
    """
    logging.getLogger("jax._src.xla_bridge").setLevel(logging.WARNING)
    import jax

    jax.config.update("jax_platform_name", "cpu")

    import jax
    from basicpy import BaSiC

    with jax.default_device(jax.devices("cpu")[0]):
        logger.info("Fitting channel {}", c)
        basic = BaSiC()  # type: ignore
        basic.fit(imgs[:, c])
        return basic


def plot_basic(basic: BaSiC):
    """Plot the flatfield, darkfield, and baseline of a BaSiC model."""
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    im = axes[0].imshow(basic.flatfield)
    fig.colorbar(im, ax=axes[0])
    axes[0].set_title("Flatfield")
    im = axes[1].imshow(basic.darkfield)
    fig.colorbar(im, ax=axes[1])
    axes[1].set_title("Darkfield")
    axes[2].plot(basic.baseline)
    axes[2].set_xlabel("Frame")
    axes[2].set_ylabel("Baseline")
    axes[0].axis("off")
    axes[1].axis("off")
    fig.tight_layout()
