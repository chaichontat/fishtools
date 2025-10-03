from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
from basicpy import BaSiC
from scipy import ndimage

from fishtools.preprocess.image_loader import load_image_components


@dataclass
class Image:
    """Lightweight container and helpers for loading and preparing images.

    This class was moved from `fishtools.preprocess.cli_register` to
    `fishtools.io.image` to centralize IO-related types. The public API is
    preserved for backward compatibility.
    """

    name: str
    idx: int
    nofid: np.ndarray
    fid: np.ndarray
    fid_raw: np.ndarray
    bits: list[str]
    powers: dict[str, float]
    metadata: dict[str, Any]
    global_deconv_scaling: np.ndarray
    basic: Callable[[], dict[str, BaSiC] | None]

    CHANNELS = [f"ilm{n}" for n in ["405", "488", "560", "650", "750"]]

    @classmethod
    def from_file(
        cls,
        path: Path,
        *,
        discards: dict[str, list[str]] | None = None,
        n_fids: int = 1,
        channels: Sequence[str] | None = None,
        image_size: int | None = None,
        context: Any | None = None,
    ) -> "Image":
        """Load components and construct Image.

        Supports both legacy and context-aware workflows:
        - Legacy: specify none of `channels`, `image_size`, `context`.
        - Context-aware: pass `context` with `system_config.available_channels`,
          `img_config.image_size`, `img_config.log_sigma`, and `img_config.percentiles`.
        """

        # Derive channels and image_size from parameters or context
        if context is not None:
            channels = channels or [f"ilm{n}" for n in context.system_config.available_channels]
            image_size = image_size or context.img_config.image_size

        components = load_image_components(
            path,
            discards=discards,
            n_fids=n_fids,
            channels=channels or tuple(f"ilm{n}" for n in ("405", "488", "560", "650", "750")),
            image_size=image_size or 2048,
        )

        # Compute fiducial with either defaults or context parameters
        if context is not None:
            fid = cls.loG_fids(
                components.fid_raw,
                sigma=float(context.img_config.log_sigma),
                percentiles=tuple(context.img_config.percentiles),
                uniform_policy="zeros",
            )
        else:
            fid = cls.loG_fids(components.fid_raw)

        return cls(
            name=components.name,
            idx=components.idx,
            nofid=components.nofid,
            fid=fid,
            fid_raw=components.fid_raw,
            bits=components.bits,
            powers=components.powers,
            metadata=components.metadata,
            global_deconv_scaling=components.global_deconv_scaling,
            basic=components.basic_loader,
        )

    @staticmethod
    def loG_fids(
        fid: np.ndarray,
        *,
        sigma: float = 3.0,
        percentiles: Sequence[float] = (1.0, 99.99),
        uniform_policy: str = "error",  # "error" or "zeros"
    ) -> np.ndarray:
        """Compute a normalized Laplacian-of-Gaussian fiducial image.

        - Accepts 2D or 3D input; 3D inputs are max-projected.
        - Normalizes using provided percentiles.
        - Uniform handling controlled by `uniform_policy`.
        """
        if len(fid.shape) == 3:
            fid = fid.max(axis=0)

        temp = -ndimage.gaussian_laplace(fid.astype(np.float32).copy(), sigma=sigma)
        temp -= temp.min()
        p0, p1 = percentiles
        percs = np.percentile(temp, [p0, p1])

        if percs[1] - percs[0] == 0:
            if uniform_policy == "zeros":
                return np.zeros_like(temp, dtype=np.float32)
            raise ValueError("Uniform image")
        temp = (temp - percs[0]) / (percs[1] - percs[0])
        return temp.astype(np.float32, copy=False)


__all__ = ["Image"]
