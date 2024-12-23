# %%

import json
import logging
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, TypeVar

import click
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tifffile
import toml
from basicpy import BaSiC
from loguru import logger
from pydantic import BaseModel, Field
from scipy import ndimage
from scipy.ndimage import shift
from tifffile import TiffFile, imread

from fishtools import align_fiducials
from fishtools.analysis.chromatic import Affine

sns.set_theme()

# %%

test = (
    imread("/mnt/archive/starmap/barrel_thicc2/pi_cFos--barrel/pi_cFos-0012.tif")[:-1]
    .reshape(-1, 2, 2048, 2048)
    .max(axis=(0))
)

ref = imread("/mnt/archive/starmap/barrel_thicc2/registered--barrel/reg-0012.tif")[-1]

# %%
plt.imshow(test[0])
# %%
plt.imshow(ref[-1])

# %%
shifts = align_fiducials(
    fids,
    reference=reference,
    debug=debug,
    iterations=4,
    threshold_sigma=config.registration.fiducial.threshold,
    fwhm=config.registration.fiducial.fwhm,
)
