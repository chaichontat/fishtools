import numpy as np
import numpy.typing as npt
from skimage.filters import unsharp_mask


def unsharp_all(
    img: npt.ArrayLike, crop: None = None, channel_axis: int = 3
):  # crop is for distributed_segmentation
    return unsharp_mask(img, preserve_range=True, radius=3, channel_axis=channel_axis)


# Backwards-compat: re-export sample_percentile from segment.normalize
from fishtools.segment.normalize import sample_percentile  # noqa: E402  (import after unsharp_all)
