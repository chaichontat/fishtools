import cupy as cp
import numpy as np
import numpy.typing as npt


def unsharp_all(
    img: npt.ArrayLike, crop: None = None, channel_axis: int = 3
):  # crop is for distributed_segmentation
    from cucim.skimage import filters as cucim_filters

    img_np = np.asarray(img)
    img_gpu = cp.asarray(img_np, dtype=cp.float32)
    result_gpu = cucim_filters.unsharp_mask(img_gpu, radius=3, preserve_range=True, channel_axis=channel_axis)
    result = cp.asnumpy(result_gpu)
    del img_gpu, result_gpu
    return result


# Backwards-compat: re-export sample_percentile from segment.normalize
