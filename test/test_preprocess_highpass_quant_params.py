from __future__ import annotations

import numpy as np
import zarr

from fishtools.preprocess.highpass import _compute_quant_params_from_reference_plane


def test_highpass_quant_params_samples_multiple_z_planes() -> None:
    # z=1 is bright, others are dark. With strong Z smoothing, highpass should be
    # positive primarily in z=1; quantization upper should reflect that (not ~1.0).
    data = np.zeros((3, 8, 8, 1), dtype=np.uint16)
    data[1, :, :, 0] = 1000

    src = zarr.array(data, chunks=data.shape, dtype=np.uint16)
    params = _compute_quant_params_from_reference_plane(
        src,
        channel=0,
        sigma_px=20.0,
        anisotropy=0.1,
        y_step=8,
        x_step=8,
        pad_xy=0,
        percentile_lo=1.0,
        percentile_hi=99.999,
        use_gpu=False,
    )

    assert params.lower >= 0.0
    assert params.upper > 10.0
