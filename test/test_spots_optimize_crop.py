from __future__ import annotations

from fishtools.preprocess.spots.align_prod import _run_spatial_slices


def test_optimize_always_uses_fixed_top_left_980_crop() -> None:
    y_slice, x_slice = _run_spatial_slices((2048, 2048), split=3, optimize_crop=True)
    assert (y_slice.start, y_slice.stop, x_slice.start, x_slice.stop) == (40, 1024, 40, 1024)


def test_decode_split_keeps_quadrant_cut_special_case_1960() -> None:
    y_slice, x_slice = _run_spatial_slices((1960, 1960), split=0, optimize_crop=False)
    assert (y_slice.start, y_slice.stop, x_slice.start, x_slice.stop) == (0, 1024, 0, 1024)
