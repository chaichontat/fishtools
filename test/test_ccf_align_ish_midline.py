import math
from pathlib import Path

import numpy as np
import pytest

from ccf.align_ish_midline import _compute_midline_warp_from_yz_affine, _parse_section_image_id
from ccf.align_ish_midline import _midline_from_dim_spacing_um
from ccf.align_ish_midline import fit_z_affine_px_to_um


@pytest.mark.unit
def test_fit_z_affine_px_to_um_recovers_coeffs() -> None:
    # z = 2x - 3y + 7
    a0, b0, c0 = 2.0, -3.0, 7.0
    pts = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0], [3.0, 7.0]], dtype=float)
    z = a0 * pts[:, 0] + b0 * pts[:, 1] + c0

    a, b, c = fit_z_affine_px_to_um(pts, z)
    assert math.isclose(a, a0, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(b, b0, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(c, c0, rel_tol=1e-12, abs_tol=1e-12)


@pytest.mark.unit
def test_parse_section_image_id_from_filename() -> None:
    assert _parse_section_image_id(Path("Acadvl_100364328.jpg")) == 100364328


@pytest.mark.unit
def test_compute_midline_warp_from_yz_affine_identity_like() -> None:
    # y_um = y_px, z_um = x_px, midline at z=10 => midline should land at output center.
    w, h = 21, 11
    ay, by, cy = 0.0, 1.0, 0.0
    az, bz, cz = 1.0, 0.0, 0.0
    midline_z = 10.0

    (out_w, out_h), M, y_scale, z_scale = _compute_midline_warp_from_yz_affine(
        w, h, ay, by, cy, az, bz, cz, midline_z
    )
    assert (out_w, out_h) == (w, h)
    assert math.isclose(y_scale, 1.0, abs_tol=1e-12)
    assert math.isclose(z_scale, 1.0, abs_tol=1e-12)

    out_cx = 0.5 * (out_w - 1)
    out_cy = 0.5 * (out_h - 1)
    # Any point on midline: x=10, arbitrary y. Pick y=center.
    x, y = 10.0, out_cy
    u, v = (M @ np.array([x, y, 1.0])).tolist()
    assert math.isclose(u, out_cx, abs_tol=1e-9)
    assert math.isclose(v, out_cy, abs_tol=1e-9)


@pytest.mark.unit
def test_midline_from_dim_spacing_um_picks_median_extent() -> None:
    # Typical 100um gridAnnotation dims: AP ~13.2mm, DV ~8.0mm, ML ~11.4mm.
    dim = [133, 81, 115]
    spacing = [100.0, 100.0, 100.0]
    # median extent is 11400 -> midline 5700
    assert math.isclose(_midline_from_dim_spacing_um(dim, spacing), 5700.0, abs_tol=1e-9)
