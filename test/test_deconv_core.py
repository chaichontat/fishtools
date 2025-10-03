from __future__ import annotations

import numpy as np
import tifffile

from pathlib import Path

from fishtools.preprocess.deconv.core import load_projectors_cached


def _write_synthetic_psf(path: Path, shape: tuple[int, int, int]) -> None:
    z, y, x = shape
    grid_z = np.linspace(-1, 1, z, dtype=np.float32)
    grid_y = np.linspace(-1, 1, y, dtype=np.float32)
    grid_x = np.linspace(-1, 1, x, dtype=np.float32)
    zz, yy, xx = np.meshgrid(grid_z, grid_y, grid_x, indexing="ij")
    psf = np.exp(-(zz**2 + yy**2 + xx**2) / 0.1)
    psf /= psf.sum()
    tifffile.imwrite(path, psf.astype(np.float32))


def test_load_projectors_handles_small_psf(tmp_path: Path, monkeypatch) -> None:
    psf_dir = tmp_path / "fishtools" / "data"
    psf_dir.mkdir(parents=True)
    psf_path = psf_dir / "PSF GL.tif"
    _write_synthetic_psf(psf_path, (11, 15, 13))

    monkeypatch.chdir(tmp_path)

    forward, backward = load_projectors_cached(step=7)

    assert forward.ndim == 4 and backward.ndim == 4
    assert forward.shape == backward.shape
    assert forward.size > 0 and backward.size > 0
    np.testing.assert_allclose(forward.sum(), 1.0, rtol=1e-5)
    np.testing.assert_allclose(backward.sum(), backward.sum(), rtol=1e-5)
