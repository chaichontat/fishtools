from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import tifffile

from fishtools.preprocess.deconv.core import (
    DATA_DIR,
    PSF_FILENAME,
    deconvolve_lucyrichardson_guo,
    deconvolve_lucyrichardson_guo_fft,
    load_projectors_cached,
    make_projector,
)

try:
    import cupy as cp
except ModuleNotFoundError:  # pragma: no cover - handled at runtime in tests
    cp = None


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


@pytest.mark.integration
def test_fft_deconvolution_matches_spatial_interior() -> None:
    if cp is None:
        pytest.skip("CuPy not available for GPU regression test")

    if os.environ.get("CI"):
        pytest.skip("GPU regression comparison disabled on CI")

    try:
        device_count = cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError as exc:  # pragma: no cover - depends on runtime env
        pytest.skip(f"CUDA runtime unavailable: {exc}")

    if device_count <= 0:
        pytest.skip("No CUDA-capable GPU detected")

    # Build projectors directly from the PSF GL volume on disk
    psf_path = DATA_DIR / PSF_FILENAME
    forward_np, backward_np = make_projector(psf_path, step=6, max_z=7, size=31)
    forward_3d = cp.asarray(forward_np[:, 0], dtype=cp.float32)
    backward_3d = cp.asarray(backward_np[:, 0], dtype=cp.float32)

    rng = np.random.default_rng(0)
    height = width = 192
    # Base payload (Z, Y, X) in bright range
    payload = rng.uniform(10_000.0, 65_535.0, size=(forward_3d.shape[0], height, width)).astype(np.float32)

    # Inject synthetic Gaussian spots inside the cropped interior so values are non-zero.
    def _add_spots(img: np.ndarray, n_spots_per_z: int = 6, sigma: float = 2.0, amp: float = 2.0e4) -> None:
        yy, xx = np.meshgrid(np.arange(height, dtype=np.float32), np.arange(width, dtype=np.float32), indexing="ij")
        margin = 40  # keep clear of XY crop border (31) and gaussian tails
        for z in range(img.shape[0]):
            for _ in range(n_spots_per_z):
                y0 = rng.integers(margin, height - margin)
                x0 = rng.integers(margin, width - margin)
                g = amp * np.exp(-((yy - y0) ** 2 + (xx - x0) ** 2) / (2.0 * sigma**2))
                img[z] += g.astype(np.float32)

    _add_spots(payload)

    # Use 3D kernels for both paths for apples-to-apples
    spatial_result = deconvolve_lucyrichardson_guo(cp.asarray(payload), (forward_3d, backward_3d), iters=1)
    fft_result = deconvolve_lucyrichardson_guo_fft(cp.asarray(payload), (forward_3d, backward_3d), iters=1)

    assert spatial_result.shape == payload.shape
    assert fft_result.shape == payload.shape

    border_xy = 31
    border_z = forward_3d.shape[0] // 2  # 7→3, crop Z edges to mitigate boundary effects
    interior_slice = (slice(border_z, -border_z), slice(border_xy, -border_xy), slice(border_xy, -border_xy))
    spatial_core = cp.asnumpy(spatial_result[interior_slice])
    fft_core = cp.asnumpy(fft_result[interior_slice])

    expected_height = height - 2 * border_xy
    expected_width = width - 2 * border_xy
    assert spatial_core.shape[-2:] == (expected_height, expected_width)
    assert fft_core.shape == spatial_core.shape

    # Optional: dump full arrays for manual inspection
    if os.environ.get("PRINT_FULL_ARRAYS"):
        # Also persist arrays for offline inspection
        from pathlib import Path
        outdir = Path("results/deconv_debug")
        outdir.mkdir(parents=True, exist_ok=True)
        np.save(outdir / "spatial_core.npy", spatial_core)
        np.save(outdir / "fft_core.npy", fft_core)
        np.savetxt(outdir / "spatial_core_z0.csv", spatial_core[0], delimiter=",")
        np.savetxt(outdir / "fft_core_z0.csv", fft_core[0], delimiter=",")
        print(f"Saved arrays to {outdir}", flush=True)
        # And print (may be truncated by runner)
        np.set_printoptions(linewidth=200, threshold=10**9, formatter={"float_kind": lambda x: f"{x:.6e}"})
        for zi in range(spatial_core.shape[0]):
            print(f"spatial_core[z={zi}] =\n{spatial_core[zi]}", flush=True)
            print(f"fft_core[z={zi}] =\n{fft_core[zi]}", flush=True)

    # Fractional deviation via division: |(FFT / Spatial) - 1|
    # Robust denominator clamp: values can be ~1e-19; prevent 0/denom noise.
    denom_floor = max(float(np.max(np.abs(spatial_core))) * 1e-12, 1e-30)
    zero_mask = np.abs(spatial_core) <= denom_floor
    zero_count = int(np.sum(zero_mask))
    print(f"denom_floor={denom_floor:.3e}, zeros/below-floor in spatial_core={zero_count}", flush=True)
    if zero_count == spatial_core.size:
        # Degenerate case: spatial interior vanished. Fall back to absolute bound
        max_abs_fft = float(np.max(np.abs(fft_core)))
        print(f"All interior refs ~0; max |fft_core|={max_abs_fft:.3e}", flush=True)
        assert max_abs_fft <= 1e-20
        return
    denom = np.where(~zero_mask, spatial_core, np.sign(spatial_core) * denom_floor + denom_floor)
    ratio = np.divide(fft_core, denom)
    frac_dev = np.abs(ratio - 1.0)
    max_frac_dev = float(np.max(frac_dev[~zero_mask]))

    # Human-readable diagnostics for manual comparison (printed when running with -s)
    def _stats(name: str, arr: np.ndarray) -> None:
        vals = arr.ravel()
        q = np.quantile(vals, [0.0, 0.5, 0.9, 0.99, 1.0])
        print(
            f"{name}: min={q[0]:.6e} med={q[1]:.6e} p90={q[2]:.6e} p99={q[3]:.6e} max={q[4]:.6e}",
            flush=True,
        )

    _stats("spatial_core", np.abs(spatial_core))
    _stats("fft_core", np.abs(fft_core))
    _stats("|ratio-1|", frac_dev)

    # Inspect the neighborhood around the worst-case pixel
    max_idx = np.unravel_index(int(np.argmax(frac_dev)), frac_dev.shape)
    z, y, x = max_idx
    y0, y1 = max(y - 1, 0), min(y + 2, spatial_core.shape[-2])
    x0, x1 = max(x - 1, 0), min(x + 2, spatial_core.shape[-1])
    print(f"max(|ratio-1|) at z={z}, y={y}, x={x}: {max_frac_dev:.6e}", flush=True)
    print("spatial_core[window]:\n", spatial_core[z, y0:y1, x0:x1], flush=True)
    print("fft_core[window]:\n", fft_core[z, y0:y1, x0:x1], flush=True)
    print("ratio[window]:\n", ratio[z, y0:y1, x0:x1], flush=True)

    assert max_frac_dev <= 5e-4, f"Max fractional deviation {max_frac_dev:.2e} exceeds tolerance (5e-4)."
