import random
import signal
import time
import sys
import types
from contextlib import suppress
from pathlib import Path

import numpy as np
import pytest
import tifffile
from scipy import fft as _scipy_fft
from scipy import ndimage as _scipy_ndimage
from scipy import sparse as _scipy_sparse


_DEFAULT_TIMEOUT_SECONDS = 30


class _TestTimeoutError(RuntimeError):
    """Raised when a test exceeds the allotted runtime."""


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item: pytest.Item):
    timeout_marker = item.get_closest_marker("timeout")
    timeout = None
    if timeout_marker is not None:
        if timeout_marker.args:
            timeout = float(timeout_marker.args[0])
        else:
            timeout = float(timeout_marker.kwargs.get("seconds", _DEFAULT_TIMEOUT_SECONDS))
    if timeout is None:
        yield
        return
    if not hasattr(signal, "SIGALRM"):
        yield
        return

    def _signal_handler(signum, frame):  # pragma: no cover - relies on OS signals
        raise _TestTimeoutError(f"Test exceeded {timeout:.1f} seconds")

    previous_handler = signal.getsignal(signal.SIGALRM)
    with suppress(AttributeError):
        signal.setitimer(signal.ITIMER_REAL, 0.0)
    signal.signal(signal.SIGALRM, _signal_handler)
    with suppress(AttributeError):
        signal.setitimer(signal.ITIMER_REAL, timeout)
    try:
        outcome = yield
    finally:
        with suppress(AttributeError):
            signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)
    exc_info = outcome.excinfo
    if exc_info is not None and issubclass(exc_info[0], _TestTimeoutError):
        pytest.fail(str(exc_info[1]), pytrace=False)


# Lightweight stub for heavy optional dependency 'scanpy' used in import-time
# paths (e.g., fishtools.postprocess). This avoids importing the real package
# during tests where it is not exercised.
if "scanpy" not in sys.modules:
    _sc = types.ModuleType("scanpy")
    _sc.pp = types.SimpleNamespace(
        filter_cells=lambda *a, **k: None,
        filter_genes=lambda *a, **k: None,
        normalize_total=lambda *a, **k: {"X": None},
        log1p=lambda *a, **k: None,
        calculate_qc_metrics=lambda *a, **k: None,
    )
    _sc.tl = types.SimpleNamespace(leiden=lambda *a, **k: None)
    _sc.pl = types.SimpleNamespace(
        embedding=lambda *a, **k: types.SimpleNamespace(axes=[]),
        rank_genes_groups=lambda *a, **k: None,
    )
    _sc.get = types.SimpleNamespace(
        rank_genes_groups_df=lambda *a, **k: types.SimpleNamespace(head=lambda n: {"names": []})
    )
    _sc.experimental = types.SimpleNamespace(
        pp=types.SimpleNamespace(
            highly_variable_genes=lambda *a, **k: None,
            normalize_pearson_residuals=lambda *a, **k: None,
        )
    )
    sys.modules["scanpy"] = _sc


if "cupy" not in sys.modules:

    class _CuPyModule(types.ModuleType):
        """NumPy-backed stub providing a minimal CuPy surface for tests."""

        ndarray = np.ndarray

        def __getattr__(self, name: str):  # type: ignore[override]
            if hasattr(np, name):
                return getattr(np, name)
            raise AttributeError(name)

        def array(self, *args, **kwargs):
            return np.array(*args, **kwargs)

        def asarray(self, *args, **kwargs):
            return np.asarray(*args, **kwargs)

        def zeros_like(self, *args, **kwargs):
            return np.zeros_like(*args, **kwargs)

        def ones_like(self, *args, **kwargs):
            return np.ones_like(*args, **kwargs)

        def empty_like(self, *args, **kwargs):
            return np.empty_like(*args, **kwargs)

        def empty(self, *args, **kwargs):
            return np.empty(*args, **kwargs)

        def clip(self, *args, **kwargs):
            return np.clip(*args, **kwargs)

        def ElementwiseKernel(self, *args, **kwargs):  # type: ignore[invalid-name]
            def _kernel(*_args: object, **_kwargs: object) -> None:
                raise RuntimeError("ElementwiseKernel stub invoked in tests; GPU path not supported.")

            return _kernel

        def asnumpy(self, array, *args, **kwargs):
            return np.asarray(array, *args, **kwargs)

        def get_array_module(self, *_: object, **__: object):
            return np

    _cupy = _CuPyModule("cupy")
    sys.modules["cupy"] = _cupy

    cupyx = types.ModuleType("cupyx")
    sys.modules["cupyx"] = cupyx

    cupyx_scipy = types.ModuleType("cupyx.scipy")
    sys.modules["cupyx.scipy"] = cupyx_scipy

    cupyx_ndimage = types.ModuleType("cupyx.scipy.ndimage")
    cupyx_ndimage.convolve = _scipy_ndimage.convolve
    cupyx_ndimage.gaussian_filter = _scipy_ndimage.gaussian_filter
    cupyx_ndimage.zoom = _scipy_ndimage.zoom
    cupyx_ndimage.rank_filter = _scipy_ndimage.rank_filter
    cupyx_ndimage.uniform_filter = _scipy_ndimage.uniform_filter
    sys.modules["cupyx.scipy.ndimage"] = cupyx_ndimage

    cupyx_scipy_sparse = types.ModuleType("cupyx.scipy.sparse")
    cupyx_scipy_sparse.csc_matrix = _scipy_sparse.csc_matrix
    cupyx_scipy_sparse.csr_matrix = _scipy_sparse.csr_matrix
    cupyx_scipy_sparse.coo_matrix = _scipy_sparse.coo_matrix
    cupyx_scipy_sparse.spmatrix = _scipy_sparse.spmatrix
    sys.modules["cupyx.scipy.sparse"] = cupyx_scipy_sparse

    cupyx_scipy.ndimage = cupyx_ndimage
    cupyx_scipy.sparse = cupyx_scipy_sparse
    cupyx.scipy = cupyx_scipy

if "cucim" not in sys.modules:
    cucim = types.ModuleType("cucim")
    cucim_skimage = types.ModuleType("cucim.skimage")
    cucim_transform = types.ModuleType("cucim.skimage.transform")
    cucim_transform.downscale_local_mean = _skimage_downscale_local_mean
    sys.modules["cucim"] = cucim
    sys.modules["cucim.skimage"] = cucim_skimage
    sys.modules["cucim.skimage.transform"] = cucim_transform
    cucim.skimage = cucim_skimage
    cucim_skimage.transform = cucim_transform

    class _CudaRuntime(types.SimpleNamespace):
        @staticmethod
        def getDeviceCount() -> int:
            return 1

        @staticmethod
        def deviceSynchronize() -> None:
            return None

        @staticmethod
        def getDeviceProperties(_: int) -> dict[str, bytes]:
            return {"name": b"FakeGPU"}

    class _CudaDevice:
        def __init__(self, idx: int):
            self.idx = idx

        def use(self) -> None:
            return None

    class _CudaStream:
        def synchronize(self) -> None:
            return None

    class _CudaEvent:
        def __init__(self) -> None:
            self._timestamp: float | None = None

        def record(self, stream: _CudaStream | None = None) -> None:
            del stream
            self._timestamp = time.perf_counter()

        def synchronize(self) -> None:
            return None

        @property
        def timestamp(self) -> float:
            if self._timestamp is None:
                return time.perf_counter()
            return self._timestamp

    _current_stream = _CudaStream()

    def _get_current_stream() -> _CudaStream:
        return _current_stream

    def _get_elapsed_time(start: _CudaEvent, end: _CudaEvent) -> float:
        return max(0.0, (end.timestamp - start.timestamp) * 1_000.0)

    _cupy.cuda = types.SimpleNamespace(
        runtime=_CudaRuntime(),
        Device=lambda idx: _CudaDevice(idx),
        Stream=_CudaStream,
        Event=_CudaEvent,
        get_current_stream=_get_current_stream,
        get_elapsed_time=_get_elapsed_time,
        nvtx=types.SimpleNamespace(RangePush=lambda *_args, **_kwargs: None, RangePop=lambda: None),
    )


@pytest.fixture(autouse=True)
def _seed_all() -> None:
    """Deterministic RNG across tests to reduce flakiness."""
    random.seed(0)
    np.random.seed(0)


@pytest.fixture
def mock_workspace(tmp_path: Path) -> Path:
    """Create a mock workspace directory structure with registered TIFFs.

    - Creates analysis/deconv/registered--{roi}+{cb}/ folders
    - Writes 5 small registered TIFFs per ROI/CB with metadata['key']
    - Creates opt_{cb} folders with minimal optimization files
    """
    base = tmp_path / "analysis" / "deconv"

    # Create registered directories with different ROIs and codebooks
    for roi in ["roi1", "roi2"]:
        for cb in ["cb1", "cb2"]:
            reg_dir = base / f"registered--{roi}+{cb}"
            reg_dir.mkdir(parents=True)

            # Create mock registered TIFF files
            for i in range(5):
                tiff_path = reg_dir / f"reg{i:04d}.tif"
                data = np.random.randint(0, 65535, (3, 4, 100, 100), dtype=np.uint16)
                tifffile.imwrite(
                    tiff_path,
                    data,
                    metadata={"key": list(range(1, 37))},
                    compression="zlib",
                )

    # Create optimization directories
    for cb in ["cb1", "cb2"]:
        opt_dir = base / f"opt_{cb}"
        opt_dir.mkdir(parents=True)
        (opt_dir / "global_scale.txt").write_text("1.0 1.1 1.2 1.3")
        (opt_dir / "global_min.txt").write_text("0.1 0.2 0.3 0.4")
        (opt_dir / "percentiles.json").write_text("[]")

    # Codebooks dir (empty, tests may write into it)
    (tmp_path / "codebooks").mkdir(exist_ok=True)

    return tmp_path


@pytest.fixture
def raw_bleed_setup(tmp_path: Path) -> tuple[Path, Path, float, float]:
    """Create paired sample/blank directories with a known bleed-through relationship."""

    analysis_path = tmp_path / "analysis"
    preprocessed_dir = tmp_path / "preprocessed"
    analysis_path.mkdir(parents=True)
    preprocessed_dir.mkdir(parents=True)

    slope = 0.18
    intercept = 75.0

    rng = np.random.default_rng(42)
    for i in range(6):
        blank_slice = rng.uniform(200, 800, size=(64, 64)).astype(np.float32)
        image_slice = blank_slice * slope + intercept + rng.normal(0, 3, size=blank_slice.shape)

        np.savez_compressed(
            preprocessed_dir / f"reg-{i:04d}_key-1.npz",
            image_slice=image_slice,
            blank_slice=blank_slice,
        )

    return analysis_path, preprocessed_dir, slope, intercept


@pytest.fixture
def registered_stack_data(
    tmp_path: Path,
) -> tuple[Path, np.ndarray, dict[str, int], float, float, np.ndarray, np.ndarray]:
    """Create a synthetic registered stack with signal tied to a blank channel."""

    path = tmp_path / "registered_stack.tif"
    slope = 0.22
    intercept = 40.0

    rng = np.random.default_rng(7)
    blank = rng.uniform(300, 900, size=(8, 64, 64)).astype(np.float32)
    signal = blank * slope + intercept + rng.normal(0, 1.5, size=blank.shape)
    other = rng.uniform(50, 120, size=blank.shape).astype(np.float32)
    fid = rng.uniform(0, 1, size=blank.shape).astype(np.float32)

    stack = np.stack([signal, blank, other, fid], axis=1).astype(np.uint16)
    channel_names = ["geneA", "Blank-560", "geneB", "fiducial"]
    channel_index = {name: idx for idx, name in enumerate(channel_names)}

    tifffile.imwrite(path, stack, metadata={"key": channel_names})

    return path, stack, channel_index, slope, intercept, signal, blank
