import os
import pickle
import queue
import threading
import types
from pathlib import Path

import json

import numpy as np
import pytest
import tifffile
import cupy as cp


def _install_cuda_stub() -> None:
    runtime = types.SimpleNamespace(
        getDeviceCount=lambda: 1,
        CUDARuntimeError=RuntimeError,
    )
    dummy_stream = types.SimpleNamespace(synchronize=lambda: None)

    def _event_factory():
        return types.SimpleNamespace(record=lambda *args, **kwargs: None, synchronize=lambda: None)

    cp.cuda = types.SimpleNamespace(  # type: ignore[attr-defined]
        runtime=runtime,
        get_current_stream=lambda: dummy_stream,
        Event=lambda: _event_factory(),
        get_elapsed_time=lambda start, end: 0.0,
        Device=lambda _: types.SimpleNamespace(use=lambda: None),
    )


_install_cuda_stub()

from fishtools.preprocess.cli_deconv import (
    DeconvolutionTileProcessor,
    ProcessorConfig,
    filter_pending_files,
    load_global_scaling,
    make_processor_factory,
    multi_run,
    _DEFAULT_OUTPUT_MODE,
    parse_device_spec,
    run,
    run_multi_gpu,
)
from fishtools.preprocess.deconv.backend import Float32HistBackend, OutputArtifacts, U16PrenormBackend
from fishtools.preprocess.config import DeconvolutionOutputMode


class DummyBasic:
    def __init__(self, darkfield: np.ndarray, flatfield: np.ndarray) -> None:
        self.darkfield = darkfield
        self.flatfield = flatfield


FAILED_PROCESS_LOG: list[Path] = []


class _FailingOnceProcessor:
    def __init__(self) -> None:
        self._failed = False

    def setup(self) -> None:
        return None

    def teardown(self) -> None:
        return None

    def process(self, path: Path) -> float:  # noqa: ARG002 - path unused for failure logic
        FAILED_PROCESS_LOG.append(path)
        if not self._failed:
            self._failed = True
            raise RuntimeError("boom")
        return 0.01


def _failing_processor_factory(_: int) -> _FailingOnceProcessor:
    return _FailingOnceProcessor()


class _DummyQueue(queue.Queue):
    def __init__(self, maxsize: int = 0) -> None:
        super().__init__(maxsize)

    def close(self) -> None:  # pragma: no cover - threads don't need explicit close
        return None

    def cancel_join_thread(self) -> None:  # pragma: no cover - threads don't need join helpers
        return None


class _DummyProcess:
    def __init__(self, *, target, kwargs, daemon: bool, **_: object) -> None:  # type: ignore[no-untyped-def]
        self._thread = threading.Thread(target=target, kwargs=kwargs, daemon=daemon)

    def start(self) -> None:
        self._thread.start()

    def join(self, timeout: float | None = None) -> None:
        self._thread.join(timeout)

    def is_alive(self) -> bool:
        return self._thread.is_alive()

    def terminate(self) -> None:  # pragma: no cover - thread termination is noop
        return None


def _ensure_psf_reference() -> None:
    data_dir = Path(__file__).resolve().parents[1] / "fishtools" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    psf_path = data_dir / "PSF GL.tif"
    if not psf_path.exists():
        psf = np.exp(-((np.linspace(-1, 1, 31)[None, :, None] ** 2) + (np.linspace(-1, 1, 31)[None, None, :] ** 2)))
        psf = np.tile(psf, (101, 1, 1)).astype(np.float32)
        tifffile.imwrite(psf_path, psf)


@pytest.fixture
def minimal_workspace(tmp_path: Path) -> tuple[Path, str, str]:
    _ensure_psf_reference()

    workspace = tmp_path / "workspace"
    round_name = "r1"
    roi = "roiA"

    data_dir = workspace / f"{round_name}--{roi}"
    data_dir.mkdir(parents=True)

    image = np.random.randint(0, 1000, size=(1, 2048, 2048), dtype=np.uint16)
    metadata = {
        "waveform": json.dumps({"params": {"step": 0.6, "powers": ["000"]}}),
    }
    tifffile.imwrite(data_dir / f"{round_name}-0000.tif", image, metadata=metadata)

    basic_dir = workspace / "basic"
    basic_dir.mkdir(parents=True)
    basic_obj = DummyBasic(
        np.zeros((2048, 2048), dtype=np.float32),
        np.ones((2048, 2048), dtype=np.float32),
    )
    with open(basic_dir / "all-000.pkl", "wb") as fh:
        pickle.dump({"basic": basic_obj}, fh)

    scale_dir = workspace / "analysis" / "deconv32" / "deconv_scaling"
    scale_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(scale_dir / f"{round_name}.txt", np.array([[0.0], [1.0]], dtype=np.float32))

    return workspace, round_name, roi


@pytest.fixture(autouse=True)
def stub_cupy_cuda(monkeypatch):
    _install_cuda_stub()
    monkeypatch.setattr(cp, "cuda", cp.cuda, raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)


def _workspace_with_tile(base: Path, *, round_name: str = "r1", roi: str = "roiA") -> Path:
    workspace = base / "workspace"
    tile_dir = workspace / f"{round_name}--{roi}"
    tile_dir.mkdir(parents=True, exist_ok=True)
    payload = np.zeros((1, 8, 8), dtype=np.uint16)
    tifffile.imwrite(tile_dir / f"{round_name}-0000.tif", payload)
    return workspace


def test_parse_device_spec_auto(monkeypatch):
    monkeypatch.setattr("cupy.cuda.runtime.getDeviceCount", lambda: 3)
    assert parse_device_spec("auto") == [0, 1, 2]


def test_parse_device_spec_subset(monkeypatch):
    monkeypatch.setattr("cupy.cuda.runtime.getDeviceCount", lambda: 4)
    os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
    assert parse_device_spec("0,2") == [0, 2]


def test_run_skip_quantized_forces_float32(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    workspace = _workspace_with_tile(tmp_path)

    captured: dict[str, object] = {}

    def _fake_plan_execute(**kwargs):
        captured.update(mode=kwargs["mode"], load_scaling=kwargs["load_scaling"])
        return []

    monkeypatch.setattr("fishtools.preprocess.cli_deconv._plan_and_execute", _fake_plan_execute)

    run.callback(
        workspace,
        "r1",
        roi_name="*",
        ref_round=None,
        limit=None,
        backend="u16",
        histogram_bins=8192,
        overwrite=False,
        delete_origin=False,
        n_fids=2,
        basic_name="all",
        debug=False,
        devices="auto",
        stop_on_error=True,
        skip_quantized=True,
    )

    assert captured["mode"] is DeconvolutionOutputMode.F32
    assert captured["load_scaling"] is False


def test_run_float32_skips_scaling(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    workspace = _workspace_with_tile(tmp_path)

    captured: dict[str, object] = {}

    def _fake_plan_execute(**kwargs):
        captured.update(mode=kwargs["mode"], load_scaling=kwargs["load_scaling"])
        return []

    monkeypatch.setattr("fishtools.preprocess.cli_deconv._plan_and_execute", _fake_plan_execute)

    run.callback(
        workspace,
        "r1",
        roi_name="*",
        ref_round=None,
        limit=None,
        backend="float32",
        histogram_bins=8192,
        overwrite=False,
        delete_origin=False,
        n_fids=2,
        basic_name="all",
        debug=False,
        devices="auto",
        stop_on_error=True,
        skip_quantized=False,
    )

    assert captured["mode"] is DeconvolutionOutputMode.F32
    assert captured["load_scaling"] is False


def test_multi_run_skip_quantized_switches_backend(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    workspace = _workspace_with_tile(tmp_path)

    captured: dict[str, object] = {}

    def _fake_plan_execute(**kwargs):
        captured.update(mode=kwargs["mode"], load_scaling=kwargs["load_scaling"])
        return []

    monkeypatch.setattr("fishtools.preprocess.cli_deconv._plan_and_execute", _fake_plan_execute)

    multi_run(
        workspace,
        "r1",
        ref=None,
        limit=None,
        backend="u16",
        histogram_bins=1024,
        skip_quantized=True,
        overwrite=True,
        n_fids=0,
        basic_name="all",
        debug=False,
        devices=[0],
        stop_on_error=True,
    )

    assert captured["mode"] is DeconvolutionOutputMode.F32
    assert captured["load_scaling"] is False


def test_filter_pending_files_uint16(tmp_path: Path):
    out_dir = tmp_path / "analysis" / "deconv"
    existing_dir = out_dir / "r1--roiA"
    existing_dir.mkdir(parents=True)

    file_done = tmp_path / "r1--roiA" / "r1-0001.tif"
    file_done.parent.mkdir(parents=True)
    file_done.touch()
    (existing_dir / file_done.name).touch()

    file_pending = tmp_path / "r1--roiB" / "r1-0002.tif"
    file_pending.parent.mkdir(parents=True)
    file_pending.touch()

    config = ProcessorConfig(
        round_name="r1",
        basic_paths=(),
        output_dir=out_dir,
        n_fids=2,
        step=6,
        mode=DeconvolutionOutputMode.U16,
        histogram_bins=8192,
        m_glob=None,
        s_glob=None,
        debug=False,
    )
    backend = U16PrenormBackend(config)

    pending = filter_pending_files(
        [file_done, file_pending],
        out_dir=out_dir,
        overwrite=False,
        backend=backend,
    )

    assert pending == [file_pending]


def test_filter_pending_files_float32(tmp_path: Path):
    out_dir = tmp_path / "analysis" / "deconv"
    out_dir.mkdir(parents=True)
    out32 = out_dir.parent / "deconv32"

    file_done = tmp_path / "r2--roiA" / "r2-0001.tif"
    file_done.parent.mkdir(parents=True, exist_ok=True)
    file_done.touch()
    hist_dir = out32 / file_done.parent.name
    hist_dir.mkdir(parents=True, exist_ok=True)
    (hist_dir / file_done.name).touch()
    (hist_dir / file_done.name).with_suffix(".histogram.csv").touch()

    file_pending = tmp_path / "r2--roiA" / "r2-0002.tif"
    file_pending.touch()

    config = ProcessorConfig(
        round_name="r2",
        basic_paths=(),
        output_dir=out_dir,
        n_fids=2,
        step=6,
        mode=DeconvolutionOutputMode.F32,
        histogram_bins=8192,
        m_glob=None,
        s_glob=None,
        debug=False,
    )
    backend = Float32HistBackend(config)

    pending = filter_pending_files(
        [file_done, file_pending],
        out_dir=out_dir,
        overwrite=False,
        backend=backend,
    )

    assert pending == [file_pending]


def _write_basic_profiles(base: Path, *, channels: int, shape: tuple[int, int]) -> list[Path]:
    (base / "basic").mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for idx in range(channels):
        profile = DummyBasic(
            np.zeros(shape, dtype=np.float32),
            np.ones(shape, dtype=np.float32),
        )
        path = base / "basic" / f"all-{idx:03d}.pkl"
        with open(path, "wb") as fh:
            pickle.dump({"basic": profile}, fh)
        paths.append(path)
    return paths


def _patch_deconv_stubs(monkeypatch, channels: int, height: int, width: int) -> None:
    import cupy as cp

    monkeypatch.setattr(
        "fishtools.preprocess.deconv.core.load_projectors_cached",
        lambda step: (
            cp.ones((channels, height, width), dtype=cp.float32),
            cp.ones((channels, height, width), dtype=cp.float32),
        ),
    )
    monkeypatch.setattr(
        "fishtools.preprocess.deconv.core.deconvolve_lucyrichardson_guo",
        lambda payload, projectors, iters: payload,
    )


def test_deconvolution_processor_dynamic_geometry(tmp_path: Path, monkeypatch):
    height, width = 512, 768
    channels = 2
    roi = "roiA"
    round_name = "rDyn"
    tile_dir = tmp_path / f"{round_name}--{roi}"
    tile_dir.mkdir(parents=True)
    tile_path = tile_dir / f"{round_name}-0000.tif"
    payload = (np.arange(channels * 2 * height * width, dtype=np.uint16)).reshape(channels * 2, height, width)
    metadata = {"waveform": json.dumps({"params": {"step": 0.6}})}
    tifffile.imwrite(tile_path, payload, metadata=metadata)

    basic_paths = _write_basic_profiles(tmp_path, channels=channels, shape=(height, width))
    _patch_deconv_stubs(monkeypatch, channels, height, width)

    output_dir = tmp_path / "analysis" / "deconv"
    config = ProcessorConfig(
        round_name=round_name,
        basic_paths=basic_paths,
        output_dir=output_dir,
        n_fids=0,
        step=6,
        mode=DeconvolutionOutputMode.F32,
        histogram_bins=16,
        m_glob=None,
        s_glob=None,
        debug=False,
    )

    processor = make_processor_factory(config)(0)
    processor.setup()
    try:
        processor.process(tile_path)
    finally:
        processor.teardown()

    float32_path = output_dir.parent / "deconv32" / tile_dir.name / tile_path.name
    assert float32_path.exists()
    with tifffile.TiffFile(float32_path) as tif:
        arr = tif.asarray()
    assert arr.shape == (channels * 2, height, width)


def test_float32_backend_appends_fiducials(tmp_path: Path) -> None:
    round_name = "rFid"
    roi = "roi"
    tile_path = tmp_path / f"{round_name}--{roi}" / f"{round_name}-0000.tif"
    tile_path.parent.mkdir(parents=True)

    height = width = 8
    payload_planes = 3
    n_fids = 2

    config = ProcessorConfig(
        round_name=round_name,
        basic_paths=(),
        output_dir=tmp_path / "analysis" / "deconv",
        n_fids=n_fids,
        step=1,
        mode=DeconvolutionOutputMode.F32,
        histogram_bins=32,
        m_glob=None,
        s_glob=None,
        debug=False,
    )

    backend = Float32HistBackend(config)
    backend.setup(types.SimpleNamespace(config=config))

    f32_payload = np.arange(payload_planes * height * width, dtype=np.float32).reshape(
        payload_planes, height, width
    )
    fid = np.arange(n_fids * height * width, dtype=np.uint16).reshape(n_fids, height, width)

    artifacts = OutputArtifacts(
        f32_payload=f32_payload,
        hist_payload={
            "C": 1,
            "bins": 1,
            "counts": [np.array([0])],
            "edges": [np.array([0.0, 1.0])],
        },
    )

    backend.write(
        tile_path,
        fid,
        metadata_out={"foo": "bar"},
        artifacts=artifacts,
        out_dir=config.output_dir,
    )

    out32 = config.output_dir.parent / "deconv32" / tile_path.parent.name / tile_path.name
    assert out32.exists()

    with tifffile.TiffFile(out32) as tif:
        stack = tif.asarray()
        meta = tif.imagej_metadata or tif.shaped_metadata[0]  # type: ignore[index]

    assert stack.shape == (payload_planes + n_fids, height, width)
    np.testing.assert_array_equal(stack[:payload_planes], f32_payload)
    np.testing.assert_array_equal(stack[-n_fids:], fid.astype(np.float32))
    assert meta["fid_planes"] == n_fids
    assert meta["fid_source_dtype"] == "uint16"


def test_deconvolution_histogram_deterministic(tmp_path: Path, monkeypatch):
    height, width = 256, 512
    channels = 2
    roi = "roiB"
    round_name = "rHist"
    tile_dir = tmp_path / f"{round_name}--{roi}"
    tile_dir.mkdir(parents=True)
    tile_path = tile_dir / f"{round_name}-0000.tif"
    payload = np.ones((channels * 4, height, width), dtype=np.uint16)
    metadata = {"waveform": json.dumps({"params": {"step": 0.6}})}
    tifffile.imwrite(tile_path, payload, metadata=metadata)

    basic_paths = _write_basic_profiles(tmp_path, channels=channels, shape=(height, width))
    _patch_deconv_stubs(monkeypatch, channels, height, width)

    def run_once(root: Path) -> str:
        output_dir = root / "analysis" / "deconv"
        config = ProcessorConfig(
            round_name=round_name,
            basic_paths=basic_paths,
            output_dir=output_dir,
            n_fids=0,
            step=6,
            mode=DeconvolutionOutputMode.F32,
            histogram_bins=8,
            m_glob=None,
            s_glob=None,
            debug=False,
        )

        processor = make_processor_factory(config)(0)
        processor.setup()
        try:
            processor.process(tile_path)
        finally:
            processor.teardown()

        hist_path = output_dir.parent / "deconv32" / tile_dir.name / tile_path.with_suffix(".histogram.csv").name
        return hist_path.read_text()

    hist_a = run_once(tmp_path / "runA")
    hist_b = run_once(tmp_path / "runB")
    assert hist_a == hist_b


def test_load_basics_mismatched_shapes(tmp_path: Path):
    roi = "roiC"
    round_name = "rShape"
    basic_dir = tmp_path / "basic"
    basic_dir.mkdir(parents=True, exist_ok=True)
    good = DummyBasic(np.zeros((32, 32), dtype=np.float32), np.ones((32, 32), dtype=np.float32))
    bad = DummyBasic(np.zeros((16, 16), dtype=np.float32), np.ones((16, 16), dtype=np.float32))
    good_path = basic_dir / "all-000.pkl"
    bad_path = basic_dir / "all-001.pkl"
    with open(good_path, "wb") as fh:
        pickle.dump({"basic": good}, fh)
    with open(bad_path, "wb") as fh:
        pickle.dump({"basic": bad}, fh)

    config = ProcessorConfig(
        round_name=round_name,
        basic_paths=[good_path, bad_path],
        output_dir=tmp_path / "analysis" / "deconv",
        n_fids=0,
        step=6,
        mode=DeconvolutionOutputMode.F32,
        histogram_bins=16,
        m_glob=None,
        s_glob=None,
        debug=False,
    )

    processor = make_processor_factory(config)(0)
    with pytest.raises(ValueError):
        processor.setup()


def test_run_multi_gpu_stop_on_error_cancels_remaining(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    file_ok = tmp_path / "tile_ok.tif"
    file_fail = tmp_path / "tile_fail.tif"
    file_ok.touch()
    file_fail.touch()

    FAILED_PROCESS_LOG.clear()

    dummy_ctx = types.SimpleNamespace(
        Queue=lambda maxsize=0: _DummyQueue(maxsize),
        Process=lambda target, kwargs, daemon, **extra: _DummyProcess(
            target=target, kwargs=kwargs, daemon=daemon, **extra
        ),
    )
    monkeypatch.setattr("fishtools.preprocess.cli_deconv._MP_CTX", dummy_ctx)
    monkeypatch.setattr("fishtools.preprocess.cli_deconv._configure_logging", lambda *a, **k: None)
    monkeypatch.setattr("fishtools.preprocess.cli_deconv.signal.signal", lambda *a, **k: None)
    monkeypatch.setattr(
        "fishtools.preprocess.cli_deconv.cp.cuda",
        types.SimpleNamespace(
            runtime=types.SimpleNamespace(getDeviceCount=lambda: 1, CUDARuntimeError=RuntimeError),
            Device=lambda _: types.SimpleNamespace(use=lambda: None),
        ),
        raising=False,
    )

    failures = run_multi_gpu(
        [file_fail, file_ok],
        devices=[0],
        processor_factory=_failing_processor_factory,
        queue_depth=1,
        stop_on_error=True,
    )

    assert [msg.path for msg in failures] == [file_fail]
    assert FAILED_PROCESS_LOG == [file_fail]


def test_load_global_scaling_missing(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_global_scaling(tmp_path, "missing")


def test_multi_run_end_to_end(minimal_workspace: tuple[Path, str, str]):
    workspace, round_name, roi = minimal_workspace

    multi_run(
        workspace,
        round_name,
        ref=None,
        limit=1,
        overwrite=True,
        n_fids=0,
        basic_name="all",
        debug=False,
        devices=[0],
        stop_on_error=True,
    )

    u16_path = workspace / "analysis" / "deconv" / f"{round_name}--{roi}" / f"{round_name}-0000.tif"
    f32_path = workspace / "analysis" / "deconv32" / f"{round_name}--{roi}" / f"{round_name}-0000.tif"
    assert u16_path.exists() or f32_path.exists()
