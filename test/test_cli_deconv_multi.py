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

from fishtools.preprocess.cli_deconv_multi import (
    DeconvolutionTileProcessor,
    ProcessorConfig,
    filter_pending_files,
    load_global_scaling,
    make_processor_factory,
    multi_run,
    _DEFAULT_OUTPUT_MODE,
    parse_device_spec,
    run_multi_gpu,
)
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
    import cupy as cp

    runtime = types.SimpleNamespace(
        getDeviceCount=lambda: 1,
        CUDARuntimeError=RuntimeError,
    )
    cuda = types.SimpleNamespace(runtime=runtime)
    monkeypatch.setattr(cp, "cuda", cuda, raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)


def test_parse_device_spec_auto(monkeypatch):
    monkeypatch.setattr("cupy.cuda.runtime.getDeviceCount", lambda: 3)
    assert parse_device_spec("auto") == [0, 1, 2]


def test_parse_device_spec_subset(monkeypatch):
    monkeypatch.setattr("cupy.cuda.runtime.getDeviceCount", lambda: 4)
    os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
    assert parse_device_spec("0,2") == [0, 2]


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

    pending = filter_pending_files(
        [file_done, file_pending],
        out_dir=out_dir,
        overwrite=False,
        mode=DeconvolutionOutputMode.U16,
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

    pending = filter_pending_files(
        [file_done, file_pending],
        out_dir=out_dir,
        overwrite=False,
        mode=DeconvolutionOutputMode.F32,
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
    monkeypatch.setattr("fishtools.preprocess.cli_deconv_multi._MP_CTX", dummy_ctx)
    monkeypatch.setattr("fishtools.preprocess.cli_deconv_multi._configure_logging", lambda *a, **k: None)
    monkeypatch.setattr("fishtools.preprocess.cli_deconv_multi.signal.signal", lambda *a, **k: None)
    monkeypatch.setattr(
        "fishtools.preprocess.cli_deconv_multi.cp.cuda",
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


def test_run_cli_progressive_scoping_all_rois(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    workspace = tmp_path / "workspace"
    for roi in ("roiA", "roiB"):
        tile_dir = workspace / f"r1--{roi}"
        tile_dir.mkdir(parents=True)
        (tile_dir / "r1-0000.tif").touch()

    calls: list[tuple[str, list[Path]]] = []

    def fake_run_round_tiles(**kwargs):  # type: ignore[no-untyped-def]
        label = kwargs.get("label")
        files = kwargs.get("files", [])
        calls.append((label, list(files)))
        return []

    monkeypatch.setattr(
        "fishtools.preprocess.cli_deconv_multi._run_round_tiles",
        fake_run_round_tiles,
    )
    monkeypatch.setattr(
        "fishtools.preprocess.cli_deconv_multi.parse_device_spec",
        lambda spec: [0],
    )
    monkeypatch.setattr(
        "fishtools.preprocess.cli_deconv_multi._configure_logging",
        lambda *args, **kwargs: None,
    )

    from fishtools.preprocess.cli_deconv_multi import run as run_cmd

    run_cmd.callback(  # type: ignore[call-arg]
        workspace,
        "*",
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
    )

    labels = {label for label, _ in calls}
    assert labels == {"r1/roiA", "r1/roiB"}
    for label, files in calls:
        assert label in labels
        assert all(path.name.startswith("r1-") for path in files)


def test_run_cli_delete_origin_respects_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    workspace = tmp_path / "workspace"
    tile_dir = workspace / "r1--roiA"
    tile_dir.mkdir(parents=True)
    files = [tile_dir / f"r1-{idx:04d}.tif" for idx in range(2)]
    for file in files:
        file.touch()

    processed: list[list[Path]] = []

    def fake_run_round_tiles(**kwargs):  # type: ignore[no-untyped-def]
        processed.append(list(kwargs.get("files", [])))
        return []

    deleted: list[list[Path]] = []

    def fake_delete(orig: list[Path], out_dir: Path) -> None:
        deleted.append(list(orig))

    monkeypatch.setattr(
        "fishtools.preprocess.cli_deconv_multi._run_round_tiles",
        fake_run_round_tiles,
    )
    monkeypatch.setattr(
        "fishtools.preprocess.cli_deconv_multi._configure_logging",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "fishtools.preprocess.cli_deconv_multi.parse_device_spec",
        lambda spec: [0],
    )
    monkeypatch.setattr(
        "fishtools.preprocess.cli_deconv_multi.safe_delete_origin_dirs",
        fake_delete,
        raising=False,
    )

    from fishtools.preprocess.cli_deconv_multi import _run_cli_scope
    from fishtools.io.workspace import Workspace

    _run_cli_scope(
        path=workspace,
        roi="roiA",
        rounds=("r1",),
        ref_round=None,
        limit=1,
        mode=DeconvolutionOutputMode.U16,
        histogram_bins=8192,
        overwrite=False,
        n_fids=2,
        basic_name="all",
        debug=False,
        devices=[0],
        stop_on_error=True,
        delete_origin=True,
        workspace=Workspace(workspace),
    )

    assert len(processed) == 1
    assert [f.name for f in processed[0]] == ["r1-0000.tif"]
    assert deleted and [f.name for f in deleted[0]] == ["r1-0000.tif"]
