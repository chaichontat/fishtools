"""
Deconvolution processing contracts and output backends (cleaned).

Notable changes:
- CLI can choose OutputBackend directly by passing a backend_factory.
- OutputBackend now exposes expected_targets() for idempotency checks.
"""

from __future__ import annotations

import csv
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

import cupy as cp
import numpy as np
import tifffile

from fishtools.io.workspace import safe_imwrite
from fishtools.preprocess.config import DeconvolutionOutputMode

from .basic_utils import load_basic_profiles
from .core import PRENORMALIZED, deconvolve_lucyrichardson_guo, projectors
from .hist import sample_histograms
from .normalize import quantize_global
from .profiling_utils import GpuTimer

timer = GpuTimer()


LEGACY_PERCENTILES = (0.1, 99.999)


@dataclass(slots=True)
class ProcessorConfig:
    round_name: str
    basic_paths: Sequence[Path]
    output_dir: Path
    n_fids: int
    step: int
    mode: DeconvolutionOutputMode
    histogram_bins: int
    m_glob: np.ndarray | None
    s_glob: np.ndarray | None
    debug: bool


# ------------------- Output backends -------------------


@dataclass(slots=True)
class OutputArtifacts:
    """Host-side artifacts produced after GPU compute."""

    towrite_u16: np.ndarray | None = None
    f32_payload: np.ndarray | None = None
    hist_payload: dict[str, Any] | None = None


class OutputBackend(Protocol):
    """Strategy: per-output-mode postprocess + write."""

    # NEW: allow CLI to compute idempotency targets without GPU setup
    def expected_targets(self, out_dir: Path, src: Path) -> Sequence[Path]:  # pragma: no cover - simple logic
        ...

    def setup(self, processor: "DeconvolutionTileProcessor") -> None:  # pragma: no cover - thin interface
        ...

    def postprocess(
        self,
        res: cp.ndarray,
        path: Path,
        hw: tuple[int, int],
    ) -> tuple[OutputArtifacts, dict[str, Any], dict[str, float]]:
        """
        Convert GPU result 'res' into host artifacts and metadata patch.

        Returns:
            artifacts: OutputArtifacts
            meta_patch: dict with extra metadata fields to merge into TIFF metadata
            timings_extra: dict with 'quant' and/or 'post' timing contributions
        """

    def write(
        self,
        path: Path,
        fid_np: np.ndarray,
        metadata_out: dict[str, Any],
        artifacts: OutputArtifacts,
        out_dir: Path,
    ) -> float:
        """Persist artifacts for one tile; returns write seconds."""


# The CLI passes one of these in.
OutputBackendFactory = Callable[[ProcessorConfig], OutputBackend]


def legacy_per_tile_quantize(
    res: cp.ndarray,
    hw: tuple[int, int],
    *,
    percentile_low: float = LEGACY_PERCENTILES[0],
    percentile_high: float = LEGACY_PERCENTILES[1],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Legacy per-tile percentile-based quantization used by cli_deconv_old."""

    height, width = hw
    if height <= 0 or width <= 0:
        raise ValueError(f"Invalid tile dimensions hw={hw}")

    payload = cp.asarray(res, dtype=cp.float32)
    if payload.ndim == 3:
        payload = payload[:, cp.newaxis, ...]
    elif payload.ndim != 4:
        raise ValueError(
            "Expected deconvolution output with 3D or 4D shape (z[, c], y, x) for legacy quantization."
        )

    if payload.shape[-2:] != (height, width):
        raise ValueError(f"Legacy quantization mismatch: payload spatial dims {payload.shape[-2:]} != {hw}.")

    mins = cp.percentile(payload, percentile_low, axis=(0, 2, 3))
    maxs = cp.percentile(payload, percentile_high, axis=(0, 2, 3))

    mins = mins.astype(cp.float32, copy=False).reshape(1, -1, 1, 1)
    maxs = maxs.astype(cp.float32, copy=False).reshape(1, -1, 1, 1)

    scale = cp.float32(65534.0) / cp.maximum(maxs - mins, cp.float32(1e-20))
    scale = scale.reshape(1, -1, 1, 1)

    scaled = cp.clip((payload - mins) * scale, 0.0, 65534.0)
    towrite = scaled.astype(cp.uint16).reshape(-1, height, width)

    mins_host = cp.asnumpy(mins.reshape(-1)).astype(np.float32, copy=False)
    scale_host = cp.asnumpy(scale.reshape(-1)).astype(np.float32, copy=False)
    towrite_host = cp.asnumpy(towrite)

    return towrite_host, mins_host, scale_host


class Float32HistBackend:
    """Emit float32 deconvolved planes (deconv32) and always compute histograms."""

    def __init__(self, config: ProcessorConfig):
        self.config = config
        self._out32: Path | None = None

    # --- idempotency contract for CLI ---
    def expected_targets(self, out_dir: Path, src: Path) -> Sequence[Path]:
        root = out_dir.parent / "deconv32"
        base = root / src.parent.name / src.name
        return [base, base.with_suffix(".histogram.csv")]

    def setup(self, processor: "DeconvolutionTileProcessor") -> None:
        self._out32 = processor.config.output_dir.parent / "deconv32"
        self._out32.mkdir(parents=True, exist_ok=True)
        if self.config.mode is not DeconvolutionOutputMode.F32:
            raise ValueError("Float32HistBackend requires mode=float32")

    def postprocess(
        self,
        res: cp.ndarray,
        path: Path,
        hw: tuple[int, int],
    ) -> tuple[OutputArtifacts, dict[str, Any], dict[str, float]]:
        H, W = hw
        t0 = time.perf_counter()

        reshaped = res.reshape(-1, H, W)
        f32_payload = cp.asnumpy(reshaped).astype(np.float32, copy=False)

        digest = hashlib.blake2b(
            f"{self.config.round_name}:{path}".encode("utf-8"),
            digest_size=8,
        ).digest()
        seed = int.from_bytes(digest, "little")
        hist_payload = sample_histograms(
            res,
            bins=int(self.config.histogram_bins),
            strides=(4, 2, 2),
            seed=seed,
        )

        timings_extra = {"quant": 0.0, "post": time.perf_counter() - t0}
        return (
            OutputArtifacts(f32_payload=f32_payload, hist_payload=hist_payload),
            {},
            timings_extra,
        )

    def write(
        self,
        path: Path,
        fid_np: np.ndarray,  # noqa: ARG002 - fid not emitted in this backend
        metadata_out: dict[str, Any],
        artifacts: OutputArtifacts,
        out_dir: Path,  # noqa: ARG002 - uses deconv32 path
    ) -> float:
        assert self._out32 is not None
        t0 = time.perf_counter()

        sub32 = self._out32 / path.parent.name
        sub32.mkdir(parents=True, exist_ok=True)

        if artifacts.f32_payload is None or artifacts.hist_payload is None:
            raise RuntimeError("Float32HistBackend expected float32 payload and histogram data.")

        payload = artifacts.f32_payload
        if fid_np.size:
            fid_float = fid_np.astype(np.float32, copy=False)
            payload = np.concatenate([payload, fid_float], axis=0)

        metadata = {
            "dtype": "float32",
            "fid_planes": int(fid_np.shape[0]),
            "fid_source_dtype": str(fid_np.dtype),
            **metadata_out,
        }

        safe_imwrite(
            sub32 / path.name,
            payload,
            dtype=np.float32,
            metadata=metadata,
            compression="zlib",
            compressionargs={"level": 6},
        )

        hist_path = (sub32 / path.name).with_suffix(".histogram.csv")
        with open(hist_path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["channel", "bin_left", "bin_right", "count"])
            C = artifacts.hist_payload["C"]
            bins = artifacts.hist_payload["bins"]
            counts_list = artifacts.hist_payload["counts"]
            edges_list = artifacts.hist_payload["edges"]
            for c in range(C):
                edges = edges_list[c]
                counts = counts_list[c]
                for b in range(bins):
                    writer.writerow([
                        c,
                        f"{edges[b]:.8g}",
                        f"{edges[b + 1]:.8g}",
                        int(counts[b]),
                    ])

        return time.perf_counter() - t0


class U16PrenormBackend:
    """Emit u16 payload + fid, and augment metadata with prenormalization fields."""

    def __init__(self, config: ProcessorConfig):
        self.config = config

    # --- idempotency contract for CLI ---
    def expected_targets(self, out_dir: Path, src: Path) -> Sequence[Path]:
        return [out_dir / src.parent.name / src.name]

    def setup(self, processor: "DeconvolutionTileProcessor") -> None:  # pragma: no cover - trivial
        return None

    def postprocess(
        self,
        res: cp.ndarray,
        path: Path,  # noqa: ARG002
        hw: tuple[int, int],  # noqa: ARG002
    ) -> tuple[OutputArtifacts, dict[str, Any], dict[str, float]]:
        if self.config.m_glob is None or self.config.s_glob is None:
            raise ValueError("Global scaling (m_glob, s_glob) is required for u16 prenormalized path.")
        t0 = time.perf_counter()
        towrite_u16 = quantize_global(
            res,
            self.config.m_glob,
            self.config.s_glob,
            i_max=65535,
            return_stats=False,
            as_numpy=True,
        )
        timings_extra = {"quant": time.perf_counter() - t0, "post": 0.0}
        meta_patch = {
            "deconv_min": list(map(float, self.config.m_glob.flatten())),
            "deconv_scale": list(map(float, self.config.s_glob.flatten())),
            PRENORMALIZED: True,
        }
        return OutputArtifacts(towrite_u16=towrite_u16), meta_patch, timings_extra

    def write(
        self,
        path: Path,
        fid_np: np.ndarray,
        metadata_out: dict[str, Any],
        artifacts: OutputArtifacts,
        out_dir: Path,
    ) -> float:
        if artifacts.towrite_u16 is None:
            raise RuntimeError("U16PrenormBackend: expected u16 payload, found None.")
        t0 = time.perf_counter()
        out_dir_roi = out_dir / path.parent.name
        out_dir_roi.mkdir(parents=True, exist_ok=True)

        deliverable = np.concatenate(
            [artifacts.towrite_u16, fid_np.astype(np.uint16, copy=False)],
            axis=0,
        )
        safe_imwrite(
            out_dir_roi / path.name,
            deliverable,
            compression=22610,
            compressionargs={"level": 0.75},
            metadata=metadata_out,
        )
        return time.perf_counter() - t0


class LegacyPerTileU16Backend:
    """Per-tile percentile quantization backend emulating cli_deconv_old outputs."""

    def __init__(self, config: ProcessorConfig):
        self.config = config
        self._out_dir: Path | None = None

    def expected_targets(self, out_dir: Path, src: Path) -> Sequence[Path]:
        return [out_dir / src.parent.name / src.name]

    def setup(self, processor: "DeconvolutionTileProcessor") -> None:
        self._out_dir = processor.config.output_dir
        self._out_dir.mkdir(parents=True, exist_ok=True)

    def postprocess(
        self,
        res: cp.ndarray,
        path: Path,  # noqa: ARG002
        hw: tuple[int, int],
    ) -> tuple[OutputArtifacts, dict[str, Any], dict[str, float]]:
        t0 = time.perf_counter()
        towrite_u16, mins, scales = legacy_per_tile_quantize(res, hw)
        quant_time = time.perf_counter() - t0
        meta_patch = {
            "deconv_min": [float(val) for val in mins],
            "deconv_scale": [float(val) for val in scales],
        }
        timings = {"quant": quant_time, "post": 0.0}
        return OutputArtifacts(towrite_u16=towrite_u16), meta_patch, timings

    def write(
        self,
        path: Path,
        fid_np: np.ndarray,
        metadata_out: dict[str, Any],
        artifacts: OutputArtifacts,
        out_dir: Path,
    ) -> float:
        if artifacts.towrite_u16 is None:
            raise RuntimeError("LegacyPerTileU16Backend: expected u16 payload, found None.")
        if self._out_dir is None:
            raise RuntimeError("LegacyPerTileU16Backend: setup was not called before write.")

        t0 = time.perf_counter()
        roi_dir = self._out_dir / path.parent.name
        roi_dir.mkdir(parents=True, exist_ok=True)

        deliverable = np.concatenate(
            [artifacts.towrite_u16, fid_np.astype(np.uint16, copy=False)],
            axis=0,
        )
        safe_imwrite(
            roi_dir / path.name,
            deliverable,
            compression=22610,
            compressionargs={"level": 0.75},
            metadata=metadata_out,
        )

        sidecar = (roi_dir / path.name).with_suffix(".deconv.json")
        sidecar.write_text(json.dumps(metadata_out, indent=2))

        return time.perf_counter() - t0


class TileProcessor(Protocol):
    """Per-worker processing contract."""

    def setup(self) -> None:  # pragma: no cover
        ...

    def teardown(self) -> None:  # pragma: no cover
        ...

    def read_tile(
        self, path: Path
    ) -> tuple[Path, np.ndarray, np.ndarray, dict[str, Any], tuple[int, int], int]: ...

    def compute_tile(
        self,
        payload_np: np.ndarray,
        path: Path,
        hw: tuple[int, int],
        metadata_in: dict[str, Any],
    ) -> tuple[OutputArtifacts, dict[str, Any], dict[str, float]]: ...

    def write_tile(
        self,
        path: Path,
        fid_np: np.ndarray,
        metadata_out: dict[str, Any],
        artifacts: OutputArtifacts,
    ) -> float: ...

    def process(self, path: Path) -> float:  # pragma: no cover - optional fast-path
        ...


ProcessorFactory = Callable[[int], TileProcessor]


class DeconvolutionTileProcessor:
    """Concrete processor with split read/compute/write for pipelining."""

    def __init__(
        self,
        config: ProcessorConfig,
        *,
        device_id: int | None = None,
        backend_factory: OutputBackendFactory | None = None,
    ):
        self.config = config
        self._darkfield: cp.ndarray | None = None
        self._flatfield: cp.ndarray | None = None
        self._projectors: tuple[cp.ndarray, cp.ndarray] | None = None
        self._inv_flatfield: cp.ndarray | None = None
        self._basic_kernel: cp.ElementwiseKernel | None = None
        self._device_id = device_id
        self._backend_factory = backend_factory
        self._backend: OutputBackend | None = None

    def set_device(self, device_id: int) -> None:
        self._device_id = device_id

    def _load_basics(self) -> None:
        self._darkfield, self._flatfield = load_basic_profiles(self.config.basic_paths)

    def setup(self) -> None:
        if self._device_id is not None and hasattr(cp.cuda, "Device"):
            cp.cuda.Device(self._device_id).use()

        cuda = getattr(cp, "cuda", None)

        if not cuda:
            raise RuntimeError("CuPy CUDA support not available; cannot proceed.")

        if hasattr(cuda, "MemoryPool") and hasattr(cuda, "set_allocator"):
            pool = cuda.MemoryPool()
            cuda.set_allocator(pool.malloc)
        if hasattr(cuda, "PinnedMemoryPool") and hasattr(cuda, "set_pinned_memory_allocator"):
            pinned_pool = cuda.PinnedMemoryPool()
            cuda.set_pinned_memory_allocator(pinned_pool.malloc)

        self._load_basics()
        # Ensure BaSiC profiles are on-device and float32
        assert self._darkfield is not None, "BaSiC darkfield missing post-setup."
        assert self._flatfield is not None, "BaSiC flatfield missing post-setup."
        self._darkfield = cp.asarray(self._darkfield, dtype=cp.float32)
        self._flatfield = cp.asarray(self._flatfield, dtype=cp.float32)
        # Precompute reciprocal once; avoid divides in the hot path
        self._inv_flatfield = 1.0 / self._flatfield
        # Single-pass BaSiC + clamp kernel
        self._basic_kernel = cp.ElementwiseKernel(
            "float32 x, float32 df, float32 inv_ff",
            "float32 y",
            "float t = (x - df) * inv_ff; y = t > 0.0f ? t : 0.0f;",
            name="basic_correct_clip",
        )

        self._projectors = projectors(self.config.step)
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        # Backend selection: prefer explicit factory; else fallback to legacy behavior
        if self._backend_factory is not None:
            self._backend = self._backend_factory(self.config)
        else:
            if self.config.mode is DeconvolutionOutputMode.F32:
                self._backend = Float32HistBackend(self.config)
            else:
                self._backend = U16PrenormBackend(self.config)
        self._backend.setup(self)

    def teardown(self) -> None:
        self._darkfield = None
        self._flatfield = None
        self._projectors = None
        self._backend = None
        self._inv_flatfield = None
        self._basic_kernel = None

    def read_tile(
        self, path: Path
    ) -> tuple[Path, np.ndarray, np.ndarray, dict[str, Any], tuple[int, int], int]:
        with tifffile.TiffFile(path) as tif:
            try:
                metadata = tif.shaped_metadata[0]  # type: ignore[index]
            except (TypeError, IndexError):
                metadata = tif.imagej_metadata or {}
            image = tif.asarray()

        if not isinstance(metadata, dict):
            metadata = dict(metadata) if metadata is not None else {}

        n_fids = self.config.n_fids
        if n_fids:
            fid = np.atleast_3d(image[-n_fids:])
            payload = image[:-n_fids]
        else:
            fid = np.zeros((0, *image.shape[-2:]), dtype=image.dtype)
            payload = image

        channels = len(self.config.basic_paths)
        if payload.ndim < 3:
            raise ValueError(
                f"{path} payload has insufficient dimensions {payload.shape}; expected at least 3-D."
            )

        height, width = payload.shape[-2:]
        plane_count = int(np.prod(payload.shape[:-2]))
        if plane_count % channels != 0:
            raise ValueError(
                (
                    f"{path} has {plane_count} payload planes not divisible by {channels} BaSiC channel(s). "
                    "This typically indicates a channel inference mismatch. Verify TIFF metadata (waveform→params→powers) "
                    "and fallback logic, or pass the correct --basic-name so that the number of BaSiC profiles matches the data."
                )
            )

        try:
            # Keep native dtype; we cast on GPU to cut H2D traffic (esp. u16 sources)
            nofid = payload.reshape(-1, channels, height, width)
        except ValueError as exc:
            raise ValueError(
                f"{path} unexpected shape {payload.shape} for {channels} channel(s) after removing {n_fids} fid slice(s)."
            ) from exc

        return path, nofid, fid, metadata, (height, width), channels

    def compute_tile(
        self,
        payload_np: np.ndarray,
        path: Path,
        hw: tuple[int, int],
        metadata_in: dict[str, Any],
    ) -> tuple[OutputArtifacts, dict[str, Any], dict[str, float]]:
        assert self._backend is not None, "Processor not set up."
        if self._device_id is not None and hasattr(cp.cuda, "Device"):
            cp.cuda.Device(self._device_id).use()

        timings: dict[str, float] = {}

        t_basic0 = time.perf_counter()
        assert (
            self._darkfield is not None and self._inv_flatfield is not None and self._basic_kernel is not None
        )

        # GPU-side cast saves bandwidth; ensure contiguous float32 on device
        x = cp.asarray(payload_np, dtype=cp.float32)
        self._basic_kernel(x, self._darkfield, self._inv_flatfield, x)

        timings["basic"] = time.perf_counter() - t_basic0

        t_dec0 = time.perf_counter()

        res = deconvolve_lucyrichardson_guo(x, self._projectors, iters=1)  # type: ignore[arg-type]

        timings["deconv"] = time.perf_counter() - t_dec0

        artifacts, meta_patch, extra_timings = self._backend.postprocess(res, path, hw)
        timings["quant"] = extra_timings.get("quant", 0.0)
        timings["post"] = extra_timings.get("post", 0.0)
        timings |= timer.to_dict()
        timer.reset()

        del res, x

        metadata_out = metadata_in | meta_patch
        return artifacts, metadata_out, timings

    def write_tile(
        self,
        path: Path,
        fid_np: np.ndarray,
        metadata_out: dict[str, Any],
        artifacts: OutputArtifacts,
    ) -> float:
        assert self._backend is not None, "Processor not set up."
        return self._backend.write(path, fid_np, metadata_out, artifacts, self.config.output_dir)

    def process(self, path: Path) -> float:
        path_r, payload, fid, metadata, hw, _ = self.read_tile(path)
        artifacts, metadata_out, timings = self.compute_tile(payload, path_r, hw, metadata)
        t_write = self.write_tile(path_r, fid, metadata_out, artifacts)
        return (
            timings.get("basic", 0.0)
            + timings.get("deconv", 0.0)
            + timings.get("quant", 0.0)
            + timings.get("post", 0.0)
            + t_write
        )


@dataclass(frozen=True, slots=True)
class _ProcessorFactory:
    config: ProcessorConfig
    backend_factory: OutputBackendFactory | None = None

    def __call__(self, device_id: int) -> TileProcessor:
        return DeconvolutionTileProcessor(
            self.config,
            device_id=device_id,
            backend_factory=self.backend_factory,
        )


def make_processor_factory(
    config: ProcessorConfig,
    backend_factory: OutputBackendFactory | None = None,
) -> ProcessorFactory:
    """Construct a processor factory with defensively copied configuration."""

    safe_config = ProcessorConfig(
        round_name=config.round_name,
        basic_paths=tuple(Path(p) for p in config.basic_paths),
        output_dir=config.output_dir,
        n_fids=config.n_fids,
        step=config.step,
        mode=config.mode,
        histogram_bins=config.histogram_bins,
        m_glob=None if config.m_glob is None else np.asarray(config.m_glob, dtype=np.float32),
        s_glob=None if config.s_glob is None else np.asarray(config.s_glob, dtype=np.float32),
        debug=config.debug,
    )
    return _ProcessorFactory(safe_config, backend_factory)
