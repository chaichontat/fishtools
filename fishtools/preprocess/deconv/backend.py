"""Deconvolution processing contracts and output backends.

This module defines the two core extension points used by the multi‑GPU
deconvolution pipeline:

- `TileProcessor` (protocol): a per‑GPU worker contract that owns the tile
  lifecycle: read → compute → postprocess → write. A concrete implementation
  (`DeconvolutionTileProcessor`) performs BaSiC correction and LR‑Guo
  deconvolution on the GPU, then delegates persistence to an output backend.

- `OutputBackend` (protocol): a strategy for turning GPU results into durable
  artifacts (files/metadata). Backends decide formatting (float32 payloads,
  histograms, prenormalized uint16, etc.) and perform the actual writes.

High‑level dataflow (per tile)
--------------------------------

    [Reader]
       │  read_tile(path)
       ▼
    payload_np (float32, shape = N·C×H×W), fid_np (last n_fids planes)
       │
       ▼
    BaSiC correction (darkfield/flatfield)              timings["basic"]
       │
       ▼
    LR‑Guo deconvolution (GPU)                          timings["deconv"]
       │               res: cp.ndarray
       ▼
    OutputBackend.postprocess(res, path, hw) →          timings["quant"], timings["post"]
       │        ┌────────────────────────────┬────────────────────────────┐
       │        │ Float32HistBackend         │ U16PrenormBackend          │
       │        │  • emit float32 planes     │  • quantize to uint16      │
       │        │  • optional CSV histograms │  • append fiducials        │
       │        └────────────────────────────┴────────────────────────────┘
       ▼
    artifacts, meta_patch
       │
       ▼
    OutputBackend.write(path, fid_np, metadata_out, artifacts) → file(s)

Concurrency schematic (inside each worker process)
--------------------------------------------------

    [reader thread]  →  q_in  →  [compute thread (GPU)]  →  q_out  →  [writer thread]

Workers are spawned and scheduled by `run_multi_gpu` (see
`fishtools/preprocess/deconv/worker.py`). Each worker receives a dedicated
`TileProcessor` bound to a specific `device_id`.

Key concepts
------------

- ProcessorConfig: immutable configuration shared by the processor and its
  backend (round name, BaSiC paths, output directory, number of fiducial
  slices, PSF step, histogram settings, global scaling for prenorm, debug).
- OutputArtifacts: host‑side containers produced by backends during
  postprocessing, later used by the backend `write` step.
- Metadata flow: `read_tile` extracts incoming TIFF metadata; `compute_tile`
  returns a merged `metadata_out` that includes backend patches such as
  `deconv_min`, `deconv_scale`, and `prenormalized` where applicable.
- Timing: processors record timing contributions for stages "basic",
  "deconv", "quant", and "post"; the writer adds its own elapsed seconds for
  logging and progress reporting.

Extending the system
--------------------

1) Add a new output format
   Implement an `OutputBackend` with three methods. Keep GPU work inside
   `postprocess` minimal—convert the CuPy array to NumPy only as needed and
   return small host‑side payloads via `OutputArtifacts`.

   Example (sketch):

       class MyZarrBackend:
           def __init__(self, config: ProcessorConfig):
               self.config = config
               self._store: zarr.Group | None = None

           def setup(self, processor: "DeconvolutionTileProcessor") -> None:
               # Create/open Zarr store under processor.config.output_dir
               ...

           def postprocess(self, res: cp.ndarray, path: Path, hw: tuple[int, int]):
               arr = cp.asnumpy(res).astype(np.float32, copy=False)
               art = OutputArtifacts(f32_payload=arr)
               return art, {"dtype": "float32"}, {"quant": 0.0, "post": 0.0}

           def write(self, path: Path, fid_np: np.ndarray, metadata_out: dict[str, Any],
                     artifacts: OutputArtifacts, out_dir: Path) -> float:
               t0 = time.perf_counter()
               # Persist into Zarr using path‑relative layout
               ...
               return time.perf_counter() - t0

   Select your backend in `DeconvolutionTileProcessor.setup()` based on config
   flags or new options added to `ProcessorConfig`.

2) Customize compute
   Implement a new `TileProcessor` if you need a different read/compute/write
   lifecycle (e.g., a different deconvolution kernel or additional GPU stages).
   Your processor must honor the protocol and should reuse `ProcessorConfig`
   where feasible for consistency.

Sequential usage (no multiprocessing)
------------------------------------

    processor = DeconvolutionTileProcessor(config, device_id=0)
    processor.setup()
    for p in files:
        processor.process(p)
    processor.teardown()

See also
--------

- `fishtools/preprocess/deconv/worker.py` for the worker processes,
  threading pipeline, and scheduling.
- `fishtools/preprocess/cli_deconv_multi.py` for CLI integration and how
  `ProcessorConfig` is constructed and passed to workers.
"""

from __future__ import annotations

import csv
import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

import cupy as cp
import numpy as np
import tifffile

from fishtools.utils.io import safe_imwrite

from .basic_utils import load_basic_profiles
from .core import deconvolve_lucyrichardson_guo, load_projectors_cached
from .hist import sample_histograms
from .normalize import quantize_global


@dataclass(slots=True)
class ProcessorConfig:
    round_name: str
    basic_paths: Sequence[Path]
    output_dir: Path
    n_fids: int
    step: int
    save_float32: bool
    save_histograms: bool
    histogram_bins: int
    m_glob: np.ndarray | None
    s_glob: np.ndarray | None
    debug: bool


# ------------------- Output backends -------------------


@dataclass(slots=True)
class OutputArtifacts:
    """Typed container for artifacts produced after GPU compute."""

    towrite_u16: np.ndarray | None = None
    f32_payload: np.ndarray | None = None
    hist_payload: dict[str, Any] | None = None


class OutputBackend(Protocol):
    """Strategy: per-output-mode postprocess + write."""

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


class TileProcessor(Protocol):
    """Protocol describing the per-worker processing contract."""

    def setup(self) -> None:  # pragma: no cover - thin interface
        """Initialize GPU resources. Called once per worker."""

    def teardown(self) -> None:  # pragma: no cover - thin interface
        """Release resources prior to worker exit."""

    def read_tile(
        self, path: Path
    ) -> tuple[Path, np.ndarray, np.ndarray, dict[str, Any], tuple[int, int], int]:
        """Read data for a tile and return payload/fiducial separation."""

    def compute_tile(
        self,
        payload_np: np.ndarray,
        path: Path,
        hw: tuple[int, int],
        metadata_in: dict[str, Any],
    ) -> tuple[OutputArtifacts, dict[str, Any], dict[str, float]]:
        """Compute GPU results for a tile."""

    def write_tile(
        self,
        path: Path,
        fid_np: np.ndarray,
        metadata_out: dict[str, Any],
        artifacts: OutputArtifacts,
    ) -> float:
        """Persist artifacts for a tile; return elapsed write seconds."""

    def process(self, path: Path) -> float:  # pragma: no cover - optional fast-path
        """Optional single-call execution used by sequential fallbacks."""


ProcessorFactory = Callable[[int], TileProcessor]


class Float32HistBackend:
    """Emit float32 deconvolved planes (deconv32) and optional per-channel histograms."""

    def __init__(self, config: ProcessorConfig):
        self.config = config
        self._out32: Path | None = None

    def setup(self, processor: "DeconvolutionTileProcessor") -> None:
        self._out32 = processor.config.output_dir.parent / "deconv32"
        self._out32.mkdir(parents=True, exist_ok=True)

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

        hist_payload: dict[str, Any] | None = None
        if self.config.save_histograms:
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
            OutputArtifacts(towrite_u16=None, f32_payload=f32_payload, hist_payload=hist_payload),
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

        if self.config.save_float32 and artifacts.f32_payload is not None:
            safe_imwrite(
                sub32 / path.name,
                artifacts.f32_payload,
                dtype=np.float32,
                metadata={"dtype": "float32", **metadata_out},
                compression="zlib",
                compressionargs={"level": 6},
            )

        if self.config.save_histograms and artifacts.hist_payload is not None:
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
                        writer.writerow([c, f"{edges[b]:.8g}", f"{edges[b + 1]:.8g}", int(counts[b])])

        return time.perf_counter() - t0


class U16PrenormBackend:
    """Emit u16 payload + fid, and augment metadata with prenormalization fields."""

    def __init__(self, config: ProcessorConfig):
        self.config = config

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
            "prenormalized": True,
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


class DeconvolutionTileProcessor:
    """Concrete processor with split read/compute/write for pipelining."""

    def __init__(self, config: ProcessorConfig, *, device_id: int | None = None):
        self.config = config
        self._darkfield: cp.ndarray | None = None
        self._flatfield: cp.ndarray | None = None
        self._projectors: tuple[cp.ndarray, cp.ndarray] | None = None
        self._device_id = device_id
        self._backend: OutputBackend | None = None

    def set_device(self, device_id: int) -> None:
        self._device_id = device_id

    def _load_basics(self) -> None:
        self._darkfield, self._flatfield = load_basic_profiles(self.config.basic_paths)

    def setup(self) -> None:
        cuda = getattr(cp, "cuda", None)
        if cuda is not None:
            if hasattr(cuda, "MemoryPool") and hasattr(cuda, "set_allocator"):
                pool = cuda.MemoryPool()
                cuda.set_allocator(pool.malloc)
            if hasattr(cuda, "PinnedMemoryPool") and hasattr(cuda, "set_pinned_memory_allocator"):
                pinned_pool = cuda.PinnedMemoryPool()
                cuda.set_pinned_memory_allocator(pinned_pool.malloc)

        self._load_basics()
        self._projectors = load_projectors_cached(self.config.step)
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        if self.config.save_float32 or self.config.save_histograms:
            self._backend = Float32HistBackend(self.config)
        else:
            self._backend = U16PrenormBackend(self.config)
        self._backend.setup(self)

    def teardown(self) -> None:
        self._darkfield = None
        self._flatfield = None
        self._projectors = None
        self._backend = None

    def read_tile(
        self, path: Path
    ) -> tuple[Path, np.ndarray, np.ndarray, dict[str, Any], tuple[int, int], int]:
        read_start = time.perf_counter()
        with tifffile.TiffFile(path) as tif:
            try:
                metadata = tif.shaped_metadata[0]  # type: ignore[index]
            except (TypeError, IndexError):
                metadata = tif.imagej_metadata or {}
            image = tif.asarray()
        _ = time.perf_counter() - read_start

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
                f"{path} has {plane_count} payload planes not divisible by {channels} BaSiC channel(s)."
            )

        try:
            nofid = payload.reshape(-1, channels, height, width).astype(np.float32, copy=False)
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
        x = cp.asarray(payload_np)
        x -= self._darkfield  # type: ignore[operator]
        x /= self._flatfield  # type: ignore[operator]
        cp.clip(x, 0, None, out=x)
        if self.config.debug:
            cp.cuda.runtime.deviceSynchronize()
        timings["basic"] = time.perf_counter() - t_basic0

        t_dec0 = time.perf_counter()
        res = deconvolve_lucyrichardson_guo(x, self._projectors, iters=1)  # type: ignore[arg-type]
        if self.config.debug:
            cp.cuda.runtime.deviceSynchronize()
        timings["deconv"] = time.perf_counter() - t_dec0

        artifacts, meta_patch, extra_timings = self._backend.postprocess(res, path, hw)
        timings["quant"] = extra_timings.get("quant", 0.0)
        timings["post"] = extra_timings.get("post", 0.0)

        del res
        del x

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
        """Convenience wrapper used in tests and simple pipelines."""
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

    def __call__(self, device_id: int) -> TileProcessor:
        return DeconvolutionTileProcessor(self.config, device_id=device_id)


def make_processor_factory(config: ProcessorConfig) -> ProcessorFactory:
    """Construct a processor factory with defensively copied configuration."""

    safe_config = ProcessorConfig(
        round_name=config.round_name,
        basic_paths=tuple(Path(p) for p in config.basic_paths),
        output_dir=config.output_dir,
        n_fids=config.n_fids,
        step=config.step,
        save_float32=config.save_float32,
        save_histograms=config.save_histograms,
        histogram_bins=config.histogram_bins,
        m_glob=None if config.m_glob is None else np.asarray(config.m_glob, dtype=np.float32),
        s_glob=None if config.s_glob is None else np.asarray(config.s_glob, dtype=np.float32),
        debug=config.debug,
    )
    return _ProcessorFactory(safe_config)
