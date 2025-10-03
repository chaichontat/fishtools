"""Multi-GPU dispatcher scaffolding for deconvolution workflows.

This module provides queue-managed worker orchestration that allows the
existing single-GPU deconvolution codepath to remain untouched while enabling
future dynamic scheduling across multiple CUDA devices. Each worker process
requests new work units from the main process once it consumes one item,
allowing faster GPUs to receive more tiles without static sharding.
"""

from __future__ import annotations

import csv
import hashlib
import json
import multiprocessing as mp
import os
import pickle
import queue
import signal
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Protocol, Sequence

import cupy as cp
import numpy as np
import rich_click as click
import tifffile
from basicpy import BaSiC
from loguru import logger

from fishtools.preprocess.deconv.normalize import quantize_global
from fishtools.utils.io import Workspace, get_channels, get_metadata
from fishtools.utils.pretty_print import progress_bar

DEFAULT_QUEUE_DEPTH = 5
# Prefer spawn to avoid inheriting CUDA context; fall back to forkserver if unavailable.
try:
    _MP_CTX = mp.get_context("spawn")
except ValueError:  # pragma: no cover - spawn should exist, but guard anyway
    _MP_CTX = mp.get_context("forkserver")  # pyright: ignore[reportConstantRedefinition]

FORBIDDEN_PREFIXES = ("10x", "analysis", "shifts", "fid", "registered", "old", "basic")


def _configure_logging(debug: bool, *, process_label: str) -> None:
    """Configure loguru output consistently for the CLI surface."""

    logger.remove()
    logger.configure(extra={"process_label": process_label})

    prefix = "[P{extra[process_label]}]"
    if debug:
        log_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{process.name: <13}</cyan> | "
            f"{prefix} <level>{{message}}</level>"
        )
        logger.add(sys.stderr, level="DEBUG", format=log_format, enqueue=True)
    else:
        logger.add(sys.stderr, level="INFO", format=f"{prefix} {{message}}", enqueue=True)


def _round_sort_key(round_name: str) -> tuple[bool, str]:
    head = round_name.split("_")[0]
    if head.isdigit():
        return (True, f"{int(head):02d}")
    return (False, round_name)


def discover_rounds(workspace_path: Path) -> list[str]:
    rounds = {
        p.name.split("--")[0]
        for p in workspace_path.iterdir()
        if p.is_dir() and "--" in p.name and not p.name.startswith(FORBIDDEN_PREFIXES)
    }
    return sorted(rounds, key=_round_sort_key)


def infer_psf_step(tile: Path, *, default: int = 6) -> tuple[int, bool]:
    try:
        metadata = get_metadata(tile)
        waveform = json.loads(metadata["waveform"])
        step = int(waveform["params"]["step"] * 10)
        return step, True
    except (KeyError, json.JSONDecodeError, TypeError, ValueError):
        return default, False


class TileProcessor(Protocol):
    """Protocol describing the per-worker processing contract."""

    def setup(self) -> None:  # pragma: no cover - thin interface
        """Initialize GPU resources. Called once per worker."""

    def process(self, path: Path) -> float:  # pyright: ignore[reportReturnType]
        """Process a single file path and return the elapsed seconds."""

    def teardown(self) -> None:  # pragma: no cover - thin interface
        """Release resources prior to worker exit."""


ProcessorFactory = Callable[[int], TileProcessor]


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


class DeconvolutionTileProcessor(TileProcessor):
    """Concrete processor mirroring the single-GPU `_run` pipeline."""

    def __init__(self, config: ProcessorConfig, *, device_id: int | None = None):
        self.config = config
        self._darkfield: cp.ndarray | None = None
        self._flatfield: cp.ndarray | None = None
        self._projectors: tuple[cp.ndarray, cp.ndarray] | None = None
        self._out32: Path | None = None
        self._device_id = device_id

    def set_device(self, device_id: int) -> None:
        self._device_id = device_id

    def _load_basics(self) -> None:
        darks: list[np.ndarray] = []
        flats: list[np.ndarray] = []
        reference_shape: tuple[int, ...] | None = None
        for pkl_path in self.config.basic_paths:
            loaded = pickle.loads(pkl_path.read_bytes())
            basic = loaded.get("basic") if isinstance(loaded, dict) else loaded
            if not hasattr(basic, "darkfield") or not hasattr(basic, "flatfield"):
                raise TypeError("Basic profile payload must expose 'darkfield' and 'flatfield' attributes.")
            dark = np.asarray(basic.darkfield, dtype=np.float32)
            flat = np.asarray(basic.flatfield, dtype=np.float32)
            if dark.shape != flat.shape:
                raise ValueError(f"Dark/flat shape mismatch for {pkl_path}: {dark.shape} vs {flat.shape}.")
            if reference_shape is None:
                reference_shape = dark.shape
            elif dark.shape != reference_shape:
                raise ValueError(
                    "Inconsistent BaSiC profile geometry detected; all dark/flat fields must match."
                )
            if not np.all(np.isfinite(flat)) or np.any(flat <= 0):
                raise ValueError(f"Invalid flatfield values encountered in {pkl_path}.")
            if not np.all(np.isfinite(dark)):
                raise ValueError(f"Invalid darkfield values encountered in {pkl_path}.")
            darks.append(dark)
            flats.append(flat)

        if not darks or not flats:
            raise ValueError("No BaSiC profiles were loaded; cannot continue.")

        darkfield_np = np.stack(darks, axis=0)[np.newaxis, ...].astype(np.float32, copy=False)
        flatfield_np = np.stack(flats, axis=0)[np.newaxis, ...].astype(np.float32, copy=False)
        self._darkfield = cp.asarray(darkfield_np)
        self._flatfield = cp.asarray(flatfield_np)

    def setup(self) -> None:
        from fishtools.preprocess.cli_deconv import projectors

        self._load_basics()
        self._projectors = projectors(self.config.step)
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        if self.config.save_float32 or self.config.save_histograms:
            self._out32 = self.config.output_dir.parent / "deconv32"
            self._out32.mkdir(parents=True, exist_ok=True)

    def teardown(self) -> None:
        self._darkfield = None
        self._flatfield = None
        self._projectors = None

    # -- Core processing ----------------------------------------------------------
    def process(self, path: Path) -> float:
        from fishtools.preprocess.cli_deconv import deconvolve_lucyrichardson_guo

        if self._darkfield is None or self._flatfield is None or self._projectors is None:
            raise RuntimeError("Processor not initialised; call setup() before process().")

        start = time.perf_counter()
        read_elapsed = 0.0
        basic_elapsed = 0.0
        deconv_elapsed = 0.0
        quant_elapsed = 0.0
        write_elapsed = 0.0

        logger.debug(f"[{path.name}] START Read/Pre-process")

        read_start = time.perf_counter()
        with tifffile.TiffFile(path) as tif:
            try:
                metadata = tif.shaped_metadata[0]  # type: ignore[index]
            except (TypeError, IndexError):
                metadata = tif.imagej_metadata or {}
            image = tif.asarray()
        read_elapsed = time.perf_counter() - read_start
        logger.debug(f"[{path.name}] Disk read took: {read_elapsed:.4f}s")

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
                f"{path} has {plane_count} payload planes which is not divisible by the {channels} BaSiC channel(s)."
            )

        try:
            nofid = payload.reshape(-1, channels, height, width).astype(np.float32, copy=False)
        except ValueError as exc:
            raise ValueError(
                f"{path} has unexpected shape {payload.shape} for {channels} channels after removing {n_fids} fiducial slice(s)."
            ) from exc

        nofid_gpu = cp.asarray(nofid)
        logger.debug(
            f"[{path.name}] Loaded payload with {nofid_gpu.shape[0]} block(s) and {channels} channel(s)."
        )

        basic_start = time.perf_counter()
        nofid_gpu -= self._darkfield
        nofid_gpu /= self._flatfield
        cp.clip(nofid_gpu, 0, None, out=nofid_gpu)

        if self.config.debug:
            cp.cuda.runtime.deviceSynchronize()
        basic_elapsed = time.perf_counter() - basic_start
        logger.debug(f"[{path.name}] BaSiC correction took: {basic_elapsed:.4f}s")

        if self.config.debug:
            cp.cuda.runtime.deviceSynchronize()

        deconv_start = time.perf_counter()
        res = deconvolve_lucyrichardson_guo(nofid_gpu, self._projectors, iters=1)

        if self.config.debug:
            cp.cuda.runtime.deviceSynchronize()
        deconv_elapsed = time.perf_counter() - deconv_start
        logger.debug(f"[{path.name}] Deconvolution kernel took: {deconv_elapsed:.4f}s")

        towrite_u16: np.ndarray | None = None
        if not self.config.save_float32:
            if self.config.m_glob is None or self.config.s_glob is None:
                raise ValueError("Global scaling not provided for quantisation path")
            quant_start = time.perf_counter()
            towrite_u16 = quantize_global(
                res,
                self.config.m_glob,
                self.config.s_glob,
                i_max=65535,
                return_stats=False,
                as_numpy=True,
            )
            quant_elapsed = time.perf_counter() - quant_start
            logger.debug(f"[{path.name}] Quantization took: {quant_elapsed:.4f}s")

        f32_payload: np.ndarray | None = None
        hist_payload: dict[str, Any] | None = None

        if self.config.save_float32 or self.config.save_histograms:
            reshaped = res.reshape(-1, height, width)
            f32_payload = cp.asnumpy(reshaped).astype(np.float32, copy=False)

            if self.config.save_histograms:
                digest = hashlib.blake2b(
                    f"{self.config.round_name}:{path}".encode("utf-8"), digest_size=8
                ).digest()
                rng = np.random.default_rng(int.from_bytes(digest, "little"))
                _, C, _, _ = res.shape
                counts_list: list[np.ndarray] = []
                edges_list: list[np.ndarray] = []
                SZ, SY, SX = 4, 2, 2
                off_z = rng.integers(SZ) if SZ > 1 else 0
                off_y = rng.integers(SY) if SY > 1 else 0
                off_x = rng.integers(SX) if SX > 1 else 0

                mins_ch = cp.min(res[off_z::SZ, :, off_y::SY, off_x::SX], axis=(0, 2, 3))
                maxs_ch = cp.max(res[off_z::SZ, :, off_y::SY, off_x::SX], axis=(0, 2, 3))

                for c in range(C):
                    lo = float(cp.asnumpy(mins_ch[c]))
                    hi = float(cp.asnumpy(maxs_ch[c]))
                    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                        lo, hi = 0.0, 1.0
                    edges_gpu = cp.linspace(lo, hi, self.config.histogram_bins + 1, dtype=cp.float32)
                    sample_c = res[off_z::SZ, c, off_y::SY, off_x::SX]
                    h_gpu, _ = cp.histogram(sample_c, bins=edges_gpu)
                    counts_list.append(cp.asnumpy(h_gpu).astype(np.int64, copy=False))
                    edges_list.append(cp.asnumpy(edges_gpu).astype(np.float32, copy=False))

                n_samp = int(res[off_z::SZ, 0, off_y::SY, off_x::SX].size)
                hist_payload = {
                    "C": int(C),
                    "bins": int(self.config.histogram_bins),
                    "counts": counts_list,
                    "edges": edges_list,
                    "strides": (SZ, SY, SX),
                    "offsets": (int(off_z), int(off_y), int(off_x)),
                    "sampled_per_channel": n_samp,
                    "tile_shape": tuple(int(x) for x in res.shape),
                }

        # All GPU work complete; release arrays before CPU writes
        del res
        del nofid_gpu

        out_dir = self.config.output_dir / path.parent.name
        out_dir.mkdir(parents=True, exist_ok=True)
        write_start = time.perf_counter()

        metadata_out = metadata.copy()
        if towrite_u16 is not None:
            metadata_out |= {
                "deconv_min": list(map(float, self.config.m_glob.flatten())),
                "deconv_scale": list(map(float, self.config.s_glob.flatten())),
                "prenormalized": True,
            }
            json_path = (out_dir / path.name).with_suffix(".deconv.json")
            json_path.write_text(json.dumps(metadata_out, indent=2))

            deliverable = np.concatenate([towrite_u16, fid.astype(np.uint16, copy=False)], axis=0)
            tifffile.imwrite(
                out_dir / path.name,
                deliverable,
                compression=22610,
                compressionargs={"level": 0.75},
                metadata=metadata_out,
            )

        if f32_payload is not None and self._out32 is not None:
            sub32 = self._out32 / path.parent.name
            sub32.mkdir(parents=True, exist_ok=True)
            if self.config.save_float32:
                tifffile.imwrite(
                    sub32 / path.name,
                    f32_payload,
                    dtype=np.float32,
                    metadata={"dtype": "float32", **metadata},
                    compression="zlib",
                )

            if self.config.save_histograms and hist_payload is not None:
                hist_path = (sub32 / path.name).with_suffix(".histogram.csv")
                with open(hist_path, "w", newline="") as fh:
                    writer = csv.writer(fh)
                    writer.writerow(["channel", "bin_left", "bin_right", "count"])
                    C = hist_payload["C"]
                    bins = hist_payload["bins"]
                    counts_list = hist_payload["counts"]
                    edges_list = hist_payload["edges"]
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

        if self.config.debug:
            cp.cuda.runtime.deviceSynchronize()
        write_elapsed = time.perf_counter() - write_start
        logger.debug(f"[{path.name}] Write stage took: {write_elapsed:.4f}s")
        total_elapsed = time.perf_counter() - start

        gpu_elapsed = basic_elapsed + deconv_elapsed + quant_elapsed
        bottleneck_label, bottleneck_value = max(
            ("read", read_elapsed),
            ("gpu", gpu_elapsed),
            ("write", write_elapsed),
            key=lambda item: item[1],
        )
        gpu_label = f"GPU{self._device_id}" if self._device_id is not None else "GPU?"
        logger.info(
            f"[{gpu_label}] {path.name}: read={read_elapsed:.2f}s gpu={gpu_elapsed:.2f}s "
            f"write={write_elapsed:.2f}s total={total_elapsed:.2f}s "
            f"bottleneck={bottleneck_label} ({bottleneck_value:.2f}s)"
        )

        return total_elapsed


@dataclass(frozen=True, slots=True)
class _ProcessorFactory:
    config: ProcessorConfig

    def __call__(self, device_id: int) -> TileProcessor:
        return DeconvolutionTileProcessor(self.config, device_id=device_id)


def make_processor_factory(config: ProcessorConfig) -> ProcessorFactory:
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


def resolve_basic_paths(
    workspace: Path,
    *,
    round_name: str,
    channels: Sequence[str],
    basic_name: str | None,
) -> list[Path]:
    paths: list[Path] = []
    for channel in channels:
        primary: Path | None = None
        if basic_name:
            candidate = workspace / "basic" / f"{basic_name}-{channel}.pkl"
            if candidate.exists():
                primary = candidate
        if primary is None:
            fallback = workspace / "basic" / f"{round_name}-{channel}.pkl"
            if not fallback.exists():
                raise FileNotFoundError(f"Missing BaSiC profile for channel {channel} in round {round_name}.")
            primary = fallback
        paths.append(primary)
        logger.debug(f"Loaded BaSiC profile for {channel} from {primary}")
    return paths


def filter_pending_files(
    files: Sequence[Path],
    *,
    out_dir: Path,
    overwrite: bool,
    save_float32: bool,
    save_histograms: bool,
) -> list[Path]:
    if overwrite:
        return list(files)

    if save_float32 or save_histograms:
        result: list[Path] = []
        out32 = out_dir.parent / "deconv32"

        for file in files:
            sub32 = out32 / file.parent.name
            targets: list[Path] = []
            if save_float32:
                targets.append(sub32 / file.name)
            if save_histograms:
                targets.append((sub32 / file.name).with_suffix(".histogram.csv"))
            if any(not target.exists() for target in targets):
                result.append(file)
        skipped = len(files) - len(result)
        if skipped:
            logger.info(f"Skipping {skipped} previously prepared tiles.")
        return result

    result = [f for f in files if not (out_dir / f.parent.name / f.name).exists()]
    skipped = len(files) - len(result)
    if skipped:
        logger.info(f"Skipping {skipped} already processed tiles.")
    return result


def load_global_scaling(path: Path, round_name: str) -> tuple[np.ndarray, np.ndarray]:
    scale_path = path / "analysis" / "deconv32" / "deconv_scaling" / f"{round_name}.txt"
    if not scale_path.exists():
        raise FileNotFoundError(
            f"Missing global scaling at {scale_path}. Run 'multi_deconv prepare' to generate histogram scaling first."
        )
    arr = np.loadtxt(scale_path).astype(np.float32).reshape((2, -1))
    return arr[0], arr[1]


def multi_prepare(
    path: Path,
    *,
    num_tiles: int,
    percent: float,
    roi: Sequence[str],
    round_names: Sequence[str],
    seed: int,
    histogram_bins: int,
    overwrite: bool,
    n_fids: int,
    basic_name: str,
    debug: bool,
    devices: Sequence[int],
    queue_depth: int,
    stop_on_error: bool,
) -> None:
    ws = Workspace(path)
    all_rois = ws.rois
    rois = list(all_rois) if not roi else [r for r in roi if r in all_rois]
    if roi and not rois:
        raise ValueError("No matching ROIs found for the provided --roi filters.")

    logger.info(f"ROIs considered: {rois if rois else list(all_rois)}")

    rounds_all = discover_rounds(path)
    logger.info(f"Discovered rounds: {rounds_all}")

    rounds = rounds_all if not round_names else [r for r in round_names if r in rounds_all]
    if round_names and not rounds:
        raise ValueError("No matching rounds found for the provided --round filters.")
    logger.info(f"Selected rounds: {rounds}")

    candidates: list[Path] = []
    for r in rounds:
        for roi_name in rois:
            candidates.extend(
                f
                for f in sorted(path.glob(f"{r}--{roi_name}/{r}-*.tif"))
                if "analysis/deconv" not in str(f)
                and not f.parent.name.endswith("basic")
                and not f.name.startswith("fid")
            )

    if not candidates:
        logger.warning("No candidate tiles found for sampling. Exiting.")
        return

    logger.info(
        "Sampling pool contains {} tile(s) across {} round(s) and {} ROI(s).",
        len(candidates),
        len(rounds),
        len(rois),
    )

    rng = np.random.default_rng(seed)
    sample_count = max(min(num_tiles, len(candidates)), int(len(candidates) * percent))
    sample_count = min(sample_count, len(candidates))
    idx = rng.choice(len(candidates), size=sample_count, replace=False)
    sampled = [candidates[i] for i in idx]
    logger.info(
        "Sampling {} tile(s) from {} candidate(s) (percent={:.2%}).",
        sample_count,
        len(candidates),
        percent,
    )

    by_round: dict[str, list[Path]] = {}
    for file in sampled:
        round_name = file.name.split("-")[0]
        by_round.setdefault(round_name, []).append(file)

    out_dir = path / "analysis" / "deconv"
    out_dir.mkdir(parents=True, exist_ok=True)

    for round_name, files in sorted(by_round.items()):
        if not files:
            continue

        logger.info(f"[{round_name}] Selected {len(files)} sampled tile(s) (pre-filter).")

        step, inferred = infer_psf_step(files[0])
        if inferred:
            logger.info(f"[{round_name}] Using PSF step={step} from waveform metadata.")
        else:
            logger.warning(
                f"[{round_name}] Could not determine PSF step from metadata; defaulting to step={step}."
            )

        channels = get_channels(files[0])
        basic_paths = resolve_basic_paths(
            path, round_name=round_name, channels=channels, basic_name=basic_name
        )

        pending = filter_pending_files(
            files,
            out_dir=out_dir,
            overwrite=overwrite,
            save_float32=True,
            save_histograms=True,
        )

        if not pending:
            logger.info(f"[{round_name}] No tiles require preparation after filtering.")
            continue

        logger.info(
            "[{}] {}/{} tile(s) pending after overwrite/histogram checks.",
            round_name,
            len(pending),
            len(files),
        )

        config = ProcessorConfig(
            round_name=round_name,
            basic_paths=basic_paths,
            output_dir=out_dir,
            n_fids=n_fids,
            step=step,
            save_float32=True,
            save_histograms=True,
            histogram_bins=histogram_bins,
            m_glob=None,
            s_glob=None,
            debug=debug,
        )

        processor_factory = make_processor_factory(config)

        with progress_bar(len(pending)) as update:

            def callback(message: WorkerMessage) -> None:
                if message.status == "ok":
                    update()
                elif message.status == "error":
                    logger.error(f"[{round_name}] Failed to process {message.path}: {message.error}")

            failures = run_multi_gpu(
                pending,
                devices=devices,
                processor_factory=processor_factory,
                queue_depth=queue_depth,
                stop_on_error=stop_on_error,
                progress_callback=callback,
                debug=debug,
            )

        if failures:
            details = ", ".join(str(msg.path) for msg in failures if msg.path is not None)
            if stop_on_error:
                raise RuntimeError(f"{round_name}: processing aborted due to failures: {details}")
            logger.warning(f"[{round_name}] Completed with failures: {details}")


def multi_batch(
    path: Path,
    *,
    ref: Path | None,
    overwrite: bool,
    n_fids: int,
    basic_name: str | None,
    delete_origin: bool,
    debug: bool,
    devices: Sequence[int],
    queue_depth: int,
    stop_on_error: bool,
) -> None:
    ws = Workspace(path)
    rois = ws.rois
    if not rois:
        raise ValueError("No ROIs found.")
    logger.info(f"Found ROIs: {rois}")

    out_dir = ws.deconved
    out_dir.mkdir(parents=True, exist_ok=True)

    rounds = discover_rounds(path)
    logger.info(f"Rounds discovered: {rounds}")

    for round_name in rounds:
        logger.info(f"Processing round: {round_name}")

        if ref is not None:
            files: list[Path] = []
            for roi_name in rois:
                allowed_indices = {
                    int(f.stem.split("-")[1])
                    for f in sorted((path / f"{ref}--{roi_name}").glob(f"{ref}-*.tif"))
                }
                if not allowed_indices:
                    logger.warning(f"No reference files found for {ref}--{roi_name}. Skipping ROI.")
                    continue

                files.extend(
                    f
                    for f in sorted(path.glob(f"{round_name}--{roi_name}/{round_name}-*.tif"))
                    if "analysis/deconv" not in str(f)
                    and not f.parent.name.endswith("basic")
                    and int(f.stem.split("-")[1]) in allowed_indices
                )
        else:
            files = [
                f
                for f in sorted(path.glob(f"{round_name}--*/{round_name}-*.tif"))
                if "analysis/deconv" not in str(f) and not f.parent.name.endswith("basic")
            ]

        if not files:
            logger.warning(f"No files found to process for round {round_name}; skipping.")
            continue

        logger.info("[{}] {} candidate tile(s) before filtering.", round_name, len(files))

        step, inferred = infer_psf_step(files[0])
        if inferred:
            logger.info(f"[{round_name}] Using PSF step={step} from metadata.")
        else:
            logger.warning(f"[{round_name}] Could not determine PSF step; defaulting to step={step}.")

        channels = get_channels(files[0])
        basic_paths = resolve_basic_paths(
            path,
            round_name=round_name,
            channels=channels,
            basic_name=basic_name,
        )

        m_glob, s_glob = load_global_scaling(path, round_name)

        pending = filter_pending_files(
            files,
            out_dir=out_dir,
            overwrite=overwrite,
            save_float32=False,
            save_histograms=False,
        )

        if not pending:
            logger.info(f"[{round_name}] All tiles already processed; skipping.")
            continue

        logger.info(
            "[{}] {}/{} tile(s) pending after overwrite checks.",
            round_name,
            len(pending),
            len(files),
        )

        config = ProcessorConfig(
            round_name=round_name,
            basic_paths=basic_paths,
            output_dir=out_dir,
            n_fids=n_fids,
            step=step,
            save_float32=False,
            save_histograms=False,
            histogram_bins=8192,
            m_glob=m_glob,
            s_glob=s_glob,
            debug=debug,
        )

        processor_factory = make_processor_factory(config)

        with progress_bar(len(pending)) as update:

            def callback(message: WorkerMessage) -> None:
                if message.status == "ok":
                    update()
                elif message.status == "error":
                    logger.error(f"[{round_name}] Failed to process {message.path}: {message.error}")

            failures = run_multi_gpu(
                pending,
                devices=devices,
                processor_factory=processor_factory,
                queue_depth=queue_depth,
                stop_on_error=stop_on_error,
                progress_callback=callback,
                debug=debug,
            )

        if failures:
            details = ", ".join(str(msg.path) for msg in failures if msg.path is not None)
            if stop_on_error:
                raise RuntimeError(f"{round_name}: processing aborted due to failures: {details}")
            logger.warning(f"[{round_name}] Completed with failures: {details}")

        if delete_origin:
            from fishtools.preprocess.cli_deconv import _safe_delete_origin_dirs

            try:
                _safe_delete_origin_dirs(files, out_dir)
            except Exception as exc:  # noqa: BLE001
                logger.error(f"[{round_name}] Failed to delete origin directories: {exc}")


def multi_run(
    path: Path,
    name: str,
    *,
    ref: Path | None,
    limit: int | None,
    overwrite: bool,
    n_fids: int,
    basic_name: str | None,
    debug: bool,
    devices: Sequence[int],
    queue_depth: int,
    stop_on_error: bool,
) -> None:
    out_dir = path / "analysis" / "deconv"
    out_dir.mkdir(parents=True, exist_ok=True)

    if ref is not None:
        files: list[Path] = []
        for roi in [r.name.split("--")[1] for r in path.glob(f"{ref}--*")]:
            allowed = {
                int(f.stem.split("-")[1]) for f in sorted((path / f"{ref}--{roi}").glob(f"{ref}-*.tif"))
            }
            if not allowed:
                continue
            files.extend(
                f
                for f in sorted(path.glob(f"{name}--{roi}/{name}-*.tif"))
                if "analysis/deconv" not in str(f)
                and not f.parent.name.endswith("basic")
                and int(f.stem.split("-")[1]) in allowed
            )
    else:
        files = [
            f
            for f in sorted(path.glob(f"{name}--*/{name}-*.tif"))
            if "analysis/deconv" not in str(f) and not f.parent.name.endswith("basic")
        ]

    if limit is not None:
        files = files[:limit]

    logger.info(f"Total: {len(files)} at {path}/{name}*" + (f" limited to {limit}" if limit else ""))

    if not files:
        logger.warning("No files found to process. Exiting.")
        return

    basic_token = basic_name or name
    logger.info(f"Using {path / 'basic'}/{basic_token}-*.pkl for BaSiC")

    channels = get_channels(files[0])
    basic_paths = resolve_basic_paths(
        path,
        round_name=name,
        channels=channels,
        basic_name=basic_token,
    )

    m_glob, s_glob = load_global_scaling(path, name)

    pending = filter_pending_files(
        files,
        out_dir=out_dir,
        overwrite=overwrite,
        save_float32=False,
        save_histograms=False,
    )

    if not pending:
        logger.info("All tiles already processed; nothing to do.")
        return

    logger.info(
        "{}/{} tile(s) pending after overwrite checks.",
        len(pending),
        len(files),
    )

    # Determine PSF step from metadata if available
    step, inferred = infer_psf_step(pending[0])
    if inferred:
        logger.info(f"[{name}] Using PSF step={step} from metadata.")
    else:
        logger.warning(f"[{name}] Could not determine PSF step; defaulting to step={step}.")

    config = ProcessorConfig(
        round_name=name,
        basic_paths=basic_paths,
        output_dir=out_dir,
        n_fids=n_fids,
        step=step,
        save_float32=False,
        save_histograms=False,
        histogram_bins=8192,
        m_glob=m_glob,
        s_glob=s_glob,
        debug=debug,
    )

    processor_factory = make_processor_factory(config)
    depth = queue_depth if queue_depth > 0 else DEFAULT_QUEUE_DEPTH

    with progress_bar(len(pending)) as update:

        def callback(message: WorkerMessage) -> None:
            if message.status == "ok":
                update()
            elif message.status == "error":
                logger.error(f"[{name}] Failed to process {message.path}: {message.error}")

        failures = run_multi_gpu(
            pending,
            devices=devices,
            processor_factory=processor_factory,
            queue_depth=depth,
            stop_on_error=stop_on_error,
            progress_callback=callback,
            debug=debug,
        )

    if failures:
        details = ", ".join(str(msg.path) for msg in failures if msg.path is not None)
        if stop_on_error:
            raise RuntimeError(f"{name}: processing aborted due to failures: {details}")
        logger.warning(f"[{name}] Completed with failures: {details}")


WorkerStatus = Literal["ok", "error", "stopped"]


@dataclass(slots=True)
class WorkerMessage:
    worker_id: int
    path: Path | None
    status: WorkerStatus
    duration: float | None = None
    error: str | None = None


def parse_device_spec(spec: str | Sequence[int] | None) -> list[int]:
    """Resolve a device specification into explicit CUDA device indices."""

    visible_env = os.environ.get("CUDA_VISIBLE_DEVICES")
    visible_mask: list[int] | None = None
    if visible_env:
        visible_mask = [int(item.strip()) for item in visible_env.split(",") if item.strip()]
    try:
        total = cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError as exc:
        raise RuntimeError("No CUDA devices available for multi-GPU execution") from exc

    if total <= 0:
        raise RuntimeError("No CUDA devices detected")

    if spec is None or spec == "auto":
        return list(range(total))

    if isinstance(spec, str):
        items = [s.strip() for s in spec.split(",") if s.strip()]
        result = [int(it) for it in items]
    else:
        result = [int(it) for it in spec]

    if not result:
        raise ValueError("Device specification resolved to an empty list")

    if visible_mask is not None:
        max_idx = len(visible_mask) - 1
        for dev in result:
            if dev < 0 or dev > max_idx:
                raise ValueError(f"Device index {dev} is outside the visible range [0, {max_idx}]")
    else:
        for dev in result:
            if dev < 0 or dev >= total:
                raise ValueError(f"Device index {dev} is outside the range [0, {total - 1}]")

    return result


def _worker_loop(
    *,
    worker_id: int,
    device_id: int,
    queue_depth: int,
    task_queue: mp.Queue,
    request_queue: mp.Queue,
    result_queue: mp.Queue,
    processor_factory: ProcessorFactory,
    stop_on_error: bool,
    debug: bool,
) -> None:
    """Worker entrypoint executed in a separate process."""

    _configure_logging(debug, process_label=str(worker_id + 1))
    logger.debug(f"Worker {worker_id} starting on device {device_id} with queue depth {queue_depth}.")
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    cp.cuda.Device(device_id).use()

    processor = processor_factory(device_id)
    processor.setup()

    try:
        # Prime dispatcher credits
        for _ in range(queue_depth):
            request_queue.put(worker_id)

        while True:
            try:
                item = task_queue.get()
            except (EOFError, OSError):
                break

            if item is None:  # Sentinel
                break

            # Request another item immediately to maintain queue_depth credits.
            request_queue.put(worker_id)

            start = time.perf_counter()
            try:
                duration = processor.process(item)
            except Exception as exc:  # noqa: BLE001
                duration = time.perf_counter() - start
                import traceback

                tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
                result_queue.put(
                    WorkerMessage(
                        worker_id=worker_id,
                        path=item,
                        status="error",
                        duration=duration,
                        error=tb,
                    )
                )
                if stop_on_error:
                    break
            else:
                elapsed = duration if duration is not None else time.perf_counter() - start
                result_queue.put(
                    WorkerMessage(
                        worker_id=worker_id,
                        path=item,
                        status="ok",
                        duration=elapsed,
                    )
                )
    finally:
        processor.teardown()
        result_queue.put(WorkerMessage(worker_id=worker_id, path=None, status="stopped"))


def run_multi_gpu(
    files: Sequence[Path],
    *,
    devices: Sequence[int],
    processor_factory: ProcessorFactory,
    queue_depth: int = DEFAULT_QUEUE_DEPTH,
    stop_on_error: bool = True,
    progress_callback: Callable[[WorkerMessage], None] | None = None,
    debug: bool = False,
) -> list[WorkerMessage]:
    """Execute deconvolution across multiple GPUs with dynamic scheduling.

    The dispatcher responds to worker credit requests, keeping at most
    ``queue_depth`` outstanding paths per worker. Faster GPUs therefore receive
    more work without explicit sharding.
    """

    if queue_depth <= 0:
        raise ValueError("queue_depth must be positive")

    if not files:
        return []

    logger.info(
        f"Dispatching {len(files)} tile(s) across {len(devices)} worker(s) with queue depth {queue_depth}."
    )

    worker_processes: list[mp.Process] = []
    task_queues: list[mp.Queue] = []
    request_queue: mp.Queue = _MP_CTX.Queue(maxsize=max(1, queue_depth * max(len(devices), 1)))
    result_queue: mp.Queue = _MP_CTX.Queue(maxsize=max(1, len(files) + len(devices)))

    assigned: dict[int, list[Path]] = {}

    # Spawn workers
    for wid, device_id in enumerate(devices):
        task_queue: mp.Queue = _MP_CTX.Queue(maxsize=queue_depth)
        proc = _MP_CTX.Process(
            target=_worker_loop,
            kwargs={
                "worker_id": wid,
                "device_id": device_id,
                "queue_depth": queue_depth,
                "task_queue": task_queue,
                "request_queue": request_queue,
                "result_queue": result_queue,
                "processor_factory": processor_factory,
                "stop_on_error": stop_on_error,
                "debug": debug,
            },
            name=f"P{wid + 1}",
            daemon=True,
        )
        proc.start()
        worker_processes.append(proc)
        task_queues.append(task_queue)
        assigned[wid] = []

    pending_files = deque(Path(path) for path in files)
    active_workers = set(range(len(devices)))
    pending_stop: set[int] = set()
    failures: list[WorkerMessage] = []
    cancelled_paths: set[Path] = set()

    try:
        while active_workers:
            try:
                wid = request_queue.get(timeout=1.0)
            except queue.Empty:
                for proc in worker_processes:
                    if not proc.is_alive():
                        active_workers.discard(worker_processes.index(proc))
                continue

            if wid not in active_workers:
                continue

            task_queue = task_queues[wid]

            if wid in pending_stop:
                task_queue.put(None)
                active_workers.discard(wid)
                continue

            if not pending_files:
                task_queue.put(None)
                active_workers.discard(wid)
                continue

            next_path = pending_files.popleft()
            task_queue.put(next_path)
            assigned[wid].append(next_path)

            # Drain result_queue opportunistically for progress updates
            while True:
                try:
                    message = result_queue.get_nowait()
                except queue.Empty:
                    break
                if message.path is not None:
                    assigned_list = assigned.get(message.worker_id, [])
                    if message.path in assigned_list:
                        assigned_list.remove(message.path)

                if progress_callback is not None:
                    progress_callback(message)

                if message.status == "error":
                    failures.append(message)
                    if stop_on_error:
                        for wid_requeue, items in assigned.items():
                            for pending_path in items:
                                cancelled_paths.add(pending_path)
                            assigned[wid_requeue] = []
                        cancelled_paths.update(pending_files)
                        pending_files.clear()
                        pending_stop.update(active_workers)
                elif message.status == "stopped":
                    active_workers.discard(message.worker_id)
    finally:
        # Deliver sentinel messages without risking deadlocks
        for proc, task_queue in zip(worker_processes, task_queues):
            if proc.is_alive():
                try:
                    task_queue.put_nowait(None)
                except queue.Full:
                    try:
                        task_queue.close()
                        task_queue.cancel_join_thread()
                    except Exception:  # pragma: no cover - best-effort cleanup
                        pass
            else:
                try:
                    task_queue.close()
                except Exception:  # pragma: no cover - best-effort cleanup
                    pass

        for proc in worker_processes:
            if proc.is_alive():
                proc.join(timeout=10.0)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=5.0)

        # Drain any remaining results
        while True:
            try:
                message = result_queue.get_nowait()
            except queue.Empty:
                break
            if progress_callback is not None:
                progress_callback(message)
            if message.status == "error":
                failures.append(message)

        for cancelled in cancelled_paths:
            failures.append(
                WorkerMessage(
                    worker_id=-1,
                    path=cancelled,
                    status="error",
                    duration=None,
                    error="Cancelled after earlier failure; tile not processed.",
                )
            )

    return failures


@click.group()
def multi_deconv() -> None:
    """Experimental multi-GPU deconvolution commands."""


@multi_deconv.command()
@click.argument("path", type=click.Path(path_type=Path))
@click.option("-n", "num_tiles", type=int, default=100, show_default=True)
@click.option("--percent", type=float, default=0.1, show_default=True)
@click.option("--roi", type=str, multiple=True)
@click.option("--round", "round_names", type=str, multiple=True)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--histogram-bins", type=int, default=8192, show_default=True)
@click.option("--overwrite", is_flag=True)
@click.option("--n-fids", type=int, default=2, show_default=True)
@click.option("--basic-name", type=str, default="all", show_default=True)
@click.option("--debug", is_flag=True)
@click.option("--devices", type=str, default="auto", show_default=True)
@click.option("--queue-depth", type=int, default=DEFAULT_QUEUE_DEPTH, show_default=True)
@click.option(
    "--stop-on-error/--continue-on-error",
    default=True,
    show_default=True,
    help="Stop all workers after the first failure.",
)
def prepare(
    path: Path,
    *,
    num_tiles: int,
    percent: float,
    roi: tuple[str, ...],
    round_names: tuple[str, ...],
    seed: int,
    histogram_bins: int,
    overwrite: bool,
    n_fids: int,
    basic_name: str,
    debug: bool,
    devices: str,
    queue_depth: int,
    stop_on_error: bool,
) -> None:
    """Sample tiles and emit float32 + histogram artifacts using multi-GPU workers."""

    _configure_logging(debug, process_label="0")
    if debug:
        logger.info("DEBUG mode enabled. Detailed profiling logs will be generated.")

    try:
        device_list = parse_device_spec(devices)
    except Exception as exc:  # noqa: BLE001
        raise click.ClickException(str(exc)) from exc

    logger.info(f"Resolved devices: {device_list}")

    depth = queue_depth if queue_depth > 0 else DEFAULT_QUEUE_DEPTH
    logger.info(f"Using queue depth={depth}")

    multi_prepare(
        path,
        num_tiles=num_tiles,
        percent=percent,
        roi=roi,
        round_names=round_names,
        seed=seed,
        histogram_bins=histogram_bins,
        overwrite=overwrite,
        n_fids=n_fids,
        basic_name=basic_name,
        debug=debug,
        devices=device_list,
        queue_depth=depth,
        stop_on_error=stop_on_error,
    )


@multi_deconv.command()
@click.argument("path", type=click.Path(path_type=Path))
@click.option("--ref", type=click.Path(path_type=Path), default=None)
@click.option("--overwrite", is_flag=True)
@click.option("--n-fids", type=int, default=2, show_default=True)
@click.option("--basic-name", type=str, default="all", show_default=True)
@click.option("--delete-origin", is_flag=True)
@click.option("--debug", is_flag=True)
@click.option("--devices", type=str, default="auto", show_default=True)
@click.option("--queue-depth", type=int, default=DEFAULT_QUEUE_DEPTH, show_default=True)
@click.option(
    "--stop-on-error/--continue-on-error",
    default=True,
    show_default=True,
    help="Stop all workers after the first failure.",
)
def batch(
    path: Path,
    *,
    ref: Path | None,
    overwrite: bool,
    n_fids: int,
    basic_name: str,
    delete_origin: bool,
    debug: bool,
    devices: str,
    queue_depth: int,
    stop_on_error: bool,
) -> None:
    """Run multi-GPU deconvolution over rounds/ROIs."""

    _configure_logging(debug, process_label="0")
    if debug:
        logger.info("DEBUG mode enabled. Detailed profiling logs will be generated.")

    try:
        device_list = parse_device_spec(devices)
    except Exception as exc:  # noqa: BLE001
        raise click.ClickException(str(exc)) from exc

    logger.info(f"Resolved devices: {device_list}")

    depth = queue_depth if queue_depth > 0 else DEFAULT_QUEUE_DEPTH
    logger.info(f"Using queue depth={depth}")

    multi_batch(
        path,
        ref=ref,
        overwrite=overwrite,
        n_fids=n_fids,
        basic_name=basic_name,
        delete_origin=delete_origin,
        debug=debug,
        devices=device_list,
        queue_depth=depth,
        stop_on_error=stop_on_error,
    )


@multi_deconv.command()
@click.argument("path", type=click.Path(path_type=Path))
@click.argument("name", type=str)
@click.option("--ref", type=click.Path(path_type=Path), default=None)
@click.option("--limit", type=int, default=None)
@click.option("--overwrite", is_flag=True)
@click.option("--n-fids", type=int, default=2, show_default=True)
@click.option("--basic-name", type=str, default="all", show_default=True)
@click.option("--debug", is_flag=True)
@click.option("--devices", type=str, default="auto", show_default=True)
@click.option("--queue-depth", type=int, default=DEFAULT_QUEUE_DEPTH, show_default=True)
@click.option(
    "--stop-on-error/--continue-on-error",
    default=True,
    show_default=True,
    help="Stop all workers after the first failure.",
)
def run(
    path: Path,
    name: str,
    *,
    ref: Path | None,
    limit: int | None,
    overwrite: bool,
    n_fids: int,
    basic_name: str,
    debug: bool,
    devices: str,
    queue_depth: int,
    stop_on_error: bool,
) -> None:
    """Multi-GPU equivalent of the legacy `deconv run` command."""

    _configure_logging(debug, process_label="0")
    if debug:
        logger.info("DEBUG mode enabled. Detailed profiling logs will be generated.")

    try:
        device_list = parse_device_spec(devices)
    except Exception as exc:  # noqa: BLE001
        raise click.ClickException(str(exc)) from exc

    logger.info(f"Resolved devices: {device_list}")

    depth = queue_depth if queue_depth > 0 else DEFAULT_QUEUE_DEPTH
    logger.info(f"Using queue depth={depth}")

    multi_run(
        path,
        name,
        ref=ref,
        limit=limit,
        overwrite=overwrite,
        n_fids=n_fids,
        basic_name=basic_name,
        debug=debug,
        devices=device_list,
        queue_depth=depth,
        stop_on_error=stop_on_error,
    )


__all__ = [
    "DEFAULT_QUEUE_DEPTH",
    "TileProcessor",
    "ProcessorFactory",
    "ProcessorConfig",
    "DeconvolutionTileProcessor",
    "WorkerMessage",
    "parse_device_spec",
    "run_multi_gpu",
    "make_processor_factory",
    "resolve_basic_paths",
    "filter_pending_files",
    "load_global_scaling",
    "multi_prepare",
    "multi_batch",
    "multi_run",
    "multi_deconv",
    "prepare",
    "batch",
    "run",
]


if __name__ == "__main__":
    multi_deconv()
