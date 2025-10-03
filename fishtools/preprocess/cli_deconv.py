from __future__ import annotations

import pickle
import signal
from pathlib import Path
from typing import Sequence

import cupy as cp
import numpy as np
import rich_click as click
from loguru import logger
from rich.console import Console
from rich.logging import RichHandler

from fishtools.io.workspace import Workspace, get_channels
from fishtools.preprocess.config import DeconvolutionConfig, DeconvolutionOutputMode
from fishtools.preprocess.deconv.backend import (
    DeconvolutionTileProcessor,
    Float32HistBackend,
    OutputBackend,
    OutputBackendFactory,
    ProcessorConfig,
    U16PrenormBackend,
    make_processor_factory,
)
from fishtools.preprocess.deconv.basic_utils import resolve_basic_paths
from fishtools.preprocess.deconv.core import (
    load_projectors_cached,
)
from fishtools.preprocess.deconv.core import rescale as core_rescale
from fishtools.preprocess.deconv.discovery import infer_psf_step
from fishtools.preprocess.deconv.helpers import safe_delete_origin_dirs
from fishtools.preprocess.deconv.logging_utils import configure_logging
from fishtools.preprocess.deconv.normalize import (
    load_global_scaling,
)
from fishtools.preprocess.deconv.normalize import (
    precompute as _normalize_precompute,
)
from fishtools.preprocess.deconv.normalize import (
    quantize as _normalize_quantize,
)
from fishtools.preprocess.deconv.worker import (
    DEFAULT_QUEUE_DEPTH,
    WorkerMessage,
    parse_device_spec,
    run_multi_gpu,
)
from fishtools.preprocess.deconv.worker import MP_CONTEXT as _MP_CTX
from fishtools.utils.pretty_print import progress_bar
from fishtools.utils.utils import batch_roi

logger.remove()
logger.configure(handlers=[{"sink": RichHandler(), "format": "{message}", "level": "INFO"}])
console = Console()


projectors = load_projectors_cached
rescale = core_rescale


@click.group()
def deconv() -> None:
    """3D deconvolution workflows.

    Includes global quantization utilities accessible as:
    - preprocess deconv precompute
    - preprocess deconv quantize
    """


@deconv.command("precompute")
@click.argument("workspace", type=click.Path(path_type=Path))
@click.argument("round_name", type=str)
@click.option("--bins", type=int, default=8192, show_default=True)
@click.option(
    "--p-low",
    type=click.FloatRange(min=0.0, max=1.0),
    default=0.001,
    show_default=True,
    help="Lower quantile (fraction) used for global offset.",
)
@click.option(
    "--p-high",
    type=click.FloatRange(min=0.0, max=1.0),
    default=0.99999,
    show_default=True,
    help="Upper quantile (fraction) defining dynamic range.",
)
@click.option(
    "--gamma",
    type=float,
    default=1.05,
    show_default=True,
    help="Headroom multiplier applied to dynamic range before scaling.",
)
@click.option(
    "--i-max",
    type=int,
    default=2**16 - 1,
    show_default=True,
    help="Maximum code value for quantization (typically 65535).",
)
def precompute(
    workspace: Path,
    round_name: str,
    *,
    bins: int,
    p_low: float,
    p_high: float,
    gamma: float,
    i_max: int,
) -> None:
    """Aggregate histograms to produce global quantization parameters."""
    _normalize_precompute(
        workspace,
        round_name,
        bins=bins,
        p_low=p_low,
        p_high=p_high,
        gamma=gamma,
        i_max=i_max,
    )


@deconv.command("quantize")
@click.argument("workspace", type=click.Path(path_type=Path))
@click.argument("round_name", type=str)
@click.option(
    "--roi",
    "rois",
    multiple=True,
    help="Restrict quantization to specific ROI names (repeatable).",
)
@click.option(
    "--n-fids",
    type=int,
    default=2,
    show_default=True,
    help="Number of fiducial planes appended to each raw tile.",
)
@click.option(
    "--overwrite/--no-overwrite",
    default=False,
    show_default=True,
    help="Overwrite existing quantized deliverables.",
)
def quantize(
    workspace: Path,
    round_name: str,
    *,
    rois: tuple[str, ...],
    n_fids: int,
    overwrite: bool,
) -> None:
    """Quantize deconvolved float32 tiles to uint16 deliverables using global scaling."""
    _normalize_quantize(
        workspace,
        round_name,
        rois=rois,
        n_fids=n_fids,
        overwrite=overwrite,
    )


__all__ = [
    "DEFAULT_QUEUE_DEPTH",
    "DeconvolutionTileProcessor",
    "ProcessorConfig",
    "OutputBackendFactory",
    "OutputBackend",
    "WorkerMessage",
    "_MP_CTX",
    "build_backend_factory",
    "filter_pending_files",
    "load_global_scaling",
    "make_processor_factory",
    "multi_batch",
    "multi_prepare",
    "multi_run",
    "parse_device_spec",
    "run_multi_gpu",
    "cp",
    "signal",
]


_DEFAULT_OUTPUT_MODE = DeconvolutionConfig().output_mode
_PREPARE_DEFAULT_MODE = DeconvolutionOutputMode.F32


def _configure_logging(debug: bool, *, process_label: str) -> None:
    configure_logging(debug, process_label=process_label)


class _PrefixedLogger:
    __slots__ = ("_label",)

    def __init__(self, label: str | None) -> None:
        self._label = label

    def _format(self, message: str, prefix: bool) -> str:
        if prefix and self._label:
            return f"[{self._label}] {message}"
        return message

    def info(self, message: str, *, prefix: bool = True) -> None:
        logger.info(self._format(message, prefix))

    def warning(self, message: str, *, prefix: bool = True) -> None:
        logger.warning(self._format(message, prefix))

    def error(self, message: str, *, prefix: bool = True) -> None:
        logger.error(self._format(message, prefix))


def _is_candidate_tile(path: Path) -> bool:
    parent_name = path.parent.name
    return (
        "analysis/deconv" not in str(path)
        and not parent_name.endswith("basic")
        and not path.name.startswith("fid")
    )


def _collect_round_tiles(
    root: Path,
    round_name: str,
    *,
    rois: Sequence[str] | None = None,
    ref_round: str | None = None,
) -> list[Path]:
    """Discover tiles for a round, optionally constrained by ROI and reference indices."""

    roi_filter = set(rois) if rois else None

    ref_indices: dict[str, set[int]] = {}
    if ref_round is not None:
        ref_pattern = f"{ref_round}--*/{ref_round}-*.tif"
        for ref_tile in root.glob(ref_pattern):
            if not _is_candidate_tile(ref_tile):
                continue
            roi_name_parts = ref_tile.parent.name.split("--", 1)
            if len(roi_name_parts) != 2:
                continue
            roi_name = roi_name_parts[1]
            try:
                tile_idx = int(ref_tile.stem.split("-")[1])
            except (IndexError, ValueError):
                continue
            ref_indices.setdefault(roi_name, set()).add(tile_idx)

        if roi_filter:
            for roi_name in roi_filter:
                if roi_name not in ref_indices:
                    logger.warning(f"No reference files found for {ref_round}--{roi_name}. Skipping ROI.")

    tiles: list[Path] = []
    pattern = f"{round_name}--*/{round_name}-*.tif"
    for tile in sorted(root.glob(pattern)):
        if not _is_candidate_tile(tile):
            continue
        roi_parts = tile.parent.name.split("--", 1)
        if len(roi_parts) != 2:
            continue
        roi_name = roi_parts[1]
        if roi_filter and roi_name not in roi_filter:
            continue
        if ref_round is not None:
            allowed = ref_indices.get(roi_name)
            if not allowed:
                continue
            try:
                tile_idx = int(tile.stem.split("-")[1])
            except (IndexError, ValueError):
                continue
            if tile_idx not in allowed:
                continue
        tiles.append(tile)

    return tiles


def _normalize_mode(value: str | DeconvolutionOutputMode) -> DeconvolutionOutputMode:
    if isinstance(value, DeconvolutionOutputMode):
        return value
    normalized = value.lower()
    if normalized in {"float32", "f32", "float32_hist", "f32hist"}:
        return DeconvolutionOutputMode.F32
    if normalized in {"u16", "uint16", "u16_prenorm", "prenorm"}:
        return DeconvolutionOutputMode.U16
    raise click.BadParameter(f"Unknown backend '{value}'. Expected one of: float32, u16.")


# ------------------------------ Backend choice ------------------------------ #


def build_backend_factory(mode: str | DeconvolutionOutputMode) -> OutputBackendFactory:
    """Return a picklable factory for OutputBackend.

    Avoid lambdas/locals so multiprocessing can pickle the factory across processes.
    """
    resolved = _normalize_mode(mode)
    if resolved is DeconvolutionOutputMode.F32:
        return _make_float32_backend
    return _make_u16_backend


def _make_float32_backend(cfg: ProcessorConfig) -> OutputBackend:
    return Float32HistBackend(cfg)


def _make_u16_backend(cfg: ProcessorConfig) -> OutputBackend:
    return U16PrenormBackend(cfg)


# ------------------------------ Pending filtering ------------------------------ #


def filter_pending_files(
    files: Sequence[Path],
    *,
    out_dir: Path,
    overwrite: bool,
    backend: OutputBackend | None = None,
    mode: DeconvolutionOutputMode = DeconvolutionOutputMode.U16,
) -> list[Path]:
    if overwrite:
        return list(files)

    pending: list[Path] = []
    root32 = out_dir.parent / "deconv32"
    for f in files:
        if backend is not None:
            targets = backend.expected_targets(out_dir, f)
        else:
            targets: list[Path]
            if mode is DeconvolutionOutputMode.F32:
                float_target = root32 / f.parent.name / f.name
                targets = [float_target, float_target.with_suffix(".histogram.csv")]
            else:
                targets = [out_dir / f.parent.name / f.name]
        if any(not t.exists() for t in targets):
            pending.append(f)

    skipped = len(files) - len(pending)
    if skipped:
        logger.info(f"Skipping {skipped} tile(s) already satisfied by existing artifacts.")
    return pending


# ------------------------------ Core round runner ------------------------------ #


def _run_round_tiles(
    *,
    path: Path,
    round_name: str,
    files: Sequence[Path],
    out_dir: Path,
    basic_name: str | None,
    n_fids: int,
    histogram_bins: int,
    load_scaling: bool,
    overwrite: bool,
    debug: bool,
    devices: Sequence[int],
    stop_on_error: bool,
    label: str | None,
    mode: DeconvolutionOutputMode,
) -> list[WorkerMessage]:
    """Run one round with explicit OutputBackend chosen by CLI."""
    file_list = list(files)
    if not file_list:
        return []

    prefixed = _PrefixedLogger(label)

    prefixed.info(f"{len(file_list)} candidate tile(s) before filtering.")

    step, inferred = infer_psf_step(file_list[0])
    if inferred:
        prefixed.info(f"Using PSF step={step} inferred from tile metadata.")
    else:
        prefixed.warning(f"Could not determine PSF step; defaulting to step={step}.")

    channels = get_channels(file_list[0])
    basic_paths = resolve_basic_paths(path, round_name=round_name, channels=channels, basic_name=basic_name)

    if load_scaling:
        m_glob, s_glob = load_global_scaling(path, round_name)
    else:
        m_glob = s_glob = None

    # Build ProcessorConfig now so the backend can compute expected targets.
    config = ProcessorConfig(
        round_name=round_name,
        basic_paths=basic_paths,
        output_dir=out_dir,
        n_fids=n_fids,
        step=step,
        mode=mode,
        histogram_bins=histogram_bins,
        m_glob=m_glob,
        s_glob=s_glob,
        debug=debug,
    )

    backend_factory = build_backend_factory(mode)
    # Use a lightweight backend instance (no GPU) to compute idempotency targets.
    backend_for_filter = backend_factory(config)

    pending = filter_pending_files(
        file_list,
        out_dir=out_dir,
        overwrite=overwrite,
        backend=backend_for_filter,
        mode=mode,
    )
    if not pending:
        prefixed.info("All tiles already processed; skipping.")
        return []

    prefixed.info(f"{len(pending)}/{len(file_list)} tile(s) pending after overwrite checks.")

    processor_factory = make_processor_factory(config, backend_factory=backend_factory)
    depth = DEFAULT_QUEUE_DEPTH

    failures: list[WorkerMessage]

    with progress_bar(len(pending)) as update:

        def callback(message: WorkerMessage) -> None:
            if message.status == "ok":
                update()
            elif message.status == "error":
                prefixed.error(f"Failed to process {message.path}: {message.error}")

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
            raise RuntimeError(f"{label or round_name}: processing aborted due to failures: {details}")
        prefixed.warning(f"Completed with failures: {details}")

    return failures


# ------------------------------ Programmatic helper ------------------------------ #


def multi_run(
    path: Path,
    round_name: str,
    *,
    ref: str | Path | None,
    limit: int | None,
    backend: str = _PREPARE_DEFAULT_MODE.value,
    histograms: bool = False,
    histogram_bins: int = 8192,
    overwrite: bool,
    n_fids: int,
    basic_name: str,
    debug: bool,
    devices: Sequence[int],
    stop_on_error: bool,
    configure_logging: bool = False,
    process_label: str = "0",
) -> list[WorkerMessage]:
    """Run multi-GPU deconvolution for a single round programmatically."""

    if configure_logging:
        _configure_logging(debug, process_label=process_label)

    ref_round = str(ref) if ref is not None else None
    files = _collect_round_tiles(path, round_name, ref_round=ref_round)

    if limit is not None:
        files = files[:limit]

    logger.info(f"Total: {len(files)} at {path}/{round_name}*" + (f" limited to {limit}" if limit else ""))

    if not files:
        logger.warning("No files found to process. Exiting.")
        return []

    basic_token = basic_name or round_name
    logger.info(f"Using {path / 'basic'}/{basic_token}-*.pkl for BaSiC")

    out_dir = path / "analysis" / "deconv"
    out_dir.mkdir(parents=True, exist_ok=True)

    mode = _normalize_mode(backend)

    return _run_round_tiles(
        path=path,
        round_name=round_name,
        files=files,
        out_dir=out_dir,
        basic_name=basic_token,
        n_fids=n_fids,
        histogram_bins=histogram_bins,
        load_scaling=True,
        overwrite=overwrite,
        debug=debug,
        devices=list(devices),
        stop_on_error=stop_on_error,
        label=round_name,
        mode=mode,
    )


def multi_prepare(
    path: Path,
    *,
    num_tiles: int,
    percent: float,
    roi: Sequence[str] | tuple[str, ...],
    round_names: Sequence[str] | tuple[str, ...],
    seed: int,
    histogram_bins: int,
    backend: str = _DEFAULT_OUTPUT_MODE.value,
    overwrite: bool,
    n_fids: int,
    basic_name: str,
    debug: bool,
    devices: Sequence[int],
    stop_on_error: bool,
    configure_logging: bool = False,
    process_label: str = "0",
) -> None:
    if configure_logging:
        _configure_logging(debug, process_label=process_label)

    ws = Workspace(path)
    all_rois = ws.rois
    selected_rois = list(all_rois) if not roi else [r for r in roi if r in all_rois]
    if roi and not selected_rois:
        raise click.ClickException("No matching ROIs found for the provided --roi filters.")

    rounds_all = Workspace.discover_rounds(path)
    selected_rounds = rounds_all if not round_names else [r for r in round_names if r in rounds_all]
    if round_names and not selected_rounds:
        raise click.ClickException("No matching rounds found for the provided --round filters.")

    candidates: list[Path] = []
    for round_name in selected_rounds:
        candidates.extend(_collect_round_tiles(path, round_name, rois=selected_rois))

    if not candidates:
        logger.warning("No candidate tiles found for sampling. Exiting.")
        return

    rng = np.random.default_rng(seed)
    sample_count = max(min(num_tiles, len(candidates)), int(len(candidates) * percent))
    sample_count = min(sample_count, len(candidates))
    idx = rng.choice(len(candidates), size=sample_count, replace=False)
    sampled = [candidates[i] for i in idx]

    by_round: dict[str, list[Path]] = {}
    for file in sampled:
        round_token = file.name.split("-")[0]
        by_round.setdefault(round_token, []).append(file)

    out_dir = path / "analysis" / "deconv"
    out_dir.mkdir(parents=True, exist_ok=True)

    mode = _normalize_mode(backend)

    for round_name, files in sorted(by_round.items()):
        if not files:
            continue

        prefixed = _PrefixedLogger(round_name)
        prefixed.info(f"Selected {len(files)} sampled tile(s) (pre-filter).")

        _run_round_tiles(
            path=path,
            round_name=round_name,
            files=files,
            out_dir=out_dir,
            basic_name=basic_name,
            n_fids=n_fids,
            histogram_bins=histogram_bins,
            load_scaling=False,
            overwrite=overwrite,
            debug=debug,
            devices=list(devices),
            stop_on_error=stop_on_error,
            label=round_name,
            mode=mode,
        )


def multi_batch(
    path: Path,
    *,
    ref: str | Path | None,
    overwrite: bool,
    n_fids: int,
    basic_name: str | None,
    delete_origin: bool,
    debug: bool,
    devices: Sequence[int],
    stop_on_error: bool,
    backend: str = _DEFAULT_OUTPUT_MODE.value,
    histogram_bins: int = 8192,
    configure_logging: bool = False,
    process_label: str = "0",
) -> None:
    if configure_logging:
        _configure_logging(debug, process_label=process_label)

    ws = Workspace(path)
    rois = ws.rois
    if not rois:
        raise click.ClickException("No ROIs found.")

    rounds = Workspace.discover_rounds(path)
    if not rounds:
        raise click.ClickException("No rounds discovered in workspace.")

    out_dir = ws.deconved
    out_dir.mkdir(parents=True, exist_ok=True)

    mode = _normalize_mode(backend)

    ref_round = str(ref) if ref is not None else None
    basic_token = basic_name or "all"

    for round_name in rounds:
        label = round_name
        files = _collect_round_tiles(path, round_name, rois=rois, ref_round=ref_round)
        if not files:
            logger.warning(f"[{label}] No files found to process; skipping.")
            continue

        _run_round_tiles(
            path=path,
            round_name=round_name,
            files=files,
            out_dir=out_dir,
            basic_name=basic_token,
            n_fids=n_fids,
            histogram_bins=histogram_bins,
            load_scaling=True,
            overwrite=overwrite,
            debug=debug,
            devices=list(devices),
            stop_on_error=stop_on_error,
            label=label,
            mode=mode,
        )

        if delete_origin:
            try:
                safe_delete_origin_dirs(files, out_dir)
            except Exception as exc:  # noqa: BLE001
                logger.error(f"[{label}] Failed to delete origin directories: {exc}")


@deconv.command()
@click.argument("path", type=click.Path(path_type=Path))
@click.option("--ref", type=click.Path(path_type=Path), default=None)
@click.option("--overwrite", is_flag=True)
@click.option("--n-fids", type=int, default=2, show_default=True)
@click.option("--basic-name", type=str, default="all", show_default=True)
@click.option("--delete-origin", is_flag=True)
@click.option("--debug", is_flag=True)
@click.option("--multi-gpu", is_flag=True, help="Use the multi-GPU scheduler.")
@click.option("--devices", type=str, default="auto", show_default=True)
@click.option(
    "--stop-on-error/--continue-on-error",
    default=True,
    show_default=True,
    help="Stop all workers after the first failure.",
)
@click.option(
    "--backend",
    type=click.Choice(["u16", "float32"]),
    default=_DEFAULT_OUTPUT_MODE.value,
    show_default=True,
)
@click.option("--histogram-bins", type=int, default=8192, show_default=True)
def batch(
    path: Path,
    *,
    ref: Path | None,
    overwrite: bool,
    n_fids: int,
    basic_name: str,
    delete_origin: bool,
    debug: bool,
    multi_gpu: bool,
    devices: str,
    stop_on_error: bool,
    backend: str,
    histogram_bins: int,
) -> None:
    """Process all rounds/ROIs; retains legacy semantics for tests."""

    configure_logging(debug)

    use_multi = multi_gpu or getattr(_run, "_is_placeholder", False)

    if use_multi:
        try:
            device_list = parse_device_spec(devices)
        except Exception as exc:  # noqa: BLE001
            raise click.ClickException(str(exc)) from exc

        multi_batch(
            path,
            ref=ref,
            overwrite=overwrite,
            n_fids=n_fids,
            basic_name=basic_name,
            delete_origin=delete_origin,
            debug=debug,
            devices=device_list,
            stop_on_error=stop_on_error,
            backend=backend,
            histogram_bins=histogram_bins,
            configure_logging=False,
            process_label="0",
        )
        return

    # Legacy/test path leveraged by unit tests that monkeypatch _run.
    ref_round = ref.name if ref is not None else None
    out_dir = path / "analysis" / "deconv"
    out_dir.mkdir(parents=True, exist_ok=True)

    rounds = sorted({p.name.split("--")[0] for p in path.iterdir() if "--" in p.name and p.is_dir()})

    for round_name in rounds:
        files = _collect_round_tiles(path, round_name, rois=None, ref_round=ref_round)
        if not files:
            logger.warning(f"[{round_name}] No files found to process; skipping.")
            continue

        try:
            step, _ = infer_psf_step(files[0])
        except Exception:  # noqa: BLE001
            step = 6

        _run(files, out_dir, {}, overwrite, n_fids, step=step, debug=debug)

        if delete_origin:
            try:
                safe_delete_origin_dirs(files, out_dir)
            except Exception as exc:  # noqa: BLE001
                logger.error(f"[{round_name}] Failed to delete origin directories: {exc}")


# ---------- prepare ----------


@deconv.command()
@click.argument("path", type=click.Path(path_type=Path))
# Accept zero or more round names as a positional argument.
@click.argument("rounds", nargs=-1)
@click.option("-n", "num_tiles", type=int, default=100, show_default=True)
@click.option("--percent", type=float, default=0.1, show_default=True)
@click.option("--roi", type=str, multiple=True)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--histogram-bins", type=int, default=8192, show_default=True)
@click.option(
    "--backend",
    type=click.Choice(["float32", "u16"]),
    default=_PREPARE_DEFAULT_MODE.value,
    show_default=True,
)
@click.option("--overwrite", is_flag=True)
@click.option("--n-fids", type=int, default=2, show_default=True)
@click.option("--basic-name", type=str, default="all", show_default=True)
@click.option("--debug", is_flag=True)
@click.option("--devices", type=str, default="auto", show_default=True)
@click.option(
    "--stop-on-error/--continue-on-error",
    default=True,
    show_default=True,
    help="Stop all workers after the first failure.",
)
def prepare(
    path: Path,
    rounds: tuple[str, ...],
    *,
    num_tiles: int,
    percent: float,
    roi: tuple[str, ...],
    seed: int,
    histogram_bins: int,
    backend: str,
    overwrite: bool,
    n_fids: int,
    basic_name: str,
    debug: bool,
    devices: str,
    stop_on_error: bool,
) -> None:
    """Sample tiles and emit artifacts using multi-GPU workers."""
    try:
        device_list = parse_device_spec(devices)
    except Exception as exc:  # noqa: BLE001
        raise click.ClickException(str(exc)) from exc

    multi_prepare(
        path,
        num_tiles=num_tiles,
        percent=percent,
        roi=roi,
        round_names=tuple(rounds),
        seed=seed,
        histogram_bins=histogram_bins,
        backend=backend,
        overwrite=overwrite,
        n_fids=n_fids,
        basic_name=basic_name,
        debug=debug,
        devices=device_list,
        stop_on_error=stop_on_error,
        configure_logging=True,
        process_label="0",
    )


# ---------- run (progressive scoping) ----------


@batch_roi()
def _run_cli_scope(
    *,
    path: Path,
    roi: str,
    rounds: Sequence[str],
    ref_round: str | None,
    limit: int | None,
    mode: DeconvolutionOutputMode,
    histogram_bins: int,
    overwrite: bool,
    n_fids: int,
    basic_name: str,
    debug: bool,
    devices: Sequence[int],
    stop_on_error: bool,
    delete_origin: bool,
    workspace: Workspace | None = None,
) -> None:
    ws = workspace or Workspace(path)

    if roi not in ws.rois:
        raise click.ClickException(f"ROI '{roi}' not found in {path}.")

    out_dir = ws.deconved
    out_dir.mkdir(parents=True, exist_ok=True)

    for round_name in rounds:
        files = _collect_round_tiles(path, round_name, rois=[roi], ref_round=ref_round)
        if not files:
            logger.warning(f"[{round_name}/{roi}] No files found to process; skipping.")
            continue

        files_to_process = files[:limit] if limit is not None else files

        label = f"{round_name}/{roi}"
        basic_token = basic_name or round_name

        logger.info(f"[{label}] Using {(path / 'basic') / f'{basic_token}-*.pkl'} for BaSiC")
        if limit is not None and len(files_to_process) < len(files):
            logger.info(
                f"[{label}] Limit applied: processing {len(files_to_process)} of {len(files)} tile(s)."
            )

        _run_round_tiles(
            path=path,
            round_name=round_name,
            files=files_to_process,
            out_dir=out_dir,
            basic_name=basic_token,
            n_fids=n_fids,
            histogram_bins=histogram_bins,
            load_scaling=True,
            overwrite=overwrite,
            debug=debug,
            devices=devices,
            stop_on_error=stop_on_error,
            label=label,
            mode=mode,
        )

        if delete_origin:
            try:
                safe_delete_origin_dirs(files_to_process, out_dir)
            except Exception as exc:  # noqa: BLE001
                logger.error(f"[{label}] Failed to delete origin directories: {exc}")


@deconv.command()
@click.argument("path", type=click.Path(path_type=Path))
@click.argument("round_name", type=str)
@click.option("--roi", "roi_name", type=str, default="*")
@click.option("--ref", "ref_round", type=str, default=None)
@click.option("--limit", type=int, default=None)
@click.option(
    "--backend",
    type=click.Choice(["u16", "float32"]),
    default=_DEFAULT_OUTPUT_MODE.value,
    show_default=True,
)
@click.option("--histogram-bins", type=int, default=8192, show_default=True)
@click.option("--overwrite", is_flag=True)
@click.option("--delete-origin", is_flag=True)
@click.option("--n-fids", type=int, default=2, show_default=True)
@click.option("--basic-name", type=str, default="all", show_default=True)
@click.option("--debug", is_flag=True)
@click.option("--devices", type=str, default="auto", show_default=True)
@click.option(
    "--stop-on-error/--continue-on-error",
    default=True,
    show_default=True,
    help="Stop all workers after the first failure.",
)
def run(
    path: Path,
    round_name: str,
    *,
    roi_name: str,
    ref_round: str | None,
    limit: int | None,
    backend: str,
    histogram_bins: int,
    overwrite: bool,
    delete_origin: bool,
    n_fids: int,
    basic_name: str,
    debug: bool,
    devices: str,
    stop_on_error: bool,
) -> None:
    """Run multi-GPU deconvolution with progressive ROI scoping."""
    _configure_logging(debug, process_label="0")

    try:
        device_list = parse_device_spec(devices)
    except Exception as exc:  # noqa: BLE001
        raise click.ClickException(str(exc)) from exc

    workspace = Workspace(path)
    all_rois = workspace.rois
    all_rounds = Workspace.discover_rounds(path)

    roi_scope = roi_name
    round_scope = round_name

    if round_scope == "*":
        rounds = tuple(all_rounds)
    elif round_scope in all_rounds:
        rounds = (round_scope,)
    else:
        # Back-compat: allow legacy positional ROI usage by interpreting unknown round as ROI
        if roi_scope == "*" and round_scope in workspace.rois:
            roi_scope = round_scope
            rounds = tuple(all_rounds)
        else:
            raise click.ClickException(f"Round '{round_scope}' not found in {path}.")

    if not rounds:
        raise click.ClickException("No rounds discovered in workspace.")

    if roi_scope != "*" and roi_scope not in all_rois:
        raise click.ClickException(f"ROI '{roi_scope}' not found in {path}.")

    mode = _normalize_mode(backend)

    _run_cli_scope(
        path=path,
        roi=roi_scope,
        rounds=rounds,
        ref_round=ref_round,
        limit=limit,
        mode=mode,
        histogram_bins=histogram_bins,
        overwrite=overwrite,
        n_fids=n_fids,
        basic_name=basic_name,
        debug=debug,
        devices=device_list,
        stop_on_error=stop_on_error,
        delete_origin=delete_origin,
        workspace=workspace,
    )


if __name__ == "__main__":
    deconv()
