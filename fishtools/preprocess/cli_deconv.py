from __future__ import annotations

import re
import signal
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Sequence

import cupy as cp
import numpy as np
import rich_click as click
from loguru import logger
from rich.logging import RichHandler

from fishtools.io.workspace import Workspace, get_channels
from fishtools.preprocess.config import DeconvolutionConfig, DeconvolutionOutputMode
from fishtools.preprocess.deconv.backend import (
    Float32HistBackend,
    OutputBackend,
    ProcessorConfig,
    ProcessorFactory,
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
from fishtools.utils.pretty_print import get_shared_console, progress_bar

logger.remove()
logger.configure(handlers=[{"sink": RichHandler(), "format": "{message}", "level": "INFO"}])


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
    "deconv",
    "precompute",
    "quantize",
    "prepare",
    "run",
    "easy",
    "multi_run",
    "multi_prepare",
]


_DEFAULT_OUTPUT_MODE = DeconvolutionConfig().output_mode
_PREPARE_DEFAULT_MODE = DeconvolutionOutputMode.F32

ProgressUpdater = Callable[[], int]


def _configure_logging(debug: bool, *, process_label: str) -> None:
    # Parent logs route through the shared Console to avoid progress duplication.
    configure_logging(
        debug,
        process_label=process_label,
        level=("DEBUG" if debug else "INFO"),
        use_console=True,
    )


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


def _parse_roi(directory_name: str) -> str | None:
    parts = directory_name.split("--", 1)
    return parts[1] if len(parts) == 2 else None


_TILE_INDEX_RE = re.compile(r".*-(\d+)$")


def _parse_tile_index(stem: str) -> int | None:
    match = _TILE_INDEX_RE.match(stem)
    if match is None:
        return None
    return int(match.group(1))


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
            roi_name = _parse_roi(ref_tile.parent.name)
            if roi_name is None:
                continue
            tile_idx = _parse_tile_index(ref_tile.stem)
            if tile_idx is None:
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
        roi_name = _parse_roi(tile.parent.name)
        if roi_name is None:
            continue
        if roi_filter and roi_name not in roi_filter:
            continue
        if ref_round is not None:
            tile_idx = _parse_tile_index(tile.stem)
            allowed = ref_indices.get(roi_name)
            if tile_idx is None or not allowed or tile_idx not in allowed:
                continue
        tiles.append(tile)

    return tiles


def _normalize_mode(value: str | DeconvolutionOutputMode) -> DeconvolutionOutputMode:
    if isinstance(value, DeconvolutionOutputMode):
        return value
    normalized = value.lower()
    if normalized in {"float32", "f32"}:
        return DeconvolutionOutputMode.F32
    if normalized == "u16":
        return DeconvolutionOutputMode.U16
    raise click.BadParameter(f"Unknown backend '{value}'. Expected one of: float32, u16.")


# ------------------------------ Backend choice ------------------------------ #


_BACKEND_CLASSES: dict[DeconvolutionOutputMode, type[OutputBackend]] = {
    DeconvolutionOutputMode.F32: Float32HistBackend,
    DeconvolutionOutputMode.U16: U16PrenormBackend,
}


# ------------------------------ Backend choice ------------------------------ #


# ------------------------------ Pending filtering ------------------------------ #


def filter_pending_files(
    files: Sequence[Path],
    *,
    out_dir: Path,
    overwrite: bool,
    backend: OutputBackend,
) -> list[Path]:
    if overwrite:
        return list(files)

    pending: list[Path] = []
    for f in files:
        targets = backend.expected_targets(out_dir, f)
        if any(not t.exists() for t in targets):
            pending.append(f)

    skipped = len(files) - len(pending)
    if skipped:
        logger.info(f"Skipping {skipped} tile(s) already satisfied by existing artifacts.")
    return pending


# ------------------------------ Core round runner ------------------------------ #


@dataclass(slots=True)
class _RoundProcessingPlan:
    label: str | None
    prefixed: _PrefixedLogger
    files: list[Path]
    pending: list[Path]
    processor_factory: ProcessorFactory
    out_dir: Path

    @property
    def has_work(self) -> bool:
        return bool(self.pending)


def _prepare_round_plan(
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
    label: str | None,
    mode: DeconvolutionOutputMode,
    prefixed: _PrefixedLogger | None = None,
) -> _RoundProcessingPlan | None:
    file_list = list(files)
    prefixed = prefixed or _PrefixedLogger(label)

    if not file_list:
        prefixed.warning("No files found to process; skipping.")
        return None

    prefixed.info(f"{len(file_list)} candidate tile(s) before filtering.")

    step, inferred = infer_psf_step(file_list[0])
    if inferred:
        prefixed.info(f"Using PSF step={step} inferred from tile metadata.")
    else:
        prefixed.warning(f"Could not determine PSF step; defaulting to step={step}.")

    channels = get_channels(file_list[0])
    basic_paths = resolve_basic_paths(
        path,
        round_name=round_name,
        channels=channels,
        basic_name=basic_name,
    )

    if load_scaling:
        m_glob, s_glob = load_global_scaling(path, round_name)
    else:
        m_glob = s_glob = None

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

    backend_cls = _BACKEND_CLASSES[mode]
    backend_for_filter = backend_cls(config)

    pending = filter_pending_files(
        file_list,
        out_dir=out_dir,
        overwrite=overwrite,
        backend=backend_for_filter,
    )

    if not pending:
        prefixed.info("All tiles already processed; skipping.")

    else:
        prefixed.info(f"{len(pending)}/{len(file_list)} tile(s) pending after overwrite checks.")

    processor_factory = make_processor_factory(config, backend_factory=backend_cls)

    return _RoundProcessingPlan(
        label=label,
        prefixed=prefixed,
        files=file_list,
        pending=pending,
        processor_factory=processor_factory,
        out_dir=out_dir,
    )


def _execute_round_plan(
    plan: _RoundProcessingPlan,
    *,
    devices: Sequence[int],
    stop_on_error: bool,
    debug: bool,
    progress: ProgressUpdater | None = None,
) -> list[WorkerMessage]:
    if not plan.has_work:
        return []

    prefixed = plan.prefixed
    pending = plan.pending
    depth = DEFAULT_QUEUE_DEPTH

    def _run_with_progress(update: ProgressUpdater) -> list[WorkerMessage]:
        def callback(message: WorkerMessage) -> None:
            if message.status == "ok":
                update()
                # Render per-tile timing above the shared progress bar without duplicating bars
                if message.path is not None and message.stages is not None:
                    s = message.stages
                    gpu = (
                        s.get("basic", 0.0) + s.get("deconv", 0.0) + s.get("quant", 0.0) + s.get("post", 0.0)
                    )
                    dev = message.device if message.device is not None else "?"
                    get_shared_console().print(
                        f"[P{message.worker_id + 1}] [GPU{dev}] {message.path.name}: "
                        f"gpu={gpu:.2f}s (basic={s.get('basic', 0.0):.2f}+"
                        f"dec={s.get('deconv', 0.0):.2f}+quant={s.get('quant', 0.0):.2f}+post={s.get('post', 0.0):.2f}) "
                        f"stage_total={(message.duration or gpu):.2f}s"
                    )
            elif message.status == "error":
                # Print errors above the progress bar as well
                get_shared_console().print(
                    f"[bold red]Failed to process {getattr(message.path, 'name', '<unknown>')}: {message.error}[/bold red]"
                )

        return run_multi_gpu(
            pending,
            devices=devices,
            processor_factory=plan.processor_factory,
            queue_depth=depth,
            stop_on_error=stop_on_error,
            progress_callback=callback,
            debug=debug,
        )

    if progress is None:
        with progress_bar(len(pending)) as local_progress:
            failures = _run_with_progress(local_progress)
    else:
        failures = _run_with_progress(progress)

    if failures:
        details = ", ".join(str(msg.path) for msg in failures if msg.path is not None)
        if stop_on_error:
            raise RuntimeError(f"{plan.label or 'run'}: processing aborted due to failures: {details}")
        prefixed.warning(f"Completed with failures: {details}")

    return failures


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
    progress: ProgressUpdater | None = None,
) -> list[WorkerMessage]:
    """Run one round with explicit OutputBackend chosen by CLI."""

    plan = _prepare_round_plan(
        path=path,
        round_name=round_name,
        files=files,
        out_dir=out_dir,
        basic_name=basic_name,
        n_fids=n_fids,
        histogram_bins=histogram_bins,
        load_scaling=load_scaling,
        overwrite=overwrite,
        debug=debug,
        label=label,
        mode=mode,
    )

    if plan is None:
        return []

    return _execute_round_plan(
        plan,
        devices=devices,
        stop_on_error=stop_on_error,
        debug=debug,
        progress=progress,
    )


# ------------------------------ Shared planning helper ------------------------------ #


def _plan_and_execute(
    *,
    path: Path,
    rounds: Sequence[str],
    rois: Sequence[str] | None,
    ref_round: str | None,
    limit: int | None,
    limit_scope: Literal["total", "per_roi"],
    basic_name: str,
    n_fids: int,
    histogram_bins: int,
    load_scaling: bool,
    overwrite: bool,
    debug: bool,
    devices: Sequence[int],
    stop_on_error: bool,
    mode: DeconvolutionOutputMode,
    delete_origin: bool,
    progress: ProgressUpdater | None = None,
) -> list[WorkerMessage]:
    workspace = Workspace(path)
    out_dir = workspace.deconved
    out_dir.mkdir(parents=True, exist_ok=True)

    available_rois = tuple(sorted(workspace.rois))
    if rois is None:
        scope_rois = available_rois
    else:
        scope_rois = tuple(rois)

    plans: list[_RoundProcessingPlan] = []
    for round_token in rounds:
        prefixed = _PrefixedLogger(round_token)
        basic_token = basic_name or round_token
        prefixed.info(f"Using {(path / 'basic') / f'{basic_token}-*.pkl'} for BaSiC")

        if not scope_rois:
            prefixed.warning("No ROIs available; skipping round.")
            continue

        files_to_process: list[Path] = []
        total_assigned = 0
        for roi in scope_rois:
            roi_files = _collect_round_tiles(path, round_token, rois=[roi], ref_round=ref_round)
            if not roi_files:
                prefixed.warning(f"No files found for ROI '{roi}'; skipping.")
                continue

            if limit is not None:
                if limit_scope == "per_roi":
                    limited = roi_files[:limit]
                    if len(limited) < len(roi_files):
                        prefixed.info(
                            f"Applying limit={limit} to ROI '{roi}': processing {len(limited)} of {len(roi_files)} tile(s)."
                        )
                    roi_files = limited
                else:
                    remaining = limit - total_assigned
                    if remaining <= 0:
                        break
                    if len(roi_files) > remaining:
                        prefixed.info(
                            f"Applying limit={limit}: processing {remaining} tile(s) from ROI '{roi}'."
                        )
                        roi_files = roi_files[:remaining]
                    total_assigned += len(roi_files)

            files_to_process.extend(roi_files)

            if limit_scope == "total" and limit is not None and total_assigned >= limit:
                break

        if not files_to_process:
            prefixed.warning("No files discovered for requested ROI scope; skipping round.")
            continue

        plan = _prepare_round_plan(
            path=path,
            round_name=round_token,
            files=files_to_process,
            out_dir=out_dir,
            basic_name=basic_token,
            n_fids=n_fids,
            histogram_bins=histogram_bins,
            load_scaling=load_scaling,
            overwrite=overwrite,
            debug=debug,
            label=round_token,
            mode=mode,
            prefixed=prefixed,
        )

        if plan is not None:
            plans.append(plan)

    if not plans:
        logger.warning("No files found to process. Exiting.")
        return []

    total_pending = sum(len(plan.pending) for plan in plans)
    if total_pending == 0:
        logger.info("All selected tiles are already processed; nothing to do.")
        if delete_origin:
            for plan in plans:
                try:
                    safe_delete_origin_dirs(plan.files, out_dir)
                except Exception as exc:  # noqa: BLE001
                    plan.prefixed.error(f"Failed to delete origin directories: {exc}")
        return []

    failures: list[WorkerMessage] = []

    def _execute(plan: _RoundProcessingPlan, progress_callback: ProgressUpdater | None) -> None:
        plan_failures = _execute_round_plan(
            plan,
            devices=devices,
            stop_on_error=stop_on_error,
            debug=debug,
            progress=progress_callback,
        )
        failures.extend(plan_failures)
        if delete_origin:
            try:
                safe_delete_origin_dirs(plan.files, out_dir)
            except Exception as exc:  # noqa: BLE001
                plan.prefixed.error(f"Failed to delete origin directories: {exc}")

    if progress is None:
        with progress_bar(total_pending) as shared_progress:
            for plan in plans:
                _execute(plan, shared_progress)
    else:
        for plan in plans:
            _execute(plan, progress)

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

    mode = _normalize_mode(backend)

    ref_round = str(ref) if ref is not None else None

    failures = _plan_and_execute(
        path=path,
        rounds=[round_name],
        rois=None,
        ref_round=ref_round,
        limit=limit,
        limit_scope="total",
        basic_name=basic_name,
        n_fids=n_fids,
        histogram_bins=histogram_bins,
        load_scaling=True,
        overwrite=overwrite,
        debug=debug,
        devices=list(devices),
        stop_on_error=stop_on_error,
        mode=mode,
        delete_origin=False,
    )

    return failures


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

    if not (0.0 < percent <= 1.0):
        raise click.ClickException("--percent must be in (0, 1].")

    rng = np.random.default_rng(seed)
    sample_count = min(max(int(len(candidates) * percent), num_tiles), len(candidates))
    idx = rng.choice(len(candidates), size=sample_count, replace=False)
    sampled = [candidates[i] for i in idx]

    by_round: dict[str, list[Path]] = {}
    for file in sampled:
        round_token = file.name.split("-")[0]
        by_round.setdefault(round_token, []).append(file)

    out_dir = ws.deconved
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


@deconv.command()
@click.argument("path", type=click.Path(path_type=Path))
@click.argument("round_name", type=str, required=False)
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
    round_name: str | None,
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
    """Run multi-GPU deconvolution across selected rounds and ROIs."""
    _configure_logging(debug, process_label="0")

    try:
        device_list = parse_device_spec(devices)
    except Exception as exc:  # noqa: BLE001
        raise click.ClickException(str(exc)) from exc

    workspace = Workspace(path)
    all_rounds = Workspace.discover_rounds(path)
    all_rois = workspace.rois

    if not all_rounds:
        raise click.ClickException("No rounds discovered in workspace.")

    if not all_rois:
        raise click.ClickException("No ROIs found in workspace.")

    if round_name is None or round_name == "*":
        selected_rounds = tuple(all_rounds)
    elif round_name in all_rounds:
        selected_rounds = (round_name,)
    else:
        raise click.ClickException(f"Round '{round_name}' not found in {path}.")

    if roi_name == "*":
        selected_rois = tuple(sorted(all_rois))
    elif roi_name in all_rois:
        selected_rois = (roi_name,)
    else:
        raise click.ClickException(f"ROI '{roi_name}' not found in {path}.")

    mode = _normalize_mode(backend)

    ref_token = ref_round
    if ref_token is not None and ref_token not in all_rounds:
        raise click.ClickException(f"Reference round '{ref_token}' not found in {path}.")

    _plan_and_execute(
        path=path,
        rounds=selected_rounds,
        rois=selected_rois,
        ref_round=ref_token,
        limit=limit,
        limit_scope="per_roi",
        basic_name=basic_name,
        n_fids=n_fids,
        histogram_bins=histogram_bins,
        load_scaling=True,
        overwrite=overwrite,
        debug=debug,
        devices=device_list,
        stop_on_error=stop_on_error,
        mode=mode,
        delete_origin=delete_origin,
    )


@deconv.command()
@click.argument("path", type=click.Path(path_type=Path))
@click.argument("round_name", type=str, required=False)
def easy(path: Path, round_name: str | None):
    import subprocess

    rounds = [round_name] if round_name else Workspace.discover_rounds(path)
    for round_ in rounds:
        if not (path / "analysis" / "deconv32" / "deconv_scaling" / f"{round_}.txt").exists():
            subprocess.run(["preprocess", "deconv", "prepare", str(path), round_], check=True)
            subprocess.run(["preprocess", "deconv", "precompute", str(path), round_], check=True)
        subprocess.run(["preprocess", "deconv", "run", str(path), round_], check=True)


if __name__ == "__main__":
    deconv()
