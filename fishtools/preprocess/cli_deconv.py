from __future__ import annotations

import re
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Sequence

import numpy as np
import rich_click as click
from click.core import ParameterSource
from loguru import logger

from fishtools.io.workspace import Workspace, get_channels
from fishtools.preprocess.config import DeconvolutionConfig, DeconvolutionOutputMode
from fishtools.preprocess.deconv.backend import (
    DeconvolutionTileProcessor,  # noqa: F401 - re-exported for tests/consumers
    Float32HistBackend,
    LEGACY_PERCENTILES,
    LegacyPerTileU16Backend,
    OutputBackend,
    ProcessorConfig,
    ProcessorFactory,
    U16PrenormBackend,
    make_processor_factory,
)
from fishtools.preprocess.deconv.basic_utils import resolve_basic_paths
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
from fishtools.utils.logging import setup_cli_logging
from fishtools.utils.pretty_print import ProgressReporter, progress_reporter, wrap_progress


@click.group()
def deconvnew() -> None:
    """3D deconvolution workflows.

    Includes global quantization utilities accessible as:
    - preprocess deconvnew precompute
    - preprocess deconvnew quantize
    """


@deconvnew.command("precompute")
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
    setup_cli_logging(
        workspace,
        component="preprocess.deconv.precompute",
        file=f"precompute-{round_name}",
        extra={"round": round_name},
    )
    _normalize_precompute(
        workspace,
        round_name,
        bins=bins,
        p_low=p_low,
        p_high=p_high,
        gamma=gamma,
        i_max=i_max,
    )


@deconvnew.command("quantize")
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
    """Convert existing float32 deconvolution outputs into uint16 deliverables.

    This command does not perform deconvolution. Its only role is to take float32 tiles
    (typically under ``analysis/deconv32/{round}--{roi}``) and write the uint16 outputs
    (under ``analysis/deconv/{round}--{roi}``) using the precomputed global scaling.
    """
    roi_tag = "-".join(sorted(rois)) if rois else "all"
    setup_cli_logging(
        workspace,
        component="preprocess.deconv.quantize",
        file=f"quantize-{round_name}",
        extra={"round": round_name, "roi": roi_tag, "overwrite": overwrite},
    )
    _normalize_quantize(
        workspace,
        round_name,
        rois=rois,
        n_fids=n_fids,
        overwrite=overwrite,
    )

    # Delete deconv32 directories only after verifying all tiles were quantized
    ws = Workspace(workspace)
    _delete_verified_deconv32(ws, round_name)


def _delete_verified_deconv32(ws: Workspace, round_name: str) -> None:
    """Delete deconv32 directories only if all tiles have been quantized.

    For each ROI directory in deconv32 matching the round, verifies that
    every float32 tile has a corresponding quantized tile in deconv.
    Skips deletion with a warning if any tiles are missing.
    """
    deconv32_base = ws.deconv32
    deconv_base = ws.deconved
    pattern = f"{round_name}--*"

    for roi_dir in deconv32_base.glob(pattern):
        if not roi_dir.is_dir():
            continue

        # Count float32 tiles in this directory
        float32_tiles = set(p.name for p in roi_dir.glob(f"{round_name}-*.tif"))
        if not float32_tiles:
            continue

        # Check corresponding quantized output directory
        quantized_dir = deconv_base / roi_dir.name
        if not quantized_dir.exists():
            logger.warning(
                f"Skipping deletion of {roi_dir}: output directory {quantized_dir} does not exist. "
                f"{len(float32_tiles)} tile(s) may not have been quantized."
            )
            continue

        quantized_tiles = set(p.name for p in quantized_dir.glob(f"{round_name}-*.tif"))
        missing = float32_tiles - quantized_tiles

        if missing:
            logger.warning(
                f"Skipping deletion of {roi_dir}: {len(missing)} tile(s) not quantized: "
                f"{', '.join(sorted(missing)[:5])}{'...' if len(missing) > 5 else ''}"
            )
            continue

        # All tiles verified - safe to delete
        shutil.rmtree(roi_dir)
        logger.info(f"Deleted {roi_dir} ({len(float32_tiles)} tiles verified)")


__all__ = [
    "deconvnew",
    "precompute",
    "quantize",
    "prepare",
    "run",
    "batch",
    "easy",
    "multi_run",
    "multi_prepare",
]


_DEFAULT_OUTPUT_MODE = DeconvolutionConfig().output_mode
_PREPARE_DEFAULT_MODE = DeconvolutionOutputMode.F32

ProgressCallback = Callable[[], int]


def _ensure_reporter(progress: ProgressReporter | ProgressCallback | None) -> ProgressReporter | None:
    if progress is None:
        return None
    if isinstance(progress, ProgressReporter):
        return progress
    return wrap_progress(progress)


def _configure_logging(debug: bool, *, process_label: str) -> None:
    # Parent logs route through the shared Console to avoid progress duplication.
    configure_logging(
        debug,
        process_label=process_label,
        level=("DEBUG" if debug else "INFO"),
        use_console=True,
        preserve_existing=True,
    )


def _devices_callback(
    ctx: click.Context,  # noqa: ARG001
    param: click.Parameter,  # noqa: ARG001
    value: str,
) -> list[int]:
    try:
        return parse_device_spec(value)
    except (RuntimeError, ValueError) as exc:
        raise click.BadParameter(str(exc)) from exc


def _is_candidate_tile(path: Path, *, deconv_root: Path) -> bool:
    parent_name = path.parent.name
    if path.is_relative_to(deconv_root):
        return False
    return not parent_name.endswith("basic") and not path.name.startswith("fid")


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
    max_idx: int | None = None,
) -> list[Path]:
    """Discover tiles for a round, optionally constrained by ROI and reference indices."""

    ws = Workspace(root)
    scan_root = ws.path
    deconv_root = ws.deconved

    roi_filter = set(rois) if rois else None

    ref_indices: dict[str, set[int]] = {}
    if ref_round is not None:
        ref_pattern = f"{ref_round}--*/{ref_round}-*.tif"
        for ref_tile in scan_root.glob(ref_pattern):
            if not _is_candidate_tile(ref_tile, deconv_root=deconv_root):
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
    for tile in sorted(scan_root.glob(pattern)):
        if not _is_candidate_tile(tile, deconv_root=deconv_root):
            continue
        roi_name = _parse_roi(tile.parent.name)
        if roi_name is None:
            continue
        if roi_filter and roi_name not in roi_filter:
            continue
        tile_idx: int | None = None
        if ref_round is not None or max_idx is not None:
            tile_idx = _parse_tile_index(tile.stem)
            if tile_idx is None:
                if max_idx is not None:
                    raise click.ClickException(
                        f"--max-idx requires tile filenames to end with '-<digits>'; cannot parse index from '{tile.name}'."
                    )
                continue

        if max_idx is not None and tile_idx is not None and tile_idx > max_idx:
            continue

        if ref_round is not None:
            allowed = ref_indices.get(roi_name)
            if not allowed or tile_idx not in allowed:
                continue
        tiles.append(tile)

    return tiles


def _normalize_mode(value: str | DeconvolutionOutputMode) -> DeconvolutionOutputMode:
    if isinstance(value, DeconvolutionOutputMode):
        return value
    try:
        return DeconvolutionOutputMode(value)
    except ValueError as exc:
        raise click.BadParameter(f"Unknown mode '{value}'. Expected one of: u16, float32, legacy.") from exc


_BACKEND_CLASSES: dict[DeconvolutionOutputMode, type[OutputBackend]] = {
    DeconvolutionOutputMode.F32: Float32HistBackend,
    DeconvolutionOutputMode.U16: U16PrenormBackend,
    DeconvolutionOutputMode.LEGACY: LegacyPerTileU16Backend,
}


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
    files: Sequence[Path],
    mode: DeconvolutionOutputMode,
    round_name: str | None = None,
    basic_name: str | None = None,
    n_fids: int = 2,
    histogram_bins: int = 8192,
    overwrite: bool = False,
    debug: bool = False,
    label: str | None = None,
    out_dir: Path | None = None,
    min_perc: float = LEGACY_PERCENTILES[0],
    max_perc: float = LEGACY_PERCENTILES[1],
) -> _RoundProcessingPlan | None:
    file_list = list(files)
    if not file_list:
        logger.warning("No files found to process; skipping.")
        return None

    inferred_round = file_list[0].name.split("-", 1)[0]
    round_token = round_name or inferred_round
    plan_label = label or round_token
    prefix = f"[{plan_label}] " if plan_label else ""

    def _info(message: str) -> None:
        logger.info(f"{prefix}{message}")

    def _warning(message: str) -> None:
        logger.warning(f"{prefix}{message}")

    _info(f"{len(file_list)} candidate tile(s) before filtering.")

    step, inferred = infer_psf_step(file_list[0])
    if inferred:
        _info(f"Using PSF step={step} inferred from tile metadata.")
    else:
        _warning(f"Could not determine PSF step; defaulting to step={step}.")

    # Resolve BaSiC lookup channels: MUST be wavelengths to match `{name}-{wavelength}.pkl`
    # Prefer explicit wavelengths from TIFF metadata (waveform→powers),
    # only fall back to workspace heuristics when absent.
    ws = Workspace(path)
    channels = get_channels(file_list[0]) or ws.infer_channel_names(round_token)
    if not channels:
        raise click.ClickException(
            f"Cannot infer channel names for round '{round_token}'. "
            "Ensure TIFF metadata includes waveform channel names, or that deconv32 tiles exist with metadata."
        )

    try:
        basic_paths = resolve_basic_paths(
            path,
            round_name=round_token,
            channels=channels,
            basic_name=basic_name,
        )
    except FileNotFoundError as exc:
        raise click.ClickException(str(exc)) from exc

    load_scaling = mode is DeconvolutionOutputMode.U16
    if load_scaling:
        m_glob, s_glob = load_global_scaling(path, round_token)
    else:
        m_glob = s_glob = None

    processor_cfg = ProcessorConfig(
        round_name=round_token,
        basic_paths=basic_paths,
        output_dir=(out_dir or ws.deconved),
        n_fids=n_fids,
        step=step,
        mode=mode,
        histogram_bins=histogram_bins,
        m_glob=m_glob,
        s_glob=s_glob,
        debug=debug,
        legacy_percentile_low=min_perc,
        legacy_percentile_high=max_perc,
    )
    logger.info(processor_cfg)

    backend_cls = _BACKEND_CLASSES[mode]
    backend_for_filter = backend_cls(processor_cfg)

    pending = filter_pending_files(
        file_list,
        out_dir=processor_cfg.output_dir,
        overwrite=overwrite,
        backend=backend_for_filter,
    )

    if not pending:
        _info("All tiles already processed; skipping.")

    else:
        _info(f"{len(pending)}/{len(file_list)} tile(s) pending after overwrite checks.")

    processor_factory = make_processor_factory(processor_cfg, backend_factory=backend_cls)

    return _RoundProcessingPlan(
        label=plan_label,
        files=file_list,
        pending=pending,
        processor_factory=processor_factory,
        out_dir=processor_cfg.output_dir,
    )


def _execute_round_plan(
    plan: _RoundProcessingPlan,
    *,
    devices: Sequence[int],
    stop_on_error: bool,
    debug: bool,
    progress: ProgressReporter | ProgressCallback | None = None,
) -> list[WorkerMessage]:
    if not plan.has_work:
        return []

    reporter = _ensure_reporter(progress)

    pending = plan.pending
    depth = DEFAULT_QUEUE_DEPTH

    def _make_progress_callback(reporter: ProgressReporter | None) -> ProgressCallback:
        def callback(message: WorkerMessage) -> None:
            if message.status == "ok":
                if reporter is not None:
                    reporter.advance()
                if message.path is not None and message.stages is not None:
                    s = message.stages
                    gpu = s.get("basic", 0.0) + s.get("deconv", 0.0) + s.get("quant", 0.0) + s.get("post", 0.0)
                    dev = message.device if message.device is not None else "?"
                    text = (
                        f"[P{message.worker_id + 1}] [GPU{dev}] {message.path.name}: "
                        f"gpu={gpu:.2f}s (basic={s.get('basic', 0.0):.2f}+"
                        f"dec={s.get('deconv', 0.0):.2f}+quant={s.get('quant', 0.0):.2f}+post={s.get('post', 0.0):.2f}) "
                        f"stage_total={(message.duration or gpu):.2f}s"
                    )
                    if reporter is not None:
                        # Render per-tile timing above the shared progress bar without duplicating bars
                        reporter.print(text)
                    else:
                        logger.info(text)
            elif message.status == "error":
                name = getattr(message.path, "name", "<unknown>")
                if reporter is not None:
                    reporter.print(f"[bold red]Failed to process {name}: {message.error}[/bold red]")
                else:
                    logger.error(f"Failed to process {name}: {message.error}")

        return callback

    def _run_with_progress(reporter: ProgressReporter | None) -> list[WorkerMessage]:
        return run_multi_gpu(
            pending,
            devices=devices,
            processor_factory=plan.processor_factory,
            queue_depth=depth,
            stop_on_error=stop_on_error,
            progress_callback=_make_progress_callback(reporter),
            debug=debug,
        )

    if reporter is None:
        if len(pending) == 1:
            failures = _run_with_progress(None)
        else:
            with progress_reporter(len(pending)) as local_reporter:
                failures = _run_with_progress(local_reporter)
    else:
        failures = _run_with_progress(reporter)

    if failures:
        details = ", ".join(str(msg.path) for msg in failures if msg.path is not None)
        if stop_on_error:
            raise RuntimeError(f"{plan.label or 'run'}: processing aborted due to failures: {details}")
        prefix = f"[{plan.label}] " if plan.label else ""
        logger.warning(f"{prefix}Completed with failures: {details}")

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
    progress: ProgressReporter | ProgressCallback | None = None,
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
    max_idx: int | None,
    basic_name: str | None,
    n_fids: int,
    histogram_bins: int,
    load_scaling: bool,
    overwrite: bool,
    debug: bool,
    devices: Sequence[int],
    stop_on_error: bool,
    mode: DeconvolutionOutputMode,
    delete_origin: bool,
    min_perc: float = LEGACY_PERCENTILES[0],
    max_perc: float = LEGACY_PERCENTILES[1],
    progress: ProgressReporter | ProgressCallback | None = None,
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
        prefix = f"[{round_token}] "
        if basic_name:
            logger.info(f"{prefix}Using {(path / 'basic') / f'{basic_name}-*.pkl'} for BaSiC")
        else:
            logger.info(
                f"{prefix}Auto-selecting BaSiC profiles: prefer "
                f"{(path / 'basic') / f'{round_token}-*.pkl'} then 'all-*.pkl'."
            )

        if not scope_rois:
            logger.warning(f"{prefix}No ROIs available; skipping round.")
            continue

        files_to_process: list[Path] = []
        total_assigned = 0
        for roi in scope_rois:
            roi_files = _collect_round_tiles(
                path,
                round_token,
                rois=[roi],
                ref_round=ref_round,
                max_idx=max_idx,
            )
            if not roi_files:
                logger.warning(f"{prefix}No files found for ROI '{roi}'; skipping.")
                continue

            if limit is not None:
                if limit_scope == "per_roi":
                    limited = roi_files[:limit]
                    if len(limited) < len(roi_files):
                        logger.info(
                            f"Applying limit={limit} to ROI '{roi}': processing {len(limited)} of {len(roi_files)} tile(s)."
                        )
                    roi_files = limited
                else:
                    remaining = limit - total_assigned
                    if remaining <= 0:
                        break
                    if len(roi_files) > remaining:
                        logger.info(
                            f"Applying limit={limit}: processing {remaining} tile(s) from ROI '{roi}'."
                        )
                        roi_files = roi_files[:remaining]
                    total_assigned += len(roi_files)

            files_to_process.extend(roi_files)

            if limit_scope == "total" and limit is not None and total_assigned >= limit:
                break

        if not files_to_process:
            logger.warning(f"{prefix}No files discovered for requested ROI scope; skipping round.")
            continue

        plan = _prepare_round_plan(
            path=path,
            round_name=round_token,
            files=files_to_process,
            out_dir=out_dir,
            basic_name=basic_name,  # None triggers round→all fallback in resolver
            n_fids=n_fids,
            histogram_bins=histogram_bins,
            overwrite=overwrite,
            debug=debug,
            label=round_token,
            mode=mode,
            min_perc=min_perc,
            max_perc=max_perc,
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
                except OSError as exc:
                    raise click.ClickException(
                        f"{plan.label or 'run'}: failed to delete origin directories: {exc}"
                    ) from exc
        return []

    failures: list[WorkerMessage] = []

    def _execute(
        plan: _RoundProcessingPlan,
        progress_callback: ProgressReporter | ProgressCallback | None,
    ) -> None:
        reporter = _ensure_reporter(progress_callback)
        plan_failures = _execute_round_plan(
            plan,
            devices=devices,
            stop_on_error=stop_on_error,
            debug=debug,
            progress=reporter,
        )
        failures.extend(plan_failures)
        if delete_origin:
            try:
                safe_delete_origin_dirs(plan.files, out_dir)
            except OSError as exc:
                raise click.ClickException(
                    f"{plan.label or 'run'}: failed to delete origin directories: {exc}"
                ) from exc

    base_reporter = _ensure_reporter(progress)

    if base_reporter is None:
        with progress_reporter(total_pending) as shared_reporter:
            for plan in plans:
                _execute(plan, shared_reporter)
    else:
        for plan in plans:
            _execute(plan, base_reporter)

    return failures


# ------------------------------ Programmatic helper ------------------------------ #


def multi_run(
    path: Path,
    round_name: str,
    *,
    ref: str | Path | None,
    limit: int | None,
    mode: str = _PREPARE_DEFAULT_MODE.value,
    histogram_bins: int = 8192,
    skip_quantized: bool = False,
    overwrite: bool,
    n_fids: int,
    basic_name: str | None,
    debug: bool,
    devices: Sequence[int],
    stop_on_error: bool,
    min_perc: float = LEGACY_PERCENTILES[0],
    max_perc: float = LEGACY_PERCENTILES[1],
    configure_logging: bool = False,
    process_label: str = "0",
) -> list[WorkerMessage]:
    """Run multi-GPU deconvolution for a single round programmatically."""

    if configure_logging:
        _configure_logging(debug, process_label=process_label)

    selected_mode = _normalize_mode(mode)
    if min_perc >= max_perc:
        raise click.ClickException("--min-perc must be < --max-perc.")

    if skip_quantized and selected_mode is DeconvolutionOutputMode.U16:
        logger.info("multi_run: skip_quantized requested; switching mode to float32 outputs.")
        selected_mode = DeconvolutionOutputMode.F32

    load_scaling = selected_mode is DeconvolutionOutputMode.U16 and not skip_quantized

    ref_round = str(ref) if ref is not None else None
    if ref_round is not None:
        all_rounds = Workspace.discover_rounds(path)
        if ref_round not in all_rounds:
            raise click.ClickException(f"Reference round '{ref_round}' not found in {path}.")

    failures = _plan_and_execute(
        path=path,
        rounds=[round_name],
        rois=None,
        ref_round=ref_round,
        limit=limit,
        limit_scope="total",
        max_idx=None,
        basic_name=basic_name,
        n_fids=n_fids,
        histogram_bins=histogram_bins,
        load_scaling=load_scaling,
        overwrite=overwrite,
        debug=debug,
        devices=list(devices),
        stop_on_error=stop_on_error,
        mode=selected_mode,
        delete_origin=False,
        min_perc=min_perc,
        max_perc=max_perc,
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
    mode: str = _DEFAULT_OUTPUT_MODE.value,
    overwrite: bool,
    n_fids: int,
    basic_name: str | None,
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

    selected_mode = _normalize_mode(mode)

    for round_name, files in sorted(by_round.items()):
        if not files:
            continue

        logger.info(f"[{round_name}] Selected {len(files)} sampled tile(s) (pre-filter).")

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
            mode=selected_mode,
        )


# ---------- prepare ----------


@deconvnew.command()
@click.argument("path", type=click.Path(path_type=Path))
# Accept zero or more round names as a positional argument.
@click.argument("rounds", nargs=-1)
@click.option("-n", "num_tiles", type=int, default=100, show_default=True)
@click.option("--percent", type=float, default=0.1, show_default=True)
@click.option("--roi", type=str, multiple=True)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--histogram-bins", type=int, default=8192, show_default=True)
@click.option("--overwrite", is_flag=True)
@click.option("--n-fids", type=int, default=2, show_default=True)
@click.option(
    "--basic-name",
    type=str,
    default=None,
    show_default=False,
    help=("BaSiC profile prefix. If omitted, uses round-specific prefix first then falls back to 'all'."),
)
@click.option("--debug", is_flag=True)
@click.option("--devices", type=str, default="auto", show_default=True, callback=_devices_callback)
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
    overwrite: bool,
    n_fids: int,
    basic_name: str | None,
    debug: bool,
    devices: list[int],
    stop_on_error: bool,
) -> None:
    """Sample tiles and emit artifacts using multi-GPU workers."""
    rounds_tag = "-".join(rounds) if rounds else "all"
    roi_tag = "-".join(roi) if roi else "all"
    setup_cli_logging(
        path,
        component="preprocess.deconv.prepare",
        file=f"prepare-{rounds_tag}",
        debug=debug,
        extra={"rounds": rounds_tag, "roi": roi_tag},
    )

    multi_prepare(
        path,
        num_tiles=num_tiles,
        percent=percent,
        roi=roi,
        round_names=tuple(rounds),
        seed=seed,
        histogram_bins=histogram_bins,
        mode=_PREPARE_DEFAULT_MODE.value,
        overwrite=overwrite,
        n_fids=n_fids,
        basic_name=basic_name,
        debug=debug,
        devices=devices,
        stop_on_error=stop_on_error,
        configure_logging=True,
        process_label="0",
    )


# ---------- run (progressive scoping) ----------


@deconvnew.command()
@click.argument("path", type=click.Path(path_type=Path))
@click.argument("round_name", type=str, required=False)
@click.option("--roi", "roi_name", type=str, default="*")
@click.option("--ref", "ref_round", type=str, default=None)
@click.option(
    "--max-idx",
    type=click.IntRange(min=0),
    default=None,
    show_default=False,
    help="Only process tiles with index <= MAX_IDX (parsed from filenames like '<round>-0007.tif').",
)
@click.option("--limit", type=int, default=None)
@click.option(
    "--mode",
    type=click.Choice(["u16", "float32", "legacy"]),
    default=_DEFAULT_OUTPUT_MODE.value,
    show_default=True,
)
@click.option("--histogram-bins", type=int, default=8192, show_default=True)
@click.option("--overwrite", is_flag=True)
@click.option("--delete-origin/--no-delete-origin", default=True, show_default=True)
@click.option("--n-fids", type=int, default=2, show_default=True)
@click.option(
    "--min-perc",
    type=click.FloatRange(min=0.0, max=100.0),
    default=LEGACY_PERCENTILES[0],
    show_default=True,
    help="Lower percentile used by legacy per-tile quantization (mode=legacy).",
)
@click.option(
    "--max-perc",
    type=click.FloatRange(min=0.0, max=100.0),
    default=LEGACY_PERCENTILES[1],
    show_default=True,
    help="Upper percentile used by legacy per-tile quantization (mode=legacy).",
)
@click.option(
    "--basic-name",
    type=str,
    default=None,
    show_default=False,
    help=("BaSiC profile prefix. If omitted, uses round-specific prefix first then falls back to 'all'."),
)
@click.option("--debug", is_flag=True)
@click.option("--devices", type=str, default="auto", show_default=True, callback=_devices_callback)
@click.option(
    "--stop-on-error/--continue-on-error",
    default=True,
    show_default=True,
    help="Stop all workers after the first failure.",
)
@click.option(
    "--skip-quantized/--include-quantized",
    default=False,
    show_default=True,
    help="Skip writing uint16 deliverables so quantize can run separately.",
)
@click.option("--skip-non-bit", is_flag=True)
def run(
    path: Path,
    round_name: str | None,
    *,
    roi_name: str,
    ref_round: str | None,
    max_idx: int | None = None,
    limit: int | None,
    mode: str = _DEFAULT_OUTPUT_MODE.value,
    histogram_bins: int,
    overwrite: bool,
    delete_origin: bool,
    n_fids: int,
    min_perc: float = LEGACY_PERCENTILES[0],
    max_perc: float = LEGACY_PERCENTILES[1],
    basic_name: str | None,
    debug: bool,
    devices: list[int],
    stop_on_error: bool,
    skip_quantized: bool,
    skip_non_bit: bool = False,
) -> None:
    """Run multi-GPU deconvolution across selected rounds and ROIs."""
    round_tag = round_name or "all"
    setup_cli_logging(
        path,
        component="preprocess.deconv.run",
        file=f"run-{round_tag}",
        debug=debug,
        extra={"round": round_tag, "roi": roi_name, "mode": mode},
    )
    _configure_logging(debug, process_label="0")

    workspace = Workspace(path)
    all_rounds = Workspace.discover_rounds(path)
    all_rois = workspace.rois

    if not all_rounds:
        raise click.ClickException("No rounds discovered in workspace.")

    if not all_rois:
        raise click.ClickException("No ROIs found in workspace.")

    if round_name is None or round_name == "*":
        if not skip_non_bit:
            selected_rounds = tuple(all_rounds)
        else:
            # Keep only bit-coded rounds (exactly three numeric tokens separated by underscores)
            selected_rounds = tuple(
                r for r in all_rounds if (all(x.isdigit() for x in r.split("_")) and (len(r.split("_")) == 3))
            )
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

    selected_mode = _normalize_mode(mode)
    if min_perc >= max_perc:
        raise click.ClickException("--min-perc must be < --max-perc.")

    if skip_quantized:
        if selected_mode is DeconvolutionOutputMode.U16:
            logger.info(
                "Skipping quantized deliverables; forcing float32 mode so "
                "'preprocess deconv quantize' can run independently."
            )
            selected_mode = DeconvolutionOutputMode.F32
        else:
            logger.info("Skipping quantized deliverables; float32 mode already active.")

    load_scaling = selected_mode is DeconvolutionOutputMode.U16 and not skip_quantized

    ref_token = ref_round
    if ref_token is not None and ref_token not in all_rounds:
        raise click.ClickException(f"Reference round '{ref_token}' not found in {path}.")

    if max_idx is not None:
        ctx = click.get_current_context(silent=True)
        if delete_origin:
            source = None if ctx is None else ctx.get_parameter_source("delete_origin")
            if source is None or source != ParameterSource.DEFAULT:
                raise click.ClickException(
                    "Refusing to delete origin directories with --max-idx: tile selection is partial, "
                    "and deletion occurs per round/ROI directory."
                )
            logger.info("--max-idx selected; forcing --no-delete-origin to avoid deleting partial directories.")
        delete_origin = False

    _plan_and_execute(
        path=path,
        rounds=selected_rounds,
        rois=selected_rois,
        ref_round=ref_token,
        limit=limit,
        limit_scope="per_roi",
        max_idx=max_idx,
        basic_name=basic_name,
        n_fids=n_fids,
        histogram_bins=histogram_bins,
        load_scaling=load_scaling,
        overwrite=overwrite,
        debug=debug,
        devices=devices,
        stop_on_error=stop_on_error,
        mode=selected_mode,
        delete_origin=delete_origin,
        min_perc=min_perc,
        max_perc=max_perc,
    )


@deconvnew.command()
@click.argument("path", type=click.Path(path_type=Path))
@click.argument("round_name", type=str, required=False)
@click.option("--roi", "roi_name", type=str, default="*")
@click.option("--ref", "ref_round", type=str, default=None)
@click.option(
    "--max-idx",
    type=click.IntRange(min=0),
    default=None,
    show_default=False,
    help="Only process tiles with index <= MAX_IDX (parsed from filenames like '<round>-0007.tif').",
)
@click.option("--limit", type=int, default=None)
@click.option(
    "--mode",
    type=click.Choice(["u16", "float32", "legacy"]),
    default=_DEFAULT_OUTPUT_MODE.value,
    show_default=True,
)
@click.option("--histogram-bins", type=int, default=8192, show_default=True)
@click.option("--overwrite", is_flag=True)
@click.option("--delete-origin/--no-delete-origin", default=True, show_default=True)
@click.option("--n-fids", type=int, default=2, show_default=True)
@click.option(
    "--min-perc",
    type=click.FloatRange(min=0.0, max=100.0),
    default=LEGACY_PERCENTILES[0],
    show_default=True,
    help="Lower percentile used by legacy per-tile quantization (mode=legacy).",
)
@click.option(
    "--max-perc",
    type=click.FloatRange(min=0.0, max=100.0),
    default=LEGACY_PERCENTILES[1],
    show_default=True,
    help="Upper percentile used by legacy per-tile quantization (mode=legacy).",
)
@click.option(
    "--basic-name",
    type=str,
    default=None,
    show_default=False,
    help=("BaSiC profile prefix. If omitted, uses round-specific prefix first then falls back to 'all'."),
)
@click.option("--debug", is_flag=True)
@click.option("--devices", type=str, default="auto", show_default=True, callback=_devices_callback)
@click.option(
    "--stop-on-error/--continue-on-error",
    default=True,
    show_default=True,
    help="Stop all workers after the first failure.",
)
@click.option(
    "--skip-quantized/--include-quantized",
    default=False,
    show_default=True,
    help="Skip writing uint16 deliverables so quantize can run separately.",
)
@click.option("--skip-non-bit", is_flag=True)
def batch(
    path: Path,
    round_name: str | None,
    *,
    roi_name: str,
    ref_round: str | None,
    max_idx: int | None = None,
    limit: int | None,
    mode: str,
    histogram_bins: int,
    overwrite: bool,
    delete_origin: bool,
    n_fids: int,
    min_perc: float = LEGACY_PERCENTILES[0],
    max_perc: float = LEGACY_PERCENTILES[1],
    basic_name: str | None,
    debug: bool,
    devices: list[int],
    stop_on_error: bool,
    skip_quantized: bool,
    skip_non_bit: bool = False,
) -> None:
    run.callback(
        path,
        round_name,
        roi_name=roi_name,
        ref_round=ref_round,
        max_idx=max_idx,
        limit=limit,
        mode=mode,
        histogram_bins=histogram_bins,
        overwrite=overwrite,
        delete_origin=delete_origin,
        n_fids=n_fids,
        min_perc=min_perc,
        max_perc=max_perc,
        basic_name=basic_name,
        debug=debug,
        devices=devices,
        stop_on_error=stop_on_error,
        skip_quantized=skip_quantized,
        skip_non_bit=skip_non_bit,
    )


@deconvnew.command()
@click.argument("path", type=click.Path(path_type=Path))
@click.argument("round_name", type=str, required=False)
def easy(path: Path, round_name: str | None) -> None:
    """One-shot wrapper that ensures scaling exists, then runs quantize + deconvolution.

    Note: this invokes ``deconvnew quantize``, which deletes verified float32 inputs under
    ``analysis/deconv32/{round}--{roi}`` after quantization.
    """
    round_tag = round_name or "all"
    setup_cli_logging(
        path,
        component="preprocess.deconv.easy",
        file=f"easy-{round_tag}",
        debug=False,
        extra={"round": round_tag},
    )
    import subprocess

    rounds = [round_name] if round_name else Workspace.discover_rounds(path)
    ws = Workspace(path)
    for round_ in rounds:
        if not ws.deconv_scaling(round_).exists():
            subprocess.run(["preprocess", "deconvnew", "prepare", str(path), round_], check=True)
            subprocess.run(["preprocess", "deconvnew", "precompute", str(path), round_], check=True)

        with ThreadPoolExecutor() as executor:
            # Run quantize alongside deconv to avoid re-deconvolving tiles just to obtain uint16 deliverables.
            promises = [
                executor.submit(
                    subprocess.run,
                    ["preprocess", "deconvnew", "quantize", str(path), round_],
                    check=True,
                ),
                executor.submit(
                    subprocess.run,
                    ["preprocess", "deconvnew", "run", "--mode=u16", str(path), round_],
                    check=True,
                ),
            ]
            for fut in as_completed(promises):
                fut.result()


if __name__ == "__main__":
    deconvnew()
