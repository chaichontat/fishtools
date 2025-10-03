"""Multi-GPU dispatcher scaffolding for deconvolution workflows.

This module provides queue-managed worker orchestration that allows the
existing single-GPU deconvolution codepath to remain untouched while enabling
future dynamic scheduling across multiple CUDA devices. Each worker process
pipelines read/compute/write so the GPU is continuously fed; faster GPUs pull
more tiles without static sharding.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import rich_click as click
from loguru import logger

from fishtools.preprocess.deconv import backend as _deconv_backend
from fishtools.preprocess.deconv.basic_utils import resolve_basic_paths
from fishtools.preprocess.deconv.discovery import infer_psf_step
from fishtools.preprocess.deconv.logging_utils import configure_logging
from fishtools.preprocess.deconv.normalize import load_global_scaling
from fishtools.preprocess.deconv.worker import (
    DEFAULT_QUEUE_DEPTH,
    WorkerMessage,
    parse_device_spec,
    run_multi_gpu,
)
from fishtools.utils.io import Workspace, get_channels
from fishtools.utils.pretty_print import progress_bar

ProcessorConfig = _deconv_backend.ProcessorConfig
TileProcessor = _deconv_backend.TileProcessor
ProcessorFactory = _deconv_backend.ProcessorFactory
DeconvolutionTileProcessor = _deconv_backend.DeconvolutionTileProcessor
make_processor_factory = _deconv_backend.make_processor_factory


def _configure_logging(debug: bool, *, process_label: str) -> None:
    """Thin wrapper to shared logger config preserving process labels."""
    configure_logging(debug, process_label=process_label)


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


def _run_round_tiles(
    *,
    path: Path,
    round_name: str,
    files: Sequence[Path],
    out_dir: Path,
    basic_name: str | None,
    n_fids: int,
    save_float32: bool,
    save_histograms: bool,
    histogram_bins: int,
    load_scaling: bool,
    overwrite: bool,
    debug: bool,
    devices: Sequence[int],
    queue_depth: int,
    stop_on_error: bool,
    label: str | None,
    psf_source: str,
    pre_filter_log: Callable[[int], str] | None,
    pending_log: Callable[[int, int], str],
    skip_log: Callable[[], str],
    failure_label: str,
    prefix_pending: bool = True,
    prefix_skip: bool = True,
) -> list[WorkerMessage]:
    """Shared round-level execution for multi-GPU deconvolution flows."""
    file_list = list(files)
    if not file_list:
        return []

    def _info(message: str, *, prefix: bool = True) -> None:
        if prefix and label:
            logger.info(f"[{label}] {message}")
        else:
            logger.info(message)

    def _warn(message: str, *, prefix: bool = True) -> None:
        if prefix and label:
            logger.warning(f"[{label}] {message}")
        else:
            logger.warning(message)

    def _error(message: str) -> None:
        if label:
            logger.error(f"[{label}] {message}")
        else:
            logger.error(message)

    if pre_filter_log is not None:
        _info(pre_filter_log(len(file_list)))

    step, inferred = infer_psf_step(file_list[0])
    if inferred:
        _info(f"Using PSF step={step} from {psf_source}.")
    else:
        _warn(f"Could not determine PSF step; defaulting to step={step}.")

    channels = get_channels(file_list[0])
    basic_paths = resolve_basic_paths(path, round_name=round_name, channels=channels, basic_name=basic_name)

    if load_scaling:
        m_glob, s_glob = load_global_scaling(path, round_name)
    else:
        m_glob = s_glob = None

    pending = filter_pending_files(
        file_list,
        out_dir=out_dir,
        overwrite=overwrite,
        save_float32=save_float32,
        save_histograms=save_histograms,
    )

    if not pending:
        _info(skip_log(), prefix=prefix_skip)
        return []

    _info(pending_log(len(pending), len(file_list)), prefix=prefix_pending)

    config = ProcessorConfig(
        round_name=round_name,
        basic_paths=basic_paths,
        output_dir=out_dir,
        n_fids=n_fids,
        step=step,
        save_float32=save_float32,
        save_histograms=save_histograms,
        histogram_bins=histogram_bins,
        m_glob=m_glob,
        s_glob=s_glob,
        debug=debug,
    )

    processor_factory = make_processor_factory(config)
    depth = queue_depth if queue_depth > 0 else DEFAULT_QUEUE_DEPTH

    failures: list[WorkerMessage]
    with progress_bar(len(pending)) as update:

        def callback(message: WorkerMessage) -> None:
            if message.status == "ok":
                update()
            elif message.status == "error":
                _error(f"Failed to process {message.path}: {message.error}")

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
            raise RuntimeError(f"{failure_label}: processing aborted due to failures: {details}")
        _warn(f"Completed with failures: {details}")

    return failures


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

    rounds_all = Workspace.discover_rounds(path)
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

        _run_round_tiles(
            path=path,
            round_name=round_name,
            files=files,
            out_dir=out_dir,
            basic_name=basic_name,
            n_fids=n_fids,
            save_float32=True,
            save_histograms=True,
            histogram_bins=histogram_bins,
            load_scaling=False,
            overwrite=overwrite,
            debug=debug,
            devices=devices,
            queue_depth=queue_depth,
            stop_on_error=stop_on_error,
            label=round_name,
            psf_source="waveform metadata",
            pre_filter_log=lambda total: f"Selected {total} sampled tile(s) (pre-filter).",
            pending_log=lambda pending,
            total: f"{pending}/{total} tile(s) pending after overwrite/histogram checks.",
            skip_log=lambda: "No tiles require preparation after filtering.",
            failure_label=round_name,
        )


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

    rounds = Workspace.discover_rounds(path)
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

        _run_round_tiles(
            path=path,
            round_name=round_name,
            files=files,
            out_dir=out_dir,
            basic_name=basic_name,
            n_fids=n_fids,
            save_float32=False,
            save_histograms=False,
            histogram_bins=8192,
            load_scaling=True,
            overwrite=overwrite,
            debug=debug,
            devices=devices,
            queue_depth=queue_depth,
            stop_on_error=stop_on_error,
            label=round_name,
            psf_source="metadata",
            pre_filter_log=lambda total: f"{total} candidate tile(s) before filtering.",
            pending_log=lambda pending, total: f"{pending}/{total} tile(s) pending after overwrite checks.",
            skip_log=lambda: "All tiles already processed; skipping.",
            failure_label=round_name,
        )

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

    _run_round_tiles(
        path=path,
        round_name=name,
        files=files,
        out_dir=out_dir,
        basic_name=basic_token,
        n_fids=n_fids,
        save_float32=False,
        save_histograms=False,
        histogram_bins=8192,
        load_scaling=True,
        overwrite=overwrite,
        debug=debug,
        devices=devices,
        queue_depth=queue_depth,
        stop_on_error=stop_on_error,
        label=name,
        psf_source="metadata",
        pre_filter_log=None,
        pending_log=lambda pending, total: f"{pending}/{total} tile(s) pending after overwrite checks.",
        skip_log=lambda: "All tiles already processed; nothing to do.",
        failure_label=name,
        prefix_pending=False,
        prefix_skip=False,
    )


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
    "filter_pending_files",
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
