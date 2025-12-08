from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Literal

import click
import numpy as np
import tifffile
import zarr
from loguru import logger
from scipy.ndimage import zoom
from tifffile import imread

from fishtools.io.workspace import Workspace
from fishtools.preprocess.segmentation import unsharp_all
from fishtools.segment.extract_helpers import (
    DEFAULT_CROP_SIZE,
    FILE_NAME,
    MAX_WIDTH_AFTER_UPSCALE,
    ZARR_TILE_SIZE,
    ZERO_PIXEL_SKIP_THRESHOLD,
    ExtractionConfig,
    MaskLike,
    SliceJob,
    TileJob,
    Volume,
    _compute_perpendicular_slice,
    _compute_tile_origins,
    _default_output_dir,
    _distribute_file_budget,
    _expand_positions_with_context,
    _format_size,
    _format_tile_filename,
    _mask_filename,
    _normalize_reporter,
    _path_size_cached,
    _prefix_with_roi,
    _resize_mask,
    _resolve_enrich_mask,
    _resolve_output_names,
    _sample_positions,
    _score_and_select_tiles,
    _select_high_diversity_positions,
    _squeeze_mask,
    _validate_max_from_path,
    _write_mask_tiff,
    _write_tiff,
    load_roi_points,
)
from fishtools.utils.pretty_print import (
    ProgressReporter,
    TaskCancelledException,
    get_cancel_event,
    progress_bar_threadpool,
    progress_reporter,
)

"""Extract utilities for the segment CLI.

This module provides the core extraction logic. The top-level `segment` CLI
in `__init__.py` exposes these functions as Click commands.
"""


@dataclass
class ExtractionContext:
    """Precomputed per-file state for extraction operations."""

    file: Path
    roi: str
    vol: Volume
    channel_names: list[str] | None
    selected_indices: list[int]
    out_names: list[str]
    other_vol: Volume | None
    mask_vol: MaskLike | None
    enrich_mask_vol: MaskLike | None
    upscale: float
    anisotropy: int
    out_dir: Path


def resolve_file_mask_path(file: Path, explicit_mask: Path | None) -> Path | None:
    """Resolve mask path for a single input file."""

    if explicit_mask is not None:
        return explicit_mask
    return _resolve_mask_path(file)


def open_and_validate_mask(mask_path: Path | None, vol: Volume, *, label: str) -> MaskLike | None:
    """Open a mask volume and validate that its shape matches the input volume.

    For TIFF/Zarr inputs we expect:
    - volume: (Z, Y, X, C)
    - mask:   (Z, Y, X)  or (Z, 1, Y, X) which is squeezed to (Z, Y, X)

    Any mismatch in Z, Y, or X dimensions is treated as an error so that
    downstream slices/tiles are guaranteed to be spatially aligned.
    """

    if mask_path is None:
        return None

    mask_vol = _open_mask_volume(mask_path)
    if mask_vol.shape[0] != vol.shape[0]:
        raise ValueError(
            f"Mask volume {mask_path} Z-dimension ({mask_vol.shape[0]}) does not match volume ({vol.shape[0]}) for {label}"
        )
    # Spatial YX dimensions must also agree; we do not attempt to resample.
    if mask_vol.shape[1] != vol.shape[1] or mask_vol.shape[2] != vol.shape[2]:
        raise ValueError(
            f"Mask volume {mask_path} spatial dimensions {mask_vol.shape[1:]} "
            f"do not match volume ({vol.shape[1]}, {vol.shape[2]}) for {label}"
        )

    logger.info(f"[{label}] Using mask: {mask_path}")
    return mask_vol


def build_extraction_context(
    file: Path,
    roi: str,
    *,
    channels: str | None,
    max_from_path: Path | None,
    mask_path: Path | None,
    enrich_path: Path | None,
    upscale: float,
    anisotropy: int,
    out_dir: Path,
) -> ExtractionContext:
    vol, channel_names = _open_volume(file)
    selected_indices = _parse_channels(channels, channel_names, vol.shape[-1])
    _ensure_channel_bounds(selected_indices, vol.shape[-1], label=file.name)
    base_names = _resolve_output_names(selected_indices, channel_names, channels)
    other_vol = _resolve_other_volume(file, max_from_path)
    out_names = [*base_names, "max_from"] if other_vol is not None else base_names
    mask_vol = open_and_validate_mask(mask_path, vol, label=file.name)
    enrich_mask_vol = open_and_validate_mask(enrich_path, vol, label=f"{file.name} (enrich)")

    return ExtractionContext(
        file=file,
        roi=roi,
        vol=vol,
        channel_names=channel_names,
        selected_indices=selected_indices,
        out_names=out_names,
        other_vol=other_vol,
        mask_vol=mask_vol,
        enrich_mask_vol=enrich_mask_vol,
        upscale=upscale,
        anisotropy=anisotropy,
        out_dir=out_dir,
    )


def write_slab_with_mask(
    *,
    ctx: ExtractionContext,
    slab: np.ndarray,
    other_max: np.ndarray | None,
    mask_slice: np.ndarray | None,
    out_file: Path,
    axes: str,
    resize_factors: tuple[float, ...],
    channels_arg: str | None,
) -> None:
    """Shared slab processing and writing logic for extracted slices."""

    processed = _prep_slab(
        slab,
        ch_idx=ctx.selected_indices,
        channel_axis=slab.ndim - 1,
        crop_slices=None,
        filter_before=False,
        append_max=other_max,
        apply_filter=False,
    )
    processed = _resize_uint16(processed, resize_factors)
    _write_tiff(
        out_file,
        processed,
        axes=axes,
        names=ctx.out_names,
        channels_arg=channels_arg,
        upscale=ctx.upscale,
    )

    if mask_slice is not None:
        if mask_slice.ndim != 2:
            raise ValueError("Mask extraction expected 2D data.")
        mask_factors = resize_factors[1:] if len(resize_factors) > 2 else resize_factors
        resized_mask = _resize_mask(mask_slice, mask_factors)
        _write_mask_tiff(out_file.parent / _mask_filename(out_file.name), resized_mask, axes=axes[1:])


def _read_channel_names(file: Path) -> list[str] | None:
    """Extract channel names from TIFF metadata, returning None if unavailable."""
    try:
        with tifffile.TiffFile(file) as tif:
            shaped = getattr(tif, "shaped_metadata", None)
            if isinstance(shaped, (list, tuple)) and shaped:
                md = shaped[0]
                if isinstance(md, dict):
                    names = md.get("key")
                    if isinstance(names, (list, tuple)):
                        return [str(x) for x in names]
    except (OSError, ValueError, KeyError, tifffile.TiffFileError) as exc:
        logger.debug(f"Failed to read channel names from {file}: {exc}")
    return None


def _normalize_channel_names(names: object) -> list[str] | None:
    if isinstance(names, (list, tuple)):
        return [str(n) for n in names]
    return None


def _read_channel_names_from_zarr(store: Path) -> list[str] | None:
    try:
        arr = zarr.open_array(store, mode="r")
    except (OSError, ValueError, Exception) as exc:  # pragma: no cover - defensive
        logger.debug(f"Failed to open Zarr store {store} for channel names: {exc}")
        return None
    raw_key = arr.attrs.get("key")
    return _normalize_channel_names(raw_key)


def _is_zarr_path(file: Path) -> bool:
    return file.suffix == ".zarr" or (file.is_dir() and file.name.endswith(".zarr"))


def _open_zarr_array(file: Path) -> zarr.Array:
    return zarr.open_array(file, mode="r")


def _open_volume(file: Path) -> tuple[Volume, list[str] | None]:
    """Open a registered volume as an array-like object with shape (Z,Y,X,C).

    - For TIFF: load and reorder (Z,C,Y,X) -> (Z,Y,X,C)
    - For Zarr: open array (no materialization)
    Returns: (volume, channel_names)
    """
    if _is_zarr_path(file):
        arr = _open_zarr_array(file)
        if arr.ndim != 4:
            raise ValueError("Expected fused Zarr to be 4D (Z,Y,X,C).")
        return arr, _read_channel_names_from_zarr(file)
    # TIFF path
    img = imread(file)
    if img.ndim != 4:
        raise ValueError("Expected registered TIFF to be 4D (Z,C,Y,X).")
    vol = np.moveaxis(img, 1, -1)  # (Z,Y,X,C)
    return vol, _read_channel_names(file)


def _resolve_mask_path(file: Path) -> Path | None:
    """Return the expected mask path for a given registered input, if it exists."""
    candidate: Path | None = None
    if _is_zarr_path(file):
        candidate = file.parent / f"{file.stem}_masks.zarr"
    elif file.suffix.lower() in {".tif", ".tiff"}:
        candidate = file.with_name(f"{file.stem}_masks{file.suffix}")
    if candidate is not None and candidate.exists():
        return candidate
    return None


def _open_mask_volume(mask_path: Path) -> MaskLike:
    """Load a mask volume produced alongside a registered stack."""
    if _is_zarr_path(mask_path):
        arr = _open_zarr_array(mask_path)
        if arr.ndim not in (3, 4):
            raise ValueError("Mask Zarr is expected to be 3D (Z,Y,X) or 4D (Z,1,Y,X).")
        return arr
    mask = imread(mask_path)
    if mask.ndim == 4 and mask.shape[1] == 1:
        mask = np.squeeze(mask, axis=1)
    if mask.ndim == 2:
        mask = mask[np.newaxis, ...]
    if mask.ndim != 3:
        raise ValueError("Mask TIFF is expected to be 3D (Z,Y,X).")
    return mask


def _process_ortho_slice(
    *,
    vol: Volume,
    mask_vol: MaskLike | None,
    other_vol: Volume | None,
    position: int,
    axis: Literal["y", "x"],
    perpendicular_slice: slice,
    selected_indices: list[int],
    out_names: list[str],
    anisotropy: int,
    upscale: float,
    out_dir: Path,
    file_stem: str,
    roi: str,
    channels: str | None,
) -> None:
    """Process a single ortho slice (Y or X) and write output files."""
    # Ensure output directory exists (safety check for parallel execution)
    out_dir.mkdir(parents=True, exist_ok=True)

    perp_start = perpendicular_slice.start or 0
    if axis == "y":
        slab = vol[:, position, perpendicular_slice, :]
        other_max = (
            other_vol[:, position, perpendicular_slice, :].max(axis=2) if other_vol is not None else None
        )
        axes_str = "CZX"
        out_name = _prefix_with_roi(f"{file_stem}_orthozx-x{perp_start}-y{position}.tif", roi)
        mask_slab = (
            _squeeze_mask(mask_vol[:, position, perpendicular_slice]) if mask_vol is not None else None
        )
    else:  # axis == "x"
        slab = vol[:, perpendicular_slice, position, :]
        other_max = (
            other_vol[:, perpendicular_slice, position, :].max(axis=2) if other_vol is not None else None
        )
        axes_str = "CZY"
        out_name = _prefix_with_roi(f"{file_stem}_orthozy-y{perp_start}-x{position}.tif", roi)
        mask_slab = (
            _squeeze_mask(mask_vol[:, perpendicular_slice, position]) if mask_vol is not None else None
        )

    processed = _prep_slab(
        slab,
        ch_idx=selected_indices,
        channel_axis=2,
        crop_slices=None,
        filter_before=False,
        append_max=other_max,
        apply_filter=False,
    )
    processed = _resize_uint16(processed, (1.0, anisotropy * upscale, upscale))
    _write_tiff(
        out_dir / out_name,
        processed,
        axes=axes_str,
        names=out_names,
        channels_arg=channels,
        upscale=upscale,
    )

    if mask_slab is not None:
        if mask_slab.ndim != 2:
            raise ValueError(f"Mask {axis.upper()} extraction expected 2D data.")
        resized_mask = _resize_mask(mask_slab, (anisotropy * upscale, upscale))
        _write_mask_tiff(out_dir / _mask_filename(out_name), resized_mask, axes=axes_str[1:])


def _execute_extraction(
    *,
    label: str,
    files: list[Path],
    mode: str,
    out_dir: Path,
    channels: str | None,
    crop: int,
    dz: int,
    n: int,
    anisotropy: int,
    threads: int,
    upscale: float,
    seed: int | None,
    max_from_path: Path | None,
    explicit_mask_path: Path | None = None,
    enrich_boundaries: Path | None = None,
    roi_points: Path | None = None,
) -> None:
    config = ExtractionConfig(
        mode=mode,
        channels=channels,
        crop=crop,
        dz=dz,
        n=n,
        anisotropy=anisotropy,
        upscale=upscale,
        seed=seed,
        threads=threads,
    )

    files = files[:config.n]
    logger.info(f"[{label}] Using {len(files)} registered {'file' if len(files) == 1 else 'files'}")
    logger.info(f"[{label}] Max-from: {max_from_path if max_from_path else 'None'}")
    if roi_points is not None:
        logger.info(f"[{label}] Using ROI points from: {roi_points}")

    all_zarr = bool(files and all(_is_zarr_path(f) for f in files))
    if all_zarr and config.mode == "z":
        _execute_zarr_z_extraction(
            label=label,
            files=files,
            config=config,
            out_dir=out_dir,
            max_from_path=max_from_path,
            explicit_mask_path=explicit_mask_path,
            enrich_boundaries=enrich_boundaries,
            roi_points=roi_points,
        )
        return

    if all_zarr and config.mode == "ortho":
        _execute_zarr_ortho_extraction(
            label=label,
            files=files,
            config=config,
            out_dir=out_dir,
            max_from_path=max_from_path,
            explicit_mask_path=explicit_mask_path,
            enrich_boundaries=enrich_boundaries,
            roi_points=roi_points,
        )
        return

    _execute_tiff_extraction(
        label=label,
        files=files,
        config=config,
        out_dir=out_dir,
        max_from_path=max_from_path,
        explicit_mask_path=explicit_mask_path,
        enrich_boundaries=enrich_boundaries,
        roi_points=roi_points,
    )


def _execute_zarr_z_extraction(
    *,
    label: str,
    files: list[Path],
    config: ExtractionConfig,
    out_dir: Path,
    max_from_path: Path | None,
    explicit_mask_path: Path | None,
    enrich_boundaries: Path | None,
    roi_points: Path | None = None,
) -> None:
    enrich_mask_vol: MaskLike | None = None
    if enrich_boundaries is not None:
        enrich_mask_vol = _open_mask_volume(enrich_boundaries)
        logger.info(f"[{label}] Using enrichment mask for tile selection: {enrich_boundaries}")

    # Load ROI points if provided (coordinates are upscaled 2x inside load_roi_points)
    point_coords: list[tuple[int, int]] | None = None
    if roi_points is not None:
        point_coords = load_roi_points(roi_points)
        logger.info(f"[{label}] Loaded {len(point_coords)} points from ROI file for Z extraction")

    tile_jobs: list[TileJob] = []
    total_outputs = 0

    for f in files:
        vol, names_all = _open_volume(f)
        mask_path = resolve_file_mask_path(f, explicit_mask_path)
        mask_vol = open_and_validate_mask(mask_path, vol, label=f"{label}:{f.name}")

        if point_coords is not None:
            # Convert point centers to tile origins (top-left corner)
            tile_half = ZARR_TILE_SIZE // 2
            tile_origins: list[tuple[int, int]] = []
            for px, py in point_coords:
                # Clamp to valid range
                y0 = max(config.crop, py - tile_half)
                x0 = max(config.crop, px - tile_half)
                # Ensure tile fits within bounds
                y0 = min(y0, vol.shape[1] - config.crop - ZARR_TILE_SIZE)
                x0 = min(x0, vol.shape[2] - config.crop - ZARR_TILE_SIZE)
                y0 = max(0, y0)
                x0 = max(0, x0)
                tile_origins.append((y0, x0))
            logger.info(f"[{label}] Using {len(tile_origins)} tile origins from ROI points")
        elif enrich_mask_vol is not None:
            all_tiles = _compute_tile_origins(
                vol.shape,
                tile_size=ZARR_TILE_SIZE,
                n_tiles=config.n * 10,
                crop=config.crop,
            )
            logger.info(f"[{label}] Scoring {len(all_tiles)} tile candidates by mask coverage...")
            tile_origins = _score_and_select_tiles(
                all_tiles,
                enrich_mask_vol,
                tile_size=ZARR_TILE_SIZE,
                count=config.n,
                score_fn=lambda tile: int(np.sum(tile > 0)),
            )
            logger.info(f"[{label}] Selected {len(tile_origins)} tiles by diversity scoring")
        else:
            tile_origins = _compute_tile_origins(
                vol.shape,
                tile_size=ZARR_TILE_SIZE,
                n_tiles=config.n,
                crop=config.crop,
            )

        z_candidates = list(range(0, vol.shape[0], config.dz))
        if not z_candidates:
            raise ValueError(f"[{label}] No Z indices available after applying dz to fused Zarr volume.")
        total_outputs += len(tile_origins) * len(z_candidates)
        tile_jobs.append(
            TileJob(
                file=f,
                vol=vol,
                channel_names=names_all,
                mask_vol=mask_vol,
                mask_path=mask_path,
                tile_origins=tile_origins,
                z_candidates=z_candidates,
            )
        )

    logger.info(f"[{label}] Tracking {total_outputs} output files from Zarr input(s)")

    with progress_reporter(total_outputs) as progress_update:
        for job in tile_jobs:
            _extract_tiles_from_zarr(
                job=job,
                roi=label,
                out_dir=out_dir,
                channels=config.channels,
                dz=config.dz,
                upscale=config.upscale,
                max_from_path=max_from_path,
                progress=progress_update,
            )


def _execute_zarr_ortho_extraction(
    *,
    label: str,
    files: list[Path],
    config: ExtractionConfig,
    out_dir: Path,
    max_from_path: Path | None,
    explicit_mask_path: Path | None,
    enrich_boundaries: Path | None,
    roi_points: Path | None = None,
) -> None:
    rng = np.random.default_rng(config.seed)
    logger.info(f"[{label}] Random seed: {config.seed if config.seed is not None else 'system entropy'}")

    slice_jobs: list[SliceJob] = []
    enrich_mask_vol: MaskLike | None = None
    if enrich_boundaries is not None:
        enrich_mask_vol = _open_mask_volume(enrich_boundaries)
        logger.info(f"[{label}] Using enrichment mask for diversity scoring: {enrich_boundaries}")

    # Load ROI points if provided (coordinates are upscaled 2x inside load_roi_points)
    point_coords: list[tuple[int, int]] | None = None
    if roi_points is not None:
        point_coords = load_roi_points(roi_points)
        logger.info(f"[{label}] Loaded {len(point_coords)} points from ROI file")

    for f in files:
        vol, names_all = _open_volume(f)
        mask_path = resolve_file_mask_path(f, explicit_mask_path)
        mask_vol = open_and_validate_mask(mask_path, vol, label=f"{label}:{f.name}")

        other_vol = _resolve_other_volume(f, max_from_path)

        selected_indices = _parse_channels(config.channels, names_all, vol.shape[-1])
        _ensure_channel_bounds(selected_indices, vol.shape[-1], label=f.name)
        base_names = _resolve_output_names(selected_indices, names_all, config.channels)
        out_names = [*base_names, "max_from"] if other_vol is not None else base_names

        max_width_pre_upscale = int(MAX_WIDTH_AFTER_UPSCALE / config.upscale)
        slab_half_width = 256  # 512px slab width / 2

        if point_coords is not None:
            # Use ROI points: for each point, create both Y and X slabs with context
            for px, py in point_coords:
                # Clamp to valid range
                base_py = max(config.crop, min(py, vol.shape[1] - config.crop - 1))
                base_px = max(config.crop, min(px, vol.shape[2] - config.crop - 1))

                # ZX slabs at position py (with context), x_slice centered at px
                x_start = max(config.crop, base_px - slab_half_width)
                x_end = min(vol.shape[2] - config.crop, base_px + slab_half_width)
                x_slice = slice(x_start, x_end)
                expanded_y = _expand_positions_with_context([base_py], crop=config.crop, axis_len=vol.shape[1])
                for yi in expanded_y:
                    slice_jobs.append(
                        SliceJob(
                            file=f,
                            vol=vol,
                            channel_names=names_all,
                            mask_vol=mask_vol,
                            other_vol=other_vol,
                            position=yi,
                            axis="y",
                            perpendicular_slice=x_slice,
                            selected_indices=selected_indices,
                            out_names=out_names,
                        )
                    )

                # ZY slabs at position px (with context), y_slice centered at py
                y_start = max(config.crop, base_py - slab_half_width)
                y_end = min(vol.shape[1] - config.crop, base_py + slab_half_width)
                y_slice = slice(y_start, y_end)
                expanded_x = _expand_positions_with_context([base_px], crop=config.crop, axis_len=vol.shape[2])
                for xi in expanded_x:
                    slice_jobs.append(
                        SliceJob(
                            file=f,
                            vol=vol,
                            channel_names=names_all,
                            mask_vol=mask_vol,
                            other_vol=other_vol,
                            position=xi,
                            axis="x",
                            perpendicular_slice=y_slice,
                            selected_indices=selected_indices,
                            out_names=out_names,
                        )
                    )
        else:
            # Original random sampling logic
            y_candidates = _sample_positions(vol.shape[1], crop=config.crop, count=config.n * 10, rng=rng)
            x_candidates = _sample_positions(vol.shape[2], crop=config.crop, count=config.n * 10, rng=rng)
            if enrich_mask_vol is not None:
                y_candidates = _select_high_diversity_positions(y_candidates, enrich_mask_vol, "y", config.n)
                x_candidates = _select_high_diversity_positions(x_candidates, enrich_mask_vol, "x", config.n)
            else:
                y_candidates = y_candidates[: config.n]
                x_candidates = x_candidates[: config.n]

            for base_y in y_candidates:
                x_slice = _compute_perpendicular_slice(
                    axis_len=vol.shape[2], crop=config.crop, max_width=max_width_pre_upscale, rng=rng
                )
                expanded_y = _expand_positions_with_context([base_y], crop=config.crop, axis_len=vol.shape[1])
                for yi in expanded_y:
                    slice_jobs.append(
                        SliceJob(
                            file=f,
                            vol=vol,
                            channel_names=names_all,
                            mask_vol=mask_vol,
                            other_vol=other_vol,
                            position=yi,
                            axis="y",
                            perpendicular_slice=x_slice,
                            selected_indices=selected_indices,
                            out_names=out_names,
                        )
                    )
            for base_x in x_candidates:
                y_slice = _compute_perpendicular_slice(
                    axis_len=vol.shape[1], crop=config.crop, max_width=max_width_pre_upscale, rng=rng
                )
                expanded_x = _expand_positions_with_context([base_x], crop=config.crop, axis_len=vol.shape[2])
                for xi in expanded_x:
                    slice_jobs.append(
                        SliceJob(
                            file=f,
                            vol=vol,
                            channel_names=names_all,
                            mask_vol=mask_vol,
                            other_vol=other_vol,
                            position=xi,
                            axis="x",
                            perpendicular_slice=y_slice,
                            selected_indices=selected_indices,
                            out_names=out_names,
                        )
                    )

    logger.info(f"[{label}] Processing {len(slice_jobs)} ortho slices in parallel with {config.threads} threads")

    with progress_bar_threadpool(len(slice_jobs), threads=config.threads, stop_on_exception=True) as submit:
        for job in slice_jobs:
            submit(
                _process_ortho_slice,
                vol=job.vol,
                mask_vol=job.mask_vol,
                other_vol=job.other_vol,
                position=job.position,
                axis=job.axis,
                perpendicular_slice=job.perpendicular_slice,
                selected_indices=job.selected_indices,
                out_names=job.out_names,
                anisotropy=config.anisotropy,
                upscale=config.upscale,
                out_dir=out_dir,
                file_stem=job.file.stem,
                roi=label,
                channels=config.channels,
            )


def _execute_tiff_extraction(
    *,
    label: str,
    files: list[Path],
    config: ExtractionConfig,
    out_dir: Path,
    max_from_path: Path | None,
    explicit_mask_path: Path | None,
    enrich_boundaries: Path | None,
    roi_points: Path | None = None,
) -> None:
    if roi_points is not None:
        logger.warning(f"[{label}] --roi-points is only supported for Zarr inputs; ignoring for TIFF extraction")

    with progress_bar_threadpool(len(files), threads=config.threads, stop_on_exception=True) as submit:
        if config.mode == "z":
            for idx, f in enumerate(files):
                mask_path = explicit_mask_path if explicit_mask_path is not None else _resolve_mask_path(f)
                if mask_path is not None:
                    logger.info(f"[{label}] Found mask stack: {mask_path}")
                file_seed = (config.seed + idx) if config.seed is not None else None
                submit(
                    _extract_z_slices,
                    file=f,
                    roi=label,
                    out_dir=out_dir,
                    channels=config.channels,
                    dz=config.dz,
                    n=config.n,
                    upscale=config.upscale,
                    max_from_path=max_from_path,
                    mask_path=mask_path,
                    enrich_boundaries=enrich_boundaries,
                    seed=file_seed,
                    progress=None,
                )
        elif config.mode == "ortho":
            for idx, f in enumerate(files):
                mask_path = explicit_mask_path if explicit_mask_path is not None else _resolve_mask_path(f)
                if mask_path is not None:
                    logger.info(f"[{label}] Found mask stack: {mask_path}")
                file_seed = (config.seed + idx) if config.seed is not None else None
                submit(
                    _extract_ortho_slices,
                    file=f,
                    roi=label,
                    out_dir=out_dir,
                    channels=config.channels,
                    crop=config.crop,
                    n=config.n,
                    anisotropy=config.anisotropy,
                    upscale=config.upscale,
                    max_from_path=max_from_path,
                    mask_path=mask_path,
                    enrich_boundaries=enrich_boundaries,
                    seed=file_seed,
                    progress=None,
                )
        else:
            raise ValueError(f"Unsupported mode: {config.mode}")


def _resize_uint16(data: np.ndarray, factors: tuple[float, ...]) -> np.ndarray:
    """Resize data with scipy.ndimage.zoom, preserving uint16 output."""

    if all(math.isclose(f, 1.0, abs_tol=1e-9, rel_tol=1e-9) for f in factors):
        # Ensure dtype consistency without extra work when no scaling requested.
        return data.astype(np.uint16, copy=False)

    resized = zoom(data.astype(np.float32, copy=False), factors, order=1)
    return np.clip(np.rint(resized), 0, 65530).astype(np.uint16)


def _prep_slab(
    slab: np.ndarray,
    *,
    ch_idx: list[int],
    channel_axis: int,
    crop_slices: tuple[slice, ...] | None,
    filter_before: bool,
    append_max: np.ndarray | None,
    apply_filter: bool = True,
) -> np.ndarray:
    """Select channels, optional filtering and max-append; return (C, ... spatial ...) uint16.

    - slab: spatial slab with channels on `channel_axis` (e.g., (Y,X,C), (Z,X,C), (Z,Y,C)).
    - ch_idx: selected channel indices (0-based).
    - crop_slices: optional slices on spatial axes (must not include channel axis).
    - filter_before: if True, filter before selection; else filter after selection.
    - append_max: optional array broadcastable to spatial shape, added as an extra channel.
    - apply_filter: enable sharpening filter (skipped for Zarr streaming to avoid extra IO).
    """
    arr = slab
    if apply_filter and filter_before:
        arr = unsharp_all(arr, channel_axis=channel_axis)

    # Select channels (channel axis last in arr)
    sel = np.take(arr, ch_idx, axis=channel_axis)

    if apply_filter and not filter_before:
        sel = unsharp_all(sel, channel_axis=sel.ndim - 1)

    # Append max channel if provided
    if append_max is not None:
        # Ensure append_max matches spatial shape
        while append_max.ndim < sel.ndim:
            append_max = append_max[..., None]
        sel = np.concatenate([sel, append_max], axis=sel.ndim - 1)

    # Move channels to first axis
    sel_c_first = np.moveaxis(sel, -1, 0)

    # Apply cropping on spatial axes if requested
    if crop_slices is not None and any(slc != slice(None) for slc in crop_slices):
        # Build slices with channel dim first
        idx = (slice(None),) + crop_slices
        sel_c_first = sel_c_first[idx]

    # Final dtype
    return np.clip(sel_c_first, 0, 65530).astype(np.uint16)


def _parse_channels(ch_arg: str | None, names: list[str] | None, channel_count: int | None) -> list[int]:
    """Parse channel specification; requires explicit indices or metadata-backed names."""
    if ch_arg is None or ch_arg.strip().lower() == "auto":
        if names:
            default_count = min(2, len(names))
            if default_count == 0:
                raise click.BadParameter("Channel metadata is empty; supply --channels explicitly.")
            return list(range(default_count))
        if channel_count is not None:
            return list(range(min(2, channel_count)))
        return [0, 1]

    parts = [p.strip() for p in ch_arg.split(",") if p.strip()]
    if not parts:
        raise click.BadParameter("Empty --channels specification.")

    try:
        return [int(p) for p in parts]
    except ValueError:
        if not names:
            raise click.BadParameter("Channel names not in metadata; pass numeric indices.")
        name_to_idx = {n: i for i, n in enumerate(names)}
        indices: list[int] = []
        for part in parts:
            try:
                indices.append(name_to_idx[part])
            except KeyError as error:
                raise click.BadParameter(f"Unknown channel name: {error.args[0]}") from error
        return indices


def _ensure_channel_bounds(indices: list[int], channel_count: int, *, label: str) -> None:
    if not indices:
        raise click.BadParameter("At least one channel index must be selected.")
    if min(indices) < 0:
        raise click.BadParameter("Channel indices must be non-negative.")
    if max(indices) >= channel_count:
        raise click.BadParameter(
            f"{label}: requested channel index {max(indices)} exceeds available channels ({channel_count})."
        )


def _load_registered_stack(file: Path) -> tuple[np.ndarray, list[str] | None, str]:
    """Load a registered stack from TIFF or Zarr and normalise to (Z,C,Y,X).

    This reuses `_open_volume` to enforce the common shape contract (Z,Y,X,C)
    and then materialises a NumPy array with channels moved to axis 1.
    """
    vol_zyxc, names = _open_volume(file)
    # Materialise lazy Zarr arrays for max-projection operations.
    if isinstance(vol_zyxc, zarr.Array):
        img = np.asarray(vol_zyxc)
    else:
        img = vol_zyxc
    if img.ndim != 4:
        raise ValueError("Expected registered volume to be 4D (Z,Y,X,C).")
    img_zcyx = np.moveaxis(img, -1, 1)  # -> (Z,C,Y,X)
    return img_zcyx, names, file.name


def _append_max_from(
    img: np.ndarray, file: Path, max_from_path: Path | None
) -> tuple[np.ndarray, int | None]:
    """Append max-projection channel from another codebook. Returns (image, appended_channel_index)."""
    if max_from_path is None:
        return img, None

    target: Path
    if max_from_path.is_dir() and not max_from_path.name.endswith(".zarr"):
        candidate = max_from_path / file.name
        if not candidate.exists():
            raise FileNotFoundError(f"Max-from file not found: {candidate}")
        target = candidate
    else:
        target = max_from_path

    other, _names, _label = _load_registered_stack(target)
    if other.shape[0] != img.shape[0] or other.shape[2:] != img.shape[2:]:
        raise ValueError("Max-from stack shape mismatch.")

    # Compute max projection in Z chunks to avoid holding the full projection at once
    max_chunk_size = 64
    running_max = np.zeros((img.shape[0], 1, img.shape[2], img.shape[3]), dtype=other.dtype)
    for z_start in range(0, other.shape[0], max_chunk_size):
        z_end = min(z_start + max_chunk_size, other.shape[0])
        chunk = other[z_start:z_end]
        channel_max = chunk.max(axis=1, keepdims=True)
        running_max[z_start:z_end] = channel_max

    new_img = np.concatenate([img, running_max], axis=1)
    return new_img, new_img.shape[1] - 1


def normalize_numeric_options(
    *,
    mode: str,
    dz: int,
    anisotropy: int,
    upscale: float | None,
    use_zarr: bool,
    has_max_from: bool,
    ortho_anisotropy_default: int,
) -> float:
    """Validate dz/anisotropy/upscale and mode-specific constraints.

    Returns the normalized ``upscale`` value while preserving existing CLI semantics.
    """
    if mode == "z" and anisotropy != ortho_anisotropy_default:
        raise click.BadParameter("--anisotropy parameter is only valid for 'ortho' mode.")
    if mode == "ortho" and dz != 1:
        raise click.BadParameter("--dz parameter is only valid for 'z' mode.")

    if use_zarr and has_max_from:
        raise click.BadParameter("--max-from cannot be used together with --zarr inputs.")

    if use_zarr:
        if upscale is not None and not math.isclose(upscale, 2.0):
            logger.info(f"Overriding --upscale value {upscale} to 2.0 for --zarr mode.")
        upscale_value = 2.0
    elif upscale is None:
        upscale_value = 1.0
    else:
        upscale_value = upscale

    if upscale_value <= 0:
        raise click.BadParameter("--upscale must be positive.")

    return upscale_value


def _iter_registered_files(reg_dir: Path) -> Iterable[Path]:
    """Get registered TIFF files sorted by size (largest first) from reg-*.tif pattern."""
    files = sorted(reg_dir.glob("reg-*.tif"), key=lambda p: p.stat().st_size, reverse=True)
    if not files:
        raise FileNotFoundError(f"No registered images found: {reg_dir}")
    return files


def _discover_registered_inputs(
    ws: Workspace,
    roi: str,
    codebook: str,
    *,
    require_zarr: bool = False,
) -> list[Path]:
    fused_zarr = ws.stitch(roi, codebook) / FILE_NAME

    if require_zarr:
        if fused_zarr.exists():
            logger.debug(f"Using fused Zarr at {fused_zarr}")
            return [fused_zarr]
        raise FileNotFoundError(f"Requested Zarr input but fused store not found at {fused_zarr}.")

    reg_dir = ws.registered(roi, codebook)
    try:
        files = list(_iter_registered_files(reg_dir))
        if files:
            logger.debug(f"Found {len(files)} registered TIFF files in {reg_dir}")
            return files
    except FileNotFoundError:
        files = []

    if fused_zarr.exists():
        logger.debug(f"Falling back to fused Zarr at {fused_zarr}")
        return [fused_zarr]

    raise FileNotFoundError(f"No registered TIFFs in {reg_dir} or fused Zarr at {fused_zarr}.")


def run_workspace_extract(
    *,
    ws: Workspace,
    mode: str,
    codebook: str,
    rois: list[str],
    out: Path | None,
    dz: int,
    n: int | None,
    anisotropy: int,
    channels: str | None,
    crop: int,
    threads: int,
    upscale: float,
    seed: int | None,
    every: int,
    max_from: str | None,
    use_zarr: bool,
    masks: Path | None,
    enrich_boundaries: Path | None,
    enable_enrich_boundaries: bool,
    roi_points: Path | None = None,
) -> None:
    inputs_by_roi: dict[str, list[Path]] = {}
    for current_roi in rois:
        inputs_by_roi[current_roi] = _discover_registered_inputs(
            ws,
            current_roi,
            codebook,
            require_zarr=use_zarr,
        )

    counts = {roi_val: len(inputs_by_roi[roi_val]) for roi_val in rois}
    quota_map = _distribute_file_budget(rois, counts, n) if n is not None else {roi_val: None for roi_val in rois}

    if len(rois) == 1:
        quota_map = {roi_val: None for roi_val in rois}

    enable_enrich = enable_enrich_boundaries
    for idx, current_roi in enumerate(rois):
        roi_out = out if out is None or len(rois) == 1 else out / current_roi

        _extract_single_roi(
            ws=ws,
            roi=current_roi,
            codebook=codebook,
            mode=mode,
            out=roi_out,
            dz=dz,
            n=n,
            anisotropy=anisotropy,
            channels=channels,
            crop=crop,
            threads=threads,
            upscale=upscale,
            seed=None if seed is None else seed + idx,
            every=every,
            max_from=max_from,
            use_zarr=use_zarr,
            prefetched_inputs=inputs_by_roi[current_roi],
            file_quota=quota_map[current_roi],
            explicit_mask_path=masks,
            enrich_boundaries=enrich_boundaries,
            enable_enrich_boundaries=enable_enrich,
            roi_points=roi_points,
        )


def run_single_file_extract(
    *,
    mode: str,
    registered: Path,
    out: Path,
    dz: int,
    n: int,
    anisotropy: int,
    channels: str | None,
    crop: int,
    threads: int,
    upscale: float,
    seed: int | None,
    max_from_path: Path | None,
    label: str,
    masks: Path | None,
    enrich_boundaries: Path | None,
) -> None:
    files = [registered]

    if max_from_path is not None:
        _validate_max_from_path(max_from_path, files, label=label)

    _execute_extraction(
        label=label,
        files=files,
        mode=mode,
        out_dir=out,
        channels=channels,
        crop=crop,
        dz=dz,
        n=n,
        anisotropy=anisotropy,
        threads=threads,
        upscale=upscale,
        seed=seed,
        max_from_path=max_from_path,
        explicit_mask_path=masks,
        enrich_boundaries=enrich_boundaries,
    )




def _extract_single_roi(
    *,
    ws: Workspace,
    roi: str,
    codebook: str,
    mode: str,
    out: Path | None,
    dz: int,
    n: int,
    anisotropy: int,
    channels: str | None,
    crop: int,
    threads: int,
    upscale: float,
    seed: int | None,
    every: int,
    max_from: str | None,
    use_zarr: bool,
    prefetched_inputs: list[Path] | None,
    file_quota: int | None,
    explicit_mask_path: Path | None = None,
    enrich_boundaries: Path | None = None,
    enable_enrich_boundaries: bool = True,
    roi_points: Path | None = None,
) -> None:
    reg_dir = ws.registered(roi, codebook)
    out_dir = out
    if out_dir is None:
        out_dir = _default_output_dir(ws, roi)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"[{roi}] Input: {reg_dir}")
    logger.info(f"[{roi}] Output: {out_dir}")
    logger.info(f"[{roi}] Upscale factor: {upscale}")

    if prefetched_inputs is not None:
        inputs = prefetched_inputs
    else:
        inputs = _discover_registered_inputs(
            ws,
            roi,
            codebook,
            require_zarr=use_zarr,
        )
    files = inputs[::every]
    if file_quota is not None:
        files = files[:file_quota]
    if not files:
        raise FileNotFoundError(f"[{roi}] No registered inputs matched the selection criteria.")

    max_from_path: Path | None = None
    if max_from:
        max_from_path = ws.registered(roi, max_from)
        _validate_max_from_path(max_from_path, files, label=roi)

    # Default to output_segmentation-sam_postproc.zarr if no explicit mask provided
    mask_path = explicit_mask_path
    if mask_path is None:
        default_mask = ws.stitch(roi, codebook) / "output_segmentation-sam_postproc.zarr"
        if default_mask.exists():
            mask_path = default_mask
            logger.info(f"[{roi}] Using default mask: {mask_path}")

    enrich_mask_path = _resolve_enrich_mask(
        ws,
        roi,
        codebook,
        enrich_boundaries,
        enable_enrich_boundaries,
    )

    _execute_extraction(
        label=roi,
        files=files,
        mode=mode,
        out_dir=out_dir,
        channels=channels,
        crop=crop,
        dz=dz,
        n=n,
        anisotropy=anisotropy,
        threads=threads,
        upscale=upscale,
        seed=seed,
        max_from_path=max_from_path,
        explicit_mask_path=mask_path,
        enrich_boundaries=enrich_mask_path,
        roi_points=roi_points,
    )


# ---------- Logging helpers ----------


def _resolve_other_volume(file: Path, max_from_path: Path | None) -> Volume | None:
    """Resolve and open the comparison volume for --max-from if it exists."""
    if max_from_path is None:
        return None

    target = (
        max_from_path
        if _is_zarr_path(max_from_path)
        else (max_from_path / file.name if max_from_path.is_dir() else max_from_path)
    )
    if not target.exists():
        return None
    other_vol, _ = _open_volume(target)
    return other_vol


def _extract_z_slices(
    *,
    file: Path,
    roi: str,
    out_dir: Path,
    channels: str | None,
    dz: int,
    n: int,
    upscale: float,
    max_from_path: Path | None,
    mask_path: Path | None,
    enrich_boundaries: Path | None,
    seed: int | None,
    progress: ProgressReporter | Callable[[], int | None] | None,
) -> None:
    reporter = _normalize_reporter(progress)

    size_str = _format_size(_path_size_cached(str(file.resolve())))
    logger.info(f"3D→Z: {file.name} [{size_str}] (dz={dz})")

    ctx = build_extraction_context(
        file,
        roi,
        channels=channels,
        max_from_path=max_from_path,
        mask_path=mask_path,
        enrich_path=enrich_boundaries,
        upscale=upscale,
        anisotropy=1,
        out_dir=out_dir,
    )

    vol = ctx.vol
    selected_indices = ctx.selected_indices
    out_names = ctx.out_names
    other_vol = ctx.other_vol
    mask_vol = ctx.mask_vol
    enrich_mask_vol = ctx.enrich_mask_vol

    z_len, y_len, x_len, _ = vol.shape

    crop_size = DEFAULT_CROP_SIZE
    max_y = max(0, y_len - crop_size)
    max_x = max(0, x_len - crop_size)

    n_crops = max(1, n)
    rng = np.random.default_rng(seed)

    if enrich_mask_vol is not None and (max_y > 0 or max_x > 0):
        n_candidates = n_crops * 10
        positions: list[tuple[int, int]] = []
        for _ in range(n_candidates):
            cy = rng.integers(0, max_y + 1) if max_y > 0 else 0
            cx = rng.integers(0, max_x + 1) if max_x > 0 else 0
            positions.append((cy, cx))

        crop_positions = _score_and_select_tiles(
            positions,
            enrich_mask_vol,
            tile_size=crop_size,
            count=n_crops,
            score_fn=lambda tile: len(np.unique(tile)),
        )
        logger.info(f"[{roi}] Selected {len(crop_positions)} crops by diversity scoring")
    else:
        crop_positions = [
            (
                rng.integers(0, max_y + 1) if max_y > 0 else 0,
                rng.integers(0, max_x + 1) if max_x > 0 else 0,
            )
            for _ in range(n_crops)
        ]

    z_idxs = list(range(0, z_len, dz))
    cancel_event = get_cancel_event()

    for crop_idx, (y_start, x_start) in enumerate(crop_positions):
        y_slice = slice(y_start, min(y_start + crop_size, y_len))
        x_slice = slice(x_start, min(x_start + crop_size, x_len))

        for index in z_idxs:
            if cancel_event.is_set():
                raise TaskCancelledException("Cancelled by user")
            t0 = perf_counter()
            plane = vol[index, y_slice, x_slice, :]
            other_max = (
                other_vol[index, y_slice, x_slice, :].max(axis=2) if other_vol is not None else None
            )
            t_read = perf_counter() - t0

            t0 = perf_counter()
            cyx_u16 = _prep_slab(
                plane,
                ch_idx=selected_indices,
                channel_axis=2,
                crop_slices=None,
                filter_before=False,
                append_max=other_max,
            )
            t_prep = perf_counter() - t0

            t0 = perf_counter()
            cyx_u16 = _resize_uint16(cyx_u16, (1.0, upscale, upscale))
            t_resize = perf_counter() - t0

            out_name = (
                _prefix_with_roi(f"{file.stem}_crop{crop_idx:02d}_z{index:02d}.tif", roi)
                if n_crops > 1
                else _prefix_with_roi(f"{file.stem}_z{index:02d}.tif", roi)
            )
            out_file = out_dir / out_name

            t0 = perf_counter()
            _write_tiff(
                out_file,
                cyx_u16,
                axes="CYX",
                names=out_names,
                channels_arg=channels,
                upscale=upscale,
            )
            t_write = perf_counter() - t0

            t_mask = 0.0
            if mask_vol is not None:
                t0 = perf_counter()
                mask_plane = _squeeze_mask(mask_vol[index, ...])
                if mask_plane.ndim != 2:
                    raise ValueError("Mask plane extraction expected 2D data.")
                mask_cropped = mask_plane[y_slice, x_slice]
                resized_mask = _resize_mask(mask_cropped, (upscale, upscale))
                mask_out_name = _mask_filename(out_name)
                _write_mask_tiff(out_dir / mask_out_name, resized_mask, axes="YX")
                t_mask = perf_counter() - t0

            logger.debug(
                f"z{index:02d} read={t_read:.3f}s prep={t_prep:.3f}s resize={t_resize:.3f}s write={t_write:.3f}s mask={t_mask:.3f}s"
            )
            if reporter is not None:
                reporter.advance()


def _extract_tiles_from_zarr(
    *,
    job: TileJob,
    roi: str,
    out_dir: Path,
    channels: str | None,
    dz: int,
    upscale: float,
    max_from_path: Path | None,
    progress: ProgressReporter | Callable[[], int | None] | None,
) -> None:
    reporter = _normalize_reporter(progress)
    size_str = _format_size(_path_size_cached(str(job.file.resolve())))
    logger.info(f"3D→Z tiles: {job.file.name} [{size_str}]")

    selected_indices = _parse_channels(channels, job.channel_names, job.vol.shape[-1])
    _ensure_channel_bounds(selected_indices, job.vol.shape[-1], label=job.file.name)
    base_names = _resolve_output_names(selected_indices, job.channel_names, channels)
    other_vol = _resolve_other_volume(job.file, max_from_path)
    out_names = [*base_names, "max_from"] if other_vol is not None else base_names

    z_len = job.vol.shape[0]
    if job.mask_vol is not None and job.mask_vol.shape[0] != z_len:
        raise ValueError(f"Mask volume {_resolve_mask_path(job.file)} does not match Z dimension of {job.file}.")

    coord_width = len(str(max(job.vol.shape[1], job.vol.shape[2])))
    z_candidates = job.z_candidates or list(range(0, z_len, dz))
    if not z_candidates:
        raise ValueError(f"[{roi}] No Z indices available after applying dz to fused Zarr volume.")

    cancel_event = get_cancel_event()
    for y0, x0 in job.tile_origins:
        y_slice_tile = slice(y0, y0 + ZARR_TILE_SIZE)
        x_slice_tile = slice(x0, x0 + ZARR_TILE_SIZE)
        skipped_for_tile = 0
        for z_index in z_candidates:
            if cancel_event.is_set():
                raise TaskCancelledException("Cancelled by user")
            plane = job.vol[z_index, y_slice_tile, x_slice_tile, :]
            zero_fraction = np.mean(plane == 0)
            skip_tile = zero_fraction > ZERO_PIXEL_SKIP_THRESHOLD
            other_max = None
            if other_vol is not None:
                other_tile = other_vol[z_index, y_slice_tile, x_slice_tile, :]
                if not skip_tile:
                    other_zero_fraction = np.mean(other_tile == 0)
                    skip_tile = other_zero_fraction > ZERO_PIXEL_SKIP_THRESHOLD
                if not skip_tile:
                    other_max = other_tile.max(axis=2)

            if skip_tile:
                skipped_for_tile += 1
                if reporter is not None:
                    reporter.advance()
                continue

            cyx_u16 = _prep_slab(
                plane,
                ch_idx=selected_indices,
                channel_axis=2,
                crop_slices=None,
                filter_before=True,
                append_max=other_max,
                apply_filter=False,
            )
            cyx_u16 = _resize_uint16(cyx_u16, (1.0, upscale, upscale))
            out_name = _format_tile_filename(
                job.file.stem,
                roi,
                z_index,
                y0,
                x0,
                coord_width=coord_width,
            )
            out_file = out_dir / out_name
            _write_tiff(
                out_file,
                cyx_u16,
                axes="CYX",
                names=out_names,
                channels_arg=channels,
                upscale=upscale,
            )
            if job.mask_vol is not None:
                mask_tile = _squeeze_mask(job.mask_vol[z_index, y_slice_tile, x_slice_tile])
                if mask_tile.ndim != 2:
                    raise ValueError("Mask tile extraction expected 2D data.")
                resized_mask = _resize_mask(mask_tile, (upscale, upscale))
                mask_out = out_dir / _mask_filename(out_name)
                _write_mask_tiff(mask_out, resized_mask, axes="YX")
            if reporter is not None:
                reporter.advance()
        if skipped_for_tile:
            logger.debug(
                f"Skipped {skipped_for_tile} z-slice(s) in tile ({y0},{x0}) of {job.file.name} due to zeros."
            )


def _extract_ortho_slices(
    *,
    file: Path,
    roi: str,
    out_dir: Path,
    channels: str | None,
    crop: int,
    n: int,
    anisotropy: int,
    upscale: float,
    max_from_path: Path | None,
    mask_path: Path | None,
    enrich_boundaries: Path | None,
    seed: int | None,
    progress: ProgressReporter | Callable[[], int | None] | None,
) -> None:
    reporter = _normalize_reporter(progress)
    size_str = _format_size(_path_size_cached(str(file.resolve())))
    logger.info(f"Ortho: {file.name} [{size_str}]")

    ctx = build_extraction_context(
        file,
        roi,
        channels=channels,
        max_from_path=max_from_path,
        mask_path=mask_path,
        enrich_path=enrich_boundaries,
        upscale=upscale,
        anisotropy=anisotropy,
        out_dir=out_dir,
    )

    vol = ctx.vol
    selected_indices = ctx.selected_indices
    out_names = ctx.out_names
    other_vol = ctx.other_vol
    mask_vol = ctx.mask_vol
    enrich_mask_vol = ctx.enrich_mask_vol

    z_len, y_len, x_len, _ = vol.shape

    y_eff = y_len - 2 * crop
    x_eff = x_len - 2 * crop
    if y_eff <= 0 or x_eff <= 0:
        raise ValueError("Image after cropping is empty.")

    enrich_mask_vol: MaskLike | None = None
    if enrich_boundaries is not None:
        enrich_mask_vol = _open_mask_volume(enrich_boundaries)
        logger.info(f"[{roi}] Using enrichment mask for diversity scoring: {enrich_boundaries}")

    oversample = 10 if enrich_mask_vol is not None else 1
    base_y = (
        np.linspace(int(0.1 * y_eff), int(0.9 * y_eff), n * oversample).astype(int) + crop
    ).tolist()
    base_x = (
        np.linspace(int(0.1 * x_eff), int(0.9 * x_eff), n * oversample).astype(int) + crop
    ).tolist()
    if enrich_mask_vol is not None:
        base_y = _select_high_diversity_positions(base_y, enrich_mask_vol, "y", n)
        base_x = _select_high_diversity_positions(base_x, enrich_mask_vol, "x", n)
    y_bases = base_y
    x_bases = base_x

    max_width_pre_upscale = int(MAX_WIDTH_AFTER_UPSCALE / upscale)

    rng = np.random.default_rng(seed)
    cancel_event = get_cancel_event()

    for base_yi in y_bases:
        if cancel_event.is_set():
            raise TaskCancelledException("Cancelled by user")
        x_slice = _compute_perpendicular_slice(
            axis_len=x_len, crop=crop, max_width=max_width_pre_upscale, rng=rng
        )
        expanded_y = _expand_positions_with_context([base_yi], crop=crop, axis_len=y_len)
        for yi in expanded_y:
            slab = vol[:, yi, x_slice, :]
            other_max = other_vol[:, yi, x_slice, :].max(axis=2) if other_vol is not None else None
            czx = _prep_slab(
                slab,
                ch_idx=selected_indices,
                channel_axis=2,
                crop_slices=None,
                filter_before=False,
                append_max=other_max,
                apply_filter=False,
            )
            czx = _resize_uint16(czx, (1.0, anisotropy * upscale, upscale))
            x_start = x_slice.start or 0
            out_name = _prefix_with_roi(f"{file.stem}_orthozx-x{x_start}-y{yi}.tif", roi)
            out_file = out_dir / out_name
            _write_tiff(
                out_file,
                czx,
                axes="CZX",
                names=out_names,
                channels_arg=channels,
                upscale=upscale,
            )
            if mask_vol is not None:
                mask_slab = _squeeze_mask(mask_vol[:, yi, x_slice])
                if mask_slab.ndim != 2:
                    raise ValueError("Mask ZX extraction expected 2D data.")
                resized_mask = _resize_mask(mask_slab, (anisotropy * upscale, upscale))
                mask_file = out_dir / _mask_filename(out_name)
                _write_mask_tiff(mask_file, resized_mask, axes="ZX")
            if reporter is not None:
                reporter.advance()

    for base_xi in x_bases:
        if cancel_event.is_set():
            raise TaskCancelledException("Cancelled by user")
        y_slice = _compute_perpendicular_slice(
            axis_len=y_len, crop=crop, max_width=max_width_pre_upscale, rng=rng
        )
        expanded_x = _expand_positions_with_context([base_xi], crop=crop, axis_len=x_len)
        for xi in expanded_x:
            slab = vol[:, y_slice, xi, :]
            other_max = other_vol[:, y_slice, xi, :].max(axis=2) if other_vol is not None else None
            czy = _prep_slab(
                slab,
                ch_idx=selected_indices,
                channel_axis=2,
                crop_slices=None,
                filter_before=False,
                append_max=other_max,
                apply_filter=False,
            )
            czy = _resize_uint16(czy, (1.0, anisotropy * upscale, upscale))
            y_start = y_slice.start or 0
            out_name = _prefix_with_roi(f"{file.stem}_orthozy-y{y_start}-x{xi}.tif", roi)
            out_file = out_dir / out_name
            _write_tiff(
                out_file,
                czy,
                axes="CZY",
                names=out_names,
                channels_arg=channels,
                upscale=upscale,
            )
            if mask_vol is not None:
                mask_slab = _squeeze_mask(mask_vol[:, y_slice, xi])
                if mask_slab.ndim != 2:
                    raise ValueError("Mask ZY extraction expected 2D data.")
                resized_mask = _resize_mask(mask_slab, (anisotropy * upscale, upscale))
                mask_file = out_dir / _mask_filename(out_name)
                _write_mask_tiff(mask_file, resized_mask, axes="ZY")
            if reporter is not None:
                reporter.advance()
