"""
FISH Image Stitching Pipeline

This module provides tools for stitching multi-tile FISH images into seamless mosaics.
The workflow includes:

1. Tile registration using ImageJ Grid/Collection stitching
2. Channel extraction and preprocessing
3. Multi-threaded image fusion
4. Zarr array creation for large datasets

Key Functions:
- create_tile_config: Generate ImageJ-compatible tile configuration files
- run_imagej: Execute ImageJ stitching operations via subprocess
- extract_channel: Extract and preprocess individual channels from multi-channel images
- extract: Process images for segmentation with proper formatting
- walk_fused: Navigate fused image directory structures

CLI Commands:
- register-simple: Basic tile registration workflow
- register: Advanced registration with fiducial markers
- fuse: Multi-threaded image fusion with compression
- combine: Combine fused tiles into Zarr arrays for analysis
- run: Complete end-to-end stitching pipeline

Uses ImageJ Grid/Collection stitching for accurate tile alignment and supports
large-scale datasets through efficient memory management and parallelization.
"""

import json
import shutil
from collections.abc import Iterable
from itertools import chain
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

import numpy as np
import pandas as pd
import rich_click as click
from loguru import logger
from PIL import Image
from tifffile import TiffFile, TiffFileError, imread

from fishtools.gpu.memory import release_all as gpu_release_all
from fishtools.io.workspace import Workspace, safe_imwrite
from fishtools.preprocess.config import StitchingConfig
from fishtools.preprocess.config_loader import load_config
from fishtools.preprocess.downsample import gpu_downsample_xy
from fishtools.preprocess.illumination import (
    apply_field_stores_to_img,
    parse_tile_index_from_path,
    resolve_roi_for_field,
    tile_origin,
)
from fishtools.preprocess.imagej import run_imagej as _run_imagej
from fishtools.preprocess.imageops import clip_range_for_dtype as clip_range_for_dtype_lib
from fishtools.preprocess.imageops import crop_xy as crop_xy_lib
from fishtools.preprocess.n4 import run_cli_workflow
from fishtools.preprocess.stitching import walk_fused as _walk_fused
from fishtools.preprocess.tileconfig import TileConfiguration
from fishtools.preprocess.tileconfig import copy_registered as _copy_registered
from fishtools.utils.logging import CONSOLE_SKIP_EXTRA, setup_cli_logging
from fishtools.utils.pretty_print import get_shared_console, progress_bar, progress_bar_threadpool
from fishtools.utils.tiff import compose_metadata as compose_meta
from fishtools.utils.tiff import normalize_channel_names as norm_names
from fishtools.utils.tiff import read_metadata_from_tif
from fishtools.utils.utils import add_file_context, batch_roi
from fishtools.utils.zarr_utils import numpy_array_to_zarr as _numpy_array_to_zarr


def _clip_range_for_dtype(dtype: np.dtype) -> tuple[float, float] | None:  # shim
    return clip_range_for_dtype_lib(dtype)


def _crop_xy(array: np.ndarray, trim: int) -> np.ndarray:  # shim
    return crop_xy_lib(array, trim)


def _read_tiff_metadata(tif: TiffFile) -> dict[str, Any]:
    return read_metadata_from_tif(tif)


def _normalize_channel_names(count: int, metadata: dict[str, Any]) -> list[str]:
    return norm_names(count, metadata)


def _compose_metadata(
    axes: str, channel_names: Sequence[str] | None, *, extra: dict[str, Any] | None = None
) -> dict[str, Any]:
    return compose_meta(axes, channel_names, extra=extra)


def _label_for_max_from(path: Path | None) -> str | None:
    if path is None:
        return None
    label = f"max_from:{path.stem}"
    try:
        with TiffFile(path) as tif:
            metadata = _read_tiff_metadata(tif)
        raw_keys = metadata.get("key")
        if isinstance(raw_keys, str):
            keys = [raw_keys]
        elif isinstance(raw_keys, (list, tuple, np.ndarray)):
            keys = [str(k) for k in list(raw_keys)]
        else:
            keys = []
        if keys:
            label = f"max_from:{'+'.join(keys)}"
    except Exception:
        pass
    return label


def _resolve_codebook_path(ws: Workspace, codebook: str) -> Path:
    """Locate the JSON file backing a codebook stem inside a workspace."""

    candidates = [
        ws.path / "codebooks" / f"{codebook}.json",
        ws.deconved / "codebooks" / f"{codebook}.json",
        ws.path / f"{codebook}.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    search_list = ", ".join(str(p) for p in candidates)
    raise ValueError(
        f"Codebook JSON for '{codebook}' not found. Looked under: {search_list}. "
        "Ensure the codebook JSON is present in the workspace."
    )


def _load_codebook_channels(ws: Workspace, codebook: str) -> set[str]:
    """Parse the set of channel identifiers used by a codebook."""

    path = _resolve_codebook_path(ws, codebook)
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:  # pragma: no cover - invalid user file
        raise ValueError(f"Failed to parse codebook JSON at {path}: {exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError(f"Expected codebook JSON at {path} to be an object mapping targets to channels.")

    channels: set[str] = set()
    for value in payload.values():
        if isinstance(value, (list, tuple, set)):
            for item in value:
                if item is None:
                    raise ValueError(f"Codebook {path} contains null channel entries.")
                channels.add(str(item))
        else:
            if value is None:
                raise ValueError(f"Codebook {path} contains null channel entries.")
            channels.add(str(value))

    if not channels:
        raise ValueError(f"Codebook '{codebook}' did not define any channels.")

    return channels


def _collect_registered_channels(
    img_paths: Sequence[Path],
    expected: set[str],
) -> set[str]:
    """Accumulate channel identifiers present in registered TIFF metadata."""

    discovered: set[str] = set()
    for img_path in img_paths:
        with TiffFile(img_path) as tif:
            metadata = _read_tiff_metadata(tif)
        raw_keys = metadata.get("key")
        if raw_keys is None:
            continue
        if isinstance(raw_keys, str):
            discovered.add(raw_keys)
        elif isinstance(raw_keys, Iterable):
            for key in raw_keys:
                discovered.add(str(key))
        else:
            discovered.add(str(raw_keys))

        if expected and expected.issubset(discovered):
            break

    return discovered


def create_tile_config(
    path: Path, df: pd.DataFrame, *, name: str = "TileConfiguration.txt", pixel: int = 1024
) -> None:
    """Shim: delegate to TileConfiguration.from_pos(...).write(...)."""
    TileConfiguration.from_pos(df).write(path / name)


def run_imagej(
    path: Path,
    *,
    compute_overlap: bool = False,
    fuse: bool = True,
    threshold: float | None = None,
    name: str = "TileConfiguration.txt",
    capture_output: bool = False,
    sc: StitchingConfig | None = None,
):  # shim for callers/tests
    return _run_imagej(
        path,
        compute_overlap=compute_overlap,
        fuse=fuse,
        threshold=threshold,
        name=name,
        capture_output=capture_output,
        sc=sc,
    )


def copy_registered(reference_path: Path, actual_path: Path) -> None:  # shim
    return _copy_registered(reference_path, actual_path)


def extract_channel(
    path: Path,
    out: Path,
    *,
    idx: int | None = None,
    trim: int = 0,
    max_proj: bool = False,
    downsample: int = 1,
    reduce_bit_depth: int = 0,
    sc: StitchingConfig | None = None,
) -> None:
    """
    Extract and preprocess a single channel from multi-channel TIFF image.

    Supports channel extraction, maximum projection, trimming, downsampling,
    and bit depth reduction with compressed output.

    Args:
        path: Input TIFF file path
        out: Output TIFF file path
        idx: Channel index to extract (required unless max_proj=True)
        trim: Pixels to trim from each edge
        max_proj: Compute maximum projection over Z and C dimensions
        downsample: Downsampling factor for spatial dimensions
        reduce_bit_depth: Number of bits to reduce (right shift)

    Raises:
        ValueError: If trim is negative or bit depth reduction on non-uint16
        TiffFileError: If input file is corrupted or unreadable
    """
    from fishtools.utils.pretty_print import TaskCancelledException, get_cancel_event

    cancel = get_cancel_event()

    try:
        with TiffFile(path) as tif:
            if trim < 0:
                raise ValueError("Trim must be positive")

            metadata_in = _read_tiff_metadata(tif)
            channel_names: list[str] = []

            if max_proj:
                full = tif.asarray()
                if full.ndim >= 4:
                    channel_names = _normalize_channel_names(full.shape[1], metadata_in)
                    img = full.max(axis=(0, 1))
                elif full.ndim == 3:
                    channel_names = _normalize_channel_names(full.shape[0], metadata_in)
                    img = full.max(axis=(0, 1))
                else:
                    img = full.squeeze()
            elif len(tif.pages) == 1 and tif.pages[0].asarray().ndim == 3:
                arr = tif.pages[0].asarray()
                channel_names = _normalize_channel_names(arr.shape[0], metadata_in)
                if idx is None:
                    raise ValueError("Channel index is required when extracting without max projection.")
                img = arr[idx]
                channel_names = [channel_names[idx]] if channel_names else [f"channel_{idx}"]
            else:
                if idx is None:
                    raise ValueError("Channel index is required when extracting without max projection.")
                channel_count = len(tif.pages)
                channel_names_all = _normalize_channel_names(channel_count, metadata_in)
                img = tif.pages[idx].asarray()
                channel_names = (
                    [channel_names_all[idx]]
                    if channel_names_all and idx < len(channel_names_all)
                    else [f"channel_{idx}"]
                )
    except TiffFileError as e:
        logger.critical(f"Error reading {path}: {e}")
        add_file_context(e, path)
        raise
    except Exception as exc:
        add_file_context(exc, path)
        raise

    try:
        if cancel.is_set():
            raise TaskCancelledException("Cancelled before processing")
        img = _crop_xy(img, trim)
        if downsample > 1:
            clip_range = _clip_range_for_dtype(img.dtype)
            try:
                img = gpu_downsample_xy(
                    img,
                    crop=0,
                    factor=downsample,
                    clip_range=clip_range,
                    output_dtype=img.dtype,
                )
            finally:
                # Release GPU allocator caches after downsampling
                try:
                    gpu_release_all()
                except Exception:
                    logger.opt(exception=True).debug("GPU cleanup failed after downsample; continuing.")

        if reduce_bit_depth:
            if img.dtype != np.uint16:
                raise ValueError("Cannot reduce bit depth if image is not uint16")
            img >>= reduce_bit_depth

        level = sc.compression_levels.get("low") if sc else 0.7
        if cancel.is_set():
            raise TaskCancelledException("Cancelled before write")
        metadata_out = _compose_metadata(
            "YX",
            channel_names,
            extra={
                "processing": {
                    "trim": trim,
                    "downsample": downsample,
                    "reduce_bit_depth": reduce_bit_depth,
                    "max_proj": bool(max_proj),
                }
            },
        )
        # Remove None values to keep metadata clean
        metadata_out = {k: v for k, v in metadata_out.items() if v is not None}
        try:
            safe_imwrite(
                out,
                img,
                compression=22610,
                metadata=metadata_out,
                compressionargs={"level": level},
            )
        except Exception as exc:
            add_file_context(exc, path, out)
            raise
    except Exception as exc:
        add_file_context(exc, path, out)
        raise


@click.group()
def stitch():
    """FISH image stitching pipeline for multi-tile datasets."""


@stitch.command()
@click.argument(
    "path",
    type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=Path),
)
@click.option(
    "--tileconfig",
    type=click.Path(exists=True, dir_okay=False, file_okay=True, path_type=Path),
)
@click.option("--fuse", is_flag=True)
@click.option("--downsample", type=int, default=2)
@click.option(
    "--config",
    "json_config",
    type=click.Path(exists=True, dir_okay=False, file_okay=True, path_type=Path),
    default=None,
    help="Optional project config to populate stitching defaults.",
)
def register_simple(path: Path, tileconfig: Path, fuse: bool, downsample: int, json_config: Path | None):
    # Workspace-scoped logging; logs to {workspace}/analysis/logs while
    # cooperating with Rich progress bars via the shared Console
    setup_cli_logging(
        path,
        component="preprocess.stitch.register-simple",
        file="stitch-register-simple",
        extra={"downsample": downsample, "fuse": fuse},
    )
    sc: StitchingConfig | None = None
    if json_config:
        try:
            sc = load_config(json_config).stitching
        except Exception as e:
            logger.warning(f"Failed to load config {json_config}: {e}")
    if downsample > 1:
        logger.info(f"Downsampling tile config by {downsample}x to {path / 'TileConfiguration.txt'}")
    TileConfiguration.from_file(tileconfig).downsample(downsample).write(path / "TileConfiguration.txt")

    run_imagej(
        path,
        compute_overlap=True,
        fuse=fuse,
        name="TileConfiguration",
        sc=sc,
    )


@stitch.command()
# "Path to registered images folder. Expects CYX images. Will reshape to CYX if not. Assumes ${name}-${idx}.tif",
@click.argument(
    "path",
    type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=Path),
)
@click.argument("roi", type=str, default="*")
@click.option(
    "--codebook",
    type=str,
    default=None,
    help="Codebook name if multiple codebooks are used in the same experiment.",
)
@click.option(
    "--position_file",
    type=click.Path(exists=True, dir_okay=False, file_okay=True, path_type=Path),
)
@click.option("--idx", "-i", type=int, help="Channel (index) to use for registration")
@click.option("--fid", is_flag=True)
@click.option("--threshold", type=float, default=None)
@click.option("--overwrite", is_flag=True)
@click.option("--max-proj", is_flag=True)
@click.option(
    "--config",
    "json_config",
    type=click.Path(exists=True, dir_okay=False, file_okay=True, path_type=Path),
    default=None,
    help="Optional project config to populate stitching defaults.",
)
@batch_roi()
def register(
    path: Path,
    roi: str,
    *,
    codebook: str | None = None,
    position_file: Path | None = None,
    idx: int | None = None,
    fid: bool = False,
    max_proj: bool = False,
    overwrite: bool = False,
    threshold: float | None = None,
    json_config: Path | None = None,
):
    setup_cli_logging(
        path,
        component="preprocess.stitch.register",
        file=f"stitch-register-{roi}",
        extra={"roi": roi, "codebook": codebook},
    )
    sc: StitchingConfig | None = None
    if json_config:
        try:
            sc = load_config(json_config).stitching
        except Exception as e:
            logger.warning(f"Failed to load config {json_config}: {e}")
    ws = Workspace(path)
    available_codebooks = ws.registered_codebooks(rois=[roi])

    if codebook is None:
        if len(available_codebooks) == 1:
            codebook = available_codebooks[0]
            logger.info(f"Using codebook {codebook}")
        elif not available_codebooks:
            raise ValueError(f"No registered codebooks found for ROI '{roi}' in workspace {ws.path}")
        else:
            raise ValueError(
                f"Multiple registered codebooks found for ROI '{roi}' in workspace {ws.path}. "
                "Please specify one using --codebook."
            )
    elif codebook not in available_codebooks:
        raise ValueError(
            f"Codebook '{codebook}' not found for ROI '{roi}' in workspace {ws.path}. "
            f"Available: {available_codebooks}"
        )

    registered_dir = ws.registered(roi, codebook)
    if not registered_dir.exists():
        raise ValueError(f"No registered images at {registered_dir.resolve()} found.")

    imgs = sorted(f for f in registered_dir.glob("*.tif") if not f.name.endswith(".hp.tif"))
    if not imgs:
        raise ValueError(f"No registered TIFF files found under {registered_dir}.")
    codebook_channels = _load_codebook_channels(ws, codebook)
    registered_channels = _collect_registered_channels(imgs, expected=codebook_channels)

    if not registered_channels:
        raise ValueError(
            f"Registered tiles for ROI '{roi}' with codebook '{codebook}' are missing TIFF metadata "
            "'key'; unable to validate channel coverage."
        )

    missing_channels = sorted(codebook_channels - registered_channels)
    if missing_channels:
        observed = ", ".join(sorted(registered_channels)) or "none"
        missing = ", ".join(missing_channels)
        raise ValueError(
            f"Registered tiles for ROI '{roi}' with codebook '{codebook}' are missing channel(s): "
            f"{missing}. Observed channels: {observed}."
        )

    logger.debug(
        f"Validated codebook channels for roi={roi}, codebook={codebook}: {sorted(registered_channels)}"
    )

    out_path = ws.stitch(roi)
    out_path.mkdir(exist_ok=True)

    if overwrite:
        [p.unlink() for p in out_path.glob("*.tif")]

    existing_digit_tifs = {p.name for p in out_path.glob("*.tif") if p.stem.isdigit()}

    def get_idx(img_path: Path) -> int:
        return int(img_path.stem.split("-")[1])

    if fid:
        logger.info(f"Found {len(imgs)} files. Extracting channel {idx} to {out_path}.")
        with progress_bar_threadpool(len(imgs), threads=6, stop_on_exception=True) as submit:
            for img_path in imgs:
                out_name = f"{get_idx(img_path):04d}.tif"
                if (out_path / out_name).exists() and not overwrite:
                    continue
                _idx = img_path.stem.split("-")[1]
                submit(
                    extract_channel,
                    ws.fid(roi, _idx),
                    out_path / out_name,
                    idx=0,
                    downsample=1,
                    max_proj=False,  # already max proj
                    sc=sc,
                )

    elif max_proj or idx is not None:
        logger.info(f"Found {len(imgs)} files. Extracting channel {idx} to {out_path}.")
        with progress_bar_threadpool(len(imgs), threads=6, stop_on_exception=True) as submit:
            for img_path in imgs:
                out_name = f"{get_idx(img_path):04d}.tif"
                if (out_path / out_name).exists() and not overwrite:
                    continue
                submit(
                    extract_channel,
                    img_path,
                    out_path / out_name,
                    idx=idx,
                    max_proj=max_proj,
                    sc=sc,
                )

    if overwrite or not (out_path / "TileConfiguration.registered.txt").exists():
        files = sorted(f for f in out_path.glob("*.tif") if not f.name.endswith(".hp.tif"))
        files_idx = [int(file.stem.split("-")[-1]) for file in files if file.stem.split("-")[-1].isdigit()]
        logger.debug(f"Using {files_idx}")
        tileconfig = TileConfiguration.from_pos(
            pd.read_csv(
                ws.tile_positions_csv(roi, position_file=position_file),
                header=None,
            ).iloc[sorted(files_idx)]
        )
        tileconfig.write(out_path / "TileConfiguration.txt")
        logger.info(f"Created TileConfiguration at {out_path}.")
        logger.info("Running first.")
    run_imagej(
        out_path,
        compute_overlap=True,
        fuse=False,
        threshold=threshold,
        name="TileConfiguration",
        sc=sc,
    )
    # Post-check: verify registered tile configuration and emit a layout plot
    try:
        tc_reg_path = out_path / "TileConfiguration.registered.txt"
        if not tc_reg_path.exists():
            logger.warning(
                f"Registered TileConfiguration not found at {tc_reg_path.resolve()}; ImageJ may have failed to write it."
            )
        else:
            try:
                tc = TileConfiguration.from_file(tc_reg_path)
            except Exception as exc:  # pragma: no cover - user file error
                logger.warning(f"Failed to parse {tc_reg_path.resolve()}: {exc}")
            else:
                # Identify any tiles at exactly (0, 0), which indicates non-registered tiles
                df = tc.df
                zero_mask = (df["x"] == 0.0) & (df["y"] == 0.0)
                bad = df.filter(zero_mask).sort("index")
                if len(bad) > 1:
                    extras = bad.slice(1)
                    extra_count = len(extras)
                    bad_dict = extras.select(["index", "filename"]).to_dict(as_series=False)
                    indices = bad_dict.get("index", [])
                    files = bad_dict.get("filename", [])
                    paired = ", ".join(f"{i}:{f}" for i, f in zip(indices, files))
                    logger.warning(
                        f"Detected {extra_count} additional tile(s) registered at (0,0): {paired}. "
                        "These tiles were likely not registered; inspect your input or rerun registration."
                    )

                # Save a per-ROI arrangement plot to the workspace outputs directory
                try:
                    import matplotlib.pyplot as plt

                    from fishtools.utils.plot import micron_tick_formatter, place_labels_avoid_overlap

                    fig, ax = plt.subplots(figsize=(6, 5), dpi=200)
                    # Plot points without labels then add labels with avoidance
                    tc.plot(ax, show_labels=False)
                    df_lab = tc.df
                    xs = df_lab["x"].to_numpy()
                    ys = df_lab["y"].to_numpy()
                    labels = [str(int(i)) for i in df_lab["index"].to_numpy()]
                    place_labels_avoid_overlap(ax, xs, ys, labels, fontsize=6, use_arrows=True)
                    # Convert axes to micrometers using default pixel size
                    fmt = micron_tick_formatter(0.108)
                    ax.xaxis.set_major_formatter(fmt)
                    ax.yaxis.set_major_formatter(fmt)
                    ax.set_xlabel("X (µm)")
                    ax.set_ylabel("Y (µm)")
                    ax.set_title(roi)
                    fig.tight_layout()

                    # Save under analysis/output and log absolute path
                    out_dir = ws.output
                    out_dir.mkdir(parents=True, exist_ok=True)
                    out_png = (out_dir / f"stitch_layout--{roi}.png").resolve()
                    fig.savefig(out_png.as_posix(), bbox_inches="tight")
                    plt.close(fig)
                    logger.info(f"Saved stitch layout plot: {out_png}")
                except Exception as exc:  # pragma: no cover - plotting environment issues
                    logger.warning(f"Failed to generate stitch layout plot: {exc}")
    except Exception:
        # Keep CLI resilient; log full context for diagnostics
        logger.opt(exception=True).warning("Post-registration checks encountered an error; continuing.")

    created_tile_tifs = [
        p for p in out_path.glob("*.tif") if p.stem.isdigit() and p.name not in existing_digit_tifs
    ]
    if created_tile_tifs:
        removed = 0
        for tif_path in created_tile_tifs:
            try:
                tif_path.unlink(missing_ok=True)
            except Exception as exc:
                logger.warning(f"Failed to remove intermediate tile TIFF {tif_path}: {exc}")
            else:
                removed += 1
        logger.info(f"Removed {removed} intermediate tile TIFF(s) from {out_path} after registration.")


def extract(
    path: Path,
    out_path: Path,
    *,
    trim: int = 0,
    downsample: int = 2,
    reduce_bit_depth: int = 0,
    subsample_z: int = 1,
    max_proj: bool = False,
    is_2d: bool = False,
    channels: list[int] | None = None,
    # Field NPZ path deprecated: use field_low_zarr/field_range_zarr
    field_corr: Path | None = None,
    max_from: Path | None = None,
    sc: StitchingConfig | None = None,
    workspace_root: Path | None = None,
    roi_for_ws: str | None = None,
    debug: bool = False,
    field_low_zarr: Path | None = None,
    field_range_zarr: Path | None = None,
) -> None:
    """
    Extract and format images for downstream segmentation analysis.

    Processes multi-dimensional images by extracting specific channels,
    downsampling, and organizing into directory structure suitable for
    segmentation workflows. Supports both 2D and 3D processing modes.

    Args:
        path: Input image file path
        out_path: Output directory for processed images
        trim: Pixels to trim from each edge
        downsample: Spatial downsampling factor
        reduce_bit_depth: Number of bits to reduce (right shift)
        subsample_z: Z-dimension subsampling factor
        max_proj: Compute maximum Z-projection
        is_2d: Process as 2D image (max project if 4D input)
        channels: List of channel indices to extract
        max_from: Additional file to merge for max projection

    Raises:
        ValueError: If is_2d=False but max_proj required for 4D input
        FileNotFoundError: If input files not found
    """
    try:
        with TiffFile(path) as tif:
            metadata_in = _read_tiff_metadata(tif)
            img = tif.asarray()
    except FileNotFoundError as exc:
        add_file_context(exc, path)
        raise
    except Exception as exc:
        add_file_context(exc, path)
        raise

    Path(out_path).mkdir(exist_ok=True)
    # Determine channel labels directly from metadata 'key' when present;
    # otherwise fall back to numeric labels sized from shape or requested channels
    maybe_keys = metadata_in.get("key")
    if isinstance(maybe_keys, str):
        base_channel_labels = [maybe_keys]
    elif isinstance(maybe_keys, (list, tuple, np.ndarray)):
        base_channel_labels = [
            str(x) for x in (maybe_keys.tolist() if isinstance(maybe_keys, np.ndarray) else maybe_keys)
        ]
    else:
        # Try to infer count from shape if possible (ZCYX => C at dim 1)
        inferred = (
            img.shape[1]
            if img.ndim >= 4
            else (
                img.shape[0] if (img.ndim == 3 and is_2d) else (len(channels) if channels is not None else 0)
            )
        )
        base_channel_labels = [f"channel_{i}" for i in range(inferred)]

    # We no longer synthesize labels from external files; stick to intrinsic names
    channel_labels_full = base_channel_labels
    channel_labels_selected = (
        [channel_labels_full[i] if i < len(channel_labels_full) else f"channel_{i}" for i in channels]
        if channels is not None
        else channel_labels_full
    )

    try:
        if is_2d:
            logger.info(
                f"Extract2D: start tile={path.name} channels={channels} subsample_z={subsample_z} field_corr={bool(field_corr)}"
            )
            if len(img.shape) == 4:
                if not max_proj:
                    raise ValueError(
                        "Please set is_3d to True if you want to segment 3D images or max_proj to True for max projection."
                    )
                img = img.max(axis=0)

            if img.ndim == 2:
                img = img[np.newaxis, ...]
            elif img.ndim != 3:
                raise ValueError("Unsupported image dimensionality for 2D extraction")

            if channels is not None:
                img = img[channels]

            # Apply field correction after downsample for 2D path using pre-exported field Zarr stores
            if field_low_zarr is not None and field_range_zarr is not None:
                if workspace_root is None:
                    raise ValueError("workspace_root is required when applying field Zarr stores")
                roi_name = resolve_roi_for_field(out_path, roi_for_ws)
                tile_index = parse_tile_index_from_path(path)
                x0, y0 = tile_origin(workspace_root, roi_name, tile_index)
                img = apply_field_stores_to_img(
                    img,
                    channel_labels_selected,
                    low_zarr=field_low_zarr,
                    range_zarr=field_range_zarr,
                    x0=x0,
                    y0=y0,
                    trim=int(trim),
                )
            elif field_corr is not None:
                # Legacy NPZ path removed; prefer pre-exported Zarr stores
                raise ValueError(
                    "field_corr NPZ is no longer supported. Use --field-low-zarr/--field-range-zarr."
                )

            clip_range = _clip_range_for_dtype(img.dtype)
            img = _crop_xy(img, trim)
            if downsample > 1:
                try:
                    img = gpu_downsample_xy(
                        img,
                        crop=0,
                        factor=downsample,
                        clip_range=clip_range,
                        output_dtype=img.dtype,
                    )
                finally:
                    try:
                        gpu_release_all()
                    except Exception:
                        logger.opt(exception=True).debug(
                            "GPU cleanup failed after 2D downsample; continuing."
                        )

            if reduce_bit_depth:
                img >>= reduce_bit_depth

            for i in range(img.shape[0]):
                channel_name = (
                    channel_labels_selected[i] if i < len(channel_labels_selected) else f"channel_{i}"
                )
                (out_path / f"{i:02d}").mkdir(exist_ok=True)
                target = out_path / f"{i:02d}" / (path.stem.split("-")[1] + ".tif")
                metadata_out = _compose_metadata(
                    "YX",
                    [channel_name],
                    extra={
                        "processing": {
                            "trim": trim,
                            "downsample": downsample,
                            "reduce_bit_depth": reduce_bit_depth,
                            "is_2d": True,
                            "subsample_z": subsample_z,
                            "max_proj": bool(max_proj),
                        }
                    },
                )
                metadata_out = {k: v for k, v in metadata_out.items() if v is not None}
                try:
                    safe_imwrite(
                        target,
                        img[i],
                        compression=22610,
                        metadata=metadata_out,
                        compressionargs={"level": 0.7},
                    )
                except Exception as exc:
                    add_file_context(exc, path, target)
                    raise
            del img
            return

        if len(img.shape) < 3:
            raise ValueError("Image must be at least 3D")

        if len(img.shape) == 3:
            img = img[np.newaxis, ...]
        elif len(img.shape) > 4:
            raise ValueError("Image must be 3D or 4D")

        img = img[::subsample_z]
        if channels is not None:
            img = img[:, channels]

        clip_range = _clip_range_for_dtype(img.dtype)
        img = _crop_xy(img, trim)
        if downsample > 1:
            try:
                img = gpu_downsample_xy(
                    img,
                    crop=0,
                    factor=downsample,
                    clip_range=clip_range,
                    output_dtype=img.dtype,
                )
            finally:
                try:
                    gpu_release_all()
                except Exception:
                    logger.opt(exception=True).debug("GPU cleanup failed after 3D downsample; continuing.")

        # Apply field correction after downsample for 3D path using pre-exported field Zarr stores
        if field_low_zarr is not None and field_range_zarr is not None:
            if workspace_root is None:
                raise ValueError("workspace_root is required when applying field Zarr stores")
            roi_name = resolve_roi_for_field(out_path, roi_for_ws)
            tile_index = parse_tile_index_from_path(path)
            x0, y0 = tile_origin(workspace_root, roi_name, tile_index)
            img = apply_field_stores_to_img(
                img,
                channel_labels_selected,
                low_zarr=field_low_zarr,
                range_zarr=field_range_zarr,
                x0=x0,
                y0=y0,
                trim=int(trim),
            ).astype(np.uint16)
        elif field_corr is not None:
            raise ValueError(
                "field_corr NPZ is no longer supported. Use --field-low-zarr/--field-range-zarr."
            )

        if max_proj:
            img = img.max(axis=0, keepdims=True)

        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                channel_name = (
                    channel_labels_selected[j] if j < len(channel_labels_selected) else f"channel_{j}"
                )
                p = out_path / f"{i:02d}" / f"{j:02d}"
                p.mkdir(exist_ok=True, parents=True)
                target = p / (path.stem.split("-")[1] + ".tif")
                metadata_out = _compose_metadata(
                    "YX",
                    [channel_name],
                    extra={
                        "processing": {
                            "trim": trim,
                            "downsample": downsample,
                            "reduce_bit_depth": reduce_bit_depth,
                            "subsample_z": subsample_z,
                            "max_proj": bool(max_proj),
                        },
                        "z_index": i,
                    },
                )
                metadata_out = {k: v for k, v in metadata_out.items() if v is not None}
                try:
                    safe_imwrite(
                        target,
                        img[i, j],
                        compression=22610,
                        metadata=metadata_out,
                        compressionargs={"level": 0.75},
                    )
                except Exception as exc:
                    add_file_context(exc, path, target)
                    raise

        del img
    except Exception as exc:
        add_file_context(exc, path)
        raise
    return


def walk_fused(path: Path) -> dict[int, list[Path]]:  # shim
    return _walk_fused(path)


@stitch.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=Path))
@click.argument("roi", type=str, default="*")
@click.option(
    "--tile_config",
    type=click.Path(exists=True, dir_okay=False, file_okay=True, path_type=Path),
)
@click.option("--codebook", type=str)
@click.option(
    "--split",
    type=int,
    default=1,
    help="Split tiles into this many parts. Mainly to avoid overflows in very large images.",
)
@click.option("--overwrite", is_flag=True)
@click.option("--downsample", "-d", type=int, default=2)
@click.option("--subsample-z", type=int, default=1)
@click.option("--is-2d", is_flag=True)
@click.option("--threads", "-t", type=int, default=8)
@click.option("--channels", type=str, default="all")
@click.option("--max-proj", is_flag=True)
@click.option("--debug", is_flag=True)
@click.option("--max-from", type=str)
@click.option(
    "--field-low-zarr",
    type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=Path),
    default=None,
    help=(
        "Pre-exported LOW field Zarr store (CYX, ds=1) from correct-illum export-field --what low --downsample 1."
    ),
)
@click.option(
    "--field-range-zarr",
    type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=Path),
    default=None,
    help=(
        "Pre-exported RANGE field Zarr store (CYX, ds=1) from correct-illum export-field --what range --downsample 1."
    ),
)
@click.option(
    "--json-config",
    type=click.Path(exists=True, dir_okay=False, file_okay=True, path_type=Path),
    default=None,
    help="Optional project config to populate stitching defaults.",
)
# @click.option("--skip-extract", is_flag=True)
@batch_roi("registered--*", include_codebook=True, split_codebook=True)
def fuse(
    path: Path,
    roi: str,
    codebook: str,
    *,
    tile_config: Path | None = None,
    split: int = 1,
    overwrite: bool = False,
    downsample: int = 2,
    is_2d: bool = False,
    threads: int = 8,
    channels: str = "all",
    subsample_z: int = 1,
    max_proj: bool = False,
    debug: bool = False,
    max_from: str | None = None,
    json_config: Path | None = None,
    # skip_extract: bool = False,
    field_low_zarr: Path | None = None,
    field_range_zarr: Path | None = None,
):
    setup_cli_logging(
        path,
        component="preprocess.stitch.fuse",
        file=f"stitch-fuse-{roi}+{codebook}",
        extra={"roi": roi, "codebook": codebook, "threads": threads},
    )
    ws = Workspace(path)
    sc: StitchingConfig | None = None
    if json_config:
        try:
            sc = load_config(json_config).stitching
        except Exception as e:
            logger.warning(f"Failed to load config {json_config}: {e}")
    if "--" in path.as_posix():
        raise ValueError("Please be in the workspace folder.")

    stitch_dir = ws.stitch(roi, codebook)
    existing_fused = [
        candidate
        for candidate in (stitch_dir / "fused.zarr", stitch_dir / "fused_n4.zarr")
        if candidate.exists()
    ]
    if existing_fused and not overwrite:
        existing_names = ", ".join(sorted(p.name for p in existing_fused))
        logger.info(
            f"Skipping ROI '{roi}' with codebook '{codebook}' — existing fused outputs ({existing_names}) present. "
            "Re-run with --overwrite to regenerate."
        )
        return

    path_img = ws.registered(roi, codebook)
    path = stitch_dir
    files = sorted(path_img.glob("*.tif"))
    if not len(files):
        raise ValueError(f"No images found at {path_img.resolve()}")
    logger.info(f"Found {len(files)} images at {path_img.resolve()}")

    skip_extract = not overwrite

    if overwrite and path.exists():
        for p in path.iterdir():
            if p.is_dir():
                shutil.rmtree(p)

    correct_count = len(list(path_img.glob("reg*.tif")))

    if skip_extract:
        try:
            folders_existing = list(chain.from_iterable(walk_fused(path).values()))
        except ValueError:
            skip_extract = False
        else:
            for folder in folders_existing:
                current_count = len([p for p in folder.glob("*.tif") if p.stem.isdigit()])
                if current_count != correct_count:
                    logger.info(
                        f"Incorrect number of files in {folder} ({current_count} != {correct_count}). Running extract."
                    )
                    skip_extract = False
                    break

    if tile_config is None:
        tile_config = ws.tileconfig_dir(roi) / "TileConfiguration.registered.txt"
        logger.info(f"Getting tile configuration from {tile_config.resolve()}")

    tileconfig = TileConfiguration.from_file(tile_config).downsample(downsample)
    n = len(tileconfig) // split

    if channels == "all":
        try:
            first_image = imread(files[0])
        except Exception as exc:
            add_file_context(exc, files[0])
            raise
        channels = ",".join(map(str, range(first_image.shape[1])))

    channel_indices = [int(c) for c in channels.split(",") if c]

    try:
        with TiffFile(files[0]) as tif_first:
            metadata_first = _read_tiff_metadata(tif_first)
        key_raw = metadata_first.get("key")
        if isinstance(key_raw, str):
            channel_names_all = _normalize_channel_names(1, metadata_first)
        elif isinstance(key_raw, (list, tuple)):
            channel_names_all = _normalize_channel_names(len(key_raw), metadata_first)
        elif isinstance(key_raw, np.ndarray):
            channel_names_all = _normalize_channel_names(len(key_raw), metadata_first)
        else:
            count_guess = (max(channel_indices) + 1) if channel_indices else 0
            channel_names_all = _normalize_channel_names(count_guess, metadata_first)
    except Exception:
        count_guess = (max(channel_indices) + 1) if channel_indices else 0
        channel_names_all = [f"channel_{i}" for i in range(count_guess)]

    if channel_indices:
        channel_labels_selected = [
            channel_names_all[i] if i < len(channel_names_all) else f"channel_{i}" for i in channel_indices
        ]
    else:
        channel_labels_selected = channel_names_all

    max_from_path = Path(max_from) if max_from else None
    max_from_label = _label_for_max_from(max_from_path)
    if max_from_label and max_from_label not in channel_labels_selected:
        channel_labels_selected.append(max_from_label)

    channel_labels_by_position = {position: label for position, label in enumerate(channel_labels_selected)}

    imgs = {int(p.stem.split("-")[1]) for p in path_img.glob("*.tif")}
    needed = set(tileconfig.df["index"])

    if len(needed & imgs) != len(needed):
        tileconfig = tileconfig.drop(list(needed - imgs))
        logger.warning(f"Not all images are present in {path_img}. Missing: {needed - imgs}. Dropping.")

    # When using pre-exported fields, enforce ds=1 for alignment
    if (field_low_zarr or field_range_zarr) and int(downsample) != 1:
        raise ValueError("When --field-low-zarr/--field-range-zarr are provided, please set --downsample 1.")

    # Sanity: both LOW and RANGE must be supplied together
    if (field_low_zarr is None) ^ (field_range_zarr is None):
        raise ValueError("Provide both --field-low-zarr and --field-range-zarr (or neither).")

    if skip_extract:
        logger.info(f"Reusing previously extracted tiles at {path}. Use --overwrite to regenerate.")
    else:
        logger.info(f"Found {len(files)} files. Extracting channel {channels} to {path}")
        with progress_bar_threadpool(len(files), threads=threads, stop_on_exception=True) as submit:
            for file in files:
                submit(
                    extract,
                    file,
                    path,
                    downsample=downsample,
                    subsample_z=subsample_z,
                    is_2d=is_2d,
                    channels=channel_indices.copy(),
                    max_proj=max_proj,
                    max_from=ws.registered(roi, max_from) / file.name if max_from else None,
                    sc=sc,
                    workspace_root=ws.path,
                    roi_for_ws=roi,
                    debug=debug,
                    field_low_zarr=field_low_zarr,
                    field_range_zarr=field_range_zarr,
                )

    def run_folder(folder: Path, capture_output: bool = False):
        def log_progress(message: str) -> None:
            if capture_output:
                logger.info(message)
            else:
                get_shared_console().log(message)
                logger.bind(**{CONSOLE_SKIP_EXTRA: True}).info(message)

        for i in range(split):
            tileconfig[i * n : (i + 1) * n].write(folder / f"TileConfiguration{i + 1}.registered.txt")
            try:
                start = perf_counter()
                log_progress(
                    f"Starting ImageJ fuse run for {folder} using TileConfiguration{i + 1}.registered.txt "
                    f"(split_index={i + 1}, capture_output={capture_output})"
                )
                run_imagej(
                    folder,
                    name=f"TileConfiguration{i + 1}",
                    capture_output=capture_output,
                    sc=sc,
                )
            except Exception as e:
                logger.critical(f"Error running ImageJ for {folder}: {e}")
                raise e
            Path(folder / "img_t1_z1_c1").rename(folder / f"fused_{folder.name}-{i + 1}.tif")
            duration = perf_counter() - start
            log_progress(f"Completed ImageJ fuse run for {folder} chunk {i + 1}/{split} in {duration:.2f}s")

    # Get all folders without subfolders
    folders = list(chain.from_iterable(walk_fused(path).values()))

    logger.info(f"Calling ImageJ on {len(folders)} folders.")
    to_runs = []
    for folder in folders:
        if not folder.is_dir():
            raise Exception("Invalid folder")
        if not folder.name.isdigit():
            raise ValueError(f"Invalid folder name {folder.name}. No external folders allowed.")

        existings = list(folder.glob("fused*"))
        if existings and not overwrite:
            logger.warning(f"{existings} already exists. Skipping this folder.")
            continue
        to_runs.append(folder)

    if not len(to_runs):
        logger.warning("No folders to run.")
        return

    with progress_bar_threadpool(len(to_runs), threads=threads, stop_on_exception=True) as submit:
        for folder in to_runs:
            submit(run_folder, folder, capture_output=not debug)

    if split > 1:
        for folder in folders:
            channel_position = int(folder.name)
            channel_name = channel_labels_by_position.get(channel_position)
            final_stitch(folder, split, channel_name=channel_name, sc=sc)


def numpy_array_to_zarr(write_path: Path | str, array: np.ndarray, chunks: tuple[int, ...]):  # shim
    return _numpy_array_to_zarr(write_path, array, chunks)


@stitch.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=Path))
@click.argument("roi", type=str, default="*")
@click.option("--codebook", type=str)
@click.option("--chunk-size", type=int, default=2048)
@click.option("--overwrite", is_flag=True)
@batch_roi("stitch--*", include_codebook=True, split_codebook=True)
def combine(path: Path, roi: str, codebook: str, chunk_size: int = 2048, overwrite: bool = True):
    setup_cli_logging(
        path,
        component="preprocess.stitch.combine",
        file=f"stitch-combine-{roi}+{codebook}",
        extra={"roi": roi, "codebook": codebook, "chunk_size": chunk_size},
    )
    import zarr

    from fishtools.io.workspace import Workspace

    ws = Workspace(path)
    target_rois = ws.resolve_rois(None if roi == "*" else [roi])

    for current_roi in target_rois:
        stitched_dir = ws.stitch(current_roi, codebook)
        # Group folders by Z index (parent directory name)
        try:
            folders_by_z = walk_fused(stitched_dir)
        except ValueError:
            logger.warning(
                f"No valid stitched folders found under {stitched_dir}. Skipping ROI '{current_roi}'."
            )
            continue
        # Sort folders within each Z index by C index (folder name)
        for z_idx in folders_by_z:
            folders_by_z[z_idx].sort(key=lambda f: int(f.name))

        if not folders_by_z:
            logger.warning(f"No stitched content discovered for ROI '{current_roi}'. Skipping.")
            continue

        zs = max(folders_by_z.keys()) + 1
        cs = max(int(f.name) for f in folders_by_z[0]) + 1  # Assume C count is same for all Z

        # Check for fused images in the first Z plane to get dimensions
        first_z_folders = folders_by_z[0]
        missing = [
            folder for folder in first_z_folders if not (folder / f"fused_{folder.name}-1.tif").exists()
        ]
        if missing:
            logger.warning(
                f"Missing fused images for ROI '{current_roi}' in Z=0: {[m.name for m in missing]}. Skipping ROI."
            )
            continue

        # Get shape from the first image of the first Z plane
        first_folder = first_z_folders[0]
        first_img = imread(first_folder / f"fused_{first_folder.name}-1.tif")
        img_shape = first_img.shape
        dtype = first_img.dtype
        final_shape = (zs, img_shape[0], img_shape[1], cs)
        logger.info(f"Final Zarr shape: {final_shape}, dtype: {dtype}")

        # Initialize the Zarr array
        zarr_path = stitched_dir / "fused.zarr"
        logger.info(f"Writing to {zarr_path.resolve()}")
        # Define chunks: chunk along Z=1, use user chunk_size for Y/X, full chunk for C
        zarr_chunks = (1, chunk_size, chunk_size, cs)
        try:
            z_array = zarr.create_array(
                zarr_path,
                shape=final_shape,
                chunks=zarr_chunks,
                dtype=dtype,
                overwrite=True,
            )
        except AttributeError:
            # New zarr API
            z_array = zarr.open_array(
                zarr_path,
                mode="w",
                shape=final_shape,
                chunks=zarr_chunks,
                dtype=dtype,
            )

        # Create thumbnail directory
        thumbnail_dir = stitched_dir / "thumbnails"
        thumbnail_dir.mkdir(exist_ok=True)

        with progress_bar(len(folders_by_z)) as progress:
            for i in sorted(folders_by_z.keys()):
                z_plane_folders = folders_by_z[i]
                z_plane_data = np.zeros((img_shape[0], img_shape[1], cs), dtype=dtype)
                thumbnail_data = None

                for folder in z_plane_folders:
                    j = int(folder.name)
                    img_path = folder / f"fused_{folder.name}-1.tif"
                    try:
                        img = imread(img_path)
                    except FileNotFoundError:
                        logger.warning(f"File not found while combining ROI '{current_roi}': {img_path}")
                        raise
                    except Exception as e:
                        logger.warning(
                            f"Error reading image while combining ROI '{current_roi}': {img_path}: {e}"
                        )
                        raise

                    z_plane_data[:, :, j] = img[:, :]

                    if thumbnail_data is None:
                        preview_c = min(3, cs)
                        thumbnail_data = np.zeros((img.shape[0], img.shape[1], preview_c), dtype=np.uint16)
                    if j < thumbnail_data.shape[2]:
                        thumbnail_data[:, :, j] = img[:, :]

                    del img

                logger.info(f"Writing Z-plane {i + 1}/{zs} to Zarr array")
                z_array[i, :, :, :] = z_plane_data

                if i % 8 == 0 and thumbnail_data is not None:
                    # Save as PNG; ensure we have 3 channels for RGB
                    td = (thumbnail_data[::8, ::8] >> 10).astype(np.uint8)
                    if td.ndim == 2:
                        td = np.repeat(td[:, :, None], 3, axis=2)
                    elif td.shape[2] == 1:
                        td = np.repeat(td, 3, axis=2)
                    elif td.shape[2] == 2:
                        td = np.concatenate([td, np.zeros_like(td[:, :, :1])], axis=2)
                    thumbnail_img = Image.fromarray(td, mode="RGB")
                    thumbnail_path = thumbnail_dir / f"thumbnail_z{i:03d}.png"
                    thumbnail_img.save(thumbnail_path)
                    logger.debug(f"Saved thumbnail for Z-plane {i} to {thumbnail_path}")

                progress()

        # Add metadata (channel names)
        try:
            first_reg_file = next(ws.registered(current_roi, codebook).glob("*.tif"))
            with TiffFile(first_reg_file) as tif:
                # Attempt to get channel names, handle potential errors
                names = tif.shaped_metadata[0].get("key") if tif.shaped_metadata else None

            if names:
                if cs == len(names) + 1:
                    names = names + ["spots"]
                z_array.attrs["key"] = names
                logger.info(f"Added channel names: {names}")
            else:
                logger.warning("Could not find channel names ('key') in TIF metadata.")
        except StopIteration:
            logger.warning(
                f"No registered TIF file found in {ws.registered(current_roi, codebook)} to read channel names."
            )
        except Exception as e:
            logger.warning(f"Error reading metadata from TIF file: {e}")

        logger.info("Deleting source folders.")
        all_folders = [f for z_folders in folders_by_z.values() for f in z_folders]
        for folder in all_folders:
            try:
                shutil.rmtree(folder.parent)  # Remove the Z-level parent folder
            except FileNotFoundError:
                ...
        logger.info("Done.")


@stitch.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=Path))
@click.argument("roi", type=str, default="*")
@click.option("--codebook", type=str, required=True, help="Codebook used in the stitch directory name.")
@click.option(
    "--channels",
    type=str,
    default=None,
    help="Comma-separated channel names or indices to correct (defaults to all channels).",
)
@click.option("--shrink", type=int, default=4, show_default=True, help="Downsample factor before running N4.")
@click.option(
    "--spline-lowres-px",
    type=float,
    default=128.0,
    show_default=True,
    help="Desired spline control-point spacing on the downsampled grid (in pixels).",
)
@click.option(
    "--z-index", type=int, required=True, help="Z index (0-based) used to compute the correction field."
)
@click.option(
    "--threshold",
    type=str,
    default=None,
    help="Foreground mask threshold (numeric or skimage.filters method name).",
)
@click.option(
    "--field-output",
    type=click.Path(exists=False, dir_okay=False, file_okay=True, path_type=Path),
    default=None,
    help="Optional output path for the correction field TIFF.",
)
@click.option(
    "--corrected-output",
    type=click.Path(exists=False, dir_okay=True, file_okay=True, path_type=Path),
    default=None,
    help="Optional output path for corrected imagery (defaults to fused_n4.zarr).",
)
@click.option(
    "--apply/--field-only",
    default=True,
    show_default=True,
    help="Apply the correction field to imagery (enabled by default).",
)
@click.option("--overwrite", is_flag=True, help="Overwrite existing outputs.")
@click.option("--single-plane", is_flag=True, help="Only correct the specified Z plane (debugging aid).")
@click.option("--debug", is_flag=True, help="Write float32 debug outputs alongside quantized corrections.")
@click.option(
    "--unsharp-mask/--no-unsharp-mask",
    default=True,
    show_default=True,
    help="Pre-filter the N4 source plane with cucim.skimage.filters.unsharp_mask (requires GPU/CuPy).",
)
@batch_roi("stitch--*", include_codebook=True, split_codebook=True)
def n4(
    path: Path,
    roi: str,
    codebook: str,
    *,
    channels: str | None,
    shrink: int,
    spline_lowres_px: float,
    z_index: int,
    threshold: str | None,
    field_output: Path | None,
    corrected_output: Path | None,
    apply: bool,
    overwrite: bool,
    single_plane: bool,
    debug: bool,
    unsharp_mask: bool,
) -> None:
    """Run N4 bias-field correction against stitched mosaics."""

    setup_cli_logging(
        path,
        component="preprocess.stitch.n4",
        file=f"stitch-n4-{roi}+{codebook}",
        extra={"roi": roi, "codebook": codebook},
    )

    try:
        results = run_cli_workflow(
            workspace=path,
            roi=roi,
            codebook=codebook,
            channels=channels,
            shrink=shrink,
            spline_lowres_px=spline_lowres_px,
            z_index=z_index,
            threshold=threshold,
            field_output=field_output,
            corrected_output=corrected_output,
            apply_correction=apply,
            overwrite=overwrite,
            single_plane=single_plane,
            debug=debug,
            use_unsharp_mask=unsharp_mask,
        )
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc

    if not results:
        click.echo("No correction results returned.")
        return

    result = results[0]
    click.echo(f"Correction field saved to {result.field_path}")
    if result.corrected_path is not None:
        click.echo(f"Corrected imagery saved to {result.corrected_path}")


@stitch.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=Path))
@click.argument("roi", type=str, default="*")
@click.option(
    "--tile_config",
    type=click.Path(exists=True, dir_okay=False, file_okay=True, path_type=Path),
)
@click.option("--codebook", type=str)
@click.option("--overwrite", is_flag=True)
@click.option("--downsample", "-d", type=int, default=2)
@click.option("--subsample-z", type=int, default=1)
@click.option("--threads", "-t", type=int, default=8)
@click.option("--channels", type=str, default="-3,-2,-1")
@batch_roi()
def run(
    path: Path,
    roi: str,
    *,
    codebook: str,
    tile_config: Path | None = None,
    overwrite: bool = False,
    downsample: int = 2,
    threads: int = 8,
    channels: str = "-3,-2,-1",
    subsample_z: int = 1,
):
    setup_cli_logging(
        path,
        component="preprocess.stitch.run",
        file=f"stitch-run-{roi}",
        extra={"roi": roi, "codebook": codebook},
    )
    register.callback(path, roi=roi, position_file=None, fid=True)
    fuse.callback(
        path,
        roi=roi,
        codebook=codebook,
        tile_config=tile_config,
        split=True,
        overwrite=overwrite,
        downsample=downsample,
        threads=threads,
        channels=channels,
        subsample_z=subsample_z,
    )
    combine.callback(path, roi=roi, codebook=codebook)


def final_stitch(
    path: Path,
    n: int,
    *,
    channel_name: str | None = None,
    sc: StitchingConfig | None = None,
):
    logger.info(f"Combining splits of {n} for {path}.")
    import polars as pl

    tcs = [TileConfiguration.from_file(f"{path}/TileConfiguration{i + 1}.registered.txt") for i in range(n)]

    bmin = pl.concat([tc.df.min() for tc in tcs])
    bmax = pl.concat([tc.df.max() for tc in tcs])

    mins = (bmin.min()[0, "y"], bmin.min()[0, "x"])
    maxs = (bmax.max()[0, "y"] + 1024, bmax.max()[0, "x"] + 1024)

    out = np.zeros((n, int(maxs[0] - mins[0] + 1), int(maxs[1] - mins[1] + 1)), dtype=np.uint16)

    for i in range(n):
        img = imread(f"{path}/fused_{i + 1}.tif")
        offsets = (int(bmin[i, "y"] - mins[i]), int(bmin[i, "x"] - mins[i]))
        out[
            i,
            offsets[0] : img.shape[0] + offsets[0],
            offsets[1] : img.shape[1] + offsets[1],
        ] = img

    out = out.max(axis=0)
    default_label = f"channel_{path.name}" if path.name.isdigit() else path.name
    label = channel_name or default_label
    metadata_out = _compose_metadata(
        "YX",
        [label] if label else None,
        extra={
            "processing": {
                "split": n,
            }
        },
    )
    metadata_out = {k: v for k, v in metadata_out.items() if v is not None}
    level = sc.compression_levels.get("high", 0.8) if sc else 0.8
    safe_imwrite(
        path / "fused.tif",
        out,
        compression=22610,
        compressionargs={"level": level},
        metadata=metadata_out,
    )
    del out


if __name__ == "__main__":
    stitch()

# %%
