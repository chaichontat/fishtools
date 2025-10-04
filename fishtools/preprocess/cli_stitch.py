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

import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import chain
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import rich_click as click
from loguru import logger
from PIL import Image
from tifffile import TiffFile, TiffFileError, imread, imwrite

from fishtools.gpu.memory import release_all as gpu_release_all
from fishtools.io.workspace import Workspace
from fishtools.preprocess.config import NumpyEncoder, StitchingConfig
from fishtools.preprocess.config_loader import load_config
from fishtools.preprocess.downsample import gpu_downsample_xy
from fishtools.preprocess.imagej import run_imagej as _run_imagej
from fishtools.preprocess.imageops import clip_range_for_dtype as clip_range_for_dtype_lib
from fishtools.preprocess.imageops import crop_xy as crop_xy_lib
from fishtools.preprocess.stitching import walk_fused as _walk_fused
from fishtools.preprocess.tileconfig import TileConfiguration
from fishtools.preprocess.tileconfig import copy_registered as _copy_registered
from fishtools.utils.logging import setup_workspace_logging
from fishtools.utils.pretty_print import progress_bar, progress_bar_threadpool
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
            imwrite(
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


@stitch.command("n4", context_settings={"ignore_unknown_options": True})
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
@click.pass_context
def n4_cmd(ctx, args):
    """Run N4 bias-field correction on fused Zarr (delegates to fishtools.preprocess.n4).

    Usage example:
    preprocess stitch n4 /workspace roi --codebook cb --z-index 0 [--channels polyA,reddot] [--debug]

    This is a thin wrapper around the Typer CLI in scripts/n4.py to keep the
    command discoverable under the stitch family.
    """
    import subprocess
    import sys

    if not args:
        args = ["--help"]

    cmd = [sys.executable, "-m", "fishtools.preprocess.n4"] + list(args)
    try:
        result = subprocess.run(cmd, check=False)
        ctx.exit(result.returncode)
    except KeyboardInterrupt:
        ctx.exit(1)


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
    setup_workspace_logging(path, component="preprocess.stitch.register-simple", file="register_simple")
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
    position_file: Path | None = None,
    idx: int | None = None,
    fid: bool = False,
    max_proj: bool = False,
    overwrite: bool = False,
    threshold: float | None = None,
    json_config: Path | None = None,
):
    setup_workspace_logging(path, component="preprocess.stitch.register", file=roi)
    sc: StitchingConfig | None = None
    if json_config:
        try:
            sc = load_config(json_config).stitching
        except Exception as e:
            logger.warning(f"Failed to load config {json_config}: {e}")
    base_path = path
    path = next(path.glob(f"registered--{roi}*"))
    if not path.exists():
        raise ValueError(f"No registered images at {path.resolve()} found.")

    imgs = sorted(f for f in path.glob("*.tif") if not f.name.endswith(".hp.tif"))
    (out_path := path.parent / f"stitch--{roi.split('+')[0]}").mkdir(exist_ok=True)

    if overwrite:
        [p.unlink() for p in out_path.glob("*.tif")]

    def get_idx(path: Path):
        return int(path.stem.split("-")[1])

    if fid:
        logger.info(f"Found {len(imgs)} files. Extracting channel {idx} to {out_path}.")
        with progress_bar(len(imgs)) as callback, ThreadPoolExecutor(6) as exc:
            futs = []
            for path in imgs:
                out_name = f"{get_idx(path):04d}.tif"
                if (out_path / out_name).exists() and not overwrite:
                    continue
                _idx = path.stem.split("-")[1]
                futs.append(
                    exc.submit(
                        extract_channel,
                        path.parent.parent / f"fids--{roi}" / f"fids-{_idx}.tif",
                        out_path / out_name,
                        idx=0,
                        downsample=1,
                        max_proj=False,  # already max proj
                        sc=sc,
                    )
                )

            for f in as_completed(futs):
                f.result()
                callback()

    elif max_proj or idx is not None:
        logger.info(f"Found {len(imgs)} files. Extracting channel {idx} to {out_path}.")
        with progress_bar(len(imgs)) as callback, ThreadPoolExecutor(6) as exc:
            futs = []
            for path in imgs:
                out_name = f"{get_idx(path):04d}.tif"
                if (out_path / out_name).exists() and not overwrite:
                    continue
                futs.append(
                    exc.submit(
                        extract_channel,
                        path,
                        out_path / out_name,
                        idx=idx,
                        max_proj=max_proj,
                        sc=sc,
                    )
                )

            for f in as_completed(futs):
                f.result()
                callback()

    del path
    if overwrite or not (out_path / "TileConfiguration.registered.txt").exists():
        files = sorted(f for f in out_path.glob("*.tif") if not f.name.endswith(".hp.tif"))
        files_idx = [int(file.stem.split("-")[-1]) for file in files if file.stem.split("-")[-1].isdigit()]
        logger.debug(f"Using {files_idx}")
        tileconfig = TileConfiguration.from_pos(
            pd.read_csv(
                position_file
                or (
                    base_path / f"{roi}.csv"
                    if (base_path / f"{roi}.csv").exists()
                    else (base_path.resolve().parent.parent / f"{roi}.csv")
                ),
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
    max_from: Path | None = None,
    sc: StitchingConfig | None = None,
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
                    imwrite(
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
                    imwrite(
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
):
    setup_workspace_logging(path, component="preprocess.stitch.fuse", file=f"{roi}+{codebook}")
    ws = Workspace(path)
    sc: StitchingConfig | None = None
    if json_config:
        try:
            sc = load_config(json_config).stitching
        except Exception as e:
            logger.warning(f"Failed to load config {json_config}: {e}")
    if "--" in path.as_posix():
        raise ValueError("Please be in the workspace folder.")

    path_img = ws.deconved / f"registered--{roi}+{codebook}"
    path = ws.stitch(roi, codebook)
    files = sorted(path_img.glob("*.tif"))
    if not len(files):
        raise ValueError(f"No images found at {path_img.resolve()}")
    logger.info(f"Found {len(files)} images at {path_img.resolve()}")

    if overwrite and path.exists():
        for p in path.iterdir():
            if p.is_dir():
                shutil.rmtree(p)

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

    skip_extract = True
    correct_count = len(list(path_img.glob("reg*.tif")))

    try:
        folders = list(chain.from_iterable(walk_fused(path).values()))
    except ValueError:
        skip_extract = False
    else:
        for folder in folders:
            if (_cnt := len([p for p in folder.glob("*.tif") if p.stem.isdigit()])) != correct_count:
                logger.info(
                    f"Incorrect number of files in {folder} ({_cnt} != {correct_count}). Running extract."
                )
                skip_extract = False
                break

    if overwrite or not skip_extract:
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
                    max_from=file.parent.parent / f"registered--{roi}+{max_from}" / file.name
                    if max_from
                    else None,
                    sc=sc,
                )

    def run_folder(folder: Path, capture_output: bool = False):
        for i in range(split):
            tileconfig[i * n : (i + 1) * n].write(folder / f"TileConfiguration{i + 1}.registered.txt")
            try:
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

    # Get all folders without subfolders
    folders = list(chain.from_iterable(walk_fused(path).values()))

    logger.info(f"Calling ImageJ on {len(folders)} folders.")
    with progress_bar_threadpool(len(folders), threads=threads, stop_on_exception=True) as submit:
        for folder in folders:
            if not folder.is_dir():
                raise Exception("Invalid folder")
            if not folder.name.isdigit():
                raise ValueError(f"Invalid folder name {folder.name}. No external folders allowed.")

            existings = list(folder.glob("fused*"))
            if existings and not overwrite:
                logger.warning(f"{existings} already exists. Skipping this folder.")
                continue
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
    setup_workspace_logging(path, component="preprocess.stitch.combine", file=f"{roi}+{codebook}")
    import zarr

    from fishtools.io.workspace import Workspace

    ws = Workspace(path)
    rois = ws.rois if roi == "*" else [roi]
    for roi in rois:
        path = Path(path / f"stitch--{roi}+{codebook}")
        # Group folders by Z index (parent directory name)
        folders_by_z = walk_fused(path)
        # Sort folders within each Z index by C index (folder name)
        for z_idx in folders_by_z:
            folders_by_z[z_idx].sort(key=lambda f: int(f.name))

        zs = max(folders_by_z.keys()) + 1
        cs = max(int(f.name) for f in folders_by_z[0]) + 1  # Assume C count is same for all Z

        # Check for fused images in the first Z plane to get dimensions
        first_z_folders = folders_by_z[0]
        for folder in first_z_folders:
            if not (folder / f"fused_{folder.name}-1.tif").exists():
                raise ValueError(f"No fused image found for {folder.name} in Z=0")

        # Get shape from the first image of the first Z plane
        first_folder = first_z_folders[0]
        first_img = imread(first_folder / f"fused_{first_folder.name}-1.tif")
        img_shape = first_img.shape
        dtype = first_img.dtype
        final_shape = (zs, img_shape[0], img_shape[1], cs)
        logger.info(f"Final Zarr shape: {final_shape}, dtype: {dtype}")

        # Initialize the Zarr array
        zarr_path = path / "fused.zarr"
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

    z_plane_data = np.zeros((img_shape[0], img_shape[1], cs), dtype=dtype)

    # Create thumbnail directory
    thumbnail_dir = path / "thumbnails"
    thumbnail_dir.mkdir(exist_ok=True)

    with progress_bar(len(folders_by_z)) as progress:
        for i in sorted(folders_by_z.keys()):
            z_plane_folders = folders_by_z[i]
            thumbnail_data = None
            for folder in z_plane_folders:
                j = int(folder.name)
                try:
                    img = imread(folder / f"fused_{folder.name}-1.tif")
                    # Write into the C dimension of the current Z-plane array
                    z_plane_data[:, :, j] = img[:, :]
                    if thumbnail_data is None:
                        # Limit preview to at most 3 channels (RGB)
                        preview_c = min(3, cs)
                        thumbnail_data = np.zeros((img.shape[0], img.shape[1], preview_c), dtype=np.uint16)
                    if j < thumbnail_data.shape[2]:
                        thumbnail_data[:, :, j] = img[:, :]
                    del img
                except FileNotFoundError:
                    raise FileNotFoundError(f"File not found: {folder / f'fused_{folder.name}-1.tif'}")
                except Exception as e:
                    raise Exception(f"Error reading {folder / f'fused_{folder.name}-1.tif'}") from e

            logger.info(f"Writing Z-plane {i}/{zs - 1} to Zarr array")
            z_array[i, :, :, :] = z_plane_data

            if i % 8 == 0:
                assert thumbnail_data is not None

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
        first_reg_file = next((path.parent / f"registered--{roi}+{codebook}").glob("*.tif"))
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
            f"No registered TIF file found in {path.parent / f'registered--{roi}'} to read channel names."
        )
    except Exception as e:
        logger.warning(f"Error reading metadata from TIF file: {e}")

    # Normalization calculation removed by request; downstream stages may compute on demand.

    logger.info("Deleting source folders.")
    all_folders = [f for z_folders in folders_by_z.values() for f in z_folders]

    for folder in all_folders:
        try:
            shutil.rmtree(folder.parent)  # Remove the Z-level parent folder
        except FileNotFoundError:
            # Can happen since some folder in all_folders are subdirectories of others
            ...
    logger.info("Done.")


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
    setup_workspace_logging(path, component="preprocess.stitch.run", file=roi)
    register.callback(path, roi=roi, position_file=None, fid=True)
    fuse.callback(
        path,
        roi=f"{roi}+{codebook}",
        tile_config=tile_config,
        split=True,
        overwrite=overwrite,
        downsample=downsample,
        threads=threads,
        channels=channels,
        subsample_z=subsample_z,
    )
    combine.callback(path, roi=f"{roi}+{codebook}", threads=threads)


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
    imwrite(
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
