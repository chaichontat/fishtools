import json
import shutil
import subprocess
from collections.abc import Callable
from itertools import chain
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import rich_click as click
from click.core import ParameterSource
from loguru import logger
from tifffile import TiffFile, TiffFileError, imread

from fishtools import IMWRITE_KWARGS
from fishtools.gpu.memory import release_all as gpu_release_all
from fishtools.io.workspace import Workspace, safe_imwrite
from fishtools.preprocess.config import StitchingConfig
from fishtools.preprocess.config_loader import load_config
from fishtools.preprocess.downsample import downsample_xy, gpu_downsample_xy
from fishtools.preprocess.illumination import parse_tile_index_from_path, resolve_roi_for_field, tile_origin
from fishtools.preprocess.imagej import run_imagej as _run_imagej
from fishtools.preprocess.imageops import clip_range_for_dtype, crop_xy
from fishtools.preprocess.stitching import walk_fused as _walk_fused
from fishtools.preprocess.tileconfig import TileConfiguration
from fishtools.preprocess.tileconfig import copy_registered as _copy_registered
from fishtools.utils.logging import CONSOLE_SKIP_EXTRA, get_shared_console, setup_cli_logging
from fishtools.utils.pretty_print import progress_bar, progress_bar_threadpool
from fishtools.utils.tiff import compose_metadata, normalize_channel_names, read_metadata_from_tif
from fishtools.utils.utils import add_file_context, batch_roi
from fishtools.utils.thumbnails import load_thumbnail_options, save_thumbnail_png
from fishtools.utils.zarr_utils import create_zarr_array
from fishtools.utils.zarr_utils import numpy_array_to_zarr as _numpy_array_to_zarr

run_cli_workflow: Callable[..., Any] | None = None


def _get_run_cli_workflow() -> Callable[..., Any]:
    global run_cli_workflow  # noqa: PLW0603 - intended shared cache
    if run_cli_workflow is None:
        from fishtools.preprocess.n4 import run_cli_workflow as _run_cli_workflow

        run_cli_workflow = _run_cli_workflow
    return run_cli_workflow


def _label_for_max_from(path: Path | None) -> str | None:
    if path is None:
        return None
    label = f"max_from:{path.stem}"
    try:
        with TiffFile(path) as tif:
            metadata = read_metadata_from_tif(tif)
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
    path: Path,
    df: pd.DataFrame,
    *,
    name: str = "TileConfiguration.txt",
    pixel: int = 1024,
    pixel_size_um: float = 0.108,
) -> None:
    TileConfiguration.from_pos(df, pixel_size_um=pixel_size_um).write(path / name)


def run_imagej(
    path: Path,
    *,
    compute_overlap: bool = False,
    fuse: bool = True,
    threshold: float | None = None,
    name: str = "TileConfiguration.txt",
    capture_output: bool = False,
    stream_to_console: bool = False,
    sc: StitchingConfig | None = None,
):  # shim for callers/tests
    return _run_imagej(
        path,
        compute_overlap=compute_overlap,
        fuse=fuse,
        threshold=threshold,
        name=name,
        capture_output=capture_output,
        stream_to_console=stream_to_console,
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
    """Extract one channel (or max-proj) and write as a YX TIFF."""
    from fishtools.utils.pretty_print import TaskCancelledException, get_cancel_event

    cancel = get_cancel_event()

    try:
        with TiffFile(path) as tif:
            if trim < 0:
                raise ValueError("Trim must be positive")

            metadata_in = read_metadata_from_tif(tif)
            channel_names: list[str] = []

            if max_proj:
                full = tif.asarray()
                if full.ndim >= 4:
                    channel_names = normalize_channel_names(full.shape[1], metadata_in)
                    img = full.max(axis=(0, 1))
                elif full.ndim == 3:
                    channel_names = normalize_channel_names(full.shape[0], metadata_in)
                    img = full.max(axis=(0, 1))
                else:
                    img = full.squeeze()
            elif len(tif.pages) == 1 and tif.pages[0].asarray().ndim == 3:
                arr = tif.pages[0].asarray()
                channel_names = normalize_channel_names(arr.shape[0], metadata_in)
                if idx is None:
                    raise ValueError("Channel index is required when extracting without max projection.")
                img = arr[idx]
                channel_names = [channel_names[idx]] if channel_names else [f"channel_{idx}"]
            else:
                if idx is None:
                    raise ValueError("Channel index is required when extracting without max projection.")
                channel_count = len(tif.pages)
                channel_names_all = normalize_channel_names(channel_count, metadata_in)
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
        img = crop_xy(img, trim)
        if downsample > 1:
            clip_range = clip_range_for_dtype(img.dtype)
            try:
                img = downsample_xy(
                    img,
                    crop=0,
                    factor=downsample,
                    clip_range=clip_range,
                    output_dtype=img.dtype,
                )
            finally:
                gpu_release_all()

        if reduce_bit_depth:
            if img.dtype != np.uint16:
                raise ValueError("Cannot reduce bit depth if image is not uint16")
            img >>= reduce_bit_depth

        level = sc.compression_levels.get("low") if sc else 0.7
        if cancel.is_set():
            raise TaskCancelledException("Cancelled before write")
        metadata_out = compose_metadata(
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
    setup_cli_logging(
        path,
        component="preprocess.stitch.register-simple",
        file="stitch-register-simple",
        extra={"downsample": downsample, "fuse": fuse},
    )
    sc = load_config(json_config).stitching if json_config else None
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
    "--debug/--no-debug",
    default=True,
    show_default=True,
    help="Write debug overlays + metadata alongside fused tiles.",
)
@click.option(
    "--drop-disconnected/--keep-disconnected",
    default=True,
    help="Drop tiles not connected to the main ROI (4-neighborhood).",
)
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
    drop_disconnected: bool = True,
    debug: bool = False,
    json_config: Path | None = None,
):
    setup_cli_logging(
        path,
        component="preprocess.stitch.register",
        file=f"stitch-register-{roi}",
        extra={"roi": roi, "codebook": codebook, "debug": debug},
    )
    ws = Workspace(path)
    if json_config is None:
        json_config = ws.config_json()
    cfg = load_config(json_config) if json_config else None
    sc = cfg.stitching if cfg else None
    pixel_size_um = cfg.pixel_size_um if cfg else 0.108
    out_path = ws.tileconfig_dir(roi)
    tileconfig_registered = ws.tileconfig_registered_txt(roi)
    if tileconfig_registered.exists() and not overwrite:
        logger.info(
            f"TileConfiguration.registered.txt already exists for roi={roi}; "
            "use --overwrite to re-register. Skipping."
        )
        return

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

    out_path.mkdir(exist_ok=True)

    if overwrite:
        for p in out_path.glob("*.tif"):
            p.unlink()

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

    if overwrite or not tileconfig_registered.exists():
        files = sorted(f for f in out_path.glob("*.tif") if not f.name.endswith(".hp.tif"))
        files_idx = [int(file.stem.split("-")[-1]) for file in files if file.stem.split("-")[-1].isdigit()]
        logger.debug(f"Using {files_idx}")
        # Build initial TileConfiguration from the positions CSV subset that
        # corresponds to the extracted files present in the stitch directory.
        tileconfig = TileConfiguration.from_pos(
            pd.read_csv(
                ws.tile_positions_csv(roi, position_file=position_file),
                header=None,
            ).iloc[sorted(files_idx)],
            pixel_size_um=pixel_size_um,
        )

        # Connected components detection and optional filtering.
        # Always log the exact set of disconnected tiles; when requested,
        # drop them from the working TileConfiguration.
        if len(tileconfig) > 0:
            keep_ids = tileconfig.main_component_ids_grid4()
            if keep_ids and len(keep_ids) < len(tileconfig):
                # Convert to python ints for clear logging
                all_ids = set(int(i) for i in tileconfig.df.get_column("index").to_list())
                disconnected_ids = sorted(int(i) for i in (all_ids - keep_ids))
                if disconnected_ids:
                    joined = ", ".join(str(i) for i in disconnected_ids)
                    logger.warning(
                        f"Found {len(disconnected_ids)} disconnected tile(s) not in main component (4-neighborhood): {joined}"
                    )
                    if drop_disconnected:
                        logger.info(
                            f"Dropping {len(disconnected_ids)} disconnected tile(s) from TileConfiguration."
                        )
                        tileconfig = tileconfig.drop(disconnected_ids)

        tileconfig.write(out_path / "TileConfiguration.txt")
        logger.info(f"Created TileConfiguration at {out_path}.")
    run_imagej(
        out_path,
        compute_overlap=True,
        fuse=False,
        threshold=threshold,
        name="TileConfiguration",
        capture_output=not debug,
        stream_to_console=debug,
        sc=sc,
    )

    for p in out_path.glob("*.tif"):
        if p.stem.isdigit():
            p.unlink()

    tc_reg_path = tileconfig_registered
    if not tc_reg_path.exists():
        logger.warning(f"Registered TileConfiguration not found at {tc_reg_path.resolve()}")
        return

    try:
        tc = TileConfiguration.from_file(tc_reg_path)
    except Exception as exc:  # pragma: no cover - user file error
        logger.warning(f"Failed to parse {tc_reg_path.resolve()}: {exc}")
        return

    df = tc.df
    zero_mask = (df["x"] == 0.0) & (df["y"] == 0.0)
    bad = df.filter(zero_mask).sort("index")
    if len(bad) > 1:
        extras = bad.slice(1).select(["index", "filename"]).to_dict(as_series=False)
        indices = extras.get("index", [])
        files = extras.get("filename", [])
        paired = ", ".join(f"{i}:{f}" for i, f in zip(indices, files))
        logger.warning(
            f"Detected {len(indices)} additional tile(s) registered at (0,0): {paired}. "
            "These tiles were likely not registered; inspect inputs or rerun registration."
        )

    try:
        import matplotlib.pyplot as plt

        from fishtools.utils.plot import micron_tick_formatter, place_labels_avoid_overlap

        fig, ax = plt.subplots(figsize=(6, 5), dpi=200)
        tc.plot(ax, show_labels=False)
        xs = df["x"].to_numpy()
        ys = df["y"].to_numpy()
        labels = [str(int(i)) for i in df["index"].to_numpy()]
        place_labels_avoid_overlap(ax, xs, ys, labels, fontsize=6, use_arrows=True)
        fmt = micron_tick_formatter(pixel_size_um)
        ax.xaxis.set_major_formatter(fmt)
        ax.yaxis.set_major_formatter(fmt)
        ax.set_xlabel("X (µm)")
        ax.set_ylabel("Y (µm)")
        ax.set_title(roi)
        fig.tight_layout()

        out_dir = ws.stitch_layout
        out_dir.mkdir(parents=True, exist_ok=True)
        out_png = (out_dir / f"stitch_layout--{roi}.png").resolve()
        fig.savefig(out_png, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved stitch layout plot: {out_png}")
    except Exception as exc:  # pragma: no cover - plotting environment issues
        logger.warning(f"Failed to generate stitch layout plot: {exc}")



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
    workspace_root: Path | None = None,
    roi_for_ws: str | None = None,
    debug: bool = False,
    field_corr: Path | None = None,
    field_zarr: Path | None = None,
    field_patch_downsample: int = 2,
    field_neighbors: int = 0,
    field_smoothing: float | None = None,
    n_channels_reshape: int | None = None,
    n_fids: int = 0,
    include_fiducials: bool = False,
) -> None:
    """Extract tiles into per-(z, channel) folders for ImageJ fusion."""
    try:
        with TiffFile(path) as tif:
            metadata_in = read_metadata_from_tif(tif)
            img = tif.asarray()
    except FileNotFoundError as exc:
        add_file_context(exc, path)
        raise
    except Exception as exc:
        add_file_context(exc, path)
        raise

    # Handle [ZC]YX deconvolved images: remove fiducials and reshape to ZCYX
    fiducials: np.ndarray | None = None
    if n_channels_reshape is not None and img.ndim == 3:
        # Image is [ZC]YX with fiducials appended at end
        if n_fids > 0:
            if include_fiducials:
                fiducials = img[-n_fids:]  # Extract fiducial frames (n_fids, H, W)
            img = img[:-n_fids]  # Remove fiducial frames from main image
        zc, h, w = img.shape
        if zc % n_channels_reshape != 0:
            raise ValueError(
                f"Cannot reshape [ZC]YX image: {zc} frames not divisible by {n_channels_reshape} channels"
            )
        n_z = zc // n_channels_reshape
        # Reshape from [ZC]YX to ZCYX - frames are interleaved as Z0C0, Z0C1, Z0C2, Z1C0, Z1C1, Z1C2, ...
        img = img.reshape(n_z, n_channels_reshape, h, w)

    out_path.mkdir(exist_ok=True)
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
                    raise ValueError("4D input with --is-2d requires --max-proj.")
                img = img.max(axis=0)

            if img.ndim == 2:
                img = img[np.newaxis, ...]
            elif img.ndim != 3:
                raise ValueError("Unsupported image dimensionality for 2D extraction")

            if channels is not None:
                img = img[channels]

            clip_range = clip_range_for_dtype(img.dtype)
            img = crop_xy(img, trim)
            if downsample > 1:
                try:
                    img = downsample_xy(
                        img,
                        crop=0,
                        factor=downsample,
                        clip_range=clip_range,
                        output_dtype=img.dtype,
                    )
                finally:
                    gpu_release_all()

            if field_corr is not None:
                if workspace_root is None:
                    raise ValueError("workspace_root is required when applying --field-corr")
                roi_name = resolve_roi_for_field(out_path, roi_for_ws)
                tile_index = parse_tile_index_from_path(path)
                from fishtools.preprocess.illumination_apply import apply_field_corr_to_tile_zcyx

                corrected = apply_field_corr_to_tile_zcyx(
                    img[np.newaxis, ...],
                    model=field_corr,
                    workspace=workspace_root,
                    roi=roi_name,
                    tile_index=tile_index,
                    downsample=downsample,
                    patch_downsample=field_patch_downsample,
                    channels_to_apply=tuple(range(img.shape[0])),
                    neighbors_override=None if field_neighbors == 0 else field_neighbors,
                    smoothing_override=field_smoothing,
                    pretrim=int(trim),
                    coords_in_downsampled_space=True,
                )
                img = corrected[0]

            if field_zarr is not None:
                if workspace_root is None:
                    raise ValueError("workspace_root is required when applying field Zarr stores")
                roi_name = resolve_roi_for_field(out_path, roi_for_ws)
                tile_index = parse_tile_index_from_path(path)
                x0, y0 = tile_origin(workspace_root, roi_name, tile_index)
                from fishtools.preprocess.illumination import apply_field_tcyx_store_to_img

                img = apply_field_tcyx_store_to_img(
                    img,
                    channel_labels_selected,
                    tcyx_zarr=field_zarr,
                    x0=x0,
                    y0=y0,
                    trim=int(trim),
                )

            if reduce_bit_depth:
                img >>= reduce_bit_depth

            for i in range(img.shape[0]):
                channel_name = (
                    channel_labels_selected[i] if i < len(channel_labels_selected) else f"channel_{i}"
                )
                (out_path / f"{i:02d}").mkdir(exist_ok=True)
                target = out_path / f"{i:02d}" / (path.stem.split("-")[1] + ".tif")
                metadata_out = compose_metadata(
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

        clip_range = clip_range_for_dtype(img.dtype)
        img = crop_xy(img, trim)
        if downsample > 1:
            try:
                img = downsample_xy(
                    img,
                    crop=0,
                    factor=downsample,
                    clip_range=clip_range,
                    output_dtype=img.dtype,
                )
            finally:
                gpu_release_all()

        if field_corr is not None:
            if workspace_root is None:
                raise ValueError("workspace_root is required when applying --field-corr")
            roi_name = resolve_roi_for_field(out_path, roi_for_ws)
            tile_index = parse_tile_index_from_path(path)
            from fishtools.preprocess.illumination_apply import apply_field_corr_to_tile_zcyx

            img = apply_field_corr_to_tile_zcyx(
                img,
                model=field_corr,
                workspace=workspace_root,
                roi=roi_name,
                tile_index=tile_index,
                downsample=downsample,
                patch_downsample=field_patch_downsample,
                channels_to_apply=tuple(range(img.shape[1])),
                neighbors_override=None if field_neighbors == 0 else field_neighbors,
                smoothing_override=field_smoothing,
                pretrim=int(trim),
                coords_in_downsampled_space=True,
            )

        if field_zarr is not None:
            if workspace_root is None:
                raise ValueError("workspace_root is required when applying field Zarr stores")
            roi_name = resolve_roi_for_field(out_path, roi_for_ws)
            tile_index = parse_tile_index_from_path(path)
            x0, y0 = tile_origin(workspace_root, roi_name, tile_index)
            from fishtools.preprocess.illumination import apply_field_tcyx_store_to_img

            img = apply_field_tcyx_store_to_img(
                img,
                channel_labels_selected,
                tcyx_zarr=field_zarr,
                x0=x0,
                y0=y0,
                trim=int(trim),
            ).astype(np.uint16)

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
                metadata_out = compose_metadata(
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

        # Save fiducial frames if extracted
        if fiducials is not None:
            # Apply same processing as main image: trim, downsample
            fid_img = fiducials
            clip_range = clip_range_for_dtype(fid_img.dtype)
            fid_img = crop_xy(fid_img, trim)
            if downsample > 1:
                try:
                    fid_img = gpu_downsample_xy(
                        fid_img,
                        crop=0,
                        factor=downsample,
                        clip_range=clip_range,
                        output_dtype=fid_img.dtype,
                    )
                finally:
                    gpu_release_all()

            # Save fiducials as 1 channel with n_fids Z slices: fid/00/, fid/01/
            for fid_z in range(fid_img.shape[0]):
                fid_folder = out_path / "fid" / f"{fid_z:02d}"
                fid_folder.mkdir(exist_ok=True, parents=True)
                target = fid_folder / (path.stem.split("-")[1] + ".tif")
                metadata_out = compose_metadata(
                    "YX",
                    ["fiducial"],
                    extra={"processing": {"fiducial_z": fid_z}},
                )
                try:
                    safe_imwrite(
                        target,
                        fid_img[fid_z],
                        compression=22610,
                        metadata=metadata_out,
                        compressionargs={"level": 0.75},
                    )
                except Exception as exc:
                    add_file_context(exc, path, target)
                    raise
            del fid_img

    except Exception as exc:
        add_file_context(exc, path)
        raise


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
@click.option(
    "--debug/--no-debug",
    default=True,
    show_default=True,
    help="Write debug overlays + metadata alongside fused tiles.",
)
@click.option("--max-from", type=str)
@click.option(
    "--field-zarr",
    type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=Path),
    default=None,
    help=(
        "Pre-exported field Zarr store (TCYX, ds=1) from correct-illum export-field --what both --downsample 1. "
        "Selects T planes ['low','range'] during application."
    ),
)
@click.option(
    "--json-config",
    type=click.Path(exists=True, dir_okay=False, file_okay=True, path_type=Path),
    default=None,
    help="Optional project config to populate stitching defaults.",
)
@click.option(
    "--coarse-shifts",
    type=click.Path(exists=True, dir_okay=False, file_okay=True, path_type=Path),
    default=None,
    help="Path to coarse_shifts.json from fix-shifts. Adjusts tile positions for large drifts.",
)
@click.option(
    "--round-name",
    type=str,
    default=None,
    help="Round name to use from coarse_shifts.json (required if multiple rounds in file).",
)
@click.option(
    "--fuse-only",
    is_flag=True,
    help="Only run fusion without combining into Zarr array.",
)
# @click.option("--skip-extract", is_flag=True)
@batch_roi("registered--*", include_codebook=True, split_codebook=True)
def fuse(
    path: Path,
    roi: str,
    codebook: str | None = None,
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
    debug: bool = True,
    max_from: str | None = None,
    json_config: Path | None = None,
    # skip_extract: bool = False,
    field_zarr: Path | None = None,
    coarse_shifts: Path | None = None,
    round_name: str | None = None,
    fuse_only: bool = False,
):
    if codebook is None and round_name is None:
        raise ValueError("Either --codebook or --round-name must be provided.")

    ws = Workspace(path)
    if json_config is None:
        json_config = ws.config_json()

    shift_lookup: dict[int, tuple[float, float]] = {}
    coarse_round_name: str | None = None

    if round_name is not None:
        ctx = click.get_current_context(silent=True)
        if ctx is not None:
            source = ctx.get_parameter_source("downsample")
            if source is None or source is ParameterSource.DEFAULT:
                downsample = 1

        if coarse_shifts is None:
            coarse_shifts = ws.coarse_shifts_json(roi)
            if not coarse_shifts.exists():
                raise ValueError(
                    f"Coarse shifts file not found at {coarse_shifts}. "
                    f"Run 'preprocess register fix-shifts' first or provide --coarse-shifts."
                )
            logger.info(f"Auto-detected coarse shifts: {coarse_shifts}")
        tiles_shifts = json.loads(coarse_shifts.read_text()).get("tiles", {})

        available_rounds: set[str] = set()
        for tile_shifts in tiles_shifts.values():
            available_rounds.update(tile_shifts.keys())

        if round_name not in available_rounds:
            available = ", ".join(sorted(available_rounds))
            raise ValueError(f"Round '{round_name}' not in coarse_shifts. Available: {available}")
        coarse_round_name = round_name

        for tile_idx_str, round_shifts in tiles_shifts.items():
            if coarse_round_name in round_shifts:
                shift_data = round_shifts[coarse_round_name]
                shift_lookup[int(tile_idx_str)] = (shift_data["dx"], shift_data["dy"])

        stitch_dir = ws.stitch_shifted(roi, coarse_round_name)
        path_img = ws.deconv_round_dir(coarse_round_name, roi)

        n_channels_reshape = len(coarse_round_name.split("_"))
        n_fids_reshape = 2
    else:
        assert codebook is not None  # validated above
        stitch_dir = ws.stitch(roi, codebook)
        path_img = ws.registered(roi, codebook)
        n_channels_reshape = None
        n_fids_reshape = 0

    # Set up logging after determining paths
    log_label = codebook if codebook else coarse_round_name
    setup_cli_logging(
        path,
        component="preprocess.stitch.fuse",
        file=f"stitch-fuse-{roi}+{log_label}",
        extra={"roi": roi, "codebook": codebook, "round": coarse_round_name, "threads": threads},
    )

    sc = load_config(json_config).stitching if json_config else None

    existing_fused = [
        candidate
        for candidate in (stitch_dir / "fused.zarr", stitch_dir / "fused_n4.zarr")
        if candidate.exists()
    ]
    if existing_fused and not overwrite:
        existing_names = ", ".join(sorted(p.name for p in existing_fused))
        logger.info(
            f"Skipping ROI '{roi}' — existing fused outputs ({existing_names}) present. "
            "Re-run with --overwrite to regenerate."
        )
        return
    stitch_dir.mkdir(parents=True, exist_ok=True)
    import json as json_module
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

    # File pattern differs: registered uses "reg*.tif", deconvolved rounds use "{round}-*.tif"
    if coarse_round_name:
        correct_count = len(list(path_img.glob(f"{coarse_round_name}-*.tif")))
    else:
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
        tile_config = ws.tileconfig_registered_txt(roi)
        logger.info(f"Getting tile configuration from {tile_config.resolve()}")

    fuse_args = {
        "roi": roi,
        "codebook": codebook,
        "round_name": coarse_round_name,
        "tile_config": str(tile_config) if tile_config else None,
        "split": split,
        "overwrite": overwrite,
        "downsample": downsample,
        "is_2d": is_2d,
        "threads": threads,
        "channels": channels,
        "subsample_z": subsample_z,
        "max_proj": max_proj,
        "debug": debug,
        "max_from": max_from,
        "json_config": str(json_config) if json_config else None,
        "field_zarr": str(field_zarr) if field_zarr else None,
        "coarse_shifts": str(coarse_shifts) if coarse_shifts else None,
        "fuse_only": fuse_only,
    }
    if sc is not None:
        fuse_args["stitching_config"] = sc.model_dump()
    (stitch_dir / "fuse_args.json").write_text(json_module.dumps(fuse_args, indent=2))

    tileconfig = TileConfiguration.from_file(tile_config).downsample(downsample)

    # Apply per-tile coarse shifts
    if shift_lookup:
        logger.info(f"Applying coarse shifts for round '{coarse_round_name}' to {len(shift_lookup)} tiles")

        tc_indices = set(int(r["index"]) for r in tileconfig.df.iter_rows(named=True))
        missing = tc_indices - set(shift_lookup.keys())
        if missing:
            logger.warning(
                f"Tiles in TileConfiguration but not in coarse_shifts (using 0,0): {sorted(missing)}"
            )

        adjusted_rows: list[dict[str, Any]] = []
        for row in tileconfig.df.iter_rows(named=True):
            tile_idx = int(row["index"])
            dx, dy = shift_lookup.get(tile_idx, (0.0, 0.0))
            row["x"] = row["x"] + dx / downsample
            row["y"] = row["y"] + dy / downsample
            adjusted_rows.append(row)
        tileconfig = TileConfiguration(pl.DataFrame(adjusted_rows, schema=tileconfig.df.schema))

        applied = [(dx, dy) for dx, dy in shift_lookup.values()]
        if applied:
            dxs, dys = zip(*applied)
            logger.info(f"  dx: [{min(dxs):.1f}, {max(dxs):.1f}], dy: [{min(dys):.1f}, {max(dys):.1f}]")

        stitch_dir.mkdir(parents=True, exist_ok=True)
        shifted_tc_path = stitch_dir / "TileConfiguration.shifted.txt"
        tileconfig.write(shifted_tc_path)
        logger.info(f"Saved shifted TileConfiguration to {shifted_tc_path}")

    n = len(tileconfig) // split

    # Read metadata first to properly determine channel count
    try:
        with TiffFile(files[0]) as tif_first:
            metadata_first = read_metadata_from_tif(tif_first)
            first_image_shape = tif_first.asarray().shape
            first_image_ndim = len(first_image_shape)
    except Exception as exc:
        add_file_context(exc, files[0])
        raise

    # Get metadata key for channel names (used later regardless of source)
    key_raw = metadata_first.get("key")

    # Determine channel count - use n_channels_reshape if set (deconvolved images)
    if n_channels_reshape is not None:
        n_channels = n_channels_reshape
        logger.info(f"Using {n_channels} channels from round name")
    elif isinstance(key_raw, str):
        n_channels = 1
    elif isinstance(key_raw, (list, tuple)):
        n_channels = len(key_raw)
    elif isinstance(key_raw, np.ndarray):
        n_channels = len(key_raw)
    else:
        # Infer from shape: 4D ZCYX -> shape[1], 3D CYX/ZYX -> need is_2d context
        if first_image_ndim >= 4:
            n_channels = first_image_shape[1]
        elif first_image_ndim == 3 and is_2d:
            n_channels = first_image_shape[0]  # CYX format
        elif first_image_ndim == 3:
            n_channels = 1  # ZYX format (single channel 3D)
        else:
            n_channels = 1  # 2D image

    if channels == "all":
        channels = ",".join(map(str, range(n_channels)))
        logger.info(f"Auto-detected {n_channels} channels")

    channel_indices = [int(c) for c in channels.split(",") if c]

    # Build channel names from metadata
    if isinstance(key_raw, str):
        channel_names_all = normalize_channel_names(1, metadata_first)
    elif isinstance(key_raw, (list, tuple)):
        channel_names_all = normalize_channel_names(len(key_raw), metadata_first)
    elif isinstance(key_raw, np.ndarray):
        channel_names_all = normalize_channel_names(len(key_raw), metadata_first)
    else:
        count_guess = (max(channel_indices) + 1) if channel_indices else n_channels
        channel_names_all = normalize_channel_names(count_guess, metadata_first)

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
    if field_zarr and int(downsample) != 1:
        raise ValueError("When providing field Zarr stores, please set --downsample 1 for alignment.")

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
                    field_zarr=field_zarr,
                    n_channels_reshape=n_channels_reshape,
                    n_fids=n_fids_reshape,
                    include_fiducials=n_channels_reshape is not None,  # Always include in round-name mode
                )

    def run_folder(folder: Path, capture_output: bool = False, stream_to_console: bool = False) -> None:
        def log_progress(message: str) -> None:
            if capture_output:
                logger.info(message)
            else:
                get_shared_console().log(message)
                logger.bind(**{CONSOLE_SKIP_EXTRA: True}).info(message)

        for i in range(split):
            tileconfig[i * n : (i + 1) * n].write(folder / f"TileConfiguration{i + 1}.registered.txt")
            start = perf_counter()
            log_progress(f"Starting ImageJ fuse for {folder} ({i + 1}/{split})")
            run_imagej(
                folder,
                name=f"TileConfiguration{i + 1}",
                capture_output=capture_output,
                stream_to_console=stream_to_console,
                sc=sc,
            )
            (folder / "img_t1_z1_c1").rename(folder / f"fused_{folder.name}-{i + 1}.tif")
            duration = perf_counter() - start
            log_progress(f"Completed ImageJ fuse for {folder} ({i + 1}/{split}) in {duration:.2f}s")

    # Get all folders without subfolders
    folders = list(chain.from_iterable(walk_fused(path).values()))

    # Also include fiducial folders if they exist (fid/00/, fid/01/)
    fid_base = path / "fid"
    if fid_base.exists():
        fid_subfolders = sorted([f for f in fid_base.iterdir() if f.is_dir() and f.name.isdigit()])
        if fid_subfolders:
            logger.info(f"Found {len(fid_subfolders)} fiducial Z-planes to fuse")
            folders.extend(fid_subfolders)

    logger.info(f"Calling ImageJ on {len(folders)} folders.")
    to_runs: list[Path] = []
    for folder in folders:
        if not folder.is_dir():
            raise ValueError(f"Invalid folder: {folder}")
        if not folder.name.isdigit():
            raise ValueError(f"Invalid folder name {folder.name}. Expected digit.")

        existings = list(folder.glob("fused*"))
        if existings and not overwrite:
            logger.warning(f"Skipping {folder}: fused outputs already exist")
            continue
        to_runs.append(folder)

    if to_runs:
        with progress_bar_threadpool(len(to_runs), threads=threads, stop_on_exception=True) as submit:
            for folder in to_runs:
                submit(run_folder, folder, capture_output=not debug, stream_to_console=debug)

        if split > 1:
            for folder in folders:
                channel_position = int(folder.name)
                channel_name = channel_labels_by_position.get(channel_position)
                final_stitch(folder, split, channel_name=channel_name, sc=sc)

    if not fuse_only:
        logger.info("Automatically running combine step...")
        cmd = ["preprocess", "stitch", "combine", str(ws.path), roi]
        if codebook:
            cmd.extend(["--codebook", codebook])
        if round_name:
            cmd.extend(["--round-name", round_name])
        if overwrite:
            cmd.append("--overwrite")
        subprocess.run(cmd, check=True)


def numpy_array_to_zarr(write_path: Path | str, array: np.ndarray, chunks: tuple[int, ...]):  # shim
    return _numpy_array_to_zarr(write_path, array, chunks)


@stitch.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=Path))
@click.argument("roi", type=str, default="*")
@click.option("--codebook", type=str, default=None)
@click.option("--round-name", type=str, default=None, help="Round name for shifted fusion folder (e.g., 1_9_17)")
@click.option("--chunk-size", type=int, default=2048)
@click.option("--overwrite", is_flag=True)
@click.option(
    "--options",
    "thumbnail_options",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    help="JSON file to override thumbnail generation options.",
)
@batch_roi("stitch--*", include_codebook=True, split_codebook=True)
def combine(
    path: Path,
    roi: str,
    codebook: str | None = None,
    round_name: str | None = None,
    chunk_size: int = 2048,
    overwrite: bool = False,
    thumbnail_options: Path | None = None,
):
    if codebook is None and round_name is None:
        raise ValueError("Either --codebook or --round-name must be provided.")

    log_label = codebook if codebook else round_name
    setup_cli_logging(
        path,
        component="preprocess.stitch.combine",
        file=f"stitch-combine-{roi}+{log_label}",
        extra={"roi": roi, "codebook": codebook, "round_name": round_name, "chunk_size": chunk_size},
    )
    import json as json_module

    ws = Workspace(path)
    target_rois = ws.resolve_rois(None if roi == "*" else [roi])

    try:
        thumb_options = load_thumbnail_options(thumbnail_options)
    except Exception as exc:
        raise click.ClickException(f"Invalid --options file: {exc}") from exc

    for current_roi in target_rois:
        if round_name is not None:
            stitched_dir = ws.stitch_shifted(current_roi, round_name)
        else:
            assert codebook is not None
            stitched_dir = ws.stitch(current_roi, codebook)
        try:
            folders_by_z = walk_fused(stitched_dir)
        except ValueError:
            logger.warning(
                f"No valid stitched folders found under {stitched_dir}. Skipping ROI '{current_roi}'."
            )
            continue
        for z_idx in folders_by_z:
            folders_by_z[z_idx].sort(key=lambda f: int(f.name))

        if not folders_by_z:
            logger.warning(f"No stitched content discovered for ROI '{current_roi}'. Skipping.")
            continue

        zs = max(folders_by_z.keys()) + 1
        first_z = min(folders_by_z.keys())
        cs = max(int(f.name) for f in folders_by_z[first_z]) + 1  # Assume C count is same for all Z

        first_z_folders = folders_by_z[first_z]
        missing = [
            folder for folder in first_z_folders if not (folder / f"fused_{folder.name}-1.tif").exists()
        ]
        if missing:
            logger.warning(
                f"Missing fused images for ROI '{current_roi}' in Z=0: {[m.name for m in missing]}. Skipping ROI."
            )
            continue

        first_folder = first_z_folders[0]
        first_img = imread(first_folder / f"fused_{first_folder.name}-1.tif")
        img_shape = first_img.shape
        dtype = first_img.dtype
        final_shape = (zs, img_shape[0], img_shape[1], cs)
        logger.info(f"Final Zarr shape: {final_shape}, dtype: {dtype}")

        zarr_path = stitched_dir / "fused.zarr"
        logger.info(f"Writing to {zarr_path.resolve()}")
        zarr_chunks = (1, chunk_size, chunk_size, 1)
        z_array = create_zarr_array(
            zarr_path,
            shape=final_shape,
            chunks=zarr_chunks,
            dtype=dtype,
            overwrite=overwrite,
        )
        fuse_args_path = stitched_dir / "fuse_args.json"
        if fuse_args_path.exists():
            fuse_args = json_module.loads(fuse_args_path.read_text())
            z_array.attrs["fuse_args"] = fuse_args

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
                    img = imread(img_path)

                    z_plane_data[:, :, j] = img[:, :]

                    if thumbnail_data is None:
                        preview_c = min(3, cs)
                        thumbnail_data = np.zeros((img.shape[0], img.shape[1], preview_c), dtype=np.uint16)
                    if j < thumbnail_data.shape[2]:
                        thumbnail_data[:, :, j] = img[:, :]

                    del img

                logger.info(f"Writing Z-plane {i + 1}/{zs} to Zarr array")
                z_array[i, :, :, :] = z_plane_data

                if (i % thumb_options.z_stride) == 0 and thumbnail_data is not None:
                    thumbnail_path = thumbnail_dir / f"thumbnail_z{i:03d}.png"
                    save_thumbnail_png(thumbnail_data, thumbnail_path, options=thumb_options)
                    logger.debug(f"Saved thumbnail for Z-plane {i} to {thumbnail_path}")

                progress()

        if codebook is not None:
            try:
                first_reg_file = next(ws.registered(current_roi, codebook).glob("*.tif"))
            except StopIteration:
                logger.warning(
                    f"No registered TIF file found in {ws.registered(current_roi, codebook)} to read channel names."
                )
            else:
                try:
                    with TiffFile(first_reg_file) as tif:
                        names = tif.shaped_metadata[0].get("key") if tif.shaped_metadata else None
                except Exception as exc:
                    logger.warning(f"Error reading metadata from {first_reg_file}: {exc}")
                else:
                    if names:
                        if isinstance(names, np.ndarray):
                            names = names.tolist()
                        elif isinstance(names, tuple):
                            names = list(names)
                        elif not isinstance(names, list):
                            names = [str(names)]

                        if cs == len(names) + 1:
                            names.append("spots")
                        z_array.attrs["key"] = names
                        logger.info(f"Added channel names: {names}")
                    else:
                        logger.warning("Could not find channel names ('key') in TIF metadata.")

        logger.info("Deleting source folders.")
        parents = sorted({folder.parent for z_folders in folders_by_z.values() for folder in z_folders})
        for parent in parents:
            if parent.exists():
                shutil.rmtree(parent)
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
@click.option(
    "--threads",
    type=int,
    default=None,
    help="Override number of CPU threads used for SimpleITK field estimation (defaults to ~half cores).",
)
@click.option(
    "--tile-size",
    type=int,
    default=None,
    help="Optional explicit GPU tile size (pixels). Defaults to auto when unset.",
)
@click.option(
    "--tile-threshold",
    type=int,
    default=None,
    help="Auto-tiling threshold in pixels (max dimension). Set <=0 to disable auto-tiling.",
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
    threads: int | None,
    tile_size: int | None,
    tile_threshold: int | None,
) -> None:
    """Run N4 bias-field correction against stitched mosaics."""

    setup_cli_logging(
        path,
        component="preprocess.stitch.n4",
        file=f"stitch-n4-{roi}+{codebook}",
        debug=debug,
        extra={"roi": roi, "codebook": codebook},
    )

    try:
        _runner = run_cli_workflow or _get_run_cli_workflow()
        results = _runner(
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
            threads=threads,
            tile_size=tile_size,
            tile_threshold=tile_threshold,
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
@click.option("--codebook", type=str, required=True, help="Codebook used in the stitch directory name.")
@click.option(
    "--highpass-px",
    type=float,
    default=20.0,
    show_default=True,
    help="Gaussian high-pass sigma in XY pixels.",
)
@click.option(
    "--anisotropy",
    type=float,
    default=2.0,
    show_default=True,
    help="Z/XY voxel-size ratio used to set sigma_z = highpass_px / anisotropy.",
)
@click.option(
    "--perc-lo",
    type=float,
    default=1.0,
    show_default=True,
    help="Lower percentile for highpass quantization (0-100).",
)
@click.option(
    "--perc-hi",
    type=float,
    default=99.999,
    show_default=True,
    help="Upper percentile for highpass quantization (0-100).",
)
@click.option("--overwrite", is_flag=True, help="Overwrite existing outputs.")
@batch_roi("stitch--*", include_codebook=True, split_codebook=True)
def highpass(
    path: Path,
    roi: str,
    codebook: str,
    *,
    highpass_px: float,
    anisotropy: float,
    perc_lo: float,
    perc_hi: float,
    overwrite: bool,
) -> None:
    """Write a high-pass filtered fused zarr as `fused_highpassed.zarr` (uint16)."""

    setup_cli_logging(
        path,
        component="preprocess.stitch.highpass",
        file=f"stitch-highpass-{roi}+{codebook}",
        extra={
            "roi": roi,
            "codebook": codebook,
            "highpass_px": highpass_px,
            "anisotropy": anisotropy,
            "perc_lo": perc_lo,
            "perc_hi": perc_hi,
        },
    )

    if anisotropy <= 0:
        raise click.ClickException("--anisotropy must be > 0.")
    if highpass_px <= 0:
        raise click.ClickException("--highpass-px must be > 0.")
    if not (0.0 <= perc_lo < perc_hi <= 100.0):
        raise click.ClickException("--perc-lo/--perc-hi must satisfy 0 <= perc_lo < perc_hi <= 100.")

    from fishtools.io.workspace import Workspace
    from fishtools.preprocess.highpass import run_highpass_workflow

    ws = Workspace(path)
    stitch_root = ws.stitch(roi, codebook)

    try:
        out = run_highpass_workflow(
            stitch_root=stitch_root,
            sigma_px=highpass_px,
            anisotropy=anisotropy,
            percentile_lo=perc_lo,
            percentile_hi=perc_hi,
            overwrite=overwrite,
        )
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(f"Highpassed imagery saved to {out}")


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
    metadata_out = compose_metadata(
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


def extract_patch(mosaic: np.ndarray, x0: int, y0: int, size: int) -> np.ndarray:
    """Extract size×size patch from 2D mosaic with zero-padding for boundaries."""
    h, w = mosaic.shape[-2:]
    y_start, x_start = max(0, y0), max(0, x0)
    y_end, x_end = min(h, y0 + size), min(w, x0 + size)

    patch = np.zeros((size, size), dtype=mosaic.dtype)
    dy, dx = max(0, -y0), max(0, -x0)
    patch[dy : dy + (y_end - y_start), dx : dx + (x_end - x_start)] = mosaic[y_start:y_end, x_start:x_end]
    return patch


def load_fiducial_mosaics(fid_dir: Path) -> dict[int, np.ndarray]:
    """Load fiducial mosaics from ImageJ fused outputs (fused_<z>-1.tif)."""

    fid_mosaics: dict[int, np.ndarray] = {}
    if not fid_dir.exists():
        return fid_mosaics

    for fid_z_folder in sorted(fid_dir.iterdir()):
        if not (fid_z_folder.is_dir() and fid_z_folder.name.isdigit()):
            continue

        fid_index = int(fid_z_folder.name)
        fused_path = fid_z_folder / f"fused_{fid_z_folder.name}-1.tif"
        if not fused_path.exists():
            logger.warning(f"No fiducial mosaic found under {fid_z_folder}")
            continue

        fid_mosaics[fid_index] = imread(fused_path)
        logger.info(f"Loaded fiducial Z={fid_z_folder.name} from {fused_path.name}")

    return fid_mosaics


def slice_tile_from_zarr(
    zarr_array,
    fid_mosaics: dict[int, np.ndarray],
    slice_x: int,
    slice_y: int,
    tile_size: int,
) -> np.ndarray:
    """Extract tile from zarr and reconstruct [ZC]YX + fiducials format.

    Args:
        zarr_array: Zarr array with shape (Z, Y, X, C)
        fid_mosaics: Dict mapping fiducial Z index to 2D mosaic array
        slice_x: X coordinate in mosaic (original_x - mosaic_origin_x)
        slice_y: Y coordinate in mosaic (original_y - mosaic_origin_y)
        tile_size: Output tile size in pixels

    Returns:
        Array of shape (Z*C + n_fids, tile_size, tile_size)
    """
    z_dim, mosaic_h, mosaic_w, c_dim = zarr_array.shape

    # Boundary-safe slicing
    y_start, y_end = max(0, slice_y), min(mosaic_h, slice_y + tile_size)
    x_start, x_end = max(0, slice_x), min(mosaic_w, slice_x + tile_size)

    # Read from zarr (lazy, efficient)
    patch_zyxc = zarr_array[:, y_start:y_end, x_start:x_end, :]  # (Z, h, w, C)

    # Zero-pad if at boundary
    if patch_zyxc.shape[1:3] != (tile_size, tile_size):
        padded = np.zeros((z_dim, tile_size, tile_size, c_dim), dtype=patch_zyxc.dtype)
        dy, dx = max(0, -slice_y), max(0, -slice_x)
        padded[:, dy : dy + patch_zyxc.shape[1], dx : dx + patch_zyxc.shape[2], :] = patch_zyxc
        patch_zyxc = padded

    # ZYXC → ZCYX → [ZC]YX
    patch_zcyx = patch_zyxc.transpose(0, 3, 1, 2)  # (Z, C, Y, X)
    z, c, h, w = patch_zcyx.shape
    patch_flat = patch_zcyx.reshape(z * c, h, w)

    # Append fiducials: fid/01 = original[-1], fid/00 = original[-2]
    # So ascending order [0, 1] gives [original[-2], original[-1]] which is correct
    fid_frames = []
    for fid_z in sorted(fid_mosaics.keys()):
        fid_patch = extract_patch(fid_mosaics[fid_z], slice_x, slice_y, tile_size)
        fid_frames.append(fid_patch)

    if fid_frames:
        return np.concatenate([patch_flat, np.stack(fid_frames)], axis=0)
    return patch_flat


@stitch.command(name="slice")
@click.argument("path", type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=Path))
@click.argument("roi", type=str, default="*")
@click.option("--round-name", type=str, required=True, help="Round name (e.g., 1_9_17)")
@click.option("--tile-size", type=int, default=2048, help="Tile size in pixels")
@click.option("--overwrite", is_flag=True)
@batch_roi("stitch--*", include_codebook=False, split_codebook=True)
def slice_mosaic(
    path: Path,
    roi: str,
    round_name: str,
    tile_size: int = 2048,
    overwrite: bool = False,
):
    """Slice shifted mosaic back into tiles in [ZC]YX format with fiducials.

    This command extracts tiles from a fused.zarr mosaic that was created with
    coarse shift correction, producing tiles that can feed back into the standard
    pipeline at their original positions.

    The coordinate logic:
    - ImageJ sets mosaic (0,0) at min(shifted_x), min(shifted_y)
    - We slice at original_position - mosaic_origin to get reference-aligned content
    """
    import zarr

    setup_cli_logging(
        path,
        component="preprocess.stitch.slice",
        file=f"stitch-slice-{roi}+{round_name}",
        extra={"roi": roi, "round_name": round_name, "tile_size": tile_size},
    )

    ws = Workspace(path)
    stitch_dir = ws.stitch_shifted(roi, round_name)
    out_dir = ws.deconv_repaired_dir(round_name, roi)

    zarr_path = stitch_dir / "fused.zarr"
    if not zarr_path.exists():
        raise click.ClickException(f"fused.zarr not found at {zarr_path}. Run 'stitch combine' first.")

    zarr_array = zarr.open(zarr_path, mode="r")
    logger.info(f"Opened zarr with shape {zarr_array.shape}")

    # Load ORIGINAL TileConfiguration (for output tile positions)
    original_tc_path = ws.tileconfig_registered_txt(roi)
    tileconfig = TileConfiguration.from_file(original_tc_path)
    logger.info(f"Loaded original TileConfiguration from {original_tc_path}")

    # Load SHIFTED TileConfiguration (to compute mosaic origin)
    shifted_tc_path = stitch_dir / "TileConfiguration.shifted.txt"
    if not shifted_tc_path.exists():
        raise click.ClickException(
            f"Shifted TileConfiguration not found at {shifted_tc_path}. "
            "Run 'stitch fuse --round-name' first (requires recent version that saves shifted config)."
        )
    shifted_tc = TileConfiguration.from_file(shifted_tc_path)
    logger.info(f"Loaded shifted TileConfiguration from {shifted_tc_path}")

    # Mosaic origin = min(shifted positions) - this is what ImageJ used as (0,0)
    origin_x = shifted_tc.df["x"].min()
    origin_y = shifted_tc.df["y"].min()
    logger.info(f"Mosaic origin (min of shifted positions): ({origin_x:.1f}, {origin_y:.1f})")

    # Load fiducial mosaics (supports fused_00-1 and fused_{z}-1 naming)
    fid_mosaics = load_fiducial_mosaics(stitch_dir / "fid")

    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {out_dir}")

    # Load metadata from original deconvolved files
    source_dir = ws.deconved / f"{round_name}--{roi}"
    if not source_dir.exists():
        logger.warning(f"Source directory {source_dir} not found - output files will have no metadata")

    skipped = 0
    rows = list(tileconfig.df.iter_rows(named=True))
    with progress_bar(len(rows)) as update:
        for row in rows:
            tile_idx = int(row["index"])
            # Slice at ORIGINAL position relative to SHIFTED origin
            slice_x = int(round(row["x"] - origin_x))
            slice_y = int(round(row["y"] - origin_y))

            out_path = out_dir / f"{round_name}-{tile_idx:04d}.tif"
            if out_path.exists() and not overwrite:
                skipped += 1
                update()
                continue

            # Read metadata from original file
            metadata = None
            source_file = source_dir / f"{round_name}-{tile_idx:04d}.tif"
            if source_file.exists():
                with TiffFile(source_file) as tif:
                    shaped = getattr(tif, "shaped_metadata", None)
                    if shaped:
                        metadata = dict(shaped[0])

            tile_data = slice_tile_from_zarr(zarr_array, fid_mosaics, slice_x, slice_y, tile_size)
            safe_imwrite(out_path, tile_data, metadata=metadata, **IMWRITE_KWARGS)
            update()

    if skipped:
        logger.info(f"Skipped {skipped} existing tiles (use --overwrite to regenerate)")
    logger.info(f"Sliced {len(tileconfig)} tiles to {out_dir}")


if __name__ == "__main__":
    stitch()
