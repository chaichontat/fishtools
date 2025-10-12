from __future__ import annotations

import math
import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Annotated

import numpy as np
import tifffile
import typer
import zarr
from loguru import logger
from numpy.random import Generator
from scipy.ndimage import zoom
from tifffile import imread, imwrite

from fishtools.io.workspace import Workspace
from fishtools.preprocess.segmentation import unsharp_all
from fishtools.utils.pretty_print import progress_bar, progress_bar_threadpool

"""Extract utilities and Typer command callback.

This module intentionally does not define its own Typer app or CLI entrypoint.
The top-level `segment` CLI registers `cmd_extract` as a command.
"""


# ---------- Helpers ----------

TIFF_KWARGS = {
    "compression": 22610,
    "photometric": "minisblack",
    "planarconfig": "separate",
    "compressionargs": {"level": 0.8},
}


@dataclass
class ChannelSelection:
    indices: list[int]
    names: list[str] | None = None


def _read_channel_names(file: Path) -> list[str] | None:
    """Extract channel names from TIFF metadata, returning None if unavailable."""
    try:
        with tifffile.TiffFile(file) as tif:
            if getattr(tif, "shaped_metadata", None):
                md = tif.shaped_metadata[0]
                if isinstance(md, dict):
                    names = md.get("key")
                    if isinstance(names, (list, tuple)):
                        return [str(x) for x in names]
    except Exception as e:  # defensive; keep quiet in CLI
        logger.debug(f"Failed to read channel names from {file}: {e}")
    return None


def _infer_channels_from_names(names: list[str]) -> list[int]:
    """Apply heuristics to select common channel pairs from available names."""
    name_to_idx = {n: i for i, n in enumerate(names)}
    # Heuristics ported from original script
    if "polyA" in name_to_idx:
        wanted = ["polyA", "reddot"]
    elif "wga" in name_to_idx and "reddot" in name_to_idx:
        wanted = ["wga", "reddot"]
    elif "af" in name_to_idx and "reddot" in name_to_idx:
        wanted = ["af", "reddot"]
    elif "reddot" in name_to_idx and "atp" in name_to_idx:
        wanted = ["reddot", "atp"]
    elif "pi" in name_to_idx and "atp" in name_to_idx:
        wanted = ["pi", "atp"]
    else:
        return [0, 1]
    idx = [name_to_idx.get(w) for w in wanted]
    if any(i is None for i in idx):
        return [0, 1]
    return [int(i) for i in idx]


def _normalize_channel_names(names: object) -> list[str] | None:
    if isinstance(names, (list, tuple)):
        return [str(n) for n in names]
    return None


def _read_channel_names_from_zarr(store: Path) -> list[str] | None:
    try:
        arr = zarr.open_array(store, mode="r")
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug(f"Failed to open Zarr store {store} for channel names: {exc}")
        return None
    return _normalize_channel_names(arr.attrs.get("key"))


def _is_zarr_path(file: Path) -> bool:
    return file.suffix == ".zarr" or (file.is_dir() and file.name.endswith(".zarr"))


def _open_zarr_array(file: Path):
    return zarr.open_array(file, mode="r")


def _open_volume(file: Path) -> tuple[object, list[str] | None]:
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


def _resolve_output_names(chsel: ChannelSelection, channels: str | None) -> list[str]:
    """Compute output channel names after selection; prefer explicit --channels names."""
    arg_names = [p.strip() for p in channels.split(",") if p.strip()] if channels else None
    if arg_names:
        return arg_names
    if chsel.names:
        try:
            return [chsel.names[i] for i in chsel.indices]
        except Exception:
            return [str(i) for i in chsel.indices]
    return [str(i) for i in chsel.indices]


def _write_tiff(
    path: Path,
    data: np.ndarray,
    axes: str,
    names: list[str] | None,
    channels_arg: str | None,
    *,
    upscale: float,
) -> None:
    imwrite(
        path,
        data,
        metadata={
            "axes": axes,
            "channel_names": names,
            "channels_arg": channels_arg,
            "upscale": upscale,
        },
        **TIFF_KWARGS,  # type: ignore
    )


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


def _parse_channels(ch_arg: str | None, names: list[str] | None) -> ChannelSelection:
    """Parse channel specification: 'auto' uses heuristics, or numeric indices, or named channels."""
    if ch_arg is None or ch_arg.lower() == "auto":
        if names:
            return ChannelSelection(_infer_channels_from_names(names), names)
        return ChannelSelection([0, 1], None)

    parts = [p.strip() for p in ch_arg.split(",") if p.strip()]
    if not parts:
        raise typer.BadParameter("Empty --channels specification.")

    # numeric
    try:
        return ChannelSelection([int(p) for p in parts], None)
    except ValueError:
        if not names:
            raise typer.BadParameter("Channel names not in metadata; pass numeric indices.")
        name_to_idx = {n: i for i, n in enumerate(names)}
        try:
            return ChannelSelection([name_to_idx[p] for p in parts], names)
        except KeyError as e:
            raise typer.BadParameter(f"Unknown channel name: {e.args[0]}") from e


def _load_registered_stack(file: Path) -> tuple[np.ndarray, list[str] | None, str]:
    """Load a registered stack from TIFF or Zarr and normalise to (Z,C,Y,X)."""
    if file.suffix.lower() in {".tif", ".tiff"}:
        img = imread(file)
        if img.ndim != 4:
            raise ValueError("Expected registered TIFF to be 4D (Z,C,Y,X).")
        names = _read_channel_names(file)
        return img, names, file.name

    if _is_zarr_path(file):
        arr = _open_zarr_array(file)
        img = np.asarray(arr)
        if img.ndim != 4:
            raise ValueError("Expected fused Zarr to be 4D (Z,Y,X,C).")
        img = np.moveaxis(img, -1, 1)  # -> (Z,C,Y,X)
        names = _read_channel_names_from_zarr(file)
        return img, names, file.name

    raise ValueError(f"Unsupported registered stack format: {file}")


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

    maxed = other.max(axis=1, keepdims=True)  # (Z,1,Y,X)
    new_img = np.concatenate([img, maxed], axis=1)
    return new_img, new_img.shape[1] - 1


def _iter_registered_files(reg_dir: Path) -> Iterable[Path]:
    """Get registered TIFF files sorted by size (largest first) from reg-*.tif pattern."""
    files = sorted(reg_dir.glob("reg-*.tif"), key=lambda p: p.stat().st_size, reverse=True)
    if not files:
        raise FileNotFoundError(f"No registered images found: {reg_dir}")
    return files


FILE_NAME = "fused_n4.zarr"
ZARR_TILE_SIZE = 512


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


def _distribute_file_budget(rois: list[str], counts: dict[str, int], total: int) -> dict[str, int]:
    """Allocate a global file budget across ROIs using proportional rounding.

    Uses largest-remainder rounding so that the total allocation sums to ``total``
    while never exceeding the per-ROI availability in ``counts``.
    """

    available = sum(counts.get(roi, 0) for roi in rois)
    if available == 0:
        return {roi: 0 for roi in rois}
    if total >= available:
        return {roi: counts.get(roi, 0) for roi in rois}

    quotas: dict[str, int] = {roi: 0 for roi in rois}
    fractional: list[tuple[str, float, int]] = []

    allocated = 0
    for idx, roi in enumerate(rois):
        count = counts.get(roi, 0)
        if count == 0:
            fractional.append((roi, 0.0, idx))
            continue
        share = (total * count) / available
        base = min(count, int(math.floor(share)))
        quotas[roi] = base
        allocated += base
        fractional.append((roi, share - base, idx))

    remainder = total - allocated
    if remainder <= 0:
        return quotas

    # Distribute leftover units to ROIs with the largest fractional remainder,
    # breaking ties by favouring ROIs with more availability first and then by index.
    fractional.sort(key=lambda item: (item[1], counts.get(item[0], 0), -item[2]), reverse=True)
    for roi, _frac, _idx in fractional:
        if remainder <= 0:
            break
        if quotas[roi] >= counts.get(roi, 0):
            continue
        quotas[roi] += 1
        remainder -= 1

    return quotas


# ---------- Commands ----------


def cmd_extract(
    mode: Annotated[str, typer.Argument(help="Mode: 'z' or 'ortho'")],
    path: Annotated[
        Path,
        typer.Argument(help="Workspace root", exists=True, file_okay=False, dir_okay=True, resolve_path=True),
    ],
    roi: Annotated[
        str | None,
        typer.Option("--roi", "-r", help="ROI name (e.g., cortex)", rich_help_panel="Inputs"),
    ] = None,
    codebook: Annotated[
        str,
        typer.Option("--codebook", "-c", help="Registration codebook", rich_help_panel="Inputs"),
    ] = ...,
    out: Annotated[
        Path | None,
        typer.Option(
            "--out", help="Output dir; defaults under analysis/deconv/segment--{roi}+{codebook}/extract_*"
        ),
    ] = None,
    dz: Annotated[int, typer.Option("--dz", help="Step between Z planes (z)", min=1)] = 1,
    n: Annotated[
        int,
        typer.Option(
            "--n",
            help=(
                "Count: z→ number of z-slices sampled uniformly (overrides --dz); "
                "ortho→ number of positions; also limits the number of files processed by "
                "taking a uniform sample across file sizes (largest→smallest)."
            ),
            min=1,
        ),
    ] = 50,
    anisotropy: Annotated[int, typer.Option("--anisotropy", help="Z scale factor (ortho)", min=1)] = 6,
    channels: Annotated[
        str | None,
        typer.Option("--channels", help="Indices or names, comma-separated; or 'auto'"),
    ] = None,
    crop: Annotated[int, typer.Option("--crop", help="Trim pixels at borders", min=0)] = 0,
    threads: Annotated[int, typer.Option("--threads", "-t", help="Parallel workers", min=1, max=64)] = 8,
    upscale: Annotated[
        float | None,
        typer.Option("--upscale", help="Additional spatial upscale factor applied before saving.", min=0.1),
    ] = None,
    seed: Annotated[int | None, typer.Option("--seed", help="Random seed for sampling.")] = None,
    every: Annotated[int, typer.Option("--every", help="Process every Nth file by size", min=1)] = 1,
    max_from: Annotated[
        str | None, typer.Option("--max-from", help="Append max across channels from this codebook")
    ] = None,
    use_zarr: Annotated[
        bool,
        typer.Option(
            "--zarr",
            help="Force reading inputs from fused Zarr store produced by 'preprocess stitch combine'",
            rich_help_panel="Inputs",
        ),
    ] = False,
) -> None:
    """Unified command for extracting Z-slices or orthogonal projections from registered stacks."""
    mode = mode.lower().strip()
    if mode not in {"z", "ortho"}:
        raise typer.BadParameter("Mode must be 'z' or 'ortho'.")

    # Validate mode-specific parameters
    if mode == "z" and anisotropy != 6:
        raise typer.BadParameter("--anisotropy parameter is only valid for 'ortho' mode.")

    if mode == "ortho" and dz != 1:
        raise typer.BadParameter("--dz parameter is only valid for 'z' mode.")

    if upscale is None:
        upscale = 1.0
    if upscale <= 0:
        raise typer.BadParameter("--upscale must be positive.")

    if use_zarr and max_from is not None:
        raise typer.BadParameter("--max-from cannot be used together with --zarr inputs.")

    ws = Workspace(path)

    if roi is not None:
        rois = ws.resolve_rois([roi])
    else:
        if not ws.rois:
            raise FileNotFoundError("Workspace contains no ROIs.")
        rois = ws.resolve_rois(ws.rois)

    inputs_by_roi: dict[str, list[Path]] = {}
    available_counts: dict[str, int] = {}
    for current_roi in rois:
        inputs = _discover_registered_inputs(
            ws,
            current_roi,
            codebook,
            require_zarr=use_zarr,
        )
        inputs_by_roi[current_roi] = inputs
        available_counts[current_roi] = len(inputs[::every])

    file_quota: dict[str, int | None] = {roi_name: None for roi_name in rois}
    if roi is None:
        total_available = sum(available_counts.values())
        if total_available == 0:
            raise FileNotFoundError("No registered inputs matched the selection criteria across any ROI.")
        file_quota = (
            {roi_name: available_counts[roi_name] for roi_name in rois}
            if n >= total_available
            else _distribute_file_budget(rois, available_counts, n)
        )

    for idx, current_roi in enumerate(rois):
        roi_out = out
        if out is not None and len(rois) > 1:
            roi_out = out / current_roi

        quota = file_quota.get(current_roi)
        if quota == 0:
            logger.info(f"[{current_roi}] Skipping ROI because global file budget allocated zero files.")
            continue

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
            file_quota=quota,
            prefetched_inputs=inputs_by_roi[current_roi],
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
    file_quota: int | None,
    prefetched_inputs: list[Path] | None,
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
    limit = file_quota if file_quota is not None else n
    if limit and len(files) > limit:
        import numpy as _np

        idxs = _np.linspace(0, len(files) - 1, limit, dtype=int)
        idxs = _np.unique(idxs)
        files = [files[i] for i in idxs]
        logger.info(f"[{roi}] Uniform file sampling: {len(files)} of {len(inputs)} across sizes")
    if not files:
        raise FileNotFoundError(f"[{roi}] No registered inputs matched the selection criteria.")

    max_from_path: Path | None = None
    if max_from:
        max_from_path = ws.registered(roi, max_from)
        if not max_from_path.exists():
            raise FileNotFoundError(f"[{roi}] --max-from codebook not found: {max_from_path}")

        missing = [
            str((max_from_path / f.name).resolve()) for f in files if not (max_from_path / f.name).exists()
        ]
        if missing:
            raise FileNotFoundError(
                f"[{roi}] --max-from is missing registered file(s): {', '.join(sorted(missing))}"
            )

    logger.info(f"[{roi}] Using {len(files)} registered {'file' if len(files) == 1 else 'files'}")
    logger.info(f"[{roi}] Max-from: {max_from_path if max_from_path else 'None'}")

    if files and all(_is_zarr_path(f) for f in files) and mode == "z":
        tile_jobs: list[tuple[Path, object, list[tuple[int, int]], list[str] | None, list[int]]] = []
        total_outputs = 0
        for f in files:
            vol, names_all = _open_volume(f)
            tile_origins = _compute_tile_origins(
                vol.shape,
                tile_size=ZARR_TILE_SIZE,
                n_tiles=n,
                crop=crop,
            )
            z_candidates = list(range(0, vol.shape[0], dz))
            if not z_candidates:
                raise ValueError(f"[{roi}] No Z indices available after applying dz to fused Zarr volume.")
            total_outputs += len(tile_origins) * len(z_candidates)
            tile_jobs.append((f, vol, tile_origins, names_all, z_candidates))

        logger.info(f"[{roi}] Tracking {total_outputs} output files from Zarr input(s)")

        with progress_bar(total_outputs) as progress_update:
            for f, vol, tile_origins, names_all, z_candidates in tile_jobs:
                _process_zarr_tiles(
                    f,
                    roi,
                    vol,
                    names_all,
                    tile_origins,
                    out_dir,
                    channels,
                    crop,
                    z_candidates,
                    max_from_path,
                    upscale,
                    progress_update,
                )
        return

    if files and all(_is_zarr_path(f) for f in files) and mode == "ortho":
        rng = np.random.default_rng(seed)
        logger.info(f"[{roi}] Random seed: {seed if seed is not None else 'system entropy'}")
        ortho_jobs: list[tuple[Path, object, list[str] | None, list[int], list[int]]] = []
        total_outputs = 0
        for f in files:
            vol, names_all = _open_volume(f)
            y_positions = _sample_positions(vol.shape[1], crop=crop, count=n, rng=rng)
            x_positions = _sample_positions(vol.shape[2], crop=crop, count=n, rng=rng)
            total_outputs += len(y_positions) + len(x_positions)
            ortho_jobs.append((f, vol, names_all, y_positions, x_positions))

        logger.info(f"[{roi}] Tracking {total_outputs} ortho outputs from Zarr input(s)")

        with progress_bar(total_outputs) as progress_update:
            for f, vol, names_all, y_positions, x_positions in ortho_jobs:
                _process_zarr_ortho(
                    f,
                    roi,
                    vol,
                    names_all,
                    y_positions,
                    x_positions,
                    out_dir,
                    channels,
                    crop,
                    anisotropy,
                    max_from_path,
                    upscale,
                    progress_update,
                )
        return

    with progress_bar_threadpool(len(files), threads=threads, stop_on_exception=True) as submit:
        if mode == "z":
            for f in files:
                submit(
                    _process_file_to_z_slices,
                    f,
                    roi,
                    out_dir,
                    channels,
                    crop,
                    dz,
                    n,
                    max_from_path,
                    upscale,
                )
        else:
            for f in files:
                submit(
                    _process_file_to_ortho,
                    f,
                    roi,
                    out_dir,
                    channels,
                    crop,
                    n,
                    anisotropy,
                    max_from_path,
                    upscale,
                )


# ---------- Logging helpers ----------


@lru_cache(maxsize=1024)
def _path_size_cached(path_str: str) -> int:
    """Return on-disk size in bytes for a file or directory.

    For directories (e.g., fused .zarr), sums contained file sizes without following symlinks.
    Cached by absolute path string to avoid repeated traversal.
    """
    p = Path(path_str)
    try:
        if p.is_file():
            return p.stat().st_size
        if p.is_dir():
            total = 0

            # Use scandir-based recursion for speed over os.walk
            def _scan(d: Path) -> int:
                t = 0
                try:
                    with os.scandir(d) as it:
                        for entry in it:
                            try:
                                if entry.is_file(follow_symlinks=False):
                                    t += entry.stat(follow_symlinks=False).st_size
                                elif entry.is_dir(follow_symlinks=False):
                                    t += _scan(Path(entry.path))
                            except OSError:
                                continue
                except OSError:
                    return 0
                return t

            total = _scan(p)
            return total
    except OSError:
        return 0
    return 0


def _format_size(num_bytes: int) -> str:
    """Format byte size into human-friendly string using binary units."""
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    n = float(num_bytes)
    for u in units:
        if n < 1024.0 or u == units[-1]:
            if u == "B":
                return f"{int(n)} {u}"
            return f"{n:.1f} {u}"
        n /= 1024.0
    return f"{n:.1f} TiB"


def _default_output_dir(ws: Workspace, roi: str) -> Path:
    """Return the default extract output directory for a given ROI."""
    return ws.deconved / "segment" / roi


def _prefix_with_roi(name: str, roi: str) -> str:
    """Prefix output filename with the ROI identifier."""
    prefix = f"{roi}--"
    return name if name.startswith(prefix) else f"{prefix}{name}"


def _process_file_to_z_slices(
    file: Path,
    roi: str,
    out_dir: Path,
    channels: str | None,
    crop: int,
    dz: int,
    n: int | None,
    max_from_path: Path | None,
    upscale: float,
    *,
    progress: Callable[[], None] | None = None,
) -> None:
    """Extract individual Z-slices as CYX TIFF files with specified step size.

    Works uniformly for TIFF and Zarr volumes; Zarr is accessed lazily per slice.
    """
    size_str = _format_size(_path_size_cached(str(file.resolve())))
    logger.info(f"3D→Z: {file.name} [{size_str}] (dz={dz})")
    vol, names_all = _open_volume(file)
    chsel = _parse_channels(channels, names_all)
    base_names = _resolve_output_names(chsel, channels)
    other_vol = None
    if max_from_path:
        # Support Zarr other; for TIFF, try matching file path under dir
        target = (
            max_from_path
            if _is_zarr_path(max_from_path)
            else (max_from_path / file.name if max_from_path.is_dir() else max_from_path)
        )
        if target.exists():
            other_vol, _ = _open_volume(target)
    out_names = [*base_names, "max_from"] if other_vol is not None else base_names

    z_len, y_len, x_len, _ = vol.shape
    # Normal mode: crop to top-left 1024x1024 region regardless of --crop.
    # If image is smaller than 1024 in either dimension, use full available size.
    y_slice = slice(0, min(1024, y_len))
    x_slice = slice(0, min(1024, x_len))

    # Determine z indices: use uniform sampling if n is provided; else step by dz
    if n is not None:
        idxs = np.linspace(0, z_len - 1, n, dtype=int)
        idxs = np.unique(idxs)
    else:
        idxs = range(0, z_len, dz)

    for i in idxs:
        plane = vol[i, :, :, :]  # (Y,X,C)
        other_max = None
        if other_vol is not None:
            # Compute max across channels for the same Z-plane. Do not crop here;
            # cropping is applied uniformly inside _prep_slab via crop_slices.
            other_max = other_vol[i, :, :, :].max(axis=2)  # (Y,X)
        cyx_u16 = _prep_slab(
            plane[:, :, :],
            ch_idx=chsel.indices,
            channel_axis=2,
            crop_slices=(y_slice, x_slice),
            filter_before=True,
            append_max=other_max,
        )
        cyx_u16 = _resize_uint16(cyx_u16, (1.0, upscale, upscale))
        out_name = _prefix_with_roi(f"{file.stem}_z{i:02d}.tif", roi)
        out_file = out_dir / out_name
        _write_tiff(
            out_file,
            cyx_u16,
            axes="CYX",
            names=out_names,
            channels_arg=channels,
            upscale=upscale,
        )
        if progress is not None:
            progress()


def _process_file_to_ortho(
    file: Path,
    roi: str,
    out_dir: Path,
    channels: str | None,
    crop: int,
    n: int,
    anisotropy: int,
    max_from_path: Path | None,
    upscale: float,
    *,
    progress: Callable[[], None] | None = None,
) -> None:
    """Generate orthogonal ZX/ZY projections with Z-axis scaling for anisotropy correction.

    Unified streaming path for both TIFF and Zarr volumes.
    """
    size_str = _format_size(_path_size_cached(str(file.resolve())))
    logger.info(f"Ortho: {file.name} [{size_str}]")

    vol, names_all = _open_volume(file)
    chsel = _parse_channels(channels, names_all)
    base_names = _resolve_output_names(chsel, channels)
    other_vol = None
    if max_from_path:
        target = (
            max_from_path
            if _is_zarr_path(max_from_path)
            else (max_from_path / file.name if max_from_path.is_dir() else max_from_path)
        )
        if target.exists():
            other_vol, _ = _open_volume(target)
    out_names = [*base_names, "max_from"] if other_vol is not None else base_names

    z_len, y_len, x_len, _ = vol.shape
    y_eff = y_len - 2 * crop
    x_eff = x_len - 2 * crop
    if y_eff <= 0 or x_eff <= 0:
        raise ValueError("Image after cropping is empty.")

    y_positions = (np.linspace(int(0.1 * y_eff), int(0.9 * y_eff), n).astype(int) + crop).tolist()
    x_positions = (np.linspace(int(0.1 * x_eff), int(0.9 * x_eff), n).astype(int) + crop).tolist()
    x_slice = slice(crop, x_len - crop) if crop > 0 else slice(None)
    y_slice = slice(crop, y_len - crop) if crop > 0 else slice(None)

    # ZX slabs (fixed Y)
    for yi in y_positions:
        slab = vol[:, yi, x_slice, :]  # (Z,X,C)
        other_max = None
        if other_vol is not None:
            other_max = other_vol[:, yi, x_slice, :].max(axis=2)  # (Z,X)
        czx = _prep_slab(
            slab,
            ch_idx=chsel.indices,
            channel_axis=2,
            crop_slices=None,  # already cropped on X via slice
            filter_before=False,
            append_max=other_max,
        )  # (C,Z,X)
        czx = _resize_uint16(czx, (1.0, anisotropy * upscale, upscale))
        out_name = _prefix_with_roi(f"{file.stem}_orthozx-{yi}.tif", roi)
        out_file = out_dir / out_name
        _write_tiff(
            out_file,
            czx,
            axes="CZX",
            names=out_names,
            channels_arg=channels,
            upscale=upscale,
        )
        if progress is not None:
            progress()

    # ZY slabs (fixed X)
    for xi in x_positions:
        slab = vol[:, y_slice, xi, :]  # (Z,Y,C)
        other_max = None
        if other_vol is not None:
            other_max = other_vol[:, y_slice, xi, :].max(axis=2)  # (Z,Y)
        czy = _prep_slab(
            slab,
            ch_idx=chsel.indices,
            channel_axis=2,
            crop_slices=None,
            filter_before=False,
            append_max=other_max,
        )  # (C,Z,Y)
        czy = _resize_uint16(czy, (1.0, anisotropy * upscale, upscale))
        out_name = _prefix_with_roi(f"{file.stem}_orthozy-{xi}.tif", roi)
        out_file = out_dir / out_name
        _write_tiff(
            out_file,
            czy,
            axes="CZY",
            names=out_names,
            channels_arg=channels,
            upscale=upscale,
        )
        if progress is not None:
            progress()


def _sample_positions(length: int, *, crop: int, count: int, rng: Generator) -> list[int]:
    start = crop
    end = length - crop
    if end <= start:
        raise ValueError("Image after cropping is empty.")

    population = np.arange(start, end)
    if not population.size:
        raise ValueError("No positions available after cropping.")

    if count >= population.size:
        return population.tolist()

    choices = rng.choice(population, size=count, replace=False)
    return sorted(int(v) for v in choices)


def _compute_tile_origins(
    shape: tuple[int, int, int, int],
    *,
    tile_size: int,
    n_tiles: int | None,
    crop: int,
) -> list[tuple[int, int]]:
    _z, y_len, x_len, _ = shape
    if y_len < tile_size + 2 * crop or x_len < tile_size + 2 * crop:
        raise ValueError("Zarr volume spatial dimensions are smaller than the requested 512×512 tile size.")

    start_y = crop
    start_x = crop
    max_y = y_len - crop - tile_size
    max_x = x_len - crop - tile_size
    if max_y < start_y or max_x < start_x:
        raise ValueError("Tile cropping leaves no valid region to sample.")

    y_range = max_y - start_y + 1
    x_range = max_x - start_x + 1
    total_positions = y_range * x_range
    desired = n_tiles or 50
    desired = min(desired, total_positions)
    idxs = np.linspace(0, total_positions - 1, desired, dtype=int)
    idxs = np.unique(idxs)
    if idxs.size < desired:
        extras = np.setdiff1d(np.arange(total_positions), idxs, assume_unique=False)[: desired - idxs.size]
        idxs = np.sort(np.concatenate([idxs, extras]))

    origins: list[tuple[int, int]] = []
    for idx in idxs:
        offset_y = idx // x_range
        offset_x = idx % x_range
        origins.append((start_y + int(offset_y), start_x + int(offset_x)))
    return origins


def _format_tile_filename(
    stem: str,
    roi: str,
    z_index: int,
    y0: int,
    x0: int,
    *,
    coord_width: int,
) -> str:
    name = f"{stem}_y{y0:0{coord_width}d}_x{x0:0{coord_width}d}_z{z_index:02d}.tif"
    return _prefix_with_roi(name, roi)


def _process_zarr_tiles(
    file: Path,
    roi: str,
    vol: object,
    names_all: list[str] | None,
    tile_origins: list[tuple[int, int]],
    out_dir: Path,
    channels: str | None,
    crop: int,
    z_candidates: list[int],
    max_from_path: Path | None,
    upscale: float,
    progress: Callable[[], None],
) -> None:
    chsel = _parse_channels(channels, names_all)
    base_names = _resolve_output_names(chsel, channels)

    other_vol = None
    if max_from_path:
        target = (
            max_from_path
            if _is_zarr_path(max_from_path)
            else (max_from_path / file.name if max_from_path.is_dir() else max_from_path)
        )
        if target.exists():
            other_vol, _ = _open_volume(target)

    out_names = [*base_names, "max_from"] if other_vol is not None else base_names
    coord_width = len(str(max(vol.shape[1], vol.shape[2])))

    for y0, x0 in tile_origins:
        y_slice = slice(y0, y0 + ZARR_TILE_SIZE)
        x_slice = slice(x0, x0 + ZARR_TILE_SIZE)
        skipped_for_tile = 0
        for z_index in z_candidates:
            plane = vol[z_index, y_slice, x_slice, :]
            zero_tile = np.any(plane == 0)
            other_max = None
            if other_vol is not None:
                other_tile = other_vol[z_index, y_slice, x_slice, :]
                if not zero_tile:
                    zero_tile = np.any(other_tile == 0)
                if not zero_tile:
                    other_max = other_tile.max(axis=2)

            if zero_tile:
                skipped_for_tile += 1
                progress()
                continue

            cyx_u16 = _prep_slab(
                plane,
                ch_idx=chsel.indices,
                channel_axis=2,
                crop_slices=None,
                filter_before=True,
                append_max=other_max,
                apply_filter=False,
            )
            cyx_u16 = _resize_uint16(cyx_u16, (1.0, upscale, upscale))

            out_name = _format_tile_filename(
                file.stem,
                roi,
                z_index,
                y0,
                x0,
                coord_width=coord_width,
            )
            _write_tiff(
                out_dir / out_name,
                cyx_u16,
                axes="CYX",
                names=out_names,
                channels_arg=channels,
                upscale=upscale,
            )
            progress()

        if skipped_for_tile:
            logger.debug(
                f"Skipped {skipped_for_tile} z-slice(s) in tile ({y0},{x0}) of {file.name} due to zeros."
            )


def _process_zarr_ortho(
    file: Path,
    roi: str,
    vol: object,
    names_all: list[str] | None,
    y_positions: list[int],
    x_positions: list[int],
    out_dir: Path,
    channels: str | None,
    crop: int,
    anisotropy: int,
    max_from_path: Path | None,
    upscale: float,
    progress: Callable[[], None],
) -> None:
    chsel = _parse_channels(channels, names_all)
    base_names = _resolve_output_names(chsel, channels)

    other_vol = None
    if max_from_path:
        target = (
            max_from_path
            if _is_zarr_path(max_from_path)
            else (max_from_path / file.name if max_from_path.is_dir() else max_from_path)
        )
        if target.exists():
            other_vol, _ = _open_volume(target)

    out_names = [*base_names, "max_from"] if other_vol is not None else base_names

    z_len, y_len, x_len, _ = vol.shape
    y_slice = slice(crop, y_len - crop) if crop > 0 else slice(None)
    x_slice = slice(crop, x_len - crop) if crop > 0 else slice(None)

    for yi in y_positions:
        slab = vol[:, yi, x_slice, :]
        other_max = None
        if other_vol is not None:
            other_max = other_vol[:, yi, x_slice, :].max(axis=2)
        czx = _prep_slab(
            slab,
            ch_idx=chsel.indices,
            channel_axis=2,
            crop_slices=None,
            filter_before=False,
            append_max=other_max,
            apply_filter=False,
        )
        czx = _resize_uint16(czx, (1.0, anisotropy * upscale, upscale))
        out_name = _prefix_with_roi(f"{file.stem}_orthozx-{yi}.tif", roi)
        out_file = out_dir / out_name
        _write_tiff(
            out_file,
            czx,
            axes="CZX",
            names=out_names,
            channels_arg=channels,
            upscale=upscale,
        )
        progress()

    for xi in x_positions:
        slab = vol[:, y_slice, xi, :]
        other_max = None
        if other_vol is not None:
            other_max = other_vol[:, y_slice, xi, :].max(axis=2)
        czy = _prep_slab(
            slab,
            ch_idx=chsel.indices,
            channel_axis=2,
            crop_slices=None,
            filter_before=False,
            append_max=other_max,
            apply_filter=False,
        )
        czy = _resize_uint16(czy, (1.0, anisotropy * upscale, upscale))
        out_name = _prefix_with_roi(f"{file.stem}_orthozy-{xi}.tif", roi)
        out_file = out_dir / out_name
        _write_tiff(
            out_file,
            czy,
            axes="CZY",
            names=out_names,
            channels_arg=channels,
            upscale=upscale,
        )
        progress()


app = typer.Typer(help="Internal helper exposing cmd_extract for tests.")


@app.callback(invoke_without_command=True)
def _app_entrypoint(
    mode: Annotated[str, typer.Argument(help="Mode: 'z' or 'ortho'")],
    path: Annotated[
        Path,
        typer.Argument(help="Workspace root", exists=True, file_okay=False, dir_okay=True, resolve_path=True),
    ],
    roi: Annotated[
        str | None,
        typer.Option("--roi", "-r", help="ROI name (e.g., cortex)", rich_help_panel="Inputs"),
    ] = None,
    codebook: Annotated[
        str,
        typer.Option("--codebook", "-c", help="Registration codebook", rich_help_panel="Inputs"),
    ] = ...,
    out: Annotated[
        Path | None,
        typer.Option(
            "--out", help="Output dir; defaults under analysis/deconv/segment--{roi}+{codebook}/extract_*"
        ),
    ] = None,
    dz: Annotated[int, typer.Option("--dz", help="Step between Z planes (z)", min=1)] = 1,
    n: Annotated[
        int,
        typer.Option(
            "--n",
            help=(
                "Count: z→ number of z-slices sampled uniformly (overrides --dz); "
                "ortho→ number of positions; also limits the number of files processed by "
                "taking a uniform sample across file sizes (largest→smallest)."
            ),
            min=1,
        ),
    ] = 50,
    anisotropy: Annotated[int, typer.Option("--anisotropy", help="Z scale factor (ortho)", min=1)] = 6,
    channels: Annotated[
        str | None,
        typer.Option("--channels", help="Indices or names, comma-separated; or 'auto'"),
    ] = None,
    crop: Annotated[int, typer.Option("--crop", help="Trim pixels at borders", min=0)] = 0,
    threads: Annotated[int, typer.Option("--threads", "-t", help="Parallel workers", min=1, max=64)] = 8,
    upscale: Annotated[
        float | None,
        typer.Option("--upscale", help="Additional spatial upscale factor applied before saving.", min=0.1),
    ] = None,
    seed: Annotated[int | None, typer.Option("--seed", help="Random seed for sampling.")] = None,
    every: Annotated[int, typer.Option("--every", help="Process every Nth file by size", min=1)] = 1,
    max_from: Annotated[
        str | None, typer.Option("--max-from", help="Append max across channels from this codebook")
    ] = None,
    use_zarr: Annotated[
        bool,
        typer.Option(
            "--zarr",
            help="Force reading inputs from fused Zarr store produced by 'preprocess stitch combine'",
            rich_help_panel="Inputs",
        ),
    ] = False,
) -> None:
    cmd_extract(
        mode,
        path,
        roi=roi,
        codebook=codebook,
        out=out,
        dz=dz,
        n=n,
        anisotropy=anisotropy,
        channels=channels,
        crop=crop,
        threads=threads,
        upscale=upscale,
        seed=seed,
        every=every,
        max_from=max_from,
        use_zarr=use_zarr,
    )


__all__ = ["cmd_extract", "app"]
