from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Iterable, Optional, Sequence

import cupy as cp
import numpy as np
import rich
import rich_click as click
import tifffile as tiff
from loguru import logger

from fishtools.io.workspace import Workspace
from fishtools.preprocess.spots.illumination import RangeFieldPointsModel
from fishtools.utils.logging import setup_workspace_logging
from fishtools.utils.pretty_print import progress_bar, progress_bar_threadpool
from fishtools.utils.tiff import normalize_channel_names, read_metadata_from_tif


def _edges(n: int, grid: int) -> list[int]:
    return list(np.linspace(0, n, grid + 1, dtype=int))


def _load_czyx(tile_path: Path, use_gpu: bool) -> tuple[np.ndarray | cp.ndarray, list[str]]:
    with tiff.TiffFile(tile_path) as tif:
        meta = read_metadata_from_tif(tif)
        arr = tif.asarray()
    if arr.ndim != 4:
        raise ValueError(f"Expected ZCYX layout in {tile_path.name}, got shape {arr.shape}")
    arr = np.swapaxes(arr, 0, 1)  # C, Z, Y, X
    arr = arr[:, ::2, :, :]
    channels = normalize_channel_names(int(arr.shape[0]), meta) or [
        f"channel_{i}" for i in range(arr.shape[0])
    ]
    if use_gpu:
        if cp is None:
            raise RuntimeError("CuPy is required for GPU percentile computation but is not installed")
        carr = cp.asarray(arr, dtype=cp.float32)
        return cp.ascontiguousarray(carr), [str(c) for c in channels]
    narr = np.asarray(arr, dtype=np.float32)
    return np.ascontiguousarray(narr), [str(c) for c in channels]


def _compute_subtile_percentiles_gpu(
    arr: "cp.ndarray",
    percentiles: tuple[float, float],
    grid: int,
    max_streams: int,
    channel_chunk: int,
) -> tuple[np.ndarray, np.ndarray]:
    if cp is None:
        raise RuntimeError("CuPy is not available for GPU percentile computation")
    c, _, h, w = arr.shape
    y_edges = _edges(h, grid)
    x_edges = _edges(w, grid)
    lo_dev = cp.empty((grid, grid, c), dtype=cp.float32)
    hi_dev = cp.empty((grid, grid, c), dtype=cp.float32)
    q = cp.asarray([percentiles[0] / 100.0, percentiles[1] / 100.0], dtype=cp.float32)
    chunk = max(1, int(channel_chunk))
    tasks = [(r, col, k) for r in range(grid) for col in range(grid) for k in range((c + chunk - 1) // chunk)]
    n_streams = max(1, min(int(max_streams or 1), len(tasks)))
    streams = [cp.cuda.Stream(non_blocking=True) for _ in range(n_streams)] if n_streams > 1 else []
    si = 0
    for r, col, k in tasks:
        y0, y1 = y_edges[r], y_edges[r + 1]
        x0, x1 = x_edges[col], x_edges[col + 1]
        c0 = k * chunk
        c1 = min(c, c0 + chunk)
        view = cp.ascontiguousarray(arr[c0:c1, :, y0:y1, x0:x1])
        if streams:
            s = streams[si]
            si = (si + 1) % n_streams
            with s:
                pct = cp.quantile(view, q, axis=(1, 2, 3))
                lo_dev[r, col, c0:c1] = pct[0]
                hi_dev[r, col, c0:c1] = pct[1]
        else:
            pct = cp.quantile(view, q, axis=(1, 2, 3))
            lo_dev[r, col, c0:c1] = pct[0]
            hi_dev[r, col, c0:c1] = pct[1]
    cp.cuda.runtime.deviceSynchronize()
    return cp.asnumpy(lo_dev), cp.asnumpy(hi_dev)


def _compute_subtile_percentiles_cpu(
    arr: np.ndarray,
    percentiles: tuple[float, float],
    grid: int,
) -> tuple[np.ndarray, np.ndarray]:
    c, _, h, w = arr.shape
    y_edges = _edges(h, grid)
    x_edges = _edges(w, grid)
    lo = np.empty((grid, grid, c), dtype=np.float32)
    hi = np.empty((grid, grid, c), dtype=np.float32)
    for r in range(grid):
        for col in range(grid):
            y0, y1 = y_edges[r], y_edges[r + 1]
            x0, x1 = x_edges[col], x_edges[col + 1]
            sub = arr[:, :, y0:y1, x0:x1]
            pct = np.percentile(sub, [percentiles[0], percentiles[1]], axis=(1, 2, 3))
            lo[r, col, :] = pct[0]
            hi[r, col, :] = pct[1]
    return lo, hi


def _skip_existing(tile: Path, out_suffix: str, grid: int, keys: tuple[str, str], overwrite: bool) -> bool:
    if overwrite:
        return False
    out = tile.with_suffix(out_suffix)
    if not out.exists():
        return False
    try:
        payload = json.loads(out.read_text())
    except Exception:
        return False
    if not isinstance(payload, dict) or not payload:
        return False
    first_entries = next(iter(payload.values()))
    if not isinstance(first_entries, list):
        return False
    if len(first_entries) != grid * grid:
        return False
    percentiles = first_entries[0].get("percentiles", {})
    return set(percentiles.keys()) == set(keys)


def _mask_union_of_tiles(
    xs: np.ndarray,
    ys: np.ndarray,
    xs0: np.ndarray,
    ys0: np.ndarray,
    tile_w: float,
    tile_h: float,
) -> np.ndarray:
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    x0 = np.asarray(xs0, dtype=np.float64)
    y0 = np.asarray(ys0, dtype=np.float64)
    if xs.size == 0 or ys.size == 0 or x0.size == 0:
        return np.ones((ys.size, xs.size), dtype=bool)
    x1 = x0 + float(tile_w)
    y1 = y0 + float(tile_h)

    dx = float(np.min(np.diff(xs))) if xs.size > 1 else 0.0
    dy = float(np.min(np.diff(ys))) if ys.size > 1 else 0.0
    tol = max(dx, dy, 1.0) * 1e-6

    X = (xs[None, :] >= (x0[:, None] - tol)) & (xs[None, :] <= (x1[:, None] + tol))
    Y = (ys[None, :] >= (y0[:, None] - tol)) & (ys[None, :] <= (y1[:, None] + tol))
    mask_counts = Y.astype(np.uint16).T @ X.astype(np.uint16)
    return mask_counts > 0


def _resolve_tile_origins(
    meta: dict[str, object], ws: Workspace, roi: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tc = ws.tileconfig(roi)
    df = tc.df
    indices = df.select("index").to_numpy().reshape(-1).astype(np.int32)
    xs0 = df.select("x").to_numpy().reshape(-1).astype(np.float64)
    ys0 = df.select("y").to_numpy().reshape(-1).astype(np.float64)
    return indices, xs0, ys0


_ROI_WILDCARD_TOKENS = {"*", "all"}


def _normalize_rois(
    ws: Workspace,
    tokens: tuple[str, ...],
    meta_roi: str | None,
) -> list[str]:
    """Resolve ROI tokens using workspace discovery with wildcard support."""

    def _resolve(rois: Iterable[str] | None) -> list[str]:
        try:
            return ws.resolve_rois(rois)
        except ValueError as exc:
            raise click.UsageError(str(exc)) from exc

    requested = [token for token in tokens if token]
    if not requested:
        if meta_roi:
            return _resolve([str(meta_roi)])
        return _resolve(None)

    includes_wildcard = any(token in _ROI_WILDCARD_TOKENS for token in requested)
    explicit: list[str] = []
    seen: set[str] = set()
    for token in requested:
        if token in _ROI_WILDCARD_TOKENS:
            continue
        if token not in seen:
            explicit.append(token)
            seen.add(token)

    if includes_wildcard and not explicit:
        return _resolve(None)

    if not explicit:
        raise click.UsageError("ROI selection resolved to an empty set.")

    return _resolve(explicit)


def _ensure_fields_dir(ws: Workspace, codebook: str) -> tuple[Path, str, str]:
    """Return (dir, codebook_label, codebook_slug) for standardized fields storage."""

    codebook_path = Path(codebook)
    codebook_label = codebook_path.stem if codebook_path.suffix else str(codebook_path)
    codebook_slug = Workspace.sanitize_codebook_name(codebook_label)
    fields_dir = ws.deconved / f"fields+{codebook_slug}"
    fields_dir.mkdir(parents=True, exist_ok=True)
    return fields_dir, codebook_label, codebook_slug


def _slugify_token(token: str) -> str:
    """Normalize tokens for filesystem usage."""

    return token.replace(" ", "_").replace("/", "-")


@click.group("correct-illum")
def correct_illum() -> None:
    """Illumination correction utilities."""


@correct_illum.command("render-global")
@click.argument(
    "model",
    type=click.Path(exists=True, dir_okay=False, file_okay=True, path_type=Path),
)
@click.option(
    "--workspace",
    type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=Path),
    default=None,
    help="Workspace root (defaults to value stored in model metadata)",
)
@click.option("--roi", type=str, default=None, help="ROI (defaults to value stored in model)")
@click.option("--codebook", type=str, default=None, help="Optional codebook (metadata fallback)")
@click.option(
    "-n",
    "--first-n",
    type=int,
    default=0,
    show_default=True,
    help="Render only the first N tiles by index (0 = all).",
)
@click.option(
    "--mode",
    type=click.Choice(["low", "high", "range"], case_sensitive=False),
    default="range",
    show_default=True,
    help="Field to stamp per tile.",
)
@click.option("--grid-step", type=float, default=None, help="Override RBF evaluation grid step")
@click.option(
    "--neighbors",
    type=int,
    default=0,
    show_default=True,
    help="Neighbors for RBF evaluation (0 = use all / defer to model).",
)
@click.option(
    "--smoothing",
    type=float,
    default=None,
    help="L2 smoothing (Tikhonov) for RBF; larger is smoother. Default uses model's value.",
)
@click.option(
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Output TIFF path (defaults next to model)",
)
def render_global(
    model: Path,
    workspace: Path | None,
    roi: str | None,
    codebook: str | None,
    first_n: int,
    mode: str,
    grid_step: float | None,
    neighbors: int,
    smoothing: float | None,
    output: Path | None,
) -> None:
    """Render a global field image by stamping per-tile patches.

    The canvas is sized from tile positions and tile dimensions. For each tile,
    we render a field patch using the illumination model and copy it into the
    canvas at the tile's origin. Overlaps are not blended (later tiles overwrite).
    """

    import numpy as np
    from tifffile import imwrite

    rng_model = RangeFieldPointsModel.from_npz(model)
    meta = rng_model.meta

    # Avoid converting missing/None ROI to literal "None"
    if roi is None:
        _roi_meta = meta.get("roi")
        roi = str(_roi_meta) if _roi_meta is not None else None
    codebook = codebook or str(meta.get("codebook")) if meta.get("codebook") is not None else None
    if not roi:
        raise click.UsageError("ROI must be provided either via --roi or present in model metadata")

    if workspace is None:
        ws_meta = meta.get("workspace")
        if ws_meta is None:
            raise click.UsageError("--workspace is required when model metadata lacks 'workspace'")
        workspace = Path(str(ws_meta))

    ws = Workspace(workspace)
    # Initialize workspace-scoped logging
    try:
        log_file_tag = f"{roi}+{codebook}" if codebook else roi
        setup_workspace_logging(
            ws.path,
            component="preprocess.correct-illum.render-global",
            file=log_file_tag,
        )
    except Exception:
        logger.opt(exception=True).debug("Workspace logging setup failed; continuing with default logger")

    logger.info(
        f"RenderGlobal: model='{model.name}', ROI='{roi}', workspace='{ws.path}', codebook='{codebook or ''}'"
    )

    # Resolve tile configuration and origins
    indices, xs0, ys0 = _resolve_tile_origins(meta, ws, roi)
    order = np.argsort(indices.astype(np.int64))
    indices = indices[order]
    xs0 = xs0[order]
    ys0 = ys0[order]
    if first_n and first_n > 0:
        keep = min(first_n, int(indices.size))
        indices = indices[:keep]
        xs0 = xs0[:keep]
        ys0 = ys0[:keep]
        logger.info(f"Limiting to first {keep} tile(s) by index")

    # Tile dimensions from model metadata (pixels in registered space)
    tile_w = int(float(meta.get("tile_w", 1968)))
    tile_h = int(float(meta.get("tile_h", 1968)))

    # Compute canvas extents from selected tiles only
    x_min = float(xs0.min()) if xs0.size else 0.0
    y_min = float(ys0.min()) if ys0.size else 0.0
    x_max = float(xs0.max()) + float(tile_w) if xs0.size else float(tile_w)
    y_max = float(ys0.max()) + float(tile_h) if ys0.size else float(tile_h)
    W = int(np.ceil(x_max - x_min))
    H = int(np.ceil(y_max - y_min))
    if W <= 0 or H <= 0:
        raise click.UsageError("Computed empty canvas (no tiles available)")

    canvas = np.zeros((H, W), dtype=np.float32)
    logger.info(f"Canvas size: {W}x{H} (tile={tile_w}x{tile_h}); tiles={int(indices.size)}; mode={mode}")

    # Prepare RBF controls
    ker = str(meta.get("kernel", "thin_plate_spline"))
    smooth_eff = float(smoothing) if smoothing is not None else float(meta.get("smoothing", 1.0))
    neigh_eff = None if neighbors is None or int(neighbors) <= 0 else int(neighbors)
    step_eff = float(grid_step) if grid_step is not None else None
    logger.info(
        "RBF params — kernel='{}', smoothing={}, neighbors={}, grid_step={}".format(
            ker,
            smooth_eff,
            neigh_eff if neigh_eff is not None else 0,
            step_eff if step_eff is not None else "meta",
        )
    )

    # If the NPZ is multi-channel, default to the first channel for rendering
    try:
        vlow = np.asarray(rng_model.vlow)
        if vlow.ndim == 2 and vlow.shape[1] > 1:
            ch_names = list(meta.get("channels", [])) if isinstance(meta.get("channels"), list) else []
            ch0 = ch_names[0] if ch_names else "ch0"
            sub_meta = dict(meta)
            sub_meta.pop("range_mean", None)
            rng_model = RangeFieldPointsModel(
                xy=rng_model.xy,
                vlow=vlow[:, 0],
                vhigh=np.asarray(rng_model.vhigh)[:, 0],
                meta=dict(sub_meta, channel=ch0),
            )
            meta = rng_model.meta
            logger.info(
                "RenderGlobal: multi-channel NPZ detected; defaulting to first channel '{}'",
                ch0,
            )
    except Exception:
        pass

    # Evaluate the global field once, normalize with the union-of-tiles mask, then slice per-tile.
    xs, ys, low_field, high_field = rng_model.evaluate(
        x_min,
        y_min,
        x_max,
        y_max,
        grid_step=step_eff,
        neighbors=neigh_eff,
        kernel=ker,
        smoothing=smooth_eff,
    )
    mask = (
        _mask_union_of_tiles(xs, ys, xs0, ys0, tile_w, tile_h) if xs0.size else np.ones_like(low_field, bool)
    )

    mode_l = mode.lower()
    if mode_l == "range":
        field_global = rng_model.range_correction(low_field, high_field, mask=mask)
    elif mode_l == "low":
        field_global = low_field
    elif mode_l == "high":
        field_global = high_field
    else:
        raise click.UsageError(f"Unsupported mode '{mode}'")

    # Stamp tiles (no blending on overlaps) with progress bar
    with progress_bar(int(indices.size)) as advance:
        for i, (x0, y0) in enumerate(zip(xs0.tolist(), ys0.tolist(), strict=False)):
            logger.debug(f"Rendering patch {i + 1}/{int(indices.size)} at ({x0:.1f}, {y0:.1f})")
            from fishtools.preprocess.spots.illumination import _slice_field_to_tile

            x1 = float(x0) + float(tile_w)
            y1 = float(y0) + float(tile_h)
            patch = _slice_field_to_tile(
                field_global,
                xs,
                ys,
                float(x0),
                float(y0),
                x1,
                y1,
                (int(tile_h), int(tile_w)),
            )
            xo = int(round(float(x0 - x_min)))
            yo = int(round(float(y0 - y_min)))
            canvas[yo : yo + tile_h, xo : xo + tile_w] = patch.astype(np.float32, copy=False)
            advance()

    # Resolve output path and write
    if output is None:
        stem = f"{model.stem}-global-{mode.lower()}"
        output = model.with_name(stem + ".tif")

    md = {
        "axes": "YX",
        "roi": roi,
        "codebook": codebook,
        "tiles": int(indices.size),
        "tile_w": int(tile_w),
        "tile_h": int(tile_h),
        "kernel": ker,
        "smoothing": float(smooth_eff),
        "neighbors": int(neigh_eff or 0),
        "grid_step": float(step_eff) if step_eff is not None else None,
    }
    # Prune None values
    md = {k: v for k, v in md.items() if v is not None}
    imwrite(output, canvas, metadata={"axes": "YX", "model_meta": md})
    logger.info(f"Saved global field ({mode}) with shape {canvas.shape} → {output}")
    rich.print(f"[green]Saved global field ({mode}) with shape {canvas.shape} → {output}[/green]")


@correct_illum.command("calculate-percentiles")
@click.argument(
    "workspace",
    type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=Path),
)
@click.argument("rois", nargs=-1, type=str)
@click.option("--idx", type=int, default=None, help="Restrict to a single tile index")
@click.option("--codebook", "codebook_opt", type=str, default=None, help="Codebook stem or path")
@click.option("--percentiles", nargs=2, type=float, default=(0.1, 99.9), show_default=True)
@click.option("--grid", type=int, default=4, show_default=True)
@click.option("--max-streams", type=int, default=1, show_default=True)
@click.option("--channel-chunk", type=int, default=1, show_default=True)
@click.option(
    "--threads",
    type=int,
    default=4,
    show_default=True,
    help="CPU threads when GPU disabled",
)
@click.option(
    "--out-suffix",
    default=".subtiles.json",
    show_default=True,
    help="Suffix for per-tile percentile JSON",
)
@click.option("--overwrite/--no-overwrite", default=False, show_default=True)
@click.option(
    "--sample-step",
    type=int,
    default=1,
    show_default=True,
    help="Spatial stride for subsampling within each tile (1 = use all pixels).",
)
def calculate_percentiles(
    workspace: Path,
    rois: tuple[str, ...],
    idx: int | None,
    codebook_opt: str | None,
    percentiles: tuple[float, float],
    grid: int,
    max_streams: int,
    channel_chunk: int,
    threads: int,
    out_suffix: str,
    overwrite: bool,
    sample_step: int,
) -> None:
    """Compute subtile illumination percentiles (GPU preferred)."""

    console = rich.get_console()
    ws = Workspace(workspace)
    resolved_rois = _normalize_rois(ws, rois, None)

    if not resolved_rois:
        raise click.UsageError("No ROIs found to process.")

    if codebook_opt is not None:
        opt_path = Path(codebook_opt)
        cb_global = opt_path.stem if (opt_path.suffix and not opt_path.is_dir()) else codebook_opt
    else:
        cb_global = None

    p_lo, p_hi = percentiles
    key_lo = format(p_lo, ".10g")
    key_hi = format(p_hi, ".10g")
    if sample_step <= 0:
        raise click.UsageError("--sample-step must be a positive integer")

    use_gpu = os.environ.get("GPU", "0") == "1" and cp is not None

    for roi_name in resolved_rois:
        if cb_global is None:
            candidates = ws.registered_codebooks(rois=[roi_name])
            if not candidates:
                raise click.UsageError(
                    f"No registered codebooks found for ROI '{roi_name}'. Provide --codebook."
                )
            if len(candidates) > 1:
                raise click.UsageError(
                    f"Multiple codebooks present for ROI '{roi_name}': {', '.join(candidates)}. Specify --codebook."
                )
            cb = candidates[0]
        else:
            cb = cb_global

        file_map, missing = ws.registered_file_map(cb, rois=[roi_name])
        if missing:
            raise click.UsageError(f"ROI(s) missing registered outputs: {', '.join(missing)}")

        tiles_all = file_map.get(roi_name, [])
        if not tiles_all:
            raise click.UsageError(f"No registered tiles for roi='{roi_name}', codebook='{cb}'.")

        if idx is not None:
            target = ws.regimg(roi_name, cb, int(idx))
            if not target.exists():
                raise click.UsageError(f"Tile {target.name} not found under registered--{roi_name}+{cb}.")
            tiles = [target]
        else:
            tiles = list(tiles_all)

        if not overwrite:
            before = len(tiles)
            tiles = [t for t in tiles if not _skip_existing(t, out_suffix, grid, (key_lo, key_hi), False)]
            skipped = before - len(tiles)
        else:
            skipped = 0

        console.print(
            f"[bold]{roi_name}[/bold]: {len(tiles)} tile(s) queued "
            f"(codebook={cb}, backend={'GPU' if use_gpu else 'CPU'}, sample_step={sample_step})",
            style="green",
        )

        def _process(tile: Path) -> None:
            # Keep coordinates in native pixel units. Do NOT pre-downsample XY globally,
            # otherwise (x, y, x_size, y_size) written to JSON would be in decimated
            # coordinates and later inferred tile_w/tile_h would be too small.
            arr, channel_names = _load_czyx(tile, use_gpu)
            # Percentiles will be computed per-subtile below with optional sampling stride.
            assert (use_gpu and cp is not None) or (not use_gpu)
            _, _, H, W = arr.shape  # type: ignore[attr-defined]
            y_edges = _edges(int(H), grid)
            x_edges = _edges(int(W), grid)
            payload: dict[str, list[dict[str, object]]] = {}
            for ci, cname in enumerate(channel_names):
                entries: list[dict[str, object]] = []
                for r in range(grid):
                    y0, y1 = y_edges[r], y_edges[r + 1]
                    for col in range(grid):
                        x0, x1 = x_edges[col], x_edges[col + 1]
                        # Apply sampling stride only when computing percentiles; keep
                        # recorded coordinates in native pixel units.
                        if use_gpu:
                            view = arr[ci : ci + 1, :, y0:y1:sample_step, x0:x1:sample_step]
                            pct = cp.quantile(
                                view,
                                cp.asarray([percentiles[0] / 100.0, percentiles[1] / 100.0]),
                                axis=(1, 2, 3),
                            )  # type: ignore[name-defined]
                            lo_val = float(cp.asnumpy(pct[0]))
                            hi_val = float(cp.asnumpy(pct[1]))
                        else:
                            sub = arr[ci : ci + 1, :, y0:y1:sample_step, x0:x1:sample_step]
                            pct = np.percentile(sub, [percentiles[0], percentiles[1]], axis=(1, 2, 3))
                            lo_val = float(pct[0])
                            hi_val = float(pct[1])

                        entries.append({
                            "x": int(x0),
                            "y": int(y0),
                            "x_size": int(x1 - x0),
                            "y_size": int(y1 - y0),
                            "percentiles": {key_lo: lo_val, key_hi: hi_val},
                        })
                payload[cname] = entries
            tile.with_suffix(out_suffix).write_text(json.dumps(payload, indent=2))

        if tiles:
            if use_gpu:
                with progress_bar(len(tiles)) as advance:
                    for tile in tiles:
                        _process(tile)
                        advance()
            else:
                with progress_bar_threadpool(len(tiles), threads=max(1, int(threads))) as submit:
                    for tile in tiles:
                        submit(_process, tile)
        else:
            console.print(
                f"[yellow]{roi_name}: all tiles already up-to-date (use --overwrite to recompute).[/yellow]"
            )
            continue

        tail = f" (skipped {skipped} existing; use --overwrite to recompute)" if skipped else ""
        console.print(f"[cyan]{roi_name}: percentile JSONs written with suffix '{out_suffix}'.{tail}[/cyan]")


@correct_illum.command("field-generate")
@click.argument(
    "workspace",
    type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=Path),
)
@click.argument("roi", type=str)
@click.argument("codebook", type=str)
@click.option(
    "--channel",
    type=str,
    default=None,
    help="Channel to model (default: ALL channels)",
)
@click.option(
    "--kernel",
    type=click.Choice(["thin_plate_spline", "cubic", "linear", "multiquadric"]),
    default="thin_plate_spline",
    show_default=True,
)
@click.option("--smoothing", type=float, default=2.0, show_default=True)
@click.option(
    "--subtile-suffix",
    type=str,
    default=".subtiles.json",
    show_default=True,
    help="Suffix of JSON produced by calculate-percentiles",
)
@click.option(
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Optional output NPZ path",
)
def field_generate(
    workspace: Path,
    roi: str,
    codebook: str,
    channel: Optional[str],
    kernel: str,
    smoothing: float,
    subtile_suffix: str,
    output: Optional[Path],
) -> None:
    """Generate an illumination field NPZ from subtile percentiles.

    Default behavior now collects ALL channels into a single multi-channel NPZ.
    Use --channel to restrict to one channel.
    """

    ws = Workspace(workspace)
    fields_dir, codebook_label, codebook_slug = _ensure_fields_dir(ws, codebook)
    reg_dir = ws.registered(roi, codebook_label)

    # Load tile indices and origins
    indices, xs0, ys0 = _resolve_tile_origins({}, ws, roi)
    order = np.argsort(indices.astype(np.int64))
    indices = indices[order]
    xs0 = xs0[order]
    ys0 = ys0[order]
    origin_map = {int(indices[i]): (float(xs0[i]), float(ys0[i])) for i in range(len(indices))}

    # Infer channels and metadata from any existing subtile JSON
    channels, p_low_key, p_high_key, tile_w, tile_h, grid_step = (
        RangeFieldPointsModel._infer_subtile_metadata(
            reg_dir,
            indices,
            subtile_suffix=subtile_suffix,
        )
    )
    if not channels:
        raise click.UsageError(
            "No channels discovered in subtile JSON; run calculate-percentiles first or check subtile suffix."
        )

    # Helper to assemble a single-channel model via existing code path
    def _single_channel_model(ch_name: str) -> RangeFieldPointsModel:
        xs, ys, lows, highs, chosen_channel = RangeFieldPointsModel._gather_subtile_values(
            reg_dir,
            indices,
            origin_map,
            channel=ch_name,
            p_low_key=p_low_key,
            p_high_key=p_high_key,
            suffix=subtile_suffix,
        )
        meta_sc: dict[str, object] = {
            "roi": roi,
            "codebook": codebook_label,
            "channel": chosen_channel,
            "kernel": kernel,
            "smoothing": float(smoothing),
            "epsilon": None,
            "neighbors": 64,
            "tile_w": int(tile_w),
            "tile_h": int(tile_h),
            "p_low_key": p_low_key,
            "p_high_key": p_high_key,
            "grid_step_suggest": float(grid_step),
            "subtile_suffix": subtile_suffix,
            "channels": channels,
            "workspace": str(ws.path),
        }
        return RangeFieldPointsModel(
            xy=np.c_[xs, ys].astype(np.float32, copy=False),
            vlow=np.asarray(lows, dtype=np.float32),
            vhigh=np.asarray(highs, dtype=np.float32),
            meta=meta_sc,
        )

    if channel is not None:
        # Legacy behavior: single-channel NPZ
        model = _single_channel_model(channel)
        meta = model.meta
        p_low = meta["p_low_key"]
        p_high = meta["p_high_key"]
        channel_used = meta["channel"]
        range_mean = float(meta.get("range_mean", getattr(model, "range_mean", 1.0)))
        range_mean = max(range_mean, 1e-6)
        if output is None:
            descriptor = "--".join([
                _slugify_token(str(channel_used)),
                _slugify_token(str(p_low)),
                _slugify_token(str(p_high)),
            ])
            output = fields_dir / f"illum-field--{roi}+{codebook_slug}--{descriptor}.npz"
        out_path = model.to_npz(output)
        rich.print(
            f"[green]Saved illumination field model → {out_path} (range mean {range_mean:.6g})[/green]"
        )
        return

    # Default: multi-channel NPZ across all discovered channels
    xs_ref: np.ndarray | None = None
    ys_ref: np.ndarray | None = None
    lows_list: list[np.ndarray] = []
    highs_list: list[np.ndarray] = []
    for ch in channels:
        xs, ys, lows, highs, _ = RangeFieldPointsModel._gather_subtile_values(
            reg_dir,
            indices,
            origin_map,
            channel=ch,
            p_low_key=p_low_key,
            p_high_key=p_high_key,
            suffix=subtile_suffix,
        )
        if xs_ref is None:
            xs_ref = xs
            ys_ref = ys
        else:
            if not (np.allclose(xs_ref, xs) and np.allclose(ys_ref, ys)):
                raise RuntimeError(
                    "Subtile centers differ across channels; cannot assemble multi-channel model."
                )
        lows_list.append(np.asarray(lows, dtype=np.float32))
        highs_list.append(np.asarray(highs, dtype=np.float32))

    assert xs_ref is not None and ys_ref is not None
    vlow_mc = np.stack(lows_list, axis=1)  # (N, C)
    vhigh_mc = np.stack(highs_list, axis=1)  # (N, C)
    meta_mc: dict[str, object] = {
        "roi": roi,
        "codebook": codebook_label,
        "kernel": kernel,
        "smoothing": float(smoothing),
        "epsilon": None,
        "neighbors": 64,
        "tile_w": int(tile_w),
        "tile_h": int(tile_h),
        "p_low_key": p_low_key,
        "p_high_key": p_high_key,
        "grid_step_suggest": float(grid_step),
        "subtile_suffix": subtile_suffix,
        "channels": channels,
        "workspace": str(ws.path),
    }
    model_mc = RangeFieldPointsModel(
        xy=np.c_[xs_ref, ys_ref].astype(np.float32, copy=False),
        vlow=vlow_mc,
        vhigh=vhigh_mc,
        meta=meta_mc,
    )
    meta = model_mc.meta
    p_low = meta["p_low_key"]
    p_high = meta["p_high_key"]
    # range_mean is per-model; we keep it for legacy readers but export code recomputes per-channel
    range_mean = float(meta.get("range_mean", getattr(model_mc, "range_mean", 1.0)))
    range_mean = max(range_mean, 1e-6)
    if output is None:
        descriptor = "--".join([
            "ALL",
            _slugify_token(str(p_low)),
            _slugify_token(str(p_high)),
        ])
        output = fields_dir / f"illum-field--{roi}+{codebook_slug}--{descriptor}.npz"
    out_path = model_mc.to_npz(output)
    rich.print(
        f"[green]Saved multi-channel illumination field model → {out_path} (channels={len(channels)}, range mean {range_mean:.6g})[/green]"
    )


__all__ = ["correct_illum"]


@correct_illum.command("plot-field")
@click.argument("model", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--workspace",
    type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=Path),
)
@click.option("--roi", type=str, required=False)
@click.option("--codebook", type=str, required=False)
@click.option("--grid-step", type=float, default=None, help="Override grid step used for plotting")
@click.option(
    "--output",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    default=None,
    help="Optional output image path",
)
def plot_field(
    model: Path,
    workspace: Path | None,
    roi: str | None,
    codebook: str | None,
    grid_step: float | None,
    output: Path | None,
) -> None:
    """Plot LOW field and RANGE correction for ALL channels in an illumination NPZ."""

    import matplotlib.pyplot as plt
    from matplotlib import colors

    rng_model = RangeFieldPointsModel.from_npz(model)
    meta = rng_model.meta

    roi = roi or meta.get("roi")
    codebook = codebook or meta.get("codebook")
    if workspace is None:
        workspace = Path(meta.get("workspace", "."))

    ws = Workspace(workspace)
    # Ensure workspace path is present for metadata; direct reg_dir is not needed here
    meta.setdefault("workspace", str(ws.path))

    _indices, xs0, ys0 = _resolve_tile_origins(meta, ws, roi)

    tile_w = float(meta.get("tile_w", 1024))
    tile_h = float(meta.get("tile_h", 1024))

    xy_points = rng_model.xy.astype(float, copy=False)
    x_min_pts = float(xy_points[:, 0].min()) if xy_points.size else 0.0
    x_max_pts = float(xy_points[:, 0].max()) if xy_points.size else tile_w
    y_min_pts = float(xy_points[:, 1].min()) if xy_points.size else 0.0
    y_max_pts = float(xy_points[:, 1].max()) if xy_points.size else tile_h

    if xs0.size:
        x0 = float(xs0.min())
        x1 = float(xs0.max() + tile_w)
    else:
        x0 = x_min_pts - tile_w * 0.5
        x1 = x_max_pts + tile_w * 0.5

    if ys0.size:
        y0 = float(ys0.min())
        y1 = float(ys0.max() + tile_h)
    else:
        y0 = y_min_pts - tile_h * 0.5
        y1 = y_max_pts + tile_h * 0.5

    vlow = np.asarray(rng_model.vlow)
    vhigh = np.asarray(rng_model.vhigh)
    if vlow.ndim == 1:
        vlow = vlow[:, None]
        vhigh = vhigh[:, None]
    n_channels = int(vlow.shape[1])
    ch_names = list(meta.get("channels", [])) if isinstance(meta.get("channels"), list) else []
    if not ch_names or len(ch_names) != n_channels:
        ch_names = [f"ch{i}" for i in range(n_channels)]

    # Build figure: 2 columns (LOW, RANGE) × n_channels rows
    height_per_row = 2.5
    fig = plt.figure(figsize=(10, max(2, int(n_channels * height_per_row))), dpi=140)
    fig.suptitle(f"Illum Field — ROI={roi} CB={codebook} (ALL channels)")
    div_norm = colors.TwoSlopeNorm(vmin=0.5, vcenter=1.0, vmax=2.0)

    # Precompute mask once at evaluation grid per channel (grids identical across channels)
    mask_ref = None
    xs_ref = None
    ys_ref = None

    for ci in range(n_channels):
        sub_meta = dict(meta)
        sub_meta.pop("range_mean", None)  # force per-channel normalization
        ch_label = ch_names[ci]
        mdl_ci = RangeFieldPointsModel(
            xy=rng_model.xy, vlow=vlow[:, ci], vhigh=vhigh[:, ci], meta=dict(sub_meta, channel=ch_label)
        )

        xs, ys, low_field, high_field = mdl_ci.evaluate(
            x0,
            y0,
            x1,
            y1,
            grid_step=grid_step,
        )

        if mask_ref is None:
            if xs0.size:
                mask_ref = _mask_union_of_tiles(xs, ys, xs0, ys0, tile_w, tile_h)
            else:
                mask_ref = np.ones_like(low_field, dtype=bool)
            xs_ref, ys_ref = xs, ys
        else:
            # Grids should match across channels
            if xs_ref is None or ys_ref is None or xs.shape != xs_ref.shape or ys.shape != ys_ref.shape:
                raise RuntimeError(
                    "Evaluated grids differ across channels; cannot plot ALL channels together."
                )

        low_masked = np.ma.array(low_field, mask=~mask_ref)
        inv_range = mdl_ci.range_correction(low_field, high_field, mask=mask_ref)
        range_masked = np.ma.array(inv_range, mask=~mask_ref)

        ax_low = fig.add_subplot(n_channels, 2, ci * 2 + 1)
        im_low = ax_low.imshow(low_masked, origin="upper", extent=[x0, x1, y0, y1])
        ax_low.set_aspect("equal")
        ax_low.set_title(f"{ch_label} — LOW")
        fig.colorbar(im_low, ax=ax_low, fraction=0.046)

        ax_rng = fig.add_subplot(n_channels, 2, ci * 2 + 2)
        im_rng = ax_rng.imshow(
            range_masked, origin="upper", extent=[x0, x1, y0, y1], cmap="coolwarm", norm=div_norm
        )
        ax_rng.set_aspect("equal")
        ax_rng.set_title(f"{ch_label} — RANGE (normalized)")
        fig.colorbar(im_rng, ax=ax_rng, fraction=0.046)

    fig.tight_layout()
    if output is None:
        output = model.with_suffix(".png")
    fig.savefig(output.as_posix(), dpi=200, bbox_inches="tight")
    plt.close(fig)
    rich.print(f"[green]Saved illumination plot → {output}[/green]")


@correct_illum.command("plot-field-zarr")
@click.argument("zarr_path", type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=Path))
@click.option(
    "--channels",
    type=str,
    default=None,
    help="Comma-separated channel names or indices to include (default: all)",
)
@click.option(
    "--output",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    default=None,
    help="Optional output image path (PNG)",
)
def plot_field_zarr(zarr_path: Path, channels: str | None, output: Path | None) -> None:
    """Plot LOW and RANGE directly from a TCYX Zarr store (no re-evaluation)."""
    import matplotlib.pyplot as plt
    import zarr
    from matplotlib import colors

    za = zarr.open_array(str(zarr_path), mode="r")
    if za.ndim != 4:
        raise click.UsageError("Expected a TCYX Zarr array (4D).")
    attrs = getattr(za, "attrs", {})
    axes = attrs.get("axes")
    if axes not in (None, "TCYX"):
        raise click.UsageError(f"Unexpected axes='{axes}'; expected 'TCYX'.")
    t_labels = attrs.get("t_labels") or ["low", "range"]
    try:
        t_low = int(t_labels.index("low"))
        t_range = int(t_labels.index("range"))
    except Exception:
        t_low, t_range = 0, 1
    ch_names = attrs.get("channel_names") or []
    C = int(za.shape[1])
    if not ch_names or len(ch_names) != C:
        ch_names = [f"ch{i}" for i in range(C)]

    # Resolve channel selection
    if channels:
        requested = [c.strip() for c in channels.split(",") if c.strip()]
        idx: list[int] = []
        for tok in requested:
            if tok.isdigit():
                i = int(tok)
                if not (0 <= i < C):
                    raise click.UsageError(f"Channel index out of range: {i}")
                idx.append(i)
            else:
                if tok not in ch_names:
                    raise click.UsageError(f"Unknown channel name: {tok}")
                idx.append(ch_names.index(tok))
        sel = idx
    else:
        sel = list(range(C))

    # Use model_meta for extent if present
    md = attrs.get("model_meta") or {}
    try:
        x0 = float(md.get("x0", 0.0))
        y0 = float(md.get("y0", 0.0))
        Wn = float(md.get("width_native", za.shape[3] * int(md.get("downsample", 1) or 1)))
        Hn = float(md.get("height_native", za.shape[2] * int(md.get("downsample", 1) or 1)))
        extent = [x0, x0 + Wn, y0, y0 + Hn]
    except Exception:
        extent = None

    # Figure: 2 columns (LOW, RANGE) × rows=len(sel); use nearest interpolation (no smoothing)
    height_per_row = 2.5
    fig = plt.figure(figsize=(10, max(2, int(len(sel) * height_per_row))), dpi=140)
    fig.suptitle(f"Field (Zarr) — {zarr_path.name}")
    div_norm = colors.TwoSlopeNorm(vmin=0.5, vcenter=1.0, vmax=2.0)

    for r, ci in enumerate(sel):
        low = za[t_low, ci, :, :]
        rng = za[t_range, ci, :, :]
        label = ch_names[ci]

        ax_low = fig.add_subplot(len(sel), 2, r * 2 + 1)
        im_low = ax_low.imshow(low, origin="upper", extent=extent, interpolation="nearest")
        ax_low.set_aspect("equal")
        ax_low.set_title(f"{label} — LOW (stored)")
        fig.colorbar(im_low, ax=ax_low, fraction=0.046)

        ax_rng = fig.add_subplot(len(sel), 2, r * 2 + 2)
        im_rng = ax_rng.imshow(
            rng, origin="upper", extent=extent, interpolation="nearest", cmap="coolwarm", norm=div_norm
        )
        ax_rng.set_aspect("equal")
        ax_rng.set_title(f"{label} — RANGE (stored)")
        fig.colorbar(im_rng, ax=ax_rng, fraction=0.046)

    fig.tight_layout()
    if output is None:
        output = zarr_path.with_suffix(".png")
    fig.savefig(output.as_posix(), dpi=200, bbox_inches="tight")
    plt.close(fig)
    rich.print(f"[green]Saved Zarr field plot → {output}[/green]")


@correct_illum.command("export-field")
@click.argument(
    "workspace",
    type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=Path),
)
@click.argument("rois", nargs=-1, type=str)
@click.option(
    "--model",
    "model_path",
    type=click.Path(exists=True, dir_okay=False, file_okay=True, path_type=Path),
    required=False,
    help="Optional illumination NPZ. When omitted, discover per-ROI models generated by field-generate.",
)
@click.option(
    "--codebook",
    type=str,
    required=False,
    help="Codebook label (falls back to NPZ metadata when omitted).",
)
@click.option(
    "--downsample",
    type=int,
    default=4,
    show_default=True,
    help="Downsample factor for output (1 = native)",
)
@click.option(
    "--grid-step",
    type=float,
    default=None,
    help="Override grid step used for evaluation",
)
@click.option(
    "--what",
    type=click.Choice(["range", "low", "both"], case_sensitive=False),
    default="both",
    show_default=True,
    help="Which field(s) to export: normalized RANGE, LOW, or BOTH.",
)
@click.option(
    "--neighbors",
    type=int,
    default=None,
    help="Neighbors for RBF evaluation (default: model metadata)",
)
@click.option(
    "--smoothing",
    type=float,
    default=None,
    help="L2 smoothing for RBF (default: model metadata)",
)
@click.option(
    "--output",
    type=click.Path(dir_okay=True, file_okay=True, writable=True, path_type=Path),
    default=None,
    help=(
        "Explicit Zarr output path. When omitted, writes to "
        "'analysis/deconv/fields+{sanitize(codebook)}/field--{roi}+{sanitize(codebook)}.zarr'."
    ),
)
@click.option(
    "--overwrite/--no-overwrite",
    default=False,
    show_default=True,
    help="When true, discard any existing export and recompute all channels.",
)
def export_field(
    workspace: Path,
    rois: tuple[str, ...],
    model_path: Path | None,
    codebook: str | None,
    downsample: int,
    grid_step: float | None,
    what: str,
    neighbors: int | None,
    smoothing: float | None,
    output: Path | None,
    overwrite: bool,
) -> None:
    """Export LOW and RANGE fields to TCYX Zarr stores per ROI."""

    import zarr
    from skimage.transform import resize

    def _update_progress(
        za_obj: zarr.Array | None,
        status: str,
        completed: Sequence[str],
        total: int | None,
    ) -> None:
        """Persist incremental export progress metadata to the Zarr store."""

        if za_obj is None or total is None:
            return
        payload = {
            "completed_channels": list(completed),
            "total_channels": int(total),
        }
        try:
            za_obj.attrs.update({"status": status, "progress": payload})
        except Exception:
            logger.opt(exception=True).warning(
                f"Failed to update export-field progress metadata (status={status})"
            )

    if downsample <= 0:
        raise click.UsageError("--downsample must be a positive integer")

    ws = Workspace(workspace)

    if output is not None and rois and len(rois) > 1:
        raise click.UsageError("--output can only be provided when exporting a single ROI.")

    fields_dir: Path | None = None
    codebook_label: str
    codebook_slug: str
    roi_to_model: dict[str, Path] = {}
    model_cache: dict[Path, RangeFieldPointsModel] = {}

    if model_path is not None:
        model_path = Path(model_path)
        rng_single = RangeFieldPointsModel.from_npz(model_path)
        meta_single = dict(rng_single.meta)
        roi_single = meta_single.get("roi")
        if roi_single is None:
            raise click.UsageError("Model metadata is missing 'roi'; specify ROI explicitly.")
        resolved_rois = _normalize_rois(ws, rois, str(roi_single))
        if not resolved_rois:
            resolved_rois = [str(roi_single)]
        if len(resolved_rois) != 1 or str(resolved_rois[0]) != str(roi_single):
            raise click.UsageError(
                "Single model NPZ can only be exported for its own ROI. "
                f"Model ROI='{roi_single}', requested={resolved_rois}"
            )
        codebook_label_raw = codebook or meta_single.get("codebook")
        if not codebook_label_raw:
            raise click.UsageError(
                "Codebook must be provided either via --codebook or present in model metadata"
            )
        fields_dir, codebook_label, codebook_slug = _ensure_fields_dir(ws, str(codebook_label_raw))
        roi_to_model[str(roi_single)] = model_path
        model_cache[model_path] = rng_single
    else:
        if not codebook:
            raise click.UsageError("Either provide --model or specify --codebook to auto-discover models.")
        fields_dir, codebook_label, codebook_slug = _ensure_fields_dir(ws, codebook)
        candidates = sorted(fields_dir.glob(f"illum-field--*+{codebook_slug}--*.npz"))
        if not candidates:
            raise click.UsageError(
                f"No illumination field models found in {fields_dir} for codebook '{codebook}'."
            )
        models_map: dict[str, Path] = {}
        for candidate in candidates:
            suffix = candidate.name[len("illum-field--") :]
            roi_part, sep, _ = suffix.partition(f"+{codebook_slug}--")
            if not sep or not roi_part:
                continue
            models_map[roi_part] = candidate
        if rois:
            resolved_rois = _normalize_rois(ws, rois, None)
        else:
            resolved_rois = sorted(models_map.keys())
        if not resolved_rois:
            raise click.UsageError("No ROIs matched available illumination field models.")
        for roi_name in resolved_rois:
            try:
                roi_to_model[roi_name] = models_map[roi_name]
            except KeyError as exc:  # coverage safety
                raise click.UsageError(
                    f"Illumination field NPZ not found for ROI '{roi_name}' in {fields_dir}."
                ) from exc

    if output is not None and len(roi_to_model) > 1:
        raise click.UsageError("--output can only be provided when exporting a single ROI.")

    if not roi_to_model:
        raise click.UsageError("No ROI/model pairs resolved for export.")

    assert fields_dir is not None

    for roi, model_path_roi in sorted(roi_to_model.items()):
        log_file_tag = f"{roi}+{codebook_label}"
        try:
            setup_workspace_logging(
                ws.path,
                component="preprocess.correct-illum.export-field",
                file=log_file_tag,
                debug=True,
            )
        except Exception:
            logger.opt(exception=True).debug("Workspace logging setup failed; continuing with default logger")

        logger.info(
            "ExportField: model='{}', ROI='{}', workspace='{}', codebook='{}', what='{}', ds={}",
            model_path_roi.name,
            roi,
            ws.path,
            codebook_label,
            what,
            downsample,
        )

        default_output_path = fields_dir / f"field--{roi}+{codebook_slug}.zarr"
        output_path = output if output is not None else default_output_path

        za_store: zarr.Array | None = None
        progress_total: int | None = None
        completed_channels: list[str] = []
        completed_set: set[str] = set()
        existing_shape: tuple[int, int, int, int] | None = None
        existing_attrs: dict[str, object] = {}
        existing_model_meta: dict[str, object] | None = None
        arr_shape: tuple[int, int, int, int] | None = None
        chunks: tuple[int, int, int, int] | None = None
        t_low_idx = 0
        t_range_idx = 1

        rng_model = model_cache.get(model_path_roi)
        if rng_model is None:
            rng_model = RangeFieldPointsModel.from_npz(model_path_roi)
            model_cache[model_path_roi] = rng_model
        meta = dict(rng_model.meta)
        meta.setdefault("roi", roi)
        meta.setdefault("codebook", codebook_label)
        meta.setdefault("workspace", str(ws.path))
        if str(meta.get("roi")) != roi:
            logger.warning(
                "Model ROI metadata '{}' differs from requested ROI '{}'; overriding metadata.",
                meta.get("roi"),
                roi,
            )
            meta["roi"] = roi

        indices, xs0, ys0 = _resolve_tile_origins(meta, ws, roi)
        if indices.size == 0:
            raise click.UsageError(f"No TileConfiguration entries found for ROI '{roi}'")

        order = np.argsort(indices.astype(np.int64))
        indices = indices[order]
        xs0 = xs0[order]
        ys0 = ys0[order]

        tile_w = float(meta.get("tile_w", 1968))
        tile_h = float(meta.get("tile_h", 1968))

        x0 = float(xs0.min())
        x1 = float(xs0.max() + tile_w)
        y0 = float(ys0.min())
        y1 = float(ys0.max() + tile_h)

        grid_step_eff = grid_step
        if grid_step_eff is None:
            grid_step_eff = float(meta.get("grid_step_suggest", meta.get("grid_step", 128.0)))
        neighbors_eff = neighbors
        if neighbors_eff is None:
            neighbors_eff = int(meta.get("neighbors", 0) or 0) or None
        smoothing_eff = smoothing
        if smoothing_eff is None:
            smoothing_eff = float(meta.get("smoothing", 1.0))

        vlow = np.asarray(rng_model.vlow)
        vhigh = np.asarray(rng_model.vhigh)
        if vlow.ndim == 1:
            vlow = vlow[:, None]
        if vhigh.ndim == 1:
            vhigh = vhigh[:, None]

        n_channels = int(vlow.shape[1])
        ch_names = list(meta.get("channels", [])) if isinstance(meta.get("channels"), list) else []
        if not ch_names or len(ch_names) != n_channels:
            ch_names = (
                [str(meta.get("channel", "channel_0"))]
                if n_channels == 1
                else [f"ch{i}" for i in range(n_channels)]
            )
        logger.info("Detected {} channel(s) in field NPZ: {}", n_channels, ",".join(ch_names))

        store_exists = output_path.exists()
        if store_exists and overwrite:
            logger.info("Overwrite requested; removing existing field store at {}", output_path)
            try:
                if output_path.is_dir():
                    shutil.rmtree(output_path)
                else:
                    output_path.unlink()
            except Exception as exc:
                raise click.ClickException(
                    f"Failed to remove existing field store at {output_path}: {exc}"
                ) from exc
            store_exists = False
            completed_channels.clear()
            completed_set.clear()
            progress_total = None
            existing_shape = None
            existing_attrs = {}
            existing_model_meta = None
        if store_exists:
            logger.info("Existing field store detected at {}; opening in r+ mode", output_path)
            za_store = zarr.open(str(output_path), mode="r+")
            existing_shape = tuple(int(dim) for dim in za_store.shape)
            if len(existing_shape) != 4 or existing_shape[0] != 2 or existing_shape[1] != n_channels:
                raise click.UsageError(
                    f"Existing field store at {output_path} has incompatible shape {existing_shape}; "
                    "delete it or export to a different path."
                )
            if za_store.dtype != np.float32:
                raise click.UsageError(
                    f"Existing field store at {output_path} has dtype {za_store.dtype}; expected float32."
                )
            existing_attrs = dict(getattr(za_store, "attrs", {}))
            axes_attr = existing_attrs.get("axes")
            if axes_attr is None:
                logger.warning(
                    "Existing field store at {} missing 'axes' metadata; setting to 'TCYX' for compatibility.",
                    output_path,
                )
                try:
                    za_store.attrs["axes"] = "TCYX"
                    existing_attrs["axes"] = "TCYX"
                except Exception as exc:
                    raise click.UsageError(
                        f"Failed to set missing 'axes' metadata on existing field store at {output_path}: {exc}"
                    ) from exc
            elif axes_attr != "TCYX":
                raise click.UsageError(
                    f"Existing field store at {output_path} has axes={axes_attr!r}; expected 'TCYX'."
                )
            t_labels_raw = existing_attrs.get("t_labels")
            if isinstance(t_labels_raw, (list, tuple)):
                t_labels_attr = [str(label) for label in t_labels_raw]
            else:
                t_labels_attr = []
            if "low" not in t_labels_attr or "range" not in t_labels_attr:
                logger.warning(
                    "Existing field store at {} missing required T labels 'low'/'range'; setting default ['low','range'].",
                    output_path,
                )
                try:
                    t_labels_attr = ["low", "range"]
                    za_store.attrs["t_labels"] = t_labels_attr
                    existing_attrs["t_labels"] = t_labels_attr
                except Exception as exc:
                    raise click.UsageError(
                        f"Failed to set missing T labels on existing field store at {output_path}: {exc}"
                    ) from exc
            channel_names_raw = existing_attrs.get("channel_names")
            if isinstance(channel_names_raw, (list, tuple)):
                channel_names_attr = [str(name) for name in channel_names_raw]
            else:
                channel_names_attr = []
            if not channel_names_attr:
                logger.warning(
                    "Existing field store at {} missing channel_names metadata; setting to {}",
                    output_path,
                    ch_names,
                )
                try:
                    za_store.attrs["channel_names"] = ch_names
                    existing_attrs["channel_names"] = ch_names
                    channel_names_attr = ch_names
                except Exception as exc:
                    raise click.UsageError(
                        f"Failed to set missing channel_names on existing field store at {output_path}: {exc}"
                    ) from exc
            if channel_names_attr != ch_names:
                raise click.UsageError(
                    "Existing field store channel list does not match model channels: "
                    f"{channel_names_attr} vs {ch_names}. Remove the store or rerun with --output."
                )
            existing_model_meta_obj = existing_attrs.get("model_meta")
            if not isinstance(existing_model_meta_obj, dict):
                raise click.UsageError(
                    f"Existing field store at {output_path} is missing model_meta metadata."
                )
            existing_model_meta = dict(existing_model_meta_obj)
            expected_meta_subset = {
                "roi": roi,
                "codebook": codebook_label,
                "tile_w": int(tile_w),
                "tile_h": int(tile_h),
                "downsample": int(downsample),
                "grid_step": float(grid_step_eff) if grid_step_eff is not None else None,
                "neighbors": int(neighbors_eff or 0),
                "smoothing": float(smoothing_eff) if smoothing_eff is not None else None,
                "cropped": True,
            }
            mismatched_keys = [
                key
                for key, expected_value in expected_meta_subset.items()
                if expected_value is not None and existing_model_meta.get(key) != expected_value
            ]
            if mismatched_keys:
                raise click.UsageError(
                    "Existing field store metadata does not match current export parameters "
                    f"(mismatched keys: {', '.join(mismatched_keys)}). "
                    "Remove the store or export to a new path."
                )
            progress_attr = existing_attrs.get("progress")
            if isinstance(progress_attr, dict):
                completed_channels = [
                    str(name) for name in progress_attr.get("completed_channels", []) if str(name) in ch_names
                ]
                total_progress = progress_attr.get("total_channels")
                if isinstance(total_progress, int):
                    progress_total = total_progress
            completed_set = set(completed_channels)
            if completed_set:
                completed_channels = [name for name in ch_names if name in completed_set]
            logger.info(
                "Resume detected: {} of {} channel(s) already exported in {}",
                len(completed_channels),
                n_channels,
                output_path,
            )
            if progress_total is None:
                progress_total = n_channels
            if len(completed_set) == n_channels:
                ordered_completed = [name for name in ch_names if name in completed_set]
                if len(ordered_completed) != n_channels:
                    ordered_completed = ch_names
                _update_progress(za_store, "complete", ordered_completed, progress_total)
                logger.info(
                    "All {} channel(s) already present in {}; skipping export-field recomputation.",
                    n_channels,
                    output_path,
                )
                rich.print(
                    f"[yellow]Field store already complete for ROI={roi} codebook={codebook_label}; "
                    f"skipping export → {output_path}[/yellow]"
                )
                continue

        # Cached grid/mask/crop computed once (identical across channels)
        xs: np.ndarray | None = None
        ys: np.ndarray | None = None
        mask: np.ndarray | None = None
        j0 = j1 = i0 = i1 = 0
        x0_crop = x1_crop = y0_crop = y1_crop = 0.0
        H = W = Hds = Wds = 0

        # Optional GPU path for resampling (set GPU=1 to enable)
        use_gpu_resize = os.environ.get("GPU", "0") == "1"
        logger.info(
            "ExportField backend: resizing on {} (GPU env GPU={} detected)",
            "GPU" if use_gpu_resize else "CPU",
            os.environ.get("GPU", "0"),
        )

        def _zoom_batch_to_output(batch: list[np.ndarray], oh: int, ow: int) -> list[np.ndarray]:
            """Resize a batch of coarse fields to the output resolution, preferring GPU."""

            if not batch:
                return []
            if not use_gpu_resize:
                logger.debug(f"zoom_batch: CPU resize for {len(batch)} field(s) to {ow}x{oh}")
                return [
                    resize(arr, (oh, ow), preserve_range=True, anti_aliasing=True).astype(np.float32)
                    for arr in batch
                ]
            try:
                from cucim.skimage.transform import rescale as _cucim_rescale  # type: ignore

                result: list[np.ndarray] = []
                last_scale_y = last_scale_x = 0.0
                for arr in batch:
                    arr_gpu = cp.asarray(arr, dtype=cp.float32)
                    scale_y = float(oh) / float(arr_gpu.shape[-2])
                    scale_x = float(ow) / float(arr_gpu.shape[-1])
                    zoomed_gpu = _cucim_rescale(
                        arr_gpu,
                        scale=(scale_y, scale_x),
                        order=2,
                        preserve_range=True,
                        anti_aliasing=False,
                    ).astype(cp.float32, copy=False)
                    if zoomed_gpu.shape != (oh, ow):
                        zoomed_gpu = zoomed_gpu[:oh, :ow]
                    result.append(cp.asnumpy(zoomed_gpu))
                    last_scale_y, last_scale_x = scale_y, scale_x
                logger.debug(
                    f"zoom_batch: cuCIM resize for {len(batch)} field(s) to {ow}x{oh} "
                    f"(scale_y={last_scale_y:.4f}, scale_x={last_scale_x:.4f})"
                )
                return result
            except Exception as exc:
                logger.opt(exception=exc).error(
                    f"zoom_batch: GPU resize failed while processing {len(batch)} field(s) to {ow}x{oh}"
                )
                raise
            finally:
                try:
                    cp.get_default_memory_pool().free_all_blocks()  # type: ignore[attr-defined]
                except Exception as pool_exc:
                    logger.opt(exception=pool_exc).debug("Failed to clear CuPy memory pool after resize")

        # Preallocate CPU-side outputs once dimensions are known; fill in channel batches of 2
        out_range: np.ndarray | None = None
        out_low: np.ndarray | None = None
        md_common: dict[str, object] | None = None
        m_bool: np.ndarray | None = None

        for ci in range(n_channels):
            ch_name = ch_names[ci]
            if ch_name in completed_set:
                logger.info(
                    "Skipping channel '{}' (already present in {})",
                    ch_name,
                    output_path,
                )
                continue

            vlo_ci = vlow[:, ci] if vlow.ndim == 2 else vlow
            vhi_ci = vhigh[:, ci] if vhigh.ndim == 2 else vhigh
            meta_ci = dict(meta)
            meta_ci.pop("range_mean", None)
            meta_ci["channel"] = ch_name
            mdl_ci = RangeFieldPointsModel(xy=rng_model.xy, vlow=vlo_ci, vhigh=vhi_ci, meta=meta_ci)

            xs_ci, ys_ci, low_field, high_field = mdl_ci.evaluate(
                x0,
                y0,
                x1,
                y1,
                grid_step=grid_step_eff,
                neighbors=neighbors_eff,
                smoothing=smoothing_eff,
            )

            # Expand the evaluated grid by one cell on all sides using nearest-neighbor padding
            # so downstream cropping/resizing has a stable exterior surface to sample.
            def _expand_grid_with_edge(
                xs_in: np.ndarray, ys_in: np.ndarray, low_in: np.ndarray, high_in: np.ndarray
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                xs_in = np.asarray(xs_in, dtype=np.float32)
                ys_in = np.asarray(ys_in, dtype=np.float32)
                if xs_in.size < 1 or ys_in.size < 1:
                    return xs_in, ys_in, low_in, high_in
                dx = float(xs_in[1] - xs_in[0]) if xs_in.size > 1 else 1.0
                dy = float(ys_in[1] - ys_in[0]) if ys_in.size > 1 else 1.0
                xs_out = np.empty(xs_in.size + 2, dtype=np.float32)
                ys_out = np.empty(ys_in.size + 2, dtype=np.float32)
                xs_out[0] = float(xs_in[0]) - dx
                xs_out[-1] = float(xs_in[-1]) + dx
                xs_out[1:-1] = xs_in
                ys_out[0] = float(ys_in[0]) - dy
                ys_out[-1] = float(ys_in[-1]) + dy
                ys_out[1:-1] = ys_in
                low_pad = np.pad(low_in, ((1, 1), (1, 1)), mode="edge").astype(np.float32, copy=False)
                high_pad = np.pad(high_in, ((1, 1), (1, 1)), mode="edge").astype(np.float32, copy=False)
                return xs_out, ys_out, low_pad, high_pad

            xs_ci, ys_ci, low_field, high_field = _expand_grid_with_edge(xs_ci, ys_ci, low_field, high_field)

            if xs is None or ys is None:
                xs, ys = xs_ci, ys_ci
                logger.info(
                    "Evaluated field grid: nx={}, ny={}, bbox=({:.1f},{:.1f})-({:.1f},{:.1f})",
                    int(xs.size),
                    int(ys.size),
                    x0,
                    y0,
                    x1,
                    y1,
                )
                mask = (
                    _mask_union_of_tiles(xs, ys, xs0, ys0, tile_w, tile_h)
                    if xs0.size
                    else np.ones((ys.size, xs.size), dtype=bool)
                )

                rows_any = mask.any(axis=1)
                cols_any = mask.any(axis=0)
                if rows_any.any() and cols_any.any():
                    j0_raw = int(np.argmax(rows_any))
                    j1_raw = int(len(rows_any) - 1 - np.argmax(rows_any[::-1]))
                    i0_raw = int(np.argmax(cols_any))
                    i1_raw = int(len(cols_any) - 1 - np.argmax(cols_any[::-1]))
                    # Clamp first to preserve ordering
                    j0_raw = max(0, min(j0_raw, j1_raw))
                    i0_raw = max(0, min(i0_raw, i1_raw))
                else:
                    j0_raw, j1_raw = 0, mask.shape[0] - 1
                    i0_raw, i1_raw = 0, mask.shape[1] - 1

                # After expanding the evaluated grid by one, also include one-cell exterior ring
                # in the cropped region to preserve that padding in the saved store.
                j0 = max(0, j0_raw - 1)
                i0 = max(0, i0_raw - 1)
                j1 = min(mask.shape[0] - 1, j1_raw + 1)
                i1 = min(mask.shape[1] - 1, i1_raw + 1)

                x0_crop = float(xs[i0])
                x1_crop = float(xs[i1])
                y0_crop = float(ys[j0])
                y1_crop = float(ys[j1])

                W = int(np.ceil(x1_crop - x0_crop))
                H = int(np.ceil(y1_crop - y0_crop))
                if W <= 0 or H <= 0:
                    raise click.UsageError(
                        "Computed empty cropped extent; check tile configuration or model metadata"
                    )
                Hds = max(1, int(round(H / float(downsample))))
                Wds = max(1, int(round(W / float(downsample))))
                logger.info(
                    "Cropped native size: {}x{} (W×H); downsampled to {}x{}",
                    int(W),
                    int(H),
                    int(Wds),
                    int(Hds),
                )
                mask_slice = mask[j0 : j1 + 1, i0 : i1 + 1]
                logger.debug(
                    (
                        f"Mask crop indices (i0={i0} i1={i1} j0={j0} j1={j1}); "
                        f"mask true ratio={float(mask_slice.mean()) if mask_slice.size else 0.0:.3f}"
                    )
                )

                # Preallocate outputs and precompute downsampled mask once
                m_sub_ref = mask_slice
                m_ds = resize(
                    m_sub_ref.astype(float),
                    (Hds, Wds),
                    preserve_range=True,
                    anti_aliasing=False,
                ).astype(np.float32)
                m_bool = m_ds > 0.5
                logger.debug(
                    f"Downsampled mask coverage: "
                    f"{float(m_bool.mean()) if m_bool.size else 0.0:.3f} (true pixels / total)"
                )
                out_range = np.empty((n_channels, Hds, Wds), dtype=np.float32)
                out_low = np.empty((n_channels, Hds, Wds), dtype=np.float32)
                mode = what.lower()
                if mode != "both":
                    logger.warning(
                        "--what={} requested; exporting both planes (low, range) and ignoring single-plane output.",
                        mode,
                    )
                if progress_total is None:
                    progress_total = n_channels
                arr_shape = (2, n_channels, Hds, Wds)
                chunk_y = min(Hds, 2048)
                chunk_x = min(Wds, 2048)
                chunks = (1, 1, chunk_y, chunk_x)
                md_common = {
                    "axes": "TCYX",
                    "roi": roi,
                    "codebook": codebook_label,
                    "tile_w": int(tile_w),
                    "tile_h": int(tile_h),
                    "grid_step": float(grid_step_eff) if grid_step_eff is not None else None,
                    "neighbors": int(neighbors_eff or 0),
                    "smoothing": float(smoothing_eff) if smoothing_eff is not None else None,
                    "downsample": int(downsample),
                    "x0": float(x0_crop),
                    "y0": float(y0_crop),
                    "width_native": int(W),
                    "height_native": int(H),
                    "cropped": True,
                }
                md_common = {k: v for k, v in md_common.items() if v is not None}
                md_common["channels"] = ch_names
                md_common["kind"] = "both"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                if za_store is None:
                    za_store = zarr.open(
                        str(output_path),
                        mode="w",
                        shape=arr_shape,
                        chunks=chunks,
                        dtype=np.float32,
                    )
                    za_store.attrs["axes"] = "TCYX"
                    za_store.attrs["t_labels"] = ["low", "range"]
                    za_store.attrs["channel_names"] = ch_names
                    za_store.attrs["model_meta"] = md_common
                    _update_progress(za_store, "initializing", completed_channels, progress_total)
                else:
                    if existing_shape is not None and existing_shape != arr_shape:
                        raise click.UsageError(
                            "Existing field store shape "
                            f"{existing_shape} does not match expected {arr_shape}; cannot resume export."
                        )
                    za_store.attrs["channel_names"] = ch_names
                    za_store.attrs["model_meta"] = md_common
            else:
                # Sanity check: grids must match across channels
                if xs_ci.shape != xs.shape or ys_ci.shape != ys.shape:
                    raise RuntimeError(
                        "Evaluated grids differ across channels; cannot assemble multi-channel export."
                    )

            if mask is None or out_low is None or out_range is None:
                raise RuntimeError("Export-field buffers were not initialized before channel processing.")

            inv_range = mdl_ci.range_correction(low_field, high_field, mask=mask)
            inv_sub = inv_range[j0 : j1 + 1, i0 : i1 + 1].astype(np.float32, copy=False)
            low_sub = low_field[j0 : j1 + 1, i0 : i1 + 1].astype(np.float32, copy=False)

            resize_backend = "GPU" if use_gpu_resize else "CPU"
            resize_start = time.perf_counter()
            inv_ds = _zoom_batch_to_output([inv_sub], Hds, Wds)[0]
            inv_resize_sec = time.perf_counter() - resize_start
            resize_start = time.perf_counter()
            low_ds = _zoom_batch_to_output([low_sub], Hds, Wds)[0]
            low_resize_sec = time.perf_counter() - resize_start

            if m_bool is not None and not m_bool.all():
                # Outside the union-of-tiles mask, use multiplicative identity for RANGE (1.0)
                # and additive identity for LOW (0.0) to match application semantics.
                inv_ds = np.where(m_bool, inv_ds, np.float32(1.0))
                low_ds = np.where(m_bool, low_ds, np.float32(0.0))

            out_range[ci, :, :] = inv_ds
            out_low[ci, :, :] = low_ds

            io_duration: float | None = None
            if za_store is not None:
                io_start = time.perf_counter()
                za_store[t_range_idx, ci, :, :] = inv_ds.astype(np.float32, copy=False)
                za_store[t_low_idx, ci, :, :] = low_ds.astype(np.float32, copy=False)
                io_duration = time.perf_counter() - io_start

            if ch_name not in completed_channels:
                completed_channels.append(ch_name)
            completed_set.add(ch_name)

            logger.debug(
                (
                    f"Channel '{ch_name}' zoom complete "
                    f"(min={float(inv_ds.min()):.3f} max={float(inv_ds.max()):.3f} "
                    f"mean={float(inv_ds.mean()):.3f})"
                )
            )

            if za_store is not None:
                _update_progress(za_store, "in_progress", completed_channels, progress_total)

            msg = (
                f"Channel {ci} ('{ch_name}') resize ({resize_backend}) "
                f"range={inv_resize_sec:.3f}s low={low_resize_sec:.3f}s"
            )
            if io_duration is not None:
                msg += f", zarr-write={io_duration:.3f}s"
            logger.info(msg)

        logger.info("Channel batch processing complete; finalizing export.")
        if out_low is None or out_range is None or md_common is None:
            raise RuntimeError("Export did not initialize output buffers; aborting without writing results.")

        arr_shape = (2, *out_low.shape)
        if za_store is None:
            chunk_y = min(arr_shape[2], 2048)
            chunk_x = min(arr_shape[3], 2048)
            chunks = (1, 1, chunk_y, chunk_x)
            logger.info(
                "Zarr store was not initialized during batching; writing full TCYX array once (shape={}, chunks={})",
                arr_shape,
                chunks,
            )
            zat_path = output_path
            zat_path.parent.mkdir(parents=True, exist_ok=True)
            io_start = time.perf_counter()
            za_store = zarr.open(
                str(zat_path),
                mode="w",
                shape=arr_shape,
                chunks=chunks,
                dtype=np.float32,
            )
            za_store.attrs["axes"] = "TCYX"
            za_store.attrs["t_labels"] = ["low", "range"]
            za_store.attrs["channel_names"] = ch_names
            md_common["kind"] = "both"
            za_store.attrs["model_meta"] = md_common
            za_store[t_low_idx, :, :, :] = out_low.astype(np.float32, copy=False)
            za_store[t_range_idx, :, :, :] = out_range.astype(np.float32, copy=False)
            full_write_sec = time.perf_counter() - io_start
            logger.info(
                "Full TCYX array write completed in %.3fs for %d channel(s)",
                full_write_sec,
                out_low.shape[0],
            )
            if progress_total is None:
                progress_total = n_channels
            _update_progress(za_store, "initializing", completed_channels, progress_total)
        else:
            md_common["kind"] = "both"
            za_store.attrs["model_meta"] = md_common

        if progress_total is None:
            progress_total = n_channels
        if len(completed_set) < n_channels:
            missing = [name for name in ch_names if name not in completed_set]
            completed_channels.extend(missing)
            completed_set.update(missing)
        _update_progress(za_store, "complete", completed_channels, progress_total)
        logger.info(
            "Saved export-field (Zarr) TCYX={}×{}×{}×{} kind={} → {}",
            int(arr_shape[0]),
            int(arr_shape[1]),
            int(arr_shape[2]),
            int(arr_shape[3]),
            md_common.get("kind"),
            output_path,
        )
        rich.print(
            f"[green]Exported field as TCYX with shape {arr_shape} (kind={md_common.get('kind')}, ds={int(downsample)}) → {output_path}[/green]"
        )


@correct_illum.command("plot-field-tile")
@click.argument("model", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--workspace",
    type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=Path),
)
@click.option("--roi", type=str)
@click.option("--codebook", type=str)
@click.option(
    "--tile",
    type=int,
    required=True,
    help="Tile index from TileConfiguration.registered.txt to render",
)
@click.option("--width", type=int, default=1968, show_default=True)
@click.option("--height", type=int, default=1968, show_default=True)
@click.option(
    "--mode",
    type=click.Choice(["low", "high", "range"], case_sensitive=False),
    default="range",
    show_default=True,
)
@click.option(
    "--grid-step",
    type=float,
    default=None,
    help="Override grid step used for interpolation",
)
@click.option("--output", type=click.Path(dir_okay=False, writable=True, path_type=Path))
def plot_field_tile(
    model: Path,
    workspace: Path | None,
    roi: str | None,
    codebook: str | None,
    tile: int,
    width: int,
    height: int,
    mode: str,
    grid_step: float | None,
    output: Path | None,
) -> None:
    """Render the correction field patch covering a single tile."""

    import matplotlib.pyplot as plt

    rng_model = RangeFieldPointsModel.from_npz(model)
    meta = rng_model.meta

    roi = roi or meta.get("roi")
    codebook = codebook or meta.get("codebook")
    if roi is None or codebook is None:
        raise click.UsageError("ROI and codebook must be provided either via CLI or the model metadata")

    if workspace is None:
        workspace = Path(meta.get("workspace", "."))

    ws = Workspace(workspace)
    indices, xs0, ys0 = _resolve_tile_origins(meta, ws, roi)
    indices_list = indices.tolist()
    if tile not in indices_list:
        raise click.UsageError(f"Tile index {tile} not found in TileConfiguration for ROI '{roi}'")

    tile_pos = indices_list.index(tile)
    x0 = float(xs0[tile_pos])
    y0 = float(ys0[tile_pos])
    tile_w = float(meta.get("tile_w", 1968.0))
    tile_h = float(meta.get("tile_h", 1968.0))

    origins_list = [(float(x), float(y)) for x, y in zip(xs0.tolist(), ys0.tolist(), strict=False)]
    patch = rng_model.field_patch(
        x0=float(x0),
        y0=float(y0),
        width=int(width),
        height=int(height),
        mode=mode,
        tile_origins=origins_list,
        tile_w=tile_w,
        tile_h=tile_h,
        grid_step=grid_step,
    )

    fig, ax = plt.subplots(figsize=(6, 6), dpi=140)
    from matplotlib import colors

    if mode.lower() == "range":
        norm = colors.TwoSlopeNorm(vmin=0.5, vcenter=1.0, vmax=2.0)
        im = ax.imshow(patch, origin="upper", zorder=1, cmap="coolwarm", norm=norm)
    else:
        im = ax.imshow(patch, origin="upper", zorder=1)
    ax.set_title(f"Field patch ({mode.upper()}) — ROI={roi} CB={codebook} at ({x0:.1f}, {y0:.1f})")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()

    if output is None:
        suffix = f"tile-{int(x0)}-{int(y0)}-{mode.lower()}.png"
        output = model.with_name(f"{model.stem}-{suffix}")

    fig.savefig(output.as_posix(), dpi=200, bbox_inches="tight")
    plt.close(fig)
    rich.print(f"[green]Saved tile field plot → {output}[/green]")
