from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

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

    roi = roi or str(meta.get("roi"))
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
            ws.path, component="preprocess.correct-illum.render-global", file=log_file_tag
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
            logger.info("RenderGlobal: multi-channel NPZ detected; defaulting to first channel '{}'", ch0)
    except Exception:
        pass

    # Evaluate the global field once, normalize with the union-of-tiles mask, then slice per-tile.
    xs, ys, low_field, high_field = rng_model.evaluate(
        x_min, y_min, x_max, y_max, grid_step=step_eff, neighbors=neigh_eff, kernel=ker, smoothing=smooth_eff
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
@click.argument("roi", type=str)
@click.argument("idx", required=False, type=int)
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
def calculate_percentiles(
    workspace: Path,
    roi: str,
    idx: int | None,
    codebook_opt: str | None,
    percentiles: tuple[float, float],
    grid: int,
    max_streams: int,
    channel_chunk: int,
    threads: int,
    out_suffix: str,
    overwrite: bool,
) -> None:
    """Compute subtile illumination percentiles (GPU preferred)."""

    console = rich.get_console()
    ws = Workspace(workspace)
    if codebook_opt is None:
        candidates = ws.registered_codebooks(rois=[roi])
        if not candidates:
            raise click.UsageError(f"No registered codebooks found for ROI '{roi}'. Provide --codebook.")
        if len(candidates) > 1:
            raise click.UsageError(
                f"Multiple codebooks present for ROI '{roi}': {', '.join(candidates)}. Specify --codebook."
            )
        cb = candidates[0]
    else:
        path = Path(codebook_opt)
        cb = path.stem if (path.suffix and not path.is_dir()) else codebook_opt

    file_map, missing = ws.registered_file_map(cb, rois=[roi])
    if missing:
        raise click.UsageError(f"ROI(s) missing registered outputs: {', '.join(missing)}")

    tiles_all = file_map.get(roi, [])
    if not tiles_all:
        raise click.UsageError(f"No registered tiles for roi='{roi}', codebook='{cb}'.")

    use_gpu = os.environ.get("GPU", "0") == "1" and cp is not None
    tiles = tiles_all
    if idx is not None:
        target = ws.regimg(roi, cb, int(idx))
        if not target.exists():
            raise click.UsageError(f"Tile {target.name} not found under registered--{roi}+{cb}.")
        tiles = [target]

    p_lo, p_hi = percentiles
    key_lo = format(p_lo, ".10g")
    key_hi = format(p_hi, ".10g")

    if not overwrite:
        before = len(tiles)
        tiles = [t for t in tiles if not _skip_existing(t, out_suffix, grid, (key_lo, key_hi), overwrite)]
        skipped = before - len(tiles)
    else:
        skipped = 0

    console.print(
        f"Found [bold]{len(tiles)}[/bold] tile(s) for roi={roi}, codebook={cb}. Backend: {'GPU' if use_gpu else 'CPU'}",
        style="green",
    )

    def _process(tile: Path) -> None:
        arr, channel_names = _load_czyx(tile, use_gpu)
        if use_gpu:
            lo, hi = _compute_subtile_percentiles_gpu(
                arr,
                percentiles=percentiles,
                grid=grid,
                max_streams=max_streams,
                channel_chunk=channel_chunk,
            )
        else:
            assert isinstance(arr, np.ndarray)
            lo, hi = _compute_subtile_percentiles_cpu(arr, percentiles=percentiles, grid=grid)
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
                    entries.append({
                        "x": int(x0),
                        "y": int(y0),
                        "x_size": int(x1 - x0),
                        "y_size": int(y1 - y0),
                        "percentiles": {
                            key_lo: float(lo[r, col, ci]),
                            key_hi: float(hi[r, col, ci]),
                        },
                    })
            payload[cname] = entries
        tile.with_suffix(out_suffix).write_text(json.dumps(payload, indent=2))

    if use_gpu:
        with progress_bar(len(tiles)) as advance:
            for tile in tiles:
                _process(tile)
                advance()
    else:
        with progress_bar_threadpool(len(tiles), threads=max(1, int(threads))) as submit:
            for tile in tiles:
                submit(_process, tile)

    tail = f" (skipped {skipped} existing; use --overwrite to recompute)" if skipped else ""
    rich.print(f"[cyan]Percentile JSONs written next to TIFFs with suffix '{out_suffix}'.{tail}[/cyan]")


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
    reg_dir = ws.registered(roi, Path(codebook).stem if Path(codebook).suffix else codebook)

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
            "codebook": Path(codebook).stem if Path(codebook).suffix else codebook,
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
            opt_dir = ws.opt(str(meta["codebook"]))
            opt_dir = opt_dir.path if hasattr(opt_dir, "path") else Path(opt_dir)
            suffix = f"{channel_used}-{p_low}-{p_high}.npz".replace("/", "-")
            output = opt_dir / f"illum-field--{roi}-{suffix}"
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
        "codebook": Path(codebook).stem if Path(codebook).suffix else codebook,
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
        opt_dir = ws.opt(str(meta["codebook"]))
        opt_dir = opt_dir.path if hasattr(opt_dir, "path") else Path(opt_dir)
        suffix = f"ALL-{p_low}-{p_high}.npz".replace("/", "-")
        output = opt_dir / f"illum-field--{roi}-{suffix}"
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
    """Plot a saved illumination field NPZ (LOW/HIGH/RANGE)."""

    import matplotlib.pyplot as plt

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

    # If the NPZ is multi-channel, pick the first channel for plotting by default
    try:
        vlow = np.asarray(rng_model.vlow)
        if vlow.ndim == 2 and vlow.shape[1] > 1:
            ch_names = list(meta.get("channels", [])) if isinstance(meta.get("channels"), list) else []
            ch0 = ch_names[0] if ch_names else "ch0"
            sub_meta = dict(meta)
            # Force per-channel scale recompute
            sub_meta.pop("range_mean", None)
            rng_model = RangeFieldPointsModel(
                xy=rng_model.xy,
                vlow=vlow[:, 0],
                vhigh=np.asarray(rng_model.vhigh)[:, 0],
                meta=dict(sub_meta, channel=ch0),
            )
            meta = rng_model.meta
    except Exception:
        pass

    xs, ys, low_field, high_field = rng_model.evaluate(
        x0,
        y0,
        x1,
        y1,
        grid_step=grid_step,
    )

    fig = plt.figure(figsize=(10, 6), dpi=140)
    title = f"Illum Field — ROI={roi} CB={codebook} CH={meta.get('channel')}"
    fig.suptitle(title)

    ax_low = fig.add_subplot(2, 2, 1)
    if xs0.size:
        mask = _mask_union_of_tiles(xs, ys, xs0, ys0, tile_w, tile_h)
    else:
        mask = np.ones_like(low_field, dtype=bool)

    low_masked = np.ma.array(low_field, mask=~mask)
    inv_range = rng_model.range_correction(low_field, high_field, mask=mask)
    range_masked = np.ma.array(inv_range, mask=~mask)

    im_low = ax_low.imshow(low_masked, origin="upper", extent=[x0, x1, y0, y1], zorder=1)
    ax_low.set_aspect("equal")
    ax_low.set_title("LOW field")
    fig.colorbar(im_low, ax=ax_low, fraction=0.046)

    ax_range = fig.add_subplot(2, 2, 2)
    from matplotlib import colors

    # Fixed colorbar for RANGE: clamp to [0.5, 2] with center at 1
    div_norm = colors.TwoSlopeNorm(vmin=0.5, vcenter=1.0, vmax=2.0)

    im_rng = ax_range.imshow(
        range_masked,
        origin="upper",
        extent=[x0, x1, y0, y1],
        zorder=1,
        cmap="coolwarm",
        norm=div_norm,
    )
    ax_range.set_aspect("equal")
    ax_range.set_title("RANGE correction (normalized inverse)")
    fig.colorbar(im_rng, ax=ax_range, fraction=0.046)

    ax_low_pts = fig.add_subplot(2, 2, 3)
    sc_low = ax_low_pts.scatter(rng_model.xy[:, 0], rng_model.xy[:, 1], c=rng_model.vlow, s=8, cmap="viridis")
    ax_low_pts.set_xlim(x0, x1)
    ax_low_pts.set_ylim(y1, y0)
    ax_low_pts.set_aspect("equal")
    ax_low_pts.set_title("Low samples")
    fig.colorbar(sc_low, ax=ax_low_pts, fraction=0.046)

    ax_high_pts = fig.add_subplot(2, 2, 4)
    sc_high = ax_high_pts.scatter(
        rng_model.xy[:, 0], rng_model.xy[:, 1], c=rng_model.vhigh, s=8, cmap="magma"
    )
    ax_high_pts.set_xlim(x0, x1)
    ax_high_pts.set_ylim(y1, y0)
    ax_high_pts.set_aspect("equal")
    ax_high_pts.set_title("High samples")
    fig.colorbar(sc_high, ax=ax_high_pts, fraction=0.046)

    fig.tight_layout()

    if output is None:
        output = model.with_suffix(".png")
    fig.savefig(output.as_posix(), dpi=200, bbox_inches="tight")
    plt.close(fig)
    rich.print(f"[green]Saved illumination plot → {output}[/green]")


@correct_illum.command("export-field")
@click.argument("model", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--workspace",
    type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=Path),
)
@click.option("--roi", type=str, required=False)
@click.option("--codebook", type=str, required=False)
@click.option(
    "--downsample", type=int, default=1, show_default=True, help="Downsample factor for output (1 = native)"
)
@click.option("--grid-step", type=float, default=None, help="Override grid step used for evaluation")
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
        "Output Zarr store path (.zarr). If --what=both and --output is provided, "
        "two stores will be written by appending -range.zarr and -low.zarr to the base name. "
        "When omitted, defaults to '<model>-field-<kind>-ds<downsample>.zarr'."
    ),
)
def export_field(
    model: Path,
    workspace: Path | None,
    roi: str | None,
    codebook: str | None,
    downsample: int,
    grid_step: float | None,
    what: str,
    neighbors: int | None,
    smoothing: float | None,
    output: Path | None,
) -> None:
    """Export LOW and normalized inverse RANGE fields cropped to the union-of-tiles.

    Parity with plot-field logic:
    - Evaluate LOW/HIGH on a grid from tile extents
    - Build a union-of-tiles (union-of-squares) mask
    - Normalize RANGE using that global mask
    - Crop to the mask's tight bounding box (remove irrelevant regions)
    - Resize to native and optionally downsample; values outside the union are NaN
    """

    import zarr
    from skimage.transform import resize

    if downsample <= 0:
        raise click.UsageError("--downsample must be a positive integer")

    rng_model = RangeFieldPointsModel.from_npz(model)
    meta = rng_model.meta

    roi = roi or meta.get("roi")
    codebook = codebook or meta.get("codebook")
    if workspace is None:
        workspace = Path(meta.get("workspace", "."))

    ws = Workspace(workspace)
    meta.setdefault("workspace", str(ws.path))

    # Initialize workspace-scoped logging like other commands
    try:
        log_file_tag = f"{roi}+{codebook}" if codebook else (roi or "roi")
        setup_workspace_logging(ws.path, component="preprocess.correct-illum.export-field", file=log_file_tag)
    except Exception:
        logger.opt(exception=True).debug("Workspace logging setup failed; continuing with default logger")

    indices, xs0, ys0 = _resolve_tile_origins(meta, ws, roi)
    logger.info(
        "ExportField: model='{}', ROI='{}', CB='{}', tiles={}, ds={}, grid_step={}, neighbors={}, smoothing={}",
        model.name,
        roi,
        codebook,
        int(indices.size),
        int(downsample),
        grid_step if grid_step is not None else "meta",
        int(neighbors or 0) if neighbors is not None else 0,
        smoothing if smoothing is not None else meta.get("smoothing"),
    )

    tile_w = float(meta.get("tile_w", 1968.0))
    tile_h = float(meta.get("tile_h", 1968.0))

    if xs0.size:
        x0 = float(xs0.min())
        x1 = float(xs0.max() + tile_w)
    else:
        # Fallback to point cloud bounds if no tile origins are available
        pts = rng_model.xy.astype(float, copy=False)
        x0 = float(pts[:, 0].min()) - tile_w * 0.5 if pts.size else 0.0
        x1 = float(pts[:, 0].max()) + tile_w * 0.5 if pts.size else tile_w

    if ys0.size:
        y0 = float(ys0.min())
        y1 = float(ys0.max() + tile_h)
    else:
        pts = rng_model.xy.astype(float, copy=False)
        y0 = float(pts[:, 1].min()) - tile_h * 0.5 if pts.size else 0.0
        y1 = float(pts[:, 1].max()) + tile_h * 0.5 if pts.size else tile_h

    # Support multi-channel NPZs by evaluating per-channel and stacking to CYX
    vlow = np.asarray(rng_model.vlow)
    vhigh = np.asarray(rng_model.vhigh)
    n_channels = int(vlow.shape[1]) if vlow.ndim == 2 else 1
    ch_names = list(meta.get("channels", [])) if isinstance(meta.get("channels"), list) else []
    if not ch_names or len(ch_names) != n_channels:
        ch_names = (
            [str(meta.get("channel", "channel_0"))]
            if n_channels == 1
            else [f"ch{i}" for i in range(n_channels)]
        )
    logger.info("Detected {} channel(s) in field NPZ: {}", n_channels, ",".join(ch_names))

    planes_range: list[np.ndarray] = []
    planes_low: list[np.ndarray] = []
    m_sub_ref: np.ndarray | None = None
    x0_crop = x1_crop = y0_crop = y1_crop = 0.0
    H = W = Hds = Wds = 0
    for ci in range(n_channels):
        vlo_ci = vlow[:, ci] if vlow.ndim == 2 else vlow
        vhi_ci = vhigh[:, ci] if vhigh.ndim == 2 else vhigh
        meta_ci = dict(meta)
        # Force per-channel scale recompute inside submodel
        meta_ci.pop("range_mean", None)
        meta_ci["channel"] = ch_names[ci]
        mdl_ci = RangeFieldPointsModel(xy=rng_model.xy, vlow=vlo_ci, vhigh=vhi_ci, meta=meta_ci)

        xs, ys, low_field, high_field = mdl_ci.evaluate(
            x0,
            y0,
            x1,
            y1,
            grid_step=grid_step,
            neighbors=neighbors,
            smoothing=smoothing,
        )
        mask = (
            _mask_union_of_tiles(xs, ys, xs0, ys0, tile_w, tile_h)
            if xs0.size
            else np.ones_like(low_field, bool)
        )
        inv_range = mdl_ci.range_correction(low_field, high_field, mask=mask)

        if ci == 0:
            logger.info(
                "Evaluated field grid: nx={}, ny={}, bbox=({:.1f},{:.1f})-({:.1f},{:.1f})",
                int(xs.size),
                int(ys.size),
                x0,
                y0,
                x1,
                y1,
            )

            rows_any = mask.any(axis=1)
            cols_any = mask.any(axis=0)
            if rows_any.any() and cols_any.any():
                j0 = int(np.argmax(rows_any))
                j1 = int(len(rows_any) - 1 - np.argmax(rows_any[::-1]))
                i0 = int(np.argmax(cols_any))
                i1 = int(len(cols_any) - 1 - np.argmax(cols_any[::-1]))
                j0 = max(0, min(j0, j1))
                i0 = max(0, min(i0, i1))
            else:
                j0, j1 = 0, mask.shape[0] - 1
                i0, i1 = 0, mask.shape[1] - 1

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
                "Cropped native size: {}x{} (W×H); downsampled to {}x{}", int(W), int(H), int(Wds), int(Hds)
            )
            m_sub_ref = mask[j0 : j1 + 1, i0 : i1 + 1]

        inv_sub = inv_range[j0 : j1 + 1, i0 : i1 + 1]
        native_range = resize(inv_sub, (H, W), preserve_range=True, anti_aliasing=True).astype(np.float32)
        out_plane_range = resize(native_range, (Hds, Wds), preserve_range=True, anti_aliasing=True).astype(
            np.float32
        )
        planes_range.append(out_plane_range)

        # Also prepare LOW field if requested
        if what.lower() in {"low", "both"}:
            low_sub = low_field[j0 : j1 + 1, i0 : i1 + 1]
            native_low = resize(low_sub, (H, W), preserve_range=True, anti_aliasing=True).astype(np.float32)
            out_plane_low = resize(native_low, (Hds, Wds), preserve_range=True, anti_aliasing=True).astype(
                np.float32
            )
            planes_low.append(out_plane_low)

    out_range = (
        np.stack(planes_range, axis=0) if planes_range else np.zeros((n_channels, 1, 1), dtype=np.float32)
    )
    out_low = np.stack(planes_low, axis=0) if planes_low else None

    if m_sub_ref is not None:
        m_native = resize(m_sub_ref.astype(float), (H, W), preserve_range=True, anti_aliasing=False).astype(
            np.float32
        )
        m_ds = resize(m_native, (Hds, Wds), preserve_range=True, anti_aliasing=False).astype(np.float32)
        m_bool = m_ds > 0.5
        if (~m_bool).any():
            out_range = out_range.copy()
            out_range[:, ~m_bool] = 0
            if out_low is not None:
                out_low = out_low.copy()
                out_low[:, ~m_bool] = 0

    # Resolve output path(s) as Zarr stores
    mode = what.lower()
    if mode == "range":
        if output is None:
            output = model.with_name(f"{model.stem}-field-range-ds{int(downsample)}.zarr")
        out_paths = {"range": output}
    elif mode == "low":
        if output is None:
            output = model.with_name(f"{model.stem}-field-low-ds{int(downsample)}.zarr")
        out_paths = {"low": output}
    else:  # both
        if output is not None:
            base = output.with_suffix("")
            out_paths = {
                "range": base.with_name(f"{base.name}-range.zarr"),
                "low": base.with_name(f"{base.name}-low.zarr"),
            }
        else:
            out_paths = {
                "range": model.with_name(f"{model.stem}-field-range-ds{int(downsample)}.zarr"),
                "low": model.with_name(f"{model.stem}-field-low-ds{int(downsample)}.zarr"),
            }

    md_common = {
        "axes": "CYX",
        "roi": roi,
        "codebook": codebook,
        "tile_w": int(tile_w),
        "tile_h": int(tile_h),
        "grid_step": float(grid_step) if grid_step is not None else None,
        "neighbors": int(neighbors or 0),
        "smoothing": float(smoothing) if smoothing is not None else None,
        "downsample": int(downsample),
        # Cropped global coordinates (plot-field parity)
        "x0": float(x0_crop),
        "y0": float(y0_crop),
        "width_native": int(W),
        "height_native": int(H),
        "cropped": True,
    }
    md_common = {k: v for k, v in md_common.items() if v is not None}
    md_common["channels"] = ch_names

    # Write requested outputs (Zarr array stores with attrs)
    if "range" in out_paths:
        md_r = dict(md_common)
        md_r["kind"] = "range"
        rng_store = out_paths["range"]
        # Ensure parent exists for nested store
        rng_store.parent.mkdir(parents=True, exist_ok=True)
        chunks = (1, min(512, Hds), min(512, Wds))
        za = zarr.open(str(rng_store), mode="w", shape=out_range.shape, chunks=chunks, dtype=np.float32)
        za[:] = out_range
        za.attrs["axes"] = "CYX"
        za.attrs["model_meta"] = md_r
        logger.info(
            "Saved export-field RANGE (Zarr) CYX={}×{}×{} → {}",
            int(out_range.shape[0]),
            int(Hds),
            int(Wds),
            rng_store,
        )
        rich.print(
            f"[green]Exported RANGE field (normalized, ds={int(downsample)}) as Zarr with CYX shape {out_range.shape} → {rng_store}[/green]"
        )
    if "low" in out_paths and out_low is not None:
        md_l = dict(md_common)
        md_l["kind"] = "low"
        low_store = out_paths["low"]
        low_store.parent.mkdir(parents=True, exist_ok=True)
        chunks_l = (1, min(512, Hds), min(512, Wds))
        zb = zarr.open(str(low_store), mode="w", shape=out_low.shape, chunks=chunks_l, dtype=np.float32)
        zb[:] = out_low
        zb.attrs["axes"] = "CYX"
        zb.attrs["model_meta"] = md_l
        logger.info(
            "Saved export-field LOW (Zarr) CYX={}×{}×{} → {}",
            int(out_low.shape[0]),
            int(Hds),
            int(Wds),
            low_store,
        )
        rich.print(
            f"[green]Exported LOW field (ds={int(downsample)}) as Zarr with CYX shape {out_low.shape} → {low_store}[/green]"
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
