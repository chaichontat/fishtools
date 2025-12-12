from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import click
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from tifffile import TiffFile

from fishtools.io.codebook import Codebook
from fishtools.io.workspace import Workspace
from fishtools.plot.diagnostics.shifts import (
    Shift,
    ShiftsAdapter,
    build_shift_layout_table,
    infer_rounds,
    make_corr_hist_figure,
    make_corr_vs_l2_figure,
    make_shifts_layout_figure,
    make_shifts_scatter_figure,
)
from fishtools.utils.logging import setup_cli_logging
from fishtools.utils.plot import save_figure

# Use the same global style baseline as other CLIs
sns.set_theme()


def _load_shifts(path: Path) -> dict[int, dict[str, Shift]]:
    """Load all shift JSONs from a directory into a tile-indexed mapping."""
    files = sorted(path.glob("*.json"))
    if not files:
        logger.warning(f"No .json files found in {path}")
        return {}

    result: dict[int, dict[str, Shift]] = {}
    for f in files:
        try:
            tile = int(f.stem.rsplit("-", 1)[-1])
        except ValueError:
            logger.warning(f"Skipping non-conforming file name (expected suffix index): {f.name}")
            continue
        try:
            result[tile] = ShiftsAdapter.validate_json(f.read_text())
        except Exception as e:  # noqa: BLE001 - fail fast with context
            raise RuntimeError(f"Failed parsing shifts JSON: {f}") from e
    return result


def _check_missing_tiles(
    ws: Workspace, roi: str, codebook: str, ref_round: str | None, shift_dir: Path
) -> None:
    # Compare the number of shifts jsons vs tiles for a reference round if available
    if ref_round is None:
        return
    ref_dir = ws.deconved / f"{ref_round}--{roi}"
    if not ref_dir.exists():
        logger.warning(f"Reference round directory not found: {ref_dir}")
        return

    ref_tiles = {p.stem.rsplit("-", 1)[-1] for p in ref_dir.glob("*.tif")}
    shift_tiles = {p.stem.rsplit("-", 1)[-1] for p in shift_dir.glob("*.json")}
    missing = sorted(ref_tiles - shift_tiles)
    if missing:
        logger.warning(f"Missing shifts for tiles: {missing}")


def _infer_tile_size_px(ws: Workspace, roi: str, codebook: str, default: float = 1968.0) -> float:
    """Infer tile edge length in pixels from a sample registered TIFF.

    Falls back to ``default`` if anything goes wrong. Logs the chosen value.
    """
    try:
        reg_dir = ws.registered(roi, codebook)
        sample = next(iter(sorted(reg_dir.glob("reg-*.tif"))), None)
        if sample is None:
            logger.warning(f"No registered TIFFs found to infer tile size; using default {default}.")
            return float(default)
        with TiffFile(sample) as tif:
            shape = tif.pages[0].shape  # YX (or CYX/ZCYX but first page is YX)
        if len(shape) < 2:
            logger.warning(f"Unexpected TIFF shape for tile size inference: {shape}; using default {default}.")
            return float(default)
        y, x = int(shape[-2]), int(shape[-1])
        if y != x:
            logger.warning(f"Non-square tile detected ({x}x{y}); using X dimension {x} as tile size.")
        size = float(x)
        logger.info(f"Inferred tile size from {sample.name}: {size}px")
        return size
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Failed inferring tile size; using default {default}: {e}")
        return float(default)


def build_metrics_table(
    shifts_by_tile: Mapping[int, Mapping[str, Shift]],
    *,
    roi: str,
) -> pd.DataFrame:
    """Return per-round metrics for each tile including correlation, L2, and additional diagnostics."""

    rounds = infer_rounds(shifts_by_tile)
    records: list[dict[str, object]] = []

    for round_name in rounds:
        per_round: list[tuple[int, Shift]] = []
        for tile_id, per_tile in shifts_by_tile.items():
            shift = per_tile.get(round_name)
            if shift is None:
                continue
            per_round.append((tile_id, shift))
        if not per_round:
            continue

        shifts = np.array([shift.shifts for _, shift in per_round], dtype=float)
        center_shift = np.median(shifts, axis=0)
        l2 = np.linalg.norm(shifts - center_shift, axis=1)

        for (tile_id, shift), dist in zip(per_round, l2, strict=False):
            records.append({
                "roi": roi,
                "round": round_name,
                "tile": int(tile_id),
                "correlation": float(shift.corr),
                "L2": float(dist),
                "residual": float(shift.residual),
                "iterations": shift.iterations,
                "final_threshold": shift.final_threshold,
                "final_fwhm": shift.final_fwhm,
                "n_spots": shift.n_spots,
                "mode": shift.mode,
                "algorithm": shift.algorithm,
            })

    records.sort(key=lambda item: (str(item["round"]), int(item["tile"])))
    return pd.DataFrame.from_records(records, columns=[
        "roi", "round", "tile", "correlation", "L2", "residual",
        "iterations", "final_threshold", "final_fwhm", "n_spots", "mode", "algorithm"
    ])


## Saving is unified via fishtools.utils.plot.save_figure


@click.command("check-shifts")
@click.argument(
    "path",
    type=click.Path(
        exists=True, file_okay=False, dir_okay=True, readable=True, resolve_path=True, path_type=Path
    ),
)
@click.argument("roi", required=False, metavar="ROI")
@click.option(
    "--codebook",
    "-c",
    "codebook_path",
    type=click.Path(
        exists=True, file_okay=True, dir_okay=False, readable=True, resolve_path=True, path_type=Path
    ),
    required=False,
    default=None,
    help="Path to the codebook JSON. If omitted, auto-discovers codebooks from registered directories.",
)
@click.option(
    "--roi",
    "rois",
    multiple=True,
    help="ROI(s) to analyze. If omitted, process all ROIs in the workspace.",
)
@click.option(
    "--output",
    "-o",
    "output_dir",
    type=click.Path(file_okay=False, dir_okay=True, writable=True, resolve_path=True, path_type=Path),
    default=None,
    help="Output directory for PNGs [default: '<workspace_parent>/output']",
)
@click.option("--cols", type=int, default=4, show_default=True, help="Number of columns in the panel grid")
@click.option(
    "--corr-threshold",
    type=float,
    default=0.8,
    show_default=True,
    help="Highlight tiles with corr < threshold",
)
@click.option(
    "--ref-round",
    type=str,
    default=None,
    help="Optional round name to check missing tiles against (e.g., '2_10_18')",
)
def check_shifts(
    path: Path,
    roi: str | None,
    codebook_path: Path | None,
    rois: tuple[str, ...],
    output_dir: Path | None,
    cols: int,
    corr_threshold: float,
    ref_round: str | None,
) -> None:
    """Inspect registration shifts across tiles and rounds; save diagnostic PNGs.

    Generates three figures per ROI:
    - shifts_scatter: X/Y shift scatter per round colored by correlation
    - shifts_corr_vs_l2: correlation vs L2 distance from per-round mean shift
    - shifts_corr_hist: correlation histograms
    """

    if roi is not None and rois:
        msg = "Specify ROI either as a positional argument or via --roi, not both."
        raise click.UsageError(msg)

    selected_rois: list[str] | None
    if roi is not None:
        selected_rois = [roi]
    elif rois:
        selected_rois = list(rois)
    else:
        selected_rois = None

    ws = Workspace(path)
    roi_list = ws.resolve_rois(selected_rois)

    # Discover codebooks if not provided
    if codebook_path is not None:
        codebook_names = [Codebook(codebook_path).name]
    else:
        codebook_names = ws.registered_codebooks(rois=roi_list)
        if not codebook_names:
            raise click.UsageError(
                "No codebooks found. Provide --codebook or ensure registered directories exist."
            )
        logger.info(f"Auto-discovered codebooks: {codebook_names}")

    roi_label = ",".join(selected_rois) if selected_rois else "all"
    codebook_label = codebook_path.stem if codebook_path else ",".join(codebook_names)

    setup_cli_logging(
        path,
        component="preprocess.check_shifts",
        file=f"check-shifts-{codebook_label}",
        extra={"codebook": codebook_label, "roi": roi_label},
    )

    # Default output directory mirrors cli_spotlook behavior
    if output_dir is None:
        output_dir = path.parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Output directory: {output_dir}")

    for codebook_name in codebook_names:
        for roi in roi_list:
            logger.info(f"Analyzing shifts for ROI '{roi}' / codebook '{codebook_name}'")
            shift_dir = ws.deconved / f"shifts--{roi}+{codebook_name}"
            if not shift_dir.exists():
                logger.warning(f"Shifts directory not found; skipping ROI {roi}: {shift_dir}")
                continue

            _check_missing_tiles(ws, roi, codebook_name, ref_round, shift_dir)
            shifts_by_tile = _load_shifts(shift_dir)
            if not shifts_by_tile:
                logger.warning(f"No shift records loaded for ROI {roi}")
                continue

            fig1 = make_shifts_scatter_figure(shifts_by_tile, ncols=cols, corr_threshold=corr_threshold)
            save_figure(fig1, output_dir / "shifts_scatter", "shifts_scatter", roi, codebook_name, log_level="INFO")

            fig2 = make_corr_vs_l2_figure(shifts_by_tile, ncols=cols, corr_threshold=corr_threshold)
            save_figure(fig2, output_dir / "shifts_corr_vs_l2", "shifts_corr_vs_l2", roi, codebook_name, log_level="INFO")

            fig3 = make_corr_hist_figure(shifts_by_tile, ncols=cols)
            save_figure(fig3, output_dir / "shifts_corr_hist", "shifts_corr_hist", roi, codebook_name, log_level="INFO")

            metrics_df = build_metrics_table(shifts_by_tile, roi=roi)
            csv_path = output_dir / "shifts_metrics" / f"shifts_metrics--{roi}+{codebook_name}.csv"
            if metrics_df.empty:
                logger.warning(f"No metrics records produced for ROI '{roi}'. Skipping CSV export.")
            else:
                csv_path.parent.mkdir(parents=True, exist_ok=True)
                metrics_df.to_csv(csv_path, index=False)
                logger.info(f"Saved metrics CSV: {csv_path}")

            # New: per-round layout scatter using TileConfiguration centers offset by shifts
            try:
                tc = ws.tileconfig(roi)
            except FileNotFoundError:
                logger.warning(f"TileConfiguration not found for ROI {roi}; skipping shifts layout plot.")
                continue
            tile_size_px = _infer_tile_size_px(ws, roi, codebook_name, default=1968.0)
            centers = {
                int(idx): (float(x) + 0.5 * tile_size_px, float(y) + 0.5 * tile_size_px)
                for idx, x, y in tc.df.select(["index", "x", "y"]).iter_rows()
            }
            records = build_shift_layout_table(centers, shifts_by_tile, roi=roi)
            try:
                fig_layout = make_shifts_layout_figure(
                    records,
                    tile_size_px=tile_size_px,
                    pixel_size_um=0.108,
                    label_skip=2,
                    corr_threshold=corr_threshold,
                )
                save_figure(fig_layout, output_dir / "shifts_layout", "shifts_layout", roi, codebook_name, log_level="INFO")
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Failed to render shifts_layout for ROI {roi}: {e}")
