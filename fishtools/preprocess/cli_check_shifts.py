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
    infer_rounds,
    build_shift_layout_table,
    make_shifts_layout_figure,
    make_corr_hist_figure,
    make_corr_vs_l2_figure,
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


def build_corr_l2_table(
    shifts_by_tile: Mapping[int, Mapping[str, Shift]],
    *,
    roi: str,
) -> pd.DataFrame:
    """Return per-round correlation and L2 distances for each tile."""

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
        mean_shift = shifts.mean(axis=0)
        l2 = np.linalg.norm(shifts - mean_shift, axis=1)

        for (tile_id, shift), dist in zip(per_round, l2, strict=False):
            records.append({
                "roi": roi,
                "round": round_name,
                "tile": int(tile_id),
                "correlation": float(shift.corr),
                "L2": float(dist),
            })

    records.sort(key=lambda item: (str(item["round"]), int(item["tile"])))
    return pd.DataFrame.from_records(records, columns=["roi", "round", "tile", "correlation", "L2"])


## Saving is unified via fishtools.utils.plot.save_figure


@click.command("check-shifts")
@click.argument(
    "path",
    type=click.Path(
        exists=True, file_okay=False, dir_okay=True, readable=True, resolve_path=True, path_type=Path
    ),
)
@click.option(
    "--codebook",
    "-c",
    "codebook_path",
    type=click.Path(
        exists=True, file_okay=True, dir_okay=False, readable=True, resolve_path=True, path_type=Path
    ),
    required=True,
    help="Path to the codebook JSON used for registration/decoding.",
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
    codebook_path: Path,
    rois: list[str] | None,
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

    setup_cli_logging(
        path,
        component="preprocess.check_shifts",
        file=f"check-shifts-{codebook_path.stem}",
        extra={"codebook": codebook_path.stem, "roi": ",".join(rois) if rois else "all"},
    )

    ws = Workspace(path)
    codebook = Codebook(codebook_path)
    roi_list = ws.resolve_rois(rois)

    # Default output directory mirrors cli_spotlook behavior
    if output_dir is None:
        output_dir = path.parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Output directory: {output_dir}")

    for roi in roi_list:
        logger.info(f"Analyzing shifts for ROI '{roi}' / codebook '{codebook.name}'")
        shift_dir = ws.deconved / f"shifts--{roi}+{codebook.name}"
        if not shift_dir.exists():
            logger.warning(f"Shifts directory not found; skipping ROI {roi}: {shift_dir}")
            continue

        _check_missing_tiles(ws, roi, codebook.name, ref_round, shift_dir)
        shifts_by_tile = _load_shifts(shift_dir)
        if not shifts_by_tile:
            logger.warning(f"No shift records loaded for ROI {roi}")
            continue

        fig1 = make_shifts_scatter_figure(shifts_by_tile, ncols=cols, corr_threshold=corr_threshold)
        save_figure(fig1, output_dir, "shifts_scatter", roi, codebook.name, log_level="INFO")

        fig2 = make_corr_vs_l2_figure(shifts_by_tile, ncols=cols, corr_threshold=corr_threshold)
        save_figure(fig2, output_dir, "shifts_corr_vs_l2", roi, codebook.name, log_level="INFO")

        fig3 = make_corr_hist_figure(shifts_by_tile, ncols=cols)
        save_figure(fig3, output_dir, "shifts_corr_hist", roi, codebook.name, log_level="INFO")

        corr_l2_df = build_corr_l2_table(shifts_by_tile, roi=roi)
        csv_path = output_dir / f"shifts_corr_l2--{roi}+{codebook.name}.csv"
        if corr_l2_df.empty:
            logger.warning(f"No correlation/L2 records produced for ROI '{roi}'. Skipping CSV export.")
        else:
            corr_l2_df.to_csv(csv_path, index=False)
            logger.info(f"Saved correlation/L2 CSV: {csv_path}")

        # New: per-round layout scatter using TileConfiguration centers offset by shifts
        try:
            tc = ws.tileconfig(roi)
        except FileNotFoundError:
            logger.warning(f"TileConfiguration not found for ROI {roi}; skipping shifts layout plot.")
            continue
        tile_size_px = _infer_tile_size_px(ws, roi, codebook.name, default=1968.0)
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
            save_figure(fig_layout, output_dir, "shifts_layout", roi, codebook.name, log_level="INFO")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Failed to render shifts_layout for ROI {roi}: {e}")
