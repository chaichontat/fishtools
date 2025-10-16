from __future__ import annotations

from pathlib import Path

import click
import seaborn as sns
from loguru import logger

from fishtools.io.codebook import Codebook
from fishtools.io.workspace import Workspace
from fishtools.plot.diagnostics.shifts import (
    Shift,
    ShiftsAdapter,
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
