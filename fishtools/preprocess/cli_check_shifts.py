from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import click
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from pydantic import ValidationError
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

sns.set_theme()


def _load_shifts(path: Path) -> dict[int, dict[str, Shift]]:
    """Load shift JSONs from a directory into a tile-indexed mapping."""
    files = sorted(path.glob("*.json"))
    if not files:
        return {}

    result: dict[int, dict[str, Shift]] = {}
    for f in files:
        try:
            tile = int(f.stem.rsplit("-", 1)[-1])
        except ValueError:
            logger.warning(f"Skipping shifts file with non-numeric suffix: {f.name}")
            continue
        try:
            result[tile] = ShiftsAdapter.validate_json(f.read_text())
        except (OSError, UnicodeDecodeError, ValidationError, ValueError) as exc:
            raise click.ClickException(f"Failed parsing shifts JSON at {f}: {exc}") from exc
    return result


def _load_coarse_shifts(ws: Workspace, roi: str) -> dict[int, dict[str, Shift]]:
    """Load coarse shifts from fix-shifts output and convert to Shift format.

    The coarse_shifts.json format:
    {
        "reference": "...",
        "use_fft": true/false,
        "tiles": {
            "0001": {
                "round_name": {"dx": ..., "dy": ..., "magnitude": ..., "residual": ...}
            }
        }
    }

    Returns a dict[tile_id, dict[round_name, Shift]] compatible with plotting functions.
    """
    coarse_path = ws.coarse_shifts_json(roi)
    if not coarse_path.exists():
        return {}

    try:
        data = json.loads(coarse_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning(f"Failed to load coarse shifts from {coarse_path}: {exc}")
        return {}

    if not isinstance(data, dict) or "tiles" not in data:
        return {}

    tiles_data = data.get("tiles", {})
    if not isinstance(tiles_data, dict):
        return {}

    result: dict[int, dict[str, Shift]] = {}
    for tile_str, rounds_data in tiles_data.items():
        try:
            tile_id = int(tile_str)
        except ValueError:
            continue

        if not isinstance(rounds_data, dict):
            continue

        tile_shifts: dict[str, Shift] = {}
        for round_name, shift_data in rounds_data.items():
            if not isinstance(shift_data, dict):
                continue

            dx = shift_data.get("dx", 0.0)
            dy = shift_data.get("dy", 0.0)
            residual = shift_data.get("residual", 0.0)
            # Coarse shifts don't have correlation; use residual-based proxy
            # Low residual = good alignment = high correlation proxy
            corr = max(0.0, 1.0 - residual) if residual > 0 else 1.0

            tile_shifts[round_name] = Shift(
                shifts=(float(dx), float(dy)),
                corr=corr,
                residual=float(residual),
            )

        if tile_shifts:
            result[tile_id] = tile_shifts

    return result


def _check_missing_tiles(
    ws: Workspace, roi: str, _codebook: str, ref_round: str | None, shift_dir: Path
) -> None:
    if ref_round is None:
        return

    ref_dir = ws.deconv_round_dir(ref_round, roi)
    repaired_dir = ws.deconv_repaired_dir(ref_round, roi)
    shift_tiles = {p.stem.rsplit("-", 1)[-1] for p in shift_dir.glob("*.json")}

    ref_tiles = {p.stem.rsplit("-", 1)[-1] for p in ref_dir.glob("*.tif")} if ref_dir.exists() else set()
    repaired_tiles = (
        {p.stem.rsplit("-", 1)[-1] for p in repaired_dir.glob("*.tif")} if repaired_dir.exists() else set()
    )

    chosen_dir = ref_dir
    chosen_tiles = ref_tiles

    if repaired_tiles and (not ref_tiles or len(repaired_tiles) > len(ref_tiles)):
        chosen_dir = repaired_dir
        chosen_tiles = repaired_tiles
        logger.info(f"Using repaired folder for reference round {ref_round}: {repaired_dir}")
    elif repaired_tiles and ref_tiles:
        missing_ref = ref_tiles - shift_tiles
        missing_repaired = repaired_tiles - shift_tiles
        if len(missing_repaired) < len(missing_ref):
            chosen_dir = repaired_dir
            chosen_tiles = repaired_tiles
            logger.info(f"Using repaired folder for reference round {ref_round}: {repaired_dir}")

    if not chosen_dir.exists():
        logger.warning(f"Reference round directory not found: {chosen_dir}")
        return

    missing = sorted(chosen_tiles - shift_tiles)
    if missing:
        logger.warning(f"Missing shifts for tiles: {missing}")


def _infer_tile_size_px(ws: Workspace, roi: str, codebook: str, default: float = 1968.0) -> float:
    """Infer tile edge length in pixels from a sample registered TIFF.

    Falls back to ``default`` when no sample exists or the sample can't be read.
    """
    reg_dir = ws.registered(roi, codebook)
    sample = next(iter(sorted(reg_dir.glob("reg-*.tif"))), None)
    if sample is None:
        return float(default)

    try:
        with TiffFile(sample) as tif:
            shape = tif.pages[0].shape
    except (OSError, ValueError) as exc:
        logger.warning(f"Failed reading {sample} to infer tile size; using default {default}: {exc}")
        return float(default)

    if len(shape) < 2:
        logger.warning(f"Unexpected TIFF shape for tile size inference: {shape}; using default {default}.")
        return float(default)

    y, x = int(shape[-2]), int(shape[-1])
    if y != x:
        logger.warning(f"Non-square tile detected ({x}x{y}); using X dimension {x} as tile size.")
    return float(x)


def build_metrics_table(
    shifts_by_tile: Mapping[int, Mapping[str, Shift]],
    *,
    roi: str,
) -> pd.DataFrame:
    """Return per-round per-tile shift metrics."""

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
            records.append(
                {
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
                }
            )

    records.sort(key=lambda item: (str(item["round"]), int(item["tile"])))
    return pd.DataFrame.from_records(
        records,
        columns=[
            "roi",
            "round",
            "tile",
            "correlation",
            "L2",
            "residual",
            "iterations",
            "final_threshold",
            "final_fwhm",
            "n_spots",
            "mode",
            "algorithm",
        ],
    )


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
    help="Output directory for PNGs [default: '<workspace>/analysis/output']",
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
@click.option(
    "--round-name",
    type=str,
    default=None,
    help="Plot coarse shifts for a repaired round (from `register fix-shifts`).",
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
    round_name: str | None,
) -> None:
    """Inspect registration shifts across tiles and rounds; save diagnostic PNGs.

    Generates figures per ROI/codebook (or per ROI/round when using --round-name):
    - shifts_scatter: X/Y shift scatter per round colored by correlation
    - shifts_corr_vs_l2: correlation vs L2 distance from per-round mean shift
    - shifts_corr_hist: correlation histograms (codebook mode only)
    - shifts_layout: spatial layout of shifts per tile

    Use --round-name to plot coarse shifts for repaired rounds (from `register fix-shifts`).
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

    if output_dir is None:
        output_dir = path.parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    roi_label = ",".join(selected_rois) if selected_rois else "all"

    # Plot coarse shifts for repaired round (from fix-shifts)
    if round_name is not None:
        setup_cli_logging(
            path,
            component="preprocess.check_shifts",
            file=f"check-shifts-{round_name}",
            extra={"round_name": round_name, "roi": roi_label},
        )
        logger.debug(f"Output directory: {output_dir}")

        for roi in roi_list:
            coarse_shifts = _load_coarse_shifts(ws, roi)
            if not coarse_shifts:
                logger.warning(f"No coarse shifts found for ROI '{roi}'")
                continue

            # Filter to only the requested round
            filtered: dict[int, dict[str, Shift]] = {}
            for tile_id, rounds_data in coarse_shifts.items():
                if round_name in rounds_data:
                    filtered[tile_id] = {round_name: rounds_data[round_name]}

            if not filtered:
                available = infer_rounds(coarse_shifts)
                logger.warning(f"Round '{round_name}' not in coarse shifts for ROI '{roi}'. Available: {available}")
                continue

            logger.info(f"Plotting coarse shifts for ROI '{roi}' / round '{round_name}'")

            fig1 = make_shifts_scatter_figure(filtered, ncols=cols, corr_threshold=corr_threshold)
            save_figure(fig1, output_dir / "shifts_scatter", "shifts_scatter", roi, round_name, log_level="INFO")

            fig2 = make_corr_vs_l2_figure(filtered, ncols=cols, corr_threshold=corr_threshold)
            save_figure(fig2, output_dir / "shifts_corr_vs_l2", "shifts_corr_vs_l2", roi, round_name, log_level="INFO")

            # Build layout if TileConfiguration is available
            try:
                tc = ws.tileconfig(roi)
                tile_size_px = _infer_tile_size_px(ws, roi, "", default=1968.0)
                centers = {
                    int(idx): (float(x) + 0.5 * tile_size_px, float(y) + 0.5 * tile_size_px)
                    for idx, x, y in tc.df.select(["index", "x", "y"]).iter_rows()
                }
                records = build_shift_layout_table(centers, filtered, roi=roi)
                try:
                    fig_layout = make_shifts_layout_figure(
                        records,
                        tile_size_px=tile_size_px,
                        pixel_size_um=0.108,
                        label_skip=2,
                        corr_threshold=corr_threshold,
                    )
                    save_figure(fig_layout, output_dir / "shifts_layout", "shifts_layout", roi, round_name, log_level="INFO")
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"Failed to render shifts_layout for ROI {roi}: {e}")
            except FileNotFoundError:
                logger.debug(f"TileConfiguration not found for ROI {roi}; skipping shifts layout plot.")
        return

    # Plot registration shifts per codebook
    if codebook_path is not None:
        codebook_names = [Codebook(codebook_path).name]
    else:
        codebook_names = ws.registered_codebooks(rois=roi_list)
        if not codebook_names:
            raise click.UsageError(
                "No codebooks found. Provide --codebook or ensure registered directories exist."
            )
        logger.info(f"Auto-discovered codebooks: {codebook_names}")

    codebook_label = codebook_path.stem if codebook_path else ",".join(codebook_names)

    setup_cli_logging(
        path,
        component="preprocess.check_shifts",
        file=f"check-shifts-{codebook_label}",
        extra={"codebook": codebook_label, "roi": roi_label},
    )

    if output_dir is None:
        output_dir = ws.output.root
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.debug(f"Output directory: {output_dir}")

    for codebook_name in codebook_names:
        for roi in roi_list:
            logger.info(f"Analyzing shifts for ROI '{roi}' / codebook '{codebook_name}'")
            shift_dir = ws.shifts(roi, codebook_name)
            if not shift_dir.exists():
                logger.warning(f"Shifts directory not found; skipping ROI {roi}: {shift_dir}")
                continue

            _check_missing_tiles(ws, roi, codebook_name, ref_round, shift_dir)
            shifts_by_tile = _load_shifts(shift_dir)
            if not shifts_by_tile:
                logger.warning(f"No shift JSON files found for ROI {roi}: {shift_dir}")
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
