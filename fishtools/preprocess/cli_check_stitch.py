from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import click
from click.core import ParameterSource
import matplotlib as mpl

# Force a non-interactive backend to avoid GUI/event-loop hangs in headless runs
mpl.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402  (import after backend selection)
import seaborn as sns  # noqa: E402
from loguru import logger

from fishtools.io.workspace import Workspace
from fishtools.plot.diagnostics.stitch import (
    StitchLayoutOptions,
    make_combined_stitch_layout,
    make_roi_stitch_layout,
)
from fishtools.preprocess.config_loader import load_config
from fishtools.preprocess.tileconfig import TileConfiguration
from fishtools.utils.logging import setup_cli_logging
from fishtools.utils.plot import scatter_spots

sns.set_theme()


if TYPE_CHECKING:
    import polars as pl


def _overlay_spots(ax: plt.Axes, spots: "pl.DataFrame", *, max_points: int = 200_000) -> None:
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    scatter_spots(ax, spots, x_col="plot_x", y_col="plot_y", max_points=max_points)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


def _load_spots_final(ws: Workspace, rois: list[str], codebook: str) -> dict[str, "pl.DataFrame"]:
    import polars as pl

    spots_by_roi: dict[str, pl.DataFrame] = {}
    for roi in rois:
        parquet_path = ws.spots_parquet(roi, codebook, must_exist=False)
        if not parquet_path.exists():
            logger.warning(f"Spots parquet not found for ROI {roi}: {parquet_path}")
            continue

        schema = pl.scan_parquet(parquet_path).collect_schema()
        if "y" in schema and "x" in schema:
            x_col = "x"
            y_col = "y"
        elif "y_" in schema and "x_" in schema:
            x_col = "x_"
            y_col = "y_"
        else:
            raise click.ClickException(
                f"Spots parquet {parquet_path} for ROI '{roi}' is missing required coordinates (x/y or x_/y_)."
            )

        df = pl.read_parquet(parquet_path, columns=[x_col, y_col]).select(
            plot_x=pl.col(x_col),
            plot_y=pl.col(y_col),
        )
        spots_by_roi[roi] = df
        logger.debug(f"ROI='{roi}': loaded spots_final parquet {parquet_path} (n={df.height:,})")

    return spots_by_roi


@click.command("check-stitch")
@click.argument(
    "path",
    type=click.Path(
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
        path_type=Path,
    ),
)
@click.option(
    "--roi",
    "rois",
    multiple=True,
    help="ROI(s) to include; defaults to all detected ROIs",
)
@click.option(
    "-o",
    "--output",
    "output_dir",
    type=click.Path(file_okay=False, dir_okay=True, writable=True, resolve_path=True, path_type=Path),
    default=None,
    help="Output directory [default: '<workspace>/analysis/output']",
)
@click.option("--cols", type=int, default=None, help="Grid columns; default sqrt(#ROIs)")
@click.option(
    "--config",
    "json_config",
    type=click.Path(exists=True, dir_okay=False, file_okay=True, readable=True, resolve_path=True, path_type=Path),
    default=None,
    help=(
        "Project config JSON; derives pixel size and tile size from config. "
        "Defaults to <workspace>/analysis/deconv/config.json then <workspace>/config.json when present."
    ),
)
@click.option(
    "--pixel-size-um",
    type=float,
    default=0.108,
    show_default=True,
    help="Pixel size in µm for axes",
)
@click.option(
    "--per-roi/--no-per-roi",
    default=False,
    show_default=True,
    help="Also save one PNG per ROI",
)
@click.option(
    "--label-skip",
    type=int,
    default=2,
    show_default=True,
    help="Label every other tile",
)
@click.option(
    "--tile-size-px",
    type=int,
    default=1968,
    show_default=True,
    help="Tile edge length in pixels. Coords are top-left; labels at centroid.",
)
@click.option(
    "--spots",
    "spots_codebook",
    default=None,
    metavar="CODEBOOK",
    help="Overlay spots_final from analysis/output/parquets (ROI+CODEBOOK parquet) on the tile layout.",
)
def check_stitch(
    path: Path,
    rois: list[str] | None,
    output_dir: Path | None,
    cols: int | None,
    json_config: Path | None,
    pixel_size_um: float,
    per_roi: bool,
    label_skip: int,
    tile_size_px: int,
    spots_codebook: str | None,
) -> None:
    """Visualize TileConfiguration layouts per ROI.

    Creates a combined panel PNG of all requested ROIs. Optionally also saves one
    per-ROI PNG. Tick labels are converted to micrometers using --pixel-size-um.
    """

    setup_cli_logging(path, component="preprocess.check_stitch", file="check-stitch", extra={})

    ws = Workspace(path)
    ctx = click.get_current_context()
    explicit_config = ctx.get_parameter_source("json_config") == ParameterSource.COMMANDLINE
    if json_config is None:
        json_config = ws.config_json()

    if json_config is not None:
        if explicit_config:
            if ctx.get_parameter_source("pixel_size_um") == ParameterSource.COMMANDLINE:
                raise click.ClickException(
                    "--pixel-size-um cannot be used with --config; set pixel_size_um in config."
                )
            if ctx.get_parameter_source("tile_size_px") == ParameterSource.COMMANDLINE:
                raise click.ClickException(
                    "--tile-size-px cannot be used with --config; tile size is derived as image_size - 2*crop."
                )

        cfg = load_config(json_config)
        if explicit_config or ctx.get_parameter_source("pixel_size_um") != ParameterSource.COMMANDLINE:
            pixel_size_um = cfg.pixel_size_um
        if explicit_config or ctx.get_parameter_source("tile_size_px") != ParameterSource.COMMANDLINE:
            tile_size_px = cfg.image_size - 2 * cfg.registration.crop
            if tile_size_px <= 0:
                raise click.ClickException(
                    f"Derived tile_size_px={tile_size_px} from image_size={cfg.image_size} and crop={cfg.registration.crop}."
                )

    logger.info(
        f"check-stitch: path={path}, rois={rois or 'ALL'}, cols={cols}, config={json_config}, "
        f"pixel_size_um={pixel_size_um}, "
        f"per_roi={per_roi}, label_skip={label_skip}, tile_size_px={tile_size_px}, spots={spots_codebook}"
    )
    roi_list = ws.resolve_rois(rois)
    if not roi_list:
        raise click.ClickException("No ROIs found.")

    if output_dir is None:
        output_dir = ws.output
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Output directory: {output_dir}")

    options = StitchLayoutOptions(
        pixel_size_um=pixel_size_um,
        label_skip=label_skip,
        tile_size_px=tile_size_px,
    )
    normalized_label_skip = max(1, label_skip)

    tileconfigs: dict[str, TileConfiguration | None] = {}
    for roi in roi_list:
        try:
            cfg_path = ws.tileconfig_registered_txt(roi)
            if not cfg_path.exists():
                raise FileNotFoundError(str(cfg_path))
            tc = TileConfiguration.from_file(cfg_path)
        except FileNotFoundError as exc:
            logger.warning(f"TileConfiguration not found for ROI {roi}: {exc}")
            tileconfigs[roi] = None
            continue
        tileconfigs[roi] = tc
        logger.debug(f"ROI='{roi}': loaded TileConfiguration at {cfg_path} (tiles={len(tc)})")

    if all(tc is None for tc in tileconfigs.values()):
        raise click.ClickException("No TileConfiguration files found for requested ROIs.")

    spots_by_roi = None
    if spots_codebook is not None:
        spots_by_roi = _load_spots_final(ws, roi_list, spots_codebook)
        if not spots_by_roi:
            logger.warning(f"No spots_final parquets loaded for codebook {spots_codebook}.")
            spots_by_roi = None

    stitch_output_dir = output_dir / "stitch_layout"
    stitch_output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes_by_roi = make_combined_stitch_layout(
        roi_list,
        tileconfigs,
        ncols=cols,
        options=options,
    )
    if spots_by_roi is not None:
        for roi, ax in axes_by_roi.items():
            spots = spots_by_roi.get(roi)
            if spots is None:
                continue
            _overlay_spots(ax, spots)
    combined = (stitch_output_dir / "stitch_layout_all.png").resolve()
    fig.savefig(combined.as_posix(), bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved combined stitch panel: {combined}")

    if per_roi:
        for roi in roi_list:
            tc = tileconfigs.get(roi)
            if tc is None:
                continue
            fig_roi = make_roi_stitch_layout(tc, roi, options=options)
            if spots_by_roi is not None and roi in spots_by_roi and fig_roi.axes:
                _overlay_spots(fig_roi.axes[0], spots_by_roi[roi])
            out = (stitch_output_dir / f"stitch_layout--{roi}.png").resolve()
            fig_roi.savefig(out.as_posix(), bbox_inches="tight")
            plt.close(fig_roi)
            total = int(len(tc))
            labeled = (total + normalized_label_skip - 1) // normalized_label_skip
            logger.info(f"Saved per-ROI stitch layout: {out} (tiles={total}, labeled={labeled})")
