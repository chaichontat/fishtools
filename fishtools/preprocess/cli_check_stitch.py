from __future__ import annotations

from pathlib import Path

import click
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
from fishtools.preprocess.tileconfig import TileConfiguration
from fishtools.utils.logging import setup_cli_logging

sns.set_theme()


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
    help="Output directory [default: '<workspace_parent>/output']",
)
@click.option("--cols", type=int, default=None, help="Grid columns; default sqrt(#ROIs)")
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
def check_stitch(
    path: Path,
    rois: list[str] | None,
    output_dir: Path | None,
    cols: int | None,
    pixel_size_um: float,
    per_roi: bool,
    label_skip: int,
    tile_size_px: int,
) -> None:
    """Visualize TileConfiguration layouts per ROI.

    Creates a combined panel PNG of all requested ROIs. Optionally also saves one
    per-ROI PNG. Tick labels are converted to micrometers using --pixel-size-um.
    """

    setup_cli_logging(path, component="preprocess.check_stitch", file="check-stitch", extra={})
    logger.info(
        f"check-stitch: path={path}, rois={rois or 'ALL'}, cols={cols}, pixel_size_um={pixel_size_um}, "
        f"per_roi={per_roi}, label_skip={label_skip}, tile_size_px={tile_size_px}"
    )

    ws = Workspace(path)
    roi_list = ws.resolve_rois(rois)
    if not roi_list:
        raise click.ClickException("No ROIs found.")

    if output_dir is None:
        output_dir = path.parent / "output"
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
        cfg_path = ws.deconved / f"stitch--{roi}" / "TileConfiguration.registered.txt"
        try:
            tc = ws.tileconfig(roi)
        except FileNotFoundError as exc:
            logger.warning(f"TileConfiguration not found for ROI {roi}: {exc}")
            tileconfigs[roi] = None
            continue
        tileconfigs[roi] = tc
        logger.debug(f"ROI='{roi}': loaded TileConfiguration at {cfg_path} (tiles={len(tc)})")

    if all(tc is None for tc in tileconfigs.values()):
        raise click.ClickException("No TileConfiguration files found for requested ROIs.")

    fig, _ = make_combined_stitch_layout(
        roi_list,
        tileconfigs,
        ncols=cols,
        options=options,
    )
    combined = (output_dir / "stitch_layout_all.png").resolve()
    fig.savefig(combined.as_posix(), bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved combined stitch panel: {combined}")

    if per_roi:
        for roi in roi_list:
            tc = tileconfigs.get(roi)
            if tc is None:
                continue
            fig_roi = make_roi_stitch_layout(tc, roi, options=options)
            out = (output_dir / f"stitch_layout--{roi}.png").resolve()
            fig_roi.savefig(out.as_posix(), bbox_inches="tight")
            plt.close(fig_roi)
            total = int(len(tc))
            labeled = (total + normalized_label_skip - 1) // normalized_label_skip
            logger.info(f"Saved per-ROI stitch layout: {out} (tiles={total}, labeled={labeled})")
