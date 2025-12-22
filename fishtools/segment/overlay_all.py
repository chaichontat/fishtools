"""Composite overlay command that runs spots then intensity."""

from pathlib import Path

import rich_click as click

from fishtools.io.workspace import Workspace
from fishtools.segment.overlay_intensity import overlay_intensity
from fishtools.segment.overlay_spots import overlay as overlay_spots


@click.command()
@click.argument(
    "path",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    required=True,
)
@click.argument("roi", required=False)
@click.option(
    "--codebook",
    type=str,
    required=True,
    help="Codebook name used for spots.",
)
@click.option(
    "--seg-codebook",
    type=str,
    required=False,
    default=None,
    help="Segmentation codebook.",
)
@click.option(
    "--intensity-codebook",
    type=str,
    required=True,
    help="Codebook label used to resolve the intensity store.",
)
@click.option(
    "--spots",
    "spots_opt",
    type=click.Path(dir_okay=True, file_okay=True, path_type=Path),
    default=None,
    help=(
        "Path to spots parquet file, or directory with per-ROI spots parquets "
        "named '{roi}+{codebook}.parquet'. Defaults to <workspace>/analysis/output."
    ),
)
@click.option(
    "--segmentation-name",
    type=str,
    default="output_segmentation-sam.zarr",
    show_default=True,
    help="Relative path to the segmentation Zarr store within the input directory.",
)
@click.option(
    "--intensity-store",
    type=str,
    default="fused.zarr",
    show_default=True,
    help="Filename of the intensity Zarr inside stitch--ROI+<codebook>.",
)
@click.option(
    "--channel",
    type=str,
    help="Channel name. When omitted, all channels in attrs['key'] are processed.",
)
@click.option(
    "--threads",
    type=int,
    default=12,
    show_default=True,
    help="Number of parallel worker processes to use per ROI.",
)
@click.option("--overwrite", is_flag=True, default=False, help="Overwrite existing output files.")
@click.option("--debug", is_flag=True, default=False, help="Enable debug logging and generate debug plots.")
def overlay_all(
    path: Path,
    roi: str | None,
    codebook: str,
    seg_codebook: str | None,
    intensity_codebook: str,
    spots_opt: Path | None,
    segmentation_name: str,
    intensity_store: str,
    channel: str | None,
    threads: int,
    overwrite: bool,
    debug: bool,
) -> None:
    """Run spots and intensity overlays sequentially for one or more ROIs."""

    workspace = Workspace(path)
    target_roi = roi or "*"
    try:
        rois = workspace.resolve_rois() if target_roi == "*" else workspace.resolve_rois([target_roi])
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    if not rois:
        raise click.ClickException(f"No ROIs discovered under workspace {path}.")

    seg_cb = seg_codebook or codebook
    for current_roi in rois:
        overlay_spots.callback(
            path=path,
            roi=current_roi,
            codebook=codebook,
            spots_opt=spots_opt,
            seg_codebook=seg_codebook,
            segmentation_name=segmentation_name,
            overwrite=overwrite,
            debug=debug,
        )
        overlay_intensity.callback(
            path=path,
            roi=current_roi,
            seg_codebook=seg_cb,
            intensity_codebook=intensity_codebook,
            segmentation_name=segmentation_name,
            intensity_store=intensity_store,
            channel=channel,
            threads=threads,
            overwrite=overwrite,
        )


if __name__ == "__main__":
    overlay_all()
