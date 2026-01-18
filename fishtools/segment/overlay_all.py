"""Composite overlay command that runs spots then intensity."""

from pathlib import Path

import rich_click as click

from fishtools.io.workspace import Workspace
from fishtools.segment.utils import StitchPaths, resolve_intensity_store


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
    "--fused-name",
    type=str,
    default="fused.zarr",
    show_default=True,
    help="Filename of the intensity Zarr inside stitch--ROI+<codebook> (e.g., fused.zarr, fused_n4.zarr, fused_highpassed.zarr).",
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
@click.option(
    "--erode",
    type=click.IntRange(min=0),
    default=2,
    show_default=True,
    help="Erode segmentation labels by N pixels before measuring intensity (2D only; set 0 to disable).",
)
@click.option(
    "--export",
    "export_opt",
    is_flag=True,
    default=False,
    help="Run `segment export` (as a subprocess) after overlays complete.",
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
    fused_name: str,
    channel: str | None,
    threads: int,
    erode: int,
    export_opt: bool,
    overwrite: bool,
    debug: bool,
) -> None:
    """Run spots and intensity overlays sequentially for one or more ROIs."""

    from fishtools.segment.overlay_intensity import overlay_intensity
    from fishtools.segment.overlay_spots import overlay as overlay_spots

    workspace = Workspace(path)
    roi_token = (roi or "*").strip()
    batch_mode = roi_token == "*" or roi_token.lower() == "all"
    try:
        rois = workspace.resolve_rois() if batch_mode else workspace.resolve_rois([roi_token])
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    if not rois:
        raise click.ClickException(f"No ROIs discovered under workspace {path}.")

    seg_cb = seg_codebook or codebook
    if seg_codebook is None:
        click.echo(f"Segmentation codebook not provided; defaulting to --codebook '{codebook}'.", err=True)

    did_overlay_any = False
    for current_roi in rois:
        stitch = StitchPaths.from_workspace(workspace, current_roi, seg_cb)
        seg_path = stitch.segmentation(segmentation_name)
        if not seg_path.exists():
            msg = f"Skipping ROI '{current_roi}': segmentation Zarr not found at {seg_path}"
            if batch_mode:
                click.echo(msg, err=True)
                continue
            raise click.ClickException(msg)

        try:
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
        except Exception as exc:
            if batch_mode:
                click.echo(f"Skipping ROI '{current_roi}': overlay spots failed: {exc}", err=True)
                continue
            raise

        try:
            resolve_intensity_store(stitch, intensity_codebook, store_name=fused_name)
        except FileNotFoundError as exc:
            if batch_mode:
                click.echo(f"Skipping ROI '{current_roi}': {exc}", err=True)
                continue
            raise click.ClickException(str(exc)) from exc

        try:
            overlay_intensity.callback(
                path=path,
                roi=current_roi,
                seg_codebook=seg_cb,
                intensity_codebook=intensity_codebook,
                segmentation_name=segmentation_name,
                fused_name=fused_name,
                erode=erode,
                channel=channel,
                threads=threads,
                overwrite=overwrite,
            )
            did_overlay_any = True
        except Exception as exc:
            if batch_mode:
                click.echo(f"Skipping ROI '{current_roi}': overlay intensity failed: {exc}", err=True)
                continue
            raise

    if export_opt and did_overlay_any:
        export_channels = channel if channel is not None else "auto"
        try:
            from fishtools.segment.export import export_cmd as segment_export_cmd

            segment_export_cmd(
                path=path,
                roi=None if batch_mode else roi_token,
                seg_codebook=seg_cb,
                codebooks=(codebook,),
                segmentation_name=segmentation_name,
                channels=export_channels,
            )
        except ValueError as exc:
            raise click.ClickException(str(exc)) from exc
    elif export_opt and not did_overlay_any:
        click.echo("No overlays completed successfully; skipping export.", err=True)


if __name__ == "__main__":
    overlay_all()
