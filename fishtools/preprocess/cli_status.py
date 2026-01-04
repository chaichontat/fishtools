from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import rich_click as click
from rich.console import Console
from rich.table import Table

from fishtools.io.workspace import Workspace


@dataclass
class StageStatus:
    complete: bool = False
    partial: bool = False
    count: int = 0
    expected: int | None = None
    details: str = ""
    last_modified: float | None = None
    stale: bool = False

    def to_cell(self, verbose: bool = False) -> str:
        if self.count == 0:
            return "[dim]-[/dim]"
        symbol = "[green]✓[/green]" if self.complete else "[yellow]⧖[/yellow]"
        stale_mark = "[red]![/red]" if self.stale else ""
        if not verbose:
            return f"{symbol}{stale_mark}"
        if self.expected and self.expected > 0:
            return f"{symbol}{stale_mark} {self.count}/{self.expected}"
        return f"{symbol}{stale_mark} {self.count}"


@dataclass
class ROIStatus:
    roi: str
    raw: StageStatus = field(default_factory=StageStatus)
    deconv: StageStatus = field(default_factory=StageStatus)
    register: StageStatus = field(default_factory=StageStatus)
    stitch_register: StageStatus = field(default_factory=StageStatus)
    stitch_fuse: StageStatus = field(default_factory=StageStatus)
    stitch_combine: StageStatus = field(default_factory=StageStatus)
    n4: StageStatus = field(default_factory=StageStatus)
    spots_decode: StageStatus = field(default_factory=StageStatus)
    spots_stitch: StageStatus = field(default_factory=StageStatus)
    segment: StageStatus = field(default_factory=StageStatus)
    postproc: StageStatus = field(default_factory=StageStatus)
    overlay_spots: StageStatus = field(default_factory=StageStatus)
    overlay_intensity: StageStatus = field(default_factory=StageStatus)
    export: StageStatus = field(default_factory=StageStatus)


_TILE_TIF_RE = re.compile(r".+-\d{4}\.tif$")


def _mtime(path: Path) -> float | None:
    mtime = path.stat().st_mtime
    return mtime if mtime > 0 else None


def _last_modified(paths: list[Path]) -> float | None:
    if not paths:
        return None

    max_mtime = 0.0
    for p in paths:
        max_mtime = max(max_mtime, p.stat().st_mtime)
    return max_mtime if max_mtime > 0 else None


def _find_roi_dirs(root: Path, roi: str, *, exclude_prefixes: tuple[str, ...]) -> list[Path]:
    dirs: list[Path] = []
    for d in root.iterdir():
        if not d.is_dir():
            continue
        if f"--{roi}" not in d.name:
            continue
        if d.name.startswith(exclude_prefixes):
            continue
        dirs.append(d)
    return dirs


def _glob_tifs(dirs: list[Path], *, name_re: re.Pattern[str] | None = None) -> list[Path]:
    tiles: list[Path] = []
    for d in dirs:
        for tile in d.glob("*.tif"):
            if name_re is not None and not name_re.match(tile.name):
                continue
            tiles.append(tile)
    return tiles


def _stage_dict(
    stage: StageStatus, *, include_count: bool = True, include_expected: bool = False
) -> dict[str, bool | int | None]:
    payload: dict[str, bool | int | None] = {"complete": stage.complete, "stale": stage.stale}
    if include_count:
        payload["count"] = stage.count
    if include_expected:
        payload["expected"] = stage.expected
    return payload


def check_raw_tiles(ws: Workspace, roi: str) -> StageStatus:
    raw_dirs = _find_roi_dirs(ws.path, roi, exclude_prefixes=("registered", "stitch", "shifts", "fids", "analysis"))
    if not raw_dirs:
        return StageStatus()

    tiles = _glob_tifs(raw_dirs)
    return StageStatus(
        complete=len(tiles) > 0,
        partial=False,
        count=len(tiles),
        expected=None,
        details=f"{len(raw_dirs)} rounds",
        last_modified=_last_modified(tiles),
    )


def check_deconv(ws: Workspace, roi: str) -> StageStatus:
    if not ws.deconved.exists():
        return StageStatus()

    deconv_dirs = _find_roi_dirs(
        ws.deconved, roi, exclude_prefixes=("registered", "stitch", "shifts", "fids", "segment", "opt")
    )
    if not deconv_dirs:
        return StageStatus()

    tiles = _glob_tifs(deconv_dirs, name_re=_TILE_TIF_RE)
    return StageStatus(
        complete=len(tiles) > 0,
        partial=False,
        count=len(tiles),
        expected=None,
        details=f"{len(deconv_dirs)} rounds",
        last_modified=_last_modified(tiles),
    )


def check_registration(ws: Workspace, roi: str, codebook: str) -> StageStatus:
    reg_path = ws.registered(roi, codebook)
    if not reg_path.exists():
        return StageStatus()

    reg_files = list(reg_path.glob("reg-*.tif"))
    count = len(reg_files)
    last_modified = _last_modified(reg_files)

    # Try to determine expected count from shifts or deconv
    expected = None
    shifts_path = ws.shifts(roi, codebook)
    if shifts_path.exists():
        shift_files = list(shifts_path.glob("shifts-*.json"))
        if shift_files:
            expected = len(shift_files)

    if expected is None:
        # Try to count from raw/deconv
        raw_status = check_raw_tiles(ws, roi)
        if raw_status.count > 0:
            # Estimate tiles per round
            deconv_status = check_deconv(ws, roi)
            if deconv_status.count > 0 and "rounds" in deconv_status.details:
                n_rounds = int(deconv_status.details.split()[0])
                if n_rounds > 0:
                    expected = deconv_status.count // n_rounds

    complete = count > 0 and (expected is None or count >= expected)
    partial = count > 0 and expected is not None and count < expected

    return StageStatus(
        complete=complete,
        partial=partial,
        count=count,
        expected=expected,
        last_modified=last_modified,
    )


def check_stitch_register(ws: Workspace, roi: str) -> StageStatus:
    tileconfig_path = ws.tileconfig_registered_txt(roi)
    if not tileconfig_path.exists():
        return StageStatus()

    return StageStatus(complete=True, count=1, last_modified=_mtime(tileconfig_path))


def check_stitch_fuse(ws: Workspace, roi: str, codebook: str) -> StageStatus:
    stitch_path = ws.stitch(roi, codebook)
    if not stitch_path.exists():
        return StageStatus()

    # Look for fused*.tif in channel subdirectories
    fused_files = list(stitch_path.rglob("fused*.tif"))
    # Also check for channel directories like 00/, 01/, etc.
    channel_dirs = [d for d in stitch_path.iterdir() if d.is_dir() and d.name.isdigit()]

    count = len(fused_files)
    if count == 0:
        return StageStatus()

    return StageStatus(
        complete=count > 0,
        count=count,
        details=f"{len(channel_dirs)} ch" if channel_dirs else "",
        last_modified=_last_modified(fused_files),
    )


def check_stitch_combine(ws: Workspace, roi: str, codebook: str) -> StageStatus:
    zarr_path = ws.stitch(roi, codebook) / "fused.zarr"
    if not zarr_path.exists():
        return StageStatus()

    # Verify it's a valid zarr (check for .zarray, .zgroup, or zarr.json for v3)
    is_valid = (
        (zarr_path / ".zarray").exists()
        or (zarr_path / ".zgroup").exists()
        or (zarr_path / "zarr.json").exists()
        or any(zarr_path.glob("*/.zarray"))  # Check subdirectories
    )
    if not is_valid:
        return StageStatus(partial=True, count=1, details="invalid zarr")

    return StageStatus(complete=True, count=1, last_modified=_mtime(zarr_path))


def check_n4(ws: Workspace, roi: str, codebook: str) -> StageStatus:
    n4_path = ws.stitch(roi, codebook) / "fused_n4.zarr"
    if not n4_path.exists():
        return StageStatus()

    return StageStatus(complete=True, count=1, last_modified=_mtime(n4_path))


def check_spots_decode(ws: Workspace, roi: str, codebook: str) -> StageStatus:
    decoded_path = ws.registered(roi, codebook) / f"decoded-{codebook}"
    if not decoded_path.exists():
        return StageStatus()

    pkl_files = list(decoded_path.glob("reg-*.pkl"))
    count = len(pkl_files)

    if count == 0:
        return StageStatus()

    last_modified = _last_modified(pkl_files)

    # Compare to registered tile count
    reg_status = check_registration(ws, roi, codebook)
    expected = reg_status.count * 4 if reg_status.count > 0 else None  # 4 quadrants per tile

    complete = expected is None or count >= expected
    partial = expected is not None and count < expected

    return StageStatus(
        complete=complete,
        partial=partial,
        count=count,
        expected=expected,
        last_modified=last_modified,
    )


def check_spots_stitch(ws: Workspace, roi: str, codebook: str) -> StageStatus:
    try:
        parquet_path = ws.spots_parquet(roi, codebook, must_exist=True)
        return StageStatus(complete=True, count=1, last_modified=_mtime(parquet_path))
    except FileNotFoundError:
        return StageStatus()


def check_segmentation(ws: Workspace, roi: str, codebook: str) -> StageStatus:
    stitch_path = ws.stitch(roi, codebook)
    if not stitch_path.exists():
        return StageStatus()

    seg_zarrs = [z for z in stitch_path.glob("output_segmentation*.zarr") if z.is_dir()]
    if not seg_zarrs:
        seg_path = ws.segment(roi, codebook)
        if seg_path.exists():
            seg_zarrs = [z for z in seg_path.glob("*.zarr") if z.is_dir()]
    if not seg_zarrs:
        return StageStatus()

    return StageStatus(complete=True, count=len(seg_zarrs), last_modified=_last_modified(seg_zarrs))


def check_postproc(ws: Workspace, roi: str, codebook: str) -> StageStatus:
    """Check for post-processed segmentation zarrs (output_segmentation*_postproc*.zarr)."""
    stitch_path = ws.stitch(roi, codebook)
    postproc_files: list[Path] = []

    if stitch_path.exists():
        postproc_files.extend(stitch_path.glob("output_segmentation*_postproc*.zarr"))

    seg_path = ws.segment(roi, codebook)
    if seg_path.exists():
        postproc_files.extend(seg_path.glob("output_segmentation*_postproc*.zarr"))
        postproc_files.extend(seg_path.glob("*_postproc*.zarr"))

    # Filter to only directories (valid zarrs)
    postproc_files = [f for f in postproc_files if f.is_dir()]

    if not postproc_files:
        return StageStatus()

    return StageStatus(complete=True, count=len(postproc_files), last_modified=_last_modified(postproc_files))


def _find_seg_zarrs(ws: Workspace, roi: str, codebook: str) -> list[Path]:
    """Find all segmentation zarr directories for a given ROI and codebook."""
    stitch_path = ws.stitch(roi, codebook)
    seg_zarrs: list[Path] = []
    if stitch_path.exists():
        seg_zarrs.extend(stitch_path.glob("output_segmentation*.zarr"))

    seg_path = ws.segment(roi, codebook)
    if seg_path.exists():
        seg_zarrs.extend(seg_path.glob("output_segmentation*.zarr"))
        seg_zarrs.extend(seg_path.glob("*.zarr"))
    return [z for z in seg_zarrs if z.is_dir()]


def check_overlay_spots(ws: Workspace, roi: str, codebook: str) -> StageStatus:
    """Check for overlay spots outputs (ident_*.parquet and polygons_*.parquet)."""
    seg_zarrs = _find_seg_zarrs(ws, roi, codebook)
    if not seg_zarrs:
        return StageStatus()

    ident_files: list[Path] = []
    polygon_files: list[Path] = []

    for seg_zarr in seg_zarrs:
        chunks_dir = seg_zarr / f"chunks+{codebook}"
        if chunks_dir.exists():
            ident_files.extend(chunks_dir.glob("ident_*.parquet"))
            polygon_files.extend(chunks_dir.glob("polygons_*.parquet"))

    if not ident_files and not polygon_files:
        return StageStatus()

    # Complete if both types exist and counts match
    complete = len(ident_files) > 0 and len(ident_files) == len(polygon_files)
    partial = (len(ident_files) > 0 or len(polygon_files) > 0) and not complete

    return StageStatus(
        complete=complete,
        partial=partial,
        count=len(ident_files),
        expected=len(polygon_files) if polygon_files else None,
        last_modified=_last_modified(ident_files + polygon_files),
    )


def check_overlay_intensity(ws: Workspace, roi: str, codebook: str) -> StageStatus:
    """Check for overlay intensity outputs (intensity_*/intensity-*.parquet)."""
    seg_zarrs = _find_seg_zarrs(ws, roi, codebook)
    stitch_path = ws.stitch(roi, codebook)

    parquet_files: list[Path] = []
    channel_dirs: set[str] = set()

    def _collect_intensity_parquets(root: Path) -> None:
        for intensity_dir in root.glob("intensity_*"):
            if not intensity_dir.is_dir():
                continue
            channel_dirs.add(intensity_dir.name.removeprefix("intensity_"))
            parquet_files.extend(intensity_dir.glob("intensity-*.parquet"))

    for seg_zarr in seg_zarrs:
        _collect_intensity_parquets(seg_zarr)

    # Legacy layout: sibling directories under stitch folder
    if stitch_path.exists():
        _collect_intensity_parquets(stitch_path)

    if not parquet_files:
        return StageStatus()

    details = f"{len(channel_dirs)} ch" if channel_dirs else ""
    return StageStatus(
        complete=True,
        count=len(parquet_files),
        details=details,
        last_modified=_last_modified(parquet_files),
    )


def check_export(ws: Workspace, roi: str, codebook: str) -> StageStatus:
    """Check for export outputs ({codebook}.h5ad or all+*.h5ad)."""
    seg_zarrs = _find_seg_zarrs(ws, roi, codebook)

    h5ad_files: list[Path] = []

    # Check inside segmentation zarrs for single-ROI exports
    for seg_zarr in seg_zarrs:
        h5ad_files.extend(seg_zarr.glob(f"{codebook}.h5ad"))
        h5ad_files.extend(seg_zarr.glob("*.h5ad"))

    # Check workspace output directory for multi-ROI exports
    if ws.output.exists():
        h5ad_files.extend(ws.output.glob(f"all+{codebook}+*.h5ad"))
        h5ad_files.extend(ws.output.glob(f"*+{codebook}+*.h5ad"))

    # Deduplicate
    h5ad_files = list(set(h5ad_files))

    if not h5ad_files:
        return StageStatus()

    return StageStatus(
        complete=True,
        count=len(h5ad_files),
        last_modified=_last_modified(h5ad_files),
    )


def mark_stale_stages(status: ROIStatus) -> None:
    # Dependency graph:
    # raw → deconv → register → stitch_register → stitch_fuse → stitch_combine → n4 → segment → postproc
    # register → spots_decode → spots_stitch
    # postproc + spots_stitch → overlay_spots
    # postproc → overlay_intensity
    # overlay_spots + overlay_intensity → export
    dependencies: list[tuple[StageStatus, list[StageStatus]]] = [
        (status.deconv, [status.raw]),
        (status.register, [status.deconv]),
        (status.stitch_register, [status.register]),
        (status.stitch_fuse, [status.stitch_register]),
        (status.stitch_combine, [status.stitch_fuse]),
        (status.n4, [status.stitch_combine]),
        (status.segment, [status.n4]),
        (status.postproc, [status.segment]),
        (status.spots_decode, [status.register]),
        (status.spots_stitch, [status.register, status.spots_decode]),
        (status.overlay_spots, [status.postproc, status.spots_stitch]),
        (status.overlay_intensity, [status.postproc]),
        (status.export, [status.overlay_spots, status.overlay_intensity]),
    ]

    for stage, upstreams in dependencies:
        if stage.last_modified is None:
            continue
        for upstream in upstreams:
            if upstream.last_modified is None:
                continue
            if upstream.last_modified > stage.last_modified:
                stage.stale = True
                break


def get_roi_status(ws: Workspace, roi: str, codebook: str) -> ROIStatus:
    status = ROIStatus(
        roi=roi,
        raw=check_raw_tiles(ws, roi),
        deconv=check_deconv(ws, roi),
        register=check_registration(ws, roi, codebook),
        stitch_register=check_stitch_register(ws, roi),
        stitch_fuse=check_stitch_fuse(ws, roi, codebook),
        stitch_combine=check_stitch_combine(ws, roi, codebook),
        n4=check_n4(ws, roi, codebook),
        spots_decode=check_spots_decode(ws, roi, codebook),
        spots_stitch=check_spots_stitch(ws, roi, codebook),
        segment=check_segmentation(ws, roi, codebook),
        postproc=check_postproc(ws, roi, codebook),
        overlay_spots=check_overlay_spots(ws, roi, codebook),
        overlay_intensity=check_overlay_intensity(ws, roi, codebook),
        export=check_export(ws, roi, codebook),
    )
    mark_stale_stages(status)
    return status


def is_spots_codebook(ws: Workspace, codebook: str, rois: list[str]) -> bool:
    for roi in rois:
        decoded_path = ws.registered(roi, codebook) / f"decoded-{codebook}"
        if decoded_path.exists():
            return True
    return False


def render_status_table(ws: Workspace, codebook: str, rois: list[str], *, verbose: bool = False) -> Table:
    is_spots = is_spots_codebook(ws, codebook, rois)

    table = Table(
        title=f"Codebook: {codebook}",
        show_header=True,
        header_style="bold",
        border_style="dim",
    )

    # Add columns based on codebook type
    table.add_column("ROI", style="cyan")
    table.add_column("Raw", justify="center")
    table.add_column("Deconv", justify="center")
    table.add_column("Register", justify="center")
    if is_spots:
        table.add_column("Spots", justify="center")
    else:
        table.add_column("Stitch", justify="center")
        table.add_column("Zarr", justify="center")
        table.add_column("N4", justify="center")
        table.add_column("Segment", justify="center")
        table.add_column("Postproc", justify="center")
        table.add_column("OvlSpots", justify="center")
        table.add_column("OvlInt", justify="center")
        table.add_column("Export", justify="center")

    for roi in rois:
        status = get_roi_status(ws, roi, codebook)
        row = [
            roi,
            status.raw.to_cell(verbose),
            status.deconv.to_cell(verbose),
            status.register.to_cell(verbose),
        ]
        if is_spots:
            spots_combined = f"{status.spots_decode.to_cell(verbose)}/{status.spots_stitch.to_cell(verbose)}"
            row.append(spots_combined)
        else:
            row.append(status.stitch_register.to_cell(verbose))
            row.append(status.stitch_combine.to_cell(verbose))
            row.append(status.n4.to_cell(verbose))
            row.append(status.segment.to_cell(verbose))
            row.append(status.postproc.to_cell(verbose))
            row.append(status.overlay_spots.to_cell(verbose))
            row.append(status.overlay_intensity.to_cell(verbose))
            row.append(status.export.to_cell(verbose))
        table.add_row(*row)

    return table


def status_to_dict(ws: Workspace, codebook: str, rois: list[str]) -> dict:
    result = {
        "workspace": str(ws.path),
        "codebook": codebook,
        "rois": {},
    }

    for roi in rois:
        status = get_roi_status(ws, roi, codebook)
        result["rois"][roi] = {
            "raw": _stage_dict(status.raw),
            "deconv": _stage_dict(status.deconv),
            "register": _stage_dict(status.register, include_expected=True),
            "stitch_register": _stage_dict(status.stitch_register, include_count=False),
            "stitch_fuse": _stage_dict(status.stitch_fuse),
            "stitch_combine": _stage_dict(status.stitch_combine, include_count=False),
            "n4": _stage_dict(status.n4, include_count=False),
            "spots_decode": _stage_dict(status.spots_decode, include_expected=True),
            "spots_stitch": _stage_dict(status.spots_stitch, include_count=False),
            "segment": _stage_dict(status.segment),
            "postproc": _stage_dict(status.postproc),
            "overlay_spots": _stage_dict(status.overlay_spots),
            "overlay_intensity": _stage_dict(status.overlay_intensity),
            "export": _stage_dict(status.export),
        }

    return result


@click.command("status")
@click.argument("path", type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path))
@click.option(
    "--codebook",
    "-c",
    type=str,
    default=None,
    help="Filter by specific codebook (auto-discovers all if not specified).",
)
@click.option(
    "--roi",
    "-r",
    "roi_filter",
    type=str,
    default=None,
    help="Filter by specific ROI (shows all if not specified).",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Show file counts in addition to status symbols.",
)
@click.option(
    "--json",
    "output_json",
    is_flag=True,
    help="Output as JSON for programmatic use.",
)
def status(path: Path, codebook: str | None, roi_filter: str | None, verbose: bool, output_json: bool) -> None:
    """Check preprocessing pipeline status for each ROI.

    Shows the completion status of each pipeline stage:
    - Raw: Raw imaging tiles in workspace
    - Deconv: Deconvolved tiles in analysis/deconv
    - Register: Registered tiles with codebook
    - Stitch: TileConfiguration.registered.txt
    - Fuse: Per-channel fused TIFFs
    - Combine: Combined fused.zarr
    - N4: N4 bias-field corrected zarr
    - Spots: Decoded spot pickles
    - Parquet: Final spots parquet
    - Segment: Segmentation zarr
    - Postproc: Post-processed segmentation zarr
    - OvlSpots: Overlay spots (ident/polygons parquets)
    - OvlInt: Overlay intensity (intensity parquets)
    - Export: Final h5ad export

    Symbols: ✓ Complete | ⧖ Partial | - Not started
    """
    try:
        ws = Workspace(path)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    # Discover ROIs
    try:
        rois = ws.resolve_rois([roi_filter] if roi_filter else None)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    # Filter out malformed ROIs (those containing "--" which are likely shifted/special directories)
    rois = [roi for roi in rois if "--" not in roi]

    if not rois:
        raise click.ClickException("No ROIs found in workspace.")

    # Discover codebooks
    if codebook:
        codebooks = [codebook]
    else:
        codebooks = ws.registered_codebooks(rois=rois)
        if not codebooks:
            # No registered codebooks, show just raw/deconv status
            codebooks = ["(none)"]

    console = Console()

    if output_json:
        # JSON output
        all_results = []
        for cb in codebooks:
            if cb == "(none)":
                continue
            all_results.append(status_to_dict(ws, cb, rois))
        click.echo(json.dumps(all_results, indent=2))
        return

    # Rich table output
    console.print(f"\n[bold]Workspace:[/bold] {ws.path}\n")

    has_spots_codebook = any(
        cb != "(none)" and is_spots_codebook(ws, cb, rois)
        for cb in codebooks
    )

    for cb in codebooks:
        if cb == "(none)":
            # Show raw/deconv only table
            table = Table(
                title="Pre-registration status",
                show_header=True,
                header_style="bold",
                border_style="dim",
            )
            table.add_column("ROI", style="cyan")
            table.add_column("Raw", justify="center")
            table.add_column("Deconv", justify="center")

            for roi in rois:
                raw_status = check_raw_tiles(ws, roi)
                deconv_status = check_deconv(ws, roi)
                table.add_row(roi, raw_status.to_cell(verbose), deconv_status.to_cell(verbose))

            console.print(table)
        else:
            table = render_status_table(ws, cb, rois, verbose=verbose)
            console.print(table)

        console.print()

    console.print("[dim]Legend: [green]✓[/green] Complete | [yellow]⧖[/yellow] Partial | - Not started | [red]![/red] Stale (upstream newer)[/dim]\n")
    if has_spots_codebook:
        console.print(
            "[dim]Spots column is `stitch/threshold`; run `preprocess spots stitch` for the first value and "
            "`preprocess spots threshold` for the second.[/dim]\n"
        )
