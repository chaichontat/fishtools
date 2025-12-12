"""CLI command to check preprocessing pipeline status for each ROI."""

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
    """Status of a single pipeline stage."""

    complete: bool = False
    partial: bool = False
    count: int = 0
    expected: int | None = None
    details: str = ""
    last_modified: float | None = None
    stale: bool = False

    def to_cell(self, verbose: bool = False) -> str:
        """Format for rich table display."""
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
    """Status of all pipeline stages for a single ROI."""

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


def check_raw_tiles(ws: Workspace, roi: str) -> StageStatus:
    """Check for raw tiles in workspace root."""
    raw_dirs = []
    for d in ws.path.iterdir():
        if not d.is_dir():
            continue
        if f"--{roi}" not in d.name:
            continue
        # Skip processed directories
        if d.name.startswith(("registered", "stitch", "shifts", "fids", "analysis")):
            continue
        raw_dirs.append(d)

    if not raw_dirs:
        return StageStatus()

    total_tiles = 0
    max_mtime = 0.0
    for d in raw_dirs:
        tiles = list(d.glob("*.tif"))
        total_tiles += len(tiles)
        for tile in tiles:
            max_mtime = max(max_mtime, tile.stat().st_mtime)

    return StageStatus(
        complete=total_tiles > 0,
        partial=False,
        count=total_tiles,
        expected=None,
        details=f"{len(raw_dirs)} rounds",
        last_modified=max_mtime if max_mtime > 0 else None,
    )


def check_deconv(ws: Workspace, roi: str) -> StageStatus:
    """Check for deconvolved tiles in analysis/deconv."""
    deconv_dirs = []
    if not ws.deconved.exists():
        return StageStatus()

    for d in ws.deconved.iterdir():
        if not d.is_dir():
            continue
        if f"--{roi}" not in d.name:
            continue
        # Only match {round}--{roi} pattern, not registered/stitch/etc
        if d.name.startswith(("registered", "stitch", "shifts", "fids", "segment", "opt")):
            continue
        deconv_dirs.append(d)

    if not deconv_dirs:
        return StageStatus()

    total_tiles = 0
    max_mtime = 0.0
    for d in deconv_dirs:
        # Match {round}-NNNN.tif pattern
        tiles = [f for f in d.glob("*.tif") if re.match(r".+-\d{4}\.tif$", f.name)]
        total_tiles += len(tiles)
        for tile in tiles:
            max_mtime = max(max_mtime, tile.stat().st_mtime)

    return StageStatus(
        complete=total_tiles > 0,
        partial=False,
        count=total_tiles,
        expected=None,
        details=f"{len(deconv_dirs)} rounds",
        last_modified=max_mtime if max_mtime > 0 else None,
    )


def check_registration(ws: Workspace, roi: str, codebook: str) -> StageStatus:
    """Check for registered tiles."""
    reg_path = ws.registered(roi, codebook)
    if not reg_path.exists():
        return StageStatus()

    reg_files = list(reg_path.glob("reg-*.tif"))
    count = len(reg_files)

    max_mtime = 0.0
    for f in reg_files:
        max_mtime = max(max_mtime, f.stat().st_mtime)

    # Try to determine expected count from shifts or deconv
    expected = None
    shifts_path = ws.deconved / f"shifts--{roi}+{codebook}"
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
        last_modified=max_mtime if max_mtime > 0 else None,
    )


def check_stitch_register(ws: Workspace, roi: str) -> StageStatus:
    """Check for TileConfiguration.registered.txt."""
    tileconfig_path = ws.tileconfig_dir(roi) / "TileConfiguration.registered.txt"
    if not tileconfig_path.exists():
        return StageStatus()

    return StageStatus(complete=True, count=1, last_modified=tileconfig_path.stat().st_mtime)


def check_stitch_fuse(ws: Workspace, roi: str, codebook: str) -> StageStatus:
    """Check for fused per-channel TIFFs."""
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

    max_mtime = 0.0
    for f in fused_files:
        max_mtime = max(max_mtime, f.stat().st_mtime)

    return StageStatus(
        complete=count > 0,
        count=count,
        details=f"{len(channel_dirs)} ch" if channel_dirs else "",
        last_modified=max_mtime if max_mtime > 0 else None,
    )


def check_stitch_combine(ws: Workspace, roi: str, codebook: str) -> StageStatus:
    """Check for fused.zarr."""
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

    return StageStatus(complete=True, count=1, last_modified=zarr_path.stat().st_mtime)


def check_n4(ws: Workspace, roi: str, codebook: str) -> StageStatus:
    """Check for N4-corrected zarr."""
    n4_path = ws.stitch(roi, codebook) / "fused_n4.zarr"
    if not n4_path.exists():
        return StageStatus()

    return StageStatus(complete=True, count=1, last_modified=n4_path.stat().st_mtime)


def check_spots_decode(ws: Workspace, roi: str, codebook: str) -> StageStatus:
    """Check for decoded spot pickles."""
    decoded_path = ws.registered(roi, codebook) / f"decoded-{codebook}"
    if not decoded_path.exists():
        return StageStatus()

    pkl_files = list(decoded_path.glob("reg-*.pkl"))
    count = len(pkl_files)

    if count == 0:
        return StageStatus()

    max_mtime = 0.0
    for f in pkl_files:
        max_mtime = max(max_mtime, f.stat().st_mtime)

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
        last_modified=max_mtime if max_mtime > 0 else None,
    )


def check_spots_stitch(ws: Workspace, roi: str, codebook: str) -> StageStatus:
    """Check for final parquet output."""
    try:
        parquet_path = ws.spots_parquet(roi, codebook, must_exist=True)
        return StageStatus(complete=True, count=1, last_modified=parquet_path.stat().st_mtime)
    except FileNotFoundError:
        return StageStatus()


def check_segmentation(ws: Workspace, roi: str, codebook: str) -> StageStatus:
    """Check for segmentation zarr outputs."""
    stitch_path = ws.stitch(roi, codebook)
    if not stitch_path.exists():
        return StageStatus()

    seg_files = list(stitch_path.glob("output_segmentation*.zarr"))
    if not seg_files:
        # Also check segment directory
        seg_path = ws.segment(roi, codebook)
        if seg_path.exists():
            seg_files = list(seg_path.glob("*.zarr"))

    count = len(seg_files)
    if count == 0:
        return StageStatus()

    max_mtime = 0.0
    for f in seg_files:
        max_mtime = max(max_mtime, f.stat().st_mtime)

    return StageStatus(complete=True, count=count, last_modified=max_mtime if max_mtime > 0 else None)


def mark_stale_stages(status: ROIStatus) -> None:
    """Mark stages as stale if an upstream stage has a newer mtime.

    Dependency graph:
    - raw → deconv → register → stitch_register → stitch_fuse → stitch_combine → n4 → segment
    - register → spots_decode → spots_stitch
    """
    # (stage, [upstream dependencies])
    dependencies: list[tuple[StageStatus, list[StageStatus]]] = [
        (status.deconv, [status.raw]),
        (status.register, [status.deconv]),
        (status.stitch_register, [status.register]),
        (status.stitch_fuse, [status.stitch_register]),
        (status.stitch_combine, [status.stitch_fuse]),
        (status.n4, [status.stitch_combine]),
        (status.segment, [status.n4]),
        (status.spots_decode, [status.register]),
        (status.spots_stitch, [status.register, status.spots_decode]),
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
    """Get status of all pipeline stages for a single ROI+codebook."""
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
    )
    mark_stale_stages(status)
    return status


def is_spots_codebook(ws: Workspace, codebook: str, rois: list[str]) -> bool:
    """Check if codebook is spots-type (has decoded-* directory) vs intensity-type."""
    for roi in rois:
        decoded_path = ws.registered(roi, codebook) / f"decoded-{codebook}"
        if decoded_path.exists():
            return True
    return False


def render_status_table(ws: Workspace, codebook: str, rois: list[str], *, verbose: bool = False) -> Table:
    """Create rich Table for a codebook's status across ROIs."""
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
        table.add_row(*row)

    return table


def status_to_dict(ws: Workspace, codebook: str, rois: list[str]) -> dict:
    """Convert status to JSON-serializable dictionary."""
    result = {
        "workspace": str(ws.path),
        "codebook": codebook,
        "rois": {},
    }

    for roi in rois:
        status = get_roi_status(ws, roi, codebook)
        result["rois"][roi] = {
            "raw": {"count": status.raw.count, "complete": status.raw.complete, "stale": status.raw.stale},
            "deconv": {"count": status.deconv.count, "complete": status.deconv.complete, "stale": status.deconv.stale},
            "register": {
                "count": status.register.count,
                "expected": status.register.expected,
                "complete": status.register.complete,
                "stale": status.register.stale,
            },
            "stitch_register": {"complete": status.stitch_register.complete, "stale": status.stitch_register.stale},
            "stitch_fuse": {"count": status.stitch_fuse.count, "complete": status.stitch_fuse.complete, "stale": status.stitch_fuse.stale},
            "stitch_combine": {"complete": status.stitch_combine.complete, "stale": status.stitch_combine.stale},
            "n4": {"complete": status.n4.complete, "stale": status.n4.stale},
            "spots_decode": {
                "count": status.spots_decode.count,
                "expected": status.spots_decode.expected,
                "complete": status.spots_decode.complete,
                "stale": status.spots_decode.stale,
            },
            "spots_stitch": {"complete": status.spots_stitch.complete, "stale": status.spots_stitch.stale},
            "segment": {"count": status.segment.count, "complete": status.segment.complete, "stale": status.segment.stale},
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
        console.print(json.dumps(all_results, indent=2))
        return

    # Rich table output
    console.print(f"\n[bold]Workspace:[/bold] {ws.path}\n")

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
