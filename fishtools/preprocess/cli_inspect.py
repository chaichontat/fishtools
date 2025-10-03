from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import rich_click as click
from rich.console import Console
from rich.pretty import Pretty
from rich.table import Table
from tifffile import TiffFile


def _format_bytes(size: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    value = float(size)

    for unit in units:
        if value < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.2f} {unit}"
        value /= 1024

    return f"{int(size)} B"


@click.command("inspect")
@click.argument("tiff_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def inspect_cli(tiff_path: Path) -> None:
    """Inspect a TIFF file and display basic metadata."""
    console = Console(force_terminal=False, color_system=None)
    file_size = tiff_path.stat().st_size

    shape: tuple[int, ...] | None = None
    dtype: str | None = None
    shaped_metadata: tuple[Any, ...] | None = None
    parsed_metadata: Any = None

    with TiffFile(tiff_path) as tif:
        if tif.series:
            series = tif.series[0]
            shape = tuple(series.shape)
            dtype = str(series.dtype)
        metadata = tif.shaped_metadata
        if metadata is not None:
            shaped_metadata = tuple(metadata)
            parsed_metadata = _parse_json_like(shaped_metadata)

    overview = Table(title="TIFF Overview", show_header=False, box=None)
    overview.add_column("field", style="bold")
    overview.add_column("value")
    overview.add_row("path", str(tiff_path.resolve()))
    overview.add_row("file size", _format_bytes(file_size))
    overview.add_row("shape", str(shape) if shape else "unknown")
    overview.add_row("dtype", dtype or "unknown")

    console.print(overview)

    if shaped_metadata is None or not shaped_metadata:
        console.print("[italic]No shaped metadata present.[/italic]")
        return

    console.print("[bold]Shaped Metadata[/bold]")
    if isinstance(parsed_metadata, (dict, list)):
        console.print_json(data=parsed_metadata)
    else:
        console.print(Pretty(parsed_metadata or shaped_metadata, overflow="ellipsis", indent_guides=True))


def _parse_json_like(value: Any) -> Any:
    """Recursively attempt to parse JSON strings contained in shaped metadata."""

    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return value
        return _parse_json_like(parsed)

    if isinstance(value, dict):
        return {k: _parse_json_like(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [_parse_json_like(v) for v in value]

    return value
