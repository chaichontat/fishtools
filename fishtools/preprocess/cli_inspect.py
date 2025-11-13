from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import rich_click as click
from rich.console import Console
from rich.pretty import Pretty
from rich.table import Table
from tifffile import TiffFile


def _is_zarr_path(p: Path) -> bool:
    """Return True if the given path points to a Zarr store.

    Accept either a directory that ends with ".zarr" or a file with the
    ".zarr" suffix. This mirrors detection used elsewhere in the project.
    """
    return p.suffix == ".zarr" or (p.is_dir() and p.name.endswith(".zarr"))


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
@click.argument("path", type=click.Path(exists=True, dir_okay=True, path_type=Path))
def inspect_cli(path: Path) -> None:
    """Inspect a TIFF file or a .zarr store and display metadata."""
    console = Console(force_terminal=False, color_system=None)

    if _is_zarr_path(path):
        _inspect_zarr(console, path)
        return

    if path.is_file():
        _inspect_tiff(console, path)
        return

    raise click.ClickException("Input must be a TIFF file or a .zarr store.")


def _inspect_tiff(console: Console, tiff_path: Path) -> None:
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


def _inspect_zarr(console: Console, zarr_path: Path) -> None:
    try:
        import zarr
    except Exception as exc:  # pragma: no cover - import failures vary
        raise click.ClickException(f"zarr is required to inspect .zarr stores: {exc}")

    # Compute on-disk size for directories; fall back to file size when applicable.
    file_size = _path_size(zarr_path)

    try:
        arr = zarr.open_array(str(zarr_path), mode="r")
    except Exception as exc:
        raise click.ClickException(f"Failed to open Zarr array at {zarr_path}: {exc}")

    shape = tuple(arr.shape)
    dtype = str(arr.dtype)
    chunks = getattr(arr, "chunks", None)

    # Compressor/codec info across zarr versions
    compressor = None
    try:
        compressor = arr.compressor  # v2 arrays; may raise on v3
    except Exception:
        compressor = None
    try:
        codec = getattr(arr, "codec", None)  # v3 arrays
    except Exception:
        codec = None
    try:
        filters = arr.filters  # v2/v3 optional; may raise
    except Exception:
        filters = None

    overview = Table(title="Zarr Overview", show_header=False, box=None)
    overview.add_column("field", style="bold")
    overview.add_column("value")
    overview.add_row("path", str(zarr_path.resolve()))
    overview.add_row("store size", _format_bytes(file_size))
    overview.add_row("shape", str(shape))
    overview.add_row("dtype", dtype)
    if chunks is not None:
        overview.add_row("chunks", str(tuple(chunks)))
    if compressor is not None:
        overview.add_row("compressor", str(compressor))
    if codec is not None and compressor is None:  # v3 terminology
        overview.add_row("codec", str(codec))
    if filters is not None:
        overview.add_row("filters", str(filters))

    console.print(overview)

    attrs = dict(getattr(arr, "attrs", {}) or {})
    if not attrs:
        console.print("[italic]No Zarr attributes present.[/italic]")
        return

    console.print("[bold]Zarr Attributes[/bold]")
    try:
        console.print_json(data=attrs)
    except Exception:
        console.print(Pretty(attrs, overflow="ellipsis", indent_guides=True))


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


def _path_size(p: Path) -> int:
    """Return on-disk size in bytes for a file or directory.

    For Zarr directories, sums contained file sizes without following symlinks.
    """
    try:
        if p.is_file():
            return p.stat().st_size
        if p.is_dir():
            total = 0

            def _scan(d: Path) -> int:
                t = 0
                try:
                    with os.scandir(d) as it:
                        for entry in it:
                            try:
                                if entry.is_file(follow_symlinks=False):
                                    t += entry.stat(follow_symlinks=False).st_size
                                elif entry.is_dir(follow_symlinks=False):
                                    t += _scan(Path(entry.path))
                            except OSError:
                                continue
                except OSError:
                    return 0
                return t

            total = _scan(p)
            return total
    except OSError:
        return 0
    return 0
