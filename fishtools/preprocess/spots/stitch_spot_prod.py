"""
Stitch decoded spot tiles into a per-ROI parquet with global coordinates.

Coordinate conventions used throughout this module:
- Each decoded tile provides spot locations in its own tile-local frame
  (``x_local``, ``y_local`` when loaded through ``fishtools.analysis.spots``).
- Using the tile configuration (ImageJ-style ``TileConfiguration.registered.txt``),
  we transform tile-local coordinates into global mosaic coordinates and
  materialize them as ``x`` and ``y`` in the output parquet.

This module does not create ``x_``/``y_`` columns. In this codebase, those
names are used in ``cli_spotlook`` for feature-space axes (not spatial
coordinates) during interactive thresholding.
"""

import re
from concurrent.futures import FIRST_EXCEPTION, ProcessPoolExecutor, wait
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import polars as pl
import rich_click as click
from loguru import logger
from rtree import index
from tifffile import imread

from fishtools.analysis.spots import load_spots, load_spots_simple
from fishtools.io.workspace import Workspace
from fishtools.preprocess.stitching import (
    Cells,
    Crosses,
    filter_spots_by_area,
    generate_cells,
    get_exclusive_area,
)
from fishtools.utils.logging import resolve_workspace_root, setup_cli_logging

FILENAME = re.compile(r"reg-(\d+)(?:-(\d+))?\.(pkl|parquet)")
OUT_SUFFIX = "_deduped.parquet"


def _setup_command_logging(
    path: Path,
    *,
    roi: str,
    codebook: Path,
    debug: bool = False,
    extra: dict[str, Any] | None = None,
) -> None:
    tag = f"stitch-spots-{roi}-{codebook.stem}"
    payload = {"roi": roi, "codebook": codebook.stem, **({} if extra is None else extra)}
    setup_cli_logging(path, component="preprocess.spots.stitch", file=tag, debug=debug, extra=payload)


def gen_out(path: Path):
    return path.with_name(path.stem + OUT_SUFFIX)


def _parse_filename(x: str):
    match = FILENAME.match(x)
    if not match:
        raise ValueError(f"Could not parse {x}")
    return {
        "idx": int(match.group(1)),
        "split": int(match.group(2)) if match.group(2) else None,
    }


def load_coords(path: Path, roi: str):
    coords = Workspace(path).tileconfig(roi).df
    if len(coords) != len(coords.unique(subset=["x", "y"], maintain_order=True)):
        logger.warning("Duplicates found in TileConfiguration.registered.txt")
        coords = coords.unique(subset=["x", "y"], maintain_order=True)
    return coords


def load_splits(
    path: Path,
    i: int,
    coords: pl.DataFrame,
    *,
    filter_: bool = True,
    size: int,
    cut: int = 1024,
    simple: bool = False,
):
    """
    Load one decoded split, add tile offset, and return a DataFrame containing
    both tile-local (``x_local``, ``y_local``) and global (``x``, ``y``)
    coordinates.

    The global coordinates incorporate the per-tile stage offsets from the
    tile configuration and, combined across all tiles, form a single mosaic
    coordinate system for the ROI.
    """
    idx, split = _parse_filename(path.name).values()
    c = coords.filter(pl.col("index") == idx).row(0, named=True)

    # logger.debug(f"Loading {path} as split {split}")
    if split == 0:  # top-left
        pass
    elif split == 1:  # top-right
        c["x"] += size - cut
    elif split == 2:  # bottom-left
        c["y"] += size - cut  # + to move down
    elif split == 3:  # bottom-right
        c["x"] += size - cut
        c["y"] += size - cut

    try:
        return (load_spots if not simple else load_spots_simple)(
            path, i, filter_=filter_, tile_coords=(c["x"], c["y"])
        )
    except pl.exceptions.ColumnNotFoundError:
        raise ValueError(f"No spots found in {path}")
    except Exception as e:
        msg = f"Error reading {path}: {e.__class__.__name__}. Please rerun."
        logger.critical(msg)
        raise Exception(msg) from e


def process_df(
    df: pl.DataFrame,
    position: int,
    coords: pl.DataFrame,  # to maintain type signature with process_pickle
    cells: Cells,
    crosses: Crosses,
) -> pl.DataFrame:
    """Process a single DataFrame for a specific region/position.

    Args:
        df: DataFrame containing spot data with x,y coordinates
        position: Index of the current cell being processed
        cells: List of all cell Polygons defining region boundaries
        crosses: Pre-calculated cell intersections. Will calculate if None
        filter_passes_thresholds: Whether to filter spots that don't pass thresholds
    """
    exclusive_area = get_exclusive_area(position, cells[position], cells, crosses)
    return filter_spots_by_area(df, exclusive_area)


def process(
    path_pickle: Path,
    curr: int,
    coords: pl.DataFrame,
    cells: Cells,
    crosses: Crosses,
    size: int,
    *,
    filter_: bool = True,
    simple: bool = False,
) -> pl.DataFrame | None:
    """Filter a decoded tile's spots to its exclusive area and write parquet.

    The input DataFrame carries both tile-local and global coordinates (see
    ``load_splits``). The written parquet preserves the global ``x``/``y`` for
    downstream stitching and visualization.
    """
    try:
        df = load_splits(path_pickle, curr, coords, size=size, filter_=filter_, simple=simple)
    except Exception as e:
        logger.error(f"{path_pickle.name}: {e}")
        return None
    (df_filtered := process_df(df, curr, coords, cells, crosses)).write_parquet(gen_out(path_pickle))

    logger.info(f"{path_pickle.name}: thrown {len(df) - len(df_filtered)} / {len(df)}.")
    logger.debug(f"Wrote to {gen_out(path_pickle)}")
    return df_filtered


@click.command()
@click.argument(
    "path_wd",
    type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=Path),
)
@click.argument("_roi", type=str, default="*")
@click.option(
    "--codebook",
    type=click.Path(exists=True, dir_okay=False, file_okay=True, path_type=Path),
)
@click.option("--no-filter", is_flag=True)
@click.option("--overwrite", is_flag=True)
@click.option("--simple", is_flag=True)
@click.option("--no-split", is_flag=True)
@click.option("--threads", type=int, default=8)
def stitch(
    path_wd: Path,
    _roi: str,
    codebook: Path,
    no_filter: bool = False,
    overwrite: bool = False,
    simple: bool = False,
    no_split: bool = False,
    threads: int = 8,
):
    """
    Consolidate decoded per-tile spot files into ROI-level parquet outputs.

    Output files at ``analysis/deconv/registered--{roi}+{codebook}/decoded-{codebook}/``
    contain global mosaic coordinates in columns ``x`` and ``y``. There are no
    ``x_``/``y_`` columns here; those names are reserved by ``cli_spotlook`` for
    non-spatial, feature-space plots.
    """
    workspace_root, _ = resolve_workspace_root(path_wd)
    roi_label = _roi if _roi not in {"*", "all"} else "all"
    _setup_command_logging(
        workspace_root,
        roi=roi_label,
        codebook=codebook,
        extra={
            "no_filter": no_filter,
            "overwrite": overwrite,
            "simple": simple,
            "no_split": no_split,
            "threads": threads,
        },
    )

    if path_wd.name.startswith("registered--"):
        raise ValueError("Path must be the main working directory, not registered.")

    paths_wd = sorted(path_wd.glob(f"registered--{_roi}+{codebook.stem}"))
    if not paths_wd:
        raise ValueError(f"No registered--{_roi}+{codebook.stem} found in {path_wd}")

    # Get size
    img = imread(next(path_wd.glob(f"registered--{_roi}+{codebook.stem}/*.tif")))
    size = img.shape[-1]
    logger.info(f"Size: {size}")
    del img

    for reg_path in paths_wd:
        roi = reg_path.name.split("--")[1]
        path = path_wd / f"registered--{roi}"
        path_cb = path / ("decoded-" + codebook.stem)
        try:
            coords = load_coords(path_wd, roi.split("+")[0])
        except FileNotFoundError:
            logger.error(
                f"Could not find {path_wd / f'stitch--{roi}' / 'TileConfiguration.registered.txt'}. Skipping {roi}."
            )
            continue

        assert len(coords) == len(set(coords["index"]))
        files = sorted(file for file in path_cb.glob("*.pkl"))

        if not files:
            raise ValueError(f"No files found in {path / codebook.stem}")
        logger.info(f"Found {len(files)} files.")

        # Filter coords to only include files that exist
        idxs = [name[:4] for name in coords["filename"]]
        # Non-split mode: extract "1234" from "something-1234.pkl"
        # Split mode: extract "1234" from "something-1234-0.pkl"
        files = [
            f
            for f in files
            if (match := FILENAME.match(f.name)) and match.group(1) in idxs and f.suffix == ".pkl"
        ]
        coords = coords.filter(pl.col("index").is_in({int(f.stem.rsplit("-", 2)[1]) for f in files}))

        # Precompute intersections
        cells = generate_cells(coords, files, split=not simple and not no_split, size=size)
        assert len(files) == len(cells)
        logger.debug(f"Generated {len(cells)} cells.")
        if not len(cells):
            raise ValueError("No tiles left to be processed after filtering.")

        idx = index.Index()
        for pos, cell in enumerate(cells):
            idx.insert(pos, cell.bounds)
        # Get the indices of the cells that cross each other
        crosses = [sorted(idx.intersection(poly.bounds)) for poly in cells]

        with ProcessPoolExecutor(max_workers=threads, mp_context=get_context("spawn")) as exc:
            futs = []
            for i, file in enumerate(files):
                if overwrite or not gen_out(file).exists():
                    futs.append(
                        exc.submit(
                            process,
                            file,
                            i,
                            size=size,
                            coords=coords,
                            cells=cells,
                            crosses=crosses,
                            filter_=not no_filter,
                            simple=simple,
                        )
                    )

            # Wait for either all futures to complete or any to raise an exception
            done, not_done = wait(futs, return_when=FIRST_EXCEPTION)

            # Check if any future raised an exception
            for fut in done:
                if fut.exception() is not None:
                    for f in not_done:
                        f.cancel()
                    fut.result()

        out_files = sorted([gen_out(file) for file in files])
        not_exists = [f for f in out_files if not f.exists()]
        if not_exists:
            logger.warning(f"{[f.name for f in not_exists]} do not exist. 0 spots or corrupted files.")
        df: pl.DataFrame = pl.scan_parquet([f for f in out_files if f.exists()]).collect()
        df.write_parquet(
            path_cb / "spots.parquet",
            metadata={"codebook": codebook.stem},
        )
        logger.info(f"Wrote to {path_cb / 'spots.parquet'}")
