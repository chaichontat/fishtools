import re
import sys
from concurrent.futures import FIRST_EXCEPTION, ProcessPoolExecutor, wait
from multiprocessing import get_context
from pathlib import Path

import polars as pl
import rich_click as click
from loguru import logger
from rtree import index
from tifffile import imread

from fishtools.analysis.spots import load_spots, load_spots_simple
from fishtools.preprocess.stitching import (
    Cells,
    Crosses,
    filter_spots_by_area,
    generate_cells,
    get_exclusive_area,
)
from fishtools.preprocess.tileconfig import TileConfiguration

FILENAME = re.compile(r"reg-(\d+)(?:-(\d+))?\.(pkl|parquet)")
OUT_SUFFIX = "_deduped.parquet"
logger.configure(handlers=[{"sink": sys.stderr}])


def gen_out(path: Path):
    return path.with_name(path.stem + OUT_SUFFIX)


def _parse_filename(x: str):
    match = FILENAME.match(x)
    if not match:
        raise ValueError(f"Could not parse {x}")
    return {"idx": int(match.group(1)), "split": int(match.group(2)) if match.group(2) else None}


def load_coords(path: Path, roi: str):
    coords = TileConfiguration.from_file(
        Path(path.parent / f"stitch--{roi}" / "TileConfiguration.registered.txt")
    ).df
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
@click.argument("path_wd", type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=Path))
@click.argument("_roi", type=str, default="*")
@click.option("--codebook", type=click.Path(exists=True, dir_okay=False, file_okay=True, path_type=Path))
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
            coords = load_coords(path, roi.split("+")[0])
        except FileNotFoundError:
            logger.error(
                f"Could not find {path / f'stitch--{roi}' / 'TileConfiguration.registered.txt'}. Skipping {roi}."
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
        df = pl.scan_parquet([f for f in out_files if f.exists()]).collect()
        df.write_parquet(path_cb / "spots.parquet")
        logger.info(f"Wrote to {path_cb / 'spots.parquet'}")
