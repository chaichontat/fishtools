import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path
import sys

import polars as pl
import rich_click as click
from loguru import logger
from rtree import index
from shapely import MultiPolygon, Point, Polygon, STRtree, intersection

from fishtools.analysis.spots import load_spots
from fishtools.preprocess.tileconfig import TileConfiguration

FILENAME = re.compile(r"reg-(\d+)(?:-(\d+))?\.pkl")
logger.configure(handlers=[{"sink": sys.stderr}])

w = 1988


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


def gen_splits(coords: tuple[float, float], n: int = 4, size: int = 1988, cut: int = 1024):
    if size < 1 or cut < 1:
        raise ValueError("width and offset must be greater than 0")
    x, y = coords
    if n == 0:  # [:cut, :cut] - top-left
        return Polygon([
            (x, y),
            (x + cut, y),
            (x + cut, y + cut),  # + for y to go down
            (x, y + cut),
        ])
    if n == 1:  # [:cut, -cut:] - top-right
        return Polygon([(x + size - cut, y), (x + size, y), (x + size, y + cut), (x + size - cut, y + cut)])
    if n == 2:  # [-cut:, :cut] - bottom-left
        return Polygon([(x, y + size - cut), (x + cut, y + size - cut), (x + cut, y + size), (x, y + size)])
    if n == 3:  # [-cut:, -cut:] - bottom-right
        return Polygon([
            (x + size - cut, y + size - cut),
            (x + size, y + size - cut),
            (x + size, y + size),
            (x + size - cut, y + size),
        ])
    raise ValueError(f"Unknown n={n}")


def load_pickle(
    path: Path, i: int, coords: pl.DataFrame, *, filter_: bool = True, size: int = 1998, cut: int = 1024
):
    idx, split = _parse_filename(path.name).values()
    c = coords.filter(pl.col("index") == idx).row(0, named=True)

    logger.debug(f"Loading {path} as split {split}")

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
        return load_spots(path, i, filter_=filter_, tile_coords=(c["x"], c["y"]))
    except Exception as e:
        logger.error(f"Error reading {path}: {e.__class__.__name__}. Please rerun.")
        raise e


def process(
    path_pickle: Path,
    curr: int,
    coords: pl.DataFrame,
    cells: list[Polygon],
    crosses: list[list[int]],
    filter_: bool = True,
):
    current_cell = cells[curr]

    lower_intersections = {j: intersection(current_cell, cells[j]) for j in crosses[curr] if j < curr}

    df = load_pickle(path_pickle, curr, coords, filter_=filter_)
    df = df.filter(
        pl.col("passes_thresholds")
    )  # & pl.col("area").is_between(15, 80) & pl.col("norm").gt(0.05))

    # Create the "exclusive" area for this cell
    exclusive_area = current_cell
    for intersect in lower_intersections.values():
        exclusive_area = exclusive_area.difference(intersect)

    # Create STRtree with the exclusive area parts
    if isinstance(exclusive_area, MultiPolygon):
        tree_geoms = list(exclusive_area.geoms)
    else:
        tree_geoms = [exclusive_area]
    tree = STRtree(tree_geoms)

    def check_point(x: float, y: float):
        point = Point(x, y)
        return len(tree.query(point, "within")) > 0

    mask = df.select([
        pl.struct(["x", "y"])
        .map_elements(lambda row: check_point(row["x"], row["y"]), return_dtype=pl.Boolean)
        .alias("keep")
    ])

    filtered_df = df.filter(mask["keep"])
    logger.info(f"{path_pickle}: thrown {len(df) - len(filtered_df)} / {df.__len__()}")
    filtered_df.write_parquet(path_pickle.with_suffix(".parquet"))
    logger.info(f"Wrote to {path_pickle.with_suffix('.parquet')}")
    return filtered_df


@click.group()
def cli(): ...


@cli.command()
@click.argument("path_wd", type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=Path))
@click.argument("_roi", type=str, default="*")
@click.option("--codebook", type=click.Path(exists=True, dir_okay=False, file_okay=True, path_type=Path))
@click.option("--no-filter", is_flag=True)
@click.option("--overwrite", is_flag=True)
def run(
    path_wd: Path,
    _roi: str,
    codebook: Path,
    split: bool = True,
    no_filter: bool = False,
    overwrite: bool = False,
):
    if path_wd.name.startswith("registered--"):
        raise ValueError("Path must be the main working directory, not registered.")

    for reg_path in path_wd.glob(f"registered--{_roi}+{codebook.stem}"):
        roi = reg_path.name.split("--")[1]
        path = path_wd / f"registered--{roi}"
        path_cb = path / ("decoded-" + codebook.stem)
        try:
            coords = load_coords(path, roi.split("+")[0])
        except FileNotFoundError as e:
            logger.error(
                f"Could not find {path / f'stitch--{roi}' / 'TileConfiguration.registered.txt'}. Skipping {roi}."
            )
            continue

        assert len(coords) == len(set(coords["index"]))

        files = sorted(file for file in path_cb.glob("*.pkl") if "opt" not in file.stem)
        if not files:
            raise ValueError(f"No files found in {path / codebook.stem}")

        idxs = [name[:4] for name in coords["filename"]]
        # pickles = list(map(lambda x: x + ".pkl", idxs))
        files = (
            [file for file in files if file.name.rsplit("-", 1)[-1] in idxs]
            if not split
            else [file for file in files if file.name.rsplit("-", 2)[1] in idxs]
        )  # [:1300]
        coords = coords.filter(pl.col("index").is_in({int(f.stem.rsplit("-", 2)[1]) for f in files}))

        if split:
            cells = []
            for file in files:
                _, _idx, _n = file.stem.rsplit("-", 2)
                row = coords.filter(pl.col("index") == int(_idx)).row(0, named=True)
                cells.append(gen_splits((row["x"], row["y"]), n=int(_n)))
        else:
            cells = [
                Polygon([
                    (row["x"], row["y"]),
                    (row["x"] + w, row["y"]),
                    (row["x"] + w, row["y"] + w),
                    (row["x"], row["y"] + w),
                ])
                for row in coords.iter_rows(named=True)
            ]

        assert len(files) == len(cells)
        idx = index.Index()
        for pos, cell in enumerate(cells):
            idx.insert(pos, cell.bounds)
        # Get the indices of the cells that cross each other
        crosses = [sorted(idx.intersection(poly.bounds)) for poly in cells]

        with ProcessPoolExecutor(max_workers=4, mp_context=get_context("spawn")) as exc:
            futs = []
            for i, file in enumerate(files):
                if overwrite or not file.with_suffix(".parquet").exists():
                    futs.append(
                        exc.submit(
                            process,
                            file,
                            i,
                            coords=coords,
                            cells=cells,
                            crosses=crosses,
                            filter_=not no_filter,
                        )
                    )
            for fut in as_completed(futs):
                fut.result()

        df = pl.scan_parquet(sorted([file.with_suffix(".parquet") for file in files])).collect()
        df.write_parquet(path_cb / "spots.parquet")
        logger.info(f"Wrote to {path_cb / 'spots.parquet'}")


if __name__ == "__main__":
    cli()
