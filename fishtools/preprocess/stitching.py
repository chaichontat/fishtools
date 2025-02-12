from pathlib import Path
from typing import TypeAlias

import polars as pl
from rtree import index
from shapely import MultiPolygon, Point, Polygon, STRtree, intersection

Cells: TypeAlias = list[Polygon]
Crosses: TypeAlias = list[list[int]]
Coords: TypeAlias = tuple[float, float]


def calculate_intersections(cells: Cells) -> Crosses:
    idx = index.Index()
    for pos, cell in enumerate(cells):
        idx.insert(pos, cell.bounds)
    return [sorted(idx.intersection(poly.bounds)) for poly in cells]


def get_exclusive_area(position: int, cell: Polygon, cells: Cells, crosses: Crosses) -> Polygon:
    lower_intersections = {j: intersection(cell, cells[j]) for j in crosses[position] if j < position}

    exclusive_area = cell
    for intersect in lower_intersections.values():
        exclusive_area = exclusive_area.difference(intersect)
    return exclusive_area


def filter_spots_by_area(df: pl.DataFrame, area: Polygon) -> pl.DataFrame:
    # Create STRtree with the area parts
    tree_geoms = list(area.geoms) if isinstance(area, MultiPolygon) else [area]
    tree = STRtree(tree_geoms)

    def check_point(x: float, y: float) -> bool:
        return len(tree.query(Point(x, y), "within")) > 0

    mask = df.select([
        pl.struct(["x", "y"])
        .map_elements(lambda row: check_point(row["x"], row["y"]), return_dtype=pl.Boolean)
        .alias("keep")
    ])

    return df.filter(mask["keep"])


def gen_splits(coords: Coords, n: int = 4, size: int = 1988, cut: int = 1024) -> Polygon:
    """Generate a polygon for a split region.

    Args:
        coords: (x,y) coordinates of the tile origin
        n: Split index (0=top-left, 1=top-right, 2=bottom-left, 3=bottom-right)
        size: Full tile size
        cut: Overlap size
    """
    if size < 1 or cut < 1:
        raise ValueError("width and offset must be greater than 0")
    x, y = coords

    match n:
        case 0:  # top-left
            return Polygon([
                (x, y),
                (x + cut, y),
                (x + cut, y + cut),
                (x, y + cut),
            ])
        case 1:  # top-right
            return Polygon([
                (x + size - cut, y),
                (x + size, y),
                (x + size, y + cut),
                (x + size - cut, y + cut),
            ])
        case 2:  # bottom-left
            return Polygon([
                (x, y + size - cut),
                (x + cut, y + size - cut),
                (x + cut, y + size),
                (x, y + size),
            ])
        case 3:  # bottom-right
            return Polygon([
                (x + size - cut, y + size - cut),
                (x + size, y + size - cut),
                (x + size, y + size),
                (x + size - cut, y + size),
            ])
        case _:
            raise ValueError(f"Unknown split index n={n}")


def generate_cells(
    coords: pl.DataFrame,
    files: list[Path],
    *,
    split: bool = True,
    size: int = 1988,
    cut: int = 1024,
) -> list[Polygon]:
    """Generate cell polygons from coordinates DataFrame and files.

    Args:
        coords: DataFrame with 'x', 'y', and 'index' columns
        files: List of files containing spot data
        split: Whether to generate split regions
        size: Full tile size
        cut: Overlap size for split regions
    """
    if split:
        cells = []
        for file in files:
            _, idx, n = file.stem.rsplit("-", 2)
            row = coords.filter(pl.col("index") == int(idx)).row(0, named=True)
            cells.append(gen_splits((row["x"], row["y"]), n=int(n), size=size, cut=cut))
        return cells

    return [
        Polygon([
            (row["x"], row["y"]),
            (row["x"] + size, row["y"]),
            (row["x"] + size, row["y"] + size),
            (row["x"], row["y"] + size),
        ])
        for row in coords.iter_rows(named=True)
    ]
