from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import polars as pl
import pyparsing as pp
from numpy.typing import NDArray
from shapely.geometry import Point, box
from shapely.ops import unary_union

if TYPE_CHECKING:
    from matplotlib.axes import Axes


class TileConfiguration:
    def __init__(self, df: pl.DataFrame):
        self.df = df

    def write(self, path: Path):
        with open(path, "w") as f:
            f.write("dim=2\n")
            for row in self.df.iter_rows(named=True):
                f.write(f"{row['index']:04d}.tif; ; ({row['x']}, {row['y']})\n")

    @classmethod
    def from_pos(cls, df: pd.DataFrame):
        pixel = 2048
        actual = pixel * 0.108
        scaling = 200 / actual
        adjusted = pd.DataFrame(dict(y=(df[0] - df[0].min()), x=(df[1] - df[1].min())))
        adjusted["x"] -= adjusted["x"].min()
        adjusted["x"] *= -(1 / 200) * pixel * scaling
        adjusted["y"] -= adjusted["y"].min()
        adjusted["y"] *= -(1 / 200) * pixel * scaling

        ats = adjusted.copy()
        ats["x"] -= ats["x"].min()
        ats["y"] -= ats["y"].min()

        # mat = ats[["y", "x"]].to_numpy() @ np.loadtxt("/home/chaichontat/fishtools/data/stage_rotation.txt")
        # ats["x"] = mat[:, 0]
        # ats["y"] = mat[:, 1]

        return cls(
            pl.DataFrame(
                ats.reset_index(), schema=pl.Schema({"index": pl.UInt32, "x": pl.Float32, "y": pl.Float32})
            )
        )

    @classmethod
    def from_file(cls, path: str | Path):
        """Parse TileConfiguration.txt file

        Returns:
            pl.DataFrame: ["prefix", "index", "filename", "x", "y"]
        """

        content = Path(path).read_text()

        integer = pp.Word(pp.nums)
        float_number = pp.Combine(
            pp.Optional(pp.oneOf("+ -")) + pp.Word(pp.nums) + "." + pp.Word(pp.nums)
        ).setParseAction(
            lambda t: float(t[0])  # type: ignore
        )

        point = pp.Suppress("(") + float_number("x") + pp.Suppress(",") + float_number("y") + pp.Suppress(")")
        entry = pp.Group(pp.Regex(r"(\w+-)?(\d+)(\.tif)")("filename") + pp.Suppress("; ; ") + point)

        comment = (pp.Literal("#") + pp.restOfLine) | (pp.Literal("dim") + pp.rest_of_line)
        parser = pp.ZeroOrMore(comment.suppress() | entry)

        # Perform this on extracted file name
        # since we want both the raw file name and the index.
        filename = (
            pp.Optional(pp.Word(pp.alphanums) + pp.Literal("-"))
            + integer("index").setParseAction(lambda t: int(t[0]))
            + pp.Literal(".tif")
        )  # type: ignore
        # Parse the content
        out = []
        for x in parser.parseString(content):
            d = x.as_dict()
            out.append(filename.parseString(d["filename"]).as_dict() | d)

        if not len(out):
            raise ValueError(f"File {path} is empty or not in the expected format.")

        return cls(
            pl.DataFrame(
                out,
                schema=pl.Schema({
                    "prefix": pl.Utf8,
                    "index": pl.UInt32,
                    "filename": pl.Utf8,
                    "x": pl.Float32,
                    "y": pl.Float32,
                }),
            )
        )

    def drop(self, idxs: list[int]):
        df = TileConfiguration(self.df.filter(~pl.col("index").is_in(idxs)))
        assert len(df) == len(self) - len(idxs)
        return df

    def downsample(self, factor: int) -> "TileConfiguration":
        if factor == 1:
            return self
        return TileConfiguration(self.df.with_columns(x=pl.col("x") / factor, y=pl.col("y") / factor))

    def __getitem__(self, sl: slice) -> "TileConfiguration":
        return TileConfiguration(self.df[sl])

    def __len__(self) -> int:
        return len(self.df)

    def plot(self, ax: "Axes | None" = None):
        import matplotlib.pyplot as plt

        if not ax:
            _, ax = plt.subplots(figsize=(8, 6), dpi=200)
        for row in self.df.iter_rows(named=True):
            ax.scatter(row["x"], row["y"], s=0.1)
            ax.text(row["x"], row["y"], str(row["index"]), fontsize=6, ha="center", va="center")
        ax.set_aspect("equal")

    # --- Selection helpers -------------------------------------------------
    def indices_at_least_n_steps_from_edges(self, n: int) -> np.ndarray:
        """Return tile indices at least ``n`` grid-steps from every edge.

        The distance is computed in ordinal grid space defined by the unique
        sorted ``x`` and ``y`` coordinates, not raw numeric spacing. This makes
        the method robust to arbitrary coordinate units (e.g., pixels vs. stage
        microns) and non-unit step sizes.

        Example: if the grid is 5 by 4 (unique x count = 5, unique y count = 4)
        and ``n=1``, only tiles with x-rank in [1, 3] and y-rank in [1, 2] are
        returned.

        Returns the values from the ``index`` column corresponding to the
        selected tiles (not dataframe row positions).
        """
        xy: NDArray[np.float64] = self.df.select(["x", "y"]).to_numpy()
        pos_idx = tiles_at_least_n_steps_from_edges(xy, n)
        # Map dataframe row positions to tile IDs in the "index" column
        return self.df.select("index").to_numpy().reshape(-1)[pos_idx]


def interior_indices_geometry(xy: NDArray[np.floating | np.integer], n: int) -> NDArray[np.int_]:
    """Geometry-backed interior selector using Shapely.

    - Builds a polygonal mosaic footprint by unioning axis-aligned boxes
      centered at each tile with half-sizes based on median grid steps.
    - Returns indices of points whose Euclidean distance to the footprint
      boundary is at least ``(n - 0.5) * step`` where ``step`` is the
      minimum median step across X and Y. This aligns with the ordinal
      interpretation: tiles one layer in have distance ~= 0.5·step.

    Args:
        xy: (N,2) array of tile centers.
        n: non-negative integer layer count.

    Returns:
        np.ndarray of positions into ``xy`` meeting the criterion.
    """
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("xy must be a (N, 2) array of [x, y] coordinates")
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0:
        return np.arange(len(xy), dtype=np.int64)

    x = np.asarray(xy[:, 0], dtype=float)
    y = np.asarray(xy[:, 1], dtype=float)
    ux = np.unique(x)
    uy = np.unique(y)

    # Need at least (2n+1) unique layers per axis to have an n-step interior
    if len(ux) < 2 * n + 1 or len(uy) < 2 * n + 1:
        return np.array([], dtype=np.int64)

    dxs = np.diff(np.sort(ux))
    dys = np.diff(np.sort(uy))
    dxs = dxs[dxs > 0]
    dys = dys[dys > 0]
    if dxs.size == 0 or dys.size == 0:
        return np.array([], dtype=np.int64)
    stepx = float(np.median(dxs))
    stepy = float(np.median(dys))
    step = float(min(stepx, stepy))
    if not np.isfinite(step) or step <= 0:
        return np.array([], dtype=np.int64)

    hx, hy = stepx / 2.0, stepy / 2.0
    # Build (multi)polygon footprint. To ensure adjacent tiles fuse into a single
    # polygon (removing internal shared edges), we enlarge slightly then shrink.
    eps = 1e-6 * max(stepx, stepy)
    enlarged = [box(xi - hx, yi - hy, xi + hx, yi + hy).buffer(eps) for xi, yi in zip(x, y)]
    footprint = unary_union(enlarged)

    # Morphological erosion by (n - 0.5) steps on the fused footprint
    thresh = (n - 0.5) * step
    eroded = footprint.buffer(-(eps + thresh))

    keep: list[int] = []
    for i, (xi, yi) in enumerate(zip(x, y)):
        if eroded.contains(Point(float(xi), float(yi))):
            keep.append(i)
    return np.asarray(keep, dtype=np.int64)


def copy_registered(reference_path: Path, actual_path: Path) -> None:
    """Copy *.registered.txt files from one directory to another.

    Kept as a library helper to avoid re-implementations across CLIs.
    """
    for file in reference_path.glob("*.registered.txt"):
        Path(actual_path).mkdir(parents=True, exist_ok=True)
        (actual_path / file.name).write_text(file.read_text())


def tiles_at_least_n_steps_from_edges(xy: NDArray[np.floating | np.integer], n: int) -> NDArray[np.int_]:
    """Return positions of tiles at least ``n`` steps from the edges.

    Parameters
    ----------
    xy
        Array of shape (N, 2) with columns ``[x, y]`` representing tile
        locations. Values need not be integers or unit-spaced.
    n
        Number of grid steps from every edge required to keep a tile.

    Returns
    -------
    np.ndarray
        1D array of 0-based positions into ``xy`` meeting the criterion.

    Notes
    -----
    "Steps" are measured in ordinal grid space derived from the unique sorted
    ``x`` and ``y`` values. For example, with unique ``x`` values
    ``[100, 200, 400]`` and unique ``y`` values ``[5, 15, 25]``, the center
    tile has x-rank=1, y-rank=1 and is one step from each edge.
    """
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("xy must be a (N, 2) array of [x, y] coordinates")
    if n < 0:
        raise ValueError("n must be non-negative")

    # Convert to float for consistent handling, then compute ordinal ranks
    x = np.asarray(xy[:, 0])
    y = np.asarray(xy[:, 1])

    ux = np.unique(x)
    uy = np.unique(y)

    # Early exit: not enough interior layers to satisfy n
    if len(ux) < 2 * n + 1 or len(uy) < 2 * n + 1:
        return np.array([], dtype=np.int64)

    x_rank_map = {v: i for i, v in enumerate(ux)}
    y_rank_map = {v: i for i, v in enumerate(uy)}

    ix = np.fromiter((x_rank_map[val] for val in x), count=len(x), dtype=np.int64)
    iy = np.fromiter((y_rank_map[val] for val in y), count=len(y), dtype=np.int64)

    dx = np.minimum(ix, (len(ux) - 1) - ix)
    dy = np.minimum(iy, (len(uy) - 1) - iy)
    dmin = np.minimum(dx, dy)

    return np.nonzero(dmin >= n)[0]
