"""
Utilities for loading decoded spot tables and unifying coordinate columns.

Spot table column conventions
- idx_local (UInt32): Row index within the decoded tile. 0‑based.
- x_local, y_local (Float32): Tile‑local pixel coordinates of the spot centroid
  within the source tile image.
- x, y (Float32): Global mosaic pixel coordinates obtained by adding the tile’s
  stage offsets to x_local/y_local. These place all tiles into a single ROI
  coordinate frame.
- z (Float32): Axial coordinate of the centroid. 0.0 for 2D tiles.
- area (Float32): Connected‑component area (in pixels) of the decoded feature
  from the segmentation/combination step.
- target (Utf8): Decoded gene/target name assigned to the spot.
- distance (Float32): Decode metric distance from the spot’s intensity vector
  to the assigned target (as returned by starfish decode_metric). Lower is
  closer/better.
- norm (Float32): Euclidean L2 norm of the spot’s intensity vector across
  rounds×channels; a proxy for overall brightness.
- tile (Utf8): Tile identifier derived from the filename (e.g., index string).
- passes_thresholds (Boolean): Per‑spot boolean mask emitted by the decoding
  stage to indicate basic QC pass.

Notes
- Columns named x_ and y_ are reserved elsewhere (cli_spotlook) for feature‑
  space used in density thresholding and are not spatial coordinates.
- Optional/derived columns encountered downstream:
  - bit0, bit1, bit2 (UInt8): From codebook join; bit/channel ids per target.
  - is_blank (Boolean): Spot target name starts with "Blank"; convenience flag.
  - roi (Utf8): ROI label attached by CLI code for multi‑ROI operations.
  - x_, y_ (Float32): Feature‑space axes created by spotlook for density
    thresholding: cube‑root area + jitter and log10(norm×(1−distance)).
  - point_density (Float64): Temporary per‑spot density value in feature space
    used only during thresholding; not persisted in ROI parquet.
"""

import pickle
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import polars as pl
from loguru import logger


def load_spots(
    path: Path | str,
    idx: int,
    *,
    filter_: bool = True,
    tile_coords: Sequence[float] | None = None,
):
    """
    Load decoded spot records from a pickle file and return a Polars DataFrame
    with both tile-local and global coordinates.

    Parameters
    - path: Path to a ``.pkl`` produced by ``preprocess spots batch``.
    - idx: Position index (used for bookkeeping only).
    - filter_: When True, restricts to the spot indices referenced by the
      decoded table (historical behavior).
    - tile_coords: (x, y) offset of the tile in the stitched mosaic, in pixels.

    Coordinate conventions
    - ``x_local``, ``y_local``: coordinates within the source tile (tile‑local).
    - ``x``, ``y``: global mosaic coordinates, computed as
      ``x_local + tile_coords[0]`` and ``y_local + tile_coords[1]`` when
      ``tile_coords`` is provided.

    Columns (core schema)
    - ``idx_local``: row index within the tile (0‑based).
    - ``area``: connected‑component area in pixels.
    - ``z``: axial coordinate; 0.0 for 2D tiles.
    - ``x_local``, ``y_local``: tile‑local pixel coordinates.
    - ``x``, ``y``: global mosaic pixel coordinates.
    - ``target``: decoded gene/target label.
    - ``distance``: decode metric distance to assigned target.
    - ``norm``: L2 norm of per‑spot intensity vector.
    - ``tile``: tile identifier string.
    - ``passes_thresholds``: boolean QC flag from decoding stage.

    Notes
    - This function does not create ``x_``/``y_`` columns. In this repository,
      names with a trailing underscore are reserved in ``cli_spotlook`` for
      feature-space axes used during density-based thresholding (e.g.,
      cube-root area and log-scaled norm×(1−distance)). Those are not spatial
      coordinates.
    """
    if Path(path).suffix != ".pkl":
        raise ValueError(f"Expected .pkl file, got {Path(path).suffix}")
    try:
        d, y, *_ = pickle.loads(Path(path).read_bytes())
    except Exception as e:
        Path(path).unlink()
        raise Exception(f"Error reading {path}: {e}.") from e

    if filter_:
        try:
            y = np.array(y)[d.coords["spot_id"]].tolist()
            # d = d[d.coords["passes_thresholds"]]
        except KeyError:
            logger.warning(f"No passes_thresholds in {path}")

    df = pl.DataFrame(y)
    n_dims = len(df[0, "centroid"])

    if n_dims == 3:
        df = df.with_columns(pl.col("centroid").list.to_struct(fields=["z", "y_local", "x_local"])).unnest(
            "centroid"
        )
    else:
        df = (
            df.with_columns(pl.col("centroid").list.to_struct(fields=["y_local", "x_local"]))
            .unnest("centroid")
            .with_columns(z=pl.lit(0.0))
        )

    SCHEMA = pl.Schema([
        ("idx_local", pl.UInt32),
        ("area", pl.Float32),
        ("z", pl.Float32),
        ("y_local", pl.Float32),
        ("x_local", pl.Float32),
        ("y", pl.Float32),
        ("x", pl.Float32),
        ("target", pl.Utf8),
        ("distance", pl.Float32),
        ("norm", pl.Float32),
        ("tile", pl.Utf8),
        ("passes_thresholds", pl.Boolean),
    ])

    df = df.with_columns(
        y=pl.col("y_local") + (tile_coords[1] if tile_coords is not None else 0),
        x=pl.col("x_local") + (tile_coords[0] if tile_coords is not None else 0),
        target=pl.Series(list(d.coords["target"].values)),
        distance=pl.Series(list(d.coords["distance"].values)),
        norm=pl.Series(np.linalg.norm(d.values, axis=(1, 2))),
        tile=pl.lit(Path(path).stem.split("-")[1]),
        passes_thresholds=pl.Series(d.coords["passes_thresholds"].values),
    ).with_row_index("idx_local")

    return pl.DataFrame(df, schema=SCHEMA)


def load_spots_simple(
    path: Path | str,
    idx: int,
    *,
    filter_: bool = True,
    tile_coords: Sequence[float] | None = None,
):
    """
    Load a simplified spot table and annotate coordinates.

    Produces the same coordinate columns as ``load_spots`` and may include a
    subset of the auxiliary fields depending on starfish versions:
    - ``x_local``, ``y_local``: tile‑local coordinates.
    - ``x``, ``y``: global mosaic coordinates derived by adding ``tile_coords``.
    - Additional columns such as ``area``, ``z``, or decoding metrics may be
      present if provided by the source table; they are passed through.

    As with ``load_spots``, no ``x_``/``y_`` feature-space columns are created
    here; those are specific to the interactive spotlook tool.
    """
    if Path(path).suffix != ".pkl":
        raise ValueError(f"Expected .pkl file, got {Path(path).suffix}")
    try:
        d, y, *_ = pickle.loads(Path(path).read_bytes())
    except Exception as e:
        Path(path).unlink()
        raise Exception(f"Error reading {path}: {e}.") from e

    return (
        pl.DataFrame(d.to_features_dataframe())
        .rename(dict(y="y_local", x="x_local"))
        .with_columns(
            y=pl.col("y_local") + (tile_coords[1] if tile_coords is not None else 0),
            x=pl.col("x_local") + (tile_coords[0] if tile_coords is not None else 0),
            tile=pl.lit(Path(path).stem.split("-")[1]),
            passes_thresholds=pl.lit(True),
        )
        .with_row_index("idx_local")
    )


def load_parquet(
    path: Path | str,
    idx: int,
    *,
    filter_: bool = True,
    tile_coords: Sequence[float] | None = None,
):
    """
    Load a parquet spot table and annotate coordinates consistently.

    - Input is expected to have per‑tile coordinates named ``x``/``y`` which are
      treated as tile‑local here and renamed to ``x_local``/``y_local``.
    - The returned DataFrame contains both the local columns and global mosaic
      coordinates ``x``/``y`` after applying ``tile_coords``.

    This mirrors the behavior of the pickle loaders to keep downstream
    consumers agnostic to the original storage format.
    """
    return (
        pl.read_parquet(path)
        .rename(dict(y="y_local", x="x_local"))
        .with_columns(
            y=pl.col("y_local") + (tile_coords[1] if tile_coords is not None else 0),
            x=pl.col("x_local") + (tile_coords[0] if tile_coords is not None else 0),
            tile=pl.lit(str(idx)),
        )
        .with_row_index("idx_local")
    )
