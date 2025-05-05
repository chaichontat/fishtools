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
