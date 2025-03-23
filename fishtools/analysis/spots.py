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

    df = (
        pl.DataFrame(y)
        .with_columns(pl.col("centroid").list.to_struct())
        .unnest("centroid")
    )

    if "field_2" in df.columns:
        df = df.rename({"field_0": "z", "field_1": "y_local", "field_2": "x_local"})
    else:
        df = df.rename({"field_0": "y_local", "field_1": "x_local"}).with_columns(
            z=pl.lit(0.0)
        )

    return df.with_columns(
        y=pl.col("y_local") + (tile_coords[1] if tile_coords is not None else 0),
        x=pl.col("x_local") + (tile_coords[0] if tile_coords is not None else 0),
        target=pl.Series(list(d.coords["target"].values)),
        distance=pl.Series(list(d.coords["distance"].values)),
        norm=pl.Series(np.linalg.norm(d.values, axis=(1, 2))),
        tile=pl.lit(Path(path).stem.split("-")[1]),
        passes_thresholds=pl.Series(d.coords["passes_thresholds"].values),
    ).with_row_index("idx_local")


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
