import pickle
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import polars as pl


def load_spots(
    path: Path | str, idx: int, *, filter_: bool = True, tile_coords: Sequence[float] | None = None
):
    try:
        d, y, *_ = pickle.loads(Path(path).read_bytes())
    except Exception as e:
        Path(path).unlink()
        raise Exception(f"Error reading {path}: {e}. Deleted.") from e

    if filter_:
        y = np.array(y)[d.coords["passes_thresholds"]].tolist()
        d = d[d.coords["passes_thresholds"]]
    return (
        pl.DataFrame(y)
        .with_columns(pl.col("centroid").list.to_struct())
        .unnest("centroid")
        .rename({"field_0": "z", "field_1": "y_local", "field_2": "x_local"})
        .with_columns(
            y=pl.col("y_local") + (tile_coords[1] if tile_coords is not None else 0),
            x=pl.col("x_local") + (tile_coords[0] if tile_coords is not None else 0),
            target=pl.Series(list(d.coords["target"].values)),
            distance=pl.Series(list(d.coords["distance"].values)),
            norm=pl.Series(np.linalg.norm(d.values, axis=(1, 2))),
            tile=pl.lit(str(idx)),
            passes_thresholds=pl.Series(d.coords["passes_thresholds"].values),
        )
        .with_row_index("idx_local")
    )
