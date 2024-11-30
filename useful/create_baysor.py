# %%
import polars as pl
from pathlib import Path

path = Path("/working/10xhuman")
spots = pl.scan_parquet("/working/10xhuman/*.parquet").collect()

spots.select("x", "y", "z", gene=pl.col("target").str.split("-").list.get(0)).write_csv(path / "spots.csv")

# %%
