import sys
from pathlib import Path

import click
import polars as pl
from loguru import logger

from fishtools.utils.samframe import SAMFrame

from ._filtration import the_filter

sys.setrecursionlimit(5000)


def run(
    data_dir: str | Path,
    gene: str,
    fpkm_path: str | Path | None = None,
    overlap: int = -2,
):
    data_dir = Path(data_dir)
    ff = SAMFrame(pl.read_parquet(data_dir / f"{gene}_crawled.parquet"))

    if fpkm_path is not None:
        fpkm = pl.read_parquet(fpkm_path)
        ff = ff.left_join(fpkm, left_on="transcript", right_on="transcript_id(s)").filter(
            pl.col("FPKM").lt(0.05 * pl.col("FPKM").first()) | pl.col("FPKM").lt(1) | pl.col("FPKM").is_null()
        )

    final = the_filter(ff, overlap=overlap)
    final.write_parquet(data_dir / f"{gene}_screened.parquet")
    logger.info("Done")


@click.command()
@click.argument("data_dir", type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=Path))
@click.argument("gene")
@click.option("--fpkm_path", type=click.Path(exists=True, dir_okay=False, file_okay=True, path_type=Path))
@click.option("--overlap", "-l", type=int, default=-2)
def screen(
    data_dir: str,
    gene: str,
    fpkm_path: Path | str | None = None,
    overlap: int = -2,
):
    """Screening of probes candidates for a gene."""
    run(
        data_dir,
        gene,
        fpkm_path=fpkm_path,
        overlap=overlap,
    )
