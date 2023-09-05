from pathlib import Path
import sys

from loguru import logger
import click

from fishtools.ext.external_data import Dataset

sys.setrecursionlimit(5000)
import polars as pl
from ._filtration import the_filter


# print(ff)
# ff.unique("name").write_parquet(f"output/{gene}_filtered.parquet")
# final = the_filter(ff, overlap=overlap).filter(pl.col("flag") & 16 == 0)
# logger.info(final)

# ff.write_parquet(
#     output
#     / f"{gene}_crawled.parquet"
#     # if overlap < 0
#     # else output / f"{gene}_final_overlap_{overlap}.parquet"
# )
def filter_pseudogene(ff: pl.DataFrame, gene: str) -> pl.DataFrame:
    """Filter pseudogenes."""


def run(
    dataset: Dataset,
    data_dir: str | Path,
    gene: str,
    output: str | Path,
    fpkm_path: str | Path | None = None,
    overlap: int = -2,
):
    data_dir = Path(data_dir)
    ff = pl.read_parquet(data_dir / f"{gene}_crawled.parquet")

    final = the_filter(ff, overlap=overlap).filter(pl.col("flag") & 16 == 0)
    logger.info(final)





@click.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=Path))
@click.argument("gene")
@click.argument("data_dir", type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=Path))
@click.option("--fpkm_path", type=click.Path(exists=True, dir_okay=False, file_okay=True, path_type=Path))
@click.option("--output", "-o", type=click.Path(), default="output/")
@click.option("--debug", "-d", is_flag=True)
@click.option("--ignore-revcomp", "-r", is_flag=True)
@click.option("--overlap", "-l", type=int, default=-2)
@click.option("--realign", is_flag=True)
def candidates(
    path: str,
    data_dir: str,
    fpkm_path: Path | str,
    gene: str,
    allow_pseudo: bool = True,
    debug: bool = False,
):
    """Initial screening of probes candidates for a gene."""
    run(
        Dataset(path),
        data_dir=data_dir,
        fpkm_path=fpkm_path,
        gene,
        allow_pseudo=allow_pseudo,
        overlap=overlap,
    )