import sys
from pathlib import Path
from typing import overload

import click
import polars as pl
from Bio import Restriction
from Bio.Seq import Seq
from loguru import logger

from .utils._filtration import the_filter
from .utils.samframe import SAMFrame

sys.setrecursionlimit(5000)


def _screen(
    data_dir: str | Path,
    gene: str,
    fpkm_path: str | Path | None = None,
    overlap: int = -2,
    restriction: list[str] | None = None,
):
    data_dir = Path(data_dir)

    ff = SAMFrame(pl.read_parquet(next(data_dir.glob(f"{gene}*_crawled.parquet"))))

    if fpkm_path is not None:
        fpkm = pl.read_parquet(fpkm_path)
        ff = ff.left_join(fpkm, left_on="transcript", right_on="transcript_id(s)").filter(
            pl.col("FPKM").lt(0.05 * pl.col("FPKM").first()) | pl.col("FPKM").lt(1) | pl.col("FPKM").is_null()
        )

    if restriction:
        ff = ff.filter(
            pl.col("seq").apply(
                lambda x: any(Restriction.RestrictionBatch(restriction).search(Seq(x)).values())
            )
        )

    final = the_filter(ff, overlap=overlap)

    final.write_parquet(
        data_dir / f"{gene}_screened_ol{overlap}.parquet"
        if overlap > 0
        else data_dir / f"{gene}_screened.parquet"
    )
    return final


@overload
def run_screen(
    data_dir: str,
    gene: str,
    fpkm_path: Path | str | None = ...,
    overlap: int = ...,
    minimum: None = ...,
    maxoverlap: int = ...,
    restriction: list[str] | None = ...,
) -> pl.DataFrame:
    ...


@overload
def run_screen(
    data_dir: str,
    gene: str,
    fpkm_path: Path | str | None = ...,
    overlap: int = ...,
    minimum: int = ...,
    maxoverlap: int = ...,
    restriction: list[str] | None = ...,
) -> dict[int, pl.DataFrame]:
    ...


def run_screen(
    data_dir: str,
    gene: str,
    fpkm_path: Path | str | None = None,
    overlap: int = -2,
    minimum: int | None = None,
    maxoverlap: int = 20,
    restriction: list[str] | None = None,
) -> dict[int, pl.DataFrame] | pl.DataFrame:
    if minimum is not None:
        if maxoverlap % 5 != 0 or maxoverlap == 0:
            raise ValueError("maxoverlap must be positive non-zero and a multiple of 5")

        res = {}
        for i in range(0, maxoverlap + 1, 5):
            res[i] = (final := _screen(data_dir, gene, fpkm_path=fpkm_path, overlap=i))
            if len(final) >= minimum:
                logger.info(f"Overlap {i} results in {len(final)} probes. Stopping.")
                return res
            logger.warning(f"Overlap {i} results in {len(final)} probes. Trying next overlap.")

        return res

    return _screen(data_dir, gene, fpkm_path=fpkm_path, overlap=overlap, restriction=restriction)


@click.command()
@click.argument("data_dir", type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=Path))
@click.argument("gene", type=str)
@click.option("--fpkm_path", type=click.Path(exists=True, dir_okay=False, file_okay=True, path_type=Path))
@click.option("--overlap", "-l", type=int, default=-2)
@click.option(
    "--minimum",
    type=int,
    help="Minimum number of probes per gene. "
    "Will generate probe sets with more overlaps (up to --maxoverlap) until the number is reached. "
    "Overrides --overlap.",
)
@click.option(
    "--maxoverlap", type=int, default=20, help="Maximum sequence overlap between probes if minimum is set."
)
@click.option("--restriction", type=str, multiple=True, help="Restriction enzymes to filter probes by.")
def screen(
    data_dir: str,
    gene: str,
    fpkm_path: Path | str | None = None,
    overlap: int = -2,
    minimum: int | None = None,
    maxoverlap: int = 20,
    restriction: list[str] | None = None,
):
    """Screening of probes candidates for a gene."""
    run_screen(
        data_dir,
        gene,
        fpkm_path=fpkm_path,
        overlap=overlap,
        minimum=minimum,
        maxoverlap=maxoverlap,
        restriction=restriction,
    )
