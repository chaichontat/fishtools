import re
import sys
from itertools import chain
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
    output_dir: str | Path,
    gene: str,
    fpkm_path: str | Path | None = None,
    overlap: int = -2,
    restriction: list[str] | None = None,
):
    output_dir = Path(output_dir)

    # if re.search("-2\d\d", gene) is None:  # not transcript
    #     try:
    #         path = next(output_dir.glob(f"{gene}-_crawled.parquet"))
    #         gene = path.name.split("_crawled")[0]
    #         ff = SAMFrame(pl.read_parquet(path))
    #     except StopIteration:
    #         logger.critical(f"No crawled data found for {gene}. Aborting.")
    #         raise Exception
    # else:
    ff = SAMFrame(pl.read_parquet(output_dir / f"{gene}_crawled.parquet"))

    if fpkm_path is not None:
        fpkm = pl.read_parquet(fpkm_path)
        ff = ff.left_join(fpkm, left_on="transcript", right_on="transcript_id(s)").filter(
            pl.col("FPKM").lt(0.05 * pl.col("FPKM").first()) | pl.col("FPKM").lt(1) | pl.col("FPKM").is_null()
        )

    if restriction:
        logger.info(f"Filtering for probes w/o {', '.join(restriction)} site(s).")
        res = Restriction.RestrictionBatch(restriction)
        ff = ff.filter(~pl.col("seq").apply(lambda x: any(res.search(Seq("NNNNNN" + x + "NNNNNN")).values())))

    final = the_filter(ff, overlap=overlap)
    assert not final["seq"].str.contains("N").any(), "N appears out of nowhere."
    final.write_parquet(
        write_path := output_dir
        / f"{gene}_screened_ol{overlap}{'_' + ''.join(restriction) if restriction else '' }.parquet"
    )

    logger.debug(f"Written to {write_path}.")
    return final


@overload
def run_screen(
    output_dir: Path | str,
    gene: str,
    fpkm_path: Path | str | None = ...,
    overlap: int = ...,
    minimum: None = ...,
    maxoverlap: int = ...,
    restriction: list[str] | None = ...,
    overwrite: bool = False,
) -> pl.DataFrame: ...


@overload
def run_screen(
    output_dir: Path | str,
    gene: str,
    fpkm_path: Path | str | None = ...,
    overlap: int = ...,
    minimum: int = ...,
    maxoverlap: int = ...,
    restriction: list[str] | None = ...,
    overwrite: bool = False,
) -> dict[int, pl.DataFrame]: ...


def run_screen(
    output_dir: Path | str,
    gene: str,
    fpkm_path: Path | str | None = None,
    overlap: int = -2,
    minimum: int | None = None,
    maxoverlap: int = 20,
    restriction: list[str] | None = None,
    overwrite: bool = False,
) -> dict[int, pl.DataFrame] | pl.DataFrame:
    output_dir = Path(output_dir)
    if (
        not overwrite
        and (
            output_dir
            / (
                file := f"{gene}_screened_ol{overlap}{'_' + ''.join(restriction) if restriction else '' }.parquet"
            )
        ).exists()
    ):
        logger.warning(f"File {file} exists. Skipping.")
        return pl.read_parquet(output_dir / file)

    for file in output_dir.glob(
        f"{gene}_screened_ol*{'_' + ''.join(restriction) if restriction else '' }.parquet"
    ):
        file.unlink(missing_ok=True)

    if minimum is not None:
        if maxoverlap % 5 != 0:
            raise ValueError("maxoverlap must be multiple of 5")

        res = {}
        for i in chain((-2,), range(5, maxoverlap + 1, 5)):
            res[i] = (
                final := _screen(output_dir, gene, fpkm_path=fpkm_path, overlap=i, restriction=restriction)
            )
            if len(final) >= minimum:
                logger.info(f"Overlap {i} results in {len(final)} probes. Stopping.")
                return res
            logger.warning(f"Overlap {i} results in {len(final)} probes. Trying next overlap.")

        return res

    return _screen(output_dir, gene, fpkm_path=fpkm_path, overlap=overlap, restriction=restriction)


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
@click.option("--restriction", type=str, help="Restriction enzymes to filter probes by.")
@click.option("--overwrite", is_flag=True, help="Overwrite existing files.")
def screen(
    data_dir: str,
    gene: str,
    fpkm_path: Path | str | None = None,
    overlap: int = -2,
    minimum: int | None = None,
    maxoverlap: int = 20,
    restriction: str | None = None,
    overwrite: bool = False,
):
    """Screening of probes candidates for a gene."""
    run_screen(
        data_dir,
        gene,
        fpkm_path=fpkm_path,
        overlap=overlap,
        minimum=minimum,
        maxoverlap=maxoverlap,
        restriction=restriction.split(",") if restriction else None,
        overwrite=overwrite,
    )
