import re
from pathlib import Path

import click
import polars as pl
from loguru import logger


# fmt: off
@click.command()
@click.argument("output_path", type=click.Path(dir_okay=True, file_okay=False, path_type=Path))
@click.option("--readouts", "-r", type=click.Path(exists=True, dir_okay=False, file_okay=True, path_type=Path))
@click.option("--codebook", "-c", type=click.Path(exists=True, dir_okay=False, file_okay=True, path_type=Path))
@click.option("--head", type=str, help="Sequence to add to the 5' end", default="")
@click.option("--tail", type=str, help="Sequence to add to the 3' end", default ="")
@click.option("--rc", is_flag=True, help="Reverse complement (for DNA probes which needs RT)")
@click.option("--maxn", type=int, help="Maximum number of probes per gene", default=64)
@click.option("--minn", type=int, help="Minimum number of probes per gene", default=48)
# fmt: on
def construct(
    output_path: Path,
    readouts: Path,
    codebook: Path,
    head: str = "",
    tail: str = "",
    rc: bool = True,
    maxn: int = 64,
    minn: int = 48,
):
    ...


@click.command()
@click.argument("output_path", type=click.Path(dir_okay=True, file_okay=False, path_type=Path))
@click.option("--genes", "-g", type=click.Path(exists=True, dir_okay=False, file_okay=True, path_type=Path))
@click.option("--min-probes", type=int, help="Minimum number of probes per gene", default=48)
@click.option(
    "--out",
    "-o",
    type=click.Path(dir_okay=False, file_okay=True, path_type=Path),
    help="Export filtered gene list",
)
def filter_genes(output_path: Path, genes: Path, min_probes: int, out: Path | None = None):
    def get_probes(gene: str):
        ols = list(output_path.glob(f"{gene}_screened_ol*.parquet"))
        if len(ols) > 0:
            ol = max([int(re.search(r"_ol(\d+)", f.stem).group(1)) for f in ols])
            return pl.read_parquet(output_path / f"{gene}_screened_ol{ol}.parquet")
        return pl.read_parquet(output_path / f"{gene}_screened.parquet")

    gene_list = genes.read_text().splitlines()
    ns: dict[str, int] = {}
    for gene in gene_list:
        if len(screened := get_probes(gene)) < min_probes:
            logger.warning(f"{gene} has {len(screened)} probes, less than {min_probes}.")
        ns[gene] = len(screened)

    if out:
        out.write_text("\n".join(gene for gene, n in ns.items() if n > min_probes))

