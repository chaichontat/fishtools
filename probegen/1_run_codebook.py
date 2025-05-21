# %%
import json
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

import click
import polars as pl
from loguru import logger
from rich.console import Console
from rich.text import Text

from fishtools.mkprobes.candidates import get_candidates
from fishtools.mkprobes.codebook.codebook import ProbeSet
from fishtools.mkprobes.codebook.finalconstruct import construct
from fishtools.mkprobes.ext.dataset import ReferenceDataset as Dataset
from fishtools.mkprobes.screen import run_screen
from fishtools.utils.pretty_print import progress_bar

# %%


def run_gene(
    path: Path,
    *,
    codebook: dict[str, list[int]],
    gene: str,
    acceptable: list[str] | None,
    overwrite: bool = False,
    log_level: str = "INFO",
    output: Path = Path("output/"),
    **kwargs,
):
    logger.remove()
    logger.add(sys.stderr, level=log_level)
    logger.add(output / f"{gene}.log", level=log_level, colorize=False, backtrace=True, diagnose=True)

    restriction = ["BamHI", "KpnI"]
    if (
        acceptable is None
        and output.joinpath(
            f"{gene}_final_{''.join(restriction)}_{','.join(map(str, sorted(codebook[gene])))}.parquet"
        ).exists()
        and not overwrite
    ):
        return

    ds = Dataset(path)
    try:
        if overwrite or not output.joinpath(f"{gene}_crawled.parquet").exists():
            get_candidates(
                ds,
                transcript=gene,
                output=output,
                ignore_revcomp=False,
                allow=acceptable,
                overwrite=overwrite,
                **kwargs,
            )
            time.sleep(1)
        overwrite = overwrite or acceptable is not None
        run_screen(output, gene, minimum=60, restriction=restriction, maxoverlap=0, overwrite=overwrite)
        construct(
            ds,
            output,
            transcript=gene,
            codebook=codebook,
            restriction=restriction,
            target_probes=48,
            overwrite=overwrite,
        )
    except Exception as e:
        raise Exception(gene) from e


@click.group()
def cli(): ...


@cli.command()
@click.argument(
    "path",
    type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=Path),
)
@click.argument(
    "codebook_path", metavar="CODEBOOK", type=click.Path(exists=True, file_okay=True, path_type=Path)
)
@click.option("--overwrite", is_flag=True)
@click.option("--listfailed", is_flag=True)
@click.option("--listfailedall", is_flag=True)
def single(
    path: Path,
    codebook_path: Path,
    overwrite: bool = False,
    listfailed: bool = False,
    listfailedall: bool = False,
):
    console = Console(stderr=True)
    logger.configure(
        handlers=[
            {
                "sink": lambda s: console.print(Text.from_ansi(s)),
                "colorize": console.is_terminal,
            }
        ]
    )

    codebook = json.loads(codebook_path.read_text())
    codebook = {k: v for k, v in codebook.items() if not k.startswith("Blank")}
    if not len(set(codebook)) == len(codebook):
        raise ValueError("Duplicated genes in codebook.")
    genes = sorted(codebook)

    if listfailed or listfailedall:
        for gene in genes:
            if not Path(
                codebook_path.parent
                / "output"
                / f"{gene}_final_BamHIKpnI_{','.join(map(str, sorted(codebook[gene])))}.parquet"
            ).exists():
                print(gene)
                if listfailedall:
                    print(pl.read_csv(codebook_path.parent / "output" / f"{gene}_offtarget_counts.csv")[:5])
        return

    acceptable_path = codebook_path.with_suffix(".acceptable.json")
    acceptable: dict[str, list[str]] = (
        json.loads(acceptable_path.read_text()) if acceptable_path.exists() else {}
    )
    logger.info(f"Acceptable: {acceptable} ")

    failed_path = codebook_path.parent / (codebook_path.stem + ".failed.txt")
    failed_path.unlink(missing_ok=True)
    with (
        ProcessPoolExecutor(16, mp_context=get_context("forkserver")) as exc,
        progress_bar(len(genes)) as pbar,
    ):
        futs = {
            gene: exc.submit(
                run_gene,
                path,
                gene=gene,
                codebook=codebook,
                acceptable=acceptable.get(gene, None),
                overwrite=overwrite,
                log_level="DEBUG",
            )
            for gene in genes
            if overwrite or not (codebook_path.parent / f"output/{gene}_final_BamHIKpnI_.parquet").exists()
        }
        failed: list[tuple[str, Exception]] = []
        for fut in futs.values():
            fut.add_done_callback(pbar)

        for fut in as_completed(futs.values()):
            try:
                fut.result()
            except Exception as e:
                # raise e
                for x, f in futs.items():  # need to trace back to which gene caused the exception
                    if f == fut:
                        logger.critical(f"{x} failed {str(e)}.")
                        traceback.print_exception(type(e), e, e.__traceback__, file=sys.stderr)
                        with failed_path.open("a") as f:
                            f.write(x + "\n")

        if failed:
            logger.critical(f"Failed on {failed}")
            for name, exc in failed:
                logger.exception(exc.with_traceback(None))


@cli.command()
@click.argument("data", type=click.Path(exists=True, dir_okay=True, path_type=Path))
@click.argument("manifest", type=click.Path(exists=True, file_okay=True, path_type=Path))
def batch(data: Path, manifest: Path):
    assert single.callback
    pss = ProbeSet.from_list_json(manifest.read_text())
    for ps in pss:
        single.callback(
            data / ps.species,
            manifest.parent / ps.codebook,
            overwrite=False,
            onlygene=None,
            listfailed=False,
            listfailedall=False,
        )


# %%

if __name__ == "__main__":
    cli()
