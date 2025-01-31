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

from fishtools.mkprobes.candidates import get_candidates
from fishtools.mkprobes.codebook.codebook import ProbeSet, hash_codebook
from fishtools.mkprobes.codebook.finalconstruct import construct
from fishtools.mkprobes.ext.external_data import Dataset
from fishtools.mkprobes.screen import run_screen

# %%


def run_gene(
    path: Path,
    *,
    codebook: dict[str, list[int]],
    gene: str,
    acceptable: list[str] | None,
    overwrite: bool = False,
    log_level: str = "INFO",
    **kwargs,
):
    logger.remove()
    logger.add(sys.stderr, level=log_level)

    hsh = hash_codebook(codebook)
    restriction = ["BamHI", "KpnI"]
    if (
        acceptable is None
        and Path(
            f"output/{gene}_final_{''.join(restriction)}_{','.join(map(str, sorted(codebook[gene])))}.parquet"
        ).exists()
        and not overwrite
    ):
        return

    ds = Dataset(path)
    try:
        if overwrite or not Path(f"output/{gene}_crawled.parquet").exists():
            get_candidates(
                ds,
                gene=gene,
                output="output/",
                ignore_revcomp=False,
                allow=acceptable,
                overwrite=overwrite,
                **kwargs,
            )
            time.sleep(1)
        overwrite = overwrite or acceptable is not None
        run_screen("output/", gene, minimum=60, restriction=restriction, maxoverlap=0, overwrite=overwrite)
        construct(
            ds,
            "output/",
            transcript=gene,
            codebook=codebook,
            restriction=restriction,
            target_probes=48,
            overwrite=overwrite,
        )
    except Exception as e:
        raise Exception(gene) from e


@click.group()
def cli():
    ...


@cli.command()
@click.argument(
    "path",
    type=click.Path(exists=True, dir_okay=True, path_type=Path),
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
    codebook = json.loads(codebook_path.read_text())
    hsh = hash_codebook(codebook)
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

    context = get_context("forkserver")
    failed_path = codebook_path.parent / (codebook_path.stem + ".failed.txt")
    failed_path.unlink(missing_ok=True)

    with ProcessPoolExecutor(6, mp_context=context) as exc:
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
