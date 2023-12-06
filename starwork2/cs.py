# %%
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

import click
from loguru import logger

from fishtools.mkprobes.candidates import get_candidates
from fishtools.mkprobes.codebook.codebook import hash_codebook
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
    **kwargs,
):
    hsh = hash_codebook(codebook)
    restriction = ["BamHI", "KpnI"]
    if (
        acceptable is None
        and Path(f"output/{gene}_final_{''.join(restriction)}_{hsh}.parquet").exists()
        and not overwrite
    ):
        return

    ds = Dataset(path)
    try:
        if acceptable is not None or overwrite or not Path(f"output/{gene}_crawled.parquet").exists():
            get_candidates(
                ds,
                gene=gene,
                output="output/",
                ignore_revcomp=True,
                allow=acceptable,
                realign=overwrite,
                **kwargs,
            )
            time.sleep(1)
        overwrite = overwrite or acceptable is not None
        run_screen("output/", gene, minimum=30, restriction=restriction, maxoverlap=0, overwrite=overwrite)
        construct(
            ds,
            "output/",
            transcript=gene,
            codebook=codebook,
            restriction=restriction,
            target_probes=30,
            overwrite=overwrite,
        )
    except Exception as e:
        raise Exception(gene) from e


@click.command()
@click.argument(
    "path",
    type=click.Path(exists=True, dir_okay=True, path_type=Path),
)
@click.argument(
    "codebook_path", metavar="CODEBOOK", type=click.Path(exists=True, file_okay=True, path_type=Path)
)
@click.option("--overwrite", is_flag=True)
@click.option("--onlygene", type=str)
# @click.option("--acceptable", "acceptable_path", type=click.Path(exists=True, file_okay=True, path_type=Path))
def main(path: Path, codebook_path: Path, overwrite: bool = False, onlygene: str | None = None):
    codebook = json.loads(codebook_path.read_text())
    genes = set(codebook)

    acceptable_path = codebook_path.with_suffix(".acceptable.json")
    acceptable: dict[str, list[str]] = (
        json.loads(acceptable_path.read_text()) if acceptable_path.exists() else {}
    )
    logger.info(f"Acceptable: {acceptable} ")

    if set(acceptable) - genes:
        raise ValueError(f"Acceptable genes {set(acceptable) - set(genes)} not in gene list.")

    if onlygene:
        overwrite = True
        genes = onlygene.split(",")
        logger.warning(f"Running only {genes}")
        # for gene in onlygenes:
        #     if gene not in genes:
        #         raise ValueError("Invalid onlygene")
        #     run_gene(
        #         path,
        #         gene=gene,
        #         codebook=codebook,
        #         acceptable=acceptable.get(gene, None),
        #         overwrite=True,
        #     )

        #     return

    context = get_context("forkserver")
    with ProcessPoolExecutor(16, mp_context=context) as exc:
        futs = {
            (gene): exc.submit(
                run_gene,
                path,
                gene=gene,
                codebook=codebook,
                acceptable=acceptable.get(gene, None),
                overwrite=overwrite,
            )
            for gene in genes
            # if overwrite or not Path(f"output/{gene}_screened_ol-2_BamHIKpnI.parquet").exists()
        }
        failed: list[tuple[str, Exception]] = []
        for fut in as_completed(futs.values()):
            try:
                fut.result()
            except Exception as e:
                # raise e
                for x, f in futs.items():  # need to trace back to which gene caused the exception
                    if f == fut:
                        print(x)
                        failed.append((x, e))

        if failed:
            logger.critical(f"Failed on {failed}")
            for name, exc in failed:
                logger.exception(exc.with_traceback(None))


# %%

if __name__ == "__main__":
    main()
