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
from fishtools.mkprobes.codebook.finalconstruct import construct
from fishtools.mkprobes.ext.dataset import Dataset
from fishtools.mkprobes.screen import run_screen

# %%


def run_gene(
    dataset_path: Path,
    output_path: Path,
    codebook: dict[str, list[int]],
    transcript: str,
    acceptable: list[str] | None,
    overwrite: bool = False,
    log_level: str = "INFO",
    **kwargs,
):
    """
    Runs the probe generation pipeline for a single gene/transcript.

    This involves finding candidate probes, screening them, and potentially
    constructing the final probe set.

    Args:
        dataset_path: Path to the dataset directory.
        output_path: Path to the directory where output files will be saved.
        codebook: A dictionary mapping transcript IDs to codebook values.
        transcript: The specific transcript ID to process.
        acceptable: An optional list of acceptable "off-target" transcript IDs.
                    If provided, binders to these transcripts will not be discarded.
        overwrite: If True, overwrite existing output files.
        log_level: Logging level for the run.
        **kwargs: Additional keyword arguments passed to `get_candidates`.
    """

    ts = transcript
    logger.remove()
    logger.add(sys.stderr, level=log_level)

    restriction = ["BamHI", "KpnI"]
    if (
        acceptable is None
        and Path(
            f"{output_path}/{ts}_final_{''.join(restriction)}_{','.join(map(str, sorted(codebook[ts])))}.parquet"
        ).exists()
        and not overwrite
    ):
        return

    ds = Dataset.from_folder(dataset_path)
    try:
        if overwrite or not (output_path / f"{ts}_crawled.parquet").exists():
            get_candidates(
                ds,
                transcript=ts,
                output=output_path,
                ignore_revcomp=False,
                allow=acceptable,
                overwrite=overwrite,
                **kwargs,
            )
            time.sleep(1)
        overwrite = overwrite or acceptable is not None
        run_screen(output_path, ts, minimum=60, restriction=restriction, maxoverlap=0, overwrite=overwrite)
        construct(
            ds,
            output_path,
            transcript=ts,
            codebook=codebook,
            restriction=restriction,
            target_probes=48,
            overwrite=overwrite,
        )
    except Exception as e:
        raise Exception(ts) from e


@click.group()
def cli(): ...


@cli.command()
@click.argument(
    "path_dataset",
    type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=Path),
)
@click.argument(
    "codebook_path", metavar="CODEBOOK", type=click.Path(exists=True, file_okay=True, path_type=Path)
)
@click.option("--allow-file", type=click.Path(exists=True, dir_okay=False, file_okay=True, path_type=Path))
@click.option("--overwrite", is_flag=True)
@click.option("--listfailed", is_flag=True)
@click.option("--listfailedall", is_flag=True)
def single(
    path_dataset: Path,
    codebook_path: Path,
    allow_file: Path | None = None,
    overwrite: bool = False,
    listfailed: bool = False,
    listfailedall: bool = False,
):
    """
    Process a single codebook to generate probes.

    PATH_DATASET: Path to the dataset directory.
    CODEBOOK: Path to the JSON codebook file.
    """
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

    acceptable_path = allow_file or codebook_path.parent / "acceptable.json"
    acceptable: dict[str, list[str]] = (
        json.loads(acceptable_path.read_text()) if acceptable_path.exists() else {}
    )
    logger.info(f"Acceptable: {acceptable} ")

    failed_path = codebook_path.parent / (codebook_path.stem + ".failed.txt")
    failed_path.unlink(missing_ok=True)

    with ProcessPoolExecutor(6, mp_context=get_context("forkserver")) as exc:
        futs = {
            gene: exc.submit(
                run_gene,
                path_dataset,
                output_path=codebook_path.parent / "output",
                transcript=gene,
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


# %%

if __name__ == "__main__":
    cli()
