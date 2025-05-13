import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Literal

import rich_click as click

from fishtools.utils.utils import setup_logging

from .candidates import candidates
from .codebook.finalconstruct import click_construct, filter_genes
from .ext.dataset import Dataset, create_dataset
from .genes.chkgenes import chkgenes, convert_to_transcripts, transcripts
from .screen import screen
from .utils._alignment import bowtie_build

log = setup_logging()
click.rich_click.SHOW_ARGUMENTS = True
click.rich_click.GROUP_ARGUMENTS_OPTIONS = True
click.rich_click.USE_MARKDOWN = True
click.rich_click.STYLE_HELPTEXT = ""


@click.group()
def main():
    r"""Combinatorial FISH Probe Design Utilities

    Basic order of operations:

    - Prepare database with:
        > mkprobes prepare \<path\> [--species <species>]
    - Initial crawling with:
        > mkprobes candidates \<path\> \<gene\> \<output\> [--allow-pseudo] [--ignore-revcomp]
    - Screening and tiling of said candidates with:
        > mkprobes screen \<path from crawling\> \<gene\> [--fpkm-path <path>] [--overlap <int>]

    """
    ...


# fmt: off
@main.command()
@click.argument("path", type=click.Path(dir_okay=True, file_okay=False, path_type=Path))
@click.option("--species", "-s", type=click.Choice(("human", "mouse")), default="mouse", help="Species to use for probe design")
@click.option("--threads", "-t", type=int, default=16, help="Number of threads to use")
# fmt: on
def prepare(path: Path, species: Literal["human", "mouse"], threads: int = 16):
    """Prepare genomic database"""
    from .ext.prepare import download_gtf_fasta, run_jellyfish

    path = path.resolve()
    download_gtf_fasta(path / species, species)
    with ThreadPoolExecutor() as exc:
        futs = [
            exc.submit(run_jellyfish, path / species),
            exc.submit(bowtie_build, path / species / "cdna_ncrna_trna.fasta", "txome"),
        ]
        for fut in as_completed(futs):
            fut.result()
    Dataset(path / species)  # test all components


@main.command()
@click.argument("path", type=click.Path(dir_okay=False, file_okay=True, path_type=Path))
def hash(path: Path):
    """Hash codebook"""
    from .codebook.codebook import hash_codebook

    print(hash_codebook(json.loads(path.read_text())))


main.add_command(candidates)
main.add_command(screen)
main.add_command(chkgenes)
main.add_command(filter_genes)
main.add_command(transcripts)
main.add_command(convert_to_transcripts)
main.add_command(click_construct)
main.add_command(create_dataset)


if __name__ == "__main__":
    main()
