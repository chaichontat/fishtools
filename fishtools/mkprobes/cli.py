from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

import rich_click as click

from fishtools.ext.prepare import run_repeatmasker
from fishtools.mkprobes.alignment import bowtie_build
from fishtools.mkprobes.initial_screen.screen import screen
from fishtools.mkprobes.misc.chkgenes import chkgenes
from fishtools.utils.utils import setup_logging

from .initial_screen.candidates import candidates

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
    from fishtools.ext.prepare import download_gtf_fasta, run_jellyfish

    download_gtf_fasta(path / species, species)
    with ThreadPoolExecutor() as exc:
        exc.submit(run_jellyfish, path / species)
        exc.submit(
            run_repeatmasker,
            path / species / "cdna_ncrna_trna.fasta",
            species=dict(human="homo sapiens", mouse="mus musculus")[species],
            threads=threads,
        )
        exc.submit(bowtie_build, path / "cdna_ncrna_trna.fasta", "txome")


main.add_command(candidates)
main.add_command(screen)
main.add_command(chkgenes)


if __name__ == "__main__":
    main()
