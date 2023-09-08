from pathlib import Path
from typing import Literal

import rich_click as click

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
# fmt: on
def prepare(path: Path, species: Literal["human", "mouse"]):
    """Prepare genomic database"""
    from fishtools.ext.prepare import download_gtf_fasta, run_jellyfish

    download_gtf_fasta(path, species)
    run_jellyfish(path)
    bowtie_build(path / "cdna_ncrna_trna.fasta", "txome")


main.add_command(candidates)
main.add_command(screen)
main.add_command(chkgenes)


if __name__ == "__main__":
    main()
