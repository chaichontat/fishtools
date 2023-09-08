from pathlib import Path
from typing import Literal

import rich_click as click

from fishtools.ext.external_data import ExternalData
from fishtools.io.pretty_print import jprint
from fishtools.mkprobes.alignment import bowtie_build
from fishtools.mkprobes.initial_screen.screen import screen
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
        > fishtools mkprobes prepare \<path\> [--species <species>]
    - Initial crawling with:
        > fishtools mkprobes candidates \<path\> \<gene\> \<output\> [--allow-pseudo] [--ignore-revcomp]
    - Screening and tiling of said candidates with:
        > fishtools mkprobes screen \<path from crawling\> \<gene\> [--fpkm-path <path>] [--overlap <int>]

    """
    ...


@main.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=Path))
def chkgenes(path: Path):
    from fishtools.ext.fix_gene_name import check_gene_names

    """Validate/check that gene names are canonical in Ensembl"""
    gtf = ExternalData.from_path("/home/chaichontat/oligocheck/data/mm39")
    genes = list(filter(lambda x: x, Path(path).read_text().splitlines()))
    if len(genes) != len(s := set(genes)):
        [genes.remove(x) for x in s]
        path.with_suffix(".unique.txt").write_text("\n".join(sorted(list(s))))
        log.error(f"Non-unique genes found: {', '.join(genes)}.\n")
        log.error(f"Unique genes written to {path.with_suffix('.unique.txt')}.\n")
        log.error("Aborting.")
        return

    converted, mapping, no_fixed_needed = check_gene_names(gtf, genes)
    if mapping:
        log.info("Mappings:")
        jprint(mapping)
        path.with_suffix(".converted.txt").write_text("\n".join(sorted(converted)))
    elif not no_fixed_needed:
        log.warning("Some genes cannot be found.")
    else:
        log.info(f"{len(s)} genes checked out. No changes needed")


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


if __name__ == "__main__":
    main()
