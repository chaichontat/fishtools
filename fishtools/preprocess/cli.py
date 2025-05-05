import rich_click as click

from .cli_basic import basic
from .cli_deconv import deconv
from .cli_register import register
from .cli_stitch import stitch
from .spots.align_prod import spots

# log = setup_logging()
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


main.add_command(basic)
main.add_command(deconv)
main.add_command(register)
main.add_command(stitch)
main.add_command(spots)

if __name__ == "__main__":
    main()
