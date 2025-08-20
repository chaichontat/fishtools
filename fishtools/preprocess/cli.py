import rich_click as click

from .cli_basic import basic
from .cli_deconv import deconv
from .cli_register import register
from .cli_stitch import stitch
from .spots.align_prod import spots


# Wrapper to integrate Typer app with Click
@click.command("registerv1", context_settings={"ignore_unknown_options": True})
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
@click.pass_context
def registerv1(ctx, args):
    """🔬 FISH Image Registration and Preprocessing Pipeline (Typer-based)

    Enhanced CLI with rich formatting, parameter validation, and comprehensive help.

    Examples:
    - Register single image: registerv1 /path/workspace roi 42 --codebook codebook.json
    - Register all files in ROI: registerv1 /path/workspace roi --codebook codebook.json
    - Register all files: registerv1 /path/workspace --codebook codebook.json
    """
    import subprocess
    import sys

    if not args:
        # Show help if no arguments provided
        args = ["--help"]

    # Call the Typer CLI as a subprocess
    cmd = [sys.executable, "-m", "fishtools.preprocess.cli_register_migrated"] + list(args)

    try:
        result = subprocess.run(cmd, check=False)
        ctx.exit(result.returncode)
    except KeyboardInterrupt:
        ctx.exit(1)


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
main.add_command(registerv1)
main.add_command(stitch)
main.add_command(spots)

if __name__ == "__main__":
    main()
