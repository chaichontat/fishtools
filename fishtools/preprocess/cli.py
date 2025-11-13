from importlib import import_module
from types import SimpleNamespace

import rich_click as click

LAZY_COMMANDS: dict[str, SimpleNamespace] = {
    "basic": SimpleNamespace(module="fishtools.preprocess.cli_basic", attr="basic"),
    "deconvnew": SimpleNamespace(module="fishtools.preprocess.cli_deconv", attr="deconvnew"),
    "deconv": SimpleNamespace(module="fishtools.preprocess.cli_deconv_old", attr="deconv"),
    "register": SimpleNamespace(module="fishtools.preprocess.cli_register", attr="register"),
    "stitch": SimpleNamespace(module="fishtools.preprocess.cli_stitch", attr="stitch"),
    "spots": SimpleNamespace(module="fishtools.preprocess.spots.align_prod", attr="spots"),
    "inspect": SimpleNamespace(module="fishtools.preprocess.cli_inspect", attr="inspect_cli"),
    "verify": SimpleNamespace(module="fishtools.preprocess.cli_verify", attr="verify"),
    "correct-illum": SimpleNamespace(module="fishtools.preprocess.cli_correct_illum", attr="correct_illum"),
    "check-shifts": SimpleNamespace(module="fishtools.preprocess.cli_check_shifts", attr="check_shifts"),
    "check-stitch": SimpleNamespace(module="fishtools.preprocess.cli_check_stitch", attr="check_stitch"),
    "check-deconv": SimpleNamespace(module="fishtools.preprocess.cli_check_deconv", attr="check_deconv"),
}


class LazyGroup(click.Group):
    """Lazy-loading Click group that defers CLI imports until invocation."""

    def __init__(self, *args, lazy_commands: dict[str, SimpleNamespace] | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._lazy_commands = lazy_commands or {}

    def list_commands(self, ctx):
        eager = super().list_commands(ctx)
        lazy = sorted(self._lazy_commands)
        # Preserve eager order, append lazy while keeping uniqueness.
        ordered = list(dict.fromkeys([*eager, *lazy]))
        return ordered

    def get_command(self, ctx, cmd_name):
        command = super().get_command(ctx, cmd_name)
        if command is not None:
            return command

        spec = self._lazy_commands.get(cmd_name)
        if spec is None:
            return None

        module = import_module(spec.module)
        return getattr(module, spec.attr)


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


@click.group(cls=LazyGroup, lazy_commands=LAZY_COMMANDS)
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


main.add_command(registerv1)

if __name__ == "__main__":
    main()
