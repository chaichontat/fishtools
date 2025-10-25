"""
Main CLI entry point for fishtools postprocessing utilities.
"""

from importlib import import_module
from types import SimpleNamespace

import rich_click as click

LAZY_COMMANDS: dict[str, SimpleNamespace] = {
    "concat": SimpleNamespace(module="fishtools.postprocess.cli_concat", attr="concat"),
    "extract-intensity": SimpleNamespace(
        module="fishtools.postprocess.cli_intensity", attr="extract_intensity"
    ),
}


class LazyGroup(click.Group):
    """Lazy-loading Click group that defers CLI imports until invocation."""

    def __init__(self, *args, lazy_commands: dict[str, SimpleNamespace] | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._lazy_commands = lazy_commands or {}

    def list_commands(self, ctx):
        eager = super().list_commands(ctx)
        lazy = sorted(self._lazy_commands)
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


# Configure rich_click for enhanced CLI experience
click.rich_click.SHOW_ARGUMENTS = True
click.rich_click.GROUP_ARGUMENTS_OPTIONS = True
click.rich_click.USE_MARKDOWN = True
click.rich_click.STYLE_HELPTEXT = ""


@click.group(cls=LazyGroup, lazy_commands=LAZY_COMMANDS)
def main() -> None:
    """
    FISH Postprocessing Utilities

    Tools for post-processing and analysis of FISH (Fluorescence In Situ Hybridization)
    experimental data, including multi-ROI concatenation, intensity extraction, and
    single-cell analysis preparation.

    Available commands:

    - **concat**: Concatenate multi-ROI data and create single-cell analysis objects
    - **extract-intensity**: Extract intensity measurements from segmentation masks

    For help with individual commands, use:

        fishtools postprocess <command> --help
    """
    pass


if __name__ == "__main__":
    main()
