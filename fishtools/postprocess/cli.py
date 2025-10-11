"""
Main CLI entry point for fishtools postprocessing utilities.
"""

import rich_click as click

from .cli_concat import concat
from .cli_intensity import extract_intensity

# Configure rich_click for enhanced CLI experience
click.rich_click.SHOW_ARGUMENTS = True
click.rich_click.GROUP_ARGUMENTS_OPTIONS = True
click.rich_click.USE_MARKDOWN = True
click.rich_click.STYLE_HELPTEXT = ""


@click.group()
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


# Add commands to the main group
main.add_command(concat)
main.add_command(extract_intensity, name="extract-intensity")


if __name__ == "__main__":
    main()
