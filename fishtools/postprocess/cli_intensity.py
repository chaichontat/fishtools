"""
CLI interface for intensity extraction from segmentation masks.

This module provides the command-line interface for the intensity extraction
pipeline, including configuration loading, validation, and execution.
"""

from pathlib import Path
from typing import Optional

import rich_click as click
from loguru import logger
from rich.console import Console
from rich.table import Table

from .intensity_config import (
    IntensityExtractionConfig,
    load_intensity_config,
    validate_intensity_config,
)
from .intensity_pipeline import IntensityExtractionPipeline

# Configure rich_click for enhanced CLI experience
click.rich_click.SHOW_ARGUMENTS = True
click.rich_click.GROUP_ARGUMENTS_OPTIONS = True
click.rich_click.USE_MARKDOWN = True
click.rich_click.STYLE_HELPTEXT = ""


def display_config_summary(config: IntensityExtractionConfig) -> None:
    """
    Display configuration summary using Rich formatting.

    Args:
        config: Configuration to display
    """
    console = Console()

    table = Table(title="Intensity Extraction Configuration")
    table.add_column("Parameter", style="cyan", no_wrap=True)
    table.add_column("Value", style="green")

    table.add_row("Workspace Path", str(config.workspace_path))
    table.add_row("ROI", config.roi)
    table.add_row("Segmentation Codebook", config.seg_codebook)
    table.add_row("Channel", config.channel)
    table.add_row("Max Workers", str(config.max_workers))
    table.add_row("Overwrite", str(config.overwrite))
    table.add_row("Segmentation Zarr", config.segmentation_name)
    table.add_row("Intensity Zarr", config.intensity_name)
    table.add_row("Output Directory", str(config.output_directory))

    console.print(table)


def display_system_info(validation_info: dict) -> None:
    """
    Display system validation information using Rich formatting.

    Args:
        validation_info: Validation results from validate_intensity_config
    """
    console = Console()

    if "zarr_info" in validation_info:
        zarr_info = validation_info["zarr_info"]

        table = Table(title="Zarr Store Information")
        table.add_column("Property", style="cyan", no_wrap=True)
        table.add_column("Segmentation", style="yellow")
        table.add_column("Intensity", style="green")

        table.add_row("Shape", str(zarr_info["segmentation_shape"]), str(zarr_info["intensity_shape"]))
        table.add_row("Data Type", zarr_info["segmentation_dtype"], zarr_info["intensity_dtype"])
        table.add_row(
            "Chunks", str(zarr_info.get("segmentation_chunks")), str(zarr_info.get("intensity_chunks"))
        )

        console.print(table)

        # Display available channels
        if zarr_info["available_channels"]:
            console.print(
                f"\n[bold cyan]Available Channels:[/bold cyan] {', '.join(map(str, zarr_info['available_channels']))}"
            )

    if "performance_info" in validation_info:
        perf_info = validation_info["performance_info"]

        table = Table(title="System Performance Information")
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="green")

        table.add_row("Estimated Peak Memory", f"{perf_info['estimated_peak_memory_gb']:.1f} GB")
        table.add_row("Available Memory", f"{perf_info['available_memory_gb']:.1f} GB")
        table.add_row("Memory Utilization", f"{perf_info['memory_utilization_pct']:.1f}%")
        table.add_row("Recommended Workers", str(perf_info["recommended_workers"]))

        console.print(table)


@click.command()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    help="Path to TOML configuration file",
    required=True,
)
@click.option(
    "--workspace",
    "-w",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    help="Override workspace path from config",
)
@click.option(
    "--roi",
    "-r",
    type=str,
    help="Override ROI identifier from config",
)
@click.option(
    "--channel",
    "-ch",
    type=str,
    help="Override channel name from config",
)
@click.option(
    "--max-workers",
    "-j",
    type=int,
    help="Override max workers from config",
)
@click.option(
    "--overwrite",
    is_flag=True,
    help="Overwrite existing output files",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Validate configuration and display information without processing",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose logging",
)
@click.option(
    "--quiet",
    "-q",
    is_flag=True,
    help="Suppress all output except errors",
)
def extract_intensity(
    config: Path,
    workspace: Optional[Path] = None,
    roi: Optional[str] = None,
    channel: Optional[str] = None,
    max_workers: Optional[int] = None,
    overwrite: bool = False,
    dry_run: bool = False,
    verbose: bool = False,
    quiet: bool = False,
) -> None:
    """
    Extract intensity measurements from segmentation masks.
    
    This command processes Zarr-format segmentation masks and intensity images
    to extract region properties including area, centroid, and intensity statistics.
    
    **Examples:**
    
    Basic usage with configuration file:
    
        fishtools postprocess extract-intensity --config intensity_config.toml
    
    Override specific parameters:
    
        fishtools postprocess extract-intensity \\
            --config intensity_config.toml \\
            --roi roi2 \\
            --channel cfse \\
            --max-workers 8
    
    Dry run to validate configuration:
    
        fishtools postprocess extract-intensity \\
            --config intensity_config.toml \\
            --dry-run
    
    **Configuration File Format:**
    
    ```toml
    workspace_path = "/path/to/experiment"
    roi = "roi1"
    channel = "cfse"
    seg_codebook = "atp"
    max_workers = 4
    overwrite = false
    segmentation_name = "output_segmentation.zarr"
    intensity_name = "input_image.zarr"
    ```
    
    The command will create parquet files in the output directory with extracted
    region properties for each Z-slice of the segmentation mask.
    """
    # Configure logging based on verbosity
    if quiet:
        logger.remove()
        logger.add(lambda m: None if m.record["level"].name != "ERROR" else print(m, end=""))
    elif verbose:
        logger.remove()
        logger.add(lambda m: print(m, end=""))

    console = Console()

    try:
        # Load configuration with overrides
        logger.info(f"Loading configuration from {config}")
        extraction_config = load_intensity_config(
            config, workspace_override=workspace, roi_override=roi, channel_override=channel
        )

        # Apply additional overrides
        if max_workers is not None:
            extraction_config.max_workers = max_workers
        if overwrite:
            extraction_config.overwrite = True

        logger.info("Configuration loaded successfully")

        # Display configuration summary
        if not quiet:
            display_config_summary(extraction_config)

        # Validate configuration and system requirements
        logger.info("Validating configuration and system requirements")
        validation_info = validate_intensity_config(extraction_config)
        logger.info("Validation completed successfully")

        # Display system information
        if not quiet:
            display_system_info(validation_info)

        # Check for dry run
        if dry_run:
            console.print(
                "\n[bold green]✓[/bold green] Dry run completed successfully. Configuration is valid."
            )
            logger.info("Dry run completed - configuration is valid")
            return

        # Create and run pipeline
        logger.info("Initializing intensity extraction pipeline")
        pipeline = IntensityExtractionPipeline(extraction_config, validate_config=False)

        logger.info("Starting intensity extraction processing")
        pipeline.run()

        # Success message
        if not quiet:
            console.print(f"\n[bold green]✓[/bold green] Intensity extraction completed successfully!")
            console.print(f"Output saved to: {extraction_config.output_directory}")

        logger.info("Intensity extraction completed successfully")

    except FileNotFoundError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        logger.error(f"File not found: {e}")
        raise click.ClickException(str(e))

    except ValueError as e:
        console.print(f"[bold red]Configuration Error:[/bold red] {e}")
        logger.error(f"Configuration error: {e}")
        raise click.ClickException(str(e))

    except RuntimeError as e:
        console.print(f"[bold red]Processing Error:[/bold red] {e}")
        logger.error(f"Processing error: {e}")
        raise click.ClickException(str(e))

    except Exception as e:
        console.print(f"[bold red]Unexpected Error:[/bold red] {e}")
        logger.error(f"Unexpected error: {e}")
        raise click.ClickException(f"Unexpected error: {e}")


if __name__ == "__main__":
    extract_intensity()
