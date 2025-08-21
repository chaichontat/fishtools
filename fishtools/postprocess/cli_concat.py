"""
CLI interface for multi-ROI FISH data concatenation pipeline.

This module provides the command-line interface for running the concat pipeline,
including configuration loading, validation, and pipeline orchestration.
"""

from pathlib import Path
from typing import Optional

import rich_click as click
from loguru import logger

from .concat_config import ConcatConfig, load_concat_config, validate_config_workspace
from .concat_pipeline import ConcatPipeline


@click.command()
@click.option(
    "--config",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to configuration TOML file",
)
@click.option("--workspace", type=click.Path(path_type=Path), help="Override workspace base path from config")
@click.option("--output", type=click.Path(path_type=Path), help="Override output H5AD path from config")
@click.option("--dry-run", is_flag=True, help="Validate configuration without processing data")
@click.option("--verbose", "-v", count=True, help="Increase verbosity (use -v, -vv, or -vvv)")
def concat(
    config: Path,
    workspace: Optional[Path] = None,
    output: Optional[Path] = None,
    dry_run: bool = False,
    verbose: int = 0,
) -> None:
    """
    Concatenate multi-ROI FISH data and create single-cell analysis object.
    
    This command processes multiple regions of interest (ROIs) from FISH experiments,
    integrates spot detection with cell segmentation, and creates a unified AnnData
    object for downstream single-cell analysis.
    
    Examples:
    
        # Basic usage with configuration file
        fishtools postprocess concat --config experiment_config.toml
        
        # Override workspace and output paths
        fishtools postprocess concat \\
            --config experiment_config.toml \\
            --workspace /alternative/data/path \\
            --output custom_output.h5ad
        
        # Validate configuration without processing
        fishtools postprocess concat --config experiment_config.toml --dry-run
        
        # Verbose output for debugging
        fishtools postprocess concat --config experiment_config.toml -vv
    """
    # Configure logging based on verbosity
    if verbose == 0:
        logger.remove()
        logger.add(lambda msg: None)  # Silent
    elif verbose == 1:
        logger.remove()
        logger.add(lambda msg: click.echo(msg), level="INFO")
    elif verbose == 2:
        logger.remove()
        logger.add(lambda msg: click.echo(msg), level="DEBUG")
    else:  # verbose >= 3
        logger.remove()
        logger.add(lambda msg: click.echo(msg), level="TRACE")

    try:
        # Load and validate configuration
        logger.info(f"Loading configuration from {config}")
        concat_config = load_concat_config(config, workspace_override=workspace, output_override=output)
        logger.info("Configuration loaded successfully")

        # Validate workspace structure
        logger.info("Validating workspace structure")
        validate_config_workspace(concat_config)
        logger.info("Workspace validation passed")

        if dry_run:
            logger.info("Dry run completed successfully - configuration is valid")
            click.echo("✅ Configuration validation passed")
            click.echo(f"   Workspace: {concat_config.workspace.path}")
            click.echo(f"   ROIs: {concat_config.rois}")
            click.echo(f"   Codebooks: {concat_config.codebooks}")
            click.echo(f"   Output: {concat_config.output.h5ad_path}")
            return

        # Execute pipeline
        logger.info("Starting pipeline execution")
        pipeline = ConcatPipeline(concat_config)
        adata = pipeline.run()

        # Save results
        output_path = concat_config.workspace.path / concat_config.output.h5ad_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        adata.write_h5ad(output_path)

        logger.info(f"Pipeline completed successfully. Results saved to {output_path}")
        click.echo("✅ Pipeline completed successfully")
        click.echo(f"   Output saved to: {output_path}")
        click.echo(f"   Final data shape: {adata.n_obs} cells × {adata.n_vars} genes")

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        click.echo(f"❌ Error: {e}", err=True)
        raise click.Abort()

    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        click.echo(f"❌ Configuration error: {e}", err=True)
        raise click.Abort()

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        click.echo(f"❌ Unexpected error: {e}", err=True)
        raise click.Abort()


if __name__ == "__main__":
    concat()
