import logging
from pathlib import Path
from typing import Annotated, Any, Optional

import torch
import typer

from fishtools.preprocess.spots.overlay_spots import overlay as overlay_spots_command
from fishtools.segment.extract import cmd_extract as extract_cmd
from fishtools.segment.extract import cmd_extract_single as extract_single_cmd
from fishtools.segment.run import run
from fishtools.segment.train import TrainConfig, build_trt_engine, run_train


def _strip_line_comments(text: str) -> str:
    """Remove lines that are comments (prefixed by //).

    This is a minimal pre-processing step to allow //-style line comments in
    JSON config files. Only whole-line comments are removed; inline fragments
    after JSON content are not altered.
    """
    lines = text.splitlines()
    kept = [line for line in lines if not line.lstrip().startswith("//")]
    return "\n".join(kept) + ("\n" if text.endswith("\n") else "")


app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command("train", help="Train a model")
def train(
    path: Annotated[
        Path,
        typer.Argument(
            help="Path to the main training directory",
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
    ],
    name: Annotated[str, typer.Argument(help="Name of the model")],
    use_te: bool = False,
    te_fp8: bool = False,
    packed: bool = typer.Option(
        False,
        "--packed",
        help="Enable packed-stripe training to match the accelerated inference path.",
        rich_help_panel="Training",
    ),
):
    models_path = path / "models"

    if not models_path.exists():
        raise FileNotFoundError(
            f"Models path {models_path} does not exist. If this is the correct directory, create one."
        )

    config_path = models_path / f"{name}.json"

    try:
        raw = config_path.read_text()
        train_config = TrainConfig.model_validate_json(_strip_line_comments(raw))
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Config file {name}.json not found in {models_path}. Please create it first."
        )

    # Allow CLI flags to override/augment config values.
    effective_use_te = train_config.use_te or use_te or train_config.te_fp8 or te_fp8
    effective_te_fp8 = train_config.te_fp8 or te_fp8

    if effective_te_fp8 and not effective_use_te:
        effective_use_te = True

    updates: dict[str, Any] = {}
    if (effective_use_te != train_config.use_te) or (effective_te_fp8 != train_config.te_fp8):
        updates.update({"use_te": effective_use_te, "te_fp8": effective_te_fp8})

    if packed != train_config.packed:
        updates["packed"] = packed

    if updates:
        train_config = train_config.model_copy(update=updates)

    # Preserve the original file (including comments). Write updated config to a new file.
    updated = run_train(name, path, train_config).model_dump_json(indent=2)
    output_path = models_path / f"{name}.trained.json"
    output_path.write_text(updated)


app.command("run", help="Run Cellpose 3D inference on a fused volume")(run)


@app.command("trt-build", help="Build a TensorRT engine for a trained model.")
def trt_build_cmd(
    model: Annotated[
        Path,
        typer.Argument(
            help="Path to the trained model weights",
            exists=True,
            file_okay=True,
            dir_okay=False,
            resolve_path=True,
        ),
    ],
    batch_size: int = typer.Option(
        1,
        "--batch-size",
        help="Maximum batch dimension to embed in the engine profile.",
        show_default=True,
    ),
):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required to build a TensorRT engine.")

    model_path = model
    name = model_path.name

    plan_path = build_trt_engine(
        model_path=model_path,
        name=name,
        device=torch.device("cuda:0"),
        bsize=256,
        batch_size=batch_size,
    )

    typer.echo(f"Saved TensorRT engine to {plan_path}")


# Register the extract command directly from the submodule function.
# This avoids double-nesting ("segment extract extract") and follows the
# Typer multi-file pattern where a function defined elsewhere can be added
# as a command of the top-level app.
app.command(
    "extract",
    help="Export training-ready slices from registered stacks (mode: z|ortho)",
)(extract_cmd)

app.command(
    "extract-single",
    help="Export training-ready slices from a single registered stack (mode: z|ortho)",
)(extract_single_cmd)


overlay_app = typer.Typer(
    help="Visualization helpers for segmentation outputs.",
    pretty_exceptions_show_locals=False,
)


@overlay_app.command("spots", help="Overlay decoded spots onto segmentation masks.")
def overlay_spots(
    path: Path = typer.Argument(
        ...,
        help="Path to the workspace root containing segmentation outputs.",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
    codebook: str = typer.Option(
        ...,
        "--codebook",
        help="Codebook name used for decoded spots.",
        rich_help_panel="Inputs",
    ),
    roi: str = typer.Argument(
        "*",
        help="ROI to process (use '*' to process all available ROIs).",
        show_default=False,
    ),
    seg_codebook: Optional[str] = typer.Option(
        None,
        "--seg-codebook",
        help="Codebook label used for the segmentation artifacts (defaults to --codebook).",
        rich_help_panel="Inputs",
    ),
    spots_opt: Optional[Path] = typer.Option(
        None,
        "--spots",
        help="Optional explicit path to the spots parquet file or directory of per-ROI parquets.",
        rich_help_panel="Inputs",
    ),
    segmentation_name: str = typer.Option(
        "output_segmentation.zarr",
        "--segmentation-name",
        help="Relative segmentation Zarr path within the ROI directory.",
        show_default=True,
    ),
    opening_radius: float = typer.Option(
        4.0,
        "--opening-radius",
        help="Radius for morphological opening.",
        show_default=True,
    ),
    closing_radius: float = typer.Option(
        6.0,
        "--closing-radius",
        help="Radius for morphological closing.",
        show_default=True,
    ),
    dilation_radius: float = typer.Option(
        2.0,
        "--dilation-radius",
        help="Radius for final dilation (set to 0 to disable).",
        show_default=True,
    ),
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        help="Overwrite existing overlay artifacts.",
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Enable verbose logging and debug plots.",
    ),
):
    overlay_spots_command.callback(
        path,
        roi,
        codebook,
        spots_opt,
        seg_codebook,
        segmentation_name,
        opening_radius,
        closing_radius,
        dilation_radius,
        overwrite,
        debug,
    )


app.add_typer(
    overlay_app,
    name="overlay",
    help="Generate overlays for segmentation quality assurance workflows.",
)


def main():
    logging.basicConfig(level=logging.INFO)
    app()


if __name__ == "__main__":
    app()
