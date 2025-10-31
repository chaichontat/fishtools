import logging
from importlib import import_module
from pathlib import Path
from typing import Any, Optional

import click
import torch

from fishtools.preprocess.spots.overlay_spots import overlay as overlay_spots_command
from fishtools.segment.extract import cmd_extract as extract_cmd
from fishtools.segment.extract import cmd_extract_single as extract_single_cmd
from fishtools.segment.run import run as run_cli
from fishtools.segment.train import TrainConfig, build_trt_engine, run_train


def _strip_line_comments(text: str) -> str:
    """Remove lines that are comments (prefixed by //)."""
    lines = text.splitlines()
    kept = [line for line in lines if not line.lstrip().startswith("//")]
    return "\n".join(kept) + ("\n" if text.endswith("\n") else "")


class SegmentCLI(click.Group):
    """Click group that surfaces exceptions (standalone_mode=False by default)."""

    def main(self, *args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("standalone_mode", False)
        return super().main(*args, **kwargs)


app = SegmentCLI(help="Segmentation tooling CLI.")


@app.command("train")
@click.argument(
    "path",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
)
@click.argument("name")
@click.option("--use-te/--no-use-te", default=False, help="Enable training-time embeddings.")
@click.option("--te-fp8/--no-te-fp8", default=False, help="Request FP8 TensorRT embedding weights.")
@click.option(
    "--packed/--no-packed",
    default=False,
    help="Enable packed-stripe training to align with accelerated inference.",
)
def train(
    path: Path,
    name: str,
    use_te: bool,
    te_fp8: bool,
    packed: bool,
) -> None:
    models_path = path / "models"
    if not models_path.exists():
        raise click.ClickException(
            f"Models path {models_path} does not exist. If this is the correct directory, create one."
        )

    config_path = models_path / f"{name}.json"
    try:
        raw = config_path.read_text()
        train_config = TrainConfig.model_validate_json(_strip_line_comments(raw))
    except FileNotFoundError as exc:  # pragma: no cover - user error path
        raise click.ClickException(
            f"Config file {name}.json not found in {models_path}. Please create it first."
        ) from exc

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

    updated = run_train(name, path, train_config).model_dump_json(indent=2)
    output_path = models_path / f"{name}.trained.json"
    output_path.write_text(updated)


@app.command("run")
@click.argument(
    "volume",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
)
@click.option(
    "-m",
    "model_path",
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    help="Pretrained Cellpose model to load.",
)
@click.option("--channels", default="1,2", show_default=True, help="Comma-separated pair of channel indices.")
@click.option(
    "--anisotropy", default=4.0, show_default=True, type=float, help="Voxel anisotropy passed to Cellpose."
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    help="Directory where masks and metadata are stored.",
)
@click.option(
    "--overwrite/--no-overwrite", default=False, show_default=True, help="Overwrite existing mask tiles."
)
@click.option(
    "--normalize-percentiles",
    default="1.0,99.0",
    show_default=True,
    help="Comma-separated low,high percentiles for normalization.",
)
@click.option(
    "--save-flows/--no-save-flows", default=False, show_default=True, help="Persist raw network flows."
)
@click.option("--ortho-weights", help="Comma-separated weights for the (XY,YZ,ZX) passes.")
@click.option(
    "--backend",
    type=click.Choice(["sam", "unet"], case_sensitive=False),
    default="sam",
    show_default=True,
    help="Segmentation backend to use.",
)
def run_command(
    volume: Path,
    model_path: Path,
    channels: str,
    anisotropy: float,
    output_dir: Optional[Path],
    overwrite: bool,
    normalize_percentiles: str,
    save_flows: bool,
    ortho_weights: Optional[str],
    backend: str,
) -> None:
    run_cli(
        volume,
        model=model_path,
        channels=channels,
        anisotropy=anisotropy,
        output_dir=output_dir,
        overwrite=overwrite,
        normalize=normalize_percentiles,
        save_flows=save_flows,
        ortho_weights=ortho_weights,
        backend=backend,
    )


@app.command("trt-build")
@click.argument(
    "model",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--batch-size",
    default=1,
    show_default=True,
    type=click.IntRange(1, None),
    help="Maximum batch dimension to embed in the engine profile.",
)
def trt_build_cmd(model: Path, batch_size: int) -> None:
    from fishtools.segment.train import build_trt_engine

    if not torch.cuda.is_available():
        raise click.ClickException("CUDA GPU is required to build a TensorRT engine.")

    plan_path = build_trt_engine(
        model_path=model,
        device=torch.device("cuda:0"),
        bsize=256,
        batch_size=batch_size,
    )
    click.echo(f"Saved TensorRT engine to {plan_path}")


@app.command("extract")
@click.argument("mode", type=click.Choice(["z", "ortho"], case_sensitive=False))
@click.argument(
    "path",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
)
@click.argument("roi", required=False)
@click.option("--codebook", "-c", required=True, help="Registration codebook label.")
@click.option(
    "--out",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    help="Output directory; defaults under analysis/deconv/segment--{roi}+{codebook}.",
)
@click.option(
    "--dz", default=1, show_default=True, type=click.IntRange(1, None), help="Step between Z planes (z mode)."
)
@click.option(
    "--n",
    default=50,
    show_default=True,
    type=click.IntRange(1, None),
    help="Number of images to sample per ROI.",
)
@click.option(
    "--anisotropy",
    default=6,
    show_default=True,
    type=click.IntRange(1, None),
    help="Z scale factor (ortho mode).",
)
@click.option("--channels", help="Indices or metadata names, comma-separated.")
@click.option(
    "--crop", default=0, show_default=True, type=click.IntRange(0, None), help="Trim pixels at borders."
)
@click.option(
    "--threads", "-t", default=8, show_default=True, type=click.IntRange(1, 64), help="Parallel workers."
)
@click.option("--upscale", type=float, help="Additional spatial upscale factor applied before saving.")
@click.option("--seed", type=int, help="Random seed for sampling.")
@click.option(
    "--every",
    default=1,
    show_default=True,
    type=click.IntRange(1, None),
    help="Process every Nth file by size.",
)
@click.option("--max-from", help="Append max across channels from this codebook.")
@click.option(
    "--zarr/--no-zarr", default=False, show_default=True, help="Force reading inputs from fused Zarr store."
)
def extract_command(
    mode: str,
    path: Path,
    roi: Optional[str],
    codebook: str,
    out: Optional[Path],
    dz: int,
    n: int,
    anisotropy: int,
    channels: Optional[str],
    crop: int,
    threads: int,
    upscale: Optional[float],
    seed: Optional[int],
    every: int,
    max_from: Optional[str],
    zarr: bool,
) -> None:
    extract_cmd(
        mode,
        path,
        roi=roi,
        codebook=codebook,
        out=out,
        dz=dz,
        n=n,
        anisotropy=anisotropy,
        channels=channels,
        crop=crop,
        threads=threads,
        upscale=upscale,
        seed=seed,
        every=every,
        max_from=max_from,
        use_zarr=zarr,
    )


@app.command("extract-single")
@click.argument("mode", type=click.Choice(["z", "ortho"], case_sensitive=False))
@click.argument("registered", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--out",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    help="Output directory; defaults to <input_parent>/segment_extract.",
)
@click.option(
    "--dz", default=1, show_default=True, type=click.IntRange(1, None), help="Step between Z planes (z mode)."
)
@click.option(
    "--n",
    default=50,
    show_default=True,
    type=click.IntRange(1, None),
    help="Number of slices/positions to sample.",
)
@click.option(
    "--anisotropy",
    default=6,
    show_default=True,
    type=click.IntRange(1, None),
    help="Z scale factor (ortho mode).",
)
@click.option("--channels", help="Indices or metadata names, comma-separated.")
@click.option(
    "--crop", default=0, show_default=True, type=click.IntRange(0, None), help="Trim pixels at borders."
)
@click.option(
    "--threads", "-t", default=8, show_default=True, type=click.IntRange(1, 64), help="Parallel workers."
)
@click.option("--upscale", type=float, help="Additional spatial upscale factor applied before saving.")
@click.option("--seed", type=int, help="Random seed for sampling.")
@click.option(
    "--max-from",
    type=click.Path(path_type=Path),
    help="Optional registered stack for max-projection channel.",
)
@click.option("--label", help="Prefix label for outputs; defaults to the input stem.")
def extract_single_command(
    mode: str,
    registered: Path,
    out: Optional[Path],
    dz: int,
    n: int,
    anisotropy: int,
    channels: Optional[str],
    crop: int,
    threads: int,
    upscale: Optional[float],
    seed: Optional[int],
    max_from: Optional[Path],
    label: Optional[str],
) -> None:
    extract_single_cmd(
        mode,
        registered,
        out=out,
        dz=dz,
        n=n,
        anisotropy=anisotropy,
        channels=channels,
        crop=crop,
        threads=threads,
        upscale=upscale,
        seed=seed,
        max_from=max_from,
        label=label,
    )


@app.group()
def overlay() -> None:
    """Visualization helpers for segmentation outputs."""


@overlay.command("spots", help="Overlay decoded spots onto segmentation masks.")
@click.argument("path", type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path))
@click.argument("roi", required=False)
@click.option("--codebook", required=True, help="Codebook name used for decoded spots.")
@click.option(
    "--seg-codebook", help="Codebook label used for segmentation artifacts (defaults to --codebook)."
)
@click.option(
    "--spots", "spots_opt", type=click.Path(path_type=Path), help="Explicit spots parquet path or directory."
)
@click.option(
    "--segmentation-name",
    default="output_segmentation.zarr",
    show_default=True,
    help="Relative segmentation Zarr path within the ROI directory.",
)
@click.option(
    "--opening-radius", default=4.0, show_default=True, type=float, help="Radius for morphological opening."
)
@click.option(
    "--closing-radius", default=6.0, show_default=True, type=float, help="Radius for morphological closing."
)
@click.option(
    "--dilation-radius", default=2.0, show_default=True, type=float, help="Radius for final dilation."
)
@click.option(
    "--overwrite/--no-overwrite",
    default=False,
    show_default=True,
    help="Overwrite existing overlay artifacts.",
)
@click.option(
    "--debug/--no-debug", default=False, show_default=True, help="Enable verbose logging and debug plots."
)
def overlay_spots(
    path: Path,
    roi: Optional[str],
    codebook: str,
    seg_codebook: Optional[str],
    spots_opt: Optional[Path],
    segmentation_name: str,
    opening_radius: float,
    closing_radius: float,
    dilation_radius: float,
    overwrite: bool,
    debug: bool,
) -> None:
    current_roi = roi if roi is not None else "*"
    overlay_spots_command.callback(
        path,
        current_roi,
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


run = run_cli

__all__ = [
    "app",
    "train",
    "run_command",
    "trt_build_cmd",
    "extract_command",
    "extract_single_command",
    "overlay",
    "overlay_spots",
    "run",
    "cp_io",
    "TrainConfig",
    "build_trt_engine",
    "run_train",
]


def __getattr__(name: str) -> Any:
    if name == "cp_io":
        module = import_module("fishtools.segment.cp_io")
        globals()["cp_io"] = module
        return module
    raise AttributeError(name)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    app()


if __name__ == "__main__":
    app()
