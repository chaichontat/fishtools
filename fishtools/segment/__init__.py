import logging
from functools import lru_cache
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import rich_click as click
import torch

from fishtools.utils.pretty_print import TaskCancelledException

if TYPE_CHECKING:  # pragma: no cover
    from fishtools.segment.train import TrainConfig as TrainConfig


@lru_cache(maxsize=None)
def _import_cached(module: str):
    return import_module(module)


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


class _LazyCommandGroup(click.Group):
    def __init__(self, *args: Any, lazy_commands: dict[str, SimpleNamespace] | None = None, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._lazy_commands = lazy_commands or {}

    def list_commands(self, ctx):  # type: ignore[override]
        eager = super().list_commands(ctx)
        lazy = sorted(self._lazy_commands)
        return list(dict.fromkeys([*eager, *lazy]))

    def get_command(self, ctx, cmd_name):  # type: ignore[override]
        command = super().get_command(ctx, cmd_name)
        if command is not None:
            return command
        spec = self._lazy_commands.get(cmd_name)
        if spec is None:
            return None
        module = _import_cached(spec.module)
        cmd = getattr(module, spec.attr)
        self.add_command(cmd, cmd_name)
        return cmd


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
    train_module = _import_cached("fishtools.segment.train")
    TrainConfigCls = train_module.TrainConfig
    run_train = train_module.run_train

    models_path = path / "models"
    if not models_path.exists():
        raise click.ClickException(
            f"Models path {models_path} does not exist. If this is the correct directory, create one."
        )

    config_path = models_path / f"{name}.json"
    try:
        raw = config_path.read_text()
        train_config = TrainConfigCls.model_validate_json(_strip_line_comments(raw))
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


@app.command("distill")
@click.argument(
    "path",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
)
@click.argument("outdir")
def distill_command(path: Path, outdir: str) -> None:
    distill_module = _import_cached("fishtools.segment.distill")
    warnings = distill_module.run_distill(path, outdir)
    for message in warnings:
        click.echo(message)


@app.command("run")
@click.argument(
    "volume",
    type=click.Path(exists=True, file_okay=True, dir_okay=True, path_type=Path),
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
@click.option(
    "--ortho-model",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    help="Path to an orthogonal-view UNet checkpoint; applies when --backend unet.",
)
@click.option(
    "--n",
    "num_files",
    default=20,
    show_default=True,
    type=int,
    help="Number of files to process when input is a directory.",
)
@click.option(
    "--seed",
    default=0,
    show_default=True,
    type=int,
    help="Random seed for reproducible file sampling.",
)
@click.option(
    "--pattern",
    default="*.tif",
    show_default=True,
    help="Glob pattern for file discovery in directory mode.",
)
@click.option(
    "--crop",
    "crop_size",
    default=1024,
    show_default=True,
    type=int,
    help="Random crop size (YxX) for batch mode. Use 0 to disable cropping.",
)
@click.option(
    "--vanilla/--no-vanilla",
    default=False,
    show_default=True,
    help="Use standard Cellpose params instead of u-Segment3D (no momentum, no KDE clustering).",
)
def run_command(
    volume: Path,
    model_path: Path,
    channels: str,
    anisotropy: float,
    output_dir: Path | None,
    overwrite: bool,
    normalize_percentiles: str,
    save_flows: bool,
    ortho_model: Path | None,
    num_files: int,
    seed: int,
    pattern: str,
    crop_size: int,
    vanilla: bool,
    *,
    ortho_weights: str | None,
    backend: str,
) -> None:
    from fishtools.segment.run import run as run_cli

    run_cli(
        volume,
        model=model_path,
        channels=channels,
        anisotropy=anisotropy,
        output_dir=output_dir,
        overwrite=overwrite,
        normalize=normalize_percentiles,
        save_flows=save_flows,
        ortho_model=ortho_model,
        ortho_weights=ortho_weights,
        backend=backend,
        num_files=num_files,
        seed=seed,
        pattern=pattern,
        crop_size=crop_size,
        vanilla=vanilla,
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
@click.option(
    "--backend",
    type=click.Choice(["sam", "unet"], case_sensitive=False),
    default="sam",
    show_default=True,
    help="Segmentation backend to export (UNet uses a 2-channel input).",
)
@click.option(
    "--opset",
    default=22,
    show_default=True,
    help="ONNX opset version to target during export.",
)
def trt_build_cmd(model: Path, batch_size: int, backend: str, opset: int) -> None:
    if not torch.cuda.is_available():
        raise click.ClickException("CUDA GPU is required to build a TensorRT engine.")

    from fishtools.segment.train import build_trt_engine

    bsize = 224 if backend.lower() == "unet" else 256
    plan_path = build_trt_engine(
        model_path=model,
        device=torch.device("cuda:0"),
        bsize=bsize,
        batch_size=batch_size,
        backend=backend,
        opset=opset,
    )
    click.echo(f"Saved TensorRT engine to {plan_path}")


@app.command("export")
@click.argument(
    "path",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
)
@click.argument("roi", required=False)
@click.option(
    "--seg-codebook",
    required=True,
    help="Codebook label used for segmentation artifacts (polygons, intensities).",
)
@click.option(
    "--codebook",
    "codebooks",
    multiple=True,
    required=True,
    help="Decoded spots codebook labels to include in the export (repeatable).",
)
@click.option(
    "--channels",
    default="auto",
    show_default=True,
    help="Comma-separated intensity channel list, or 'auto' to discover from intensity_* outputs.",
)
@click.option(
    "--out-dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    help="Optional output directory; defaults under analysis/deconv/segment_export.",
)
@click.option(
    "--diag/--no-diag",
    default=False,
    show_default=True,
    help="Emit matching diagnostics for polygons/intensity shards per ROI.",
)
def export_command(
    path: Path,
    roi: str | None,
    seg_codebook: str,
    codebooks: tuple[str, ...],
    channels: str,
    out_dir: Path | None,
    diag: bool,
) -> None:
    """Export Baysor-ready spots plus aggregated per-cell intensities."""

    from fishtools.segment.export import export_cmd as segment_export_cmd

    segment_export_cmd(
        path=path,
        roi=roi,
        seg_codebook=seg_codebook,
        codebooks=codebooks,
        channels=channels,
        out_dir=out_dir,
        diag=diag,
    )


def _postproc_single(
    masks_path: Path,
    output: Path | None,
    sigma_val: float | tuple[float, float, float],
    max_expansion: int,
    erosion_fwhm_frac: float,
    v_min: int,
    min_contact_fraction: float,
    backend: str,
    skip_smooth: bool,
    skip_absorb: bool,
    skip_donate: bool,
) -> None:
    """Process a single masks file."""
    import time

    import numpy as np
    import tifffile

    from fishtools.segment.postproc3d import (
        absorb_encircled_rois,
        compute_metadata_and_adjacency,
        donate_small_cells,
        gaussian_erosion_to_margin_and_scale,
        gaussian_smooth_labels,
        gaussian_smooth_labels_cupy,
        relabel_connected_components,
    )

    click.echo(f"Loading masks from {masks_path}...")
    masks = tifffile.imread(masks_path)

    if masks.ndim == 2:
        masks = masks[None, ...]
    if masks.ndim != 3:
        raise click.ClickException(f"Expected 2D or 3D masks, got shape {masks.shape}")
    if not np.issubdtype(masks.dtype, np.integer):
        masks = masks.astype(np.int32)

    click.echo(f"Loaded masks {masks.shape}, dtype={masks.dtype}")
    t_start = time.perf_counter()

    # Phase 1: Gaussian smooth
    if not skip_smooth:
        _, bg_scale = gaussian_erosion_to_margin_and_scale(
            sigma=float(sigma_val) if isinstance(sigma_val, (int, float)) else max(sigma_val),
            fwhm_fraction=erosion_fwhm_frac,
        )
        click.echo(
            f"Phase 1: Gaussian smoothing (σ={sigma_val}, max_expansion={max_expansion}, "
            f"bg_scale={bg_scale:.4f}, backend={backend})..."
        )
        t0 = time.perf_counter()

        if backend.lower() == "cupy":
            try:
                import cupy

                _ = cupy.cuda.runtime.getDevice()
                masks = gaussian_smooth_labels_cupy(
                    masks,
                    sigma=sigma_val,
                    in_place=False,
                    bg_scale=bg_scale,
                    max_expansion=max_expansion,
                )
                click.echo("  Using CuPy-accelerated backend")
            except Exception:
                click.echo("  CuPy not available, falling back to CPU")
                masks = gaussian_smooth_labels(
                    masks,
                    sigma=sigma_val,
                    in_place=False,
                    bg_scale=bg_scale,
                    max_expansion=max_expansion,
                )
        else:
            masks = gaussian_smooth_labels(
                masks,
                sigma=sigma_val,
                in_place=False,
                bg_scale=bg_scale,
                max_expansion=max_expansion,
            )

        click.echo(f"  ✓ Phase 1 done: {(time.perf_counter() - t0) * 1000:.1f} ms")

    # Phase 1.5: Absorb encircled ROIs
    if not skip_absorb:
        click.echo("Phase 1.5: Absorbing encircled ROIs...")
        t0 = time.perf_counter()
        masks = absorb_encircled_rois(masks, in_place=False)
        click.echo(f"  ✓ Phase 1.5 done: {(time.perf_counter() - t0) * 1000:.1f} ms")

    # Phase 2: Relabel connected components
    click.echo("Phase 2: Relabeling connected components...")
    t0 = time.perf_counter()
    masks = relabel_connected_components(masks, in_place=False)
    click.echo(f"  ✓ Phase 2 done: {(time.perf_counter() - t0) * 1000:.1f} ms")

    # Phase 3 & 4: Compute metadata and donate small cells
    if not skip_donate and v_min > 0:
        click.echo("Phase 3: Computing metadata and adjacency...")
        t0 = time.perf_counter()
        volumes, adjacency, contact_areas = compute_metadata_and_adjacency(masks)
        click.echo(f"  ✓ Phase 3 done: {(time.perf_counter() - t0) * 1000:.1f} ms")

        click.echo(f"Phase 4: Donating small cells (V_min={v_min})...")
        t0 = time.perf_counter()
        masks = donate_small_cells(
            masks,
            volumes=volumes,
            adjacency=adjacency,
            contact_areas=contact_areas,
            V_min=v_min,
            min_contact_fraction=min_contact_fraction,
            in_place=False,
        )
        click.echo(f"  ✓ Phase 4 done: {(time.perf_counter() - t0) * 1000:.1f} ms")

    t_total = time.perf_counter() - t_start
    click.echo(f"\nTotal time: {t_total * 1000:.1f} ms")

    # Determine output path
    if output is None:
        output = masks_path.parent / f"{masks_path.stem}_postproc.tif"

    click.echo(f"Saving to {output}...")
    tifffile.imwrite(output, masks.astype(np.uint16))
    click.echo("Done.")


@app.command("postproc")
@click.argument(
    "masks_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=True, path_type=Path),
)
@click.option(
    "--output",
    "-o",
    type=click.Path(file_okay=True, dir_okay=True, path_type=Path),
    help="Output path. For single file: output file. For directory: output directory (defaults to same).",
)
@click.option(
    "--pattern",
    default="*_masks.tif",
    show_default=True,
    help="Glob pattern for finding mask files in directory mode.",
)
@click.option(
    "--sigma",
    default="1.5,3.0,3.0",
    show_default=True,
    help="Gaussian sigma (ZYX) for smoothing. Can be scalar or comma-separated ZYX values.",
)
@click.option(
    "--max-expansion",
    default=1,
    show_default=True,
    type=int,
    help="Max voxels labels can expand beyond original foreground.",
)
@click.option(
    "--erosion-fwhm-frac",
    default=-0.1,
    show_default=True,
    type=float,
    help="Erosion as fraction of FWHM. Negative = dilation.",
)
@click.option(
    "--v-min",
    default=4000,
    show_default=True,
    type=int,
    help="Volume threshold (voxels) for small cell donation.",
)
@click.option(
    "--min-contact-fraction",
    default=0.0,
    show_default=True,
    type=float,
    help="Minimum contact_area/volume ratio to donate.",
)
@click.option(
    "--backend",
    type=click.Choice(["cpu", "cupy"], case_sensitive=False),
    default="cupy",
    show_default=True,
    help="Backend for Gaussian smoothing.",
)
@click.option(
    "--skip-smooth/--no-skip-smooth",
    default=False,
    show_default=True,
    help="Skip Gaussian smoothing phase.",
)
@click.option(
    "--skip-absorb/--no-skip-absorb",
    default=False,
    show_default=True,
    help="Skip encircled ROI absorption phase.",
)
@click.option(
    "--skip-donate/--no-skip-donate",
    default=False,
    show_default=True,
    help="Skip small cell donation phase.",
)
def postproc_command(
    masks_path: Path,
    output: Path | None,
    pattern: str,
    sigma: str,
    max_expansion: int,
    erosion_fwhm_frac: float,
    v_min: int,
    min_contact_fraction: float,
    backend: str,
    skip_smooth: bool,
    skip_absorb: bool,
    skip_donate: bool,
) -> None:
    """Post-process 3D segmentation masks.

    Runs a 4-phase pipeline: Gaussian smoothing, encircled ROI absorption,
    connected component relabeling, and small cell donation.

    MASKS_PATH can be a single file or a directory. If a directory, processes
    all files matching --pattern (default: *_masks.tif).
    """
    # Parse sigma
    sigma_parts = sigma.split(",")
    if len(sigma_parts) == 1:
        sigma_val: float | tuple[float, float, float] = float(sigma_parts[0])
    elif len(sigma_parts) == 3:
        sigma_val = tuple(float(s) for s in sigma_parts)  # type: ignore[assignment]
    else:
        raise click.ClickException("sigma must be a single value or 3 comma-separated ZYX values")

    if masks_path.is_file():
        _postproc_single(
            masks_path,
            output,
            sigma_val,
            max_expansion,
            erosion_fwhm_frac,
            v_min,
            min_contact_fraction,
            backend,
            skip_smooth,
            skip_absorb,
            skip_donate,
        )
    else:
        # Directory mode
        files = sorted(masks_path.glob(pattern))
        if not files:
            raise click.ClickException(f"No files matching '{pattern}' found in {masks_path}")

        click.echo(f"Found {len(files)} files matching '{pattern}'")
        out_dir = output if output else masks_path

        try:
            for i, f in enumerate(files, 1):
                click.echo(f"\n{'=' * 60}")
                click.echo(f"[{i}/{len(files)}] Processing {f.name}")
                click.echo("=" * 60)
                out_file = out_dir / f"{f.stem}_postproc.tif"
                _postproc_single(
                    f,
                    out_file,
                    sigma_val,
                    max_expansion,
                    erosion_fwhm_frac,
                    v_min,
                    min_contact_fraction,
                    backend,
                    skip_smooth,
                    skip_absorb,
                    skip_donate,
                )
        except (KeyboardInterrupt, TaskCancelledException):
            click.echo("\n\nInterrupted by user. Exiting...")
            raise SystemExit(1)

        click.echo(f"\n{'=' * 60}")
        click.echo(f"Completed processing {len(files)} files.")


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
    default=None,
    type=click.IntRange(1, None),
    help="Number of images to sample per ROI. Default: 50 for z, 20 for ortho.",
)
@click.option(
    "--anisotropy",
    default=None,
    type=click.IntRange(1, None),
    help="Z scale factor (ortho mode). Default: 2 for zarr, 4 otherwise.",
)
@click.option("--channels", help="Indices or metadata names, comma-separated.")
@click.option(
    "--crop", default=0, show_default=True, type=click.IntRange(0, None), help="Trim pixels at borders."
)
@click.option(
    "--threads", "-t", default=8, show_default=True, type=click.IntRange(1, 64), help="Parallel workers."
)
@click.option("--upscale", type=float, help="Additional spatial upscale factor applied before saving.")
@click.option("--seed", type=int, default=42, show_default=True, help="Random seed for sampling.")
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
@click.option(
    "--masks",
    type=click.Path(exists=True, dir_okay=True, path_type=Path),
    help="Path to a label mask file (TIF or Zarr) to extract alongside the registered stack.",
)
@click.option(
    "--enrich-boundaries",
    type=click.Path(dir_okay=True, path_type=Path),
    default=None,
    help="Mask for diversity scoring. Default: output_segmentation-sam.zarr. Use --no-enrich-boundaries to disable.",
)
@click.option(
    "--no-enrich-boundaries",
    is_flag=True,
    default=False,
    help="Disable boundary enrichment sampling.",
)
@click.option(
    "--overwrite/--no-overwrite",
    default=False,
    show_default=True,
    help="Allow writing into an existing output directory when --out is set.",
)
@click.option(
    "--roi-points",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="ImageJ ROI file (.roi or .zip) with point coordinates for targeted extraction. Zarr mode only.",
)
def extract_command(
    mode: str,
    path: Path,
    roi: str | None,
    codebook: str,
    out: Path | None,
    dz: int,
    n: int | None,
    anisotropy: int | None,
    channels: str | None,
    crop: int,
    threads: int,
    upscale: float | None,
    seed: int | None,
    every: int,
    max_from: str | None,
    zarr: bool,
    masks: Path | None,
    enrich_boundaries: Path | str | None,
    no_enrich_boundaries: bool,
    overwrite: bool,
    roi_points: Path | None,
) -> None:
    from fishtools.segment.extract import cmd_extract

    # Apply mode-specific default for n
    n_value = n if n is not None else (20 if mode.lower() == "ortho" else 50)
    # Apply zarr-specific default for anisotropy (only relevant for ortho mode)
    if mode.lower() == "ortho":
        anisotropy_value = anisotropy if anisotropy is not None else (2 if zarr else 4)
    else:
        anisotropy_value = anisotropy if anisotropy is not None else 4

    # Resolve enrich_boundaries: None means disabled, "AUTO" means use default, Path means explicit
    enrich_value: Path | str | None = None if no_enrich_boundaries else enrich_boundaries

    # If a custom output directory is provided and already populated, require --overwrite
    if out is not None and out.exists() and any(out.iterdir()) and not overwrite:
        raise click.ClickException(
            f"Output directory {out} already exists and is not empty; use --overwrite to reuse it."
        )

    cmd_extract(
        mode,
        path,
        roi=roi,
        codebook=codebook,
        out=out,
        dz=dz,
        n=n_value,
        anisotropy=anisotropy_value,
        channels=channels,
        crop=crop,
        threads=threads,
        upscale=upscale,
        seed=seed,
        every=every,
        max_from=max_from,
        use_zarr=zarr,
        masks=masks,
        enrich_boundaries=enrich_value,
        roi_points=roi_points,
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
    "--pattern",
    default="*.tif",
    show_default=True,
    help="Glob pattern for files in directory mode (mask files excluded).",
)
@click.option(
    "--dz", default=1, show_default=True, type=click.IntRange(1, None), help="Step between Z planes (z mode)."
)
@click.option(
    "--n",
    default=None,
    type=click.IntRange(1, None),
    help="Number of slices/positions to sample. Default: 50 for z, 20 for ortho.",
)
@click.option(
    "--anisotropy",
    default=None,
    type=click.IntRange(1, None),
    help="Z scale factor (ortho mode). Default: 6.",
)
@click.option("--channels", help="Indices or metadata names, comma-separated.")
@click.option(
    "--crop", default=0, show_default=True, type=click.IntRange(0, None), help="Trim pixels at borders."
)
@click.option(
    "--threads", "-t", default=8, show_default=True, type=click.IntRange(1, 64), help="Parallel workers."
)
@click.option("--upscale", type=float, help="Additional spatial upscale factor applied before saving.")
@click.option("--seed", type=int, default=42, show_default=True, help="Random seed for sampling.")
@click.option(
    "--max-from",
    type=click.Path(path_type=Path),
    help="Optional registered stack for max-projection channel.",
)
@click.option("--label", help="Prefix label for outputs; defaults to the input stem.")
@click.option(
    "--masks",
    type=click.Path(exists=True, dir_okay=True, path_type=Path),
    help="Path to a label mask file (TIF or Zarr) to extract alongside the registered stack.",
)
@click.option(
    "--enrich-boundaries",
    type=click.Path(exists=True, dir_okay=True, path_type=Path),
    default=None,
    help="Mask for diversity scoring. Required for enrichment in single-file mode.",
)
@click.option(
    "--overwrite/--no-overwrite",
    default=False,
    show_default=True,
    help="Overwrite existing outputs in directory mode.",
)
@click.option("--debug", is_flag=True, default=False, help="Enable debug logging with timing info.")
def extract_single_command(
    mode: str,
    registered: Path,
    out: Path | None,
    pattern: str,
    dz: int,
    n: int | None,
    anisotropy: int | None,
    channels: str | None,
    crop: int,
    threads: int,
    upscale: float | None,
    seed: int | None,
    max_from: Path | None,
    label: str | None,
    masks: Path | None,
    enrich_boundaries: Path | None,
    overwrite: bool,
    debug: bool,
) -> None:
    """Extract slices from registered stacks for segmentation training.

    REGISTERED can be a single file or a directory. If a directory, processes
    all files matching --pattern (default: *.tif), excluding mask files.
    """
    from loguru import logger

    from fishtools.segment.extract import cmd_extract_single

    if debug:
        logger.enable("fishtools")
        import sys

        logger.add(sys.stderr, level="DEBUG")

    # Apply mode-specific default for n
    n_value = n if n is not None else (20 if mode.lower() == "ortho" else 50)
    anisotropy_value = anisotropy if anisotropy is not None else 6

    if registered.is_file():
        cmd_extract_single(
            mode,
            registered,
            out=out,
            dz=dz,
            n=n_value,
            anisotropy=anisotropy_value,
            channels=channels,
            crop=crop,
            threads=threads,
            upscale=upscale,
            seed=seed,
            max_from=max_from,
            label=label,
            masks=masks,
            enrich_boundaries=enrich_boundaries,
        )
    else:
        # Directory mode - exclude mask files (specifically *_masks.tif outputs)
        files = sorted(
            f for f in registered.glob(pattern) if not f.name.endswith("_masks.tif") and f.is_file()
        )
        if not files:
            raise click.ClickException(f"No non-mask files matching '{pattern}' found in {registered}")

        click.echo(f"Found {len(files)} files matching '{pattern}' (excluding masks)")

        # Determine output directory for idempotency check
        out_dir = out if out is not None else registered / "segment_extract"

        try:
            for i, f in enumerate(files, 1):
                # Check if outputs already exist (idempotency)
                # Label defaults to file stem; output pattern is {label}--{stem}_z*.tif
                check_pattern = (
                    f"{f.stem}--{f.stem}_z*.tif" if mode.lower() == "z" else f"{f.stem}--{f.stem}_ortho*.tif"
                )
                if out_dir.exists():
                    existing = list(out_dir.glob(check_pattern))
                    if existing and not overwrite:
                        click.echo(f"[{i}/{len(files)}] Skipping {f.name} ({len(existing)} outputs exist)")
                        continue

                click.echo(f"\n{'=' * 60}")
                click.echo(f"[{i}/{len(files)}] Processing {f.name}")
                click.echo("=" * 60)
                cmd_extract_single(
                    mode,
                    f,
                    out=out,
                    dz=dz,
                    n=n_value,
                    anisotropy=anisotropy_value,
                    channels=channels,
                    crop=crop,
                    threads=threads,
                    upscale=upscale,
                    seed=seed,
                    max_from=max_from,
                    label=None,  # Use default (file stem) for each file
                    masks=masks,
                    enrich_boundaries=enrich_boundaries,
                )
        except (KeyboardInterrupt, TaskCancelledException):
            click.echo("\n\nInterrupted by user. Exiting...")
            raise SystemExit(1)

        click.echo(f"\n{'=' * 60}")
        click.echo(f"Completed processing {len(files)} files.")


_OVERLAY_LAZY_COMMANDS = {
    "intensity": SimpleNamespace(module="fishtools.segment.overlay_intensity", attr="overlay_intensity"),
}


@app.group(cls=_LazyCommandGroup, lazy_commands=_OVERLAY_LAZY_COMMANDS)
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
    roi: str | None,
    codebook: str,
    seg_codebook: str | None,
    spots_opt: Path | None,
    segmentation_name: str,
    overwrite: bool,
    debug: bool,
) -> None:
    current_roi = roi if roi is not None else "*"
    from fishtools.segment.overlay_spots import overlay as overlay_impl

    # Call the underlying callback function instead of the Click/RichCommand wrapper.
    # `overlay_impl` is a Click command object; invoking it directly would route
    # positional arguments through `RichCommand.main`, causing the TypeError you saw.
    overlay_impl.callback(
        path,
        current_roi,
        codebook,
        spots_opt,
        seg_codebook,
        segmentation_name,
        overwrite,
        debug,
    )


def run(*args, **kwargs):
    from fishtools.segment.run import run as run_cli

    return run_cli(*args, **kwargs)


_LAZY_EXPORT_ATTRS = {
    "TrainConfig": ("fishtools.segment.train", "TrainConfig"),
    "build_trt_engine": ("fishtools.segment.train", "build_trt_engine"),
    "run_train": ("fishtools.segment.train", "run_train"),
}


__all__ = [
    "app",
    "train",
    "run_command",
    "trt_build_cmd",
    "export_command",
    "postproc_command",
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
        module = _import_cached("fishtools.segment.cp_io")
        globals()["cp_io"] = module
        return module
    if name in _LAZY_EXPORT_ATTRS:
        module_name, attr = _LAZY_EXPORT_ATTRS[name]
        value = getattr(_import_cached(module_name), attr)
        globals()[name] = value
        return value
    raise AttributeError(name)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    app()


if __name__ == "__main__":
    main()
