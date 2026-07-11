from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import rich_click as click

from fishtools.segment.cli import app, main
from fishtools.utils.pretty_print import TaskCancelledException, progress_bar
from fishtools.utils.thumbnails import load_thumbnail_options, thumbnail_rgb
from fishtools.utils.utils import batch_roi

if TYPE_CHECKING:  # pragma: no cover
    from fishtools.segment.train import TrainConfig as TrainConfig


def _strip_line_comments(text: str) -> str:
    """Remove lines that are comments (prefixed by //)."""
    lines = text.splitlines()
    kept = [line for line in lines if not line.lstrip().startswith("//")]
    return "\n".join(kept) + ("\n" if text.endswith("\n") else "")


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
@click.option("--skip-trt", is_flag=True, help="Skip TensorRT engine generation after training.")
def train(
    path: Path,
    name: str,
    use_te: bool,
    te_fp8: bool,
    packed: bool,
    skip_trt: bool,
) -> None:
    from fishtools.segment.train import TrainConfig as TrainConfigCls
    from fishtools.segment.train import run_train

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
    if skip_trt:
        updates["skip_trt"] = True
    if updates:
        train_config = train_config.model_copy(update=updates)

    updated = run_train(name, path, train_config).model_dump_json(indent=2)
    output_path = models_path / f"{name}.trained.json"
    output_path.write_text(updated)


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
    import torch

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
    "--segmentation-name",
    default="output_segmentation-sam_postproc_s1-2-2_v500.zarr",
    show_default=True,
    help="Post-processed segmentation zarr name (contains chunks and intensity outputs).",
)
@click.option(
    "--channels",
    default="auto",
    show_default=True,
    help="Comma-separated intensity channel list, or 'auto' to discover from intensity_* outputs.",
)
@click.option(
    "--thumbnail-scale",
    default=8.0,
    show_default=True,
    type=float,
    help="Scale factor to map RoiSet thumbnail coordinates back to segmentation pixels.",
)
@click.option(
    "--debug",
    default=False,
    is_flag=True,
    help="Emit matching diagnostics for polygons/intensity shards per ROI.",
)
def export_command(
    path: Path,
    roi: str | None,
    seg_codebook: str,
    codebooks: tuple[str, ...],
    segmentation_name: str,
    channels: str,
    thumbnail_scale: float,
    debug: bool,
) -> None:
    """Export Baysor-ready spots plus aggregated per-cell intensities."""

    from fishtools.io.workspace import Workspace
    from fishtools.segment.export import export_cmd as segment_export_cmd

    ws = Workspace(path)
    roi_label = roi if roi is not None else "*"

    rois_to_check = [roi] if roi else ws.rois
    if any((ws.stitch(r, seg_codebook) / segmentation_name).exists() for r in rois_to_check):
        seg_name = segmentation_name
    else:
        raise click.ClickException(
            f"No segmentation outputs found for roi={roi_label!r}, seg_codebook={seg_codebook!r}. "
            f"Checked: {segmentation_name!r}"
        )

    segment_export_cmd(
        path=path,
        roi=roi,
        seg_codebook=seg_codebook,
        codebooks=codebooks,
        segmentation_name=seg_name,
        channels=channels,
        thumbnail_scale=thumbnail_scale,
        diag=debug,
    )


def _parse_xyz_triple(name: str, val: str) -> tuple[float, float, float]:
    s = val.strip().replace(" ", ",")
    parts = [p for p in s.split(",") if p]
    if len(parts) != 3:
        raise click.BadParameter(f"{name} must be a triple like 'x,y,z' (commas/spaces ok).")
    try:
        x, y, z = (float(p) for p in parts)
    except ValueError as exc:
        raise click.BadParameter(f"{name} must contain numeric values.") from exc
    return (x, y, z)


def _parse_zyx_triple(name: str, val: str) -> tuple[float, float, float]:
    s = val.strip().replace(" ", ",")
    parts = [p for p in s.split(",") if p]
    if len(parts) != 3:
        raise click.BadParameter(f"{name} must be a triple like 'z,y,x' (commas/spaces ok).")
    try:
        z, y, x = (float(p) for p in parts)
    except ValueError as exc:
        raise click.BadParameter(f"{name} must contain numeric values.") from exc
    return (z, y, x)


def _parse_labels_csv(val: str) -> list[int]:
    raw = val.strip()
    if not raw:
        return []
    parts = [p.strip() for p in raw.replace(" ", ",").split(",") if p.strip()]
    try:
        return [int(p) for p in parts]
    except ValueError as exc:
        raise click.BadParameter("--labels must be a comma-separated list of integers.") from exc


@app.command("export-mesh")
@click.argument(
    "path",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
)
@click.argument("roi", required=False)
@click.option(
    "--seg-codebook",
    required=True,
    help="Codebook label used for segmentation artifacts (stitch--<roi>+<seg_codebook>).",
)
@click.option(
    "--segmentation-name",
    default="output_segmentation.zarr",
    show_default=True,
    help="Segmentation zarr name (Z,Y,X integer labels).",
)
@click.option(
    "--output",
    type=click.Path(file_okay=True, dir_okay=False, path_type=Path),
    help="Optional output .ply path (defaults to <segmentation.zarr>/mesh.ply).",
)
@click.option(
    "--labels",
    default="",
    show_default=False,
    help="Optional comma-separated label IDs to export (defaults to all non-zero labels).",
)
@click.option(
    "--spacing",
    default="2,1,1",
    show_default=True,
    help="Voxel spacing in ZYX order, as 'z,y,x' (commas/spaces ok).",
)
@click.option(
    "--origin",
    default="0,0,0",
    show_default=True,
    help="World-space origin in ZYX order, as 'z,y,x' (commas/spaces ok).",
)
@click.option(
    "--downsample",
    type=click.IntRange(1, None),
    default=1,
    show_default=True,
    help="Downsample factor for Y,X (1 = native).",
)
@click.option(
    "--progress-bar/--no-progress-bar",
    default=True,
    show_default=True,
    help="Show a progress bar while loading large Zarr volumes.",
)
def export_mesh_command(
    path: Path,
    roi: str | None,
    seg_codebook: str,
    segmentation_name: str,
    output: Path | None,
    labels: str,
    spacing: str,
    origin: str,
    downsample: int,
    progress_bar: bool,
) -> None:
    """Export surface meshes for Blender (PLY)."""

    from fishtools.segment.export_mesh import export_mesh_cmd

    label_ids = _parse_labels_csv(labels) if labels.strip() else None
    export_mesh_cmd(
        path=path,
        roi=roi,
        seg_codebook=seg_codebook,
        segmentation_name=segmentation_name,
        output=output,
        labels=label_ids,
        spacing_zyx=_parse_zyx_triple("--spacing", spacing),
        origin_zyx=_parse_zyx_triple("--origin", origin),
        downsample=downsample,
        show_progress_bar=progress_bar,
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
            except ImportError:
                click.echo("  CuPy not installed, falling back to CPU")
                masks = gaussian_smooth_labels(
                    masks,
                    sigma=sigma_val,
                    in_place=False,
                    bg_scale=bg_scale,
                    max_expansion=max_expansion,
                )
            else:
                try:
                    _ = cupy.cuda.runtime.getDevice()
                except cupy.cuda.runtime.CUDARuntimeError:
                    click.echo("  CUDA unavailable, falling back to CPU")
                    masks = gaussian_smooth_labels(
                        masks,
                        sigma=sigma_val,
                        in_place=False,
                        bg_scale=bg_scale,
                        max_expansion=max_expansion,
                    )
                else:
                    masks = gaussian_smooth_labels_cupy(
                        masks,
                        sigma=sigma_val,
                        in_place=False,
                        bg_scale=bg_scale,
                        max_expansion=max_expansion,
                    )
                    click.echo("  Using CuPy-accelerated backend")
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


@app.command("batch")
@click.argument(
    "workspace",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
)
@click.argument("roi", required=False, default="*", type=str)
@click.option(
    "--codebook",
    required=True,
    type=str,
    help="Codebook label used to locate stitch--ROI+<codebook> folders.",
)
@click.option("--channels", default=None, type=str, help="Comma-separated list of channel names to use.")
@click.option("--overwrite/--no-overwrite", default=False, show_default=True, help="Overwrite existing segmentation.")
@click.option(
    "--stitched-name",
    default="fused_n4.zarr",
    show_default=True,
    help="Stitched zarr filename inside stitch--ROI+CB.",
)
@click.option(
    "--config",
    "-c",
    "config_path",
    default=None,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Explicit path to config.json. Defaults to <path>/../config.json when omitted.",
)
@click.option(
    "--workers-per-gpu",
    default=4,
    show_default=True,
    type=int,
    help="Number of workers to spawn per GPU (>=2 enables multi-worker SpecCluster).",
)
@click.option(
    "--threads-per-worker",
    default=1,
    show_default=True,
    type=int,
    help="Threads per worker (GPU-bound work typically uses 1).",
)
@click.option(
    "--use-localcuda/--no-use-localcuda",
    default=False,
    show_default=True,
    help="If true and workers_per_gpu<=1, use dask-cuda LocalCUDACluster.",
)
@click.option("--n-workers", default=None, type=int, help="For LocalCUDACluster: number of workers (defaults to #GPUs).")
@click.option(
    "--target-ny",
    default=None,
    type=int,
    help="Desired internal Cellpose ny tiles (SAM backend only).",
)
@click.option(
    "--target-nx",
    default=None,
    type=int,
    help="Desired internal Cellpose nx tiles (SAM backend only).",
)
@click.option(
    "--cellpose-only/--no-cellpose-only",
    default=False,
    show_default=True,
    help="Stop after cellpose phase, save intermediate state for later stitching.",
)
@click.option(
    "--stagger-seconds",
    default=5.0,
    show_default=True,
    type=float,
    help="Seconds to stagger worker starts on the same GPU (0 to disable).",
)
@click.option(
    "--roi-retries",
    default=2,
    show_default=True,
    type=int,
    help="Number of retries per ROI when segmentation fails (0 disables retries).",
)
@click.option(
    "--roi-retry-delay-s",
    default=0.0,
    show_default=True,
    type=float,
    help="Delay (seconds) between ROI retries.",
)
def batch_command(
    workspace: Path,
    roi: str,
    codebook: str,
    channels: str | None,
    overwrite: bool,
    stitched_name: str,
    config_path: Path | None,
    workers_per_gpu: int,
    threads_per_worker: int,
    use_localcuda: bool,
    n_workers: int | None,
    target_ny: int | None,
    target_nx: int | None,
    cellpose_only: bool,
    stagger_seconds: float,
    roi_retries: int,
    roi_retry_delay_s: float,
) -> None:
    """Run distributed Cellpose segmentation over stitched Zarr volumes."""
    from fishtools.segmentation.distributed import distributed_segmentation as ds

    callback = ds.run.callback
    if callback is None:  # pragma: no cover
        raise click.ClickException("distributed_segmentation.run callback missing.")

    callback(
        workspace=workspace,
        roi=roi,
        codebook=codebook,
        channels=channels,
        overwrite=overwrite,
        stitched_name=stitched_name,
        config_path=config_path,
        workers_per_gpu=workers_per_gpu,
        threads_per_worker=threads_per_worker,
        use_localcuda=use_localcuda,
        n_workers=n_workers,
        target_ny=target_ny,
        target_nx=target_nx,
        cellpose_only=cellpose_only,
        stagger_seconds=stagger_seconds,
        roi_retries=roi_retries,
        roi_retry_delay_s=roi_retry_delay_s,
    )


@app.command("postproc-batch")
@click.argument(
    "workspace",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
)
@click.argument("rois", nargs=-1, type=str)
@click.option(
    "--output-path",
    default=None,
    type=click.Path(path_type=Path),
    help="Output path (only valid when processing a single input zarr).",
)
@click.option("--blocksize", default=1024, show_default=True, type=int, help="XY block size for tiled processing.")
@click.option(
    "--sigma",
    default="1,2,2",
    show_default=True,
    help="Gaussian smoothing sigma; scalar or 'z,y,x' triple.",
)
@click.option("--v-min", default=500, show_default=True, type=int, help="Minimum volume threshold for small cell donation.")
@click.option("--margin", default=50, show_default=True, type=int, help="Margin parameter (overlap = 2*margin for overlap removal).")
@click.option("--workers-per-gpu", default=4, show_default=True, type=int, help="Workers per GPU.")
@click.option("--overwrite/--no-overwrite", default=False, show_default=True, help="Overwrite existing output.")
def postproc_batch_command(
    workspace: Path,
    rois: tuple[str, ...],
    output_path: Path | None,
    blocksize: int,
    sigma: str,
    v_min: int,
    margin: int,
    workers_per_gpu: int,
    overwrite: bool,
) -> None:
    """Distributed post-processing for stitched segmentation Zarrs in a workspace.

    Post-processes `output_segmentation-sam.zarr` under `analysis/deconv/stitch--ROI+*`
    using the distributed 3D post-processing pipeline.
    """
    import zarr

    from fishtools.io.workspace import Workspace
    from fishtools.segmentation.distributed import distributed_postproc as dpp

    ws = Workspace(workspace)
    if (not rois) or any(roi in {"*", "all"} for roi in rois):
        resolved_rois = ws.rois
    else:
        resolved_rois = ws.resolve_rois(list(rois))

    input_paths: list[Path] = []
    for roi in resolved_rois:
        seg_paths = dpp._iter_segmentation_paths_for_roi(ws, roi)
        if not seg_paths:
            click.echo(f"[{roi}] No {dpp._DEFAULT_SEGMENTATION_NAME} found; skipping.")
            continue
        input_paths.extend(seg_paths)

    if not input_paths:
        click.echo("No segmentation zarrs found to post-process.")
        return

    if output_path is not None and len(input_paths) > 1:
        raise click.ClickException("--output-path can only be used when processing a single input zarr.")

    try:
        sigma_val = dpp._parse_sigma_option(sigma)
    except (ValueError, click.ClickException) as exc:
        raise click.ClickException(str(exc)) from exc

    cluster_kwargs = {
        "workers_per_gpu": workers_per_gpu,
        "threads_per_worker": 1,
    }

    for input_path in input_paths:
        input_zarr = zarr.open(input_path, mode="r")

        resolved_output_path = output_path
        if resolved_output_path is None:
            sigma_str = sigma.replace(",", "-").replace(" ", "")
            resolved_output_path = input_path.parent / f"{input_path.stem}_postproc_s{sigma_str}_v{v_min}.zarr"

        if resolved_output_path.exists() and not overwrite:
            click.echo(f"Output already exists: {resolved_output_path}. Skipping (use --overwrite to force).")
            continue

        dpp.distributed_postproc(
            input_zarr=input_zarr,
            write_path=resolved_output_path,
            blocksize=(input_zarr.shape[0], blocksize, blocksize),
            margin=margin,
            sigma=sigma_val,
            V_min=v_min,
            input_path=input_path,
            cluster_kwargs=cluster_kwargs,
        )


@app.command("extract")
@click.argument("mode", type=click.Choice(["z", "ortho", "maxproj"], case_sensitive=False))
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
    "--dz",
    default=1,
    show_default=True,
    type=click.IntRange(1, None),
    help="Step between Z planes (z/maxproj modes).",
)
@click.option(
    "--n",
    default=None,
    type=click.IntRange(1, None),
    help="Number of images to sample per ROI. Default: 50 for z/maxproj, 20 for ortho.",
)
@click.option(
    "--n-crops",
    default=None,
    type=click.IntRange(1, None),
    help="Number of crops per image (z/maxproj modes). Default: 1.",
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
    n_crops: int | None,
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
    enrich_boundaries: Path | None,
    no_enrich_boundaries: bool,
    overwrite: bool,
    roi_points: Path | None,
) -> None:
    from fishtools.segment.extract import cmd_extract

    # Apply mode-specific default for n
    n_value = n if n is not None else (20 if mode.lower() == "ortho" else 50)
    z_crops_value = n_crops if n_crops is not None else (1 if mode.lower() == "z" else 1)
    # Apply zarr-specific default for anisotropy (only relevant for ortho mode)
    if mode.lower() == "ortho":
        anisotropy_value = anisotropy if anisotropy is not None else (2 if zarr else 4)
    else:
        anisotropy_value = anisotropy if anisotropy is not None else 4

    enable_enrich_boundaries = not no_enrich_boundaries

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
        z_crops_per_file=z_crops_value,
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
        enrich_boundaries=enrich_boundaries,
        enable_enrich_boundaries=enable_enrich_boundaries,
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


@app.command("thumbnail")
@click.argument(
    "path",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
)
@click.argument("roi", required=False, default="*")
@click.option(
    "--seg-codebook",
    type=str,
    default=None,
    help="Segmentation codebook used to locate the mask zarr for boundary overlays (enables outlines).",
)
@click.option(
    "--segmentation-name",
    default="output_segmentation-sam_postproc_s1-2-2_v500.zarr",
    show_default=True,
    help="Segmentation zarr name (Z,Y,X integer labels) inside stitch--ROI+<seg_codebook>.",
)
@click.option(
    "--z-stride",
    type=click.IntRange(min=1),
    default=None,
    help="Generate thumbnail every N Z-planes (overrides --options).",
)
@click.option(
    "--z-range",
    default=None,
    help="Z range as start:end (e.g., 0:50). Empty start/end is allowed (e.g., :50, 10:).",
)
@click.option(
    "--zs",
    "zs_spec",
    default=None,
    help="Comma-separated 0-based Z indices to render (overrides --z-stride/--z-range). Example: --zs 0,5,10",
)
@click.option(
    "--downsample",
    type=click.IntRange(min=1),
    default=None,
    help="Spatial downsampling factor (overrides --options).",
)
@click.option(
    "--channels",
    default=None,
    help=(
        "Comma-separated channel names (from fused.zarr attrs['key']) or 0-based indices to render (max 3 for RGB). "
        "Example: --channels Cy3,ATTO647N or --channels 0,2. Default: first up to 3 channels."
    ),
)
@click.option(
    "--options",
    "thumbnail_options",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    help="JSON file to override thumbnail generation options.",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    help="Output root for thumbnails (defaults to analysis/output/thumbnails).",
)
@click.option("--codebook", "-c", required=True, help="Codebook label for fused.zarr lookup.")
@click.option(
    "--include-n4",
    is_flag=True,
    default=False,
    help="Also generate thumbnails from fused_n4.zarr when present.",
)
@click.option(
    "--boundary-color",
    default="128,128,128",
    show_default=True,
    help="Boundary RGB triplet like '0,255,0' (only used when --seg-codebook is set).",
)
@click.option(
    "--spots",
    "spots_codebook",
    default=None,
    metavar="CODEBOOK",
    help="Overlay detected spots from analysis/output/parquets (ROI+CODEBOOK parquet) on thumbnails.",
)
@click.option(
    "--ccf-rotate/--no-ccf-rotate",
    default=True,
    show_default=True,
    help="Rotate/flip thumbnails using CCF transforms when available.",
)
@batch_roi("stitch--*", include_codebook=True, split_codebook=True)
def thumbnail_command(
    path: Path,
    roi: str,
    seg_codebook: str | None,
    segmentation_name: str,
    z_stride: int | None,
    z_range: str | None,
    zs_spec: str | None,
    downsample: int | None,
    channels: str | None,
    thumbnail_options: Path | None,
    output_dir: Path | None,
    codebook: str,
    include_n4: bool,
    boundary_color: str,
    spots_codebook: str | None,
    ccf_rotate: bool,
) -> None:
    """Generate RGB PNG thumbnails from stitched fused.zarr volumes."""
    import numpy as np
    import zarr
    from loguru import logger
    from scipy.ndimage import rotate as ndimage_rotate

    from fishtools.ccf.landmark import LandmarkRegistrationOutputs
    from fishtools.io.workspace import Workspace
    from fishtools.segment.normalize import sample_percentile

    ws = Workspace(path)
    try:
        thumb_options = load_thumbnail_options(thumbnail_options)
    except Exception as exc:
        raise click.ClickException(f"Invalid --options file: {exc}") from exc

    channels_spec = channels

    from dataclasses import replace

    # When outlines are requested, default to the historical `segment plot` subsample (2)
    # unless the user explicitly overrides via --downsample or --options.
    if seg_codebook is not None and downsample is None and thumbnail_options is None:
        thumb_options = replace(thumb_options, xy_downsample=2)

    if z_stride is not None:
        thumb_options = replace(thumb_options, z_stride=z_stride)
    if downsample is not None:
        thumb_options = replace(thumb_options, xy_downsample=downsample)

    z_start: int | None = None
    z_end: int | None = None
    if z_range is not None:
        parts = z_range.split(":", maxsplit=1)
        if len(parts) != 2:
            raise click.ClickException("Invalid --z-range format. Use start:end (e.g., 0:50).")
        start_raw, end_raw = (p.strip() for p in parts)
        try:
            z_start = int(start_raw) if start_raw else None
            z_end = int(end_raw) if end_raw else None
        except ValueError as exc:
            raise click.ClickException("--z-range must contain integer start/end values.") from exc

    output_root = output_dir if output_dir is not None else ws.output / "thumbnails"
    stitched_dir = ws.stitch(roi, codebook)
    thumbnail_dir = output_root / f"{roi}+{codebook}"

    boundary_rgb = _parse_rgb_triplet(boundary_color)
    mask_path: Path | None = None
    if seg_codebook is not None:
        seg_dir = ws.stitch(roi, seg_codebook)
        mask_path = seg_dir / segmentation_name
        if not mask_path.exists():
            raise click.ClickException(f"ROI '{roi}': segmentation zarr not found at {mask_path}")

    spots_parquet_path: Path | None = None
    if spots_codebook is not None:
        spots_parquet_path = ws.spots_parquet(roi, spots_codebook, must_exist=True)

    pose: tuple[float, bool] | None = None
    if ccf_rotate:
        try:
            pose = _try_read_thumbnail_pose(ws, roi, LandmarkRegistrationOutputs=LandmarkRegistrationOutputs)
        except (OSError, ValueError) as exc:
            raise click.ClickException(f"Failed to read CCF pose transform for ROI '{roi}': {exc}") from exc

    def _spot_marker_radius(*, n_spots: int, xy_downsample: int) -> int:
        # Inspired by `spots plotall` scatter sizing: bigger markers for sparse plots,
        # but capped so dense plots don't turn into solid blobs.
        xy = max(1, int(xy_downsample))
        n = max(1, int(n_spots))
        r_res = int(np.ceil(2.0 / float(np.sqrt(xy))))
        r_den = int(np.ceil(float(np.sqrt(200.0 / float(n)))))
        return int(np.clip(max(r_res, r_den), 1, 3))

    def _overlay_spots_rgb(rgb: np.ndarray, *, rows: np.ndarray, cols: np.ndarray, radius: int) -> None:
        color = np.asarray([255, 0, 0], dtype=np.uint8)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"Expected RGB image shaped (Y,X,3), got shape={rgb.shape}")

        in_bounds = (rows >= 0) & (cols >= 0) & (rows < int(rgb.shape[0])) & (cols < int(rgb.shape[1]))
        if not np.any(in_bounds):
            return
        rows = rows[in_bounds]
        cols = cols[in_bounds]

        r = max(1, int(radius))
        for dy in range(-r, r + 1):
            rr = rows + dy
            in_y = (rr >= 0) & (rr < int(rgb.shape[0]))
            if not np.any(in_y):
                continue
            rr = rr[in_y]
            cc0 = cols[in_y]
            for dx in range(-r, r + 1):
                cc = cc0 + dx
                in_x = (cc >= 0) & (cc < int(rgb.shape[1]))
                if not np.any(in_x):
                    continue
                rgb[rr[in_x], cc[in_x]] = color

    def _process_zarr(*, zarr_path: Path, prefix: str) -> None:
        if not zarr_path.exists():
            logger.warning(f"Skipping ROI '{roi}': {zarr_path.name} not found at {zarr_path}")
            return

        try:
            z_array = zarr.open_array(zarr_path, mode="r")
        except Exception as exc:
            logger.warning(f"Skipping ROI '{roi}': failed to open {zarr_path}: {exc}")
            return

        if z_array.ndim != 4:
            logger.warning(f"Skipping ROI '{roi}': {zarr_path.name} has shape {z_array.shape}, expected 4D.")
            return

        zs, _, _, cs = z_array.shape
        channel_names = _read_channel_names_from_zarr_array(z_array)
        selected_channels = _resolve_thumbnail_channels(
            channels_spec=channels_spec, channel_names=channel_names, channel_count=int(cs)
        )
        preview_c = len(selected_channels)
        if preview_c <= 0:
            logger.warning(f"Skipping ROI '{roi}': {zarr_path.name} has no channels.")
            return

        mask_arr = None
        if mask_path is not None:
            try:
                mask_arr = zarr.open_array(mask_path, mode="r")
            except Exception as exc:
                raise click.ClickException(f"ROI '{roi}': failed to open segmentation zarr at {mask_path}: {exc}") from exc
            if mask_arr.ndim != 3:
                raise click.ClickException(
                    f"ROI '{roi}': segmentation zarr has shape {mask_arr.shape}, expected 3D (Z,Y,X)."
                )
            if int(mask_arr.shape[0]) != int(zs):
                raise click.ClickException(
                    f"ROI '{roi}': Z mismatch between {zarr_path} (Z={zs}) and {mask_path} (Z={mask_arr.shape[0]})."
                )
            if int(mask_arr.shape[1]) != int(z_array.shape[1]) or int(mask_arr.shape[2]) != int(z_array.shape[2]):
                raise click.ClickException(
                    f"ROI '{roi}': XY mismatch between {zarr_path} (YX={z_array.shape[1:3]}) and "
                    f"{mask_path} (YX={mask_arr.shape[1:3]})."
                )

        if zs_spec is not None:
            z_indices = _parse_thumbnail_zs(zs_spec, zs=int(zs))
            if not z_indices:
                logger.warning(f"Skipping ROI '{roi}': --zs resolved to no Z-planes.")
                return
        else:
            start = z_start if z_start is not None else 0
            end = z_end if z_end is not None else zs
            start = max(0, start)
            end = min(zs, end)
            if start >= end:
                logger.warning(f"Skipping ROI '{roi}': no Z-planes to process in range [{start}, {end}).")
                return
            z_indices = list(range(start, end, thumb_options.z_stride))
            if not z_indices:
                logger.warning(f"Skipping ROI '{roi}': no Z-planes to process in range [{start}, {end}).")
                return

        low, high = (1.0, 99.9)
        if thumb_options.percentiles is not None:
            low, high = thumb_options.percentiles

        lowhigh_by_channel: dict[int, np.ndarray] = {}
        missing: list[int] = []
        for ch_idx in selected_channels:
            cache_path = _thumbnail_channel_percentiles_cache_path(
                ws=ws,
                roi=roi,
                codebook=codebook,
                zarr_name=zarr_path.name,
                channel_key=_thumbnail_channel_key(ch_idx, channel_names),
                low=low,
                high=high,
            )
            try:
                cached = _try_load_thumbnail_channel_percentiles(cache_path)
            except (OSError, ValueError) as exc:
                logger.warning(f"Ignoring invalid percentile cache at {cache_path}: {exc}")
                cached = None
            if cached is None:
                missing.append(ch_idx)
            else:
                lowhigh_by_channel[ch_idx] = cached

        if missing:
            try:
                block_y = max(1, min(256, int(z_array.shape[1]) - 1))
                block_x = max(1, min(1024, int(z_array.shape[2]) - 1))
                computed, _ = sample_percentile(
                    z_array,
                    channels=[ch + 1 for ch in missing],
                    block=(block_y, block_x),
                    n=30,
                    low=low,
                    high=high,
                )
                computed = np.asarray(computed, dtype=np.float64)
            except (ValueError, RuntimeError) as exc:
                logger.warning(
                    f"Failed to compute random-crop percentiles for {zarr_path} ({exc}); falling back to strided sampling."
                )
                z_step = max(1, zs // 8)
                y_step = max(1, int(z_array.shape[1]) // 256)
                x_step = max(1, int(z_array.shape[2]) // 256)
                sample = np.asarray(z_array[::z_step, ::y_step, ::x_step, missing])
                bounds = np.percentile(sample, [low, high], axis=(0, 1, 2)).T  # (len(missing), 2)
                computed = np.asarray(bounds, dtype=np.float64)

            if computed.shape != (len(missing), 2):
                raise click.ClickException(
                    f"ROI '{roi}': computed percentile bounds have shape {computed.shape}, expected ({len(missing)}, 2)."
                )

            for out_idx, ch_idx in enumerate(missing):
                cache_path = _thumbnail_channel_percentiles_cache_path(
                    ws=ws,
                    roi=roi,
                    codebook=codebook,
                    zarr_name=zarr_path.name,
                    channel_key=_thumbnail_channel_key(ch_idx, channel_names),
                    low=low,
                    high=high,
                )
                row = np.asarray(computed[out_idx], dtype=np.float64)
                _save_thumbnail_channel_percentiles(cache_path, row)
                lowhigh_by_channel[ch_idx] = row

        lowhigh = np.stack([lowhigh_by_channel[ch_idx] for ch_idx in selected_channels], axis=0).astype(np.float64)

        spots_xy_by_z: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        if spots_parquet_path is not None:
            import polars as pl

            try:
                tileconfig = ws.tileconfig(roi)
            except FileNotFoundError as exc:
                raise click.ClickException(
                    f"ROI '{roi}': TileConfiguration.registered.txt is required for --spots overlay ({exc})."
                ) from exc

            schema = pl.scan_parquet(spots_parquet_path).collect_schema()
            if "y" in schema and "x" in schema:
                x_col = "x"
                y_col = "y"
            elif "y_" in schema and "x_" in schema:
                x_col = "x_"
                y_col = "y_"
            else:
                raise click.ClickException(
                    f"Spots parquet {spots_parquet_path} for ROI '{roi}' is missing required coordinates (x/y or x_/y_)."
                )
            if "z" not in schema:
                raise click.ClickException(f"Spots parquet {spots_parquet_path} for ROI '{roi}' is missing column 'z'.")

            tc_ds = tileconfig.downsample(int(thumb_options.xy_downsample))
            x_offset = float(tc_ds.df["x"].min())
            y_offset = float(tc_ds.df["y"].min())

            zs_needed = sorted(set(z_indices))
            z_min = float(zs_needed[0]) - 0.5
            z_max = float(zs_needed[-1]) + 0.5
            df = (
                pl.scan_parquet(spots_parquet_path)
                .select(
                    x=pl.col(x_col).cast(pl.Float64),
                    y=pl.col(y_col).cast(pl.Float64),
                    z=pl.col("z").cast(pl.Float64),
                )
                .filter(pl.col("z").is_between(z_min, z_max, closed="both"))
                .with_columns(z_idx=(pl.col("z") + 0.5).floor().cast(pl.Int64))
                .filter(pl.col("z_idx").is_in(zs_needed))
                .collect()
            )

            if df.height:
                for z_idx, df_z in df.partition_by("z_idx", as_dict=True).items():
                    z_key = int(z_idx[0]) if isinstance(z_idx, tuple) else int(z_idx)
                    x = df_z.get_column("x").to_numpy() / float(thumb_options.xy_downsample) - x_offset
                    y = df_z.get_column("y").to_numpy() / float(thumb_options.xy_downsample) - y_offset
                    cols = np.floor(x + 0.5).astype(np.int64)
                    rows = np.floor(y + 0.5).astype(np.int64)
                    spots_xy_by_z[z_key] = (rows, cols)

        thumbnail_dir.mkdir(parents=True, exist_ok=True)
        from PIL import Image
        from skimage.segmentation import find_boundaries

        with progress_bar(len(z_indices)) as progress:
            for i in z_indices:
                stem = f"{prefix}{_channels_filename_suffix(channels_spec)}"
                thumbnail_path = thumbnail_dir / f"{stem}_z{i:03d}.png"
                overlay_path = thumbnail_dir / f"{stem}_mask_z{i:03d}.png"

                thumbnail_data = z_array[i, :, :, selected_channels]
                base_raw = thumbnail_rgb(thumbnail_data, options=thumb_options, lowhigh=lowhigh)

                base_to_save = base_raw
                spots_xy = spots_xy_by_z.get(i)
                if spots_xy is not None:
                    rows, cols = spots_xy
                    if rows.size and cols.size:
                        radius = _spot_marker_radius(n_spots=int(rows.size), xy_downsample=int(thumb_options.xy_downsample))
                        base_to_save = base_raw.copy()
                        _overlay_spots_rgb(base_to_save, rows=rows, cols=cols, radius=radius)

                base = _apply_thumbnail_pose(base_to_save, pose=pose, ndimage_rotate=ndimage_rotate) if pose else base_to_save
                Image.fromarray(base, mode="RGB").save(thumbnail_path)
                logger.debug(f"Saved thumbnail for Z-plane {i} to {thumbnail_path}")

                if mask_arr is not None:
                    mask_slice = np.asarray(mask_arr[i, :: thumb_options.xy_downsample, :: thumb_options.xy_downsample])
                    boundaries = find_boundaries(mask_slice, mode="outer")
                    overlay_raw = base_raw.copy()
                    overlay_raw[boundaries] = boundary_rgb
                    if spots_xy is not None:
                        rows, cols = spots_xy
                        if rows.size and cols.size:
                            radius = _spot_marker_radius(
                                n_spots=int(rows.size),
                                xy_downsample=int(thumb_options.xy_downsample),
                            )
                            _overlay_spots_rgb(overlay_raw, rows=rows, cols=cols, radius=radius)
                    overlay = (
                        _apply_thumbnail_pose(overlay_raw, pose=pose, ndimage_rotate=ndimage_rotate) if pose else overlay_raw
                    )
                    Image.fromarray(overlay, mode="RGB").save(overlay_path)
                    logger.debug(f"Saved thumbnail overlay for Z-plane {i} to {overlay_path}")
                progress()

    _process_zarr(zarr_path=stitched_dir / "fused.zarr", prefix="thumbnail")
    highpass_path = stitched_dir / "fused_highpassed.zarr"
    if highpass_path.exists():
        _process_zarr(zarr_path=highpass_path, prefix="thumbnail_highpass")
    if include_n4:
        _process_zarr(zarr_path=stitched_dir / "fused_n4.zarr", prefix="thumbnail_n4")


def _parse_rgb_triplet(val: str) -> tuple[int, int, int]:
    raw = val.strip().replace(" ", ",")
    parts = [p for p in raw.split(",") if p]
    if len(parts) != 3:
        raise click.BadParameter("--boundary-color must be an RGB triplet like '255,255,255'.")
    try:
        r, g, b = (int(p) for p in parts)
    except ValueError as exc:
        raise click.BadParameter("--boundary-color must contain integer values.") from exc
    for x in (r, g, b):
        if x < 0 or x > 255:
            raise click.BadParameter("--boundary-color values must be in [0, 255].")
    return (r, g, b)


def _parse_thumbnail_zs(val: str, *, zs: int) -> list[int]:
    if zs <= 0:
        raise click.ClickException("Invalid --zs: Z dimension must be positive.")

    raw = val.strip()
    if not raw:
        raise click.ClickException("Invalid --zs: must not be empty.")

    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        raise click.ClickException("Invalid --zs: must not be empty.")

    seen: set[int] = set()
    out: list[int] = []
    for part in parts:
        try:
            idx = int(part)
        except ValueError as exc:
            raise click.ClickException(f"Invalid --zs: z index {part!r} is not an integer.") from exc
        if idx < 0:
            raise click.ClickException("Invalid --zs: indices must be >= 0.")
        if idx >= zs:
            raise click.ClickException(f"Invalid --zs: index {idx} out of range for Z={zs}.")
        if idx in seen:
            continue
        seen.add(idx)
        out.append(idx)

    return out


def _normalize_channel_names(names: object) -> list[str] | None:
    if isinstance(names, (list, tuple)):
        return [str(x) for x in names]
    return None


def _read_channel_names_from_zarr_array(arr: object) -> list[str] | None:
    attrs = getattr(arr, "attrs", None)
    if attrs is None:
        return None
    if hasattr(attrs, "get"):
        raw = attrs.get("key") or attrs.get("channel_names")
        return _normalize_channel_names(raw)
    return None


def _resolve_thumbnail_channels(
    *, channels_spec: str | None, channel_names: list[str] | None, channel_count: int
) -> list[int]:
    if channel_count <= 0:
        return []

    if channels_spec is None:
        return list(range(min(3, channel_count)))

    raw = channels_spec.strip().replace(" ", ",")
    parts = [p for p in raw.split(",") if p]
    if not parts:
        raise click.BadParameter("--channels must not be empty.", param_hint="--channels")
    if len(parts) > 3:
        raise click.BadParameter("--channels supports up to 3 channels (RGB).", param_hint="--channels")

    selected: list[int] = []
    for part in parts:
        try:
            idx = int(part)
        except ValueError:
            if not channel_names:
                raise click.BadParameter(
                    "Channel names are missing from fused.zarr metadata (attrs['key']); "
                    "pass numeric indices instead.",
                    param_hint="--channels",
                )
            try:
                idx = channel_names.index(part)
            except ValueError as exc:
                raise click.BadParameter(
                    f"Unknown channel name {part!r}. Available: {channel_names}", param_hint="--channels"
                ) from exc
        if idx < 0:
            raise click.BadParameter("--channels indices must be >= 0.", param_hint="--channels")
        if idx >= channel_count:
            raise click.BadParameter(
                f"--channels index {idx} out of range for C={channel_count}.", param_hint="--channels"
            )
        if idx in selected:
            raise click.BadParameter("--channels must not contain duplicates.", param_hint="--channels")
        selected.append(idx)

    return selected


def _read_similarity2d_angle_rad(tfm_path: Path) -> float:
    if not tfm_path.exists():
        raise FileNotFoundError(f"Missing similarity transform: {tfm_path}")
    for line in tfm_path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("Parameters:"):
            continue
        parts = line.split(":", 1)[1].strip().split()
        if len(parts) < 2:
            raise ValueError(f"Invalid Similarity2DTransform parameters line in {tfm_path}: {line!r}")
        return float(parts[1])
    raise ValueError(f"Missing 'Parameters:' line in Similarity2DTransform file {tfm_path}")


def _try_read_thumbnail_pose(
    ws: Any,
    roi: str,
    *,
    LandmarkRegistrationOutputs: type[Any],
) -> tuple[float, bool] | None:
    import math

    out = LandmarkRegistrationOutputs(ws.ccf_transforms(roi))
    if not out.p1_similarity_tfm.exists():
        return None

    landmarks = out.try_read_p1_landmarks()
    prior_rotation_deg = 0 if landmarks is None else int(landmarks.prior_rotation_deg)
    prior_flip_x = False if landmarks is None else bool(landmarks.prior_flip_x)

    theta_rad = _read_similarity2d_angle_rad(out.p1_similarity_tfm)
    theta_deg = float(math.degrees(theta_rad))
    net_rotation_deg = float(prior_rotation_deg - theta_deg)
    return (net_rotation_deg, prior_flip_x)


def _apply_thumbnail_pose(
    rgb: Any,
    *,
    pose: tuple[float, bool] | None,
    ndimage_rotate: Any,
) -> Any:
    import numpy as np

    if pose is None:
        return rgb

    rotation_deg, flip_x = pose
    out = rgb
    if flip_x:
        out = out[:, ::-1, :]
    if rotation_deg != 0:
        # Expand the canvas so rotation does not crop content.
        out = ndimage_rotate(out, angle=-rotation_deg, reshape=True, order=1, mode="nearest")
    if out.dtype != np.uint8:
        out = np.clip(out, 0, 255).astype(np.uint8)
    return out


def _channels_filename_suffix(channels_spec: str | None) -> str:
    if channels_spec is None:
        return ""
    raw = channels_spec.strip()
    if not raw:
        return ""

    out_chars: list[str] = []
    for ch in raw:
        if ch.isalnum():
            out_chars.append(ch)
            continue
        if ch in {",", " ", "+", ":", ";"}:
            out_chars.append("-")
            continue
        if ch in {"_", "-", "."}:
            out_chars.append(ch)
            continue
        out_chars.append("_")

    cleaned = "".join(out_chars).strip("-_.")
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    if not cleaned:
        return ""
    if len(cleaned) > 80:
        cleaned = cleaned[:80].rstrip("-_.")
    return f"_channels-{cleaned}"


def _thumbnail_channel_key(channel_index: int, channel_names: list[str] | None) -> str:
    if channel_names is not None and channel_index < len(channel_names):
        return str(channel_names[channel_index])
    return f"channel_{channel_index}"


def _sanitize_thumbnail_channel_key(key: str) -> str:
    raw = key.strip()
    if not raw:
        return "unknown"

    out_chars: list[str] = []
    for ch in raw:
        if ch.isalnum():
            out_chars.append(ch)
            continue
        if ch in {"_", "-", "."}:
            out_chars.append(ch)
            continue
        out_chars.append("_")

    cleaned = "".join(out_chars).strip("-_.")
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    if not cleaned:
        return "unknown"
    if len(cleaned) > 80:
        cleaned = cleaned[:80].rstrip("-_.")
    return cleaned


def _thumbnail_channel_percentiles_cache_path(
    *,
    ws: Any,
    roi: str,
    codebook: str,
    zarr_name: str,
    channel_key: str,
    low: float,
    high: float,
) -> Path:
    low_str = str(float(low)).replace(".", "p")
    high_str = str(float(high)).replace(".", "p")
    base = ws.output / "thumbnail_percentiles" / f"{roi}+{codebook}"
    ch = _sanitize_thumbnail_channel_key(channel_key)
    return base / f"{zarr_name}__ch-{ch}__p-{low_str}-{high_str}.npy"


def _try_load_thumbnail_channel_percentiles(path: Path) -> Any | None:
    import numpy as np

    if not path.exists():
        return None
    try:
        arr = np.load(path)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Failed to load cached percentiles at {path}: {exc}") from exc
    if arr.shape != (2,):
        raise ValueError(f"Cached channel percentiles at {path} have shape {arr.shape}; expected (2,).")
    if not np.isfinite(arr).all():
        raise ValueError(f"Cached percentiles at {path} contain non-finite values.")
    return arr.astype(np.float64, copy=False)


def _save_thumbnail_channel_percentiles(path: Path, lowhigh: Any) -> None:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    row = np.asarray(lowhigh, dtype=np.float64)
    if row.shape != (2,):
        raise ValueError(f"Expected lowhigh shape (2,), got shape={row.shape}")
    np.save(tmp, row)
    tmp.with_suffix(tmp.suffix + ".npy").replace(path)


@app.command("plot")
@click.argument(
    "path",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
)
@click.argument("roi", required=False, default="*")
@click.option("--codebook", "-c", required=True, help="Codebook label for stitched fused_n4.zarr lookup.")
@click.option(
    "--seg-codebook",
    type=str,
    default=None,
    help="Segmentation codebook (defaults to --codebook).",
)
@click.option(
    "--segmentation-name",
    default="output_segmentation-sam_postproc_s1-2-2_v500.zarr",
    show_default=True,
    help="Segmentation zarr name (Z,Y,X integer labels) inside stitch--ROI+<seg_codebook>.",
)
@click.option(
    "--image-store",
    default="fused.zarr",
    show_default=True,
    help="Image zarr name inside stitch--ROI+<codebook> (usually fused.zarr).",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    help=(
        "Output root for PNGs (defaults to analysis/output/plots). "
        "Outputs are written under <output-dir>/<roi>+<seg_codebook>/ (like `segment thumbnail`)."
    ),
)
@click.option("--channel", type=click.IntRange(min=0), default=0, show_default=True, help="Channel index.")
@click.option(
    "--subsample",
    type=click.IntRange(min=1),
    default=2,
    show_default=True,
    help="Subsample factor for Y,X (output size is Y//subsample, X//subsample).",
)
@click.option(
    "--z-step",
    type=click.IntRange(min=1),
    default=8,
    show_default=True,
    help="Export every Nth z-slice.",
)
@click.option(
    "--z-range",
    default=None,
    help="Z range as start:end (e.g., 0:50). Empty start/end is allowed (e.g., :50, 10:).",
)
@click.option("--cmap", "cmap_name", default="magma", show_default=True, help="Matplotlib colormap name.")
@click.option("--p-low", type=float, default=1.0, show_default=True, help="Lower percentile for normalization.")
@click.option("--p-high", type=float, default=99.99, show_default=True, help="Upper percentile for normalization.")
@click.option(
    "--boundary-color",
    default="255,255,255",
    show_default=True,
    help="Boundary RGB triplet like '255,255,255'.",
)
@click.option("--overwrite", is_flag=True, default=False, help="Overwrite existing PNGs.")
@batch_roi("stitch--*", include_codebook=True, split_codebook=True)
def plot_command(
    path: Path,
    roi: str,
    codebook: str,
    seg_codebook: str | None,
    segmentation_name: str,
    image_store: str,
    output_dir: Path | None,
    channel: int,
    subsample: int,
    z_step: int,
    z_range: str | None,
    cmap_name: str,
    p_low: float,
    p_high: float,
    boundary_color: str,
    overwrite: bool,
) -> None:
    """Export per-Z PNGs with and without segmentation boundaries."""
    from fishtools.io.workspace import Workspace
    from fishtools.segment.plot import export_zslices_with_boundaries

    ws = Workspace(path)
    seg_cb = seg_codebook or codebook

    z_start: int = 0
    z_end: int | None = None
    if z_range is not None:
        parts = z_range.split(":", maxsplit=1)
        if len(parts) != 2:
            raise click.ClickException("Invalid --z-range format. Use start:end (e.g., 0:50).")
        start_raw, end_raw = (p.strip() for p in parts)
        try:
            z_start = int(start_raw) if start_raw else 0
            z_end = int(end_raw) if end_raw else None
        except ValueError as exc:
            raise click.ClickException("--z-range must contain integer start/end values.") from exc

    stitched_dir = ws.stitch(roi, codebook)
    image_path = stitched_dir / image_store
    if not image_path.exists():
        raise click.ClickException(f"ROI '{roi}': image zarr not found at {image_path}")

    seg_dir = ws.stitch(roi, seg_cb)
    mask_path = seg_dir / segmentation_name
    if not mask_path.exists():
        raise click.ClickException(f"ROI '{roi}': segmentation zarr not found at {mask_path}")

    rgb = _parse_rgb_triplet(boundary_color)

    out_root = (ws.output / "plots") if output_dir is None else output_dir
    out = out_root / f"{roi}+{seg_cb}"

    export_zslices_with_boundaries(
        image_path=image_path,
        mask_path=mask_path,
        output_dir=out,
        channel=channel,
        subsample=subsample,
        boundary_color=rgb,
        z_step=z_step,
        z_start=z_start,
        z_end=z_end,
        cmap_name=cmap_name,
        p_low=p_low,
        p_high=p_high,
        overwrite=overwrite,
    )


LAZY_OVERLAY_COMMANDS: dict[str, SimpleNamespace] = {
    "all": SimpleNamespace(module="fishtools.segment.overlay_all", attr="overlay_all"),
    "intensity": SimpleNamespace(module="fishtools.segment.overlay_intensity", attr="overlay_intensity"),
    "spots": SimpleNamespace(module="fishtools.segment.overlay_spots", attr="overlay"),
}


class LazyGroup(click.Group):
    """Lazy-loading Click group that defers CLI imports until invocation."""

    def __init__(self, *args: Any, lazy_commands: dict[str, SimpleNamespace] | None = None, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._lazy_commands = lazy_commands or {}

    def list_commands(self, ctx: click.Context) -> list[str]:
        eager = super().list_commands(ctx)
        lazy = sorted(self._lazy_commands)
        ordered = list(dict.fromkeys([*eager, *lazy]))
        return ordered

    def get_command(self, ctx: click.Context, cmd_name: str) -> click.Command | None:
        command = super().get_command(ctx, cmd_name)
        if command is not None:
            return command

        spec = self._lazy_commands.get(cmd_name)
        if spec is None:
            return None

        module = import_module(spec.module)
        return getattr(module, spec.attr)


@app.group(cls=LazyGroup, lazy_commands=LAZY_OVERLAY_COMMANDS)
def overlay() -> None:
    """Visualization helpers for segmentation outputs."""


__all__ = [
    "app",
    "train",
    "run_command",
    "batch_command",
    "trt_build_cmd",
    "export_command",
    "postproc_command",
    "extract_command",
    "extract_single_command",
    "thumbnail_command",
    "plot_command",
    "overlay",
]


if __name__ == "__main__":
    main()
