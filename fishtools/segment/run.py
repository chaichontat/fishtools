from __future__ import annotations

import datetime
import pickle
import shutil
from pathlib import Path
from typing import Literal

import click
import numpy as np
import polars as pl
import tifffile
import torch
from loguru import logger
from pydantic import BaseModel, Field, model_validator
from skimage.measure import regionprops_table

from fishtools.io.workspace import Workspace
from fishtools.segment.train import plan_path_for_device
from fishtools.utils.logging import setup_cli_logging

# warnings.filterwarnings('error')

TIFF_COMPRESSION = 22610


class RunConfig(BaseModel):
    volume_path: Path = Field(description="Path to a fused 4D TIFF (Z,C,Y,X) volume.")
    model_path: Path = Field(description="Pretrained Cellpose model to load for inference.")
    backend: Literal["sam", "unet"] = Field(
        default="sam",
        description="Model backend to use: 'sam' for transformer, 'unet' for legacy UNet.",
    )

    channels: tuple[int, ...] = Field(
        default=(1, 2),
        description="Channel indices exposed to Cellpose. UNet requires two channels; SAM can accept more.",
    )
    anisotropy: float = Field(default=4.0, gt=0.0, description="Voxel Z-to-XY anisotropy passed to Cellpose.")
    output_dir: Path | None = Field(
        default=None,
        description="Optional output directory; defaults to <volume>/../cellpose when omitted.",
    )
    overwrite: bool = Field(default=False, description="Recompute tiles even when artifacts already exist.")
    use_gpu: bool = Field(default=True, description="Request GPU execution when available.")
    normalize_percentiles: tuple[float, float] = Field(
        default=(1.0, 99.9),
        description="Percentiles used for Cellpose normalization low/high cutoffs.",
    )
    save_flows: bool = Field(
        default=False,
        description="Persist raw network flows (dP + cell probabilities) alongside masks.",
    )
    ortho_model_path: Path | None = Field(
        default=None,
        description="Optional UNet orthogonal-view model to supply when using the UNet backend.",
    )
    ortho_weights: tuple[float, float, float] | None = Field(
        default=None,
        description="Optional weights applied to the (XY, YZ, ZX) passes when aggregating Cellpose 3D flows.",
    )
    momentum: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Momentum coefficient for suppressed gradient descent (u-Segment3D). 0.0 = standard Euler, 0.95 = recommended.",
    )
    step_decay: float = Field(
        default=0.01,
        ge=0.0,
        description="Step decay factor for gradient descent (u-Segment3D). 0.0 = constant step, ~0.01 = recommended.",
    )
    flow2D_smooth: float = Field(
        default=1.0,
        ge=0.0,
        description="Gaussian sigma for pre-smoothing 2D flows before 3D aggregation (u-Segment3D). 0.0 = no smoothing, 1.0 = recommended.",
    )
    use_kde_clustering: bool = Field(
        default=True,
        description="Use u-Segment3D KDE-based clustering instead of peak detection + seed growth.",
    )
    kde_sigma: float = Field(
        default=1.0,
        ge=0.0,
        description="Gaussian sigma for KDE smoothing of endpoint histogram (u-Segment3D).",
    )
    kde_threshold_k: float = Field(
        default=1.0,
        description="Threshold multiplier k for adaptive threshold: mean(rho) + k*std(rho).",
    )
    use_variance_fusion: bool = Field(
        default=True,
        description="Use inverse-variance weighted fusion for 3D flow aggregation (u-Segment3D). Automatically downweights noisy planes.",
    )
    variance_alpha_flow: float = Field(
        default=0.5,
        gt=0.0,
        description="Alpha for flow variance weighting. Larger = more averaging, smaller = trust smoothest. 0.5 = conservative.",
    )
    variance_alpha_cellprob: float = Field(
        default=1e-5,
        gt=0.0,
        description="Alpha for cellprob variance weighting. 1e-5 = aggressive (strongly trust smooth predictions).",
    )

    @model_validator(mode="after")
    def _validate_config(self) -> RunConfig:
        # Semantic validation: channels must be distinct and non-negative
        num_channels = len(self.channels)
        if num_channels < 2:
            raise ValueError("At least two channel indices are required for 3D Cellpose.")
        if len(set(self.channels)) != num_channels:
            raise ValueError("Channel indices must be distinct.")
        if self.backend == "unet" and num_channels != 2:
            raise ValueError("UNet backend expects exactly two channel indices.")
        if any(idx < 0 for idx in self.channels):
            raise ValueError("Channel indices must be non-negative.")

        # Semantic validation: percentiles must be ordered
        low, high = self.normalize_percentiles
        if not 0.0 <= low < high <= 100.0:
            raise ValueError("Normalization percentiles must satisfy 0 <= low < high <= 100.")

        # Semantic validation: ortho weights must be positive and length three
        if self.ortho_weights is not None:
            if len(self.ortho_weights) != 3:
                raise ValueError("ortho_weights must provide exactly three values (XY,YZ,ZX).")
            if any(weight < 0 for weight in self.ortho_weights):
                raise ValueError("ortho_weights must be non-negative.")
            if all(weight == 0 for weight in self.ortho_weights):
                raise ValueError("At least one ortho weight must be positive.")

        if self.ortho_model_path is not None and not self.ortho_model_path.is_file():
            raise ValueError(f"ortho_model_path {self.ortho_model_path} does not exist or is not a file.")

        # Path validation: output_dir cannot be an existing file
        if self.output_dir is not None and self.output_dir.is_file():
            raise ValueError("output_dir must be a directory, not a file path.")

        return self


#     config: RunConfig,
#     working_stack: Path,
# ) -> tuple[int, int, int, int]:
#     if volume.ndim != 4:
#         raise ValueError("Registered volume must be 4D with axes (Z,C,Y,X).")

#     if max(config.channels) >= volume.shape[1]:
#         raise ValueError(
#             f"Requested channels {config.channels} exceed available channels {volume.shape[1]} in {config.volume_path}."
#         )

#     slice_ = volume[:, list(config.channels), :, :]
#     array = np.asarray(slice_)
#     tifffile.imwrite(
#         working_stack,
#         array,

#         bigtiff=True,
#         metadata={"axes": "ZCYX", "channels": config.channels},
#     )
#     del volume
#     return tuple(int(dim) for dim in array.shape)


def _discover_workspace_rois(volume_path: Path) -> tuple[Workspace, list[str]]:
    """Resolve the workspace root for the given volume and return its ROI list.

    The segmentation CLI is typically invoked on files nested under
    `<workspace>/analysis/deconv/...`. We therefore ascend the directory tree
    until we encounter the canonical `*.DONE` workspace marker, then leverage
    the `Workspace` helper to enumerate ROI names. We bound traversal depth to
    avoid walking arbitrarily far up the filesystem when a caller passes an
    unexpected path.
    """
    candidate = volume_path.parent
    steps = 0
    while True:
        if not candidate.is_dir():
            raise ValueError(f"Volume path {volume_path} is not located inside a valid workspace directory.")
        if any(child.is_file() and child.suffix == ".DONE" for child in candidate.iterdir()):
            workspace = Workspace(candidate)
            return workspace, workspace.rois
        parent = candidate.parent
        steps += 1
        if steps > 4 or parent == candidate:
            break
        candidate = parent
    raise ValueError(
        "Unable to infer workspace root from the supplied volume; ensure a *.DONE marker exists "
        "at or above the analysis directory."
    )


def _find_trt_plan(model_path: Path) -> tuple[Path, str] | None:
    """Return the TensorRT plan path paired with the raw device name, if available."""
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        return None

    device_index = 0
    device_name = torch.cuda.get_device_name(device_index)
    plan_candidate = plan_path_for_device(model_path, device_name)
    if plan_candidate.is_file():
        return plan_candidate, device_name
    return None


def _cellpose(model, image: np.ndarray, *, config: RunConfig):
    """Run Cellpose 3D eval on a single tile.

    Parameter mapping preserved from legacy pipeline for compatibility with trained models.
    Normalization uses tile-based percentile stretching; flow/diameter/rescale settings
    match the training regime.
    """
    backend = config.backend
    normalization = {
        "lowhigh": None,
        "normalize": True,
        "percentile": config.normalize_percentiles,
        "norm3D": True,
        "tile_norm_smooth3D": 1,
        "sharpen_radius": 0,
        "smooth_radius": 0,
        "invert": False,
    }

    eval_image = image
    backend_kwargs: dict[str, tuple[int, ...] | int]
    if backend == "sam":
        sam_channels = [c - 1 for c in config.channels]
        eval_image = eval_image[:, sam_channels, :, :]
        backend_kwargs = {"z_axis": 0}
    else:
        backend_kwargs = {"channels": config.channels}

    masks, flows, styles = model.eval(
        eval_image,  # [:, :, :400, :1386],
        channel_axis=1,
        normalize=normalization,
        batch_size=1,
        anisotropy=4,
        flow_threshold=0.4,
        cellprob_threshold=0,
        flow3D_smooth=3,
        resample=False,
        niter=1000,
        # stitch_threshold=0.24,
        diameter=60,
        do_3D=True,
        min_size=8000,
        ortho_weights=config.ortho_weights,
        momentum=config.momentum,
        step_decay=config.step_decay,
        flow2D_smooth=config.flow2D_smooth,
        use_kde_clustering=config.use_kde_clustering,
        kde_sigma=config.kde_sigma,
        kde_threshold_k=config.kde_threshold_k,
        use_variance_fusion=config.use_variance_fusion,
        variance_alpha_flow=config.variance_alpha_flow,
        variance_alpha_cellprob=config.variance_alpha_cellprob,
        # bsize=224,
        # augment=True,
        **backend_kwargs,
    )

    return masks, flows, styles


def _build_label_metadata(
    masks: np.ndarray,
    flow_field: np.ndarray,
    *,
    use_gpu: bool,
) -> pl.DataFrame:
    """Generate per-label metadata including flow-error QC metrics."""

    if masks.size == 0:
        return pl.DataFrame(
            {
                "label": pl.Series([], dtype=pl.UInt32),
                "flow_error": pl.Series([], dtype=pl.Float32),
                "voxel_count": pl.Series([], dtype=pl.UInt64),
            }
        )

    mask_int = np.asarray(masks, dtype=np.int32)
    props_dict = regionprops_table(mask_int, properties=("label", "area", "bbox", "centroid"))
    props_df = pl.DataFrame(props_dict)
    if props_df.is_empty():
        return pl.DataFrame(
            {
                "label": pl.Series([], dtype=pl.UInt32),
                "flow_error": pl.Series([], dtype=pl.Float32),
                "voxel_count": pl.Series([], dtype=pl.UInt64),
            }
        )

    props_df = props_df.with_columns(
        pl.col("label").cast(pl.UInt32),
        pl.col("area").cast(pl.UInt64).alias("voxel_count"),
    ).drop("area")

    device = torch.device("cpu")
    if use_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
    elif use_gpu and torch.backends.mps.is_available():
        device = torch.device("mps")

    # flow_errors, dP_masks = dynamics.flow_error(mask_int, np.asarray(flow_field), device=device)
    # del dP_masks

    # labels = props_df.select("label").to_numpy().ravel()
    # flow_df = pl.DataFrame(
    #     {
    #         "label": pl.Series(np.arange(1, flow_errors.size + 1, dtype=np.uint32)),
    #         "flow_error": pl.Series(flow_errors.astype(np.float32)),
    #     }
    # ).filter(pl.col("label").is_in(labels.astype(np.uint32)))

    # .join(flow_df, on="label", how="left").sort("label")
    metadata = props_df

    primary = ["label"]
    other_cols = [col for col in metadata.columns if col not in primary]
    return metadata.select([*(c for c in primary if c in metadata.columns), *other_cols])


def _sample_files_by_size(
    directory: Path,
    *,
    n: int,
    seed: int,
    pattern: str = "*.tif",
) -> list[Path]:
    """Sample n files with uniform size coverage.

    Strategy:
    1. Glob all matching files
    2. Randomly sample min(5*n, total) candidates
    3. Sort candidates by file size
    4. Pick every (len/n)th file for uniform size distribution
    """
    all_files = list(directory.glob(pattern))
    if not all_files:
        raise click.BadParameter(f"No files matching '{pattern}' found in {directory}")

    rng = np.random.default_rng(seed)

    # Sample 5x candidates (or all if fewer available)
    candidate_count = min(5 * n, len(all_files))
    if candidate_count < len(all_files):
        indices = rng.choice(len(all_files), size=candidate_count, replace=False)
        candidates = [all_files[i] for i in indices]
    else:
        candidates = all_files

    # Sort by file size
    candidates.sort(key=lambda p: p.stat().st_size)

    # Pick every Nth for uniform size coverage
    stride = max(1, len(candidates) // n)
    selected = candidates[::stride][:n]

    return selected


def _run_single_volume(
    volume_path: Path,
    cellpose_model,
    config: RunConfig,
    *,
    crop_size: int = 0,
    crop_seed: int = 0,
) -> Path:
    """Run segmentation on a single volume and return mask path."""
    img = tifffile.imread(volume_path)

    # Random crop if enabled (assumes ZCYX or ZYX layout)
    if crop_size > 0:
        rng = np.random.default_rng(crop_seed)
        *leading, h, w = img.shape
        if h > crop_size and w > crop_size:
            y0 = rng.integers(0, h - crop_size)
            x0 = rng.integers(0, w - crop_size)
            img = img[..., y0 : y0 + crop_size, x0 : x0 + crop_size]
            logger.info(f"Cropped to {crop_size}x{crop_size} at ({y0}, {x0})")

    masks, flows, styles = _cellpose(cellpose_model, img, config=config)
    metadata_df = _build_label_metadata(masks, flows[1], use_gpu=config.use_gpu)

    # Save outputs
    out = volume_path.parent.parent / "segment2_5d"
    out.mkdir(exist_ok=True)
    pickle_path = out / f"{volume_path.stem}.pkl"
    dest_volume = pickle_path.with_name(volume_path.name)
    if dest_volume.resolve() != volume_path.resolve():
        shutil.copyfile(volume_path, dest_volume)
    mask_path = pickle_path.with_name(volume_path.stem + "_masks.tif")
    tifffile.imwrite(
        mask_path,
        masks.astype(np.uint32),
        compression="zstd",
    )
    metadata_path = pickle_path.with_name(volume_path.stem + "_labels.parquet")
    metadata_df.write_parquet(metadata_path)

    sidecar = {
        "config": {
            "model_path": str(config.model_path),
            "backend": config.backend,
            "channels": list(config.channels),
            "anisotropy": config.anisotropy,
            "normalize_percentiles": list(config.normalize_percentiles),
            "ortho_model_path": str(config.ortho_model_path) if config.ortho_model_path else None,
            "ortho_weights": list(config.ortho_weights) if config.ortho_weights is not None else None,
        },
    }
    with pickle_path.open("wb") as handle:
        pickle.dump({"flows": flows, "styles": styles, **sidecar}, handle)

    if config.save_flows:
        flow_path = pickle_path.with_name(volume_path.stem + "_flows.npz")
        np.savez_compressed(
            flow_path,
            dp=np.asarray(flows[1], dtype=np.float32),
            cellprob=np.asarray(flows[2], dtype=np.float32),
        )
        logger.info(f"Saved flows to {flow_path}")

    logger.info(f"Wrote masks to {mask_path}")
    return mask_path


def run(
    volume: Path,
    *,
    model: Path,
    channels: str = "1,2",
    anisotropy: float = 4.0,
    output_dir: Path | None = None,
    overwrite: bool = False,
    normalize: str = "1.0,99.0",
    save_flows: bool = False,
    ortho_model: Path | None = None,
    ortho_weights: str | None = None,
    backend: str = "sam",
    num_files: int = 20,
    seed: int = 0,
    pattern: str = "*.tif",
    crop_size: int = 0,
    momentum: float = 0.95,
    step_decay: float = 0.01,
    flow2D_smooth: float = 1.0,
    use_kde_clustering: bool = True,
    kde_sigma: float = 2.0,
    kde_threshold_k: float = 1.0,
    use_variance_fusion: bool = True,
    variance_alpha_flow: float = 0.5,
    variance_alpha_cellprob: float = 1e-5,
    vanilla: bool = False,
):
    # --vanilla overrides u-Segment3D params with standard Cellpose defaults
    if vanilla:
        momentum = 0.0
        step_decay = 0.0
        flow2D_smooth = 1
        use_kde_clustering = True
        use_variance_fusion = False

    backend_normalized = backend.lower()
    if backend_normalized not in {"sam", "unet"}:
        raise click.BadParameter(f"Unsupported backend {backend!r}; choose 'sam' or 'unet'.")

    # Handle directory input (batch mode) vs single file
    if volume.is_dir():
        files = _sample_files_by_size(volume, n=num_files, seed=seed, pattern=pattern)
        logger.info(f"Batch mode: processing {len(files)} files from {volume}")
        batch_mode = True
    else:
        files = [volume]
        batch_mode = False

    setup_cli_logging(
        files[0],
        component="segment.run",
        file="segment-run",
        extra={
            "backend": backend_normalized,
            "volume": volume.name,
            "batch_mode": batch_mode,
            "num_files": len(files),
        },
    )

    channels_tuple = tuple(map(int, channels.split(",")))
    normalize_tuple = tuple(map(float, normalize.split(",")))
    ortho_tuple = tuple(map(float, ortho_weights.split(","))) if ortho_weights is not None else None

    # Build config (volume_path will be updated per-file)
    config = RunConfig(
        volume_path=files[0],
        model_path=model,
        backend=backend_normalized,
        anisotropy=anisotropy,
        output_dir=output_dir,
        overwrite=overwrite,
        use_gpu=True,
        normalize_percentiles=normalize_tuple,
        channels=channels_tuple,
        save_flows=save_flows,
        ortho_model_path=ortho_model,
        ortho_weights=ortho_tuple,
        momentum=momentum,
        step_decay=step_decay,
        flow2D_smooth=flow2D_smooth,
        use_kde_clustering=use_kde_clustering,
        kde_sigma=kde_sigma,
        kde_threshold_k=kde_threshold_k,
        use_variance_fusion=use_variance_fusion,
        variance_alpha_flow=variance_alpha_flow,
        variance_alpha_cellprob=variance_alpha_cellprob,
    )

    logger.info(f"Segment run configuration: {config.model_dump()}")

    # Load model once (expensive operation)
    if config.backend == "sam":
        from cellpose.contrib.packed_infer import PackedCellposeModel as TorchModel
        from cellpose.contrib.packed_infer import PackedCellposeModelTRT as TRTModel
    else:
        from cellpose.contrib.packed_infer import CellposeUNetModel as TorchModel
        from cellpose.contrib.packed_infer import CellposeUNetModelTRT as TRTModel

    plan_selection = _find_trt_plan(config.model_path)

    ortho_plan_path = None
    if config.ortho_model_path is not None:
        ortho_plan_selection = _find_trt_plan(config.ortho_model_path)
        if ortho_plan_selection is not None:
            ortho_plan_path, ortho_device_name = ortho_plan_selection
            logger.info(
                f"Using TensorRT plan {ortho_plan_path.name} for ortho model on CUDA device '{ortho_device_name}'."
            )
        else:
            raise FileNotFoundError(
                f"TensorRT plan not found for ortho model at {config.ortho_model_path}. "
                f"When using TensorRT for the main model, ortho model must also have a TRT plan built."
            )

    if plan_selection is not None:
        plan_path, device_name = plan_selection
        plan_mtime = plan_path.stat().st_mtime
        plan_time_local = datetime.datetime.fromtimestamp(plan_mtime).strftime("%Y-%m-%d %H:%M:%S")
        logger.info(
            f"Using TensorRT plan {plan_path.name} for CUDA device '{device_name}' "
            f"(backend={config.backend}, mtime={plan_time_local})."
        )
        ortho_kwargs = {}
        if ortho_plan_path is not None:
            ortho_kwargs["pretrained_model_ortho"] = str(ortho_plan_path)
        elif config.ortho_model_path is not None:
            ortho_kwargs["pretrained_model_ortho"] = str(config.ortho_model_path)

        trt_kwargs = {"gpu": config.use_gpu, "pretrained_model": str(plan_path), **ortho_kwargs}
        cellpose_model = TRTModel(**trt_kwargs)
    else:
        logger.info(f"TensorRT plan not found; falling back to Torch for backend={config.backend}.")
        ortho_kwargs = {}
        if config.ortho_model_path is not None:
            ortho_kwargs["pretrained_model_ortho"] = str(config.ortho_model_path)

        cellpose_model = TorchModel(
            gpu=config.use_gpu,
            pretrained_model=str(model),
            **ortho_kwargs,
        )

    # Process files
    results: list[Path] = []
    for i, file_path in enumerate(files, 1):
        logger.info(f"[{i}/{len(files)}] Processing {file_path.name}")
        # Per-file seed for reproducible crops (combine global seed with file index)
        crop_seed = seed + i
        try:
            mask_path = _run_single_volume(
                file_path,
                cellpose_model,
                config,
                crop_size=0,
                crop_seed=crop_seed,
            )
            results.append(mask_path)
        except Exception as e:
            if batch_mode:
                logger.warning(f"Failed {file_path.name}: {e}")
                continue
            raise

    if batch_mode:
        logger.info(f"Batch complete: {len(results)}/{len(files)} succeeded")

    return results
