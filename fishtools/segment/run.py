from __future__ import annotations

import pickle
import shutil
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import numpy as np
import tifffile
import typer
from loguru import logger
from pydantic import BaseModel, Field, model_validator

from fishtools.io.workspace import Workspace
from fishtools.segment.train import plan_path_for_device

if TYPE_CHECKING:
    from cellpose.contrib.packed_infer import PackedCellposeModelTRT as CellposeModel

TIFF_COMPRESSION = 22610


class RunConfig(BaseModel):
    volume_path: Path = Field(description="Path to a fused 4D TIFF (Z,C,Y,X) volume.")
    model_path: Path = Field(
        description="Pretrained Cellpose model to load for inference."
    )

    channels: tuple[int, int] = Field(
        default=(1, 2),
        description="Pair of channel indices exposed to Cellpose. 1-based indices are computed internally.",
    )
    anisotropy: float = Field(
        default=4.0, gt=0.0, description="Voxel Z-to-XY anisotropy passed to Cellpose."
    )
    output_dir: Path | None = Field(
        default=None,
        description="Optional output directory; defaults to <volume>/../cellpose when omitted.",
    )
    overwrite: bool = Field(
        default=False, description="Recompute tiles even when artifacts already exist."
    )
    use_gpu: bool = Field(
        default=True, description="Request GPU execution when available."
    )
    normalize_percentiles: tuple[float, float] = Field(
        default=(1.0, 99.9),
        description="Percentiles used for Cellpose normalization low/high cutoffs.",
    )
    save_flows: bool = Field(
        default=False,
        description="Persist raw network flows (dP + cell probabilities) alongside masks.",
    )
    ortho_weights: tuple[float, float, float] | None = Field(
        default=None,
        description="Optional weights applied to the (XY, YZ, ZX) passes when aggregating Cellpose 3D flows.",
    )

    @model_validator(mode="after")
    def _validate_config(self) -> "RunConfig":
        # Semantic validation: channels must be distinct and non-negative
        if len(self.channels) != 2:
            raise ValueError(
                "Exactly two channel indices are required for 3D Cellpose."
            )
        if self.channels[0] == self.channels[1]:
            raise ValueError(
                "Channel indices must be distinct to provide a dual-channel input."
            )
        if any(idx < 0 for idx in self.channels):
            raise ValueError("Channel indices must be non-negative.")

        # Semantic validation: percentiles must be ordered
        low, high = self.normalize_percentiles
        if not 0.0 <= low < high <= 100.0:
            raise ValueError(
                "Normalization percentiles must satisfy 0 <= low < high <= 100."
            )

        # Semantic validation: ortho weights must be positive and length three
        if self.ortho_weights is not None:
            if len(self.ortho_weights) != 3:
                raise ValueError("ortho_weights must provide exactly three values (XY,YZ,ZX).")
            if any(weight < 0 for weight in self.ortho_weights):
                raise ValueError("ortho_weights must be non-negative.")
            if all(weight == 0 for weight in self.ortho_weights):
                raise ValueError("At least one ortho weight must be positive.")

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
            raise ValueError(
                f"Volume path {volume_path} is not located inside a valid workspace directory."
            )
        if any(
            child.is_file() and child.suffix == ".DONE" for child in candidate.iterdir()
        ):
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


def _cellpose(model: CellposeModel, image: np.ndarray, *, config: RunConfig):
    """Run Cellpose 3D eval on a single tile.

    Parameter mapping preserved from legacy pipeline for compatibility with trained models.
    Normalization uses tile-based percentile stretching; flow/diameter/rescale settings
    match the training regime.
    """
    IS_CELLPOSE_SAM = version("cellpose").startswith("4.")

    channels = config.channels
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

    masks, flows, styles = model.eval(
        image,
        channel_axis=1,
        normalize=normalization,
        batch_size=1,
        anisotropy=2,
        flow_threshold=0.4,
        cellprob_threshold=-1.0,
        flow3D_smooth=3,
        ortho_weights=config.ortho_weights,
        niter=2000,
        # stitch_threshold=0.24,
        diameter=60,
        do_3D=True,
        # bsize=224,
        augment=True,
        **(dict(channels=channels) if not IS_CELLPOSE_SAM else dict(z_axis=0)),
    )

    return masks, flows, styles


def run(
    volume: Annotated[
        Path,
        typer.Argument(
            help="Path to the fused registered stack (Z,C,Y,X) used for inference.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            resolve_path=True,
        ),
    ],
    model: Annotated[
        Path,
        typer.Option("--model", "-m", help="Pretrained Cellpose model to load."),
    ],
    channels: Annotated[
        str,
        typer.Option(
            "--channels",
            "-c",
            help="Comma-separated pair of channel indices (1-based).",
        ),
    ] = "1,2",
    anisotropy: Annotated[
        float,
        typer.Option("--anisotropy", help="Voxel anisotropy passed to Cellpose."),
    ] = 4.0,
    output_dir: Annotated[
        Path | None,
        typer.Option(
            "--output-dir", help="Directory where masks and metadata are stored."
        ),
    ] = None,
    overwrite: Annotated[
        bool,
        typer.Option(
            "--overwrite/--no-overwrite", help="Overwrite existing mask tiles."
        ),
    ] = False,
    normalize: Annotated[
        str,
        typer.Option(
            "--normalize-percentiles",
            help="Comma-separated low,high percentiles for Cellpose normalization.",
        ),
    ] = "1.0,99.0",
    save_flows: Annotated[
        bool,
        typer.Option(
            "--save-flows/--no-save-flows",
            help="Persist raw network flows to NPZ for downstream reuse.",
        ),
    ] = False,
    ortho_weights: Annotated[
        str | None,
        typer.Option(
            "--ortho-weights",
            help="Comma-separated weights for the (XY,YZ,ZX) model passes.",
        ),
    ] = None,
):
    from cellpose.contrib.packed_infer import (
        PackedCellposeModel as CellposeModel,
    )
    from cellpose.contrib.packed_infer import (
        PackedCellposeModelTRT,
    )

    IS_CELLPOSE_SAM = version("cellpose").startswith("4.")

    try:
        config = RunConfig(
            volume_path=volume,
            model_path=model,
            anisotropy=anisotropy,
            output_dir=output_dir,
            overwrite=overwrite,
            use_gpu=True,
            normalize_percentiles=tuple(map(float, normalize.split(","))),
            channels=tuple(map(int, channels.split(","))),
            save_flows=save_flows,
            ortho_weights=(
                tuple(map(float, ortho_weights.split(",")))
                if ortho_weights is not None
                else None
            ),
        )
    except Exception as exc:
        raise typer.BadParameter(str(exc)) from exc

    logger.info(f"Segment run configuration: {config.model_dump()}")

    try:
        workspace, discovered_rois = _discover_workspace_rois(config.volume_path)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    img = tifffile.imread(volume)
    ortho_kwargs = (
        dict(
            pretrained_model_ortho="/working/cellpose-training/models/embryonic2/ortho"
        )
        if not IS_CELLPOSE_SAM
        else {}
    )
    plan_selection = _find_trt_plan(config.model_path)
    if plan_selection is not None:
        plan_path, device_name = plan_selection
        logger.info(
            f"Using TensorRT plan {plan_path.name} for CUDA device '{device_name}'."
        )
        cellpose_model = PackedCellposeModelTRT(
            gpu=True, pretrained_model=str(plan_path)
        )
    else:
        cellpose_model = CellposeModel(
            gpu=True,
            pretrained_model=model,
            **ortho_kwargs,
        )

    sidecar = {
        "config": {
            "volume_path": str(config.volume_path),
            "model_path": str(config.model_path),
            "channels": list(config.channels),
            "anisotropy": config.anisotropy,
            "normalize_percentiles": list(config.normalize_percentiles),
            "ortho_weights": list(config.ortho_weights)
            if config.ortho_weights is not None
            else None,
        },
    }

    masks, flows, styles = _cellpose(cellpose_model, img, config=config)

    # Save outputs
    out = volume.parent.parent / "segment2_5d"
    out.mkdir(exist_ok=True)
    pickle_path = out / f"{volume.stem}.pkl"
    dest_volume = pickle_path.with_name(volume.name)
    if dest_volume.resolve() != volume.resolve():
        shutil.copyfile(volume, dest_volume)
    mask_path = pickle_path.with_name(volume.stem + "_masks.tif")
    tifffile.imwrite(
        mask_path,
        masks.astype(np.uint32),
        compression="zstd",
    )
    with pickle_path.open("wb") as handle:
        pickle.dump({"flows": flows, "styles": styles, **sidecar}, handle)

    if config.save_flows:
        flow_path = pickle_path.with_name(volume.stem + "_flows.npz")
        np.savez_compressed(
            flow_path,
            dp=np.asarray(flows[1], dtype=np.float32),
            cellprob=np.asarray(flows[2], dtype=np.float32),
        )
        logger.info(f"Saved flows to {flow_path}")

    logger.info(
        f"Wrote masks to {mask_path}"
    )
    roi_count = len(discovered_rois)
    plural = "ROI" if roi_count == 1 else "ROIs"
    logger.info(f"Discovered {roi_count} {plural} in workspace {workspace.path}")
