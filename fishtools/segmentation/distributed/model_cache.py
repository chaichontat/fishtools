"""
Cellpose model caching for distributed workers.

Provides a Dask WorkerPlugin that initializes and caches a CellPose model
once per worker, avoiding repeated model loading overhead.
"""

import logging
from pathlib import Path
from typing import Any

import distributed
from distributed import WorkerPlugin

from fishtools.segment.train import IS_CELLPOSE_SAM, plan_path_for_device

logger = logging.getLogger("rich")


def _build_packed_cellpose_model(model_kwargs: dict[str, Any]):
    """Instantiate a PackedCellpose model, preferring TensorRT plans when available."""
    import torch
    from cellpose.contrib.packed_infer import (
        PackedCellposeModel,
        PackedCellposeModelTRT,
        PackedCellposeUNetModel,
        PackedCellposeUNetModelTRT,
    )

    resolved_kwargs = dict(model_kwargs)
    backend = resolved_kwargs.pop("backend", "sam").lower()
    if backend not in {"sam", "unet"}:
        raise ValueError("backend must be either 'sam' or 'unet'.")

    pretrained_model = resolved_kwargs.get("pretrained_model")
    if pretrained_model is None:
        raise ValueError("model_kwargs must include 'pretrained_model'.")

    pretrained_path = Path(pretrained_model)
    resolved_kwargs["pretrained_model"] = str(pretrained_path)

    plan_selection: tuple[Path, str] | None = None
    if torch is not None and getattr(torch.cuda, "is_available", lambda: False)():
        if getattr(torch.cuda, "device_count", lambda: 0)() > 0:
            device_index = 0
            device_name = torch.cuda.get_device_name(device_index)
            plan_candidate = plan_path_for_device(pretrained_path, device_name)
            if plan_candidate.is_file():
                plan_selection = (plan_candidate, device_name)
    plan_candidate = plan_candidate if "plan_candidate" in locals() else plan_path_for_device(pretrained_path, "cuda")

    backend_to_classes = {
        "sam": (PackedCellposeModel, PackedCellposeModelTRT),
        "unet": (PackedCellposeUNetModel, PackedCellposeUNetModelTRT),
    }
    base_cls, trt_cls = backend_to_classes[backend]

    if plan_selection is not None:
        plan_path, device_name = plan_selection
        logger.info(f"Using TensorRT plan {plan_path.name} for CUDA device '{device_name}'")
        trt_kwargs = dict(resolved_kwargs)
        trt_kwargs["pretrained_model"] = str(plan_path)
        trt_kwargs.setdefault("gpu", True)
        return trt_cls(**trt_kwargs)

    raise FileNotFoundError(
        f"TensorRT plan required. Expected plan at {plan_candidate} for the current GPU."
    )

    if backend == "sam" and IS_CELLPOSE_SAM:
        resolved_kwargs.pop("pretrained_model_ortho", None)

    resolved_kwargs.setdefault("gpu", True)
    return base_cls(**resolved_kwargs)


def get_cached_model(model_kwargs: dict[str, Any]):
    """Get model from worker attribute, building if necessary (fallback).

    Uses Dask's documented pattern of storing state on worker.my_attribute.
    Falls back to building the model if the CellposeModelPlugin wasn't registered.
    """
    worker = distributed.get_worker()

    if hasattr(worker, "cellpose_model"):
        return worker.cellpose_model

    # Fallback: build and cache (handles case where plugin wasn't registered)
    logger.warning(f"Worker {worker.name}: Model not cached, building fresh")
    model = _build_packed_cellpose_model(model_kwargs)
    worker.cellpose_model = model
    return model


class CellposeModelPlugin(WorkerPlugin):
    """Plugin to initialize and cache CellPose model once per worker.

    Uses the officially documented pattern of storing state on worker attributes
    (get_worker().my_attribute) rather than get_worker().data which is reserved
    for Dask's internal memory management.

    References:
    - https://distributed.dask.org/en/latest/plugins.html
    - https://stackoverflow.com/questions/58126830/
    """

    name = "cellpose-model-cache"

    def __init__(self, model_kwargs: dict[str, Any]):
        self.model_kwargs = model_kwargs

    def setup(self, worker):
        """Initialize model when worker starts. Stores on worker.cellpose_model."""
        if hasattr(worker, "cellpose_model"):
            logger.info(f"Worker {worker.name}: Model already initialized, skipping")
            return

        backend = self.model_kwargs.get("backend", "sam")
        logger.info(f"Worker {worker.name}: Initializing CellPose model (backend={backend})")
        model = _build_packed_cellpose_model(self.model_kwargs)
        worker.cellpose_model = model
        worker.cellpose_model_kwargs = self.model_kwargs
        logger.info(f"Worker {worker.name}: Model cached on worker.cellpose_model")

    def teardown(self, worker):
        """Clean up model when worker shuts down."""
        if hasattr(worker, "cellpose_model"):
            del worker.cellpose_model
            logger.info(f"Worker {worker.name}: Cleared cellpose_model")
