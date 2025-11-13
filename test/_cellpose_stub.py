from __future__ import annotations

import sys
import types


def ensure_cellpose_stub() -> None:
    """Provide lightweight stand-ins for cellpose modules when package isn't installed."""

    if "cellpose" not in sys.modules:
        mod = types.ModuleType("cellpose")
        mod.dynamics = object()
        sys.modules["cellpose"] = mod
    cellpose_mod = sys.modules["cellpose"]

    if "cellpose.contrib" not in sys.modules:
        contrib = types.ModuleType("cellpose.contrib")
        sys.modules["cellpose.contrib"] = contrib
        cellpose_mod.contrib = contrib  # type: ignore[attr-defined]
    if "cellpose.contrib.cellposetrt" not in sys.modules:
        trt_mod = types.ModuleType("cellpose.contrib.cellposetrt")

        def _dummy_trt_build(*_args, **_kwargs):
            raise RuntimeError("cellpose TRT build stub invoked during tests")

        trt_mod.trt_build = _dummy_trt_build  # type: ignore[attr-defined]
        sys.modules["cellpose.contrib.cellposetrt"] = trt_mod
        sys.modules["cellpose.contrib"].cellposetrt = trt_mod  # type: ignore[attr-defined]

    if "cellpose.models" not in sys.modules:
        models_mod = types.ModuleType("cellpose.models")

        class _DummyCellposeModel:  # pylint: disable=too-few-public-methods
            def __init__(self, *_args, **_kwargs):
                raise RuntimeError("cellpose model stub invoked during tests")

        models_mod.CellposeModel = _DummyCellposeModel  # type: ignore[attr-defined]
        sys.modules["cellpose.models"] = models_mod
        cellpose_mod.models = models_mod  # type: ignore[attr-defined]

    if "cellpose.train" not in sys.modules:
        train_mod = types.ModuleType("cellpose.train")

        def _dummy_train_seg(*_args, **_kwargs):
            raise RuntimeError("cellpose train stub invoked during tests")

        train_mod.train_seg = _dummy_train_seg  # type: ignore[attr-defined]
        sys.modules["cellpose.train"] = train_mod
        cellpose_mod.train = train_mod  # type: ignore[attr-defined]

    if "cellpose.train_unet" not in sys.modules:
        train_unet_mod = types.ModuleType("cellpose.train_unet")

        def _dummy_train_unet(*_args, **_kwargs):
            raise RuntimeError("cellpose train_unet stub invoked during tests")

        train_unet_mod.train_seg = _dummy_train_unet  # type: ignore[attr-defined]
        sys.modules["cellpose.train_unet"] = train_unet_mod
        cellpose_mod.train_unet = train_unet_mod  # type: ignore[attr-defined]

    if "cellpose.unet" not in sys.modules:
        unet_mod = types.ModuleType("cellpose.unet")

        class _DummyCellposeUnetModel:  # pylint: disable=too-few-public-methods
            def __init__(self, *_args, **_kwargs):
                raise RuntimeError("cellpose unet stub invoked during tests")

        unet_mod.CellposeUNetModel = _DummyCellposeUnetModel  # type: ignore[attr-defined]
        sys.modules["cellpose.unet"] = unet_mod
        cellpose_mod.unet = unet_mod  # type: ignore[attr-defined]
