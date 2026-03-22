from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


def _load_train_module():  # type: ignore[return-type]
    script_path = Path(__file__).parents[1] / "scripts/segmentation/brdu_edu_train_dinov3_logreg.py"
    spec = importlib.util.spec_from_file_location("brdu_edu_train_dinov3_logreg", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load trainer script from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_cell_id_roi_and_label_supports_dataset_prefixed_roi() -> None:
    module = _load_train_module()
    parse = getattr(module, "_cell_id_roi_and_label", None)
    if parse is None:
        raise AttributeError("brdu_edu_train_dinov3_logreg missing _cell_id_roi_and_label")

    assert parse("2r|39329") == ("2r", 39329)
    assert parse("20251001_JaxA3_Coro11:2r|39329") == ("2r", 39329)


def test_resolve_workspace_root_checks_nvme_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    module = _load_train_module()
    resolve_workspace_root = getattr(module, "_resolve_workspace_root", None)
    if resolve_workspace_root is None:
        raise AttributeError("brdu_edu_train_dinov3_logreg missing _resolve_workspace_root")

    working_root = tmp_path / "working"
    nvme_root = tmp_path / "nvme"
    dataset = "DS1"
    dataset_root = nvme_root / dataset
    dataset_root.mkdir(parents=True)
    monkeypatch.setattr(module, "WORKSPACE_ROOT_BASES", (working_root, nvme_root))

    assert resolve_workspace_root(dataset) == dataset_root
