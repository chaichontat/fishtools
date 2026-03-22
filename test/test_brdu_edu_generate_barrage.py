from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import anndata as ad
import numpy as np
import pandas as pd


def _load_barrage_module():  # type: ignore[return-type]
    script_path = Path(__file__).parents[1] / "scripts/segmentation/brdu_edu_generate_barrage.py"
    spec = importlib.util.spec_from_file_location("brdu_edu_generate_barrage", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load barrage script from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_h5ad(path: Path, obs: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    X = np.empty((len(obs), 0), dtype=np.float32)
    adata = ad.AnnData(X=X, obs=obs)
    adata.obs_names = pd.Index(obs.index.astype(str))
    adata.write_h5ad(path)


def test_brdu_edu_generate_barrage_can_load_per_roi_h5ads_dir(tmp_path: Path) -> None:
    module = _load_barrage_module()
    load = getattr(module, "load_barrage_adata", None)
    if load is None:
        raise AttributeError("brdu_edu_generate_barrage missing load_barrage_adata")

    h5ads_dir = tmp_path / "h5ads"
    required = {
        "roi": str,
        "x": float,
        "y": float,
        "z": float,
        "brdu_mean": float,
        "brdu_min": float,
        "brdu_max": float,
        "brdu_median": float,
        "brdu_std": float,
        "edu_mean": float,
        "edu_min": float,
        "edu_max": float,
        "edu_median": float,
        "edu_std": float,
    }

    obs_a = pd.DataFrame({k: [t("1")] if t is str else [t(1)] for k, t in required.items()}, index=["roiA|1"])
    obs_a["roi"] = "roiA"
    obs_b = pd.DataFrame({k: [t("2")] if t is str else [t(2)] for k, t in required.items()}, index=["roiB|2"])
    obs_b["roi"] = "roiB"

    _write_h5ad(h5ads_dir / "roiA.h5ad", obs_a)
    _write_h5ad(h5ads_dir / "roiB.h5ad", obs_b)

    adata = load(h5ad_path=h5ads_dir, dataset="DS1")
    assert set(adata.obs_names.astype(str).tolist()) == {"roiA|1", "roiB|2"}
    assert set(adata.obs["dataset"].astype(str).unique().tolist()) == {"DS1"}


def test_select_barrage_adata_supports_all_datasets_and_single_dataset() -> None:
    module = _load_barrage_module()
    select = getattr(module, "_select_barrage_adata", None)
    all_datasets = getattr(module, "ALL_DATASETS", None)
    if select is None:
        raise AttributeError("brdu_edu_generate_barrage missing _select_barrage_adata")
    if all_datasets is None:
        raise AttributeError("brdu_edu_generate_barrage missing ALL_DATASETS")

    obs = pd.DataFrame(
        {
            "dataset": ["DS1", "DS2", "DS1"],
            "roi": ["1", "2", "3"],
        },
        index=["a", "b", "c"],
    )
    adata = ad.AnnData(X=np.empty((3, 0), dtype=np.float32), obs=obs)
    adata.obs_names = pd.Index(obs.index.astype(str))

    got_all = select(adata=adata, dataset=all_datasets)
    assert set(got_all.obs_names.astype(str).tolist()) == {"a", "b", "c"}

    got_ds1 = select(adata=adata, dataset="DS1")
    assert set(got_ds1.obs_names.astype(str).tolist()) == {"a", "c"}


def test_cell_id_roi_and_label_supports_dataset_prefixed_roi() -> None:
    module = _load_barrage_module()
    parse = getattr(module, "_cell_id_roi_and_label", None)
    if parse is None:
        raise AttributeError("brdu_edu_generate_barrage missing _cell_id_roi_and_label")

    assert parse("2r|39329") == ("2r", 39329)
    assert parse("20251001_JaxA3_Coro11:2r|39329") == ("2r", 39329)


def test_workspace_for_dataset_checks_nvme_root(monkeypatch, tmp_path: Path) -> None:
    module = _load_barrage_module()
    workspace_for_dataset = getattr(module, "_workspace_for_dataset", None)
    if workspace_for_dataset is None:
        raise AttributeError("brdu_edu_generate_barrage missing _workspace_for_dataset")

    working_root = tmp_path / "working"
    nvme_root = tmp_path / "nvme"
    dataset = "DS1"
    dataset_root = nvme_root / dataset
    dataset_root.mkdir(parents=True)
    (dataset_root / "ready.DONE").write_text("", encoding="utf-8")
    monkeypatch.setattr(module, "WORKSPACE_ROOT_BASES", (working_root, nvme_root))

    ws = workspace_for_dataset(dataset=dataset, cache={})
    assert ws.path == dataset_root.resolve()


def test_select_additional_barrage_cells_preserves_existing_and_adds_new_ones() -> None:
    module = _load_barrage_module()
    select_additional = getattr(module, "_select_additional_barrage_cells", None)
    if select_additional is None:
        raise AttributeError("brdu_edu_generate_barrage missing _select_additional_barrage_cells")

    obs = pd.DataFrame(
        {
            "brdu_mean": [1.0, 2.0, 3.0, 4.0],
            "brdu_std": [1.0, 1.0, 1.0, 1.0],
            "edu_mean": [1.0, 2.0, 3.0, 4.0],
            "edu_std": [1.0, 1.0, 1.0, 1.0],
        },
        index=["existing", "labeled_new", "fresh_a", "fresh_b"],
    )

    preserved, new_cells = select_additional(
        obs=obs,
        feature_cols=["brdu_mean", "brdu_std", "edu_mean", "edu_std"],
        existing_cells=["existing"],
        preselected=["labeled_new"],
        n_cells=2,
        n_bins=2,
        seed=0,
    )

    assert preserved == ["existing"]
    assert new_cells[0] == "labeled_new"
    assert len(new_cells) == 2
    assert "existing" not in new_cells
