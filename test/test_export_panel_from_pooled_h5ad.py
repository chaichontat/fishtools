from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import anndata as ad
import numpy as np
import pandas as pd


def _load_module():
    script_path = Path(__file__).parents[1] / "scripts/gam/export_panel_from_pooled_h5ad.py"
    spec = importlib.util.spec_from_file_location("export_panel_from_pooled_h5ad", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load script from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_pooled_export_writes_unique_cell_ids_and_source_batch_columns(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()

    h5ad_path = tmp_path / "pooled.h5ad"
    out_dir = tmp_path / "panel_out"

    obs = pd.DataFrame(
        {
            "ccf_adjusted": ["cortex", "cortex2", "other", "cortex", "cortex2", "cortex"],
            "dataset": ["d1", "d1", "d1", "d2", "d2", "d2"],
            "roi": ["1", "1", "1", "2", "2", "2"],
            "leiden": ["7", "7", "7", "7", "4", "7"],
            "tricycle": np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float32),
            "t_local": np.array([0.2, 0.3, 0.4, 0.6, 0.2, 0.8], dtype=np.float32),
            "r_um": np.array([10.0, 15.0, 20.0, 25.0, 30.0, 35.0], dtype=np.float32),
            "total_counts": np.array([10, 20, 30, 40, 50, 60], dtype=np.float32),
        },
        index=pd.Index(["a", "a", "b", "b", "c", "c"], dtype=object),
    )

    X = np.array(
        [
            [1, 2, 3],
            [2, 3, 4],
            [3, 4, 5],
            [4, 5, 6],
            [5, 6, 7],
            [6, 7, 8],
        ],
        dtype=np.float32,
    )
    var = pd.DataFrame(index=pd.Index(["G1", "G2", "G3"], dtype=object))
    obsm = {
        "principal": np.array(
            [
                [0.2, 10.0],
                [0.3, 15.0],
                [0.4, 20.0],
                [0.6, 25.0],
                [0.2, 30.0],
                [0.8, 35.0],
            ],
            dtype=np.float32,
        ),
        "AP_ML_um": np.array(
            [
                [1.0, 10.0],
                [2.0, 20.0],
                [3.0, 30.0],
                [4.0, 40.0],
                [5.0, 50.0],
                [6.0, 60.0],
            ],
            dtype=np.float32,
        ),
    }
    adata = ad.AnnData(X=X, obs=obs, var=var, obsm=obsm, layers={"raw": X.copy()})
    adata.write_h5ad(h5ad_path)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_panel_from_pooled_h5ad.py",
            str(h5ad_path),
            "--out-dir",
            str(out_dir),
            "--region-col",
            "ccf_adjusted",
            "--region",
            "cortex,cortex2",
            "--dataset-col",
            "dataset",
            "--roi-col",
            "roi",
            "--theta-col",
            "tricycle",
            "--t-min",
            "0.0",
            "--t-max",
            "1.0",
            "--r-min",
            "0.0",
            "--genes",
            "all",
        ],
    )
    rc = module.main()
    assert rc == 0

    cells = pd.read_csv(out_dir / "cells.tsv", sep="\t")
    counts = pd.read_csv(out_dir / "counts.tsv", sep="\t")

    # one "other" region row dropped => 5 rows kept
    assert len(cells) == 5
    assert len(counts) == 5
    assert "source" in cells.columns
    assert "batch" in cells.columns
    assert set(cells["source"]) == {"d1.1", "d2.2"}
    assert set(cells["batch"]) == {"d1", "d2"}
    assert cells["cell_id"].nunique() == len(cells)
    assert counts["cell_id"].tolist() == cells["cell_id"].tolist()


def test_pooled_export_filters_genes_by_fraction_expressing(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()

    h5ad_path = tmp_path / "pooled_expr_filter.h5ad"
    out_dir = tmp_path / "panel_out_expr_filter"

    n = 1000
    obs = pd.DataFrame(
        {
            "ccf_adjusted": ["cortex"] * n,
            "dataset": ["d1"] * n,
            "roi": ["1"] * n,
            "tricycle": np.linspace(0.0, 2.0 * np.pi, n, dtype=np.float32),
            "t_local": np.linspace(0.0, 1.0, n, dtype=np.float32),
            "r_um": np.linspace(10.0, 20.0, n, dtype=np.float32),
            "total_counts": np.full((n,), 10.0, dtype=np.float32),
        },
        index=pd.Index([f"c{i:04d}" for i in range(n)], dtype=object),
    )
    obsm = {"AP_ML_um": np.zeros((n, 2), dtype=np.float32)}

    X = np.zeros((n, 3), dtype=np.float32)
    # Expressed in all cells.
    X[:, 0] = 1.0
    # Expressed in 2/1000 cells => 0.2% (kept).
    X[:2, 1] = 1.0
    # Expressed in 1/1000 cells => 0.1% (dropped because threshold is >0.1%).
    X[:1, 2] = 1.0

    var = pd.DataFrame(index=pd.Index(["G_all", "G_keep", "G_drop"], dtype=object))
    adata = ad.AnnData(X=X, obs=obs, var=var, obsm=obsm, layers={"raw": X.copy()})
    adata.write_h5ad(h5ad_path)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_panel_from_pooled_h5ad.py",
            str(h5ad_path),
            "--out-dir",
            str(out_dir),
            "--region-col",
            "ccf_adjusted",
            "--region",
            "cortex",
            "--dataset-col",
            "dataset",
            "--roi-col",
            "roi",
            "--theta-col",
            "tricycle",
            "--t-col",
            "t_local",
            "--t-min",
            "0.0",
            "--t-max",
            "1.0",
            "--r-min",
            "-1.0",
            "--genes",
            "all",
        ],
    )
    rc = module.main()
    assert rc == 0

    counts = pd.read_csv(out_dir / "counts.tsv", sep="\t")
    exported = [c for c in counts.columns if c != "cell_id"]
    assert exported == ["G_all", "G_keep"]


def test_pooled_export_subset_col_and_values_filter_rows(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()

    h5ad_path = tmp_path / "pooled_subset.h5ad"
    out_dir = tmp_path / "panel_out_subset"

    obs = pd.DataFrame(
        {
            "ccf_adjusted": ["cortex", "cortex2", "other", "cortex", "cortex2", "cortex"],
            "dataset": ["d1", "d1", "d1", "d2", "d2", "d2"],
            "roi": ["1", "1", "1", "2", "2", "2"],
            "leiden": ["7", "7", "7", "7", "4", "7"],
            "tricycle": np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float32),
            "t_local": np.array([0.2, 0.3, 0.4, 0.6, 0.2, 0.8], dtype=np.float32),
            "r_um": np.array([10.0, 15.0, 20.0, 25.0, 30.0, 35.0], dtype=np.float32),
            "total_counts": np.array([10, 20, 30, 40, 50, 60], dtype=np.float32),
        },
        index=pd.Index(["a", "a", "b", "b", "c", "c"], dtype=object),
    )

    X = np.array(
        [
            [1, 2, 3],
            [2, 3, 4],
            [3, 4, 5],
            [4, 5, 6],
            [5, 6, 7],
            [6, 7, 8],
        ],
        dtype=np.float32,
    )
    var = pd.DataFrame(index=pd.Index(["G1", "G2", "G3"], dtype=object))
    obsm = {
        "principal": np.array(
            [
                [0.2, 10.0],
                [0.3, 15.0],
                [0.4, 20.0],
                [0.6, 25.0],
                [0.2, 30.0],
                [0.8, 35.0],
            ],
            dtype=np.float32,
        ),
        "AP_ML_um": np.array(
            [
                [1.0, 10.0],
                [2.0, 20.0],
                [3.0, 30.0],
                [4.0, 40.0],
                [5.0, 50.0],
                [6.0, 60.0],
            ],
            dtype=np.float32,
        ),
    }
    adata = ad.AnnData(X=X, obs=obs, var=var, obsm=obsm, layers={"raw": X.copy()})
    adata.write_h5ad(h5ad_path)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_panel_from_pooled_h5ad.py",
            str(h5ad_path),
            "--out-dir",
            str(out_dir),
            "--region-col",
            "ccf_adjusted",
            "--region",
            "cortex,cortex2",
            "--dataset-col",
            "dataset",
            "--roi-col",
            "roi",
            "--theta-col",
            "tricycle",
            "--subset-col",
            "leiden",
            "--subset-values",
            "7",
	            "--t-min",
	            "0.0",
	            "--t-max",
	            "1.0",
	            "--r-min",
	            "0.0",
	            "--genes",
	            "all",
	        ],
	    )
    rc = module.main()
    assert rc == 0

    cells = pd.read_csv(out_dir / "cells.tsv", sep="\t")
    counts = pd.read_csv(out_dir / "counts.tsv", sep="\t")

    # Region filtering drops the one 'other' row, leaving 5. Subsetting leiden==7 drops one row (leiden==4), leaving 4.
    assert len(cells) == 4
    assert len(counts) == 4


def test_pooled_export_drop_nans_allows_nonfinite_r_when_enabled(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()

    h5ad_path = tmp_path / "pooled_with_nans.h5ad"
    out_dir = tmp_path / "panel_out_drop_nans"

    obs = pd.DataFrame(
        {
            "ccf_adjusted": ["cortex", "cortex2", "cortex", "cortex", "cortex2"],
            "dataset": ["d1", "d1", "d2", "d1", "d2"],
            "roi": ["1", "1", "2", "1", "2"],
            "tricycle": np.array([0.1, 0.2, 0.3, 0.4, 0.35], dtype=np.float32),
            "t_local": np.array([0.2, 0.3, 0.2, 0.25, 0.28], dtype=np.float32),
            "r_um": np.array([10.0, np.nan, 20.0, 12.0, 22.0], dtype=np.float32),
            "total_counts": np.array([10, 20, 30, 40, 50], dtype=np.float32),
        },
        index=pd.Index(["a", "a", "b", "c", "d"], dtype=object),
    )

    X = np.array(
        [
            [1, 2, 3],
            [2, 3, 4],
            [3, 4, 5],
            [4, 5, 6],
            [5, 6, 7],
        ],
        dtype=np.float32,
    )
    var = pd.DataFrame(index=pd.Index(["G1", "G2", "G3"], dtype=object))
    obsm = {
        "AP_ML_um": np.array(
            [
                [1.0, 10.0],
                [2.0, 20.0],
                [3.0, 30.0],
                [4.0, 40.0],
                [5.0, 50.0],
            ],
            dtype=np.float32,
        ),
    }
    adata = ad.AnnData(X=X, obs=obs, var=var, obsm=obsm, layers={"raw": X.copy()})
    adata.write_h5ad(h5ad_path)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_panel_from_pooled_h5ad.py",
            str(h5ad_path),
            "--out-dir",
            str(out_dir),
            "--region-col",
            "ccf_adjusted",
            "--region",
            "cortex,cortex2",
            "--dataset-col",
            "dataset",
            "--roi-col",
            "roi",
            "--theta-col",
            "tricycle",
            "--t-col",
            "t_local",
            "--t-min",
            "0.0",
            "--t-max",
            "0.75",
            "--r-min",
            "0.0",
            "--genes",
            "all",
        ],
    )
    rc = module.main()
    assert rc == 0

    cells = pd.read_csv(out_dir / "cells.tsv", sep="\t")
    counts = pd.read_csv(out_dir / "counts.tsv", sep="\t")
    assert len(cells) == 4
    assert len(counts) == 4
    assert np.isfinite(cells["r_um"]).all()


def test_pooled_export_no_drop_nans_fails_on_nonfinite_r(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()

    h5ad_path = tmp_path / "pooled_with_nans_no_drop.h5ad"
    out_dir = tmp_path / "panel_out_no_drop_nans"

    obs = pd.DataFrame(
        {
            "ccf_adjusted": ["cortex", "cortex2", "cortex", "cortex", "cortex2"],
            "dataset": ["d1", "d1", "d2", "d1", "d2"],
            "roi": ["1", "1", "2", "1", "2"],
            "tricycle": np.array([0.1, 0.2, 0.3, 0.4, 0.35], dtype=np.float32),
            "t_local": np.array([0.2, 0.3, 0.2, 0.25, 0.28], dtype=np.float32),
            "r_um": np.array([10.0, np.nan, 20.0, 12.0, 22.0], dtype=np.float32),
            "total_counts": np.array([10, 20, 30, 40, 50], dtype=np.float32),
        },
        index=pd.Index(["a", "a", "b", "c", "d"], dtype=object),
    )

    X = np.array(
        [
            [1, 2, 3],
            [2, 3, 4],
            [3, 4, 5],
            [4, 5, 6],
            [5, 6, 7],
        ],
        dtype=np.float32,
    )
    var = pd.DataFrame(index=pd.Index(["G1", "G2", "G3"], dtype=object))
    obsm = {
        "AP_ML_um": np.array(
            [
                [1.0, 10.0],
                [2.0, 20.0],
                [3.0, 30.0],
                [4.0, 40.0],
                [5.0, 50.0],
            ],
            dtype=np.float32,
        ),
    }
    adata = ad.AnnData(X=X, obs=obs, var=var, obsm=obsm, layers={"raw": X.copy()})
    adata.write_h5ad(h5ad_path)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_panel_from_pooled_h5ad.py",
            str(h5ad_path),
            "--out-dir",
            str(out_dir),
            "--region-col",
            "ccf_adjusted",
            "--region",
            "cortex,cortex2",
            "--dataset-col",
            "dataset",
            "--roi-col",
            "roi",
            "--theta-col",
            "tricycle",
            "--t-col",
            "t_local",
            "--t-min",
            "0.0",
            "--t-max",
            "0.75",
            "--r-min",
            "0.0",
            "--no-drop-nans",
            "--genes",
            "all",
        ],
    )

    try:
        module.main()
        raise AssertionError("Expected export to fail with --no-drop-nans when r_um is non-finite.")
    except ValueError as e:
        assert "obs.r_um must be finite" in str(e)


def test_pooled_export_theta_from_obsm_overrides_missing_obs_theta(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()

    h5ad_path = tmp_path / "pooled_theta_obsm.h5ad"
    out_dir = tmp_path / "panel_out_theta_obsm"

    obs = pd.DataFrame(
        {
            "ccf_adjusted": ["cortex", "cortex2", "cortex", "cortex2"],
            "dataset": ["d1", "d1", "d2", "d2"],
            "roi": ["1", "1", "2", "2"],
            "t_local": np.array([0.2, 0.3, 0.2, 0.25], dtype=np.float32),
            "r_um": np.array([10.0, 15.0, 20.0, 25.0], dtype=np.float32),
            "total_counts": np.array([10, 20, 30, 40], dtype=np.float32),
        },
        index=pd.Index(["a", "b", "c", "d"], dtype=object),
    )

    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]], dtype=np.float32)
    var = pd.DataFrame(index=pd.Index(["Nr2f1", "G2"], dtype=object))
    obsm = {
        "AP_ML_um": np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]], dtype=np.float32),
        "cycle_xy": np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]], dtype=np.float32),
    }
    adata = ad.AnnData(X=X, obs=obs, var=var, obsm=obsm, layers={"raw": X.copy()})
    adata.write_h5ad(h5ad_path)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_panel_from_pooled_h5ad.py",
            str(h5ad_path),
            "--out-dir",
            str(out_dir),
            "--region-col",
            "ccf_adjusted",
            "--region",
            "cortex,cortex2",
            "--dataset-col",
            "dataset",
            "--roi-col",
            "roi",
            "--theta-obsm",
            "cycle_xy",
            "--t-min",
            "0.0",
            "--t-max",
            "1.0",
            "--r-min",
            "-1.0",
            "--genes",
            "Nr2f1",
        ],
    )
    rc = module.main()
    assert rc == 0

    cells = pd.read_csv(out_dir / "cells.tsv", sep="\t")
    assert "theta" in cells.columns
    assert np.isfinite(cells["theta"]).all()


def test_pooled_export_no_theta_skips_theta_requirement_and_column(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()

    h5ad_path = tmp_path / "pooled_no_theta.h5ad"
    out_dir = tmp_path / "panel_out_no_theta"

    obs = pd.DataFrame(
        {
            "ccf_adjusted": ["cortex", "cortex2", "cortex", "cortex2"],
            "dataset": ["d1", "d1", "d2", "d2"],
            "roi": ["1", "1", "2", "2"],
            "t_local": np.array([0.2, 0.3, 0.2, 0.25], dtype=np.float32),
            "r_um": np.array([10.0, 15.0, 20.0, 25.0], dtype=np.float32),
            "total_counts": np.array([10, 20, 30, 40], dtype=np.float32),
        },
        index=pd.Index(["a", "b", "c", "d"], dtype=object),
    )
    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]], dtype=np.float32)
    var = pd.DataFrame(index=pd.Index(["Nr2f1", "G2"], dtype=object))
    obsm = {
        "AP_ML_um": np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]], dtype=np.float32),
    }
    adata = ad.AnnData(X=X, obs=obs, var=var, obsm=obsm, layers={"raw": X.copy()})
    adata.write_h5ad(h5ad_path)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_panel_from_pooled_h5ad.py",
            str(h5ad_path),
            "--out-dir",
            str(out_dir),
            "--region-col",
            "ccf_adjusted",
            "--region",
            "cortex,cortex2",
            "--dataset-col",
            "dataset",
            "--roi-col",
            "roi",
            "--no-theta",
            "--t-min",
            "0.0",
            "--t-max",
            "1.0",
            "--r-min",
            "-1.0",
            "--genes",
            "Nr2f1",
        ],
    )
    rc = module.main()
    assert rc == 0

    cells = pd.read_csv(out_dir / "cells.tsv", sep="\t")
    assert "theta" not in cells.columns

    with (out_dir / "panel_meta.json").open() as f:
        meta = json.load(f)
    assert bool(meta["theta_exported"]) is False
