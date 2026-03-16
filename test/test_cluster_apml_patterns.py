from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.gam.cluster_apml_patterns import extract_ref_levels


def _make_small_panel(panel_dir: Path, *, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    n = 140
    theta = rng.uniform(0.0, 2.0 * np.pi, size=n)
    r_um = rng.uniform(0.0, 240.0, size=n)
    ap_um = rng.normal(0.0, 120.0, size=n)
    ml_um = rng.normal(0.0, 100.0, size=n)
    s = rng.lognormal(mean=6.0, sigma=0.25, size=n)
    x = 0.4 + 0.2 * np.cos(theta) + rng.normal(0.0, 0.05, size=n)

    ap_n = ap_um / 120.0
    ml_n = ml_um / 100.0
    r_n = (r_um - np.mean(r_um)) / np.std(r_um)
    cyc = np.cos(theta)
    eta_1 = -6.1 + 0.35 * ap_n - 0.28 * ml_n + 0.20 * r_n
    eta_2 = -6.0 - 0.30 * ap_n + 0.32 * ml_n - 0.25 * r_n
    eta_3 = -6.2 + 0.25 * ap_n + 0.22 * ml_n + 0.40 * r_n * cyc
    eta_4 = -6.1 + 0.18 * ap_n - 0.14 * ml_n - 0.30 * r_n * cyc
    eta_5 = -6.0 + 0.22 * cyc + 0.20 * r_n

    mu = np.vstack(
        [
            s * np.exp(eta_1),
            s * np.exp(eta_2),
            s * np.exp(eta_3),
            s * np.exp(eta_4),
            s * np.exp(eta_5),
        ]
    ).T
    counts = rng.poisson(lam=np.maximum(mu, 1e-4)).astype(np.int64)

    cells = pd.DataFrame(
        {
            "cell_id": [f"c{i:04d}" for i in range(n)],
            "x": x.astype(np.float64),
            "r_um": r_um.astype(np.float64),
            "AP_um": ap_um.astype(np.float64),
            "ML_um": ml_um.astype(np.float64),
            "theta": theta.astype(np.float64),
            "s": s.astype(np.float64),
        }
    )
    panel_dir.mkdir(parents=True, exist_ok=True)
    cells.to_csv(panel_dir / "cells.tsv", sep="\t", index=False)

    counts_df = pd.DataFrame(
        {
            "cell_id": cells["cell_id"].to_numpy(),
            "G1": counts[:, 0],
            "G2": counts[:, 1],
            "G3": counts[:, 2],
            "G4": counts[:, 3],
            "G5": counts[:, 4],
        }
    )
    counts_df.to_csv(panel_dir / "counts.tsv", sep="\t", index=False)


class _StubPredictor:
    def coefficient_names(self, _fit: object) -> list[str]:
        return ["(Intercept)", "animalJaxA4", "batchB"]

    def mean_log_sf(self, _fit: object) -> float:
        return 0.0

    def batch_levels(self, _fit: object) -> list[str]:
        return []

    def animal_levels(self, _fit: object) -> list[str]:
        return []


def test_extract_ref_levels_falls_back_to_cells_levels() -> None:
    cells = pd.DataFrame(
        {
            "animal": ["JaxA4", "JaxA5", "JaxA4", None],
            "batch": ["b1", "b2", "b1", "b3"],
        }
    )
    (
        include_pos,
        include_log_sf_c,
        include_batch,
        mean_log_sf,
        batch_ref,
        batch_levels,
        include_animal,
        animal_ref,
        animal_levels,
    ) = extract_ref_levels(_StubPredictor(), fit=object(), cells=cells)
    assert include_pos is False
    assert include_log_sf_c is False
    assert mean_log_sf is None
    assert include_batch is True
    assert batch_ref == "b1"
    assert batch_levels == ["b1", "b2", "b3"]
    assert include_animal is True
    assert animal_ref == "JaxA4"
    assert animal_levels == ["JaxA4", "JaxA5"]


def test_cluster_apml_patterns_smoke(tmp_path: Path) -> None:
    panel_dir = tmp_path / "panel"
    _make_small_panel(panel_dir)

    fit_out = panel_dir / "fit_results.tsv"
    subprocess.run(
        [
            "Rscript",
            "scripts/gam/fit_inm_panel.R",
            str(panel_dir),
            str(fit_out),
            "--no-pos",
            "--threads",
            "1",
            "--k-uv",
            "8",
            "--no-diagnostics",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    out_dir = panel_dir / "cluster_out"
    subprocess.run(
        [
            sys.executable,
            "scripts/gam/cluster_apml_patterns.py",
            str(panel_dir),
            "--fit-results",
            str(fit_out),
            "--out-dir",
            str(out_dir),
            "--top-n",
            "3",
            "--rank-by",
            "p_apml_r_um",
            "--pattern",
            "apml_at_r",
            "--r-quantiles",
            "0.5",
            "--apml-n",
            "20",
            "--grid-mode",
            "data",
            "--pointwise-confidence",
            "z_surface",
            "--pc-k",
            "2",
            "--k",
            "2",
            "--linkage",
            "average",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    summary = pd.read_csv(out_dir / "summary.tsv", sep="\t")
    assert len(summary) == 3
    assert "cluster" in summary.columns
    assert np.all(np.isfinite(pd.to_numeric(summary["pc1"], errors="coerce").to_numpy(dtype=float)))
    assert np.all(np.isfinite(pd.to_numeric(summary["pc2"], errors="coerce").to_numpy(dtype=float)))

    assert (out_dir / "dendrogram.png").exists()
    assert (out_dir / "pc_scatter.png").exists()
    assert (out_dir / "surfaces.npz").exists()
    assert (out_dir / "amplitude.tsv").exists()
    assert (out_dir / "amplitude_rank.png").exists()

    cluster_pngs = sorted((out_dir / "cluster_means").glob("cluster_*_rQ50.png"))
    assert len(cluster_pngs) >= 1
