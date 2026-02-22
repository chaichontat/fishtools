from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


def test_fit_inm_panel_uses_all_cells_by_default(tmp_path: Path) -> None:
    panel_dir = tmp_path / "panel"
    panel_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)
    n = 60
    cells = pd.DataFrame(
        {
            "cell_id": [f"c{i:03d}" for i in range(n)],
            "x": rng.normal(0.0, 0.1, size=n).astype(np.float64),
            "r_um": rng.uniform(0.0, 600.0, size=n).astype(np.float64),
            "AP_um": rng.normal(0.0, 50.0, size=n).astype(np.float64),
            "ML_um": rng.normal(0.0, 50.0, size=n).astype(np.float64),
            "theta": rng.uniform(0.0, 2.0 * np.pi, size=n).astype(np.float64),
            "s": rng.lognormal(mean=6.0, sigma=0.3, size=n).astype(np.float64),
            "batch": np.array(["20250101_JaxA1_Test1"] * (n // 2) + ["20250102_JaxA2_Test2"] * (n - n // 2), dtype=object),
            "source": np.array(["d1.1"] * n, dtype=object),
        }
    )
    cells.to_csv(panel_dir / "cells.tsv", sep="\t", index=False)

    counts = rng.poisson(lam=1.0, size=n).astype(int)
    counts_df = pd.DataFrame({"cell_id": cells["cell_id"].to_numpy(), "G1": counts})
    counts_df.to_csv(panel_dir / "counts.tsv", sep="\t", index=False)

    out_tsv = panel_dir / "fit_results.tsv"
    subprocess.run(
        ["Rscript", "scripts/gam/fit_inm_panel.R", str(panel_dir), str(out_tsv), "--no-pos"],
        check=True,
        capture_output=True,
        text=True,
    )

    df = pd.read_csv(out_tsv, sep="\t")
    assert "r_um_max" not in df.columns
    assert list(df["gene"]) == ["G1"]
