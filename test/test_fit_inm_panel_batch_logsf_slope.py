from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


def test_fit_inm_panel_uses_animal_and_animal_batch_random_effects(tmp_path: Path) -> None:
    panel_dir = tmp_path / "panel"
    panel_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)
    n = 60
    dataset = np.array(
        ["20250101_JaxA1_Test1"] * (n // 2) + ["20250102_JaxA2_Test2"] * (n - n // 2),
        dtype=object,
    )
    cells = pd.DataFrame(
        {
            "cell_id": [f"c{i:03d}" for i in range(n)],
            "x": (0.3 * np.sin(rng.uniform(0.0, 2.0 * np.pi, size=n)) + rng.normal(0.0, 0.05, size=n)).astype(
                np.float64
            ),
            "r_um": rng.uniform(0.0, 299.0, size=n).astype(np.float64),
            "AP_um": rng.normal(0.0, 50.0, size=n).astype(np.float64),
            "ML_um": rng.normal(0.0, 50.0, size=n).astype(np.float64),
            "theta": rng.uniform(0.0, 2.0 * np.pi, size=n).astype(np.float64),
            "s": rng.lognormal(mean=6.0, sigma=0.3, size=n).astype(np.float64),
            "batch": dataset,
            "source": [f"{d}.1" for d in dataset],
        }
    )
    cells.to_csv(panel_dir / "cells.tsv", sep="\t", index=False)

    lam = np.exp(0.5 + 0.2 * np.sin(cells["theta"].to_numpy()))
    counts = rng.poisson(lam=lam).astype(int)
    counts_df = pd.DataFrame({"cell_id": cells["cell_id"].to_numpy(), "G1": counts})
    counts_df.to_csv(panel_dir / "counts.tsv", sep="\t", index=False)

    subprocess.run(
        ["Rscript", "scripts/gam/fit_inm_panel.R", str(panel_dir), str(panel_dir / "fit_results.tsv"), "--no-pos"],
        check=True,
        capture_output=True,
        text=True,
    )

    fit_rds = panel_dir / "fits_rds" / "0001_G1.gam.rds"
    assert fit_rds.exists()

    check_cmd = [
        "Rscript",
        "-e",
        (
            f"fit<-readRDS('{fit_rds}'); "
            "fml<-paste(deparse(fit$formula), collapse=' '); "
            "fmlc<-gsub('[[:space:]]+','',fml); "
            "fmlc<-gsub('\"',\"'\",fmlc,fixed=TRUE); "
            "stopifnot(grepl(\"s(animal,bs='re')\", fmlc, fixed=TRUE)); "
            "stopifnot(grepl(\"s(ab,bs='re')\", fmlc, fixed=TRUE));"
        ),
    ]
    subprocess.run(check_cmd, check=True, capture_output=True, text=True)
