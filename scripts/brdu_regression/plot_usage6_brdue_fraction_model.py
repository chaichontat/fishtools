#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(str(path))
    df = pd.read_csv(path)
    df.columns = [str(c).lstrip("\ufeff").strip() for c in df.columns]
    return df


def _savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def _weighted_ci_normal(x: np.ndarray, w: np.ndarray) -> tuple[float, float, float]:
    x = np.asarray(x, dtype=float).reshape(-1)
    w = np.asarray(w, dtype=float).reshape(-1)
    ok = np.isfinite(x) & np.isfinite(w) & (w > 0)
    x = x[ok]
    w = w[ok]
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")

    w_sum = float(np.sum(w))
    mean = float(np.sum(w * x) / w_sum)
    var = float(np.sum(w * (x - mean) ** 2) / w_sum)
    w2 = float(np.sum(w * w))
    n_eff = (w_sum * w_sum) / w2 if w2 > 0 else float("nan")
    se = math.sqrt(var / n_eff) if np.isfinite(n_eff) and n_eff > 1 else float("nan")
    if not np.isfinite(se):
        return mean, float("nan"), float("nan")
    z = 1.96
    return mean, float(mean - z * se), float(mean + z * se)


def _plot_grid(
    by_animal: pd.DataFrame,
    *,
    by_unit: pd.DataFrame | None,
    value_col: str,
    ylabel: str,
    out_png: Path,
    title_prefix: str,
) -> None:
    needed = {"animal", "leiden", "usage_bin", value_col}
    missing = sorted(needed - set(by_animal.columns))
    if missing:
        raise ValueError(f"{out_png} missing columns in input: {missing}")

    df = by_animal.copy()
    if "fail_reason" in df.columns:
        df = df.loc[df["fail_reason"].isna() | (df["fail_reason"].astype(str).str.len() == 0)].copy()
    df = df.dropna(subset=[value_col])

    df["leiden"] = df["leiden"].astype(str)
    df["usage_bin"] = df["usage_bin"].astype(int)
    df["animal"] = df["animal"].astype(str)
    df[value_col] = df[value_col].astype(float)

    def _leiden_key(x: str) -> tuple[int, str]:
        sx = str(x)
        return (0, f"{int(sx):06d}") if sx.isdigit() else (1, sx)

    leidens = sorted(df["leiden"].unique().tolist(), key=_leiden_key)
    if not leidens:
        raise ValueError("No rows to plot after filtering.")

    ncols = 2
    nrows = int(np.ceil(len(leidens) / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10.5, 3.2 * nrows), sharex=True)
    axes = np.atleast_1d(axes).reshape(nrows, ncols)

    for idx, leiden in enumerate(leidens):
        ax = axes[idx // ncols, idx % ncols]
        sub = df.loc[df["leiden"] == leiden].copy()

        if by_unit is not None:
            du = by_unit.loc[by_unit["leiden"].astype(str) == str(leiden)].copy()
            if du.empty:
                raise ValueError(f"by_unit provided but has no rows for leiden={leiden!r}")

            du["usage_bin"] = du["usage_bin"].astype(int)
            du["unit_weight_b1"] = du["unit_weight_b1"].astype(float)
            du["animal"] = du["animal"].astype(str)
            du[value_col] = du[value_col].astype(float)

            animals = sorted(du["animal"].unique().tolist())
            cmap = plt.get_cmap("tab10")

            # Per-animal weighted error bars (unit = dataset×roi×ccf_adjusted; weights = BrdU+ cells per unit).
            for a_idx, animal in enumerate(animals):
                da = du.loc[du["animal"] == animal]
                xs: list[int] = []
                ys: list[float] = []
                ylo: list[float] = []
                yhi: list[float] = []
                for t in sorted(da["usage_bin"].unique().tolist()):
                    dt = da.loc[da["usage_bin"] == t]
                    m, lo, hi = _weighted_ci_normal(
                        dt[value_col].to_numpy(float),
                        dt["unit_weight_b1"].to_numpy(float),
                    )
                    xs.append(int(t))
                    ys.append(float(m))
                    ylo.append(float(lo))
                    yhi.append(float(hi))
                yerr = np.vstack([np.array(ys) - np.array(ylo), np.array(yhi) - np.array(ys)])
                yerr = np.nan_to_num(yerr, nan=0.0, posinf=0.0, neginf=0.0)
                color = cmap(a_idx % 10)
                ax.errorbar(
                    np.array(xs, dtype=int),
                    np.array(ys, dtype=float),
                    yerr=yerr,
                    color=color,
                    linewidth=1.0,
                    alpha=0.7,
                    capsize=2,
                )

            # Pooled weighted mean+CI across all units (across all animals).
            xs: list[int] = []
            ys: list[float] = []
            ylo: list[float] = []
            yhi: list[float] = []
            for t in sorted(du["usage_bin"].unique().tolist()):
                dt = du.loc[du["usage_bin"] == t]
                m, lo, hi = _weighted_ci_normal(dt[value_col].to_numpy(float), dt["unit_weight_b1"].to_numpy(float))
                xs.append(int(t))
                ys.append(float(m))
                ylo.append(float(lo))
                yhi.append(float(hi))
            yerr = np.vstack([np.array(ys) - np.array(ylo), np.array(yhi) - np.array(ys)])
            yerr = np.nan_to_num(yerr, nan=0.0, posinf=0.0, neginf=0.0)
            ax.errorbar(
                np.array(xs, dtype=int),
                np.array(ys, dtype=float),
                yerr=yerr,
                color="#2b8cbe",
                linewidth=2.5,
                capsize=3,
            )
        else:
            animals = sorted(sub["animal"].unique().tolist())
            for animal in animals:
                a = sub.loc[sub["animal"] == animal].sort_values("usage_bin")
                ax.plot(
                    a["usage_bin"].to_numpy(int),
                    a[value_col].to_numpy(float),
                    color="#888888",
                    linewidth=1.0,
                    alpha=0.6,
                )

            mean = sub.groupby("usage_bin", as_index=False)[value_col].mean().sort_values("usage_bin")
            ax.plot(
                mean["usage_bin"].to_numpy(int),
                mean[value_col].to_numpy(float),
                color="#2b8cbe",
                linewidth=2.5,
            )

        ax.set_title(f"{title_prefix} (leiden={leiden}; n_animals={len(animals)})")
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Usage_6 bin (global quantiles)")
        ax.grid(True, axis="y", alpha=0.2)

    for j in range(len(leidens), nrows * ncols):
        axes[j // ncols, j % ncols].axis("off")

    fig.suptitle(title_prefix, y=1.02)
    fig.tight_layout()
    _savefig(out_png)


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Plot results from usage6_brdue_fraction_model.py (per-animal curves + mean). "
            "Expects usage6_by_animal.csv in --indir."
        )
    )
    p.add_argument("--indir", type=Path, required=True, help="Directory containing usage6_by_animal.csv")
    args = p.parse_args()

    indir: Path = args.indir
    by_animal_csv = indir / "usage6_by_animal.csv"
    by_animal = _read_csv(by_animal_csv)
    by_unit_csv = indir / "usage6_by_unit.csv"
    by_unit = _read_csv(by_unit_csv) if by_unit_csv.exists() else None

    _plot_grid(
        by_animal=by_animal,
        by_unit=by_unit,
        value_col="f_hat",
        ylabel="f_hat = P(EdU+ | BrdU+)",
        out_png=indir / "usage6_f_hat.png",
        title_prefix="BrdU+ retention vs Usage_6",
    )
    _plot_grid(
        by_animal=by_animal,
        by_unit=by_unit,
        value_col="inv_f_hat",
        ylabel="1 / f_hat",
        out_png=indir / "usage6_inv_f_hat.png",
        title_prefix="BrdU+EdU+ / BrdU+ inverse fraction vs Usage_6",
    )

    if "f_hat" not in by_animal.columns:
        raise ValueError("usage6_by_animal.csv missing f_hat (needed for T_S/Δt plot).")
    if np.any(pd.to_numeric(by_animal["f_hat"], errors="coerce").to_numpy(float) >= 1.0):
        raise ValueError("Some f_hat values are >= 1.0; T_S/Δt = 1/(1-f_hat) would be undefined.")
    by_animal_ts = by_animal.copy()
    by_animal_ts["ts_over_dt"] = 1.0 / (1.0 - by_animal_ts["f_hat"].astype(float))
    if by_unit is not None and "ts_over_dt" not in by_unit.columns:
        by_unit = by_unit.copy()
        by_unit["ts_over_dt"] = 1.0 / (1.0 - by_unit["f_hat"].astype(float))
    _plot_grid(
        by_animal=by_animal_ts,
        by_unit=by_unit,
        value_col="ts_over_dt",
        ylabel="T_S / Δt ≈ 1 / (1 − f_hat)",
        out_png=indir / "usage6_Ts_over_dt.png",
        title_prefix="S-phase time (dimensionless) vs Usage_6",
    )

    print(f"Wrote: {indir / 'usage6_f_hat.png'}")
    print(f"Wrote: {indir / 'usage6_inv_f_hat.png'}")
    print(f"Wrote: {indir / 'usage6_Ts_over_dt.png'}")


if __name__ == "__main__":
    main()
