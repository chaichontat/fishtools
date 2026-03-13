#!/usr/bin/env python3
from __future__ import annotations

import argparse
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


def _weighted_quantile(x: np.ndarray, w: np.ndarray, q: float) -> float:
    x = np.asarray(x, dtype=float).reshape(-1)
    w = np.asarray(w, dtype=float).reshape(-1)
    if not (0.0 <= float(q) <= 1.0):
        raise ValueError("q must be in [0,1]")
    ok = np.isfinite(x) & np.isfinite(w) & (w > 0)
    x = x[ok]
    w = w[ok]
    if x.size == 0:
        return float("nan")
    order = np.argsort(x, kind="mergesort")
    x = x[order]
    w = w[order]
    cw = np.cumsum(w)
    cutoff = float(q) * float(cw[-1])
    j = int(np.searchsorted(cw, cutoff, side="left"))
    j = max(0, min(j, int(x.size) - 1))
    return float(x[j])


def _weighted_band(x: np.ndarray, w: np.ndarray) -> tuple[float, float, float]:
    x = np.asarray(x, dtype=float).reshape(-1)
    w = np.asarray(w, dtype=float).reshape(-1)
    ok = np.isfinite(x) & np.isfinite(w) & (w > 0)
    x = x[ok]
    w = w[ok]
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")

    med = _weighted_quantile(x, w, 0.5)
    lo = _weighted_quantile(x, w, 0.1)
    hi = _weighted_quantile(x, w, 0.9)
    return float(med), float(lo), float(hi)


def _plot_grid(
    by_animal: pd.DataFrame,
    *,
    by_unit: pd.DataFrame | None,
    value_col: str,
    weight_col: str | None,
    min_unit_trials: int,
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
            if weight_col is None:
                raise ValueError("by_unit provided but weight_col is None")
            if weight_col not in du.columns:
                raise ValueError(f"by_unit missing weight_col={weight_col!r} for plot {out_png.name}")
            b1_trial_col = "eff_n_b1_bin_total" if "eff_n_b1_bin_total" in du.columns else "n_b1_bin_total"
            if b1_trial_col not in du.columns or "n_all_bin_total" not in du.columns:
                raise ValueError(
                    "by_unit missing required columns for min-trials filtering: "
                    "n_b1_bin_total/eff_n_b1_bin_total and/or n_all_bin_total"
                )
            du[weight_col] = du[weight_col].astype(float)
            du["animal"] = du["animal"].astype(str)
            du[value_col] = du[value_col].astype(float)
            du[b1_trial_col] = pd.to_numeric(du[b1_trial_col], errors="raise").astype(float)
            du["n_all_bin_total"] = pd.to_numeric(du["n_all_bin_total"], errors="raise").astype(int)

            min_trials = int(min_unit_trials)
            if min_trials < 0:
                raise ValueError("min_unit_trials must be >= 0")
            if min_trials > 0:
                if value_col in {"f_hat", "inv_f_hat", "ts_over_dt", "ts_minutes"}:
                    du = du.loc[du[b1_trial_col] >= min_trials].copy()
                elif value_col in {"pE_hat"}:
                    du = du.loc[du["n_all_bin_total"] >= min_trials].copy()
                elif value_col in {"tc_over_dt", "tc_minutes"}:
                    du = du.loc[(du["n_all_bin_total"] >= min_trials) & (du[b1_trial_col] >= min_trials)].copy()
                else:
                    raise ValueError(f"Unexpected value_col={value_col!r} for min-trials filtering.")
                if du.empty:
                    raise ValueError(
                        f"All unit×bin rows dropped for leiden={leiden!r} after min-trials filtering "
                        f"(value_col={value_col!r}, min_trials={min_trials})."
                    )

            animals = sorted(du["animal"].unique().tolist())
            cmap = plt.get_cmap("tab10")

            # Per-animal weighted bands across units (unit = dataset×roi×ccf_adjusted).
            # NOTE: these are NOT confidence intervals; they summarize between-unit heterogeneity.
            for a_idx, animal in enumerate(animals):
                da = du.loc[du["animal"] == animal]
                xs: list[int] = []
                ys: list[float] = []
                ylo: list[float] = []
                yhi: list[float] = []
                for t in sorted(da["usage_bin"].unique().tolist()):
                    dt = da.loc[da["usage_bin"] == t]
                    m, lo, hi = _weighted_band(
                        dt[value_col].to_numpy(float),
                        dt[weight_col].to_numpy(float),
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
                    label=f"{animal} (10–90%)",
                )

            # Pooled weighted median and unit-heterogeneity band across all units (across all animals).
            xs: list[int] = []
            ys: list[float] = []
            ylo: list[float] = []
            yhi: list[float] = []
            for t in sorted(du["usage_bin"].unique().tolist()):
                dt = du.loc[du["usage_bin"] == t]
                m, lo, hi = _weighted_band(dt[value_col].to_numpy(float), dt[weight_col].to_numpy(float))
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
                label="pooled (10–90% units; not CI)",
            )
            ax.legend(loc="best", fontsize=8, frameon=False)
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
    p.add_argument(
        "--delta-t-minutes",
        type=float,
        default=None,
        help="If set, also write Ts/Tc plots in minutes (e.g. 90 for a 90 min pulse lag).",
    )
    p.add_argument(
        "--min-unit-trials",
        type=int,
        default=20,
        help=(
            "When usage6_by_unit.csv is present, treat unit×bin rows with very low/zero underlying trials as missing "
            "for plotting."
        ),
    )
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
        weight_col="unit_weight_b1" if by_unit is not None else None,
        min_unit_trials=int(args.min_unit_trials),
        ylabel="f_hat = P(EdU+ | BrdU+)",
        out_png=indir / "usage6_f_hat.png",
        title_prefix="BrdU+ retention vs Usage_6",
    )
    _plot_grid(
        by_animal=by_animal,
        by_unit=by_unit,
        value_col="inv_f_hat",
        weight_col="unit_weight_b1" if by_unit is not None else None,
        min_unit_trials=int(args.min_unit_trials),
        ylabel="1 / f_hat",
        out_png=indir / "usage6_inv_f_hat.png",
        title_prefix="BrdU+EdU+ / BrdU+ inverse fraction vs Usage_6",
    )

    if "ts_over_dt" not in by_animal.columns:
        if "f_hat" not in by_animal.columns:
            raise ValueError("usage6_by_animal.csv missing f_hat (needed for T_S/Δt plot).")
        if np.any(pd.to_numeric(by_animal["f_hat"], errors="coerce").to_numpy(float) >= 1.0):
            raise ValueError("Some f_hat values are >= 1.0; T_S/Δt = 1/(1-f_hat) would be undefined.")
        by_animal_ts = by_animal.copy()
        by_animal_ts["ts_over_dt"] = 1.0 / (1.0 - by_animal_ts["f_hat"].astype(float))
    else:
        by_animal_ts = by_animal
    if by_unit is not None and "ts_over_dt" not in by_unit.columns:
        by_unit = by_unit.copy()
        by_unit["ts_over_dt"] = 1.0 / (1.0 - by_unit["f_hat"].astype(float))
    _plot_grid(
        by_animal=by_animal_ts,
        by_unit=by_unit,
        value_col="ts_over_dt",
        weight_col="unit_weight_b1" if by_unit is not None else None,
        min_unit_trials=int(args.min_unit_trials),
        ylabel="T_S / Δt ≈ 1 / (1 − f_hat)",
        out_png=indir / "usage6_Ts_over_dt.png",
        title_prefix="S-phase time (dimensionless) vs Usage_6",
    )

    print(f"Wrote: {indir / 'usage6_f_hat.png'}")
    print(f"Wrote: {indir / 'usage6_inv_f_hat.png'}")
    print(f"Wrote: {indir / 'usage6_Ts_over_dt.png'}")

    if "pE_hat" in by_animal.columns:
        _plot_grid(
            by_animal=by_animal,
            by_unit=by_unit,
            value_col="pE_hat",
            weight_col="unit_weight_all" if by_unit is not None else None,
            min_unit_trials=int(args.min_unit_trials),
            ylabel="pE_hat = P(EdU+)",
            out_png=indir / "usage6_pE_hat.png",
            title_prefix="EdU labeling index vs Usage_6",
        )
        print(f"Wrote: {indir / 'usage6_pE_hat.png'}")

    if "tc_over_dt" in by_animal.columns:
        _plot_grid(
            by_animal=by_animal,
            by_unit=by_unit,
            value_col="tc_over_dt",
            weight_col="unit_weight_all" if by_unit is not None else None,
            min_unit_trials=int(args.min_unit_trials),
            ylabel="T_C proxy / Δt ≈ (T_S/Δt) / P(EdU+)  (GF-sensitive)",
            out_png=indir / "usage6_Tc_over_dt.png",
            title_prefix="Cell-cycle time proxy (dimensionless) vs Usage_6",
        )
        print(f"Wrote: {indir / 'usage6_Tc_over_dt.png'}")

    if args.delta_t_minutes is not None:
        dt_min = float(args.delta_t_minutes)
        if not (dt_min > 0):
            raise ValueError("--delta-t-minutes must be > 0")

        if "ts_over_dt" in by_animal_ts.columns:
            by_animal_ts_min = by_animal_ts.copy()
            by_animal_ts_min["ts_minutes"] = dt_min * by_animal_ts_min["ts_over_dt"].astype(float)
            by_unit_ts_min = None
            if by_unit is not None:
                by_unit_ts_min = by_unit.copy()
                by_unit_ts_min["ts_minutes"] = dt_min * by_unit_ts_min["ts_over_dt"].astype(float)
            _plot_grid(
                by_animal=by_animal_ts_min,
                by_unit=by_unit_ts_min,
                value_col="ts_minutes",
                weight_col="unit_weight_b1" if by_unit_ts_min is not None else None,
                min_unit_trials=int(args.min_unit_trials),
                ylabel="T_S (minutes) = (T_S/Δt) × Δt",
                out_png=indir / "usage6_Ts_minutes.png",
                title_prefix=f"S-phase time vs Usage_6 (Δt={dt_min:g} min)",
            )
            print(f"Wrote: {indir / 'usage6_Ts_minutes.png'}")

        if "tc_over_dt" in by_animal.columns:
            by_animal_tc_min = by_animal.copy()
            by_animal_tc_min["tc_minutes"] = dt_min * by_animal_tc_min["tc_over_dt"].astype(float)
            by_unit_tc_min = None
            if by_unit is not None:
                by_unit_tc_min = by_unit.copy()
                by_unit_tc_min["tc_minutes"] = dt_min * by_unit_tc_min["tc_over_dt"].astype(float)
            _plot_grid(
                by_animal=by_animal_tc_min,
                by_unit=by_unit_tc_min,
                value_col="tc_minutes",
                weight_col="unit_weight_all" if by_unit_tc_min is not None else None,
                min_unit_trials=int(args.min_unit_trials),
                ylabel="T_C proxy (minutes) = (T_C proxy/Δt) × Δt  (GF-sensitive)",
                out_png=indir / "usage6_Tc_minutes.png",
                title_prefix=f"Cell-cycle time proxy vs Usage_6 (Δt={dt_min:g} min)",
            )
            print(f"Wrote: {indir / 'usage6_Tc_minutes.png'}")


if __name__ == "__main__":
    main()
