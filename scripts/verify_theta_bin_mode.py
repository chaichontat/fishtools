#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pathlib

import numpy as np
import pandas as pd


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if int(np.sum(m)) < 3:
        return float("nan")
    aa = a[m].astype(float, copy=False)
    bb = b[m].astype(float, copy=False)
    if float(np.std(aa)) == 0.0 or float(np.std(bb)) == 0.0:
        return float("nan")
    return float(np.corrcoef(aa, bb)[0, 1])


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare quantile vs fixed-angle theta bin modes on exec summary outputs.")
    ap.add_argument("--quantile-csv", type=pathlib.Path, required=True)
    ap.add_argument("--angle-csv", type=pathlib.Path, required=True)
    ap.add_argument("--top-n", type=int, default=50)
    ap.add_argument("--outdir", type=pathlib.Path, default=pathlib.Path("scripts/_out/theta_bin_mode_verify"))
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    q = pd.read_csv(args.quantile_csv)
    a = pd.read_csv(args.angle_csv)
    for need in ["gene", "delta_int_E", "rd_int"]:
        if need not in q.columns:
            raise ValueError(f"quantile-csv missing column: {need}")
        if need not in a.columns:
            raise ValueError(f"angle-csv missing column: {need}")

    merged = q.merge(a, on="gene", suffixes=("_quantile", "_angle"), how="inner", validate="one_to_one")
    if merged.empty:
        raise ValueError("No overlapping genes between the two CSVs.")

    out_rows: list[dict[str, object]] = []
    for col in ["delta_int_E", "rd_int", "delta_BrdUplus_Edu", "delta_BrdUminus_Edu"]:
        cq = f"{col}_quantile"
        ca = f"{col}_angle"
        if cq not in merged.columns or ca not in merged.columns:
            continue
        x = merged[cq].to_numpy(float)
        y = merged[ca].to_numpy(float)
        out_rows.append(
            {
                "metric": col,
                "corr": _safe_corr(x, y),
                "sign_match_frac": float(np.mean(np.sign(x) == np.sign(y))),
                "n": int(np.sum(np.isfinite(x) & np.isfinite(y))),
            }
        )

    summary = pd.DataFrame(out_rows)
    summary.to_csv(args.outdir / "theta_bin_mode_metric_agreement.csv", index=False)

    # Top-N overlap by |delta_int_E| and by |rd_int|.
    def _top_set(df: pd.DataFrame, col: str) -> set[str]:
        sub = df.loc[:, ["gene", col]].dropna()
        sub = sub.iloc[np.argsort(np.abs(sub[col].to_numpy(float)))[::-1]]
        return set(sub["gene"].head(int(args.top_n)).astype(str).tolist())

    top_q_int = _top_set(q, "delta_int_E")
    top_a_int = _top_set(a, "delta_int_E")
    top_q_rd = _top_set(q, "rd_int")
    top_a_rd = _top_set(a, "rd_int")

    md = []
    md.append("# Theta Bin Mode Verification\n")
    md.append("Compared `quantile` vs `angle` theta-bin modes using exec summaries:\n")
    md.append(f"- quantile: `{args.quantile_csv}`\n")
    md.append(f"- angle: `{args.angle_csv}`\n")
    md.append(f"\nTop-N for overlap: N={int(args.top_n)}.\n\n")

    md.append("## Metric Agreement\n")
    md.append("Wrote `theta_bin_mode_metric_agreement.csv` with correlation and sign-match fractions.\n\n")

    md.append("## Top-N Overlap\n")
    md.append(f"- Top-{int(args.top_n)} by `|delta_int_E|` overlap: {len(top_q_int & top_a_int)} / {int(args.top_n)}\n")
    md.append(f"- Top-{int(args.top_n)} by `|rd_int|` overlap: {len(top_q_rd & top_a_rd)} / {int(args.top_n)}\n")
    md.append("\n")

    # Include the disagreement sets for debugging.
    only_q = sorted(top_q_int - top_a_int)[:25]
    only_a = sorted(top_a_int - top_q_int)[:25]
    md.append("## Example Disagreements (Top-|delta_int_E|)\n")
    md.append(f"- In quantile-only (first 25): {', '.join(only_q) if only_q else '(none)'}\n")
    md.append(f"- In angle-only (first 25): {', '.join(only_a) if only_a else '(none)'}\n")
    md.append("\n")

    (args.outdir / "theta_bin_mode_verify.md").write_text("".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()
