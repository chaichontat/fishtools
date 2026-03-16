from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _spearman_corr(anchor: pd.DataFrame, other: pd.DataFrame) -> np.ndarray:
    genes = anchor.index.intersection(other.index)
    if genes.empty:
        raise ValueError("No shared genes between anchor and other GEP tables.")
    a = anchor.loc[genes]
    b = other.loc[genes]

    ar = a.rank(axis=0, method="average").to_numpy(float)
    br = b.rank(axis=0, method="average").to_numpy(float)
    ar = (ar - ar.mean(0, keepdims=True)) / ar.std(0, ddof=1, keepdims=True)
    br = (br - br.mean(0, keepdims=True)) / br.std(0, ddof=1, keepdims=True)
    return (ar.T @ br) / (ar.shape[0] - 1)


def _best2_per_row(corr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Returns best_idx, best_score, second_score
    order = np.argsort(-corr, axis=1)
    best = order[:, 0]
    best_s = corr[np.arange(corr.shape[0]), best]
    if corr.shape[1] >= 2:
        second = order[:, 1]
        second_s = corr[np.arange(corr.shape[0]), second]
    else:
        second_s = np.full_like(best_s, np.nan, dtype=float)
    return best, best_s, second_s


def _redundancy_stats(gep: pd.DataFrame, thr: float = 0.7) -> tuple[float, float]:
    c = gep.corr(method="spearman").to_numpy()
    np.fill_diagonal(c, np.nan)
    return float(np.nanmean(c > thr)), float(np.nanmax(c))


def _label_mix(outdir: Path, k: int, dt: str) -> dict[str, int]:
    ann_path = outdir / "annotations" / f"program_annotations.k{k}.dt{dt}.tsv"
    ann = pd.read_csv(ann_path, sep="\t")
    vc = ann["label"].value_counts()
    out: dict[str, int] = {}
    out["Cell cycle"] = int(vc.get("Cell cycle (S phase)", 0) + vc.get("Cell cycle (G2/M)", 0))
    out["RG/NSC"] = int(vc.get("Radial glia / neural stem-like", 0))
    out["IPC"] = int(vc.get("Neurogenic progenitors / IPC", 0))
    out["Astroglial"] = int(vc.get("Astroglial / gliogenic", 0))
    out["GABA"] = int(vc.get("GABAergic neurons", 0))
    out["Sex/X"] = int(vc.get("Sex / X-inactivation", 0))
    out["Mixed"] = int(sum(int(v) for i, v in vc.items() if str(i).startswith("Mixed:")))
    out["Unassigned"] = int(vc.get("Unassigned", 0))
    out["Other"] = int(len(ann) - sum(out.values()))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Compare candidate ks using stability + annotation summaries.")
    p.add_argument("--outdir", type=Path, default=Path("/fast2/cs_outputs/cnmf_all_progenitors"))
    p.add_argument("--dt", type=str, default="20.0")
    p.add_argument("--anchor-k", type=int, default=8)
    p.add_argument("--cands", type=int, nargs="+", default=[10, 12, 14])
    p.add_argument("--write-tsv", action="store_true")
    args = p.parse_args()

    outdir: Path = args.outdir
    dt: str = args.dt
    anchor_k: int = int(args.anchor_k)
    cands = [int(x) for x in args.cands]

    anchor = pd.read_csv(outdir / f"gep_scores.k{anchor_k}.dt{dt}.tsv", sep="\t", index_col=0)

    rows: list[dict[str, object]] = []
    for k in cands:
        gep = pd.read_csv(outdir / f"gep_scores.k{k}.dt{dt}.tsv", sep="\t", index_col=0)
        corr = _spearman_corr(anchor, gep)  # (anchor_programs x k_programs)

        # k -> anchor (does k split anchor programs?)
        best_a, best_s, second_s = _best2_per_row(corr.T)
        delta = best_s - second_s
        collisions = pd.Series(best_a).value_counts()

        # anchor -> k (does k cover anchor programs?)
        _, best_s2, second_s2 = _best2_per_row(corr)
        delta2 = best_s2 - second_s2

        frac_red, max_red = _redundancy_stats(gep, thr=0.7)
        mix = _label_mix(outdir, k, dt)

        rows.append(
            {
                "k": k,
                "n_programs": int(gep.shape[1]),
                "mean_best(k->anchor)": float(best_s.mean()),
                "median_delta(k->anchor)": float(np.nanmedian(delta)),
                "collisions(k->anchor)": int((collisions > 1).sum()),
                "max_collision": int(collisions.max()),
                "mean_best(anchor->k)": float(best_s2.mean()),
                "median_delta(anchor->k)": float(np.nanmedian(delta2)),
                "redundancy_frac_pairs_corr>0.7": frac_red,
                "redundancy_max_corr": max_red,
                **mix,
            }
        )

    df = pd.DataFrame(rows).sort_values("k")
    pd.set_option("display.max_columns", 200)
    print(df.round(3).to_string(index=False))

    if args.write_tsv:
        out_path = outdir / f"choose_k_compare.anchor_k{anchor_k}.dt{dt}.tsv"
        df.to_csv(out_path, sep="\t", index=False)
        print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()

