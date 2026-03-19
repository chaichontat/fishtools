from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import math
from multiprocessing import get_context
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ccf.refextract.plot_ap_ml_heatmap import compute_ap_ml_support_mask_native_grid
from fishtools.ccf.transforms import build_apml_native_surface_projection_context
from fishtools.gam.io_helpers import build_fit_map
from fishtools.gam.io_helpers import safe_gene_name
from fishtools.gam.mgcv_predict import RPredictor
from fishtools.gam.mgcv_predict import is_smooth_term
from fishtools.gam.mgcv_predict import make_newdata_base
from fishtools.gam.mgcv_predict import predict_response_shrunk
from fishtools.gam.mgcv_predict import shrink_hard
from fishtools.gam.mgcv_predict import shrink_soft
from fishtools.gam.mgcv_predict import select_ti_apml_r
from fishtools.gam.native_surface_plotting import native_proj_png_name
from fishtools.gam.native_surface_plotting import plot_coronal_surface_projection
from fishtools.gam.surface_predict import predict_term_effect_shrunk


PVAL_COLUMNS = [
    "p_spatial",
    "p_cycle",
    "p_interaction",
    "p_apml",
    "p_apml_r_um",
    "p_brdu_pos",
    "p_edu_pos",
    "p_brdu_edu",
]
SPATIAL_PVAL_COLUMNS = {"p_spatial", "p_apml", "p_apml_r_um"}

def read_fit_results(panel_dir: Path) -> pd.DataFrame:
    candidates = [
        panel_dir / "fit_results_allgenes.tsv",
        panel_dir / "fit_results.tsv",
        panel_dir / "fit_results_first50.tsv",
    ]
    for path in candidates:
        if path.exists():
            return pd.read_csv(path, sep="\t")
    raise FileNotFoundError("Could not find fit_results_allgenes.tsv, fit_results.tsv, or fit_results_first50.tsv")


def resolve_fit_results_path(panel_dir: Path) -> Path:
    candidates = [
        panel_dir / "fit_results_allgenes.tsv",
        panel_dir / "fit_results.tsv",
        panel_dir / "fit_results_first50.tsv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("Could not find fit_results_allgenes.tsv, fit_results.tsv, or fit_results_first50.tsv")


def pick_fits_dir(panel_dir: Path) -> Path:
    preferred = panel_dir / "fits_rds"
    if preferred.is_dir():
        return preferred
    candidates = sorted(panel_dir.glob("fits_rds__*"))
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError("Could not find fits_rds directory")

def pick_diagnostics_dir(panel_dir: Path, fit_results_path: Path) -> Path:
    default_fit_results = panel_dir / "fit_results.tsv"
    if fit_results_path.resolve() == default_fit_results.resolve():
        return panel_dir / "diagnostics_gam"
    stem = fit_results_path.name.removesuffix(".tsv")
    return panel_dir / f"diagnostics_gam__{stem}"


def copy_diagnostics_for_fit(*, diagnostics_dir: Path, fit_path: Path, dest_gene_dir: Path) -> None:
    src = diagnostics_dir / fit_path.name.removesuffix(".gam.rds")
    if not src.is_dir():
        return
    dest = dest_gene_dir / "diagnostics"
    if dest.exists():
        return
    shutil.copytree(src, dest)

def _parse_only(only: list[str] | None) -> set[str] | None:
    if not only:
        return None
    out = {str(x).strip() for x in only if str(x).strip()}
    return out or None



def plot_line(x: np.ndarray, y: np.ndarray, *, out_png: Path, title: str, xlab: str) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5.5), dpi=140)
    plt.plot(x, y, lw=2.0)
    plt.xlabel(xlab)
    plt.ylabel("fitted mean (response scale)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def plot_heatmap(
    matrix: np.ndarray,
    *,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    out_png: Path,
    title: str,
    xlab: str,
    ylab: str,
    support_mask: np.ndarray | None = None,
    y_origin_top: bool = False,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    x_grid = np.asarray(x_grid, dtype=float).reshape(-1)
    y_grid = np.asarray(y_grid, dtype=float).reshape(-1)
    extent = [float(x_grid.min()), float(x_grid.max()), float(y_grid.min()), float(y_grid.max())]
    matrix_plot = np.asarray(matrix, dtype=float)
    expected = (int(y_grid.size), int(x_grid.size))
    if matrix_plot.shape != expected:
        raise ValueError(f"Heatmap matrix shape {matrix_plot.shape} does not match (len(y_grid), len(x_grid))={expected}")
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="lightgray")
    if support_mask is not None:
        matrix_plot = apply_support_mask_for_imshow(matrix_plot, support_mask)
    plt.figure(figsize=(8.5, 6.5), dpi=140)
    kwargs = {}
    if vmin is not None:
        kwargs["vmin"] = float(vmin)
    if vmax is not None:
        kwargs["vmax"] = float(vmax)
    image = plt.imshow(matrix_plot, origin="lower", aspect="auto", extent=extent, cmap=cmap, **kwargs)
    if support_mask is not None:
        plt.contour(
            support_mask.astype(float),
            levels=[0.5],
            origin="lower",
            extent=extent,
            colors="white",
            linewidths=1.2,
        )
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(title)
    plt.colorbar(image, label="fitted mean (response)")
    if y_origin_top:
        plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def plot_heatmap_signed(
    matrix: np.ndarray,
    *,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    out_png: Path,
    title: str,
    xlab: str,
    ylab: str,
    support_mask: np.ndarray | None = None,
    y_origin_top: bool = False,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    matrix = np.asarray(matrix, dtype=float)
    if vmin is None or vmax is None:
        vmax0 = float(np.nanmax(np.abs(matrix)))
        if not np.isfinite(vmax0) or vmax0 <= 0:
            vmax0 = 1.0
        vmin = -vmax0
        vmax = vmax0
    out_png.parent.mkdir(parents=True, exist_ok=True)
    x_grid = np.asarray(x_grid, dtype=float).reshape(-1)
    y_grid = np.asarray(y_grid, dtype=float).reshape(-1)
    extent = [float(x_grid.min()), float(x_grid.max()), float(y_grid.min()), float(y_grid.max())]
    matrix_plot = matrix
    expected = (int(y_grid.size), int(x_grid.size))
    if matrix_plot.shape != expected:
        raise ValueError(f"Heatmap matrix shape {matrix_plot.shape} does not match (len(y_grid), len(x_grid))={expected}")
    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad(color="lightgray")
    if support_mask is not None:
        matrix_plot = apply_support_mask_for_imshow(matrix_plot, support_mask)
    plt.figure(figsize=(8.5, 6.5), dpi=140)
    image = plt.imshow(matrix_plot, origin="lower", aspect="auto", extent=extent, cmap=cmap, vmin=float(vmin), vmax=float(vmax))
    if support_mask is not None:
        plt.contour(
            support_mask.astype(float),
            levels=[0.5],
            origin="lower",
            extent=extent,
            colors="black",
            linewidths=1.2,
        )
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(title)
    plt.colorbar(image, label="effect (link scale)")
    if y_origin_top:
        plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def apply_support_mask_for_imshow(matrix: np.ndarray, support_mask: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    support_mask = np.asarray(support_mask, dtype=bool)
    if support_mask.shape != matrix.shape:
        raise ValueError(f"support_mask shape {support_mask.shape} does not match matrix shape {matrix.shape}")
    out = matrix.copy()
    out[~support_mask] = np.nan
    return out


def compute_apml_percentile_bounds(
    *,
    ap_um: np.ndarray,
    ml_um: np.ndarray,
    percentile_range: tuple[float, float],
) -> tuple[float, float, float, float]:
    lo, hi = (float(percentile_range[0]), float(percentile_range[1]))
    if not np.isfinite(lo) or not np.isfinite(hi) or not (0.0 <= lo < hi <= 100.0):
        raise ValueError(f"Invalid AP/ML percentile range: {percentile_range!r}")
    ap = np.asarray(ap_um, dtype=np.float64).reshape(-1)
    ml = np.asarray(ml_um, dtype=np.float64).reshape(-1)
    keep = np.isfinite(ap) & np.isfinite(ml)
    if not np.any(keep):
        raise ValueError("Cannot compute AP/ML percentile bounds without finite AP_um and ML_um values.")
    ap_lo, ap_hi = np.nanpercentile(ap[keep], [lo, hi])
    ml_lo, ml_hi = np.nanpercentile(ml[keep], [lo, hi])
    return float(ap_lo), float(ap_hi), float(ml_lo), float(ml_hi)


def build_native_tri_alpha(
    *,
    ordered_geom: dict[str, np.ndarray],
    ap_um_flat: np.ndarray,
    ml_um_flat: np.ndarray,
    percentile_bounds: tuple[float, float, float, float] | None,
) -> np.ndarray | None:
    if percentile_bounds is None:
        return None
    ap_lo, ap_hi, ml_lo, ml_hi = percentile_bounds
    tris = np.asarray(ordered_geom["tris"], dtype=np.int32)
    cent_ap = np.mean(np.asarray(ap_um_flat, dtype=np.float64)[tris], axis=1)
    cent_ml = np.mean(np.asarray(ml_um_flat, dtype=np.float64)[tris], axis=1)
    inside = (cent_ap >= ap_lo) & (cent_ap <= ap_hi) & (cent_ml >= ml_lo) & (cent_ml <= ml_hi)
    return inside.astype(np.float64, copy=False)


def _predict_link_shrunk_excluding_terms(
    *,
    predictor: RPredictor,
    fit,
    newdata: pd.DataFrame,
    shrink: str,
    hard_z: float,
    exclude_terms: tuple[str, ...],
) -> np.ndarray:
    eta = predictor.predict_link(fit, newdata).astype(np.float64, copy=False)
    terms, se_terms, term_names = predictor.predict_terms_se(fit, newdata)
    exclude_norm = tuple(term.replace(" ", "") for term in exclude_terms)
    exclude_idx = [
        i
        for i, name in enumerate(term_names)
        if any(str(name).replace(" ", "").startswith(prefix) for prefix in exclude_norm)
    ]
    if exclude_idx:
        eta = eta - np.sum(terms[:, exclude_idx], axis=1)
    smooth_idx = [i for i, name in enumerate(term_names) if is_smooth_term(name) and i not in exclude_idx]
    if not smooth_idx:
        return eta
    smooth = terms[:, smooth_idx]
    smooth_se = se_terms[:, smooth_idx]
    if shrink == "none":
        smooth_shrunk = smooth
    elif shrink == "hard":
        smooth_shrunk = shrink_hard(smooth, smooth_se, z=float(hard_z))
    elif shrink == "soft":
        smooth_shrunk = shrink_soft(smooth, smooth_se)
    else:
        raise ValueError(f"Unknown shrink mode: {shrink}")
    return eta - np.sum(smooth, axis=1) + np.sum(smooth_shrunk, axis=1)


def mean_response_from_link(etas: list[np.ndarray]) -> np.ndarray:
    if not etas:
        raise ValueError("Expected at least one link-scale prediction to average.")
    eta_mat = np.stack([np.asarray(eta, dtype=np.float64) for eta in etas], axis=0)
    eta_max = np.max(eta_mat, axis=0)
    centered = np.exp(eta_mat - eta_max[None, :])
    log_mean = eta_max + np.log(np.mean(centered, axis=0))
    return np.exp(log_mean)


def predict_response_shrunk_marginalizing_animal(
    *,
    predictor: RPredictor,
    fit,
    newdata: pd.DataFrame,
    animal_levels: list[str],
    ab_levels: list[str],
    batch_ref: str | None,
    shrink: str,
    hard_z: float,
) -> np.ndarray:
    if not animal_levels:
        raise ValueError("Requested --marginalize-animal but fit has no animal levels.")
    eta_by_animal: list[np.ndarray] = []
    for animal_level in animal_levels:
        nd = newdata.copy()
        nd["animal"] = pd.Categorical([animal_level] * len(nd), categories=animal_levels)
        if ab_levels:
            ab_ref = None
            if batch_ref is not None:
                candidate = f"{animal_level}.{batch_ref}"
                if candidate in ab_levels:
                    ab_ref = candidate
            if ab_ref is None:
                for level in ab_levels:
                    if str(level).startswith(f"{animal_level}."):
                        ab_ref = str(level)
                        break
            if ab_ref is None:
                ab_ref = str(ab_levels[0])
            nd["ab"] = pd.Categorical([ab_ref] * len(nd), categories=ab_levels)
        eta_by_animal.append(
            _predict_link_shrunk_excluding_terms(
                predictor=predictor,
                fit=fit,
                newdata=nd,
                shrink=shrink,
                hard_z=hard_z,
                exclude_terms=("s(ab)",),
            )
        )
    return mean_response_from_link(eta_by_animal)


def chunk_genes(sig_genes: list[str], workers: int) -> list[list[str]]:
    if workers <= 1 or len(sig_genes) <= 1:
        return [sig_genes]
    n_workers = min(len(sig_genes), int(workers))
    chunk_size = max(1, int(math.ceil(len(sig_genes) / float(n_workers))))
    return [sig_genes[i : i + chunk_size] for i in range(0, len(sig_genes), chunk_size)]


def run_plot_jobs(args: argparse.Namespace, sig_genes: list[str]) -> None:
    chunks = chunk_genes(sig_genes, int(args.workers))
    if len(chunks) <= 1:
        plot_selected_genes(args, sig_genes)
        return
    with ProcessPoolExecutor(max_workers=min(len(chunks), int(args.workers)), mp_context=get_context("spawn")) as executor:
        futures = [executor.submit(plot_selected_genes, args, chunk) for chunk in chunks]
        for future in as_completed(futures):
            future.result()


def plot_selected_genes(args: argparse.Namespace, sig_genes: list[str]) -> int:
    only = _parse_only(args.only)

    def want(png_name: str) -> bool:
        return only is None or png_name in only

    panel_dir = Path(args.panel_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cells = pd.read_csv(panel_dir / "cells.tsv", sep="\t")
    required_cells = {"r_um", "theta", "AP_um", "ML_um"}
    missing_cells = sorted(required_cells - set(cells.columns))
    if missing_cells:
        raise ValueError(f"cells.tsv is missing required columns: {missing_cells}")

    r_um_max = float(args.r_um_max)
    if not np.isfinite(r_um_max) or r_um_max <= 0:
        raise ValueError(f"--r-um-max must be finite and >0, got {args.r_um_max}")

    r_um = pd.to_numeric(cells["r_um"], errors="coerce").to_numpy(dtype=float)
    if not np.all(np.isfinite(r_um)):
        raise ValueError("cells.tsv column r_um must be finite (no NA/Inf).")
    if not args.no_r_um_filter:
        keep = r_um < r_um_max
        n_keep = int(np.sum(keep))
        if n_keep <= 0:
            raise ValueError(f"r_um < {r_um_max:g} filter removed all cells.")
        if n_keep < len(cells):
            print(f"Filtering cells: keeping {n_keep}/{len(cells)} with r_um < {r_um_max:g}")
            cells = cells.loc[keep].reset_index(drop=True)
    else:
        print("r_um filter disabled for plotting: using all cells (subject to finite AP/ML filtering)")

    ap_um = pd.to_numeric(cells["AP_um"], errors="coerce").to_numpy(dtype=float)
    ml_um = pd.to_numeric(cells["ML_um"], errors="coerce").to_numpy(dtype=float)
    keep_apml = np.isfinite(ap_um) & np.isfinite(ml_um)
    n_keep_apml = int(np.sum(keep_apml))
    if n_keep_apml <= 0:
        raise ValueError("No rows remain after requiring finite AP_um and ML_um.")
    if n_keep_apml < len(cells):
        print(f"Filtering cells: keeping {n_keep_apml}/{len(cells)} with finite AP_um/ML_um")
        cells = cells.loc[keep_apml].reset_index(drop=True)
        ap_um = ap_um[keep_apml]
        ml_um = ml_um[keep_apml]

    fit_results_path = Path(args.fit_results)
    results = pd.read_csv(fit_results_path, sep="\t")
    if "gene" not in results.columns:
        raise ValueError("fit results table must contain a 'gene' column")

    fits_dir = Path(args.fits_dir)
    fit_map = build_fit_map(fits_dir)
    predictor = RPredictor()
    diagnostics_dir = pick_diagnostics_dir(panel_dir, fit_results_path)

    diag_summary_df: pd.DataFrame | None = None
    if bool(args.gate_by_edf):
        diag_summary_path = Path(args.diagnostics_summary) if args.diagnostics_summary is not None else None
        if diag_summary_path is None:
            candidate = fit_results_path.parent / "diagnostics_summary.tsv"
            if candidate.exists():
                diag_summary_path = candidate
        if diag_summary_path is None or not diag_summary_path.exists():
            raise ValueError(
                "EDF gating is enabled but diagnostics summary was not found. "
                "Provide --diagnostics-summary PATH or write diagnostics_summary.tsv next to the fit-results TSV."
            )
        diag_summary_df = pd.read_csv(diag_summary_path, sep="\t")
        if "gene" not in diag_summary_df.columns:
            raise ValueError("diagnostics summary must contain a 'gene' column")
        diag_summary_df["gene"] = diag_summary_df["gene"].astype(str)
        diag_summary_df = diag_summary_df.set_index("gene", drop=False)

    r_grid = np.linspace(0.0, float(np.nanmax(cells["r_um"])), args.rtheta_nr)
    theta_grid = np.linspace(0.0, 2.0 * np.pi, args.rtheta_ntheta)
    refextract_res_ijk_um = (
        None
        if args.refextract_res_ijk_um is None
        else (
            float(args.refextract_res_ijk_um[0]),
            float(args.refextract_res_ijk_um[1]),
            float(args.refextract_res_ijk_um[2]),
        )
    )
    ap_grid, ml_grid, apml_support_mask = compute_ap_ml_support_mask_native_grid(
        outdir=Path(args.refextract_outdir).expanduser(),
        slice_i_min=int(args.refextract_slice_i_min),
        slice_i_max=int(args.refextract_slice_i_max),
        n_t=int(args.refextract_n_t),
        n_ml=int(args.apml_n),
        ref_t=float(args.refextract_ref_t),
        band_frac=float(args.refextract_band_frac),
        res_ijk_um=refextract_res_ijk_um,
        restrict_t_neomeso=bool(args.restrict_t_neomeso),
    )
    percentile_bounds = (
        None
        if args.apml_percentile_range is None
        else compute_apml_percentile_bounds(
            ap_um=ap_um,
            ml_um=ml_um,
            percentile_range=(float(args.apml_percentile_range[0]), float(args.apml_percentile_range[1])),
        )
    )
    native_ctx = None
    native_ordered_geom = None
    native_tri_alpha = None
    if bool(args.native_proj):
        native_ctx = build_apml_native_surface_projection_context(
            outdir=Path(args.refextract_outdir).expanduser(),
            slice_i_min=int(args.refextract_slice_i_min),
            slice_i_max=int(args.refextract_slice_i_max),
            n_t=int(args.refextract_n_t),
            ref_t=float(args.refextract_ref_t),
            band_frac=float(args.refextract_band_frac),
            res_ijk_um=refextract_res_ijk_um,
            elev_deg=float(args.elev_deg),
            azim_deg=float(args.azim_deg),
            roll_deg=float(args.roll_deg),
        )
        native_ordered_geom = (
            native_ctx["ordered_geom_neomeso"]
            if bool(args.restrict_t_neomeso) and not bool(args.gray_context)
            else native_ctx["ordered_geom_support"]
        )
        native_vertex_neomeso = np.asarray(native_ctx["neomeso_flat"], dtype=bool)
        native_tri_neomeso = np.asarray(native_ctx["tri_neomeso"], dtype=bool)
        native_tri_alpha = build_native_tri_alpha(
            ordered_geom=native_ordered_geom,
            ap_um_flat=np.asarray(native_ctx["ap_um_flat"], dtype=np.float64),
            ml_um_flat=np.asarray(native_ctx["ml_um_flat"], dtype=np.float64),
            percentile_bounds=percentile_bounds,
        )

    r0 = float(np.nanmedian(cells["r_um"]))
    apml_r_values = np.unique(
        np.nanquantile(cells["r_um"].to_numpy(dtype=float), np.asarray([0.1, 0.3, 0.5, 0.7, 0.9], dtype=np.float64))
    ).astype(np.float64, copy=False)
    theta0 = 0.0
    ap0 = float(np.nanmedian(ap_um))
    ml0 = float(np.nanmedian(ml_um))
    source_ref = str(cells["source"].dropna().astype(str).iloc[0]) if "source" in cells.columns and cells["source"].notna().any() else None

    for index, gene in enumerate(sig_genes, start=1):
        print(f"[{index}/{len(sig_genes)}] {gene}")
        safe = safe_gene_name(gene)
        if safe not in fit_map:
            print(f"  missing fit for {gene} (safe={safe}); skipping")
            continue

        fit_path = fit_map[safe]
        fit = predictor.read_fit(fit_path)
        coef_names = predictor.coefficient_names(fit)
        include_pos = ("brdu_pos" in coef_names) and ("edu_pos" in coef_names)
        include_log_sf_c = "log_sf_c" in coef_names
        mean_log_sf = predictor.mean_log_sf(fit) if include_log_sf_c else None
        batch_levels = predictor.batch_levels(fit)
        include_batch = bool(batch_levels)
        animal_levels = predictor.animal_levels(fit)
        include_animal = bool(animal_levels)
        ab_levels = predictor.ab_levels(fit)
        batch_ref = batch_levels[0] if batch_levels else source_ref
        animal_ref = animal_levels[0] if animal_levels else None
        ab_ref = ab_levels[0] if ab_levels else None
        if include_animal and animal_levels and batch_ref is not None:
            import re

            m = re.search(r"(JaxA\\d+)", str(batch_ref))
            if m is not None and m.group(1) in animal_levels:
                animal_ref = m.group(1)
        if ab_ref is None and animal_ref is not None and batch_ref is not None:
            ab_ref = f"{animal_ref}.{batch_ref}"
        gene_row = results.loc[results["gene"].astype(str) == gene].iloc[0]
        gene_dir = out_dir / safe
        gene_dir.mkdir(parents=True, exist_ok=True)

        def predict_response_values(newdata: pd.DataFrame) -> np.ndarray:
            if not bool(args.marginalize_animal):
                return predict_response_shrunk(
                    predictor,
                    fit,
                    newdata,
                    shrink=args.shrink,
                    hard_z=float(args.hard_z),
                )
            return predict_response_shrunk_marginalizing_animal(
                predictor=predictor,
                fit=fit,
                newdata=newdata,
                animal_levels=animal_levels,
                ab_levels=ab_levels,
                batch_ref=batch_ref,
                shrink=args.shrink,
                hard_z=float(args.hard_z),
            )

        if bool(args.copy_diagnostics):
            copy_diagnostics_for_fit(diagnostics_dir=diagnostics_dir, fit_path=fit_path, dest_gene_dir=gene_dir)

        p_spatial = float(gene_row["p_spatial"]) if "p_spatial" in gene_row and pd.notna(gene_row["p_spatial"]) else np.nan
        p_cycle = float(gene_row["p_cycle"]) if "p_cycle" in gene_row and pd.notna(gene_row["p_cycle"]) else np.nan
        p_interaction = (
            float(gene_row["p_interaction"]) if "p_interaction" in gene_row and pd.notna(gene_row["p_interaction"]) else np.nan
        )
        p_apml = float(gene_row["p_apml"]) if "p_apml" in gene_row and pd.notna(gene_row["p_apml"]) else np.nan
        p_apml_r_um = (
            float(gene_row["p_apml_r_um"]) if "p_apml_r_um" in gene_row and pd.notna(gene_row["p_apml_r_um"]) else np.nan
        )

        if want("fit_r.png") and np.isfinite(p_spatial) and p_spatial < float(args.alpha_spatial):
            nd_r = make_newdata_base(
                len(r_grid),
                r_um=r_grid,
                theta=theta0,
                ap_um=ap0,
                ml_um=ml0,
                mean_log_sf=mean_log_sf,
                include_pos=include_pos,
                include_batch=include_batch,
                batch_ref=batch_ref,
                batch_levels=batch_levels,
                include_animal=include_animal,
                animal_ref=animal_ref,
                animal_levels=animal_levels,
                include_log_sf_c=include_log_sf_c,
                ab_ref=ab_ref,
                ab_levels=ab_levels,
            )
            mu_r = predict_response_values(nd_r)
            plot_line(r_grid, mu_r, out_png=gene_dir / "fit_r.png", title=f"{gene}: fitted vs r_um", xlab="r_um")

        if want("fit_theta.png") and np.isfinite(p_cycle) and p_cycle < args.alpha:
            nd_theta = make_newdata_base(
                len(theta_grid),
                r_um=r0,
                theta=theta_grid,
                ap_um=ap0,
                ml_um=ml0,
                mean_log_sf=mean_log_sf,
                include_pos=include_pos,
                include_batch=include_batch,
                batch_ref=batch_ref,
                batch_levels=batch_levels,
                include_animal=include_animal,
                animal_ref=animal_ref,
                animal_levels=animal_levels,
                include_log_sf_c=include_log_sf_c,
                ab_ref=ab_ref,
                ab_levels=ab_levels,
            )
            mu_theta = predict_response_values(nd_theta)
            plot_line(
                theta_grid,
                mu_theta,
                out_png=gene_dir / "fit_theta.png",
                title=f"{gene}: fitted vs theta",
                xlab="theta (radians)",
            )

        if want("fit_r_theta.png") and np.isfinite(p_interaction) and p_interaction < args.alpha:
            if diag_summary_df is not None:
                if gene not in diag_summary_df.index:
                    raise ValueError(f"gene {gene!r} not found in diagnostics summary")
                edf_rt = pd.to_numeric(diag_summary_df.at[gene, "edf_ti_r_theta"], errors="coerce")
                if not np.isfinite(edf_rt) or float(edf_rt) < float(args.min_edf_plot):
                    print(f"  skip fit_r_theta: edf_ti_r_theta={edf_rt} < {float(args.min_edf_plot):g}")
                    continue
            rr, tt = np.meshgrid(r_grid, theta_grid, indexing="ij")
            nd_rt = make_newdata_base(
                rr.size,
                r_um=rr.ravel(),
                theta=tt.ravel(),
                ap_um=ap0,
                ml_um=ml0,
                mean_log_sf=mean_log_sf,
                include_pos=include_pos,
                include_batch=include_batch,
                batch_ref=batch_ref,
                batch_levels=batch_levels,
                include_animal=include_animal,
                animal_ref=animal_ref,
                animal_levels=animal_levels,
                include_log_sf_c=include_log_sf_c,
                ab_ref=ab_ref,
                ab_levels=ab_levels,
            )
            mu_rt = predict_response_values(nd_rt).reshape(len(r_grid), len(theta_grid))
            plot_heatmap(
                mu_rt,
                x_grid=theta_grid,
                y_grid=r_grid,
                out_png=gene_dir / "fit_r_theta.png",
                title=f"{gene}: fitted over r_um/theta",
                xlab="theta (radians)",
                ylab="r_um",
            )

        spatial_sig = (
            (np.isfinite(p_spatial) and p_spatial < float(args.alpha_spatial))
            or (np.isfinite(p_apml) and p_apml < float(args.alpha_spatial))
            or (np.isfinite(p_apml_r_um) and p_apml_r_um < float(args.alpha_spatial))
        )
        if spatial_sig:
            sf_ref = float(np.exp(mean_log_sf)) if mean_log_sf is not None else 1.0
            ap_mesh, ml_mesh = np.meshgrid(ap_grid, ml_grid, indexing="ij")

            def predict_apml_values(ap_vals: np.ndarray, ml_vals: np.ndarray) -> np.ndarray:
                mu_acc = np.zeros((len(ap_vals),), dtype=np.float64)
                for r_val in apml_r_values:
                    nd = make_newdata_base(
                        len(ap_vals),
                        r_um=float(r_val),
                        theta=theta0,
                        ap_um=ap_vals,
                        ml_um=ml_vals,
                        mean_log_sf=mean_log_sf,
                        include_pos=include_pos,
                        include_batch=include_batch,
                        batch_ref=batch_ref,
                        batch_levels=batch_levels,
                        include_animal=include_animal,
                        animal_ref=animal_ref,
                        animal_levels=animal_levels,
                        include_log_sf_c=include_log_sf_c,
                        ab_ref=ab_ref,
                        ab_levels=ab_levels,
                    )
                    if include_log_sf_c and mean_log_sf is not None:
                        nd["sf"] = sf_ref
                        nd["log_sf_c"] = 0.0
                    mu_acc += predict_response_values(nd)
                return mu_acc / float(len(apml_r_values))

            if want("fit_ap_ml.png"):
                mu_apml = predict_apml_values(ap_mesh.ravel(), ml_mesh.ravel()).reshape(len(ap_grid), len(ml_grid))
                if bool(args.native_proj):
                    assert native_ctx is not None
                    mu_native = np.full(np.asarray(native_ctx["support_flat"], dtype=bool).shape, np.nan, dtype=np.float64)
                    native_support = np.asarray(native_ctx["support_flat"], dtype=bool)
                    predict_mask = native_support
                    mu_native[predict_mask] = predict_apml_values(
                        np.asarray(native_ctx["ap_um_flat"], dtype=np.float64)[predict_mask],
                        np.asarray(native_ctx["ml_um_flat"], dtype=np.float64)[predict_mask],
                    )
                    plot_coronal_surface_projection(
                        mu_native,
                        x2d=np.asarray(native_ctx["x2d"], dtype=np.float64),
                        y2d=np.asarray(native_ctx["y2d"], dtype=np.float64),
                        z2d=np.asarray(native_ctx["z2d"], dtype=np.float64),
                        x3d=np.asarray(native_ctx["x3"], dtype=np.float64),
                        y3d=np.asarray(native_ctx["y3"], dtype=np.float64),
                        z3d=np.asarray(native_ctx["z3"], dtype=np.float64),
                        faces=np.asarray(native_ctx["faces"], dtype=np.int32),
                        tri_support=np.asarray(native_ctx["tri_support"], dtype=bool),
                        tri_neomeso=native_tri_neomeso,
                        restrict_t_neomeso=bool(args.restrict_t_neomeso),
                        gray_context=bool(args.gray_context),
                        latlon=bool(args.latlon),
                        graticule=str(args.graticule),
                        lat_stride=int(args.lat_stride),
                        lon_stride=int(args.lon_stride),
                        max_lat_lines=int(args.max_lat_lines),
                        max_lon_lines=int(args.max_lon_lines),
                        vertex_support=np.asarray(native_ctx["support_flat"], dtype=bool),
                        vertex_neomeso=native_vertex_neomeso,
                        vertex_ap_um=np.asarray(native_ctx["ap_um_flat"], dtype=np.float64),
                        vertex_ml_um=np.asarray(native_ctx["ml_um_flat"], dtype=np.float64),
                        n_rows=int(native_ctx["n_rows"]),
                        n_cols=int(native_ctx["n_cols"]),
                        shade=False,
                        shade_strength=0.75,
                        shade_elev_deg=float(args.elev_deg),
                        shade_azim_deg=float(args.azim_deg),
                        camera_elev_deg=float(args.elev_deg),
                        camera_azim_deg=float(args.azim_deg),
                        camera_roll_deg=float(args.roll_deg),
                        proj_type=str(args.proj_type),
                        focal_length=float(args.focal_length),
                        ordered_geometry=native_ordered_geom,
                        neomeso_start_fit=None
                        if native_ctx["neomeso_start_fit"] is None
                        else np.asarray(native_ctx["neomeso_start_fit"], dtype=np.float64),
                        neomeso_end_fit=None
                        if native_ctx["neomeso_end_fit"] is None
                        else np.asarray(native_ctx["neomeso_end_fit"], dtype=np.float64),
                        tri_alpha=native_tri_alpha,
                        out_png=gene_dir / native_proj_png_name("fit_ap_ml.png"),
                        title=f"{gene}: fitted over AP/ML (marginalized over r_um, theta={theta0:.3g})",
                        cmap=plt.get_cmap("viridis"),
                        cbar_label="fitted mean (response)",
                        cbar_ticks=None,
                        cbar_ticklabels=None,
                        vmin=None,
                        vmax=None,
                    )
                else:
                    plot_heatmap(
                        mu_apml,
                        x_grid=ml_grid,
                        y_grid=ap_grid,
                        out_png=gene_dir / "fit_ap_ml.png",
                        title=f"{gene}: fitted over AP/ML (marginalized over r_um, theta={theta0:.3g})",
                        xlab="ML_um",
                        ylab="AP_um",
                        support_mask=apml_support_mask,
                        y_origin_top=True,
                    )

            if bool(args.plot_apmlr_interaction):
                if diag_summary_df is not None:
                    if gene not in diag_summary_df.index:
                        raise ValueError(f"gene {gene!r} not found in diagnostics summary")
                    edf_apmlr = pd.to_numeric(diag_summary_df.at[gene, "edf_ti_apml_r"], errors="coerce")
                    if not np.isfinite(edf_apmlr) or float(edf_apmlr) < float(args.min_edf_plot):
                        print(f"  skip apmlr interaction: edf_ti_apml_r={edf_apmlr} < {float(args.min_edf_plot):g}")
                        continue
                nd_1 = make_newdata_base(
                    1,
                    r_um=r0,
                    theta=theta0,
                    ap_um=ap0,
                    ml_um=ml0,
                    mean_log_sf=mean_log_sf,
                    include_pos=include_pos,
                    include_batch=include_batch,
                    batch_ref=batch_ref,
                    batch_levels=batch_levels,
                    include_animal=include_animal,
                    animal_ref=animal_ref,
                    animal_levels=animal_levels,
                    include_log_sf_c=include_log_sf_c,
                    ab_ref=ab_ref,
                    ab_levels=ab_levels,
                )
                if include_log_sf_c and mean_log_sf is not None:
                    nd_1["sf"] = sf_ref
                    nd_1["log_sf_c"] = 0.0
                ti_idx = select_ti_apml_r(predictor.predict_terms_se(fit, nd_1)[2])
                if not ti_idx:
                    print("  no ti(AP_um,ML_um,r_um) term found; skipping interaction-term plots")
                else:
                    want_fit_lo = want("fit_ap_ml__r_lo.png")
                    want_fit_hi = want("fit_ap_ml__r_hi.png")
                    want_term_lo = want("term_ti_apml_r__r_lo.png")
                    want_term_hi = want("term_ti_apml_r__r_hi.png")
                    want_term_delta = want("term_ti_apml_r__delta_rhi_rlo.png")
                    if not (want_fit_lo or want_fit_hi or want_term_lo or want_term_hi or want_term_delta):
                        continue

                    r_lo = float(np.nanquantile(cells["r_um"].to_numpy(dtype=float), 0.1))
                    r_hi = float(np.nanquantile(cells["r_um"].to_numpy(dtype=float), 0.9))
                    nd_lo = make_newdata_base(
                        ap_mesh.size,
                        r_um=r_lo,
                        theta=theta0,
                        ap_um=ap_mesh.ravel(),
                        ml_um=ml_mesh.ravel(),
                        mean_log_sf=mean_log_sf,
                        include_pos=include_pos,
                        include_batch=include_batch,
                        batch_ref=batch_ref,
                        batch_levels=batch_levels,
                        include_animal=include_animal,
                        animal_ref=animal_ref,
                        animal_levels=animal_levels,
                        include_log_sf_c=include_log_sf_c,
                        ab_ref=ab_ref,
                        ab_levels=ab_levels,
                    )
                    nd_hi = make_newdata_base(
                        ap_mesh.size,
                        r_um=r_hi,
                        theta=theta0,
                        ap_um=ap_mesh.ravel(),
                        ml_um=ml_mesh.ravel(),
                        mean_log_sf=mean_log_sf,
                        include_pos=include_pos,
                        include_batch=include_batch,
                        batch_ref=batch_ref,
                        batch_levels=batch_levels,
                        include_animal=include_animal,
                        animal_ref=animal_ref,
                        animal_levels=animal_levels,
                        include_log_sf_c=include_log_sf_c,
                        ab_ref=ab_ref,
                        ab_levels=ab_levels,
                    )
                    if include_log_sf_c and mean_log_sf is not None:
                        for nd in (nd_lo, nd_hi):
                            nd["sf"] = sf_ref
                            nd["log_sf_c"] = 0.0

                    eff_lo = eff_hi = eff_delta = None
                    vmax = None
                    if want_term_lo or want_term_hi or want_term_delta:
                        eff_lo = predict_term_effect_shrunk(
                            predictor=predictor,
                            fit=fit,
                            newdata=nd_lo,
                            idx=ti_idx,
                            shrink=args.shrink,
                            hard_z=float(args.hard_z),
                        ).reshape(len(ap_grid), len(ml_grid))
                        eff_hi = predict_term_effect_shrunk(
                            predictor=predictor,
                            fit=fit,
                            newdata=nd_hi,
                            idx=ti_idx,
                            shrink=args.shrink,
                            hard_z=float(args.hard_z),
                        ).reshape(len(ap_grid), len(ml_grid))
                        eff_delta = eff_hi - eff_lo
                        vmax = float(np.nanmax(np.abs(np.stack([eff_lo, eff_hi, eff_delta], axis=0))))
                        if not np.isfinite(vmax) or vmax <= 0:
                            vmax = 1.0

                    if want_fit_lo or want_fit_hi:
                        mu_lo = predict_response_values(nd_lo).reshape(len(ap_grid), len(ml_grid))
                        mu_hi = predict_response_values(nd_hi).reshape(len(ap_grid), len(ml_grid))
                        mu_lo_m = apply_support_mask_for_imshow(mu_lo, apml_support_mask)
                        mu_hi_m = apply_support_mask_for_imshow(mu_hi, apml_support_mask)
                        mu_min = float(np.nanmin(np.stack([mu_lo_m, mu_hi_m], axis=0)))
                        mu_max = float(np.nanmax(np.stack([mu_lo_m, mu_hi_m], axis=0)))
                        if not np.isfinite(mu_min) or not np.isfinite(mu_max) or mu_min >= mu_max:
                            mu_min, mu_max = None, None

                        if want_fit_lo:
                            if bool(args.native_proj):
                                assert native_ctx is not None
                                native_support = np.asarray(native_ctx["support_flat"], dtype=bool)
                                predict_mask = native_support
                                mu_lo_native = np.full(native_support.shape, np.nan, dtype=np.float64)
                                nd_lo_native = make_newdata_base(
                                    int(np.sum(predict_mask)),
                                    r_um=r_lo,
                                    theta=theta0,
                                    ap_um=np.asarray(native_ctx["ap_um_flat"], dtype=np.float64)[predict_mask],
                                    ml_um=np.asarray(native_ctx["ml_um_flat"], dtype=np.float64)[predict_mask],
                                    mean_log_sf=mean_log_sf,
                                    include_pos=include_pos,
                                    include_batch=include_batch,
                                    batch_ref=batch_ref,
                                    batch_levels=batch_levels,
                                    include_animal=include_animal,
                                    animal_ref=animal_ref,
                                    animal_levels=animal_levels,
                                    include_log_sf_c=include_log_sf_c,
                                    ab_ref=ab_ref,
                                    ab_levels=ab_levels,
                                )
                                if include_log_sf_c and mean_log_sf is not None:
                                    nd_lo_native["sf"] = sf_ref
                                    nd_lo_native["log_sf_c"] = 0.0
                                mu_lo_native[predict_mask] = predict_response_values(nd_lo_native)
                                plot_coronal_surface_projection(
                                    mu_lo_native,
                                    x2d=np.asarray(native_ctx["x2d"], dtype=np.float64),
                                    y2d=np.asarray(native_ctx["y2d"], dtype=np.float64),
                                    z2d=np.asarray(native_ctx["z2d"], dtype=np.float64),
                                    x3d=np.asarray(native_ctx["x3"], dtype=np.float64),
                                    y3d=np.asarray(native_ctx["y3"], dtype=np.float64),
                                    z3d=np.asarray(native_ctx["z3"], dtype=np.float64),
                                    faces=np.asarray(native_ctx["faces"], dtype=np.int32),
                                    tri_support=np.asarray(native_ctx["tri_support"], dtype=bool),
                                    tri_neomeso=native_tri_neomeso,
                                    restrict_t_neomeso=bool(args.restrict_t_neomeso),
                                    gray_context=bool(args.gray_context),
                                    latlon=bool(args.latlon),
                                    graticule=str(args.graticule),
                                    lat_stride=int(args.lat_stride),
                                    lon_stride=int(args.lon_stride),
                                    max_lat_lines=int(args.max_lat_lines),
                                    max_lon_lines=int(args.max_lon_lines),
                                    vertex_support=np.asarray(native_ctx["support_flat"], dtype=bool),
                                    vertex_neomeso=native_vertex_neomeso,
                                    vertex_ap_um=np.asarray(native_ctx["ap_um_flat"], dtype=np.float64),
                                    vertex_ml_um=np.asarray(native_ctx["ml_um_flat"], dtype=np.float64),
                                    n_rows=int(native_ctx["n_rows"]),
                                    n_cols=int(native_ctx["n_cols"]),
                                    shade=False,
                                    shade_strength=0.75,
                                    shade_elev_deg=float(args.elev_deg),
                                    shade_azim_deg=float(args.azim_deg),
                                    camera_elev_deg=float(args.elev_deg),
                                    camera_azim_deg=float(args.azim_deg),
                                    camera_roll_deg=float(args.roll_deg),
                                    proj_type=str(args.proj_type),
                                    focal_length=float(args.focal_length),
                                    ordered_geometry=native_ordered_geom,
                                    neomeso_start_fit=None
                                    if native_ctx["neomeso_start_fit"] is None
                                    else np.asarray(native_ctx["neomeso_start_fit"], dtype=np.float64),
                                    neomeso_end_fit=None
                                    if native_ctx["neomeso_end_fit"] is None
                                    else np.asarray(native_ctx["neomeso_end_fit"], dtype=np.float64),
                                    tri_alpha=native_tri_alpha,
                                    out_png=gene_dir / native_proj_png_name("fit_ap_ml__r_lo.png"),
                                    title=f"{gene}: fitted over AP/ML (r_um={r_lo:.3g}, theta={theta0:.3g})",
                                    cmap=plt.get_cmap("viridis"),
                                    cbar_label="fitted mean (response)",
                                    cbar_ticks=None,
                                    cbar_ticklabels=None,
                                    vmin=mu_min,
                                    vmax=mu_max,
                                )
                            else:
                                plot_heatmap(
                                    mu_lo,
                                    x_grid=ml_grid,
                                    y_grid=ap_grid,
                                    out_png=gene_dir / "fit_ap_ml__r_lo.png",
                                    title=f"{gene}: fitted over AP/ML (r_um={r_lo:.3g}, theta={theta0:.3g})",
                                    xlab="ML_um",
                                    ylab="AP_um",
                                    support_mask=apml_support_mask,
                                    y_origin_top=True,
                                    vmin=mu_min,
                                    vmax=mu_max,
                                )
                        if want_fit_hi:
                            if bool(args.native_proj):
                                assert native_ctx is not None
                                native_support = np.asarray(native_ctx["support_flat"], dtype=bool)
                                predict_mask = native_support
                                mu_hi_native = np.full(native_support.shape, np.nan, dtype=np.float64)
                                nd_hi_native = make_newdata_base(
                                    int(np.sum(predict_mask)),
                                    r_um=r_hi,
                                    theta=theta0,
                                    ap_um=np.asarray(native_ctx["ap_um_flat"], dtype=np.float64)[predict_mask],
                                    ml_um=np.asarray(native_ctx["ml_um_flat"], dtype=np.float64)[predict_mask],
                                    mean_log_sf=mean_log_sf,
                                    include_pos=include_pos,
                                    include_batch=include_batch,
                                    batch_ref=batch_ref,
                                    batch_levels=batch_levels,
                                    include_animal=include_animal,
                                    animal_ref=animal_ref,
                                    animal_levels=animal_levels,
                                    include_log_sf_c=include_log_sf_c,
                                    ab_ref=ab_ref,
                                    ab_levels=ab_levels,
                                )
                                if include_log_sf_c and mean_log_sf is not None:
                                    nd_hi_native["sf"] = sf_ref
                                    nd_hi_native["log_sf_c"] = 0.0
                                mu_hi_native[predict_mask] = predict_response_values(nd_hi_native)
                                plot_coronal_surface_projection(
                                    mu_hi_native,
                                    x2d=np.asarray(native_ctx["x2d"], dtype=np.float64),
                                    y2d=np.asarray(native_ctx["y2d"], dtype=np.float64),
                                    z2d=np.asarray(native_ctx["z2d"], dtype=np.float64),
                                    x3d=np.asarray(native_ctx["x3"], dtype=np.float64),
                                    y3d=np.asarray(native_ctx["y3"], dtype=np.float64),
                                    z3d=np.asarray(native_ctx["z3"], dtype=np.float64),
                                    faces=np.asarray(native_ctx["faces"], dtype=np.int32),
                                    tri_support=np.asarray(native_ctx["tri_support"], dtype=bool),
                                    tri_neomeso=native_tri_neomeso,
                                    restrict_t_neomeso=bool(args.restrict_t_neomeso),
                                    gray_context=bool(args.gray_context),
                                    latlon=bool(args.latlon),
                                    graticule=str(args.graticule),
                                    lat_stride=int(args.lat_stride),
                                    lon_stride=int(args.lon_stride),
                                    max_lat_lines=int(args.max_lat_lines),
                                    max_lon_lines=int(args.max_lon_lines),
                                    vertex_support=np.asarray(native_ctx["support_flat"], dtype=bool),
                                    vertex_neomeso=native_vertex_neomeso,
                                    vertex_ap_um=np.asarray(native_ctx["ap_um_flat"], dtype=np.float64),
                                    vertex_ml_um=np.asarray(native_ctx["ml_um_flat"], dtype=np.float64),
                                    n_rows=int(native_ctx["n_rows"]),
                                    n_cols=int(native_ctx["n_cols"]),
                                    shade=False,
                                    shade_strength=0.75,
                                    shade_elev_deg=float(args.elev_deg),
                                    shade_azim_deg=float(args.azim_deg),
                                    camera_elev_deg=float(args.elev_deg),
                                    camera_azim_deg=float(args.azim_deg),
                                    camera_roll_deg=float(args.roll_deg),
                                    proj_type=str(args.proj_type),
                                    focal_length=float(args.focal_length),
                                    ordered_geometry=native_ordered_geom,
                                    neomeso_start_fit=None
                                    if native_ctx["neomeso_start_fit"] is None
                                    else np.asarray(native_ctx["neomeso_start_fit"], dtype=np.float64),
                                    neomeso_end_fit=None
                                    if native_ctx["neomeso_end_fit"] is None
                                    else np.asarray(native_ctx["neomeso_end_fit"], dtype=np.float64),
                                    tri_alpha=native_tri_alpha,
                                    out_png=gene_dir / native_proj_png_name("fit_ap_ml__r_hi.png"),
                                    title=f"{gene}: fitted over AP/ML (r_um={r_hi:.3g}, theta={theta0:.3g})",
                                    cmap=plt.get_cmap("viridis"),
                                    cbar_label="fitted mean (response)",
                                    cbar_ticks=None,
                                    cbar_ticklabels=None,
                                    vmin=mu_min,
                                    vmax=mu_max,
                                )
                            else:
                                plot_heatmap(
                                    mu_hi,
                                    x_grid=ml_grid,
                                    y_grid=ap_grid,
                                    out_png=gene_dir / "fit_ap_ml__r_hi.png",
                                    title=f"{gene}: fitted over AP/ML (r_um={r_hi:.3g}, theta={theta0:.3g})",
                                    xlab="ML_um",
                                    ylab="AP_um",
                                    support_mask=apml_support_mask,
                                    y_origin_top=True,
                                    vmin=mu_min,
                                    vmax=mu_max,
                                )

                    if want_term_lo and eff_lo is not None and vmax is not None:
                        plot_heatmap_signed(
                            eff_lo,
                            x_grid=ml_grid,
                            y_grid=ap_grid,
                            out_png=gene_dir / "term_ti_apml_r__r_lo.png",
                            title=f"{gene}: ti(AP,ML,r) term (link; r_um={r_lo:.3g})",
                            xlab="ML_um",
                            ylab="AP_um",
                            support_mask=apml_support_mask,
                            y_origin_top=True,
                            vmin=-vmax,
                            vmax=vmax,
                        )
                    if want_term_hi and eff_hi is not None and vmax is not None:
                        plot_heatmap_signed(
                            eff_hi,
                            x_grid=ml_grid,
                            y_grid=ap_grid,
                            out_png=gene_dir / "term_ti_apml_r__r_hi.png",
                            title=f"{gene}: ti(AP,ML,r) term (link; r_um={r_hi:.3g})",
                            xlab="ML_um",
                            ylab="AP_um",
                            support_mask=apml_support_mask,
                            y_origin_top=True,
                            vmin=-vmax,
                            vmax=vmax,
                        )
                    if want_term_delta and eff_delta is not None and vmax is not None:
                        plot_heatmap_signed(
                            eff_delta,
                            x_grid=ml_grid,
                            y_grid=ap_grid,
                            out_png=gene_dir / "term_ti_apml_r__delta_rhi_rlo.png",
                            title=f"{gene}: ti(AP,ML,r) delta (r_hi - r_lo; link)",
                            xlab="ML_um",
                            ylab="AP_um",
                            support_mask=apml_support_mask,
                            y_origin_top=True,
                            vmin=-vmax,
                            vmax=vmax,
                        )

        if want("fit_ap_r.png") and np.isfinite(p_apml_r_um) and p_apml_r_um < float(args.alpha_spatial):
            ap_mesh_r, r_mesh_ap = np.meshgrid(ap_grid, r_grid, indexing="ij")
            nd_apr = make_newdata_base(
                ap_mesh_r.size,
                r_um=r_mesh_ap.ravel(),
                theta=theta0,
                ap_um=ap_mesh_r.ravel(),
                ml_um=ml0,
                mean_log_sf=mean_log_sf,
                include_pos=include_pos,
                include_batch=include_batch,
                batch_ref=batch_ref,
                batch_levels=batch_levels,
                include_animal=include_animal,
                animal_ref=animal_ref,
                animal_levels=animal_levels,
                include_log_sf_c=include_log_sf_c,
                ab_ref=ab_ref,
                ab_levels=ab_levels,
            )
            mu_apr = predict_response_values(nd_apr).reshape(len(ap_grid), len(r_grid))
            plot_heatmap(
                mu_apr,
                x_grid=r_grid,
                y_grid=ap_grid,
                out_png=gene_dir / "fit_ap_r.png",
                title=f"{gene}: fitted over AP/r (ML_um={ml0:.3g}, theta={theta0:.3g})",
                xlab="r_um",
                ylab="AP_um",
            )

    print(f"Wrote plots under {out_dir}")
    return len(sig_genes)


def main() -> int:
    parser = argparse.ArgumentParser(description="Python plotting for significant GAM genes (includes AP/ML fitted heatmaps).")
    parser.add_argument("panel_dir", type=Path)
    parser.add_argument("alpha", type=float, nargs="?", default=0.05)
    parser.add_argument("out_dir", type=Path, nargs="?")
    parser.add_argument("--fit-results", type=Path, default=None, help="Optional fit results TSV path (overrides panel_dir defaults).")
    parser.add_argument("--fits-dir", type=Path, default=None, help="Optional directory containing per-gene .gam.rds fits.")
    parser.add_argument(
        "--genes",
        type=str,
        default=None,
        help="Optional comma-separated gene list to plot (overrides significance-based selection).",
    )
    parser.add_argument(
        "--genes-file",
        type=Path,
        default=None,
        help="Optional newline-delimited gene list file to plot (overrides significance-based selection).",
    )
    parser.add_argument(
        "--gate-by-edf",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "If enabled, use per-gene EDF from diagnostics_summary.tsv to skip plotting interaction terms "
            "that are penalized away (edf ~ 0). (default: enabled)"
        ),
    )
    parser.add_argument(
        "--diagnostics-summary",
        type=Path,
        default=None,
        help=(
            "Optional diagnostics_summary.tsv path (generated by summarize_gam_diagnostics.py). "
            "If omitted and --gate-by-edf is enabled, looks for diagnostics_summary.tsv next to the fit-results TSV."
        ),
    )
    parser.add_argument(
        "--min-edf-plot",
        type=float,
        default=1e-3,
        help="Minimum EDF threshold for a term to be considered non-collapsed when --gate-by-edf is enabled.",
    )
    parser.add_argument(
        "--copy-diagnostics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Copy per-gene diagnostics into each gene plot folder (default: enabled).",
    )
    parser.add_argument(
        "--alpha-spatial",
        type=float,
        default=0.01,
        help="Significance threshold for p_spatial when selecting and plotting spatial effects.",
    )
    parser.add_argument("--apml-n", type=int, default=512, help="Number of ML bins for AP/ML plotting on refextract support grid.")
    parser.add_argument("--refextract-outdir", type=Path, default=Path("ccf/out/refextract/midsurface_neocortex_mesocortex_allocortex_3d"))
    parser.add_argument("--refextract-slice-i-min", type=int, default=161)
    parser.add_argument("--refextract-slice-i-max", type=int, default=305)
    parser.add_argument("--refextract-n-t", type=int, default=257)
    parser.add_argument("--refextract-ref-t", type=float, default=0.5)
    parser.add_argument("--refextract-band-frac", type=float, default=0.15)
    parser.add_argument("--r-um-max", type=float, default=300.0, help="Filter cells to r_um < r-um-max before plotting.")
    parser.add_argument("--no-r-um-filter", action="store_true", help="Disable r_um filtering for plotting.")
    parser.add_argument(
        "--refextract-res-ijk-um",
        type=float,
        nargs=3,
        default=(20.0, 20.0, 20.0),
        metavar=("RI", "RJ", "RK"),
    )
    parser.add_argument(
        "--restrict-t-neomeso",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Restrict AP/ML support mask to neocortex+mesocortex overlap t-ranges (default: enabled).",
    )
    parser.add_argument("--rtheta-nr", type=int, default=140)
    parser.add_argument("--rtheta-ntheta", type=int, default=220)
    parser.add_argument(
        "--plot-apmlr-interaction",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Plot the ti(AP_um,ML_um,r_um) interaction term at low/high r_um and its delta (default: disabled).",
    )
    parser.add_argument(
        "--shrink",
        choices=("soft", "hard", "none"),
        default="soft",
        help="Post-fit SE-aware shrinkage mode (applied to smooth terms on the link scale before exponentiating).",
    )
    parser.add_argument("--hard-z", type=float, default=1.96, help="Z-threshold for hard shrinkage (default: 1.96).")
    parser.add_argument(
        "--only",
        action="append",
        default=None,
        help=(
            "Only render specified output PNG basenames (e.g. fit_ap_ml.png). "
            "Can be repeated; when omitted, renders all applicable plots."
        ),
    )
    parser.add_argument(
        "--native-proj",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Render AP/ML plots on the native cortical surface instead of flat AP/ML heatmaps.",
    )
    parser.add_argument(
        "--marginalize-animal",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Average predictions across animal levels while excluding s(ab), matching simplex native plots.",
    )
    parser.add_argument(
        "--apml-percentile-range",
        type=float,
        nargs=2,
        default=(2.5, 97.5),
        metavar=("LO", "HI"),
        help="Restrict native AP/ML rendering to the independent [LO, HI] percentiles of AP_um and ML_um from cells.tsv.",
    )
    parser.add_argument("--gray-context", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--elev-deg", type=float, default=-10.0)
    parser.add_argument("--azim-deg", type=float, default=-110.0)
    parser.add_argument("--roll-deg", type=float, default=180.0)
    parser.add_argument("--proj-type", choices=("ortho", "persp"), default="ortho")
    parser.add_argument("--focal-length", type=float, default=0.5)
    parser.add_argument("--latlon", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--graticule", choices=("apml", "param", "ijk"), default="ijk")
    parser.add_argument("--lat-stride", type=int, default=10)
    parser.add_argument("--lon-stride", type=int, default=10)
    parser.add_argument("--max-lat-lines", type=int, default=8)
    parser.add_argument("--max-lon-lines", type=int, default=8)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    if not (0.0 < float(args.alpha) <= 1.0):
        raise ValueError(f"alpha must be in (0,1], got {args.alpha}")
    if not (0.0 < float(args.alpha_spatial) <= 1.0):
        raise ValueError(f"--alpha-spatial must be in (0,1], got {args.alpha_spatial}")

    panel_dir = args.panel_dir
    args.out_dir = args.out_dir if args.out_dir is not None else panel_dir / "plots_gam_significant_py"
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    if args.fit_results is None:
        fit_results_path = resolve_fit_results_path(panel_dir)
    else:
        fit_results_path = args.fit_results.expanduser()
    results = pd.read_csv(fit_results_path, sep="\t")
    if "gene" not in results.columns:
        raise ValueError("fit results table must contain a 'gene' column")

    pcols = [col for col in PVAL_COLUMNS if col in results.columns]
    if not pcols:
        raise ValueError("No expected p-value columns found in fit results table")

    forced_genes: list[str] | None = None
    if args.genes is not None and args.genes_file is not None:
        raise ValueError("Use only one of --genes or --genes-file")
    if args.genes is not None:
        forced_genes = [g.strip() for g in str(args.genes).split(",") if g.strip()]
        if not forced_genes:
            raise ValueError("--genes was provided but parsed to an empty list")
    if args.genes_file is not None:
        raw = args.genes_file.expanduser().read_text().splitlines()
        forced_genes = [line.strip() for line in raw if line.strip() and not line.strip().startswith("#")]
        if not forced_genes:
            raise ValueError("--genes-file was provided but contains no genes (after stripping blanks/comments)")

    if forced_genes is not None:
        known = set(results["gene"].astype(str).to_list())
        missing = [g for g in forced_genes if g not in known]
        if missing:
            raise ValueError(f"{len(missing)} gene(s) not found in fit results: {missing[:10]!r}")
        sig_genes = list(forced_genes)
        print(f"Forced gene list: {len(sig_genes)} gene(s)")
    else:
        sig_mask = np.zeros(len(results), dtype=bool)
        for col in pcols:
            values = pd.to_numeric(results[col], errors="coerce").to_numpy()
            threshold = float(args.alpha_spatial) if col in SPATIAL_PVAL_COLUMNS else float(args.alpha)
            sig_mask |= np.isfinite(values) & (values < threshold)
        sig_genes = results.loc[sig_mask, "gene"].astype(str).to_list()
    if sig_genes:
        original_index = {gene: idx for idx, gene in enumerate(sig_genes)}

        def sort_key(gene: str) -> tuple[bool, int]:
            safe = safe_gene_name(gene)
            exists = (Path(args.out_dir) / safe).is_dir()
            return (exists, original_index.get(gene, 10**9))

        sig_genes = sorted(sig_genes, key=sort_key)
    print(
        f"alpha={args.alpha:.3g}, alpha_spatial={args.alpha_spatial:.3g}: "
        f"significant genes={len(sig_genes)}/{len(results)}"
    )
    if int(args.workers) <= 0:
        raise ValueError(f"--workers must be >=1, got {args.workers}")
    args.fit_results = fit_results_path
    args.fits_dir = args.fits_dir.expanduser() if args.fits_dir is not None else pick_fits_dir(panel_dir)
    args.only = None if args.only is None else list(args.only)
    run_plot_jobs(args, sig_genes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
