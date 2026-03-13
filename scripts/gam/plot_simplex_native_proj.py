from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ccf.refextract.plot_ap_ml_heatmap import compute_ap_ml_native_grid_details
from ccf.refextract.plot_ap_ml_heatmap import compute_ap_ml_support_mask_neomeso_on_native_grid
from fishtools.ccf.transforms import build_apml_native_surface_projection_context
from fishtools.gam.io_helpers import safe_gene_name
from fishtools.gam.mgcv_predict import RPredictor
from fishtools.gam.native_surface_plotting import _mask_r_ap_ml_pair_by_support
from fishtools.gam.native_surface_plotting import plot_coronal_surface_projection
from fishtools.gam.native_surface_plotting import write_apml_native_proj_montage
from fishtools.gam.surface_predict import GAMPredictorConfig
from fishtools.gam.surface_predict import make_newdata_for_fit


PNG_NAME = "fit_ap_ml_native_proj_simplex_u.png"
MONTAGE_NAME = "montage_fit_ap_ml_native_proj_simplex_u.png"
APMLR_PNG_NAME = "fit_ap_r_ml_r_simplex_u.png"


def _load_meta(meta_path: Path) -> dict[str, object]:
    raw = json.loads(meta_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{meta_path}: expected JSON object")
    for key in ["topic_ids", "fits_dir"]:
        if key not in raw:
            raise KeyError(f"{meta_path}: missing key {key!r}")
    return raw


def _load_topic_titles(topic_ids: list[int], label_tsv: Path | None) -> list[str]:
    if label_tsv is None:
        return [f"P{int(i)}" for i in topic_ids]
    ann = pd.read_csv(label_tsv, sep="\t")
    if "program" not in ann.columns:
        raise KeyError(f"{label_tsv}: expected 'program' column")
    label_col = "curated_label" if "curated_label" in ann.columns else ("label" if "label" in ann.columns else None)
    if label_col is None:
        raise KeyError(f"{label_tsv}: expected label column 'curated_label' or 'label'")
    prog = pd.to_numeric(ann["program"], errors="coerce")
    labels = ann[label_col].astype(str)
    mapping = {int(p): str(lbl) for p, lbl in zip(prog, labels, strict=False) if np.isfinite(p)}
    out: list[str] = []
    for pid in topic_ids:
        lbl = mapping.get(int(pid))
        out.append(f"P{int(pid)} {lbl}" if lbl else f"P{int(pid)}")
    return out


def _infer_animal_ref_from_batch_values(batch_values: pd.Series, animal_levels: list[str]) -> str | None:
    """
    Try to infer an animal reference level from batch strings like "...JaxA123...".
    Returns None if nothing matches.
    """
    if not animal_levels:
        return None
    pat = re.compile(r"(JaxA\\d+)")
    for raw in batch_values.dropna().astype(str).tolist():
        m = pat.search(raw)
        if m is None:
            continue
        candidate = m.group(1)
        if candidate in animal_levels:
            return candidate
    return None


def inv_alr(z: np.ndarray, *, ref_index: int) -> np.ndarray:
    """
    Invert ALR coordinates to simplex.

    z: (n, K-1) where columns correspond to topic_ids excluding ref topic.
    returns: (n, K) in [0,1] summing to 1.
    """
    z = np.asarray(z, dtype=np.float64)
    if z.ndim != 2:
        raise ValueError(f"z must be 2D (n,K-1), got {z.shape}")
    if not np.all(np.isfinite(z)):
        raise ValueError("z contains non-finite values")
    k_minus_1 = int(z.shape[1])
    k = k_minus_1 + 1
    ref = int(ref_index)
    if ref < 0 or ref >= k:
        raise ValueError(f"ref_index out of bounds for K={k}: {ref_index}")

    # ALR inverse is just a softmax over logits with ref logit fixed to 0.
    logits = np.zeros((z.shape[0], k), dtype=np.float64)
    nonref_cols = [i for i in range(k) if i != ref]
    if len(nonref_cols) != k_minus_1:
        raise RuntimeError("internal nonref_cols mismatch")
    for j, col in enumerate(nonref_cols):
        logits[:, col] = z[:, j]

    logits -= np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    denom = np.sum(exp_logits, axis=1)
    if not np.all(np.isfinite(denom)) or np.any(denom <= 0.0):
        raise ValueError("Invalid softmax denominator in inv_alr")
    u = exp_logits / denom[:, None]
    s = np.sum(u, axis=1)
    if not np.all(np.isfinite(s)) or np.any(np.abs(s - 1.0) > 1e-6):
        raise ValueError("inv_alr produced values that do not sum to 1")
    if np.any(u < -1e-9) or np.any(u > 1.0 + 1e-9):
        raise ValueError("inv_alr produced values outside [0,1]")
    return np.clip(u, 0.0, 1.0)


def ilr_basis_pivot(k: int) -> np.ndarray:
    """Pivot ILR basis (K x (K-1)), orthonormal in clr space."""
    kk = int(k)
    if kk < 2:
        raise ValueError("K must be >=2 for ILR")
    V = np.zeros((kk, kk - 1), dtype=np.float64)
    for j in range(1, kk):
        V[:j, j - 1] = np.sqrt(1.0 / (j * (j + 1)))
        V[j, j - 1] = -np.sqrt(j / (j + 1))
    return V


def inv_ilr(z: np.ndarray, *, V: np.ndarray) -> np.ndarray:
    """
    Invert ILR coordinates to simplex via clr.

    z: (n, K-1), V: (K, K-1)
    returns: (n, K) in [0,1] summing to 1.
    """
    z = np.asarray(z, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    if z.ndim != 2:
        raise ValueError(f"z must be 2D (n,K-1), got {z.shape}")
    if V.ndim != 2 or V.shape[0] != (z.shape[1] + 1) or V.shape[1] != z.shape[1]:
        raise ValueError(f"V shape {V.shape} incompatible with z shape {z.shape}")
    if not np.all(np.isfinite(z)):
        raise ValueError("z contains non-finite values")

    clr = z @ V.T
    clr -= np.max(clr, axis=1, keepdims=True)
    exp_logits = np.exp(clr)
    denom = np.sum(exp_logits, axis=1)
    if not np.all(np.isfinite(denom)) or np.any(denom <= 0.0):
        raise ValueError("Invalid softmax denominator in inv_ilr")
    u = exp_logits / denom[:, None]
    s = np.sum(u, axis=1)
    if not np.all(np.isfinite(s)) or np.any(np.abs(s - 1.0) > 1e-6):
        raise ValueError("inv_ilr produced values that do not sum to 1")
    if np.any(u < -1e-9) or np.any(u > 1.0 + 1e-9):
        raise ValueError("inv_ilr produced values outside [0,1]")
    return np.clip(u, 0.0, 1.0)


def plot_r_ap_ml_pair(
    u_r_ap: np.ndarray,
    u_r_ml: np.ndarray,
    *,
    ap_grid: np.ndarray,
    ml_grid: np.ndarray,
    r_grid: np.ndarray,
    alpha_r_ap: np.ndarray | None = None,
    alpha_r_ml: np.ndarray | None = None,
    out_png: Path,
    title: str,
) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    ap_grid = np.asarray(ap_grid, dtype=float).reshape(-1)
    ml_grid = np.asarray(ml_grid, dtype=float).reshape(-1)
    r_grid = np.asarray(r_grid, dtype=float).reshape(-1)
    u_r_ap = np.asarray(u_r_ap, dtype=float)
    u_r_ml = np.asarray(u_r_ml, dtype=float)

    expected_ap = (int(r_grid.size), int(ap_grid.size))
    expected_ml = (int(r_grid.size), int(ml_grid.size))
    if u_r_ap.shape != expected_ap:
        raise ValueError(f"u_r_ap shape {u_r_ap.shape} does not match (len(r_grid), len(ap_grid))={expected_ap}")
    if u_r_ml.shape != expected_ml:
        raise ValueError(f"u_r_ml shape {u_r_ml.shape} does not match (len(r_grid), len(ml_grid))={expected_ml}")

    if alpha_r_ap is not None:
        alpha_r_ap = np.asarray(alpha_r_ap, dtype=float)
        if alpha_r_ap.shape != expected_ap:
            raise ValueError(f"alpha_r_ap shape {alpha_r_ap.shape} does not match {expected_ap}")
        if not np.all(np.isfinite(alpha_r_ap)) or np.any(alpha_r_ap < 0.0) or np.any(alpha_r_ap > 1.0):
            raise ValueError("alpha_r_ap must be finite and in [0,1]")
    if alpha_r_ml is not None:
        alpha_r_ml = np.asarray(alpha_r_ml, dtype=float)
        if alpha_r_ml.shape != expected_ml:
            raise ValueError(f"alpha_r_ml shape {alpha_r_ml.shape} does not match {expected_ml}")
        if not np.all(np.isfinite(alpha_r_ml)) or np.any(alpha_r_ml < 0.0) or np.any(alpha_r_ml > 1.0):
            raise ValueError("alpha_r_ml must be finite and in [0,1]")
    if (alpha_r_ap is None) != (alpha_r_ml is None):
        raise ValueError("alpha_r_ap and alpha_r_ml must be provided together (or both None)")

    extent_ap = [float(ap_grid.min()), float(ap_grid.max()), float(r_grid.min()), float(r_grid.max())]
    extent_ml = [float(ml_grid.min()), float(ml_grid.max()), float(r_grid.min()), float(r_grid.max())]

    cmap = plt.get_cmap("turbo").copy()
    cmap.set_bad(color="lightgray")
    norm = matplotlib.colors.Normalize(vmin=0.0, vmax=1.0, clip=False)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.5), dpi=140, sharey=True)
    ax_ap, ax_ml = axes
    if alpha_r_ap is None or alpha_r_ml is None:
        ax_ap.imshow(u_r_ap, origin="lower", aspect="auto", extent=extent_ap, cmap=cmap, norm=norm)
        ax_ml.imshow(u_r_ml, origin="lower", aspect="auto", extent=extent_ml, cmap=cmap, norm=norm)
    else:
        # Instead of making outside-hull regions "transparent", blend them towards gray.
        base = np.array([0.65, 0.65, 0.65, 1.0], dtype=np.float64)
        rgba_ap = np.asarray(cmap(norm(u_r_ap)), dtype=np.float64)
        rgba_ml = np.asarray(cmap(norm(u_r_ml)), dtype=np.float64)
        w_ap = np.asarray(alpha_r_ap, dtype=np.float64)
        w_ml = np.asarray(alpha_r_ml, dtype=np.float64)
        w_ap = np.where(np.isfinite(u_r_ap), w_ap, 1.0)
        w_ml = np.where(np.isfinite(u_r_ml), w_ml, 1.0)
        rgba_ap = base[None, None, :] * (1.0 - w_ap[:, :, None]) + rgba_ap * w_ap[:, :, None]
        rgba_ml = base[None, None, :] * (1.0 - w_ml[:, :, None]) + rgba_ml * w_ml[:, :, None]
        rgba_ap[..., 3] = 1.0
        rgba_ml[..., 3] = 1.0
        ax_ap.imshow(rgba_ap, origin="lower", aspect="auto", extent=extent_ap)
        ax_ml.imshow(rgba_ml, origin="lower", aspect="auto", extent=extent_ml)

    ax_ap.set_xlabel("AP_um")
    ax_ap.set_ylabel("r_um")
    ax_ap.set_title("loading over AP/r")

    ax_ml.set_xlabel("ML_um")
    ax_ml.set_title("loading over ML/r")

    # Avoid tight_layout/colorbar overlap by allocating an explicit colorbar axis.
    fig.subplots_adjust(left=0.07, right=0.88, bottom=0.1, top=0.86, wspace=0.15)
    cax = fig.add_axes((0.9, 0.15, 0.02, 0.65))
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, cax=cax, label="predicted loading (simplex)")
    fig.suptitle(str(title))
    fig.savefig(out_png)
    plt.close(fig)

def _convex_hull_path(ap: np.ndarray, ml: np.ndarray):
    ap = np.asarray(ap, dtype=float).reshape(-1)
    ml = np.asarray(ml, dtype=float).reshape(-1)
    ok = np.isfinite(ap) & np.isfinite(ml)
    pts = np.column_stack([ap[ok], ml[ok]])
    if pts.shape[0] < 3:
        return None
    pts_u = np.unique(pts, axis=0)
    if pts_u.shape[0] < 3:
        return None
    from matplotlib.path import Path as MplPath
    from scipy.spatial import ConvexHull

    hull = ConvexHull(pts_u)
    poly = pts_u[hull.vertices]
    return MplPath(poly, closed=True)


def _predict_link_no_re(
    *,
    predictor: RPredictor,
    fit,
    newdata: pd.DataFrame,
    exclude_terms: tuple[str, ...] = ("s(animal)", "s(ab)"),
) -> np.ndarray:
    """
    Return link predictions with selected random-effect smooth terms removed.

    We compute eta=E[z|x] on the link scale and subtract the corresponding columns
    from predict(type='terms'). This avoids mgcv's 'exclude=' plumbing and keeps
    factor requirements (animal/ab) satisfied in newdata.
    """
    eta = predictor.predict_link(fit, newdata).astype(np.float64, copy=False)
    terms, _se, term_names = predictor.predict_terms_se(fit, newdata)
    drop_idx: list[int] = []
    for i, nm in enumerate(term_names):
        s = str(nm).replace(" ", "")
        if any(s.startswith(t.replace(" ", "")) for t in exclude_terms):
            drop_idx.append(i)
    if not drop_idx:
        return eta
    return eta - np.sum(terms[:, drop_idx], axis=1)


def _predict_link_excluding_terms(
    *,
    predictor: RPredictor,
    fit,
    newdata: pd.DataFrame,
    exclude_terms: tuple[str, ...],
) -> np.ndarray:
    eta = predictor.predict_link(fit, newdata).astype(np.float64, copy=False)
    if not exclude_terms:
        return eta
    terms, _se, term_names = predictor.predict_terms_se(fit, newdata)
    drop_idx: list[int] = []
    for i, nm in enumerate(term_names):
        s = str(nm).replace(" ", "")
        if any(s.startswith(t.replace(" ", "")) for t in exclude_terms):
            drop_idx.append(i)
    if not drop_idx:
        return eta
    return eta - np.sum(terms[:, drop_idx], axis=1)


def main() -> int:
    p = argparse.ArgumentParser(description="Plot logistic-normal (ALR) simplex topic fits on native projection.")
    p.add_argument("panel_dir", type=Path, help="Panel directory containing cells.tsv and simplex_meta.json.")
    p.add_argument("--meta", type=Path, default=None, help="Path to simplex_meta.json (default: <panel_dir>/simplex_meta.json).")
    p.add_argument(
        "--label-tsv",
        type=Path,
        default=Path(
            "/fast2/cs_outputs/fishtools2/_out/cnmf_all_progenitors/annotations/program_annotations.k9.dt0.1.curated.tsv"
        ),
    )
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument(
        "--exclude-random-effects",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If set, subtract s(animal) and s(ab) terms from predictions before simplex inversion.",
    )
    p.add_argument(
        "--marginalize-animal",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "If set, average predicted simplex loadings across animal levels. "
            "This keeps s(animal) but excludes s(ab) to avoid conditioning on a specific batch."
        ),
    )
    p.add_argument("--plot-apmlr", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--apmlr-out-dir", type=Path, default=None)
    p.add_argument("--apml-n", type=int, default=256, help="Number of ML bins for AP/ML native grid (also used for AP grid).")
    p.add_argument("--apmlr-r-n", type=int, default=200, help="Number of r bins for r×AP and r×ML plots.")
    p.add_argument(
        "--mask-apml-by-hull",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If set, gray out AP/ML native projection vertices outside the convex hull of (AP_um, ML_um) samples.",
    )
    p.add_argument(
        "--mask-apmlr-by-hull",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If set, gray out AP/r and ML/r pixels whose (AP,ML) location is outside the convex hull of samples.",
    )
    p.add_argument(
        "--hull-fade-alpha",
        type=float,
        default=0.0,
        help=(
            "Outside-hull weight in [0,1]. 0 renders outside-hull regions as solid gray; "
            "1 disables hull grayout (no effect)."
        ),
    )
    p.add_argument("--mask-apmlr-by-support", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--restrict-t-neomeso", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--gray-context", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--elev-deg", type=float, default=-10.0)
    p.add_argument("--azim-deg", type=float, default=-110.0)
    p.add_argument("--roll-deg", type=float, default=180.0)
    p.add_argument("--latlon", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--graticule", choices=("apml", "param", "ijk"), default="ijk")
    p.add_argument("--lat-stride", type=int, default=10)
    p.add_argument("--lon-stride", type=int, default=10)
    p.add_argument("--max-lat-lines", type=int, default=10)
    p.add_argument("--max-lon-lines", type=int, default=10)
    p.add_argument("--montage-cols", type=int, default=3)
    p.add_argument("--montage-scale-bar", choices=("first", "all", "none"), default="first")
    p.add_argument("--refextract-outdir", type=Path, default=Path("ccf/out/refextract/midsurface_neocortex_mesocortex_allocortex_3d"))
    p.add_argument("--refextract-slice-i-min", type=int, default=161)
    p.add_argument("--refextract-slice-i-max", type=int, default=305)
    p.add_argument("--refextract-n-t", type=int, default=257)
    p.add_argument("--refextract-ref-t", type=float, default=0.5)
    p.add_argument("--refextract-band-frac", type=float, default=0.15)
    p.add_argument("--refextract-res-ijk-um", type=float, nargs=3, default=(20.0, 20.0, 20.0), metavar=("RI", "RJ", "RK"))
    args = p.parse_args()
    hull_fade_alpha = float(args.hull_fade_alpha)
    if not np.isfinite(hull_fade_alpha) or hull_fade_alpha < 0.0 or hull_fade_alpha > 1.0:
        raise ValueError("--hull-fade-alpha must be in [0,1]")

    panel_dir = Path(args.panel_dir).expanduser()
    meta_path = (panel_dir / "simplex_meta.json") if args.meta is None else Path(args.meta).expanduser()
    meta = _load_meta(meta_path)

    topic_ids = [int(x) for x in meta["topic_ids"]]  # type: ignore[arg-type]
    transform = str(meta.get("transform") or "alr")
    if transform not in {"alr", "ilr"}:
        raise ValueError(f"{meta_path}: unsupported transform={transform!r} (expected 'alr' or 'ilr')")
    ref_topic = None if transform != "alr" else int(meta["ref_topic"])  # type: ignore[arg-type]
    fits_dir = Path(str(meta["fits_dir"])).expanduser()
    if not fits_dir.is_dir():
        raise FileNotFoundError(f"fits_dir not found: {fits_dir}")
    if transform == "alr":
        if ref_topic is None or int(ref_topic) not in set(topic_ids):
            raise ValueError(f"ref_topic {ref_topic} not in topic_ids")
        ref_index = topic_ids.index(int(ref_topic))
    else:
        ref_index = -1

    out_dir = (panel_dir / "plots_native_proj_simplex_u") if args.out_dir is None else Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    cells = pd.read_csv(panel_dir / "cells.tsv", sep="\t")
    required = {"r_um", "AP_um", "ML_um"}
    missing = sorted(required.difference(cells.columns))
    if missing:
        raise ValueError(f"{panel_dir / 'cells.tsv'}: missing required columns: {missing}")
    r_um = pd.to_numeric(cells["r_um"], errors="coerce").to_numpy(dtype=float)
    if not np.all(np.isfinite(r_um)):
        raise ValueError("cells.tsv r_um must be finite")
    r0 = float(np.median(r_um))
    theta0 = 0.0
    if "theta" in cells.columns:
        theta = pd.to_numeric(cells["theta"], errors="coerce").to_numpy(dtype=float)
        if not np.all(np.isfinite(theta)):
            raise ValueError("cells.tsv theta must be finite when present")
        theta0 = float(0.0 if theta.size <= 0 else np.median(theta))
    ap0 = float(np.nanmedian(pd.to_numeric(cells["AP_um"], errors="coerce").to_numpy(dtype=float)))
    ml0 = float(np.nanmedian(pd.to_numeric(cells["ML_um"], errors="coerce").to_numpy(dtype=float)))
    if not np.isfinite(ap0) or not np.isfinite(ml0):
        raise ValueError("cells.tsv AP_um/ML_um must be finite")

    predictor = RPredictor()
    if bool(args.marginalize_animal) and bool(args.exclude_random_effects):
        raise ValueError("--marginalize-animal is incompatible with --exclude-random-effects (animal term would be removed).")

    surface_ctx = build_apml_native_surface_projection_context(
        outdir=Path(args.refextract_outdir).expanduser(),
        slice_i_min=int(args.refextract_slice_i_min),
        slice_i_max=int(args.refextract_slice_i_max),
        n_t=int(args.refextract_n_t),
        ref_t=float(args.refextract_ref_t),
        band_frac=float(args.refextract_band_frac),
        res_ijk_um=(
            float(args.refextract_res_ijk_um[0]),
            float(args.refextract_res_ijk_um[1]),
            float(args.refextract_res_ijk_um[2]),
        ),
        elev_deg=float(args.elev_deg),
        azim_deg=float(args.azim_deg),
        roll_deg=float(args.roll_deg),
    )

    support_flat = np.asarray(surface_ctx["support_flat"], dtype=bool).reshape(-1)
    neomeso_flat = np.asarray(surface_ctx["neomeso_flat"], dtype=bool).reshape(-1)
    predict_mask = support_flat & (neomeso_flat if bool(args.restrict_t_neomeso) else True)
    if not np.any(predict_mask):
        raise ValueError("No vertices available for prediction after applying support masks")

    ap_flat = np.asarray(surface_ctx["ap_um_flat"], dtype=np.float64).reshape(-1)
    ml_flat = np.asarray(surface_ctx["ml_um_flat"], dtype=np.float64).reshape(-1)
    ap_s = ap_flat[predict_mask]
    ml_s = ml_flat[predict_mask]

    hull_path = None
    if bool(args.mask_apml_by_hull) or bool(args.mask_apmlr_by_hull):
        ap_cells = pd.to_numeric(cells["AP_um"], errors="coerce").to_numpy(dtype=float)
        ml_cells = pd.to_numeric(cells["ML_um"], errors="coerce").to_numpy(dtype=float)
        r_cells = pd.to_numeric(cells["r_um"], errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(ap_cells) & np.isfinite(ml_cells) & np.isfinite(r_cells)
        fit_r_max = meta.get("r_max")
        if fit_r_max is not None:
            try:
                ok &= r_cells <= float(fit_r_max)
            except Exception:
                raise ValueError(f"simplex_meta.json r_max is not numeric: {fit_r_max!r}")
        hull_path = _convex_hull_path(ap_cells[ok], ml_cells[ok])
        if hull_path is None:
            raise ValueError("Requested convex-hull masking, but could not construct a hull (too few unique points).")

    batch_ref_meta = meta.get("batch_ref")
    batch_ref = None if batch_ref_meta is None else str(batch_ref_meta)
    animal_ref_meta = meta.get("animal_ref")
    animal_ref_meta_str = None if animal_ref_meta is None else str(animal_ref_meta)

    if transform == "alr":
        assert ref_topic is not None
        # Predict ALR components at surface vertices.
        nonref_topics = [pid for pid in topic_ids if pid != int(ref_topic)]
        z_hat = np.zeros((int(ap_s.size), len(nonref_topics)), dtype=np.float64)
        for j, pid in enumerate(nonref_topics):
            gene = f"ALR_P{int(pid)}_vs_P{int(ref_topic)}"
            fit_path = fits_dir / f"{gene}.gam.rds"
            if not fit_path.exists():
                raise FileNotFoundError(fit_path)
            fit = predictor.read_fit(fit_path)
            animal_levels = predictor.animal_levels(fit)
            animal_ref = animal_ref_meta_str
            if animal_levels and (animal_ref is None or animal_ref not in animal_levels):
                animal_ref = _infer_animal_ref_from_batch_values(pd.Series(cells["batch"]), animal_levels)
            if animal_levels and animal_ref is None:
                animal_ref = animal_levels[0]

            config = GAMPredictorConfig(
                mean_log_sf=None,
                include_pos=False,
                include_batch=False,
                batch_ref=batch_ref,
                batch_levels=[],
                include_animal=bool(animal_levels),
                animal_ref=animal_ref,
                animal_levels=animal_levels,
                include_log_sf_c=False,
            )
            nd = make_newdata_for_fit(
                predictor=predictor,
                fit=fit,
                n=int(ap_s.size),
                r_um=r0,
                theta=theta0,
                ap_um=ap_s,
                ml_um=ml_s,
                config=config,
            )
            if bool(args.exclude_random_effects):
                z_hat[:, j] = _predict_link_no_re(predictor=predictor, fit=fit, newdata=nd)
            elif bool(args.marginalize_animal):
                # We'll fill this in below via per-animal averaging on the simplex.
                pass
            else:
                z_hat[:, j] = predictor.predict_link(fit, nd).astype(np.float64, copy=False)

        if bool(args.marginalize_animal):
            u_acc = np.zeros((int(ap_s.size), len(topic_ids)), dtype=np.float64)
            n_animals = 0
            # Discover animal levels from the first fit (they're shared across fits).
            first_fit = predictor.read_fit(fits_dir / f"ALR_P{int(nonref_topics[0])}_vs_P{int(ref_topic)}.gam.rds")
            animal_levels_all = predictor.animal_levels(first_fit)
            if not animal_levels_all:
                raise ValueError("Requested --marginalize-animal but fit has no animal levels.")
            for animal_level in animal_levels_all:
                z_hat_a = np.zeros((int(ap_s.size), len(nonref_topics)), dtype=np.float64)
                for j, pid in enumerate(nonref_topics):
                    gene = f"ALR_P{int(pid)}_vs_P{int(ref_topic)}"
                    fit = predictor.read_fit(fits_dir / f"{gene}.gam.rds")
                    animal_levels = predictor.animal_levels(fit)
                    config = GAMPredictorConfig(
                        mean_log_sf=None,
                        include_pos=False,
                        include_batch=False,
                        batch_ref=batch_ref,
                        batch_levels=[],
                        include_animal=bool(animal_levels),
                        animal_ref=str(animal_level),
                        animal_levels=animal_levels,
                        include_log_sf_c=False,
                    )
                    nd = make_newdata_for_fit(
                        predictor=predictor,
                        fit=fit,
                        n=int(ap_s.size),
                        r_um=r0,
                        theta=theta0,
                        ap_um=ap_s,
                        ml_um=ml_s,
                        config=config,
                    )
                    # Exclude only ab to avoid conditioning on a specific batch within animal.
                    z_hat_a[:, j] = _predict_link_excluding_terms(
                        predictor=predictor, fit=fit, newdata=nd, exclude_terms=("s(ab)",)
                    )
                u_acc += inv_alr(z_hat_a, ref_index=ref_index)
                n_animals += 1
            u_hat = u_acc / float(n_animals)
        else:
            u_hat = inv_alr(z_hat, ref_index=ref_index)
        if u_hat.shape != (int(ap_s.size), len(topic_ids)):
            raise RuntimeError("Unexpected inv_alr output shape")
    else:
        coord_names = meta.get("coord_names")
        if not isinstance(coord_names, list) or not coord_names:
            raise ValueError(f"{meta_path}: missing/invalid coord_names for ILR")
        V = ilr_basis_pivot(len(topic_ids))
        z_hat = np.zeros((int(ap_s.size), int(len(coord_names))), dtype=np.float64)
        for j, nm in enumerate(coord_names):
            gene = str(nm)
            fit_path = fits_dir / f"{gene}.gam.rds"
            if not fit_path.exists():
                raise FileNotFoundError(fit_path)
            fit = predictor.read_fit(fit_path)
            animal_levels = predictor.animal_levels(fit)
            animal_ref = animal_ref_meta_str
            if animal_levels and (animal_ref is None or animal_ref not in animal_levels):
                animal_ref = _infer_animal_ref_from_batch_values(pd.Series(cells["batch"]), animal_levels)
            if animal_levels and animal_ref is None:
                animal_ref = animal_levels[0]
            config = GAMPredictorConfig(
                mean_log_sf=None,
                include_pos=False,
                include_batch=False,
                batch_ref=batch_ref,
                batch_levels=[],
                include_animal=bool(animal_levels),
                animal_ref=animal_ref,
                animal_levels=animal_levels,
                include_log_sf_c=False,
            )
            nd = make_newdata_for_fit(
                predictor=predictor,
                fit=fit,
                n=int(ap_s.size),
                r_um=r0,
                theta=theta0,
                ap_um=ap_s,
                ml_um=ml_s,
                config=config,
            )
            if bool(args.exclude_random_effects):
                z_hat[:, j] = _predict_link_no_re(predictor=predictor, fit=fit, newdata=nd)
            elif bool(args.marginalize_animal):
                pass
            else:
                z_hat[:, j] = predictor.predict_link(fit, nd).astype(np.float64, copy=False)
        if bool(args.marginalize_animal):
            u_acc = np.zeros((int(ap_s.size), len(topic_ids)), dtype=np.float64)
            n_animals = 0
            first_fit = predictor.read_fit(fits_dir / f"{str(coord_names[0])}.gam.rds")
            animal_levels_all = predictor.animal_levels(first_fit)
            if not animal_levels_all:
                raise ValueError("Requested --marginalize-animal but fit has no animal levels.")
            for animal_level in animal_levels_all:
                z_hat_a = np.zeros((int(ap_s.size), int(len(coord_names))), dtype=np.float64)
                for j, nm in enumerate(coord_names):
                    gene = str(nm)
                    fit = predictor.read_fit(fits_dir / f"{gene}.gam.rds")
                    animal_levels = predictor.animal_levels(fit)
                    config = GAMPredictorConfig(
                        mean_log_sf=None,
                        include_pos=False,
                        include_batch=False,
                        batch_ref=batch_ref,
                        batch_levels=[],
                        include_animal=bool(animal_levels),
                        animal_ref=str(animal_level),
                        animal_levels=animal_levels,
                        include_log_sf_c=False,
                    )
                    nd = make_newdata_for_fit(
                        predictor=predictor,
                        fit=fit,
                        n=int(ap_s.size),
                        r_um=r0,
                        theta=theta0,
                        ap_um=ap_s,
                        ml_um=ml_s,
                        config=config,
                    )
                    z_hat_a[:, j] = _predict_link_excluding_terms(
                        predictor=predictor, fit=fit, newdata=nd, exclude_terms=("s(ab)",)
                    )
                u_acc += inv_ilr(z_hat_a, V=V)
                n_animals += 1
            u_hat = u_acc / float(n_animals)
        else:
            u_hat = inv_ilr(z_hat, V=V)
        if u_hat.shape != (int(ap_s.size), len(topic_ids)):
            raise RuntimeError("Unexpected inv_ilr output shape")

    topic_titles = _load_topic_titles(topic_ids, None if args.label_tsv is None else Path(args.label_tsv).expanduser())

    x2d = np.asarray(surface_ctx["x2d"], dtype=np.float64).reshape(-1)
    y2d = np.asarray(surface_ctx["y2d"], dtype=np.float64).reshape(-1)
    z2d = np.asarray(surface_ctx["z2d"], dtype=np.float64).reshape(-1)
    x3d = np.asarray(surface_ctx["x3"], dtype=np.float64).reshape(-1)
    y3d = np.asarray(surface_ctx["y3"], dtype=np.float64).reshape(-1)
    z3d = np.asarray(surface_ctx["z3"], dtype=np.float64).reshape(-1)
    faces = np.asarray(surface_ctx["faces"], dtype=np.int32)
    tri_support = np.asarray(surface_ctx["tri_support"], dtype=bool).reshape(-1)
    tri_neomeso = np.asarray(surface_ctx["tri_neomeso"], dtype=bool).reshape(-1)
    ordered_geom = (
        surface_ctx["ordered_geom_neomeso"]
        if (bool(args.restrict_t_neomeso) and not bool(args.gray_context))
        else surface_ctx["ordered_geom_support"]
    )
    n_rows = int(surface_ctx["n_rows"])
    n_cols = int(surface_ctx["n_cols"])

    cmap = plt.get_cmap("turbo").copy()
    cmap.set_bad(color="lightgray")
    gene_values: list[tuple[str, np.ndarray]] = []
    for k, title in enumerate(topic_titles):
        vals_all = np.full(support_flat.shape, np.nan, dtype=np.float64)
        vals_all[predict_mask] = u_hat[:, k]
        out_png = out_dir / safe_gene_name(f"P{topic_ids[k]}") / PNG_NAME
        tri_alpha = None
        if hull_path is not None and bool(args.mask_apml_by_hull):
            if ordered_geom is None:
                raise ValueError("--mask-apml-by-hull requires ordered geometry for stable triangle alpha fading")
            tris_k = np.asarray(ordered_geom["tris"], dtype=np.int32)
            # Use triangle centroid-in-hull for a more visible boundary than vertex-fraction smoothing.
            cent_ap = np.mean(ap_flat[tris_k], axis=1)
            cent_ml = np.mean(ml_flat[tris_k], axis=1)
            inside_t = hull_path.contains_points(np.column_stack([cent_ap, cent_ml]), radius=1e-9).astype(np.float64)
            tri_alpha = hull_fade_alpha + (1.0 - hull_fade_alpha) * inside_t
        plot_coronal_surface_projection(
            vals_all,
            x2d=x2d,
            y2d=y2d,
            z2d=z2d,
            x3d=x3d,
            y3d=y3d,
            z3d=z3d,
            faces=faces,
            tri_support=tri_support,
            tri_neomeso=tri_neomeso,
            restrict_t_neomeso=bool(args.restrict_t_neomeso),
            gray_context=bool(args.gray_context),
            latlon=bool(args.latlon),
            graticule=str(args.graticule),
            lat_stride=int(args.lat_stride),
            lon_stride=int(args.lon_stride),
            max_lat_lines=int(args.max_lat_lines),
            max_lon_lines=int(args.max_lon_lines),
            vertex_support=support_flat,
            vertex_neomeso=neomeso_flat,
            vertex_ap_um=ap_flat,
            vertex_ml_um=ml_flat,
            n_rows=n_rows,
            n_cols=n_cols,
            shade=False,
            shade_strength=0.75,
            shade_elev_deg=float(args.elev_deg),
            shade_azim_deg=float(args.azim_deg),
            ordered_geometry=ordered_geom,
            tri_alpha=tri_alpha,
            out_png=out_png,
            title=str(title),
            cmap=cmap,
            cbar_label="predicted loading (simplex)",
            cbar_ticks=None,
            cbar_ticklabels=None,
            vmin=0.0,
            vmax=1.0,
        )
        gene_values.append((str(title), vals_all))

    write_apml_native_proj_montage(
        gene_values=gene_values,
        out_png=out_dir / MONTAGE_NAME,
        ncols=max(1, min(int(args.montage_cols), len(gene_values))),
        suptitle="Simplex topic model (logistic-normal)",
        scale_bar=str(args.montage_scale_bar),
        x2d=x2d,
        y2d=y2d,
        z2d=z2d,
        x3d=x3d,
        y3d=y3d,
        z3d=z3d,
        faces=faces,
        tri_support=tri_support,
        tri_neomeso=tri_neomeso,
        restrict_t_neomeso=bool(args.restrict_t_neomeso),
        gray_context=bool(args.gray_context),
        latlon=bool(args.latlon),
        graticule=str(args.graticule),
        lat_stride=int(args.lat_stride),
        lon_stride=int(args.lon_stride),
        max_lat_lines=int(args.max_lat_lines),
        max_lon_lines=int(args.max_lon_lines),
        vertex_support=support_flat,
        vertex_neomeso=neomeso_flat,
        vertex_ap_um=ap_flat,
        vertex_ml_um=ml_flat,
        n_rows=n_rows,
        n_cols=n_cols,
        shade=False,
        shade_strength=0.75,
        shade_elev_deg=float(args.elev_deg),
        shade_azim_deg=float(args.azim_deg),
        ordered_geometry=ordered_geom,
        cmap=cmap,
        cbar_label="predicted loading (simplex)",
        cbar_ticks=None,
        cbar_ticklabels=None,
        vmin=0.0,
        vmax=1.0,
    )

    if bool(args.plot_apmlr):
        apmlr_out_dir = (
            (panel_dir / "plots_apmlr_simplex_u") if args.apmlr_out_dir is None else Path(args.apmlr_out_dir).expanduser()
        )
        apmlr_out_dir.mkdir(parents=True, exist_ok=True)

        refextract_res_ijk_um = (
            None
            if args.refextract_res_ijk_um is None
            else (
                float(args.refextract_res_ijk_um[0]),
                float(args.refextract_res_ijk_um[1]),
                float(args.refextract_res_ijk_um[2]),
            )
        )
        apml_native_grid = compute_ap_ml_native_grid_details(
            outdir=Path(args.refextract_outdir).expanduser(),
            slice_i_min=int(args.refextract_slice_i_min),
            slice_i_max=int(args.refextract_slice_i_max),
            n_t=int(args.refextract_n_t),
            n_ml=int(args.apml_n),
            ref_t=float(args.refextract_ref_t),
            band_frac=float(args.refextract_band_frac),
            res_ijk_um=refextract_res_ijk_um,
            restrict_t_neomeso=False,
        )
        ap_grid = np.asarray(apml_native_grid.ap_um, dtype=float)
        ml_grid = np.asarray(apml_native_grid.ml_um, dtype=float)
        apml_support_mask = np.asarray(apml_native_grid.support_mask, dtype=bool)
        if bool(args.restrict_t_neomeso):
            apml_support_mask = compute_ap_ml_support_mask_neomeso_on_native_grid(
                outdir=Path(args.refextract_outdir).expanduser(),
                grid=apml_native_grid,
            )

        j_ml0 = int(np.nanargmin(np.abs(ml_grid - float(ml0))))
        i_ap0 = int(np.nanargmin(np.abs(ap_grid - float(ap0))))
        keep_ap = apml_support_mask[:, j_ml0].astype(bool, copy=False)
        keep_ml = apml_support_mask[i_ap0, :].astype(bool, copy=False)
        ap_keep_idx = np.flatnonzero(keep_ap)
        ml_keep_idx = np.flatnonzero(keep_ml)
        ap_slice = slice(int(ap_keep_idx.min()), int(ap_keep_idx.max()) + 1) if ap_keep_idx.size else slice(None)
        ml_slice = slice(int(ml_keep_idx.min()), int(ml_keep_idx.max()) + 1) if ml_keep_idx.size else slice(None)

        r_hi = float(np.nanmax(pd.to_numeric(cells["r_um"], errors="coerce").to_numpy(dtype=float)))
        fit_r_max = meta.get("r_max")
        if fit_r_max is not None:
            try:
                r_hi = min(r_hi, float(fit_r_max))
            except Exception:
                raise ValueError(f"simplex_meta.json r_max is not numeric: {fit_r_max!r}")
        r_grid = np.linspace(0.0, float(r_hi), int(args.apmlr_r_n))
        rr_ap, aa = np.meshgrid(r_grid, ap_grid, indexing="ij")
        rr_ml, mm = np.meshgrid(r_grid, ml_grid, indexing="ij")

        if transform == "alr":
            assert ref_topic is not None
            nonref_topics = [pid for pid in topic_ids if pid != int(ref_topic)]
            if bool(args.marginalize_animal):
                u_apr_acc = np.zeros((int(rr_ap.size), len(topic_ids)), dtype=np.float64)
                u_mlr_acc = np.zeros((int(rr_ml.size), len(topic_ids)), dtype=np.float64)
                n_animals = 0
                first_fit = predictor.read_fit(fits_dir / f"ALR_P{int(nonref_topics[0])}_vs_P{int(ref_topic)}.gam.rds")
                animal_levels_all = predictor.animal_levels(first_fit)
                if not animal_levels_all:
                    raise ValueError("Requested --marginalize-animal but fit has no animal levels.")
                for animal_level in animal_levels_all:
                    z_hat_apr = np.zeros((int(rr_ap.size), len(nonref_topics)), dtype=np.float64)
                    z_hat_mlr = np.zeros((int(rr_ml.size), len(nonref_topics)), dtype=np.float64)
                    for j, pid in enumerate(nonref_topics):
                        gene = f"ALR_P{int(pid)}_vs_P{int(ref_topic)}"
                        fit = predictor.read_fit(fits_dir / f"{gene}.gam.rds")
                        animal_levels = predictor.animal_levels(fit)
                        config = GAMPredictorConfig(
                            mean_log_sf=None,
                            include_pos=False,
                            include_batch=False,
                            batch_ref=batch_ref,
                            batch_levels=[],
                            include_animal=bool(animal_levels),
                            animal_ref=str(animal_level),
                            animal_levels=animal_levels,
                            include_log_sf_c=False,
                        )
                        nd_apr = make_newdata_for_fit(
                            predictor=predictor,
                            fit=fit,
                            n=int(rr_ap.size),
                            r_um=rr_ap.ravel(),
                            theta=theta0,
                            ap_um=aa.ravel(),
                            ml_um=ml0,
                            config=config,
                        )
                        nd_mlr = make_newdata_for_fit(
                            predictor=predictor,
                            fit=fit,
                            n=int(rr_ml.size),
                            r_um=rr_ml.ravel(),
                            theta=theta0,
                            ap_um=ap0,
                            ml_um=mm.ravel(),
                            config=config,
                        )
                        z_hat_apr[:, j] = _predict_link_excluding_terms(
                            predictor=predictor, fit=fit, newdata=nd_apr, exclude_terms=("s(ab)",)
                        )
                        z_hat_mlr[:, j] = _predict_link_excluding_terms(
                            predictor=predictor, fit=fit, newdata=nd_mlr, exclude_terms=("s(ab)",)
                        )
                    u_apr_acc += inv_alr(z_hat_apr, ref_index=ref_index)
                    u_mlr_acc += inv_alr(z_hat_mlr, ref_index=ref_index)
                    n_animals += 1
                u_apr = u_apr_acc / float(n_animals)
                u_mlr = u_mlr_acc / float(n_animals)
            else:
                z_hat_apr = np.zeros((int(rr_ap.size), len(nonref_topics)), dtype=np.float64)
                z_hat_mlr = np.zeros((int(rr_ml.size), len(nonref_topics)), dtype=np.float64)
                for j, pid in enumerate(nonref_topics):
                    gene = f"ALR_P{int(pid)}_vs_P{int(ref_topic)}"
                    fit_path = fits_dir / f"{gene}.gam.rds"
                    fit = predictor.read_fit(fit_path)
                    animal_levels = predictor.animal_levels(fit)
                    animal_ref = animal_ref_meta_str
                    if animal_levels and (animal_ref is None or animal_ref not in animal_levels):
                        animal_ref = _infer_animal_ref_from_batch_values(pd.Series(cells["batch"]), animal_levels)
                    if animal_levels and animal_ref is None:
                        animal_ref = animal_levels[0]
                    config = GAMPredictorConfig(
                        mean_log_sf=None,
                        include_pos=False,
                        include_batch=False,
                        batch_ref=batch_ref,
                        batch_levels=[],
                        include_animal=bool(animal_levels),
                        animal_ref=animal_ref,
                        animal_levels=animal_levels,
                        include_log_sf_c=False,
                    )
                    nd_apr = make_newdata_for_fit(
                        predictor=predictor,
                        fit=fit,
                        n=int(rr_ap.size),
                        r_um=rr_ap.ravel(),
                        theta=theta0,
                        ap_um=aa.ravel(),
                        ml_um=ml0,
                        config=config,
                    )
                    nd_mlr = make_newdata_for_fit(
                        predictor=predictor,
                        fit=fit,
                        n=int(rr_ml.size),
                        r_um=rr_ml.ravel(),
                        theta=theta0,
                        ap_um=ap0,
                        ml_um=mm.ravel(),
                        config=config,
                    )
                    if bool(args.exclude_random_effects):
                        z_hat_apr[:, j] = _predict_link_no_re(predictor=predictor, fit=fit, newdata=nd_apr)
                        z_hat_mlr[:, j] = _predict_link_no_re(predictor=predictor, fit=fit, newdata=nd_mlr)
                    else:
                        z_hat_apr[:, j] = predictor.predict_link(fit, nd_apr).astype(np.float64, copy=False)
                        z_hat_mlr[:, j] = predictor.predict_link(fit, nd_mlr).astype(np.float64, copy=False)
                u_apr = inv_alr(z_hat_apr, ref_index=ref_index)
                u_mlr = inv_alr(z_hat_mlr, ref_index=ref_index)
        else:
            coord_names = meta.get("coord_names")
            if not isinstance(coord_names, list) or not coord_names:
                raise ValueError(f"{meta_path}: missing/invalid coord_names for ILR")
            V = ilr_basis_pivot(len(topic_ids))
            if bool(args.marginalize_animal):
                u_apr_acc = np.zeros((int(rr_ap.size), len(topic_ids)), dtype=np.float64)
                u_mlr_acc = np.zeros((int(rr_ml.size), len(topic_ids)), dtype=np.float64)
                n_animals = 0
                first_fit = predictor.read_fit(fits_dir / f"{str(coord_names[0])}.gam.rds")
                animal_levels_all = predictor.animal_levels(first_fit)
                if not animal_levels_all:
                    raise ValueError("Requested --marginalize-animal but fit has no animal levels.")
                for animal_level in animal_levels_all:
                    z_hat_apr = np.zeros((int(rr_ap.size), int(len(coord_names))), dtype=np.float64)
                    z_hat_mlr = np.zeros((int(rr_ml.size), int(len(coord_names))), dtype=np.float64)
                    for j, nm in enumerate(coord_names):
                        gene = str(nm)
                        fit = predictor.read_fit(fits_dir / f"{gene}.gam.rds")
                        animal_levels = predictor.animal_levels(fit)
                        config = GAMPredictorConfig(
                            mean_log_sf=None,
                            include_pos=False,
                            include_batch=False,
                            batch_ref=batch_ref,
                            batch_levels=[],
                            include_animal=bool(animal_levels),
                            animal_ref=str(animal_level),
                            animal_levels=animal_levels,
                            include_log_sf_c=False,
                        )
                        nd_apr = make_newdata_for_fit(
                            predictor=predictor,
                            fit=fit,
                            n=int(rr_ap.size),
                            r_um=rr_ap.ravel(),
                            theta=theta0,
                            ap_um=aa.ravel(),
                            ml_um=ml0,
                            config=config,
                        )
                        nd_mlr = make_newdata_for_fit(
                            predictor=predictor,
                            fit=fit,
                            n=int(rr_ml.size),
                            r_um=rr_ml.ravel(),
                            theta=theta0,
                            ap_um=ap0,
                            ml_um=mm.ravel(),
                            config=config,
                        )
                        z_hat_apr[:, j] = _predict_link_excluding_terms(
                            predictor=predictor, fit=fit, newdata=nd_apr, exclude_terms=("s(ab)",)
                        )
                        z_hat_mlr[:, j] = _predict_link_excluding_terms(
                            predictor=predictor, fit=fit, newdata=nd_mlr, exclude_terms=("s(ab)",)
                        )
                    u_apr_acc += inv_ilr(z_hat_apr, V=V)
                    u_mlr_acc += inv_ilr(z_hat_mlr, V=V)
                    n_animals += 1
                u_apr = u_apr_acc / float(n_animals)
                u_mlr = u_mlr_acc / float(n_animals)
            else:
                z_hat_apr = np.zeros((int(rr_ap.size), int(len(coord_names))), dtype=np.float64)
                z_hat_mlr = np.zeros((int(rr_ml.size), int(len(coord_names))), dtype=np.float64)
                for j, nm in enumerate(coord_names):
                    gene = str(nm)
                    fit_path = fits_dir / f"{gene}.gam.rds"
                    fit = predictor.read_fit(fit_path)
                    animal_levels = predictor.animal_levels(fit)
                    animal_ref = animal_ref_meta_str
                    if animal_levels and (animal_ref is None or animal_ref not in animal_levels):
                        animal_ref = _infer_animal_ref_from_batch_values(pd.Series(cells["batch"]), animal_levels)
                    if animal_levels and animal_ref is None:
                        animal_ref = animal_levels[0]
                    config = GAMPredictorConfig(
                        mean_log_sf=None,
                        include_pos=False,
                        include_batch=False,
                        batch_ref=batch_ref,
                        batch_levels=[],
                        include_animal=bool(animal_levels),
                        animal_ref=animal_ref,
                        animal_levels=animal_levels,
                        include_log_sf_c=False,
                    )
                    nd_apr = make_newdata_for_fit(
                        predictor=predictor,
                        fit=fit,
                        n=int(rr_ap.size),
                        r_um=rr_ap.ravel(),
                        theta=theta0,
                        ap_um=aa.ravel(),
                        ml_um=ml0,
                        config=config,
                    )
                    nd_mlr = make_newdata_for_fit(
                        predictor=predictor,
                        fit=fit,
                        n=int(rr_ml.size),
                        r_um=rr_ml.ravel(),
                        theta=theta0,
                        ap_um=ap0,
                        ml_um=mm.ravel(),
                        config=config,
                    )
                    if bool(args.exclude_random_effects):
                        z_hat_apr[:, j] = _predict_link_no_re(predictor=predictor, fit=fit, newdata=nd_apr)
                        z_hat_mlr[:, j] = _predict_link_no_re(predictor=predictor, fit=fit, newdata=nd_mlr)
                    else:
                        z_hat_apr[:, j] = predictor.predict_link(fit, nd_apr).astype(np.float64, copy=False)
                        z_hat_mlr[:, j] = predictor.predict_link(fit, nd_mlr).astype(np.float64, copy=False)
                u_apr = inv_ilr(z_hat_apr, V=V)
                u_mlr = inv_ilr(z_hat_mlr, V=V)

        for k, title in enumerate(topic_titles):
            u_r_ap = u_apr[:, k].reshape(len(r_grid), len(ap_grid))
            u_r_ml = u_mlr[:, k].reshape(len(r_grid), len(ml_grid))
            if bool(args.mask_apmlr_by_support):
                u_r_ap, u_r_ml = _mask_r_ap_ml_pair_by_support(
                    mu_r_ap=u_r_ap,
                    mu_r_ml=u_r_ml,
                    ap_grid=ap_grid,
                    ml_grid=ml_grid,
                    apml_support_mask=apml_support_mask,
                    ap0=ap0,
                    ml0=ml0,
                )
            # Crop plot bounds to the supported AP/ML ranges at ML0/AP0 to avoid huge empty margins.
            u_r_ap_plot = u_r_ap[:, ap_slice]
            u_r_ml_plot = u_r_ml[:, ml_slice]
            ap_grid_plot = ap_grid[ap_slice]
            ml_grid_plot = ml_grid[ml_slice]
            alpha_r_ap = None
            alpha_r_ml = None
            if hull_path is not None and bool(args.mask_apmlr_by_hull):
                assert ap_grid_plot.ndim == 1 and ml_grid_plot.ndim == 1
                ap_keep = hull_path.contains_points(
                    np.column_stack([ap_grid_plot, np.full_like(ap_grid_plot, float(ml0))]),
                    radius=1e-9,
                )
                ml_keep = hull_path.contains_points(
                    np.column_stack([np.full_like(ml_grid_plot, float(ap0)), ml_grid_plot]),
                    radius=1e-9,
                )
                alpha_r_ap = np.ones_like(u_r_ap_plot, dtype=np.float64)
                alpha_r_ml = np.ones_like(u_r_ml_plot, dtype=np.float64)
                alpha_r_ap[:, ~ap_keep] = hull_fade_alpha
                alpha_r_ml[:, ~ml_keep] = hull_fade_alpha
            out_png = apmlr_out_dir / safe_gene_name(f"P{topic_ids[k]}") / APMLR_PNG_NAME
            plot_r_ap_ml_pair(
                u_r_ap_plot,
                u_r_ml_plot,
                ap_grid=ap_grid_plot,
                ml_grid=ml_grid_plot,
                r_grid=r_grid,
                alpha_r_ap=alpha_r_ap,
                alpha_r_ml=alpha_r_ml,
                out_png=out_png,
                title=str(title),
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
