from __future__ import annotations

import argparse
import pathlib
import re

import anndata as ad
from fishtools.ccf.transforms import DEFAULT_NATIVE_CAMERA_AZIM_DEG
from fishtools.ccf.transforms import DEFAULT_NATIVE_CAMERA_ELEV_DEG
from fishtools.ccf.transforms import DEFAULT_NATIVE_CAMERA_ROLL_DEG
from fishtools.ccf.transforms import DEFAULT_NATIVE_FOCAL_LENGTH
from fishtools.ccf.transforms import DEFAULT_NATIVE_PROJ_TYPE
from fishtools.ccf.transforms import build_apml_native_surface_projection_context
from fishtools.gam.native_surface_plotting import plot_coronal_surface_projection
from fishtools.postprocess.utils_h5ad import read_obsm_h5ad
import matplotlib
import numpy as np
import pandas as pd

PLOT_CMAP = "turbo"
TS_CMAP = PLOT_CMAP
SCALAR_CMAP = PLOT_CMAP


def _expit(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return 1.0 / (1.0 + np.exp(-values))


def _dataset_animal(dataset: str) -> str:
    m = re.search(r"(jaxa\d+)", str(dataset), flags=re.IGNORECASE)
    if m is None:
        raise ValueError(f"Cannot infer animal from dataset={dataset!r}.")
    token = m.group(1)
    return token[0].upper() + token[1:]


def _read_gene_layer_values(
    adata: ad.AnnData,
    gene: str,
    *,
    layer: str,
    obs_idx: np.ndarray | None = None,
) -> np.ndarray:
    where = np.where(np.asarray(adata.var_names, dtype=object) == str(gene))[0]
    if where.size == 0:
        raise ValueError(f"Gene not found in adata.var_names: {gene}")
    if str(layer) not in adata.layers:
        raise ValueError(f"Layer {layer!r} not found in h5ad layers: {sorted(adata.layers.keys())}")
    col = int(where[0])
    rows = slice(None) if obs_idx is None else np.asarray(obs_idx, dtype=int)
    values = adata.layers[str(layer)][rows, col]
    if hasattr(values, "toarray"):
        values = values.toarray()
    return np.asarray(values, dtype=float).reshape(-1)


def _load_adata_with_external_obsm(*, h5ad_path: pathlib.Path, obsm_h5ad_path: pathlib.Path) -> ad.AnnData:
    adata = ad.read_h5ad(h5ad_path, backed="r")
    read_obsm_h5ad(obsm_h5ad_path, adata=adata)
    return adata


def _fit_binomial_logit(
    x: np.ndarray, k: np.ndarray, n: np.ndarray, *, max_iter: int = 100, tol: float = 1e-8
) -> tuple[float, float]:
    """Fit pooled binomial-logit regression: logit(p) = intercept + slope * x."""
    x = np.asarray(x, dtype=float)
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)
    if x.ndim != 1 or k.ndim != 1 or n.ndim != 1:
        raise ValueError("x, k, n must be 1D.")
    if not (x.shape == k.shape == n.shape):
        raise ValueError(f"Shape mismatch: x={x.shape} k={k.shape} n={n.shape}")
    if x.size < 2:
        raise ValueError("Need at least 2 points for regression.")
    if np.any(n <= 0):
        raise ValueError("All n must be > 0.")

    x_mean = float(np.mean(x))
    x_std = float(np.std(x))
    if x_std <= 0:
        raise ValueError("x has zero variance.")
    x_scaled = (x - x_mean) / x_std

    X = np.column_stack([np.ones_like(x_scaled), x_scaled])
    beta = np.zeros(2, dtype=float)

    for _ in range(max_iter):
        eta = X @ beta
        p = _expit(eta)
        w = n * p * (1.0 - p)
        xtwx = X.T @ (X * w[:, None])
        grad = X.T @ (k - (n * p))
        step = np.linalg.solve(xtwx, grad)
        beta_new = beta + step
        if np.max(np.abs(step)) < tol:
            beta = beta_new
            break
        beta = beta_new

    slope = float(beta[1] / x_std)
    intercept = float(beta[0] - (beta[1] * x_mean / x_std))
    return intercept, slope


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Quantify and plot BrdU/EdU only/dual proportions binned along AP and ML "
            "from obsm['AP_ML_um'], restricted to selected Leiden clusters."
        )
    )
    p.add_argument(
        "--h5ad",
        type=pathlib.Path,
        default=pathlib.Path("~/nvme/all_r300.h5ad"),
        help="Input .h5ad (default: %(default)s)",
    )
    p.add_argument(
        "--obsm-h5ad",
        type=pathlib.Path,
        default=pathlib.Path("~/nvme/obsm.h5ad"),
        help="Obsm-only .h5ad used to overwrite the loaded obsm keys by obs index (default: %(default)s)",
    )
    p.add_argument(
        "--clusters",
        type=str,
        default="6,7,9",
        help="Comma-separated leiden clusters to include (default: %(default)s)",
    )
    p.add_argument(
        "--manual-layer",
        type=str,
        default=None,
        help="Optional exact-match filter on obs['manual_layer'].",
    )
    p.add_argument(
        "--eomes-gt",
        type=float,
        default=None,
        help="Optional raw-count filter: keep only cells with Eomes > this threshold in layers['raw'].",
    )
    p.add_argument(
        "--bin-width-um",
        type=float,
        default=200.0,
        help="Bin width in microns (default: %(default)s)",
    )
    p.add_argument(
        "--outdir",
        type=pathlib.Path,
        default=pathlib.Path("scripts/_out/brdu_edu_bins"),
        help="Output directory for plots/CSVs (default: %(default)s)",
    )
    p.add_argument(
        "--plot-native-ts",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Render fitted Ts(AP,ML)=delta_t_hours / P(BrdU-only | BrdU+) on the native AP/ML manifold.",
    )
    p.add_argument(
        "--plot-native-tc",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Render fitted Tc(AP,ML)=Ts(AP,ML) / P(BrdU+) on the native AP/ML manifold.",
    )
    p.add_argument(
        "--plot-native-brdu-only-fraction",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Render fitted P(BrdU-only | BrdU+) on the native AP/ML manifold.",
    )
    p.add_argument(
        "--plot-native-edu-pos-fraction",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Render fitted P(EdU+) on the native AP/ML manifold.",
    )
    p.add_argument(
        "--plot-2d-edu-pos-fraction",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Write a direct 2D AP/ML plane for observed and fitted P(EdU+).",
    )
    p.add_argument(
        "--delta-t-hours",
        type=float,
        default=1.5,
        help="Pulse spacing Δt in hours for Ts = Δt / frac_brdu_only (default: %(default)s)",
    )
    p.add_argument(
        "--restrict-t-neomeso",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Restrict native rendering to neocortex+mesocortex supported t-ranges (default: %(default)s)",
    )
    p.add_argument(
        "--gray-context",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Show unsupported/native-context triangles in gray outside the colored Ts support (default: %(default)s)",
    )
    p.add_argument(
        "--native-elev-deg",
        type=float,
        default=DEFAULT_NATIVE_CAMERA_ELEV_DEG,
        help="Native view elevation in degrees.",
    )
    p.add_argument(
        "--native-azim-deg",
        type=float,
        default=DEFAULT_NATIVE_CAMERA_AZIM_DEG,
        help="Native view azimuth in degrees.",
    )
    p.add_argument(
        "--native-roll-deg",
        type=float,
        default=DEFAULT_NATIVE_CAMERA_ROLL_DEG,
        help="Native view roll in degrees.",
    )
    p.add_argument(
        "--native-proj-type",
        choices=("ortho", "persp"),
        default=DEFAULT_NATIVE_PROJ_TYPE,
        help="Projection type for native rendering (default: %(default)s)",
    )
    p.add_argument(
        "--native-focal-length",
        type=float,
        default=DEFAULT_NATIVE_FOCAL_LENGTH,
        help="Perspective focal length for native rendering (default: %(default)s)",
    )
    p.add_argument(
        "--native-latlon",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Overlay native manifold graticule lines (default: %(default)s)",
    )
    p.add_argument(
        "--native-graticule",
        choices=("apml", "param", "ijk"),
        default="ijk",
        help="Graticule mode passed to the native surface renderer (default: %(default)s)",
    )
    p.add_argument("--native-lat-stride", type=int, default=10)
    p.add_argument("--native-lon-stride", type=int, default=10)
    p.add_argument("--native-max-lat-lines", type=int, default=8)
    p.add_argument("--native-max-lon-lines", type=int, default=8)
    p.add_argument(
        "--refextract-outdir",
        type=pathlib.Path,
        default=pathlib.Path("ccf/out/refextract/midsurface_neocortex_mesocortex_allocortex_3d"),
        help="Refextract midsurface artifact directory for native AP/ML rendering.",
    )
    p.add_argument("--refextract-slice-i-min", type=int, default=161)
    p.add_argument("--refextract-slice-i-max", type=int, default=305)
    p.add_argument("--refextract-n-t", type=int, default=257)
    p.add_argument("--refextract-ref-t", type=float, default=0.5)
    p.add_argument("--refextract-band-frac", type=float, default=0.15)
    p.add_argument(
        "--refextract-res-ijk-um",
        type=float,
        nargs=3,
        default=(20.0, 20.0, 20.0),
        metavar=("RI", "RJ", "RK"),
        help="Voxel size override for native AP/ML projection context.",
    )
    return p.parse_args()


def _bins_for(values: np.ndarray, bin_width_um: float) -> tuple[np.ndarray, np.ndarray]:
    if values.size == 0:
        raise ValueError("No values to bin.")
    if not np.isfinite(bin_width_um) or bin_width_um <= 0:
        raise ValueError(f"--bin-width-um must be > 0, got {bin_width_um!r}")

    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    start = bin_width_um * np.floor(vmin / bin_width_um)
    # Ensure the last edge is strictly greater than vmax even when vmax is an exact multiple.
    end = bin_width_um * (np.floor(vmax / bin_width_um) + 1.0)
    edges = np.arange(start, end + bin_width_um, bin_width_um, dtype=float)
    centers = edges[:-1] + (bin_width_um / 2.0)
    return edges, centers


def _binned_counts(
    coord_um: np.ndarray,
    brdu_pos: np.ndarray,
    edu_pos: np.ndarray,
    bin_width_um: float,
    *,
    edges: np.ndarray | None = None,
) -> pd.DataFrame:
    coord_um = np.asarray(coord_um, dtype=float)
    brdu_pos = np.asarray(brdu_pos, dtype=bool)
    edu_pos = np.asarray(edu_pos, dtype=bool)
    if coord_um.shape != brdu_pos.shape or coord_um.shape != edu_pos.shape:
        raise ValueError(
            f"Shape mismatch: coord {coord_um.shape}, brdu_pos {brdu_pos.shape}, edu_pos {edu_pos.shape}"
        )

    finite = np.isfinite(coord_um)
    coord_um = coord_um[finite]
    brdu_pos = brdu_pos[finite]
    edu_pos = edu_pos[finite]

    if edges is None:
        edges, centers = _bins_for(coord_um, bin_width_um=bin_width_um)
    else:
        edges = np.asarray(edges, dtype=float)
        if edges.ndim != 1 or edges.size < 2:
            raise ValueError(f"edges must be 1D with >=2 entries; got shape {edges.shape}")
        if not np.all(np.isfinite(edges)):
            raise ValueError("edges contains non-finite values")
        if not np.all(np.diff(edges) > 0):
            raise ValueError("edges must be strictly increasing")
        centers = edges[:-1] + (np.diff(edges) / 2.0)
    bin_idx = np.digitize(coord_um, edges, right=False) - 1
    nbins = edges.size - 1
    in_range = (bin_idx >= 0) & (bin_idx < nbins)
    bin_idx = bin_idx[in_range]
    brdu_pos = brdu_pos[in_range]
    edu_pos = edu_pos[in_range]

    brdu_only = brdu_pos & (~edu_pos)
    edu_only = edu_pos & (~brdu_pos)
    dual = brdu_pos & edu_pos
    any_pos = brdu_pos | edu_pos

    total = np.bincount(bin_idx, minlength=nbins)
    n_brdu_only = np.bincount(bin_idx[brdu_only], minlength=nbins)
    n_edu_only = np.bincount(bin_idx[edu_only], minlength=nbins)
    n_dual = np.bincount(bin_idx[dual], minlength=nbins)
    n_any = np.bincount(bin_idx[any_pos], minlength=nbins)

    df = pd.DataFrame(
        {
            "bin_center_um": centers,
            "n_total": total,
            "n_any": n_any,
            "n_brdu_only": n_brdu_only,
            "n_edu_only": n_edu_only,
            "n_dual": n_dual,
        }
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        df["frac_total_brdu_only"] = df["n_brdu_only"] / df["n_total"]
        df["frac_total_edu_only"] = df["n_edu_only"] / df["n_total"]
        df["frac_total_dual"] = df["n_dual"] / df["n_total"]

        df["frac_any_brdu_only"] = df["n_brdu_only"] / df["n_any"]
        df["frac_any_edu_only"] = df["n_edu_only"] / df["n_any"]
        df["frac_any_dual"] = df["n_dual"] / df["n_any"]

    return df


def _binned_counts_2d(
    ap_um: np.ndarray,
    ml_um: np.ndarray,
    brdu_pos: np.ndarray,
    edu_pos: np.ndarray,
    *,
    bin_width_um: float,
) -> dict[str, np.ndarray | pd.DataFrame]:
    ap_um = np.asarray(ap_um, dtype=float)
    ml_um = np.asarray(ml_um, dtype=float)
    brdu_pos = np.asarray(brdu_pos, dtype=bool)
    edu_pos = np.asarray(edu_pos, dtype=bool)
    if not (ap_um.shape == ml_um.shape == brdu_pos.shape == edu_pos.shape):
        raise ValueError(
            f"Shape mismatch: ap={ap_um.shape}, ml={ml_um.shape}, brdu={brdu_pos.shape}, edu={edu_pos.shape}"
        )

    finite = np.isfinite(ap_um) & np.isfinite(ml_um)
    ap_um = ap_um[finite]
    ml_um = ml_um[finite]
    brdu_pos = brdu_pos[finite]
    edu_pos = edu_pos[finite]
    if ap_um.size == 0:
        raise ValueError("No finite AP/ML values for 2D binning.")

    ap_edges, ap_centers = _bins_for(ap_um, bin_width_um=bin_width_um)
    ml_edges, ml_centers = _bins_for(ml_um, bin_width_um=bin_width_um)
    ap_idx = np.digitize(ap_um, ap_edges, right=False) - 1
    ml_idx = np.digitize(ml_um, ml_edges, right=False) - 1

    n_ap = ap_edges.size - 1
    n_ml = ml_edges.size - 1
    in_range = (ap_idx >= 0) & (ap_idx < n_ap) & (ml_idx >= 0) & (ml_idx < n_ml)
    ap_idx = ap_idx[in_range]
    ml_idx = ml_idx[in_range]
    brdu_pos = brdu_pos[in_range]
    edu_pos = edu_pos[in_range]

    brdu_only = brdu_pos & (~edu_pos)
    edu_only = edu_pos & (~brdu_pos)
    dual = brdu_pos & edu_pos

    flat = (ap_idx * n_ml) + ml_idx
    n_total = np.bincount(flat, minlength=n_ap * n_ml).reshape(n_ap, n_ml)
    n_brdu_only = np.bincount(flat[brdu_only], minlength=n_ap * n_ml).reshape(n_ap, n_ml)
    n_edu_only = np.bincount(flat[edu_only], minlength=n_ap * n_ml).reshape(n_ap, n_ml)
    n_dual = np.bincount(flat[dual], minlength=n_ap * n_ml).reshape(n_ap, n_ml)

    ap_grid, ml_grid = np.meshgrid(ap_centers, ml_centers, indexing="ij")
    table = pd.DataFrame(
        {
            "ap_center_um": ap_grid.ravel(),
            "ml_center_um": ml_grid.ravel(),
            "n_total": n_total.ravel(),
            "n_brdu_only": n_brdu_only.ravel(),
            "n_edu_only": n_edu_only.ravel(),
            "n_dual": n_dual.ravel(),
        }
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        table["frac_brdu_only"] = table["n_brdu_only"] / table["n_total"]
        table["frac_edu_only"] = table["n_edu_only"] / table["n_total"]
        table["frac_dual"] = table["n_dual"] / table["n_total"]

    return {
        "ap_edges": ap_edges,
        "ml_edges": ml_edges,
        "ap_centers": ap_centers,
        "ml_centers": ml_centers,
        "n_total": n_total,
        "n_brdu_only": n_brdu_only,
        "n_edu_only": n_edu_only,
        "n_dual": n_dual,
        "table": table,
    }


def _fit_binomial_logit_2d(
    ap_um: np.ndarray,
    ml_um: np.ndarray,
    k: np.ndarray,
    n: np.ndarray,
    animal: np.ndarray | None = None,
    *,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit logit(p) = b0 + b_ap*AP + b_ml*ML with optional animal fixed effects.

    When `animal` is provided, the fit includes animal dummy covariates and then
    collapses them back into a pooled intercept using the observed animal mix.
    """
    ap_um = np.asarray(ap_um, dtype=float)
    ml_um = np.asarray(ml_um, dtype=float)
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)
    animal_arr = None if animal is None else np.asarray(animal, dtype=object)
    if animal_arr is None:
        if not (ap_um.shape == ml_um.shape == k.shape == n.shape):
            raise ValueError(f"Shape mismatch in 2D fit: {ap_um.shape}, {ml_um.shape}, {k.shape}, {n.shape}")
    elif not (ap_um.shape == ml_um.shape == k.shape == n.shape == animal_arr.shape):
        raise ValueError(
            "Shape mismatch in 2D fit: "
            f"{ap_um.shape}, {ml_um.shape}, {k.shape}, {n.shape}, {animal_arr.shape}"
        )
    if ap_um.ndim != 1:
        raise ValueError("2D fit inputs must be 1D arrays.")
    if ap_um.size < 3:
        raise ValueError("Need at least 3 bins for 2D regression.")

    ok = n > 0
    ap_um = ap_um[ok]
    ml_um = ml_um[ok]
    k = k[ok]
    n = n[ok]
    if animal_arr is not None:
        animal_arr = animal_arr[ok]
    if ap_um.size < 3:
        raise ValueError("Need at least 3 non-empty bins for 2D regression.")

    ap_mean = float(np.mean(ap_um))
    ml_mean = float(np.mean(ml_um))
    ap_std = float(np.std(ap_um))
    ml_std = float(np.std(ml_um))
    if ap_std <= 0.0 or ml_std <= 0.0:
        raise ValueError("AP and ML must have non-zero variance.")

    ap_s = (ap_um - ap_mean) / ap_std
    ml_s = (ml_um - ml_mean) / ml_std
    X = np.column_stack([np.ones_like(ap_s), ap_s, ml_s])
    n_base_cols = X.shape[1]
    z_mean = np.zeros(0, dtype=float)
    if animal_arr is not None:
        animal_codes, animal_levels = pd.factorize(pd.Series(animal_arr, copy=False), sort=True)
        if animal_levels.size > 1:
            X_animal = np.eye(animal_levels.size, dtype=float)[animal_codes][:, 1:]
            X = np.column_stack([X, X_animal])
            z_mean = np.mean(X_animal, axis=0, dtype=float)
    beta = np.zeros(X.shape[1], dtype=float)

    ridge = 1e-8
    for _ in range(max_iter):
        eta = X @ beta
        p = _expit(eta)
        w = n * p * (1.0 - p)
        xtwx = X.T @ (X * w[:, None]) + (ridge * np.eye(X.shape[1]))
        grad = X.T @ (k - (n * p))
        step = np.linalg.solve(xtwx, grad)
        beta_new = beta + step
        if np.max(np.abs(step)) < tol:
            beta = beta_new
            break
        beta = beta_new

    eta = X @ beta
    p = _expit(eta)
    w = n * p * (1.0 - p)
    xtwx = X.T @ (X * w[:, None]) + (ridge * np.eye(X.shape[1]))
    cov_full = np.linalg.inv(xtwx)

    marginalize = np.zeros((3, X.shape[1]), dtype=float)
    marginalize[0, 0] = 1.0
    marginalize[1, 1] = 1.0
    marginalize[2, 2] = 1.0
    if z_mean.size:
        marginalize[0, n_base_cols:] = z_mean

    transform = np.array(
        [
            [1.0, -ap_mean / ap_std, -ml_mean / ml_std],
            [0.0, 1.0 / ap_std, 0.0],
            [0.0, 0.0, 1.0 / ml_std],
        ],
        dtype=float,
    )
    total_transform = transform @ marginalize
    coef = total_transform @ beta
    cov = total_transform @ cov_full @ total_transform.T
    return coef, cov


def _plot_2d_plane_brdu_dual(
    plane: dict[str, np.ndarray | pd.DataFrame],
    coef: np.ndarray,
    cov: np.ndarray,
    *,
    title: str,
    out_png: pathlib.Path,
) -> None:
    matplotlib.use("Agg", force=True)
    from scipy.stats import norm

    ap_edges = np.asarray(plane["ap_edges"], dtype=float)
    ml_edges = np.asarray(plane["ml_edges"], dtype=float)
    ap_centers = np.asarray(plane["ap_centers"], dtype=float)
    ml_centers = np.asarray(plane["ml_centers"], dtype=float)
    n_total = np.asarray(plane["n_total"], dtype=float)
    n_brdu_only = np.asarray(plane["n_brdu_only"], dtype=float)

    obs = np.divide(n_brdu_only, n_total, out=np.full_like(n_total, np.nan), where=n_total > 0)
    ap_grid, ml_grid = np.meshgrid(ap_centers, ml_centers, indexing="ij")
    pred = _expit(coef[0] + (coef[1] * ap_grid) + (coef[2] * ml_grid))

    se_ap = float(np.sqrt(cov[1, 1])) if cov[1, 1] > 0 else float("nan")
    se_ml = float(np.sqrt(cov[2, 2])) if cov[2, 2] > 0 else float("nan")
    p_ap = float(2.0 * norm.sf(abs(coef[1] / se_ap))) if np.isfinite(se_ap) and se_ap > 0 else float("nan")
    p_ml = float(2.0 * norm.sf(abs(coef[2] / se_ml))) if np.isfinite(se_ml) and se_ml > 0 else float("nan")
    annotation_text = (
        f"logit(p)=b0+bAP*AP+bML*ML\n"
        f"bAP={coef[1]*1000:+.3f}/mm, p={p_ap:.2e}\n"
        f"bML={coef[2]*1000:+.3f}/mm, p={p_ml:.2e}"
    )

    _plot_2d_plane_fraction(
        ap_edges=ap_edges,
        ml_edges=ml_edges,
        obs=obs,
        pred=pred,
        title=title,
        out_png=out_png,
        obs_title="Observed BrdU-only fraction",
        pred_title="Fitted 2D logistic plane",
        cbar_label="BrdU-only fraction",
        cmap_name=SCALAR_CMAP,
        annotation_text=annotation_text,
    )


def _plot_2d_plane_fraction(
    *,
    ap_edges: np.ndarray,
    ml_edges: np.ndarray,
    obs: np.ndarray,
    pred: np.ndarray,
    title: str,
    out_png: pathlib.Path,
    obs_title: str,
    pred_title: str,
    cbar_label: str,
    cmap_name: str,
    annotation_text: str | None = None,
) -> None:
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(13, 5.5), sharex=True, sharey=True, constrained_layout=True)
    im0 = axes[0].pcolormesh(ml_edges, ap_edges, obs, shading="auto", vmin=0.0, vmax=1.0, cmap=cmap_name)
    im1 = axes[1].pcolormesh(ml_edges, ap_edges, pred, shading="auto", vmin=0.0, vmax=1.0, cmap=cmap_name)

    axes[0].set_title(obs_title)
    axes[1].set_title(pred_title)
    for ax in axes:
        ax.set_xlabel("ML (um)")
        ax.set_ylabel("AP (um)")
        ax.invert_xaxis()
        ax.invert_yaxis()

    fig.colorbar(im0, ax=axes[0], label=cbar_label)
    fig.colorbar(im1, ax=axes[1], label=cbar_label)
    if annotation_text is not None:
        axes[1].text(
            0.02,
            0.98,
            annotation_text,
            transform=axes[1].transAxes,
            va="top",
            ha="left",
            fontsize=9,
            color="white",
            bbox={"facecolor": "black", "alpha": 0.35, "edgecolor": "none", "pad": 3.0},
        )
    fig.suptitle(title)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def _plot_2d_plane_edu_pos_fraction(
    plane_all: dict[str, np.ndarray | pd.DataFrame],
    *,
    coef_edu_pos: np.ndarray,
    title: str,
    out_png: pathlib.Path,
) -> None:
    ap_edges = np.asarray(plane_all["ap_edges"], dtype=float)
    ml_edges = np.asarray(plane_all["ml_edges"], dtype=float)
    ap_centers = np.asarray(plane_all["ap_centers"], dtype=float)
    ml_centers = np.asarray(plane_all["ml_centers"], dtype=float)
    n_total = np.asarray(plane_all["n_total"], dtype=float)
    n_edu_only = np.asarray(plane_all["n_edu_only"], dtype=float)
    n_dual = np.asarray(plane_all["n_dual"], dtype=float)
    n_edu_pos = n_edu_only + n_dual

    obs = np.divide(n_edu_pos, n_total, out=np.full_like(n_total, np.nan), where=n_total > 0)
    ap_grid, ml_grid = np.meshgrid(ap_centers, ml_centers, indexing="ij")
    pred = _predict_edu_pos_fraction_2d(coef=coef_edu_pos, ap_um=ap_grid, ml_um=ml_grid)

    _plot_2d_plane_fraction(
        ap_edges=ap_edges,
        ml_edges=ml_edges,
        obs=obs,
        pred=pred,
        title=title,
        out_png=out_png,
        obs_title="Observed EdU+ fraction",
        pred_title="Fitted EdU+ plane",
        cbar_label="EdU+ fraction",
        cmap_name=SCALAR_CMAP,
    )


def _plot_2d_plane_tc_hours(
    plane_all: dict[str, np.ndarray | pd.DataFrame],
    *,
    coef_ts: np.ndarray,
    coef_dual: np.ndarray,
    delta_t_hours: float,
    title: str,
    out_png: pathlib.Path,
) -> None:
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    ap_edges = np.asarray(plane_all["ap_edges"], dtype=float)
    ml_edges = np.asarray(plane_all["ml_edges"], dtype=float)
    ap_centers = np.asarray(plane_all["ap_centers"], dtype=float)
    ml_centers = np.asarray(plane_all["ml_centers"], dtype=float)
    obs_tc = _tc_hours_obs_from_counts(
        n_total=np.asarray(plane_all["n_total"], dtype=float),
        n_brdu_only=np.asarray(plane_all["n_brdu_only"], dtype=float),
        n_edu_only=np.asarray(plane_all["n_edu_only"], dtype=float),
        n_dual=np.asarray(plane_all["n_dual"], dtype=float),
        delta_t_hours=float(delta_t_hours),
    )

    ap_grid, ml_grid = np.meshgrid(ap_centers, ml_centers, indexing="ij")
    pred_tc = _predict_tc_hours_2d(
        coef_ts=coef_ts,
        coef_dual=coef_dual,
        ap_um=ap_grid,
        ml_um=ml_grid,
        delta_t_hours=float(delta_t_hours),
    )

    finite = np.isfinite(obs_tc) | np.isfinite(pred_tc)
    if not np.any(finite):
        raise ValueError("No finite Tc values available for 2D plane plotting.")
    vmax = float(np.nanquantile(np.concatenate([obs_tc[finite], pred_tc[finite]]), 0.99))
    vmin = float(np.nanmin(np.concatenate([obs_tc[finite], pred_tc[finite]])))
    if not np.isfinite(vmax) or not np.isfinite(vmin) or vmax <= vmin:
        vmax = vmin + 1.0

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(13, 5.5), sharex=True, sharey=True, constrained_layout=True)
    im0 = axes[0].pcolormesh(ml_edges, ap_edges, obs_tc, shading="auto", vmin=vmin, vmax=vmax, cmap=SCALAR_CMAP)
    im1 = axes[1].pcolormesh(ml_edges, ap_edges, pred_tc, shading="auto", vmin=vmin, vmax=vmax, cmap=SCALAR_CMAP)

    axes[0].set_title("Observed Tc (hours)")
    axes[1].set_title("Fitted Tc plane (hours)")
    for ax in axes:
        ax.set_xlabel("ML (um)")
        ax.set_ylabel("AP (um)")
        ax.invert_xaxis()
        ax.invert_yaxis()

    fig.colorbar(im0, ax=axes[0], label="Tc (hours)")
    fig.colorbar(im1, ax=axes[1], label="Tc (hours)")
    fig.suptitle(title)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def _plot_2d_plane_ts_hours(
    plane_brdu: dict[str, np.ndarray | pd.DataFrame],
    *,
    coef_ts: np.ndarray,
    delta_t_hours: float,
    title: str,
    out_png: pathlib.Path,
) -> None:
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    ap_edges = np.asarray(plane_brdu["ap_edges"], dtype=float)
    ml_edges = np.asarray(plane_brdu["ml_edges"], dtype=float)
    ap_centers = np.asarray(plane_brdu["ap_centers"], dtype=float)
    ml_centers = np.asarray(plane_brdu["ml_centers"], dtype=float)
    n_brdu_only = np.asarray(plane_brdu["n_brdu_only"], dtype=float)
    n_dual = np.asarray(plane_brdu["n_dual"], dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        obs_ts = np.divide(
            float(delta_t_hours) * n_dual,
            n_brdu_only,
            out=np.full_like(n_brdu_only, np.nan),
            where=n_brdu_only > 0,
        )

    ap_grid, ml_grid = np.meshgrid(ap_centers, ml_centers, indexing="ij")
    pred_ts = _predict_ts_hours_2d(
        coef=coef_ts,
        ap_um=ap_grid,
        ml_um=ml_grid,
        delta_t_hours=float(delta_t_hours),
    )

    finite = np.isfinite(obs_ts) | np.isfinite(pred_ts)
    if not np.any(finite):
        raise ValueError("No finite Ts values available for 2D plane plotting.")
    vals = np.concatenate([obs_ts[finite], pred_ts[finite]])
    vmax = float(np.nanquantile(vals, 0.99))
    vmin = float(np.nanmin(vals))
    if not np.isfinite(vmax) or not np.isfinite(vmin) or vmax <= vmin:
        vmax = vmin + 1.0

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(13, 5.5), sharex=True, sharey=True, constrained_layout=True)
    im0 = axes[0].pcolormesh(ml_edges, ap_edges, obs_ts, shading="auto", vmin=vmin, vmax=vmax, cmap=TS_CMAP)
    im1 = axes[1].pcolormesh(ml_edges, ap_edges, pred_ts, shading="auto", vmin=vmin, vmax=vmax, cmap=TS_CMAP)

    axes[0].set_title("Observed Ts (hours)")
    axes[1].set_title("Fitted Ts plane (hours)")
    for ax in axes:
        ax.set_xlabel("ML (um)")
        ax.set_ylabel("AP (um)")
        ax.invert_xaxis()
        ax.invert_yaxis()

    fig.colorbar(im0, ax=axes[0], label="Ts (hours)")
    fig.colorbar(im1, ax=axes[1], label="Ts (hours)")
    fig.suptitle(title)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def _plot_frac_total(df: pd.DataFrame, *, axis_label: str, title: str, out_png: pathlib.Path) -> None:
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    ok = df["n_total"].to_numpy() > 0
    x = df.loc[ok, "bin_center_um"].to_numpy()

    fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    ax.plot(x, df.loc[ok, "frac_total_brdu_only"].to_numpy(), label="BrdU only")
    ax.plot(x, df.loc[ok, "frac_total_edu_only"].to_numpy(), label="EdU only")
    ax.plot(x, df.loc[ok, "frac_total_dual"].to_numpy(), label="Dual (BrdU+EdU)")
    ax.set_xlabel(axis_label)
    ax.set_ylabel("Proportion among all cells")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, ncol=3, loc="upper center")
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def _plot_tc_hours_1d(
    df: pd.DataFrame,
    *,
    axis_label: str,
    title: str,
    out_png: pathlib.Path,
) -> None:
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    ok = np.isfinite(df["tc_hours_obs"].to_numpy(dtype=float)) & np.isfinite(df["tc_hours_fit"].to_numpy(dtype=float))
    if not np.any(ok):
        raise ValueError("No finite Tc values available for 1D plotting.")

    x = df.loc[ok, "bin_center_um"].to_numpy(dtype=float)
    y_obs = df.loc[ok, "tc_hours_obs"].to_numpy(dtype=float)
    y_fit = df.loc[ok, "tc_hours_fit"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    obs_color = matplotlib.colormaps[SCALAR_CMAP](0.2)
    fit_color = matplotlib.colormaps[SCALAR_CMAP](0.8)
    ax.plot(x, y_obs, color=obs_color, linewidth=1.6, marker="o", markersize=3.5, label="Observed Tc")
    ax.plot(x, y_fit, color=fit_color, linewidth=2.2, label="Fitted Tc")
    ax.set_xlabel(axis_label)
    ax.set_ylabel("Tc (hours)")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def _plot_frac_total_by_dataset(
    dfs: dict[str, pd.DataFrame],
    *,
    axis_label: str,
    title: str,
    out_png: pathlib.Path,
) -> None:
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from scipy.stats import norm

    if not dfs:
        raise ValueError("No datasets to plot.")

    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(12.5, 7.5),
        sharex=True,
        constrained_layout=True,
    )

    ds_names = list(dfs.keys())
    cmap = plt.get_cmap("tab20")
    color_map = {ds: cmap(i % 20) for i, ds in enumerate(ds_names)}

    def combined_counts(dfs_in: dict[str, pd.DataFrame]) -> pd.DataFrame:
        cols = ["bin_center_um", "n_total", "n_brdu_only", "n_dual"]
        return (
            pd.concat([df.loc[:, cols] for df in dfs_in.values()], ignore_index=True)
            .groupby("bin_center_um", as_index=False)
            .sum(numeric_only=True)
            .sort_values("bin_center_um")
        )

    def fit_pooled_binomial(x: np.ndarray, k: np.ndarray, n: np.ndarray) -> tuple[float, float, float]:
        intercept, slope = _fit_binomial_logit(x, k, n)
        p = _expit(intercept + (slope * x))
        w = n * p * (1.0 - p)
        X = np.column_stack([np.ones_like(x), x])
        xtwx = X.T @ (X * w[:, None])
        cov = np.linalg.inv(xtwx)
        se_slope = float(np.sqrt(cov[1, 1]))
        if not np.isfinite(se_slope) or se_slope <= 0.0:
            return intercept, slope, float("nan")
        z = slope / se_slope
        pval = float(2.0 * norm.sf(abs(z)))
        return intercept, slope, pval

    def bootstrap_band(
        x_grid: np.ndarray,
        *,
        dfs_in: dict[str, pd.DataFrame],
        n_boot: int = 250,
        seed: int = 0,
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        rng = np.random.default_rng(seed)
        ds_list = list(dfs_in.keys())
        preds: list[np.ndarray] = []
        failures = 0
        shown = 0
        for i in range(n_boot):
            sample = rng.choice(ds_list, size=len(ds_list), replace=True)
            pooled_i = combined_counts({ds: dfs_in[ds] for ds in sample})
            ok_i = pooled_i["n_total"].to_numpy(dtype=float) > 0
            x_i = pooled_i.loc[ok_i, "bin_center_um"].to_numpy(dtype=float)
            n_i = pooled_i.loc[ok_i, "n_total"].to_numpy(dtype=float)
            k_i = pooled_i.loc[ok_i, "n_brdu_only"].to_numpy(dtype=float)
            try:
                intercept_i, slope_i, _p_i = fit_pooled_binomial(x_i, k_i, n_i)
            except (RuntimeError, ValueError, np.linalg.LinAlgError) as e:
                failures += 1
                if shown < 5:
                    print(f"warning: bootstrap binomial fit failed at iter {i}: {e}")
                    shown += 1
                continue
            preds.append(_expit(intercept_i + (slope_i * x_grid)))
        if not preds:
            raise RuntimeError("All bootstrap fits failed; cannot compute uncertainty band.")
        if failures:
            if failures > 5:
                print(f"warning: bootstrap failures={failures}/{n_boot} (showing first 5)")
            else:
                print(f"warning: bootstrap failures={failures}/{n_boot}")
        arr = np.stack(preds, axis=0)  # (boot, grid)
        lo = np.quantile(arr, 0.025, axis=0)
        hi = np.quantile(arr, 0.975, axis=0)
        delta = arr[:, -1] - arr[:, 0]
        delta_lo = float(np.quantile(delta, 0.025))
        delta_hi = float(np.quantile(delta, 0.975))
        return lo, hi, delta_lo, delta_hi

    panels = [
        ("frac_total_brdu_only", "n_brdu_only", "BrdU only"),
        ("frac_total_dual", "n_dual", "Dual (BrdU+EdU)"),
    ]

    pooled = combined_counts(dfs)
    ok_pool = pooled["n_total"].to_numpy(dtype=float) > 0
    x_pool = pooled.loc[ok_pool, "bin_center_um"].to_numpy(dtype=float)
    n_pool = pooled.loc[ok_pool, "n_total"].to_numpy(dtype=float)
    k_pool = pooled.loc[ok_pool, "n_brdu_only"].to_numpy(dtype=float)
    intercept_pool, slope_pool, pval_pool = fit_pooled_binomial(x_pool, k_pool, n_pool)
    x_grid = np.linspace(float(np.min(x_pool)), float(np.max(x_pool)), 250)
    p_brdu_grid = _expit(intercept_pool + (slope_pool * x_grid))
    p_dual_grid = 1.0 - p_brdu_grid
    lo_grid, hi_grid, delta_lo, delta_hi = bootstrap_band(x_grid, dfs_in=dfs)

    # Predicted change across the full x-range (min->max) from pooled binomial fit.
    delta_fit_brdu = float(p_brdu_grid[-1] - p_brdu_grid[0])
    delta_fit_dual = -delta_fit_brdu
    x0 = float(x_grid[0])
    x1 = float(x_grid[-1])
    dx_um = x1 - x0
    print(
        f"{title}: pooled delta across range {x0:.1f}->{x1:.1f} um (d={dx_um:.1f} um): "
        f"brdu_only={delta_fit_brdu:+.4f} dual={delta_fit_dual:+.4f}"
    )
    print(
        f"{title}: bootstrap 95% delta CI: "
        f"brdu_only=[{delta_lo:+.4f},{delta_hi:+.4f}] "
        f"dual=[{-delta_hi:+.4f},{-delta_lo:+.4f}]"
    )

    for ax, (ycol, kcol, ylabel) in zip(axes, panels, strict=True):
        for ds, df in dfs.items():
            ok = df["n_total"].to_numpy() > 0
            x = df.loc[ok, "bin_center_um"].to_numpy()
            y = df.loc[ok, ycol].to_numpy()
            color = color_map[ds]

            if x.size == 0:
                continue
            ax.plot(x, y, label=ds, linewidth=1.0, alpha=0.8, color=color, zorder=1)

        if kcol == "n_brdu_only":
            band_lo = lo_grid
            band_hi = hi_grid
            y_fit = p_brdu_grid
            slope_dx = slope_pool
        else:
            band_lo = 1.0 - hi_grid
            band_hi = 1.0 - lo_grid
            y_fit = p_dual_grid
            slope_dx = -slope_pool

        ax.fill_between(x_grid, band_lo, band_hi, color="black", alpha=0.15, linewidth=0.0, zorder=3)
        ax.plot(x_grid, y_fit, color="black", linewidth=2.2, alpha=0.95, zorder=4, label="Pooled binomial fit")

        slope_per_mm = slope_dx * 1000.0
        p_txt = f"p={pval_pool:.2e}" if np.isfinite(pval_pool) else "p=nan"
        ax.text(
            0.02,
            0.92,
            f"slope@mean={slope_per_mm:+.3f}/mm, {p_txt}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            color="black",
        )
        ax.set_ylabel(ylabel)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.25)

    axes[-1].set_xlabel(axis_label)
    fig.suptitle(title)
    # Put legend outside; lots of datasets.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=False,
        fontsize=8,
    )
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def _predict_brdu_only_fraction_2d(*, coef: np.ndarray, ap_um: np.ndarray, ml_um: np.ndarray) -> np.ndarray:
    ap = np.asarray(ap_um, dtype=float)
    ml = np.asarray(ml_um, dtype=float)
    if ap.shape != ml.shape:
        raise ValueError(f"AP and ML shapes must match, got {ap.shape} and {ml.shape}")
    coef_arr = np.asarray(coef, dtype=float).reshape(-1)
    if coef_arr.shape != (3,):
        raise ValueError(f"coef must have shape (3,), got {coef_arr.shape}")
    return _expit(coef_arr[0] + (coef_arr[1] * ap) + (coef_arr[2] * ml))


def _predict_edu_pos_fraction_2d(*, coef: np.ndarray, ap_um: np.ndarray, ml_um: np.ndarray) -> np.ndarray:
    return _predict_brdu_only_fraction_2d(coef=coef, ap_um=ap_um, ml_um=ml_um)


def _predict_dual_fraction_2d(*, coef: np.ndarray, ap_um: np.ndarray, ml_um: np.ndarray) -> np.ndarray:
    return _predict_brdu_only_fraction_2d(coef=coef, ap_um=ap_um, ml_um=ml_um)


def _predict_brdu_only_fraction_1d(*, intercept: float, slope: float, coord_um: np.ndarray) -> np.ndarray:
    coord = np.asarray(coord_um, dtype=float)
    return _expit(float(intercept) + (float(slope) * coord))


def _predict_edu_pos_fraction_1d(*, intercept: float, slope: float, coord_um: np.ndarray) -> np.ndarray:
    return _predict_brdu_only_fraction_1d(intercept=intercept, slope=slope, coord_um=coord_um)


def _predict_dual_fraction_1d(*, intercept: float, slope: float, coord_um: np.ndarray) -> np.ndarray:
    return _predict_brdu_only_fraction_1d(intercept=intercept, slope=slope, coord_um=coord_um)


def _tc_hours_obs_from_counts(
    *,
    n_total: np.ndarray,
    n_brdu_only: np.ndarray,
    n_edu_only: np.ndarray,
    n_dual: np.ndarray,
    delta_t_hours: float,
) -> np.ndarray:
    n_total_arr = np.asarray(n_total, dtype=float)
    n_brdu_only_arr = np.asarray(n_brdu_only, dtype=float)
    n_dual_arr = np.asarray(n_dual, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ts_obs = np.divide(
            float(delta_t_hours) * n_dual_arr,
            n_brdu_only_arr,
            out=np.full_like(n_total_arr, np.nan),
            where=n_brdu_only_arr > 0,
        )
        frac_dual = np.divide(
            n_dual_arr,
            n_total_arr,
            out=np.full_like(n_total_arr, np.nan),
            where=n_total_arr > 0,
        )
        return np.divide(
            ts_obs,
            frac_dual,
            out=np.full_like(n_total_arr, np.nan),
            where=np.isfinite(frac_dual) & (frac_dual > 0),
        )


def _predict_ts_hours_2d(*, coef: np.ndarray, ap_um: np.ndarray, ml_um: np.ndarray, delta_t_hours: float) -> np.ndarray:
    if not np.isfinite(delta_t_hours) or float(delta_t_hours) <= 0.0:
        raise ValueError(f"delta_t_hours must be > 0, got {delta_t_hours!r}")
    frac_brdu_only = _predict_brdu_only_fraction_2d(coef=coef, ap_um=ap_um, ml_um=ml_um)
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(delta_t_hours) * ((1.0 - frac_brdu_only) / frac_brdu_only)


def _predict_ts_hours_1d(*, intercept: float, slope: float, coord_um: np.ndarray, delta_t_hours: float) -> np.ndarray:
    if not np.isfinite(delta_t_hours) or float(delta_t_hours) <= 0.0:
        raise ValueError(f"delta_t_hours must be > 0, got {delta_t_hours!r}")
    frac_brdu_only = _predict_brdu_only_fraction_1d(intercept=intercept, slope=slope, coord_um=coord_um)
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(delta_t_hours) * ((1.0 - frac_brdu_only) / frac_brdu_only)


def _predict_tc_hours_2d(
    *,
    coef_ts: np.ndarray,
    coef_dual: np.ndarray,
    ap_um: np.ndarray,
    ml_um: np.ndarray,
    delta_t_hours: float,
) -> np.ndarray:
    ts_hours = _predict_ts_hours_2d(coef=coef_ts, ap_um=ap_um, ml_um=ml_um, delta_t_hours=delta_t_hours)
    frac_dual = _predict_dual_fraction_2d(coef=coef_dual, ap_um=ap_um, ml_um=ml_um)
    with np.errstate(divide="ignore", invalid="ignore"):
        return ts_hours / frac_dual


def _predict_tc_hours_1d(
    *,
    intercept_ts: float,
    slope_ts: float,
    intercept_dual: float,
    slope_dual: float,
    coord_um: np.ndarray,
    delta_t_hours: float,
) -> np.ndarray:
    ts_hours = _predict_ts_hours_1d(
        intercept=intercept_ts,
        slope=slope_ts,
        coord_um=coord_um,
        delta_t_hours=delta_t_hours,
    )
    frac_dual = _predict_dual_fraction_1d(
        intercept=intercept_dual,
        slope=slope_dual,
        coord_um=coord_um,
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        return ts_hours / frac_dual


def _percentile_support_bounds(values: np.ndarray, *, lo: float = 2.5, hi: float = 97.5) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    finite = np.isfinite(arr)
    if not np.any(finite):
        raise ValueError("Cannot compute percentile support bounds from all-nonfinite values.")
    return tuple(float(x) for x in np.nanpercentile(arr[finite], [lo, hi]))


def _build_native_tri_alpha(
    *,
    ordered_geom: dict[str, np.ndarray],
    ap_um_flat: np.ndarray,
    ml_um_flat: np.ndarray,
    ap_support_bounds_um: tuple[float, float] | None,
    ml_support_bounds_um: tuple[float, float] | None,
) -> np.ndarray | None:
    if ap_support_bounds_um is None or ml_support_bounds_um is None:
        return None
    tris = np.asarray(ordered_geom["tris"], dtype=np.int32)
    cent_ap = np.mean(np.asarray(ap_um_flat, dtype=np.float64)[tris], axis=1)
    cent_ml = np.mean(np.asarray(ml_um_flat, dtype=np.float64)[tris], axis=1)
    inside = (
        (cent_ap >= float(ap_support_bounds_um[0]))
        & (cent_ap <= float(ap_support_bounds_um[1]))
        & (cent_ml >= float(ml_support_bounds_um[0]))
        & (cent_ml <= float(ml_support_bounds_um[1]))
    )
    return inside.astype(np.float64, copy=False)


def _plot_native_scalar_surface(
    *,
    context: dict[str, object],
    values_flat: np.ndarray,
    out_png: pathlib.Path,
    restrict_t_neomeso: bool,
    gray_context: bool,
    native_latlon: bool,
    native_graticule: str,
    native_lat_stride: int,
    native_lon_stride: int,
    native_max_lat_lines: int,
    native_max_lon_lines: int,
    native_elev_deg: float,
    native_azim_deg: float,
    native_roll_deg: float,
    native_proj_type: str,
    native_focal_length: float,
    ap_support_bounds_um: tuple[float, float] | None,
    ml_support_bounds_um: tuple[float, float] | None,
    title: str,
    cbar_label: str,
    cmap_name: str,
) -> None:
    values = np.asarray(values_flat, dtype=np.float64)
    support_flat = np.asarray(context["support_flat"], dtype=bool)
    if values.shape != support_flat.shape:
        raise ValueError(f"values_flat shape {values.shape} does not match native surface shape {support_flat.shape}.")
    finite_support = support_flat & np.isfinite(values)
    if restrict_t_neomeso:
        finite_support &= np.asarray(context["neomeso_flat"], dtype=bool)
    if not np.any(finite_support):
        raise ValueError(f"No finite {cbar_label} values available on the requested native manifold support.")
    ordered_geom = (
        context["ordered_geom_neomeso"]
        if bool(restrict_t_neomeso) and not bool(gray_context)
        else context["ordered_geom_support"]
    )
    tri_alpha = _build_native_tri_alpha(
        ordered_geom=ordered_geom,
        ap_um_flat=np.asarray(context["ap_um_flat"], dtype=np.float64),
        ml_um_flat=np.asarray(context["ml_um_flat"], dtype=np.float64),
        ap_support_bounds_um=ap_support_bounds_um,
        ml_support_bounds_um=ml_support_bounds_um,
    )

    plot_coronal_surface_projection(
        values,
        x2d=np.asarray(context["x2d"], dtype=np.float64),
        y2d=np.asarray(context["y2d"], dtype=np.float64),
        z2d=np.asarray(context["z2d"], dtype=np.float64),
        x3d=np.asarray(context["x3"], dtype=np.float64),
        y3d=np.asarray(context["y3"], dtype=np.float64),
        z3d=np.asarray(context["z3"], dtype=np.float64),
        faces=np.asarray(context["faces"], dtype=np.int32),
        tri_support=np.asarray(context["tri_support"], dtype=bool),
        tri_neomeso=np.asarray(context["tri_neomeso"], dtype=bool),
        restrict_t_neomeso=bool(restrict_t_neomeso),
        gray_context=bool(gray_context),
        latlon=bool(native_latlon),
        graticule=str(native_graticule),
        lat_stride=int(native_lat_stride),
        lon_stride=int(native_lon_stride),
        max_lat_lines=int(native_max_lat_lines),
        max_lon_lines=int(native_max_lon_lines),
        vertex_support=np.asarray(context["support_flat"], dtype=bool),
        vertex_neomeso=np.asarray(context["neomeso_flat"], dtype=bool),
        vertex_ap_um=np.asarray(context["ap_um_flat"], dtype=np.float64),
        vertex_ml_um=np.asarray(context["ml_um_flat"], dtype=np.float64),
        n_rows=int(context["n_rows"]),
        n_cols=int(context["n_cols"]),
        shade=False,
        shade_strength=0.75,
        shade_elev_deg=float(native_elev_deg),
        shade_azim_deg=float(native_azim_deg),
        camera_elev_deg=float(native_elev_deg),
        camera_azim_deg=float(native_azim_deg),
        camera_roll_deg=float(native_roll_deg),
        proj_type=str(native_proj_type),
        focal_length=float(native_focal_length),
        ordered_geometry=ordered_geom,
        neomeso_start_fit=None
        if context["neomeso_start_fit"] is None
        else np.asarray(context["neomeso_start_fit"], dtype=np.float64),
        neomeso_end_fit=None
        if context["neomeso_end_fit"] is None
        else np.asarray(context["neomeso_end_fit"], dtype=np.float64),
        tri_alpha=tri_alpha,
        out_png=out_png,
        title=title,
        cmap=matplotlib.colormaps[cmap_name],
        cbar_label=cbar_label,
        cbar_ticks=None,
        cbar_ticklabels=None,
        vmin=None,
        vmax=None,
    )


def _plot_native_ts_surface(
    *,
    coef: np.ndarray,
    delta_t_hours: float,
    out_png: pathlib.Path,
    refextract_outdir: pathlib.Path,
    refextract_slice_i_min: int,
    refextract_slice_i_max: int,
    refextract_n_t: int,
    refextract_ref_t: float,
    refextract_band_frac: float,
    refextract_res_ijk_um: tuple[float, float, float],
    native_elev_deg: float,
    native_azim_deg: float,
    native_roll_deg: float,
    native_latlon: bool,
    native_graticule: str,
    native_lat_stride: int,
    native_lon_stride: int,
    native_max_lat_lines: int,
    native_max_lon_lines: int,
    native_proj_type: str,
    native_focal_length: float,
    ap_support_bounds_um: tuple[float, float] | None,
    ml_support_bounds_um: tuple[float, float] | None,
    restrict_t_neomeso: bool,
    gray_context: bool,
    title: str,
) -> None:
    context = build_apml_native_surface_projection_context(
        outdir=refextract_outdir,
        slice_i_min=int(refextract_slice_i_min),
        slice_i_max=int(refextract_slice_i_max),
        n_t=int(refextract_n_t),
        ref_t=float(refextract_ref_t),
        band_frac=float(refextract_band_frac),
        res_ijk_um=tuple(float(x) for x in refextract_res_ijk_um),
        elev_deg=float(native_elev_deg),
        azim_deg=float(native_azim_deg),
        roll_deg=float(native_roll_deg),
    )
    ts_hours = _predict_ts_hours_2d(
        coef=coef,
        ap_um=np.asarray(context["ap_um_flat"], dtype=np.float64),
        ml_um=np.asarray(context["ml_um_flat"], dtype=np.float64),
        delta_t_hours=float(delta_t_hours),
    )
    _plot_native_scalar_surface(
        context=context,
        values_flat=ts_hours,
        out_png=out_png,
        restrict_t_neomeso=restrict_t_neomeso,
        gray_context=gray_context,
        native_latlon=native_latlon,
        native_graticule=native_graticule,
        native_lat_stride=native_lat_stride,
        native_lon_stride=native_lon_stride,
        native_max_lat_lines=native_max_lat_lines,
        native_max_lon_lines=native_max_lon_lines,
        native_elev_deg=native_elev_deg,
        native_azim_deg=native_azim_deg,
        native_roll_deg=native_roll_deg,
        native_proj_type=native_proj_type,
        native_focal_length=native_focal_length,
        ap_support_bounds_um=ap_support_bounds_um,
        ml_support_bounds_um=ml_support_bounds_um,
        title=title,
        cbar_label="Ts (hours)",
        cmap_name=TS_CMAP,
    )


def _plot_native_brdu_only_fraction_surface(
    *,
    coef: np.ndarray,
    out_png: pathlib.Path,
    refextract_outdir: pathlib.Path,
    refextract_slice_i_min: int,
    refextract_slice_i_max: int,
    refextract_n_t: int,
    refextract_ref_t: float,
    refextract_band_frac: float,
    refextract_res_ijk_um: tuple[float, float, float],
    native_elev_deg: float,
    native_azim_deg: float,
    native_roll_deg: float,
    native_latlon: bool,
    native_graticule: str,
    native_lat_stride: int,
    native_lon_stride: int,
    native_max_lat_lines: int,
    native_max_lon_lines: int,
    native_proj_type: str,
    native_focal_length: float,
    ap_support_bounds_um: tuple[float, float] | None,
    ml_support_bounds_um: tuple[float, float] | None,
    restrict_t_neomeso: bool,
    gray_context: bool,
    title: str,
) -> None:
    context = build_apml_native_surface_projection_context(
        outdir=refextract_outdir,
        slice_i_min=int(refextract_slice_i_min),
        slice_i_max=int(refextract_slice_i_max),
        n_t=int(refextract_n_t),
        ref_t=float(refextract_ref_t),
        band_frac=float(refextract_band_frac),
        res_ijk_um=tuple(float(x) for x in refextract_res_ijk_um),
        elev_deg=float(native_elev_deg),
        azim_deg=float(native_azim_deg),
        roll_deg=float(native_roll_deg),
    )
    frac_brdu_only = _predict_brdu_only_fraction_2d(
        coef=coef,
        ap_um=np.asarray(context["ap_um_flat"], dtype=np.float64),
        ml_um=np.asarray(context["ml_um_flat"], dtype=np.float64),
    )
    _plot_native_scalar_surface(
        context=context,
        values_flat=frac_brdu_only,
        out_png=out_png,
        restrict_t_neomeso=restrict_t_neomeso,
        gray_context=gray_context,
        native_latlon=native_latlon,
        native_graticule=native_graticule,
        native_lat_stride=native_lat_stride,
        native_lon_stride=native_lon_stride,
        native_max_lat_lines=native_max_lat_lines,
        native_max_lon_lines=native_max_lon_lines,
        native_elev_deg=native_elev_deg,
        native_azim_deg=native_azim_deg,
        native_roll_deg=native_roll_deg,
        native_proj_type=native_proj_type,
        native_focal_length=native_focal_length,
        ap_support_bounds_um=ap_support_bounds_um,
        ml_support_bounds_um=ml_support_bounds_um,
        title=title,
        cbar_label="BrdU-only fraction",
        cmap_name=SCALAR_CMAP,
    )


def _plot_native_tc_surface(
    *,
    coef_ts: np.ndarray,
    coef_dual: np.ndarray,
    delta_t_hours: float,
    out_png: pathlib.Path,
    refextract_outdir: pathlib.Path,
    refextract_slice_i_min: int,
    refextract_slice_i_max: int,
    refextract_n_t: int,
    refextract_ref_t: float,
    refextract_band_frac: float,
    refextract_res_ijk_um: tuple[float, float, float],
    native_elev_deg: float,
    native_azim_deg: float,
    native_roll_deg: float,
    native_latlon: bool,
    native_graticule: str,
    native_lat_stride: int,
    native_lon_stride: int,
    native_max_lat_lines: int,
    native_max_lon_lines: int,
    native_proj_type: str,
    native_focal_length: float,
    ap_support_bounds_um: tuple[float, float] | None,
    ml_support_bounds_um: tuple[float, float] | None,
    restrict_t_neomeso: bool,
    gray_context: bool,
    title: str,
) -> None:
    context = build_apml_native_surface_projection_context(
        outdir=refextract_outdir,
        slice_i_min=int(refextract_slice_i_min),
        slice_i_max=int(refextract_slice_i_max),
        n_t=int(refextract_n_t),
        ref_t=float(refextract_ref_t),
        band_frac=float(refextract_band_frac),
        res_ijk_um=tuple(float(x) for x in refextract_res_ijk_um),
        elev_deg=float(native_elev_deg),
        azim_deg=float(native_azim_deg),
        roll_deg=float(native_roll_deg),
    )
    tc_hours = _predict_tc_hours_2d(
        coef_ts=coef_ts,
        coef_dual=coef_dual,
        ap_um=np.asarray(context["ap_um_flat"], dtype=np.float64),
        ml_um=np.asarray(context["ml_um_flat"], dtype=np.float64),
        delta_t_hours=float(delta_t_hours),
    )
    _plot_native_scalar_surface(
        context=context,
        values_flat=tc_hours,
        out_png=out_png,
        restrict_t_neomeso=restrict_t_neomeso,
        gray_context=gray_context,
        native_latlon=native_latlon,
        native_graticule=native_graticule,
        native_lat_stride=native_lat_stride,
        native_lon_stride=native_lon_stride,
        native_max_lat_lines=native_max_lat_lines,
        native_max_lon_lines=native_max_lon_lines,
        native_elev_deg=native_elev_deg,
        native_azim_deg=native_azim_deg,
        native_roll_deg=native_roll_deg,
        native_proj_type=native_proj_type,
        native_focal_length=native_focal_length,
        ap_support_bounds_um=ap_support_bounds_um,
        ml_support_bounds_um=ml_support_bounds_um,
        title=title,
        cbar_label="Tc (hours)",
        cmap_name=SCALAR_CMAP,
    )


def _plot_native_edu_pos_fraction_surface(
    *,
    coef_edu_pos: np.ndarray,
    out_png: pathlib.Path,
    refextract_outdir: pathlib.Path,
    refextract_slice_i_min: int,
    refextract_slice_i_max: int,
    refextract_n_t: int,
    refextract_ref_t: float,
    refextract_band_frac: float,
    refextract_res_ijk_um: tuple[float, float, float],
    native_elev_deg: float,
    native_azim_deg: float,
    native_roll_deg: float,
    native_latlon: bool,
    native_graticule: str,
    native_lat_stride: int,
    native_lon_stride: int,
    native_max_lat_lines: int,
    native_max_lon_lines: int,
    native_proj_type: str,
    native_focal_length: float,
    ap_support_bounds_um: tuple[float, float] | None,
    ml_support_bounds_um: tuple[float, float] | None,
    restrict_t_neomeso: bool,
    gray_context: bool,
    title: str,
) -> None:
    context = build_apml_native_surface_projection_context(
        outdir=refextract_outdir,
        slice_i_min=int(refextract_slice_i_min),
        slice_i_max=int(refextract_slice_i_max),
        n_t=int(refextract_n_t),
        ref_t=float(refextract_ref_t),
        band_frac=float(refextract_band_frac),
        res_ijk_um=tuple(float(x) for x in refextract_res_ijk_um),
        elev_deg=float(native_elev_deg),
        azim_deg=float(native_azim_deg),
        roll_deg=float(native_roll_deg),
    )
    frac_edu_pos = _predict_edu_pos_fraction_2d(
        coef=coef_edu_pos,
        ap_um=np.asarray(context["ap_um_flat"], dtype=np.float64),
        ml_um=np.asarray(context["ml_um_flat"], dtype=np.float64),
    )
    _plot_native_scalar_surface(
        context=context,
        values_flat=frac_edu_pos,
        out_png=out_png,
        restrict_t_neomeso=restrict_t_neomeso,
        gray_context=gray_context,
        native_latlon=native_latlon,
        native_graticule=native_graticule,
        native_lat_stride=native_lat_stride,
        native_lon_stride=native_lon_stride,
        native_max_lat_lines=native_max_lat_lines,
        native_max_lon_lines=native_max_lon_lines,
        native_elev_deg=native_elev_deg,
        native_azim_deg=native_azim_deg,
        native_roll_deg=native_roll_deg,
        native_proj_type=native_proj_type,
        native_focal_length=native_focal_length,
        ap_support_bounds_um=ap_support_bounds_um,
        ml_support_bounds_um=ml_support_bounds_um,
        title=title,
        cbar_label="EdU+ fraction",
        cmap_name=SCALAR_CMAP,
    )


def _dataset_orientation(dataset: str) -> str:
    s = str(dataset).lower()
    if "sag" in s:
        return "sag"
    if "coro" in s:
        return "coro"
    raise ValueError(f"Cannot infer dataset orientation from dataset={dataset!r} (expected contains 'Sag' or 'Coro').")


def _filename_filter_tag(*, manual_layer: str | None, eomes_gt: float | None) -> str:
    parts: list[str] = []
    if manual_layer is not None:
        parts.append(f"manual_layer_{manual_layer}")
    if eomes_gt is not None:
        eomes_tag = int(eomes_gt) if float(eomes_gt).is_integer() else eomes_gt
        parts.append(f"Eomes_gt_{eomes_tag}")
    return "__".join(parts) if parts else "all_cells"


def main() -> None:
    args = _parse_args()
    h5ad_path = args.h5ad.expanduser()
    obsm_h5ad_path = args.obsm_h5ad.expanduser()
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    clusters = [c.strip() for c in str(args.clusters).split(",") if c.strip()]
    if not clusters:
        raise ValueError("--clusters parsed to empty list")
    manual_layer = None if args.manual_layer is None else str(args.manual_layer)
    eomes_gt = None if args.eomes_gt is None else float(args.eomes_gt)

    adata = _load_adata_with_external_obsm(h5ad_path=h5ad_path, obsm_h5ad_path=obsm_h5ad_path)
    if "leiden" not in adata.obs.columns:
        raise KeyError("obs['leiden'] not found")
    if "brdu_pos" not in adata.obs.columns or "edu_pos" not in adata.obs.columns:
        raise KeyError("obs['brdu_pos'] and/or obs['edu_pos'] not found")
    if "AP_ML_um" not in adata.obsm.keys():
        raise KeyError("obsm['AP_ML_um'] not found")

    mask = adata.obs["leiden"].astype(str).isin(clusters).to_numpy()
    if manual_layer is not None:
        if "manual_layer" not in adata.obs.columns:
            raise KeyError("obs['manual_layer'] not found")
        mask &= adata.obs["manual_layer"].astype(str).to_numpy() == manual_layer
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        raise ValueError(f"No cells matched leiden clusters {clusters!r} and manual_layer={manual_layer!r}")
    n_selected_before_expr = int(idx.size)
    if eomes_gt is not None:
        eomes = _read_gene_layer_values(adata, "Eomes", layer="raw", obs_idx=idx)
        idx = idx[eomes > eomes_gt]
        if idx.size == 0:
            raise ValueError(
                f"No cells matched leiden clusters {clusters!r}, manual_layer={manual_layer!r}, and Eomes>{eomes_gt:g}."
            )
    n_selected = int(idx.size)

    coords = np.asarray(adata.obsm["AP_ML_um"][idx, :], dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"Expected obsm['AP_ML_um'] to be (n,2); got {coords.shape}")

    brdu_pos = adata.obs["brdu_pos"].to_numpy(dtype=bool)[idx]
    edu_pos = adata.obs["edu_pos"].to_numpy(dtype=bool)[idx]
    dataset = adata.obs["dataset"].astype(str).to_numpy()[idx] if "dataset" in adata.obs.columns else None
    if dataset is None:
        raise KeyError("obs['dataset'] not found")

    finite_both = np.isfinite(coords).all(axis=1)
    coords = coords[finite_both]
    brdu_pos = brdu_pos[finite_both]
    edu_pos = edu_pos[finite_both]
    dataset = dataset[finite_both]
    n_finite = int(coords.shape[0])

    coords_all = coords.copy()
    brdu_pos_all = brdu_pos.copy()
    edu_pos_all = edu_pos.copy()
    dataset_all = dataset.copy()
    animal_all = np.array([_dataset_animal(d) for d in dataset_all], dtype=object)

    brdu_subset = brdu_pos
    coords = coords[brdu_subset]
    brdu_pos = brdu_pos[brdu_subset]
    edu_pos = edu_pos[brdu_subset]
    dataset = dataset[brdu_subset]
    animal = animal_all[brdu_subset]
    if coords.shape[0] == 0:
        raise ValueError("No BrdU or EdU positive cells after filtering.")

    ap_col, ml_col = 0, 1
    ap_support_bounds_um = _percentile_support_bounds(coords_all[:, ap_col])
    ml_support_bounds_um = _percentile_support_bounds(coords_all[:, ml_col])

    filter_desc_parts = [f"leiden {clusters}"]
    if manual_layer is not None:
        filter_desc_parts.append(f"manual_layer={manual_layer}")
    if eomes_gt is not None:
        filter_desc_parts.append(f"Eomes>{eomes_gt:g}")
    filter_tag = _filename_filter_tag(manual_layer=manual_layer, eomes_gt=eomes_gt)
    filter_desc = ", ".join(filter_desc_parts)
    bw = int(float(args.bin_width_um)) if float(args.bin_width_um).is_integer() else float(args.bin_width_um)

    n_total = int(coords.shape[0])

    # Per-dataset plots: only plot AP for sagittal, ML for coronal.
    orientations = np.array([_dataset_orientation(d) for d in dataset], dtype=object)
    sag_mask = orientations == "sag"
    coro_mask = orientations == "coro"
    orientations_all = np.array([_dataset_orientation(d) for d in dataset_all], dtype=object)
    sag_mask_all = orientations_all == "sag"
    coro_mask_all = orientations_all == "coro"

    edges_sag_ap = None
    if int(sag_mask.sum()) > 0:
        edges_sag_ap, _ = _bins_for(coords[sag_mask, ap_col], bin_width_um=float(args.bin_width_um))
    edges_coro_ml = None
    if int(coro_mask.sum()) > 0:
        edges_coro_ml, _ = _bins_for(coords[coro_mask, ml_col], bin_width_um=float(args.bin_width_um))

    # Overall (orientation-filtered) summaries.
    ds_outdir = outdir / "by_dataset"
    ds_outdir.mkdir(parents=True, exist_ok=True)

    sag_ap_dfs: dict[str, pd.DataFrame] = {}
    coro_ml_dfs: dict[str, pd.DataFrame] = {}

    for ds in pd.unique(dataset):
        ds_mask = dataset == ds
        ori = _dataset_orientation(ds)
        safe_ds = "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in ds)
        if ori == "sag":
            if edges_sag_ap is None:
                continue
            df = _binned_counts(
                coords[ds_mask, ap_col],
                brdu_pos[ds_mask],
                edu_pos[ds_mask],
                bin_width_um=float(args.bin_width_um),
                edges=edges_sag_ap,
            )
            csv_path = ds_outdir / f"brdu_edu_dual_by_AP_um_leiden_{filter_tag}_bin{bw}__{safe_ds}.csv"
            df.to_csv(csv_path, index=False)
            sag_ap_dfs[ds] = df
        else:
            if edges_coro_ml is None:
                continue
            df = _binned_counts(
                coords[ds_mask, ml_col],
                brdu_pos[ds_mask],
                edu_pos[ds_mask],
                bin_width_um=float(args.bin_width_um),
                edges=edges_coro_ml,
            )
            csv_path = ds_outdir / f"brdu_edu_dual_by_ML_um_leiden_{filter_tag}_bin{bw}__{safe_ds}.csv"
            df.to_csv(csv_path, index=False)
            coro_ml_dfs[ds] = df

    print(f"leiden clusters: {clusters}")
    if manual_layer is not None:
        print(f"manual_layer: {manual_layer}")
    if eomes_gt is not None:
        print(f"Eomes raw filter: > {eomes_gt:g} (selected before filter: {n_selected_before_expr:,})")
    print(
        f"cells (BrdU+ subset = BrdU-only U Dual): total={n_total:,} "
        f"(finite AP/ML before BrdU+ filter: {n_finite:,}) "
        f"(selected after metadata/gene filters before coord filter: {n_selected:,})"
    )
    print(
        f"AP/ML mapping: using obsm['AP_ML_um'] column {ap_col} as AP and column {ml_col} as ML"
    )

    if sag_ap_dfs:
        ap_png = outdir / f"brdu_edu_dual_frac_total_vs_AP_um_SagOnly_leiden_{filter_tag}_bin{bw}_by_dataset.png"
        _plot_frac_total_by_dataset(
            sag_ap_dfs,
            axis_label="AP (um, binned)",
            title=f"Sag only: BrdU/EdU composition vs AP ({filter_desc}, bin {args.bin_width_um:g} um)",
            out_png=ap_png,
        )
        print(f"wrote: {ap_png}")
    if coro_ml_dfs:
        ml_png = outdir / f"brdu_edu_dual_frac_total_vs_ML_um_CoroOnly_leiden_{filter_tag}_bin{bw}_by_dataset.png"
        _plot_frac_total_by_dataset(
            coro_ml_dfs,
            axis_label="ML (um, binned)",
            title=f"Coro only: BrdU/EdU composition vs ML ({filter_desc}, bin {args.bin_width_um:g} um)",
            out_png=ml_png,
        )
        print(f"wrote: {ml_png}")

    plane2d = _binned_counts_2d(
        coords[:, ap_col],
        coords[:, ml_col],
        brdu_pos,
        edu_pos,
        bin_width_um=float(args.bin_width_um),
    )
    plane2d_table = plane2d["table"]
    if not isinstance(plane2d_table, pd.DataFrame):
        raise TypeError("2D plane table is not a DataFrame.")
    plane2d_csv = outdir / f"brdu_dual_2d_plane_AP_ML_leiden_{filter_tag}_bin{bw}.csv"
    plane2d_table.to_csv(plane2d_csv, index=False)

    coef2d, cov2d = _fit_binomial_logit_2d(
        coords[:, ap_col],
        coords[:, ml_col],
        (~edu_pos).astype(float),
        np.ones(coords.shape[0], dtype=float),
        animal=animal,
    )

    plane2d_png = outdir / f"brdu_dual_2d_plane_AP_ML_leiden_{filter_tag}_bin{bw}.png"
    _plot_2d_plane_brdu_dual(
        plane2d,
        coef2d,
        cov2d,
        title=f"Pooled 2D plane (BrdU-only vs Dual): {filter_desc}, bin {args.bin_width_um:g} um",
        out_png=plane2d_png,
    )
    print(
        f"2D plane slopes: AP={coef2d[1]*1000:+.3f}/mm, ML={coef2d[2]*1000:+.3f}/mm; "
        f"intercept={coef2d[0]:+.3f}"
    )
    plane2d_ts_table = plane2d_table.copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        plane2d_ts_table["ts_hours_obs"] = np.divide(
            float(args.delta_t_hours) * plane2d_ts_table["n_dual"].to_numpy(dtype=float),
            plane2d_ts_table["n_brdu_only"].to_numpy(dtype=float),
            out=np.full(plane2d_ts_table.shape[0], np.nan, dtype=float),
            where=plane2d_ts_table["n_brdu_only"].to_numpy(dtype=float) > 0,
        )
    plane2d_ts_csv = outdir / f"ts_2d_plane_AP_ML_leiden_{filter_tag}_bin{bw}.csv"
    plane2d_ts_table.to_csv(plane2d_ts_csv, index=False)
    plane2d_ts_png = outdir / f"ts_2d_plane_AP_ML_leiden_{filter_tag}_bin{bw}.png"
    _plot_2d_plane_ts_hours(
        plane2d,
        coef_ts=coef2d,
        delta_t_hours=float(args.delta_t_hours),
        title=f"Pooled 2D plane (Ts hours): {filter_desc}, bin {args.bin_width_um:g} um",
        out_png=plane2d_ts_png,
    )
    print(f"wrote: {plane2d_png}")
    print(f"wrote: {plane2d_csv}")
    print(f"wrote: {plane2d_ts_png}")
    print(f"wrote: {plane2d_ts_csv}")

    coef2d_by_orientation: dict[str, np.ndarray] = {}
    for ori_label, mask_ori in (
        ("SagOnly", sag_mask),
        ("CoroOnly", coro_mask),
    ):
        if not np.any(mask_ori):
            continue
        plane2d_ori = _binned_counts_2d(
            coords[mask_ori, ap_col],
            coords[mask_ori, ml_col],
            brdu_pos[mask_ori],
            edu_pos[mask_ori],
            bin_width_um=float(args.bin_width_um),
        )
        plane2d_ori_table = plane2d_ori["table"]
        if not isinstance(plane2d_ori_table, pd.DataFrame):
            raise TypeError(f"{ori_label} 2D plane table is not a DataFrame.")
        plane2d_ori_csv = outdir / f"brdu_dual_2d_plane_AP_ML_{ori_label}_leiden_{filter_tag}_bin{bw}.csv"
        plane2d_ori_table.to_csv(plane2d_ori_csv, index=False)
        coef2d_ori, cov2d_ori = _fit_binomial_logit_2d(
            coords[mask_ori, ap_col],
            coords[mask_ori, ml_col],
            (~edu_pos[mask_ori]).astype(float),
            np.ones(int(np.sum(mask_ori)), dtype=float),
            animal=animal[mask_ori],
        )
        coef2d_by_orientation[ori_label] = coef2d_ori
        plane2d_ori_png = outdir / f"brdu_dual_2d_plane_AP_ML_{ori_label}_leiden_{filter_tag}_bin{bw}.png"
        _plot_2d_plane_brdu_dual(
            plane2d_ori,
            coef2d_ori,
            cov2d_ori,
            title=f"{ori_label} 2D plane (BrdU-only vs Dual): {filter_desc}, bin {args.bin_width_um:g} um",
            out_png=plane2d_ori_png,
        )
        plane2d_ts_ori_table = plane2d_ori_table.copy()
        with np.errstate(divide="ignore", invalid="ignore"):
            plane2d_ts_ori_table["ts_hours_obs"] = np.divide(
                float(args.delta_t_hours) * plane2d_ts_ori_table["n_dual"].to_numpy(dtype=float),
                plane2d_ts_ori_table["n_brdu_only"].to_numpy(dtype=float),
                out=np.full(plane2d_ts_ori_table.shape[0], np.nan, dtype=float),
                where=plane2d_ts_ori_table["n_brdu_only"].to_numpy(dtype=float) > 0,
            )
        plane2d_ts_ori_csv = outdir / f"ts_2d_plane_AP_ML_{ori_label}_leiden_{filter_tag}_bin{bw}.csv"
        plane2d_ts_ori_table.to_csv(plane2d_ts_ori_csv, index=False)
        plane2d_ts_ori_png = outdir / f"ts_2d_plane_AP_ML_{ori_label}_leiden_{filter_tag}_bin{bw}.png"
        _plot_2d_plane_ts_hours(
            plane2d_ori,
            coef_ts=coef2d_ori,
            delta_t_hours=float(args.delta_t_hours),
            title=f"{ori_label} 2D plane (Ts hours): {filter_desc}, bin {args.bin_width_um:g} um",
            out_png=plane2d_ts_ori_png,
        )
        print(
            f"{ori_label} 2D plane slopes: AP={coef2d_ori[1]*1000:+.3f}/mm, "
            f"ML={coef2d_ori[2]*1000:+.3f}/mm; intercept={coef2d_ori[0]:+.3f}"
        )
        print(f"wrote: {plane2d_ori_png}")
        print(f"wrote: {plane2d_ori_csv}")
        print(f"wrote: {plane2d_ts_ori_png}")
        print(f"wrote: {plane2d_ts_ori_csv}")

    coef_dual_2d: np.ndarray | None = None
    coef_edu_pos_2d: np.ndarray | None = None
    plane2d_all: dict[str, np.ndarray | pd.DataFrame] | None = None
    if bool(args.plot_native_tc) or bool(args.plot_native_edu_pos_fraction) or bool(args.plot_2d_edu_pos_fraction):
        plane2d_all = _binned_counts_2d(
            coords_all[:, ap_col],
            coords_all[:, ml_col],
            brdu_pos_all,
            edu_pos_all,
            bin_width_um=float(args.bin_width_um),
        )
        plane2d_all_table = plane2d_all["table"]
        if not isinstance(plane2d_all_table, pd.DataFrame):
            raise TypeError("2D all-cell plane table is not a DataFrame.")
        plane2d_all_table = plane2d_all_table.copy()
        with np.errstate(divide="ignore", invalid="ignore"):
            plane2d_all_table["ts_hours_obs"] = np.divide(
                float(args.delta_t_hours) * plane2d_all_table["n_dual"].to_numpy(dtype=float),
                plane2d_all_table["n_brdu_only"].to_numpy(dtype=float),
                out=np.full(plane2d_all_table.shape[0], np.nan, dtype=float),
                where=plane2d_all_table["n_brdu_only"].to_numpy(dtype=float) > 0,
            )
            plane2d_all_table["tc_hours_obs"] = _tc_hours_obs_from_counts(
                n_total=plane2d_all_table["n_total"].to_numpy(dtype=float),
                n_brdu_only=plane2d_all_table["n_brdu_only"].to_numpy(dtype=float),
                n_edu_only=plane2d_all_table["n_edu_only"].to_numpy(dtype=float),
                n_dual=plane2d_all_table["n_dual"].to_numpy(dtype=float),
                delta_t_hours=float(args.delta_t_hours),
            )
        if bool(args.plot_native_tc):
            coef_dual_2d, _cov_dual_2d = _fit_binomial_logit_2d(
                coords_all[:, ap_col],
                coords_all[:, ml_col],
                (brdu_pos_all & edu_pos_all).astype(float),
                np.ones(coords_all.shape[0], dtype=float),
                animal=animal_all,
            )
            print(
                f"2D dual plane slopes: AP={coef_dual_2d[1]*1000:+.3f}/mm, "
                f"ML={coef_dual_2d[2]*1000:+.3f}/mm; intercept={coef_dual_2d[0]:+.3f}"
            )
        if bool(args.plot_native_edu_pos_fraction) or bool(args.plot_2d_edu_pos_fraction):
            plane2d_all_table["n_edu_pos"] = plane2d_all_table["n_edu_only"] + plane2d_all_table["n_dual"]
            coef_edu_pos_2d, _cov_edu_pos_2d = _fit_binomial_logit_2d(
                coords_all[:, ap_col],
                coords_all[:, ml_col],
                edu_pos_all.astype(float),
                np.ones(coords_all.shape[0], dtype=float),
                animal=animal_all,
            )
            print(
                f"2D EdU+ plane slopes: AP={coef_edu_pos_2d[1]*1000:+.3f}/mm, "
                f"ML={coef_edu_pos_2d[2]*1000:+.3f}/mm; intercept={coef_edu_pos_2d[0]:+.3f}"
            )
        if bool(args.plot_2d_edu_pos_fraction):
            if coef_edu_pos_2d is None:
                raise RuntimeError("EdU+ plotting requested but EdU+ plane was not fit.")
            plane2d_edu_pos_csv = outdir / f"edu_pos_fraction_2d_plane_AP_ML_leiden_{filter_tag}_bin{bw}.csv"
            plane2d_all_table.to_csv(plane2d_edu_pos_csv, index=False)
            plane2d_edu_pos_png = outdir / f"edu_pos_fraction_2d_plane_AP_ML_leiden_{filter_tag}_bin{bw}.png"
            _plot_2d_plane_edu_pos_fraction(
                plane2d_all,
                coef_edu_pos=coef_edu_pos_2d,
                title=f"Pooled 2D plane (EdU+ fraction): {filter_desc}, bin {args.bin_width_um:g} um",
                out_png=plane2d_edu_pos_png,
            )
            print(f"wrote: {plane2d_edu_pos_png}")
            print(f"wrote: {plane2d_edu_pos_csv}")
        if bool(args.plot_native_tc):
            if coef_dual_2d is None:
                raise RuntimeError("Tc plotting requested but dual plane was not fit.")
            plane2d_tc_csv = outdir / f"tc_2d_plane_AP_ML_leiden_{filter_tag}_bin{bw}.csv"
            plane2d_all_table.to_csv(plane2d_tc_csv, index=False)
            plane2d_tc_png = outdir / f"tc_2d_plane_AP_ML_leiden_{filter_tag}_bin{bw}.png"
            _plot_2d_plane_tc_hours(
                plane2d_all,
                coef_ts=coef2d,
                coef_dual=coef_dual_2d,
                delta_t_hours=float(args.delta_t_hours),
                title=f"Pooled 2D plane (Tc hours): {filter_desc}, bin {args.bin_width_um:g} um",
                out_png=plane2d_tc_png,
            )
            print(f"wrote: {plane2d_tc_png}")
            print(f"wrote: {plane2d_tc_csv}")

            for ori_label, axis_label, coord_col, mask_ori_brdu, mask_ori_all in (
                ("SagOnly", "AP (um, binned)", ap_col, sag_mask, sag_mask_all),
                ("CoroOnly", "ML (um, binned)", ml_col, coro_mask, coro_mask_all),
            ):
                if not np.any(mask_ori_brdu) or not np.any(mask_ori_all):
                    continue
                edges_1d, _ = _bins_for(coords_all[mask_ori_all, coord_col], bin_width_um=float(args.bin_width_um))
                df_tc = _binned_counts(
                    coords_all[mask_ori_all, coord_col],
                    brdu_pos_all[mask_ori_all],
                    edu_pos_all[mask_ori_all],
                    bin_width_um=float(args.bin_width_um),
                    edges=edges_1d,
                )
                intercept_ts_1d, slope_ts_1d = _fit_binomial_logit(
                    coords[mask_ori_brdu, coord_col],
                    (~edu_pos[mask_ori_brdu]).astype(float),
                    np.ones(int(np.sum(mask_ori_brdu)), dtype=float),
                )
                intercept_dual_1d, slope_dual_1d = _fit_binomial_logit(
                    coords_all[mask_ori_all, coord_col],
                    (brdu_pos_all[mask_ori_all] & edu_pos_all[mask_ori_all]).astype(float),
                    np.ones(int(np.sum(mask_ori_all)), dtype=float),
                )
                df_tc["tc_hours_obs"] = _tc_hours_obs_from_counts(
                    n_total=df_tc["n_total"].to_numpy(dtype=float),
                    n_brdu_only=df_tc["n_brdu_only"].to_numpy(dtype=float),
                    n_edu_only=df_tc["n_edu_only"].to_numpy(dtype=float),
                    n_dual=df_tc["n_dual"].to_numpy(dtype=float),
                    delta_t_hours=float(args.delta_t_hours),
                )
                df_tc["tc_hours_fit"] = _predict_tc_hours_1d(
                    intercept_ts=intercept_ts_1d,
                    slope_ts=slope_ts_1d,
                    intercept_dual=intercept_dual_1d,
                    slope_dual=slope_dual_1d,
                    coord_um=df_tc["bin_center_um"].to_numpy(dtype=float),
                    delta_t_hours=float(args.delta_t_hours),
                )
                tc_line_csv = outdir / f"tc_hours_line_{ori_label}_leiden_{filter_tag}_bin{bw}.csv"
                df_tc.to_csv(tc_line_csv, index=False)
                tc_line_png = outdir / f"tc_hours_line_{ori_label}_leiden_{filter_tag}_bin{bw}.png"
                _plot_tc_hours_1d(
                    df_tc,
                    axis_label=axis_label,
                    title=f"{ori_label} Tc line plot: {filter_desc}, bin {args.bin_width_um:g} um",
                    out_png=tc_line_png,
                )
                print(f"wrote: {tc_line_png}")
                print(f"wrote: {tc_line_csv}")

        if bool(args.plot_native_tc):
            for ori_label, mask_ori_all in (("SagOnly", sag_mask_all), ("CoroOnly", coro_mask_all)):
                if not np.any(mask_ori_all):
                    continue
                if ori_label not in coef2d_by_orientation:
                    raise RuntimeError(f"Missing {ori_label} Ts plane fit needed for Tc 2D plane.")
                plane2d_all_ori = _binned_counts_2d(
                    coords_all[mask_ori_all, ap_col],
                    coords_all[mask_ori_all, ml_col],
                    brdu_pos_all[mask_ori_all],
                    edu_pos_all[mask_ori_all],
                    bin_width_um=float(args.bin_width_um),
                )
                plane2d_all_ori_table = plane2d_all_ori["table"]
                if not isinstance(plane2d_all_ori_table, pd.DataFrame):
                    raise TypeError(f"{ori_label} 2D all-cell plane table is not a DataFrame.")
                plane2d_all_ori_table = plane2d_all_ori_table.copy()
                with np.errstate(divide="ignore", invalid="ignore"):
                    plane2d_all_ori_table["ts_hours_obs"] = np.divide(
                        float(args.delta_t_hours) * plane2d_all_ori_table["n_dual"].to_numpy(dtype=float),
                        plane2d_all_ori_table["n_brdu_only"].to_numpy(dtype=float),
                        out=np.full(plane2d_all_ori_table.shape[0], np.nan, dtype=float),
                        where=plane2d_all_ori_table["n_brdu_only"].to_numpy(dtype=float) > 0,
                    )
                    plane2d_all_ori_table["tc_hours_obs"] = _tc_hours_obs_from_counts(
                        n_total=plane2d_all_ori_table["n_total"].to_numpy(dtype=float),
                        n_brdu_only=plane2d_all_ori_table["n_brdu_only"].to_numpy(dtype=float),
                        n_edu_only=plane2d_all_ori_table["n_edu_only"].to_numpy(dtype=float),
                        n_dual=plane2d_all_ori_table["n_dual"].to_numpy(dtype=float),
                        delta_t_hours=float(args.delta_t_hours),
                    )
                plane2d_tc_ori_csv = outdir / f"tc_2d_plane_AP_ML_{ori_label}_leiden_{filter_tag}_bin{bw}.csv"
                plane2d_all_ori_table.to_csv(plane2d_tc_ori_csv, index=False)
                coef_dual_2d_ori, _cov_dual_2d_ori = _fit_binomial_logit_2d(
                    coords_all[mask_ori_all, ap_col],
                    coords_all[mask_ori_all, ml_col],
                    (brdu_pos_all[mask_ori_all] & edu_pos_all[mask_ori_all]).astype(float),
                    np.ones(int(np.sum(mask_ori_all)), dtype=float),
                    animal=animal_all[mask_ori_all],
                )
                plane2d_tc_ori_png = outdir / f"tc_2d_plane_AP_ML_{ori_label}_leiden_{filter_tag}_bin{bw}.png"
                _plot_2d_plane_tc_hours(
                    plane2d_all_ori,
                    coef_ts=coef2d_by_orientation[ori_label],
                    coef_dual=coef_dual_2d_ori,
                    delta_t_hours=float(args.delta_t_hours),
                    title=f"{ori_label} 2D plane (Tc hours): {filter_desc}, bin {args.bin_width_um:g} um",
                    out_png=plane2d_tc_ori_png,
                )
                print(
                    f"{ori_label} 2D dual plane slopes: AP={coef_dual_2d_ori[1]*1000:+.3f}/mm, "
                    f"ML={coef_dual_2d_ori[2]*1000:+.3f}/mm; intercept={coef_dual_2d_ori[0]:+.3f}"
                )
                print(f"wrote: {plane2d_tc_ori_png}")
                print(f"wrote: {plane2d_tc_ori_csv}")

    if bool(args.plot_native_ts):
        native_ts_png = outdir / f"ts_hours_native_proj_AP_ML_leiden_{filter_tag}_bin{bw}.png"
        _plot_native_ts_surface(
            coef=coef2d,
            delta_t_hours=float(args.delta_t_hours),
            out_png=native_ts_png,
            refextract_outdir=args.refextract_outdir.expanduser(),
            refextract_slice_i_min=int(args.refextract_slice_i_min),
            refextract_slice_i_max=int(args.refextract_slice_i_max),
            refextract_n_t=int(args.refextract_n_t),
            refextract_ref_t=float(args.refextract_ref_t),
            refextract_band_frac=float(args.refextract_band_frac),
            refextract_res_ijk_um=tuple(float(x) for x in args.refextract_res_ijk_um),
            native_elev_deg=float(args.native_elev_deg),
            native_azim_deg=float(args.native_azim_deg),
            native_roll_deg=float(args.native_roll_deg),
            native_latlon=bool(args.native_latlon),
            native_graticule=str(args.native_graticule),
            native_lat_stride=int(args.native_lat_stride),
            native_lon_stride=int(args.native_lon_stride),
            native_max_lat_lines=int(args.native_max_lat_lines),
            native_max_lon_lines=int(args.native_max_lon_lines),
            native_proj_type=str(args.native_proj_type),
            native_focal_length=float(args.native_focal_length),
            ap_support_bounds_um=ap_support_bounds_um,
            ml_support_bounds_um=ml_support_bounds_um,
            restrict_t_neomeso=bool(args.restrict_t_neomeso),
            gray_context=bool(args.gray_context),
            title=(
                f"Ts native projection: {filter_desc}, "
                f"Δt={float(args.delta_t_hours):g} h, bin {args.bin_width_um:g} um"
            ),
        )
        print(f"wrote: {native_ts_png}")

    if bool(args.plot_native_brdu_only_fraction):
        native_brdu_only_png = outdir / f"brdu_only_fraction_native_proj_AP_ML_leiden_{filter_tag}_bin{bw}.png"
        _plot_native_brdu_only_fraction_surface(
            coef=coef2d,
            out_png=native_brdu_only_png,
            refextract_outdir=args.refextract_outdir.expanduser(),
            refextract_slice_i_min=int(args.refextract_slice_i_min),
            refextract_slice_i_max=int(args.refextract_slice_i_max),
            refextract_n_t=int(args.refextract_n_t),
            refextract_ref_t=float(args.refextract_ref_t),
            refextract_band_frac=float(args.refextract_band_frac),
            refextract_res_ijk_um=tuple(float(x) for x in args.refextract_res_ijk_um),
            native_elev_deg=float(args.native_elev_deg),
            native_azim_deg=float(args.native_azim_deg),
            native_roll_deg=float(args.native_roll_deg),
            native_latlon=bool(args.native_latlon),
            native_graticule=str(args.native_graticule),
            native_lat_stride=int(args.native_lat_stride),
            native_lon_stride=int(args.native_lon_stride),
            native_max_lat_lines=int(args.native_max_lat_lines),
            native_max_lon_lines=int(args.native_max_lon_lines),
            native_proj_type=str(args.native_proj_type),
            native_focal_length=float(args.native_focal_length),
            ap_support_bounds_um=ap_support_bounds_um,
            ml_support_bounds_um=ml_support_bounds_um,
            restrict_t_neomeso=bool(args.restrict_t_neomeso),
            gray_context=bool(args.gray_context),
            title=f"BrdU-only fraction native projection: {filter_desc}, bin {args.bin_width_um:g} um",
        )
        print(f"wrote: {native_brdu_only_png}")

    if bool(args.plot_native_tc):
        if coef_dual_2d is None:
            raise RuntimeError("Tc plotting requested but dual plane was not fit.")
        native_tc_png = outdir / f"tc_hours_native_proj_AP_ML_leiden_{filter_tag}_bin{bw}.png"
        _plot_native_tc_surface(
            coef_ts=coef2d,
            coef_dual=coef_dual_2d,
            delta_t_hours=float(args.delta_t_hours),
            out_png=native_tc_png,
            refextract_outdir=args.refextract_outdir.expanduser(),
            refextract_slice_i_min=int(args.refextract_slice_i_min),
            refextract_slice_i_max=int(args.refextract_slice_i_max),
            refextract_n_t=int(args.refextract_n_t),
            refextract_ref_t=float(args.refextract_ref_t),
            refextract_band_frac=float(args.refextract_band_frac),
            refextract_res_ijk_um=tuple(float(x) for x in args.refextract_res_ijk_um),
            native_elev_deg=float(args.native_elev_deg),
            native_azim_deg=float(args.native_azim_deg),
            native_roll_deg=float(args.native_roll_deg),
            native_latlon=bool(args.native_latlon),
            native_graticule=str(args.native_graticule),
            native_lat_stride=int(args.native_lat_stride),
            native_lon_stride=int(args.native_lon_stride),
            native_max_lat_lines=int(args.native_max_lat_lines),
            native_max_lon_lines=int(args.native_max_lon_lines),
            native_proj_type=str(args.native_proj_type),
            native_focal_length=float(args.native_focal_length),
            ap_support_bounds_um=ap_support_bounds_um,
            ml_support_bounds_um=ml_support_bounds_um,
            restrict_t_neomeso=bool(args.restrict_t_neomeso),
            gray_context=bool(args.gray_context),
            title=(
                f"Tc native projection: {filter_desc}, "
                f"Δt={float(args.delta_t_hours):g} h, bin {args.bin_width_um:g} um"
            ),
        )
        print(f"wrote: {native_tc_png}")

    if bool(args.plot_native_edu_pos_fraction):
        if coef_edu_pos_2d is None:
            raise RuntimeError("EdU+ fraction plotting requested but EdU+ plane was not fit.")
        native_edu_pos_png = outdir / f"edu_pos_fraction_native_proj_AP_ML_leiden_{filter_tag}_bin{bw}.png"
        _plot_native_edu_pos_fraction_surface(
            coef_edu_pos=coef_edu_pos_2d,
            out_png=native_edu_pos_png,
            refextract_outdir=args.refextract_outdir.expanduser(),
            refextract_slice_i_min=int(args.refextract_slice_i_min),
            refextract_slice_i_max=int(args.refextract_slice_i_max),
            refextract_n_t=int(args.refextract_n_t),
            refextract_ref_t=float(args.refextract_ref_t),
            refextract_band_frac=float(args.refextract_band_frac),
            refextract_res_ijk_um=tuple(float(x) for x in args.refextract_res_ijk_um),
            native_elev_deg=float(args.native_elev_deg),
            native_azim_deg=float(args.native_azim_deg),
            native_roll_deg=float(args.native_roll_deg),
            native_latlon=bool(args.native_latlon),
            native_graticule=str(args.native_graticule),
            native_lat_stride=int(args.native_lat_stride),
            native_lon_stride=int(args.native_lon_stride),
            native_max_lat_lines=int(args.native_max_lat_lines),
            native_max_lon_lines=int(args.native_max_lon_lines),
            native_proj_type=str(args.native_proj_type),
            native_focal_length=float(args.native_focal_length),
            ap_support_bounds_um=ap_support_bounds_um,
            ml_support_bounds_um=ml_support_bounds_um,
            restrict_t_neomeso=bool(args.restrict_t_neomeso),
            gray_context=bool(args.gray_context),
            title=f"EdU+ fraction native projection: {filter_desc}, bin {args.bin_width_um:g} um",
        )
        print(f"wrote: {native_edu_pos_png}")


if __name__ == "__main__":
    main()
