# %% [markdown]
# # Pick BrdU/EdU thresholds per ROI/subROI (click UI)
#
# For each ROI/subROI job, this script:
# 1) computes tricycle embedding from log1p counts using `neuroRef.csv`,
# 2) shows:
#    - tricycle[:, 0] vs log_brdu_mean
#    - tricycle[:, 0] vs log_edu_mean
# 3) lets you click each panel to set a horizontal threshold line.
#
# Run cells sequentially. Outputs are written next to each `.annotated.h5ad`.

# %%
from __future__ import annotations

import os
from pathlib import Path

_MPLRC_IGNORE = Path("/tmp/fishtools_empty_matplotlibrc")
_MPLRC_IGNORE.touch(exist_ok=True)
os.environ["MATPLOTLIBRC"] = str(_MPLRC_IGNORE)

import json

import anndata as ad
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
import scipy.sparse as sp
from IPython import get_ipython
from matplotlib.widgets import Button  # noqa: E402

from fishtools.io.workspace import Workspace
from fishtools.utils.plot import plot_embedding

# Ignore local/user matplotlibrc by pointing to an empty rc file.



# Use widget backend for VS Code interactive mode
ip = get_ipython()
ip.run_line_magic("matplotlib", "widget")


# --------------------------------
folders = sorted(Path("/working").expanduser().glob("2025*JaxA1*"))
FOLDER = folders[1]
# --------------------------------

print("Using folder:", FOLDER)
ws = Workspace(str(FOLDER))

# Optional filters
ROI_NAMES: tuple[str, ...] | None = None
ROI_OBS_KEY = "ccf_adjusted"
ROI_VALUES: tuple[str, ...] | None = None
SKIP_EXISTING_JSON = False  # True: skip jobs when output threshold JSON already exists.
SKIP_EXISTING_JSON = False  # True: skip jobs when output threshold JSON already exists.
PLOT_MAX_POINTS = 200_000
POINT_SIZE = 2
POINT_ALPHA = 0.3
PLOT_JITTER_LINEAR_SIGMA = 10.0
RUN_SPATIAL_DIAGNOSTICS = True

# Job index (incremented in the plotting cell below)
i = 0

ALL_PROGENITORS_H5AD = Path("~/nvme/all_progenitors.h5ad").expanduser()
USAGE_NORM_K9_PARQUET = Path("~/nvme/cnmf_all_progenitors/usage_norm.k9.dt0.1.parquet").expanduser()

usage_norm_k9 = pd.read_parquet(
    USAGE_NORM_K9_PARQUET,
    columns=["index", "dataset", "Usage_4", "Usage_7"],
    engine="pyarrow",
)
usage_norm_k9["index"] = usage_norm_k9["index"].astype(str)
usage_norm_k9["dataset"] = usage_norm_k9["dataset"].astype(str)
all_progenitors_backed: ad.AnnData | None = None


def _valid_roi_values(series: np.ndarray) -> list[str]:
    return [v for v in series.tolist() if v != "" and v.lower() not in {"nan", "none"}]


def _threshold_json_path(input_h5ad: Path, roi_value: str | None, *, multi: bool) -> Path:
    if roi_value is None:
        return input_h5ad.parent / f"{input_h5ad.stem}.brdu_edu_thresholds.json"
    if multi:
        return input_h5ad.parent / f"{input_h5ad.stem}.{roi_value}.brdu_edu_thresholds.json"
    return input_h5ad.parent / f"{input_h5ad.stem}.brdu_edu_thresholds.json"


def build_jobs(*, folder: Path, skip_completed: bool) -> list[tuple[Path, str | None, Path]]:
    ccf_dir = folder / "analysis" / "output" / "ccf-transforms"
    jobs: list[tuple[Path, str | None, Path]] = []

    for h5ad in sorted(ccf_dir.glob("*/*.annotated.h5ad")):
        roi_name = h5ad.parent.name
        if ROI_NAMES is not None and roi_name not in set(ROI_NAMES):
            continue

        adata_scan = ad.read_h5ad(h5ad, backed="r")
        try:
            if ROI_OBS_KEY in adata_scan.obs.columns:
                roi_series = adata_scan.obs[ROI_OBS_KEY].astype(str).to_numpy()
                roi_unique = sorted(set(_valid_roi_values(roi_series)))
            else:
                roi_unique = []
        finally:
            if getattr(adata_scan, "isbacked", False) and getattr(adata_scan, "file", None) is not None:
                adata_scan.file.close()

        if roi_unique:
            selected = roi_unique if ROI_VALUES is None else [v for v in ROI_VALUES if v in set(roi_unique)]
            multi = len(selected) > 1
            for roi_value in selected:
                out_json = _threshold_json_path(h5ad, roi_value, multi=multi)
                if skip_completed and out_json.exists():
                    continue
                jobs.append((h5ad, roi_value, out_json))
        else:
            out_json = _threshold_json_path(h5ad, None, multi=False)
            if skip_completed and out_json.exists():
                continue
            jobs.append((h5ad, None, out_json))

    return jobs


def _load_log1p_matrix(adata: ad.AnnData, genes: list[str]) -> np.ndarray:
    if "log1p" in adata.layers:
        x = adata[:, genes].layers["log1p"]
    elif "raw" in adata.layers:
        x = np.log1p(adata[:, genes].layers["raw"])
    else:
        x = np.log1p(adata[:, genes].X)

    if sp.issparse(x):
        x = x.toarray()
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Unexpected expression matrix shape: {arr.shape}")
    return arr


def compute_tricycle_from_log1p(adata: ad.AnnData, *, repo_root: Path) -> None:
    ref_path = repo_root / "neuroRef.csv"
    if not ref_path.exists():
        raise FileNotFoundError(ref_path)

    trc = pd.read_csv(ref_path)
    needed = {"symbol", "pc1.rot", "pc2.rot"}
    missing = needed - set(trc.columns)
    if missing:
        raise KeyError(f"Missing neuroRef columns: {sorted(missing)}")

    shared = sorted(set(trc["symbol"].astype(str)) & set(adata.var_names.astype(str)))
    if not shared:
        raise ValueError("No shared genes between adata.var_names and neuroRef.csv")

    loadings = (
        trc[trc["symbol"].isin(shared)]
        .set_index("symbol")
        .reindex(shared)[["pc1.rot", "pc2.rot"]]
        .to_numpy(dtype=np.float32)
    )

    x = _load_log1p_matrix(adata, shared)
    x_centered = x - np.mean(x, axis=0, keepdims=True)
    pls = x_centered @ loadings

    adata.obsm["tricycle"] = pls
    adata.obsm["X_tricycle"] = pls
    adata.obs["tricycle"] = (np.arctan2(pls[:, 1], pls[:, 0]) + 2 * np.pi) % (2 * np.pi)


def load_existing_thresholds(path: Path) -> dict[str, float | None]:
    if not path.exists():
        return {"log_brdu_mean": None, "log_edu_mean": None}
    data = json.loads(path.read_text())
    vals = data.get("thresholds", {})
    return {
        "log_brdu_mean": float(vals["log_brdu_mean"]) if vals.get("log_brdu_mean") is not None else None,
        "log_edu_mean": float(vals["log_edu_mean"]) if vals.get("log_edu_mean") is not None else None,
    }


def ensure_log_brdu_edu_mean(adata: ad.AnnData) -> None:
    def _minmax_0_65535(values: np.ndarray) -> np.ndarray:
        finite = np.isfinite(values)
        if not np.any(finite):
            return np.zeros_like(values, dtype=float)
        p1 = float(np.percentile(values[finite], 10.0))
        clipped = values.copy()
        clipped[finite & (clipped < p1)] = p1
        lo = float(np.min(clipped[finite]))
        hi = float(np.max(clipped[finite]))
        if hi <= lo:
            out = np.zeros_like(values, dtype=float)
            out[~finite] = np.nan
            return out
        out = (clipped - lo) / (hi - lo) * 65535.0
        out[~finite] = np.nan
        return np.clip(out, 0.0, 65535.0)

    for marker in ("brdu", "edu"):
        mean_col = f"{marker}_mean"
        log_col = f"log_{marker}_mean"
        if mean_col in adata.obs.columns:
            scaled = _minmax_0_65535(adata.obs[mean_col].to_numpy(dtype=float))
            adata.obs[log_col] = np.log1p(scaled)
            continue
        if log_col not in adata.obs.columns:
            raise KeyError(
                f"Missing required obs columns: expected `{mean_col}` to compute `{log_col}` with np.log1p."
            )


def get_all_progenitors_backed(path: Path) -> ad.AnnData:
    global all_progenitors_backed
    if all_progenitors_backed is not None:
        return all_progenitors_backed
    if not path.exists():
        raise FileNotFoundError(path)
    all_progenitors_backed = ad.read_h5ad(path)
    all_progenitors_backed = all_progenitors_backed[all_progenitors_backed.obs['leiden'].isin(list(map(str,[7,8,9,10])))]
    return all_progenitors_backed


def load_joined_progenitors_usage(
    *,
    dataset: str,
    roi_name: str,
    roi_value: str | None,
    roi_obs_key: str,
    all_h5ad: Path,
    usage_df: pd.DataFrame,
) -> tuple[ad.AnnData, np.ndarray, np.ndarray]:
    usage_cols = {"index", "dataset", "Usage_4", "Usage_7"}
    usage_missing = usage_cols - set(usage_df.columns)
    if usage_missing:
        raise KeyError(f"Missing usage parquet columns: {sorted(usage_missing)}")

    adata_backed = get_all_progenitors_backed(all_h5ad)
    obs = adata_backed.obs
    obs_required = {"dataset", "roi"}
    if roi_value is not None:
        obs_required.add(roi_obs_key)
    obs_missing = obs_required - set(obs.columns)
    if obs_missing:
        raise KeyError(f"Missing all_progenitors obs columns: {sorted(obs_missing)}")

    mask = obs["dataset"].astype(str) == str(dataset)
    mask &= obs["roi"].astype(str) == str(roi_name)
    if roi_value is not None:
        mask &= obs[roi_obs_key].astype(str) == str(roi_value)
    keep_idx = np.flatnonzero(mask.to_numpy())
    if keep_idx.size == 0:
        raise ValueError(
            f"No cells in {all_h5ad} for dataset={dataset}, roi={roi_name}, {roi_obs_key}={roi_value}."
        )
    adata_sel = adata_backed[keep_idx].to_memory()

    usage_subset = usage_df.loc[
        usage_df["dataset"] == str(dataset),
        ["index", "Usage_4", "Usage_7"],
    ].copy()
    if usage_subset.empty:
        raise ValueError(f"No usage rows found for dataset={dataset} in {USAGE_NORM_K9_PARQUET}.")
    usage_dup = usage_subset["index"].duplicated(keep=False)
    if bool(usage_dup.any()):
        n_dup = int(usage_dup.sum())
        raise ValueError(f"Found {n_dup} duplicate `index` values in usage rows for dataset={dataset}.")

    usage_lookup = usage_subset.set_index("index")
    join_keys_raw = pd.Index(adata_sel.obs_names.astype(str))
    prefix_candidates = (f"{dataset}:", f"{dataset}|")
    join_keys = join_keys_raw
    for prefix in prefix_candidates:
        has_prefix = join_keys_raw.str.startswith(prefix)
        if bool(has_prefix.all()):
            join_keys = join_keys_raw.str.slice(len(prefix), None)
            break
        if bool(has_prefix.any()):
            raise ValueError(
                f"Mixed dataset prefix in obs_names for dataset={dataset}: prefix={prefix!r} matched only some rows."
            )
    usage_aligned = usage_lookup.reindex(join_keys)
    valid = usage_aligned[["Usage_4", "Usage_7"]].notna().all(axis=1).to_numpy(dtype=bool)
    n_joined = int(valid.sum())
    if n_joined == 0:
        raise ValueError(
            f"Inner join is empty for dataset={dataset}, roi={roi_name}, {roi_obs_key}={roi_value}. "
            f"All-progenitors filtered cells={adata_sel.n_obs}, usage rows={len(usage_subset)}."
        )

    adata_joined = adata_sel[valid].copy()
    usage_joined = usage_aligned.loc[valid, ["Usage_4", "Usage_7"]]
    usage_4 = usage_joined["Usage_4"].to_numpy(dtype=float, copy=False)
    usage_7 = usage_joined["Usage_7"].to_numpy(dtype=float, copy=False)
    return adata_joined, usage_4, usage_7


def pick_thresholds(
    *,
    adata_in: ad.AnnData,
    input_h5ad: Path,
    out_json: Path,
    title: str,
    roi_value: str | None,
    initial: dict[str, float | None],
) -> None:
    ensure_log_brdu_edu_mean(adata_in)
    for col in ("log_brdu_mean", "log_edu_mean"):
        if col not in adata_in.obs.columns:
            raise KeyError(f"Missing required obs column: {col}")
    if "tricycle" not in adata_in.obsm:
        raise KeyError("Missing adata.obsm['tricycle']")

    x_all = np.asarray(adata_in.obsm["tricycle"], dtype=float)[:, 0]
    y_brdu_all = adata_in.obs["log_brdu_mean"].to_numpy(dtype=float)
    y_edu_all = adata_in.obs["log_edu_mean"].to_numpy(dtype=float)

    rng = np.random.default_rng(0)
    plot_idx = np.arange(adata_in.n_obs, dtype=int)
    if plot_idx.size > PLOT_MAX_POINTS:
        plot_idx = rng.choice(plot_idx, size=PLOT_MAX_POINTS, replace=False)

    x = x_all[plot_idx]
    y_brdu = y_brdu_all[plot_idx]
    y_edu = y_edu_all[plot_idx]
    spot_color = plt.get_cmap("CMRmap_r")(0.6)

    thresholds: dict[str, float | None] = {
        "log_brdu_mean": initial["log_brdu_mean"],
        "log_edu_mean": initial["log_edu_mean"],
    }

    fig, axs = plt.subplots(ncols=3, figsize=(12, 5), sharex=False)
    fig.suptitle(title)

    axs[0].scatter(
        x,
        y_brdu,
        color=spot_color,
        s=POINT_SIZE,
        alpha=POINT_ALPHA,
        edgecolors="black",
        linewidths=0.05,
    )
    axs[0].set_xlabel("tricycle[:,0]")
    axs[0].set_ylabel("log_brdu_mean")
    axs[0].set_title("BrdU")

    axs[1].scatter(
        x,
        y_edu,
        color=spot_color,
        s=POINT_SIZE,
        alpha=POINT_ALPHA,
        edgecolors="black",
        linewidths=0.04,
    )
    axs[1].set_xlabel("tricycle[:,0]")
    axs[1].set_ylabel("log_edu_mean")
    axs[1].set_title("EdU")

    axs[2].scatter(
        y_brdu,
        y_edu,
        c=x,
        s=POINT_SIZE,
        alpha=POINT_ALPHA,
        cmap="CMRmap_r",
        edgecolors="black",
        linewidths=0.04,
        vmin=-1.5,
        vmax=-0.2
    )
    axs[2].set_xlabel("log_brdu_mean")
    axs[2].set_ylabel("log_edu_mean")
    axs[2].set_title("BrdU vs EdU")

    line_brdu = axs[0].axhline(
        thresholds["log_brdu_mean"] if thresholds["log_brdu_mean"] is not None else 0.0,
        color="crimson",
        linestyle="--",
        linewidth=1.0,
        visible=thresholds["log_brdu_mean"] is not None,
    )
    line_edu = axs[1].axhline(
        thresholds["log_edu_mean"] if thresholds["log_edu_mean"] is not None else 0.0,
        color="crimson",
        linestyle="--",
        linewidth=1.0,
        visible=thresholds["log_edu_mean"] is not None,
    )
    line_brdu_cross = axs[2].axvline(
        thresholds["log_brdu_mean"] if thresholds["log_brdu_mean"] is not None else 0.0,
        color="crimson",
        linestyle="--",
        linewidth=1.0,
        visible=thresholds["log_brdu_mean"] is not None,
    )
    line_edu_cross = axs[2].axhline(
        thresholds["log_edu_mean"] if thresholds["log_edu_mean"] is not None else 0.0,
        color="crimson",
        linestyle="--",
        linewidth=1.0,
        visible=thresholds["log_edu_mean"] is not None,
    )

    status = fig.text(0.01, 0.01, "", ha="left", va="bottom", fontsize=10)

    def refresh_status() -> None:
        brdu_thr = thresholds["log_brdu_mean"]
        edu_thr = thresholds["log_edu_mean"]
        brdu_pct = (
            f"{100.0 * float(np.mean(y_brdu_all >= brdu_thr)):.2f}%"
            if brdu_thr is not None
            else "unset"
        )
        edu_pct = f"{100.0 * float(np.mean(y_edu_all >= edu_thr)):.2f}%" if edu_thr is not None else "unset"
        double_pct = (
            f"{100.0 * float(np.mean((y_brdu_all >= brdu_thr) & (y_edu_all >= edu_thr))):.2f}%"
            if brdu_thr is not None and edu_thr is not None
            else "unset"
        )
        status.set_text(
            "log_brdu_mean="
            + (f"{brdu_thr:.4f}" if brdu_thr is not None else "unset")
            + " | log_edu_mean="
            + (f"{edu_thr:.4f}" if edu_thr is not None else "unset")
            + "\nBrdU+="
            + brdu_pct
            + " | EdU+="
            + edu_pct
            + " | Double+="
            + double_pct
        )
        fig.canvas.draw_idle()

    def on_click(event) -> None:
        if event.xdata is None or event.ydata is None:
            return
        if event.inaxes is axs[0]:
            y = float(event.ydata)
            thresholds["log_brdu_mean"] = y
            line_brdu.set_ydata([y, y])
            line_brdu.set_visible(True)
            line_brdu_cross.set_xdata([y, y])
            line_brdu_cross.set_visible(True)
            print(f"Set log_brdu_mean threshold: {y:.6f}")
            refresh_status()
            return
        if event.inaxes is axs[1]:
            y = float(event.ydata)
            thresholds["log_edu_mean"] = y
            line_edu.set_ydata([y, y])
            line_edu.set_visible(True)
            line_edu_cross.set_ydata([y, y])
            line_edu_cross.set_visible(True)
            print(f"Set log_edu_mean threshold: {y:.6f}")
            refresh_status()
            return
        if event.inaxes is axs[2]:
            brdu_y = float(event.xdata)
            edu_y = float(event.ydata)
            thresholds["log_brdu_mean"] = brdu_y
            thresholds["log_edu_mean"] = edu_y
            line_brdu.set_ydata([brdu_y, brdu_y])
            line_brdu.set_visible(True)
            line_edu.set_ydata([edu_y, edu_y])
            line_edu.set_visible(True)
            line_brdu_cross.set_xdata([brdu_y, brdu_y])
            line_brdu_cross.set_visible(True)
            line_edu_cross.set_ydata([edu_y, edu_y])
            line_edu_cross.set_visible(True)
            print(f"Set thresholds from BrdU vs EdU: log_brdu_mean={brdu_y:.6f}, log_edu_mean={edu_y:.6f}")
            refresh_status()

    def on_reset(_event) -> None:
        thresholds["log_brdu_mean"] = None
        thresholds["log_edu_mean"] = None
        line_brdu.set_visible(False)
        line_edu.set_visible(False)
        line_brdu_cross.set_visible(False)
        line_edu_cross.set_visible(False)
        print("Reset thresholds.")
        refresh_status()

    def on_save(_event) -> None:
        if thresholds["log_brdu_mean"] is None or thresholds["log_edu_mean"] is None:
            print("Set both thresholds before saving.")
            return
        payload = {
            "input_h5ad": str(input_h5ad),
            "roi_obs_key": ROI_OBS_KEY if roi_value is not None else None,
            "roi_value": roi_value,
            "n_obs": int(adata_in.n_obs),
            "thresholds": {
                "log_brdu_mean": float(thresholds["log_brdu_mean"]),
                "log_edu_mean": float(thresholds["log_edu_mean"]),
            },
        }
        out_json.write_text(json.dumps(payload, indent=2) + "\n")
        print("Wrote:", out_json)

    fig.canvas.mpl_connect("button_press_event", on_click)

    ax_reset = fig.add_axes([0.79, 0.01, 0.1, 0.06])
    ax_save = fig.add_axes([0.9, 0.01, 0.1, 0.06])
    btn_reset = Button(ax_reset, "Reset")
    btn_save = Button(ax_save, "Save")
    btn_reset.on_clicked(on_reset)
    btn_save.on_clicked(on_save)

    # Keep widget objects alive for `%matplotlib widget`.
    fig._pick_brdu_edu_threshold_handles = (btn_reset, btn_save)  # type: ignore[attr-defined]

    refresh_status()
    plt.tight_layout(rect=[0.0, 0.08, 1.0, 0.95])
    plt.show()


def pick_thresholds_modules(
    *,
    adata_in: ad.AnnData,
    input_h5ad: Path,
    out_json: Path,
    title: str,
    roi_value: str | None,
    initial: dict[str, float | None],
    usage_4_all: np.ndarray,
    usage_7_all: np.ndarray,
) -> None:
    ensure_log_brdu_edu_mean(adata_in)
    for col in ("log_brdu_mean", "log_edu_mean"):
        if col not in adata_in.obs.columns:
            raise KeyError(f"Missing required obs column: {col}")

    usage_4 = np.asarray(usage_4_all, dtype=float)
    usage_7 = np.asarray(usage_7_all, dtype=float)
    x_all = np.maximum(usage_4, usage_7)
    if usage_4.shape != (adata_in.n_obs,) or usage_7.shape != (adata_in.n_obs,):
        raise ValueError(
            "Usage arrays must match adata_in.n_obs exactly; got "
            f"Usage_4={usage_4.shape}, Usage_7={usage_7.shape}, n_obs={adata_in.n_obs}."
        )

    y_brdu_all = adata_in.obs["log_brdu_mean"].to_numpy(dtype=float)
    y_edu_all = adata_in.obs["log_edu_mean"].to_numpy(dtype=float)

    rng = np.random.default_rng(0)
    plot_idx = np.arange(adata_in.n_obs, dtype=int)
    if plot_idx.size > PLOT_MAX_POINTS:
        plot_idx = rng.choice(plot_idx, size=PLOT_MAX_POINTS, replace=False)

    x_brdu = x_all[plot_idx]
    x_edu = x_all[plot_idx]
    y_brdu = y_brdu_all[plot_idx]
    y_edu = y_edu_all[plot_idx]

    def jitter_log1p_in_linear_space(values: np.ndarray) -> np.ndarray:
        finite = np.isfinite(values)
        if not np.any(finite):
            return values
        jittered = values.copy()
        linear = np.expm1(values[finite])
        noise = np.abs(rng.normal(loc=0.0, scale=PLOT_JITTER_LINEAR_SIGMA, size=linear.shape))
        jittered_linear = np.clip(linear + noise, a_min=0.0, a_max=None)
        jittered[finite] = np.log1p(jittered_linear)
        return jittered

    x_brdu_plot = x_brdu
    x_edu_plot = x_edu
    y_brdu_plot = jitter_log1p_in_linear_space(y_brdu)
    y_edu_plot = jitter_log1p_in_linear_space(y_edu)
    spot_color = plt.get_cmap("CMRmap_r")(0.6)
    cross_color_all = x_all
    cross_color = cross_color_all[plot_idx]
    cross_vmin = float(np.nanpercentile(cross_color_all, 1.0))
    cross_vmax = float(np.nanpercentile(cross_color_all, 99.0))

    thresholds: dict[str, float | None] = {
        "log_brdu_mean": initial["log_brdu_mean"],
        "log_edu_mean": initial["log_edu_mean"],
    }

    fig, axs = plt.subplots(ncols=3, figsize=(12, 5), sharex=False)
    fig.suptitle(title)

    axs[0].scatter(
        x_brdu_plot,
        y_brdu_plot,
        color=spot_color,
        s=POINT_SIZE,
        alpha=POINT_ALPHA,
        edgecolors="black",
        linewidths=0.05,
    )
    axs[0].set_xlabel("max(Usage_4, Usage_7)")
    axs[0].set_ylabel("log_brdu_mean")
    axs[0].set_title("BrdU")

    axs[1].scatter(
        x_edu_plot,
        y_edu_plot,
        color=spot_color,
        s=POINT_SIZE,
        alpha=POINT_ALPHA,
        edgecolors="black",
        linewidths=0.04,
    )
    axs[1].set_xlabel("max(Usage_4, Usage_7)")
    axs[1].set_ylabel("log_edu_mean")
    axs[1].set_title("EdU")

    axs[2].scatter(
        y_brdu_plot,
        y_edu_plot,
        c=cross_color,
        s=POINT_SIZE * 2.0,
        alpha=POINT_ALPHA,
        cmap="turbo",
        edgecolors="black",
        linewidths=0.04,
        vmin=cross_vmin,
        vmax=cross_vmax,
    )
    axs[2].set_xlabel("log_brdu_mean")
    axs[2].set_ylabel("log_edu_mean")
    axs[2].set_title("BrdU vs EdU")

    line_brdu = axs[0].axhline(
        thresholds["log_brdu_mean"] if thresholds["log_brdu_mean"] is not None else 0.0,
        color="crimson",
        linestyle="--",
        linewidth=1.0,
        visible=thresholds["log_brdu_mean"] is not None,
    )
    line_edu = axs[1].axhline(
        thresholds["log_edu_mean"] if thresholds["log_edu_mean"] is not None else 0.0,
        color="crimson",
        linestyle="--",
        linewidth=1.0,
        visible=thresholds["log_edu_mean"] is not None,
    )
    line_brdu_cross = axs[2].axvline(
        thresholds["log_brdu_mean"] if thresholds["log_brdu_mean"] is not None else 0.0,
        color="crimson",
        linestyle="--",
        linewidth=1.0,
        visible=thresholds["log_brdu_mean"] is not None,
    )
    line_edu_cross = axs[2].axhline(
        thresholds["log_edu_mean"] if thresholds["log_edu_mean"] is not None else 0.0,
        color="crimson",
        linestyle="--",
        linewidth=1.0,
        visible=thresholds["log_edu_mean"] is not None,
    )

    status = fig.text(0.01, 0.01, "", ha="left", va="bottom", fontsize=10)

    def refresh_status() -> None:
        brdu_thr = thresholds["log_brdu_mean"]
        edu_thr = thresholds["log_edu_mean"]
        brdu_pct = (
            f"{100.0 * float(np.mean(y_brdu_all >= brdu_thr)):.2f}%"
            if brdu_thr is not None
            else "unset"
        )
        edu_pct = f"{100.0 * float(np.mean(y_edu_all >= edu_thr)):.2f}%" if edu_thr is not None else "unset"
        double_pct = (
            f"{100.0 * float(np.mean((y_brdu_all >= brdu_thr) & (y_edu_all >= edu_thr))):.2f}%"
            if brdu_thr is not None and edu_thr is not None
            else "unset"
        )
        status.set_text(
            "log_brdu_mean="
            + (f"{brdu_thr:.4f}" if brdu_thr is not None else "unset")
            + " | log_edu_mean="
            + (f"{edu_thr:.4f}" if edu_thr is not None else "unset")
            + "\nBrdU+="
            + brdu_pct
            + " | EdU+="
            + edu_pct
            + " | Double+="
            + double_pct
        )
        fig.canvas.draw_idle()

    def on_click(event) -> None:
        if event.xdata is None or event.ydata is None:
            return
        if event.inaxes is axs[0]:
            y = float(event.ydata)
            thresholds["log_brdu_mean"] = y
            line_brdu.set_ydata([y, y])
            line_brdu.set_visible(True)
            line_brdu_cross.set_xdata([y, y])
            line_brdu_cross.set_visible(True)
            print(f"Set log_brdu_mean threshold: {y:.6f}")
            refresh_status()
            return
        if event.inaxes is axs[1]:
            y = float(event.ydata)
            thresholds["log_edu_mean"] = y
            line_edu.set_ydata([y, y])
            line_edu.set_visible(True)
            line_edu_cross.set_ydata([y, y])
            line_edu_cross.set_visible(True)
            print(f"Set log_edu_mean threshold: {y:.6f}")
            refresh_status()
            return
        if event.inaxes is axs[2]:
            brdu_y = float(event.xdata)
            edu_y = float(event.ydata)
            thresholds["log_brdu_mean"] = brdu_y
            thresholds["log_edu_mean"] = edu_y
            line_brdu.set_ydata([brdu_y, brdu_y])
            line_brdu.set_visible(True)
            line_edu.set_ydata([edu_y, edu_y])
            line_edu.set_visible(True)
            line_brdu_cross.set_xdata([brdu_y, brdu_y])
            line_brdu_cross.set_visible(True)
            line_edu_cross.set_ydata([edu_y, edu_y])
            line_edu_cross.set_visible(True)
            print(f"Set thresholds from BrdU vs EdU: log_brdu_mean={brdu_y:.6f}, log_edu_mean={edu_y:.6f}")
            refresh_status()

    def on_reset(_event) -> None:
        thresholds["log_brdu_mean"] = None
        thresholds["log_edu_mean"] = None
        line_brdu.set_visible(False)
        line_edu.set_visible(False)
        line_brdu_cross.set_visible(False)
        line_edu_cross.set_visible(False)
        print("Reset thresholds.")
        refresh_status()

    def on_save(_event) -> None:
        if thresholds["log_brdu_mean"] is None or thresholds["log_edu_mean"] is None:
            print("Set both thresholds before saving.")
            return
        payload = {
            "input_h5ad": str(input_h5ad),
            "roi_obs_key": ROI_OBS_KEY if roi_value is not None else None,
            "roi_value": roi_value,
            "n_obs": int(adata_in.n_obs),
            "thresholds": {
                "log_brdu_mean": float(thresholds["log_brdu_mean"]),
                "log_edu_mean": float(thresholds["log_edu_mean"]),
            },
        }
        out_json.write_text(json.dumps(payload, indent=2) + "\n")
        print("Wrote:", out_json)

    fig.canvas.mpl_connect("button_press_event", on_click)

    ax_reset = fig.add_axes([0.79, 0.01, 0.1, 0.06])
    ax_save = fig.add_axes([0.9, 0.01, 0.1, 0.06])
    btn_reset = Button(ax_reset, "Reset")
    btn_save = Button(ax_save, "Save")
    btn_reset.on_clicked(on_reset)
    btn_save.on_clicked(on_save)

    # Keep widget objects alive for `%matplotlib widget`.
    fig._pick_brdu_edu_threshold_handles = (btn_reset, btn_save)  # type: ignore[attr-defined]

    refresh_status()
    plt.tight_layout(rect=[0.0, 0.08, 1.0, 0.95])
    plt.show()


jobs = build_jobs(folder=FOLDER, skip_completed=SKIP_EXISTING_JSON)
print(f"Found jobs: {len(jobs)}")
for job_h5ad, job_subroi, job_out in jobs:
    print(f"  - {job_h5ad.parent.name} subroi={job_subroi} -> {job_out.name}")


# %% [markdown]
# ## Run next job (re-run this cell)
#
# Run one job per execution for reliable widget/button behavior in VS Code/Jupyter.
# If `RUN_SPATIAL_DIAGNOSTICS` is True, spatial diagnostics are shown before threshold picking.
if i >= len(jobs):
    print(f"Done: i={i} >= n_jobs={len(jobs)}")
else:
    input_h5ad, roi_value, out_json = jobs[i]
    adata = ad.read_h5ad(input_h5ad)
    if roi_value is not None:
        mask = adata.obs[ROI_OBS_KEY].astype(str) == str(roi_value)
        adata = adata[mask].copy()

    compute_tricycle_from_log1p(adata, repo_root=Path(__file__).resolve().parents[2])
    initial = load_existing_thresholds(out_json)

    title = (
        f"{input_h5ad.parent.name} | all cells"
        if roi_value is None
        else f"{input_h5ad.parent.name} | {ROI_OBS_KEY}={roi_value}"
    )
    print(f"Job {i + 1}/{len(jobs)}: {title} n_obs={int(adata.n_obs)} out={out_json}")
    if RUN_SPATIAL_DIAGNOSTICS:
        ensure_log_brdu_edu_mean(adata)
        if "spatial" in adata.obsm:
            basis = "spatial"
        elif "spatial_trans" in adata.obsm:
            basis = "spatial_trans"
        else:
            raise KeyError(
                "Missing spatial coordinates: expected `adata.obsm['spatial']` or `adata.obsm['spatial_trans']`."
            )

        brdu_vals = adata.obs["log_brdu_mean"].to_numpy(dtype=float)
        edu_vals = adata.obs["log_edu_mean"].to_numpy(dtype=float)
        vmin_vals = [float(np.nanmedian(brdu_vals)), float(np.nanmedian(edu_vals))]
        vmax_vals = [float(np.nanpercentile(brdu_vals, 99.9)), float(np.nanpercentile(edu_vals, 99.9))]

        fig_diag, axs_diag = plot_embedding(
            adata,
            color=["log_brdu_mean", "log_edu_mean"],
            basis=basis,
            dpi=250,
            figsize=(6, 4),
            s=2,
            cmap="turbo",
            vmin=vmin_vals,
            vmax=vmax_vals,
        )
        for ax in axs_diag:
            ax.invert_yaxis()
            ax.axis("off")
        fig_diag.suptitle(f"Spatial diagnostics ({basis})")
        plt.show()

    pick_thresholds(
        adata_in=adata,
        input_h5ad=input_h5ad,
        out_json=out_json,
        title=title,
        roi_value=roi_value,
        initial=initial,
    )
# i+=1

    # if i < len(jobs):
    #     raise RuntimeError("Re-run this cell to process the next job.")

# %%
# %% [markdown]
# ## Run next job using joined all_progenitors + cNMF usages (re-run this cell)
#
# This keeps the same click UI, but uses:
# - max(Usage_4, Usage_7) vs log_brdu_mean
# - max(Usage_4, Usage_7) vs log_edu_mean
# with an inner join on cell id for the current job's dataset/ROI/subROI.
if i >= len(jobs):
    print(f"Done: i_usage={i} >= n_jobs={len(jobs)}")
else:
    input_h5ad, roi_value, out_json = jobs[i]
    roi_name = input_h5ad.parent.name
    try:
        adata, usage_4, usage_7 = load_joined_progenitors_usage(
            dataset=FOLDER.name,
            roi_name=roi_name,
            roi_value=roi_value,
            roi_obs_key=ROI_OBS_KEY,
            all_h5ad=ALL_PROGENITORS_H5AD,
            usage_df=usage_norm_k9,
        )
        initial = load_existing_thresholds(out_json)

        title = (
            f"{roi_name} | all cells"
            if roi_value is None
            else f"{roi_name} | {ROI_OBS_KEY}={roi_value}"
        )
        print(
            f"Job {i + 1}/{len(jobs)}: {title} "
            f"n_obs={int(adata.n_obs)} out={out_json} source=all_progenitors+usage_norm_k9(inner join)"
        )
        if RUN_SPATIAL_DIAGNOSTICS:
            ensure_log_brdu_edu_mean(adata)
            if "spatial" in adata.obsm:
                basis = "spatial"
            elif "spatial_trans" in adata.obsm:
                basis = "spatial_trans"
            else:
                raise KeyError(
                    "Missing spatial coordinates: expected `adata.obsm['spatial']` or `adata.obsm['spatial_trans']`."
                )

            brdu_vals = adata.obs["log_brdu_mean"].to_numpy(dtype=float)
            edu_vals = adata.obs["log_edu_mean"].to_numpy(dtype=float)
            vmin_vals = [float(np.nanmedian(brdu_vals)), float(np.nanmedian(edu_vals))]
            vmax_vals = [float(np.nanpercentile(brdu_vals, 99.9)), float(np.nanpercentile(edu_vals, 99.9))]

            fig_diag, axs_diag = plot_embedding(
                adata,
                color=["log_brdu_mean", "log_edu_mean"],
                basis=basis,
                dpi=250,
                figsize=(6, 4),
                s=2,
                cmap="turbo",
                vmin=vmin_vals,
                vmax=vmax_vals,
            )
            for ax in axs_diag:
                ax.invert_yaxis()
                ax.axis("off")
            fig_diag.suptitle(f"Spatial diagnostics ({basis})")
            plt.show()

        pick_thresholds_modules(
            adata_in=adata,
            input_h5ad=input_h5ad,
            out_json=out_json,
            title=title,
            roi_value=roi_value,
            initial=initial,
            usage_4_all=usage_4,
            usage_7_all=usage_7,
        )
    except ValueError as e:
        print(f"Skipping job {i + 1}/{len(jobs)} due to error: {e}")

    # i += 1
i += 1
# %%

# %% [markdown]
# ## Optional diagnostics: thresholded cells in spatial space
#
# Run this after setting and saving thresholds for the current job.


ensure_log_brdu_edu_mean(adata)
thresholds = load_existing_thresholds(out_json)
brdu_thr = thresholds["log_brdu_mean"]
edu_thr = thresholds["log_edu_mean"]
if brdu_thr is None or edu_thr is None:
    raise RuntimeError(f"Thresholds are missing in {out_json}. Save thresholds first.")

brdu_pos = adata.obs["log_brdu_mean"].to_numpy(dtype=float) >= float(brdu_thr)
edu_pos = adata.obs["log_edu_mean"].to_numpy(dtype=float) >= float(edu_thr)
adata.obs["thr_brdu_pos"] = brdu_pos.astype(np.int8)
adata.obs["thr_edu_pos"] = edu_pos.astype(np.int8)
adata.obs["thr_double_pos"] = (brdu_pos & edu_pos).astype(np.int8)

if "spatial" in adata.obsm:
    basis = "spatial"
elif "spatial_trans" in adata.obsm:
    basis = "spatial_trans"
else:
    raise KeyError("Missing spatial coordinates: expected `adata.obsm['spatial']` or `adata.obsm['spatial_trans']`.")

fig_thr, axs_thr = plot_embedding(
    adata,
    color=["thr_brdu_pos", "thr_edu_pos"],
    basis=basis,
    dpi=250,
    figsize=(8, 4),
    s=2,
    cmap="Blues",
    vmin=-0.2,
    vmax=1,
)
for ax in axs_thr:
    ax.invert_yaxis()
    ax.axis("off")
n_cells = int(adata.n_obs)
n_brdu = int(brdu_pos.sum())
n_edu = int(edu_pos.sum())
n_double = int((brdu_pos & edu_pos).sum())
fig_thr.suptitle(
    f"Thresholded spatial ({basis}) | BrdU+ {n_brdu}/{n_cells} | EdU+ {n_edu}/{n_cells} | Double+ {n_double}/{n_cells}"
)
plt.show()

# %%
