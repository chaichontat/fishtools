# ruff: noqa
# %%
import os
from collections.abc import Iterable
from pathlib import Path

# os.environ['MATPLOTLIBR']
import anndata as ad
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rapids_singlecell as rsc
import scanpy as sc
import scFates as scf
import seaborn as sns
from loguru import logger

from fishtools.postprocess import auto_translate_groups, normalize_pearson
from fishtools.postprocess.utils_h5ad import read_obs_parquet, read_obsm_h5ad, run_tricycle
from fishtools.utils.plot import add_scale_bar, plot_embedding

#%%
# mpl.rcParams["figure.dpi"] = 300
sns.set_theme()


def add_elpigraph_vz_r(adata: ad.AnnData, elpigraph_vz_dir: Path) -> ad.AnnData:
    """Attach per-cell `r_vz` by matching within dataset/roi/ccf_adjusted on cell geometry."""

    key_cols = ["dataset", "roi", "ccf_adjusted", "x", "y", "z", "area"]
    missing = [col for col in key_cols if col not in adata.obs.columns]
    if missing:
        raise ValueError(f"adata.obs is missing required columns for r_vz join: {missing}")

    lookup_parts: list[pd.DataFrame] = []
    for job_path in sorted(elpigraph_vz_dir.glob("*.job_obs.parquet")):
        metrics_path = elpigraph_vz_dir / job_path.name.replace(".job_obs.parquet", ".elpigraph_vz_metrics.parquet")
        if not metrics_path.exists():
            raise ValueError(f"Missing matching metrics parquet for {job_path.name}")

        job = pd.read_parquet(job_path, columns=[*key_cols, "obs_ix"])
        metrics = pd.read_parquet(metrics_path, columns=["obs_ix", "r_vz"])
        lookup_parts.append(job.merge(metrics, on="obs_ix", how="left", validate="many_to_one")[key_cols + ["r_vz"]])

    lookup = pd.concat(lookup_parts, ignore_index=True)
    if lookup.duplicated(key_cols).any():
        raise ValueError("Duplicate dataset/roi/ccf_adjusted/x/y/z/area keys in elpigraph_vz lookup.")

    obs = adata.obs[key_cols].copy()
    obs["dataset"] = obs["dataset"].astype(str)
    obs["roi"] = obs["roi"].astype(str)
    obs["ccf_adjusted"] = obs["ccf_adjusted"].astype(str)
    joined = obs.merge(lookup, on=key_cols, how="left", validate="many_to_one")
    if joined["r_vz"].isna().any():
        missing_n = int(joined["r_vz"].isna().sum())
        raise ValueError(f"Failed to join r_vz for {missing_n} cells.")

    adata.obs["r_vz"] = joined["r_vz"].to_numpy()
    return adata


def add_dinoinfer_scores(adata: ad.AnnData, dinoinfer_dir: Path) -> ad.AnnData:
    """Join per-dataset Cell-DINO inference parquets onto `adata.obs` by cell id."""

    parquet_paths = sorted(dinoinfer_dir.glob("*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"No dinoinfer parquet files found in {dinoinfer_dir}.")

    parts: list[pd.DataFrame] = []
    required_cols = {"dataset", "roi", "brdu_prob", "edu_prob"}
    for path in parquet_paths:
        frame = pd.read_parquet(path)
        missing = required_cols.difference(frame.columns)
        if missing:
            raise KeyError(f"dinoinfer parquet {path} is missing required columns: {sorted(missing)}")
        frame.index = frame.index.astype(str)
        if frame.index.has_duplicates:
            raise ValueError(f"Duplicate cell ids in dinoinfer parquet: {path}")
        parts.append(frame)

    joined = pd.concat(parts, axis=0, copy=False)
    if joined.index.has_duplicates:
        raise ValueError("Duplicate cell ids across dinoinfer parquet files.")

    obs_keys = adata.obs[["dataset", "roi"]].copy()
    obs_keys["dataset"] = obs_keys["dataset"].astype(str)
    obs_keys["roi"] = obs_keys["roi"].astype(str)
    infer_keys = joined[["dataset", "roi"]].copy()
    infer_keys["dataset"] = infer_keys["dataset"].astype(str)
    infer_keys["roi"] = infer_keys["roi"].astype(str)
    validated = obs_keys.join(
        infer_keys.rename(columns={"dataset": "dataset_dinoinfer", "roi": "roi_dinoinfer"}),
        how="left",
        validate="one_to_one",
    )
    mismatch = validated[
        validated["dataset_dinoinfer"].notna()
        & (
            (validated["dataset"] != validated["dataset_dinoinfer"])
            | (validated["roi"] != validated["roi_dinoinfer"])
        )
    ]
    if len(mismatch) > 0:
        sample = mismatch.index[:3].tolist()
        raise ValueError(f"dinoinfer parquet metadata mismatched for {len(mismatch)} cells, e.g. {sample}")

    score_cols = [col for col in joined.columns if col not in {"dataset", "roi"}]
    adata.obs = adata.obs.drop(columns=score_cols, errors="ignore").join(
        joined[score_cols],
        how="left",
        validate="one_to_one",
    )
    return adata


def add_filtration(adata: ad.AnnData, cluster_params:dict[str, float]|None=None, selected: list[str] | None=None, excluded: list[str] | None = None,custom: dict|None=None) -> ad.AnnData:
    import json
    try:
        parsed: list = json.loads(adata.uns["filtration"])
    except KeyError:
        parsed = []

    if custom:
        parsed.append({"custom": custom})
        adata.uns["filtration"] = json.dumps(parsed)
        return adata

    if sum(x is not None for x in [cluster_params, selected, excluded]) != 1:
        raise ValueError("Must provide exactly one of `cluster_params`, `selected`, or `excluded`.")

    if cluster_params and (sorted(cluster_params.keys()) != sorted(['n_neighbors', 'n_pcs', 'resolution'])):
        raise ValueError("`cluster_params` must have keys: 'n_neighbors', 'n_pcs', 'resolution'.")

    # if selected and (not isinstance(selected, list) or not all(s in adata.obs['leiden'].cat.categories for s in selected)):
    #     raise ValueError("`selected` must be a list of valid Leiden cluster labels in `adata.obs['leiden']`.")


    if cluster_params:
        if parsed and isinstance(parsed[-1], dict) and sorted(parsed[-1].keys()) == sorted(['n_neighbors', 'n_pcs', 'resolution']):
            parsed[-1] = cluster_params
        else:
            parsed.append(cluster_params)
    elif excluded is not None:
        parsed.append({"excluded": excluded})
    else:
        parsed.append({"selected": selected})
    adata.uns["filtration"] = json.dumps(parsed)
    return adata

def filter_(adata, *,include=None, exclude=None,write=True):
    if include and exclude:
        raise ValueError("Cannot specify both `include` and `exclude`.")
    if include:
        mask = adata.obs['leiden'].isin(list(map(str, include)))
    elif exclude:
        mask = ~adata.obs['leiden'].isin(list(map(str, exclude)))
    else:
        raise ValueError("Must specify either `include` or `exclude`.")
    if write:
        add_filtration(adata, selected=include, excluded=exclude)
    return adata[mask]


def plot_clusters(adata: ad.AnnData, color:list, basis: str = "umap", ) -> None:
    clusters = (
        list(adata.obs["leiden"].cat.categories)
        if hasattr(adata.obs["leiden"], "cat")
        else sorted(adata.obs["leiden"].unique(), key=str)
    )

    cluster_cell_types = {
        str(leiden): cell_type
        for leiden, cell_type in adata.obs.groupby("leiden")["leiden_cell_type"].first().items()
    }

    import matplotlib.patheffects as pe

    embedding = adata.obsm[f"X_{basis}"]
    axs = sc.pl.embedding(
        adata,
        basis=basis,
        color=color,
        legend_loc=None,
        title="Leiden + Gemini cell type",
        show=False,
        cmap="CMRmap_r",
        ncols=2,
    )
    for cluster in clusters:
        cluster_mask = (adata.obs["leiden"].astype(str) == str(cluster)).to_numpy()
        if cluster_mask.sum() == 0:
            continue
        x, y = np.nanmedian(embedding[cluster_mask], axis=0)
        cell_type = cluster_cell_types.get(str(cluster), "unassigned")
        ax=axs[0]
        ax.text(
            x,
            y,
            f"{cluster}: {cell_type}",
            ha="center",
            va="center",
            fontsize=4,
            fontweight="bold",
            path_effects=[pe.withStroke(linewidth=1, foreground="white")],
        )
    return axs

#%%

# %%
# Input: raw-count filtered h5ad generated by `concat_copy copy.py`.
in_h5ad = Path.home() / "nvme" / "vzsvz.h5ad"
obs_parquet = Path.home() / "nvme" / "obs.parquet"
obsm_h5ad = Path.home() / "nvme" / "all.fresh.obsm_only.h5ad"
manual_layers_parquet = Path.home() / "nvme" / "all.mclust_manual_boundaries" / "manual_layers.parquet"
elpigraph_vz_dir = Path.home() / "nvme" / "all.elpigraph_vz"
cnmf_usage_parquet = Path.home() / "nvme" / "cnmf_all_progenitors_all" / "usage_norm.k8.dt0.1.parquet"
dinoinfer_dir = Path.home() / "nvme" / "dinoinfer"

print(f"Loading: {in_h5ad}")
adata = sc.read_h5ad(in_h5ad)
#%%
read_obs_parquet(obs_parquet, adata=adata)
read_obsm_h5ad(obsm_h5ad, adata=adata)
manual_layers = pd.read_parquet(manual_layers_parquet, columns=["index", "manual_layer"]).set_index("index")
adata.obs = adata.obs.drop(columns=["manual_layer"], errors="ignore").join(
    manual_layers,
    how="left",
    validate="one_to_one",
)
cnmf_usage = pd.read_parquet(cnmf_usage_parquet).set_index("index")
cnmf_usage.index = cnmf_usage.index.astype(str)
cnmf_usage = cnmf_usage[[col for col in cnmf_usage.columns if col.startswith("Usage_")]]
adata.obs = adata.obs.drop(columns=[x for x in adata.obs.columns if x.startswith("Usage_")], errors="ignore").join(
    cnmf_usage,
    how="left",
    validate="one_to_one",
)
adata = add_dinoinfer_scores(adata, dinoinfer_dir)
adata = add_elpigraph_vz_r(adata, elpigraph_vz_dir)
#%%
adata.obs["log_edu_mean"] = np.log(adata.obs["edu_mean"] + 1)
adata.obs["log_brdu_mean"] = np.log(adata.obs["brdu_mean"] + 1)
# %% Tricycle (cell-cycle) projection
if 1:
    repo_root = Path(__file__).resolve().parents[2]
    trc = pd.read_csv(repo_root / "neuroRef.csv")
    adata=run_tricycle(adata, trc, layer="raw_sct_corrected",center_on=adata[adata.obs["manual_layer"].isin(["1", "5"])].obs_names)

#%%
adata=adata[adata.obs["leiden"].isin(["5", "6", "7"])]

#%%
kmeans = pd.read_parquet("/home/chaichontat/nvme/cnmf_all_neurons/kmeans_cells.k8_on_usage.k11.parquet").set_index("index")
adata.obs = adata.obs.drop(columns=["kmeans_k8_on_usage_k11_cluster"], errors="ignore")
adata.obs = adata.obs.join(
    kmeans[["kmeans_k8_on_usage_k11_cluster"]],
    how="left",
    validate="one_to_one",
)
kmeans = kmeans.drop(columns=["kmeans_k8_on_usage_k11_cluster"], errors="ignore")
adata.obs["kmeans_k8_on_usage_k11_cluster"] = adata.obs["kmeans_k8_on_usage_k11_cluster"].astype(str).astype("category")
adata.obs = adata.obs.join(kmeans, how="left", validate="one_to_one")
#%%


sc.pl.embedding(
    adata,
    basis="umap",
    color=['Eomes', 'leiden']
)

#%%
adata = adata[adata.obs["manual_layer"].isin([ "1","5"]) & adata.obs["leiden"].isin(["6", "5", "7"])]
rsc.tl.umap(adata, min_dist=0.4, n_components=2, random_state=0)
#%%
mask = adata.obs["brdu_pos"].eq(1) & adata.obs["edu_pos"].eq(0) & ~adata.obs['leiden'].isin(["7"])
plt.scatter(
    adata.obs.loc[mask, "tricycle"],
    adata.obs.loc[mask, "r_vz"],
    s=0.1,
    alpha=0.5,
)

#%%

#%%
adata.obs['vz_pseudo'] = adata.obs['Usage_7'] / (adata.obs['Usage_1'] + adata.obs['Usage_7'])

# adata =filter_(adata, include=[5,6,7,8,9,10],write=True)
# sc.tl.leiden(adata, n_iterations=2, resolution=0.2,  flavor="igraph")
# rsc.tl.umap(adata, min_dist=0.2, n_components=2, random_state=0)
sc.pl.embedding(
    adata,
    # adata[adata.obs['leiden'].isin(['5','6','7'])],
    color=["leiden", 'tricycle', "Usage_1", "Usage_7","vz_pseudo", "Notch2", "Hes5", "manual_layer"],
    basis="umap",
    # groups=['13', '7', '4'],
    ncols=3,
    # vmax=5,
    cmap="CMRmap_r",
    legend_loc="on data",
    legend_fontsize=8,
)  # type: ignore
#%%

plot_embedding(adata,color=["edu_pos", "brdu_pos", "Usage_3"], cmap="CMRmap_r", basis="umap")

#%%
plot_embedding(adata[adata.obs["manual_layer"] == "5"],color=[*[f"Usage_{i+1}" for i in range(8)], "manual_layer"], basis="umap", cmap="CMRmap_r")

#%% Usage programs vs r_vz
def plot_usage_vs_r_vz(
    usage_rvz: pd.DataFrame,
    usage_cols: list[str],
    *,
    title: str,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot faceted usage-vs-r_vz hexbin panels for a set of usage programs."""
    if not usage_cols:
        raise ValueError("Expected at least one usage column to plot.")
    if usage_rvz.empty:
        raise ValueError("No finite `r_vz`/usage values available for plotting.")

    rvz_min = float(usage_rvz["r_vz"].min())
    rvz_max = float(usage_rvz["r_vz"].max())
    if rvz_min >= rvz_max:
        raise ValueError(f"Expected `r_vz` to span a range for plotting, got min=max={rvz_min}.")

    usage_max = float(usage_rvz[usage_cols].to_numpy().max())
    ncols = min(3, len(usage_cols))
    nrows = int(np.ceil(len(usage_cols) / ncols))
    fig, axs = plt.subplots(
        nrows,
        ncols,
        figsize=(4 * ncols, 3 * nrows),
        dpi=200,
        sharex=True,
        sharey=True,
        facecolor="white",
    )
    axs = np.atleast_1d(axs).ravel()

    for ax, usage_col in zip(axs, usage_cols, strict=False):
        ax.scatter(
            usage_rvz["r_vz"],
            usage_rvz[usage_col],
            s=0.1,
            alpha=0.4,
            linewidths=0,
            rasterized=True,
            color="black",
        )
        ax.set_title(usage_col, color="black")
        ax.grid(True, alpha=0.2)
        ax.set_xlim(rvz_min, rvz_max)
        ax.set_ylim(0, usage_max)

    for ax in axs[len(usage_cols) :]:
        ax.set_visible(False)

    for ax in axs[-ncols:]:
        if ax.get_visible():
            ax.set_xlabel("r_vz", color="black")
    for ax in axs[::ncols]:
        if ax.get_visible():
            ax.set_ylabel("Usage", color="black")

    fig.suptitle(title, color="black")
    plt.tight_layout()
    return fig, axs


usage_cols = sorted(
    [col for col in adata.obs.columns if col.startswith("Usage_")],
    key=lambda col: int(col.split("_", maxsplit=1)[1]),
)
if not usage_cols:
    raise ValueError("Expected at least one `Usage_*` column in `adata.obs`.")

usage_rvz = adata[adata.obs["manual_layer"] == "1"].obs[["r_vz", *usage_cols]].apply(pd.to_numeric, errors="coerce").dropna()
plot_usage_vs_r_vz(usage_rvz, usage_cols, title="Usage programs vs r_vz")

excluded_usage_cols = ["Usage_5", "Usage_8"]
missing_excluded = [col for col in excluded_usage_cols if col not in usage_cols]
if missing_excluded:
    raise ValueError(f"Missing usage columns requested for exclusion: {missing_excluded}")

renorm_usage_cols = [col for col in usage_cols if col not in excluded_usage_cols]
renorm_usage_rvz = usage_rvz[["r_vz", *renorm_usage_cols]].copy()
renorm_denominator = renorm_usage_rvz[renorm_usage_cols].sum(axis=1)
positive_mask = renorm_denominator > 0
if not positive_mask.any():
    raise ValueError("No cells have positive renormalized usage mass after excluding `Usage_5` and `Usage_8`.")

renorm_usage_rvz = renorm_usage_rvz.loc[positive_mask].copy()
renorm_usage_rvz.loc[:, renorm_usage_cols] = renorm_usage_rvz[renorm_usage_cols].div(
    renorm_denominator.loc[positive_mask],
    axis=0,
)
plot_usage_vs_r_vz(
    renorm_usage_rvz,
    renorm_usage_cols,
    title="Usage programs vs r_vz (renormalized without Usage_5/Usage_8)",
)


#%%
facet_usage_cols = sorted(
    [col for col in adata.obs.columns if col.startswith("Usage_")],
    key=lambda col: int(col.split("_", maxsplit=1)[1]),
)
if not facet_usage_cols:
    raise ValueError("Expected at least one `Usage_*` column in `adata.obs`.")

hist_df = adata.obs.loc[:, ["manual_layer", *facet_usage_cols]].copy()
manual_layer_num = pd.to_numeric(hist_df["manual_layer"], errors="coerce")
hist_df = hist_df.loc[manual_layer_num.isin([1, 5])].copy()
hist_df["manual_layer"] = manual_layer_num.loc[hist_df.index].astype("Int64").astype(str)

layer_counts = hist_df["manual_layer"].value_counts()
if "1" not in layer_counts or "5" not in layer_counts:
    raise ValueError(f"Expected both manual layers 1 and 5 in data, got counts={layer_counts.to_dict()}")

hist_long = hist_df.melt(
    id_vars="manual_layer",
    value_vars=facet_usage_cols,
    var_name="program",
    value_name="usage",
)
hist_long["usage"] = pd.to_numeric(hist_long["usage"], errors="coerce")
hist_long = hist_long.dropna(subset=["usage"])
if hist_long.empty:
    raise ValueError("No finite `Usage_*` values found for manual_layer 1/5 histogram plotting.")

g = sns.displot(
    data=hist_long,
    x="usage",
    hue="manual_layer",
    hue_order=["1", "5"],
    palette={"1": "tab:blue", "5": "tab:orange"},
    col="program",
    col_wrap=3,
    kind="hist",
    bins=80,
    element="step",
    fill=False,
    linewidth=1.2,
    stat="density",
    common_norm=False,
    facet_kws={"sharex": False, "sharey": False},
    height=2.6,
    aspect=1.2,
)
g.set_axis_labels("Usage", "Density")
g.set_titles("{col_name}")
if g._legend is not None:
    g._legend.set_title("manual_layer")
plt.tight_layout()


#%% VZ tricycle plotting
layer1_adata = adata[adata.obs["manual_layer"].isin(["1"]) & adata.obs["leiden"].isin(["1", "6", "5", "7"])]
#%%
layer15 = adata[adata.obs["manual_layer"].isin([ "1","5"])]
#%%
facet_masks = [
    ("EdU only", layer1_adata.obs["edu_pos"] & ~layer1_adata.obs["brdu_pos"]),
    ("Dual pulse", layer1_adata.obs["edu_pos"] & layer1_adata.obs["brdu_pos"]),
    ("BrdU only", ~layer1_adata.obs["edu_pos"] & layer1_adata.obs["brdu_pos"]),
    ("Dual neg", ~layer1_adata.obs["edu_pos"] & ~layer1_adata.obs["brdu_pos"]),
]

tricycle_min = float(layer1_adata.obs["tricycle"].min())
tricycle_max = float(layer1_adata.obs["tricycle"].max())

fig, axs = plt.subplots(2, 2, figsize=(10, 5), dpi=200, sharex=True, sharey=True, facecolor="white")
for ax, (title, mask) in zip(axs.flat, facet_masks, strict=True):
    subset = layer1_adata[np.asarray(mask)]
    ax.set_facecolor("black")
    ax.hexbin(
        subset.obs["tricycle"],
        subset.obs["r_vz"],
        gridsize=200,
        cmap="viridis",
        mincnt=1,
        linewidths=0,
    )
    ax.set_title(title, color="black")
    ax.set_xlim(tricycle_min, tricycle_max)
    ax.set_ylim(0, 50)
    ax.tick_params(colors="black")
    for spine in ax.spines.values():
        spine.set_color("black")

for ax in axs[-1]:
    ax.set_xlabel("tricycle", color="black")
for ax in axs[:, 0]:
    ax.set_ylabel("r_vz", color="black")



#%%
# a=adata[adata.obs['brdu_pos']&~adata.obs['edu_pos']]
# adata=adata[adata.obs['brdu_pos'] | adata.obs['edu_pos']]
# %%
# Keep behavior consistent with `concat_copy copy.py`, but don't double-translate if already done.
if "spatial_trans" not in adata.obsm and "spatial" in adata.obsm:
    auto_translate_groups(adata, padding=500.0)
#%%

# %% 3D UMAP (interactive via plotly)
import plotly.graph_objects as go
import rapids_singlecell as rsc

rsc.tl.umap(adata, min_dist=0.2, n_components=3, random_state=0)
umap3d = adata.obsm["X_umap"]
#%%
leiden_labels = adata.obs["leiden"].astype(str).tolist()

unique_clusters = sorted(set(leiden_labels), key=int)
colors = plt.colormaps["tab20"](np.linspace(0, 1, len(unique_clusters)))
color_map = {c: f"rgb({int(r*255)},{int(g*255)},{int(b*255)})" for c, (r, g, b, _) in zip(unique_clusters, colors)}

def make_3d_umap(adata: ad.AnnData, color: str, height: int = 900) -> go.Figure:
    """
    Plot 3D UMAP colored by a column in adata.obs or a gene in adata.var_names.
    """
    umap3d = adata.obsm["X_umap"]
    if color in adata.obs.columns:
        values = adata.obs[color]
        is_categorical = hasattr(values, "cat") or values.dtype == object
        if is_categorical:
            labels = values.astype(str).tolist()
            unique = sorted(set(labels), key=lambda x: (int(x) if x.isdigit() else x))
            cmap = plt.colormaps["tab20"](np.linspace(0, 1, len(unique)))
            cmap_dict = {c: f"rgb({int(r*255)},{int(g*255)},{int(b*255)})" for c, (r, g, b, _) in zip(unique, cmap)}
            traces = []
            for cluster in unique:
                mask = np.array(labels) == cluster
                traces.append(
                    go.Scatter3d(
                        x=umap3d[mask, 0],
                        y=umap3d[mask, 1],
                        z=umap3d[mask, 2],
                        mode="markers",
                        name=cluster,
                        marker=dict(size=1, color=cmap_dict[cluster], opacity=0.7),
                        hovertemplate=f"<b>{cluster}</b><br>x=%{{x:.2f}}, y=%{{y:.2f}}, z=%{{z:.2f}}<extra></extra>",
                    )
                )
        else:
            vals = values.to_numpy().astype(float)
            traces = [go.Scatter3d(
                x=umap3d[:, 0], y=umap3d[:, 1], z=umap3d[:, 2],
                mode="markers",
                marker=dict(size=1, color=vals, colorscale="turbo", opacity=0.7,
                            colorbar=dict(title=color, thickness=15)),
                hovertemplate=f"{color}=%{{marker.color:.3f}}<br>x=%{{x:.2f}}, y=%{{y:.2f}}, z=%{{z:.2f}}<extra></extra>",
            )]
    elif color in adata.var_names:
        from scipy.sparse import issparse
        gene_idx = adata.var_names.get_loc(color)
        X = adata.X
        vals = (X[:, gene_idx].toarray().flatten() if issparse(X) else np.asarray(X[:, gene_idx]).flatten())
        traces = [go.Scatter3d(
            x=umap3d[:, 0], y=umap3d[:, 1], z=umap3d[:, 2],
            mode="markers",
            marker=dict(size=1, color=vals, colorscale="turbo", opacity=0.7,
                        colorbar=dict(title=color, thickness=15)),
            hovertemplate=f"{color}=%{{marker.color:.3f}}<br>x=%{{x:.2f}}, y=%{{y:.2f}}, z=%{{z:.2f}}<extra></extra>",
        )]
    else:
        raise KeyError(f"{color!r} not found in adata.obs or adata.var_names.")

    return go.Figure(
        data=traces,
        layout=go.Layout(
            title=f"3D UMAP — {color}",
            height=height,
            scene=dict(
                xaxis_title="UMAP1",
                yaxis_title="UMAP2",
                zaxis_title="UMAP3",
                bgcolor="black",
                xaxis=dict(color="white", gridcolor="#333"),
                yaxis=dict(color="white", gridcolor="#333"),
                zaxis=dict(color="white", gridcolor="#333"),
            ),
            paper_bgcolor="black",
            font_color="white",
            legend=dict(font=dict(size=9)),
            margin=dict(l=0, r=0, b=0, t=40),
        ),
    )

from scipy.sparse import issparse

make_3d_umap(adata, "tricycle", height=1200).show()





# %%
# adata=adata[adata.obs['r_um'] < 350]
adata.X = adata.layers["raw"]
adata, plot = normalize_pearson(adata, batch_key="dataset")
#%%
import plotly.graph_objects as go

umap = adata.obsm['X_umap']
categories = adata.obs['leiden'].cat.categories
palette = sns.color_palette("tab20", len(categories))
color_map = {cat: f"rgb({int(r*255)},{int(g*255)},{int(b*255)})" for cat, (r, g, b) in zip(categories, palette)}

fig = go.Figure()
for cat in categories:
    mask = adata.obs['leiden'].values == cat
    fig.add_trace(go.Scatter3d(
        x=umap[mask, 0],
        y=umap[mask, 1],
        z=umap[mask, 2],
        mode='markers',
        name=str(cat),
        marker=dict(size=2, color=color_map[cat], opacity=0.5),
    ))

fig.update_layout(
    scene=dict(xaxis_title='UMAP1', yaxis_title='UMAP2', zaxis_title='UMAP3'),
    title='UMAP colored by Leiden',
    legend=dict(itemsizing='constant'),
)
fig.show()
#%%
plot_clusters(adata,color=["leiden", "Eomes"])


# %%

fig, axs = plot_embedding(
    adata,
    color=["edu_mean", "brdu_mean", "pi_mean"],
    basis="spatial_trans",
    dpi=300,
    figsize=(10, 6),
    s=5,
    cmap="Blues",
)
for ax in axs:
    ax.invert_yaxis()
axs[0].axis("off")
add_scale_bar(axs[0], 1000 / 0.216, "1000 μm")

# %%
import rapids_singlecell as rsc


def std_umap(adata: ad.AnnData,n_neighbors:int=20, n_pcs:int=20, resolution:float=0.8) -> ad.AnnData:

    non_finite = ~np.isfinite(adata.X)
    if bool(non_finite.any()):
        n_bad = int(non_finite.sum())
        n_bad_cells = int(non_finite.any(axis=1).sum())
        n_bad_genes = int(non_finite.any(axis=0).sum())
        print(
            "Found non-finite values in `adata.X` before PCA; replacing with 0. "
            f"(entries={n_bad}, cells={n_bad_cells}, genes={n_bad_genes})"
        )
        adata.X = np.nan_to_num(adata.X, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
        if not np.isfinite(adata.X).all():
            raise ValueError("`adata.X` still contains non-finite values after nan_to_num().")

    # cuML PCA can fail to converge if some features have (near-)zero variance.
    gene_var = np.var(adata.X, axis=0)
    keep_genes = gene_var > 1e-8
    if int(np.sum(keep_genes)) < 2:
        raise ValueError(f"Too few variable genes for PCA after variance filter: kept={int(np.sum(keep_genes))}")
    adata = adata[:, keep_genes]

    n_comps = min(50, adata.n_obs - 1, adata.n_vars - 1)
    if n_comps < 2:
        raise ValueError(f"Not enough cells/genes for PCA: n_obs={adata.n_obs}, n_vars={adata.n_vars}")
    rsc.tl.pca(adata, n_comps=int(n_comps))
    sc.pl.pca_variance_ratio(adata, log=True)

    rsc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs, metric="cosine", random_state=12)
    sc.tl.leiden(adata, n_iterations=2, resolution=resolution, flavor="igraph")
    rsc.tl.umap(adata, min_dist=0.1, n_components=2, random_state=0)
    adata = add_filtration(adata, cluster_params={"n_neighbors": n_neighbors, "n_pcs": n_pcs, "resolution": resolution})

    return adata

adata=std_umap(adata, n_neighbors=25, n_pcs=18, resolution=0.6)
#%%
adata.obs['brduedu'] = 'neg'
adata.obs.loc[adata.obs['edu_pos'] & ~adata.obs['brdu_pos'], 'brduedu'] = 'edu'
adata.obs.loc[adata.obs['brdu_pos'] & ~adata.obs['edu_pos'], 'brduedu'] = 'brdu'
adata.obs.loc[adata.obs['brdu_pos'] & adata.obs['edu_pos'], 'brduedu'] = 'dual'
# %%
sc.pl.embedding(
    adata,
    color=["leiden", "Eomes", "Tnc", "Ptprz1"],
    basis="umap",
    # groups=['13', '7', '4'],
    ncols=3,
    # vmax=5,
    cmap="CMRmap_r",
    legend_loc="on data",
    legend_fontsize=8,
)  # type: ignore
#%%
sc.pl.embedding(
    adata,
    color=["leiden", "manual_layer"],
    basis="umap",
    # vmax=5,
    # groups=['edu'],
    ncols=3,
    cmap="turbo",
    # legend_loc="on data",
    legend_fontsize=8,
)  # type: ignore

#%%

adata=adata[adata.obs['leiden'].isin(list(map(str, range(5))))]

if "leiden" not in adata.obs.columns or "manual_layer" not in adata.obs.columns:
    raise ValueError("Expected `leiden` and `manual_layer` in `adata.obs`.")

leiden_groups = (
    list(adata.obs["leiden"].cat.categories)
    if hasattr(adata.obs["leiden"], "cat")
    else sorted(adata.obs["leiden"].astype(str).unique(), key=lambda x: int(x) if x.isdigit() else x)
)

invalid: dict[str, list[str]] = {}
for leiden in leiden_groups:
    subset = adata[adata.obs["leiden"].astype(str) == str(leiden)]
    present_layers = sorted(pd.unique(subset.obs["manual_layer"].astype(str)))
    if len(present_layers) < 2:
        invalid[str(leiden)] = present_layers

if invalid:
    raise ValueError(
        f"Cannot run DE by `manual_layer` for these leiden clusters (need >=2 layers): {invalid}"
    )

de_manual_layer_by_leiden: dict[str, dict[str, pd.DataFrame]] = {}
de_manual_layer_tables: list[pd.DataFrame] = []

for leiden in leiden_groups:
    subset = adata[adata.obs["leiden"].astype(str) == str(leiden)].copy()
    subset.obs["manual_layer"] = subset.obs["manual_layer"].astype(str).astype("category")

    sc.tl.rank_genes_groups(
        subset,
        groupby="manual_layer",
        method="wilcoxon",
        use_raw=False,
    )
    from fishtools.postprocess import plot_ranked_genes
    plot_ranked_genes(subset)
    plt.show()

    de_manual_layer_by_leiden[str(leiden)] = {}
    for layer in subset.obs["manual_layer"].cat.categories:
        table = sc.get.rank_genes_groups_df(subset, group=str(layer)).copy()
        table.insert(0, "manual_layer", str(layer))
        table.insert(0, "leiden", str(leiden))
        de_manual_layer_by_leiden[str(leiden)][str(layer)] = table
        de_manual_layer_tables.append(table)

de_manual_layer = pd.concat(de_manual_layer_tables, ignore_index=True)
# #
# from itertools import batched

# clusters = [str(c) for c in adata.obs["leiden"].cat.categories]
# clusters_per_fig = 3
# panel_inches = 5.5
# for pair in batched(clusters, clusters_per_fig):
#     fig, axs = plt.subplots(
#         1,
#         len(pair),
#         figsize=(panel_inches * len(pair), panel_inches),
#         dpi=300,
#         constrained_layout=True,
#     )
#     axs = np.atleast_1d(axs)
#     for ax, cluster in zip(axs, pair):
#         sc.pl.embedding(
#             adata,
#             color=["leiden"],
#             basis="spatial_trans",
#             groups=[cluster],
#             ax=ax,
#             show=False,
#             s=0.5,
#             legend_loc=None,
#             colorbar_loc=None,
#             frameon=False,
#         )  # type: ignoreZZ
#         ax.invert_yaxis()
#         ax.set_aspect("equal", adjustable="box")
#         ax.axis("off")
#         ax.set_title(f"Leiden {cluster}")
#         add_scale_bar(ax, 1000 / 0.216, "1000 μm")
#     plt.show()
#%%

def ask_gemini_cell_type_from_markers(
    adata: ad.AnnData,
    *,
    clusters: Iterable[str | int] | None = None,
    groupby: str = "leiden",
    n_genes: int = 10,
    model: str = "gemini-3-flash-preview",
    client: object | None = None,
) -> dict[str, str]:
    """Ask Gemini to name likely cell types for Leiden clusters."""

    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)

    import json

    from google import genai

    if clusters is None:
        clusters = (
            list(adata.obs[groupby].cat.categories)
            if hasattr(adata.obs[groupby], "cat")
            else sorted(adata.obs[groupby].unique(), key=str)
        )

    if "rank_genes_groups" not in adata.uns:
        sc.tl.rank_genes_groups(adata, groupby=groupby, method="wilcoxon")

    cluster_lines: list[str] = []
    for cluster in clusters:
        ranked = sc.get.rank_genes_groups_df(adata, group=str(cluster))
        genes = ranked["names"].head(n_genes).tolist()
        cluster_lines.append(f"{cluster}: {', '.join(map(str, genes))}")

    prompt = (
        "Please help annotate these single-cell transcriptomic clusters of E15.5 embryonic mouse cortex.\n"
        "For each cluster, return only a short cell type/state label (2-6 words).\n"
        "Use this exact JSON format (no markdown):\n"
        "{\"<cluster>\": \"<label>\", ...}\n\n"
        f"Top {n_genes} marker genes by cluster:\n"
        + "\n".join(cluster_lines)
    )

    if client is None:
        client = genai.Client()

    stream = client.models.generate_content_stream(model=model, contents=[prompt])
    chunks: list[str] = []
    for chunk in stream:
        chunk_text = getattr(chunk, "text", None)
        if chunk_text:
            logger.opt(raw=True).info(chunk_text)
            chunks.append(chunk_text)

    response_text = "".join(chunks)
    annotations = json.loads(response_text)

    return {str(cluster): str(label) for cluster, label in annotations.items()}

sc.tl.rank_genes_groups(adata, groupby="leiden", method="wilcoxon")
#%%
from fishtools.postprocess import plot_ranked_genes

plot_ranked_genes(adata)

#%%
cluster_cell_types = ask_gemini_cell_type_from_markers(adata)

adata.obs["leiden_cell_type"] = adata.obs["leiden"].astype(str).map(cluster_cell_types)
#%%
plot_clusters(adata, color=["leiden", "manual_layer", "r_vz", "tricycle", "dual_pos", "Eomes"], basis="umap")
plt.show()
#%%

adata.obs['dual_pos'] = (adata.obs['brdu_pos'] & adata.obs['edu_pos'])
adata.obs['brdu_only'] = adata.obs['brdu_pos'] & ~adata.obs['edu_pos']
adata.obs['edu_only'] = adata.obs['edu_pos'] & ~adata.obs['brdu_pos']

#%%

(
    adata.obs[["leiden", "leiden_cell_type"]]
    .astype({"leiden": str, "leiden_cell_type": str})
    .drop_duplicates()
    .sort_values("leiden", key=lambda s: pd.to_numeric(s))
    .reset_index(drop=True)
)

#%%
plot_embedding(
    adata,
    basis="umap",
    color=[ "edu_only", "r_um", "Pax6", "Dcx", "Eomes", "Tubb3"],
    cmap="turbo",
)
# "leiden", "ap", "ml", "tricycle",
#%%


plot_clusters(adata, color=["leiden", "Pax6", "Eomes", "Vim", "Tnc", "ap", "ml", "r_um", "tricycle"], basis="umap")

#%%
for i in range(5):
    plot_embedding(
        adata,
        basis="AP_ML_um_jitter",
        color=["leiden"],
        groups=[str(2*i), str(2*i+1)],
        title=f"Leiden {2*i}+{2*i+1}",
    )
    plt.show()
#%%
# adata[~adata.obs['leiden'].isin(["2"])]
adata_sub=adata[~adata.obs['leiden'].isin(list(map(str, [2,10,12])))]
adata_sub.X = adata_sub.layers['raw']
#%%
adata=sc.read_h5ad(Path.home() / "nvme"/"all_excit.h5ad")
#%%

plot_embedding(adata[adata.obs['manual_layer']=='1'],basis='spatial_trans', color=['edu_pos', 'brdu_pos', 'r_um'], cmap='turbo')
plt.savefig(Path.home() / "nvme" / "spatial_edu_brdu.png", dpi=300, bbox_inches="tight")

#%%

adata.obs['dataset_roi'] = adata.obs['dataset'].astype(str) + "\n" + adata.obs['roi'].astype(str) + " " + adata.obs['ccf_adjusted'].astype(str)
#%%
sc.pl.embedding(
    adata,
    basis="spatial_trans",
    color=["dataset_roi", "r_um"],
    legend_loc="on data",

    alpha=0.7,
    legend_fontsize=3,
)
#%%

# %% Tricycle plots
import colorcet  # noqa: F401

sc.pl.embedding(
    adata,
    color=["tricycle"],
    basis="umap",
    ncols=3,
    s=0.2,
    frameon=False,
    cmap="cet_colorwheel",
)

fig, axs = plot_embedding(
    adata,
    color=["log_edu_mean"],
    basis="tricycle",
    cmap="CMRmap_r",
    return_fig=True,
    dpi=300,
)
for ax in axs:
    ax.set_aspect("equal")


# %%
top_n = 10
for cluster in adata.obs["leiden"].cat.categories:
    genes = sc.get.rank_genes_groups_df(adata, group=cluster).head(top_n)["names"].tolist()
    print(f"Cluster {cluster}: {', '.join(genes)}")

# %%
top_k = 3
for cluster in adata.obs["leiden"].cat.categories:
    print(f"Cluster {cluster}")
    genes = sc.get.rank_genes_groups_df(adata, group=cluster).head(top_k)["names"].tolist()
    sc.pl.umap(
        adata,
        color=genes,
        legend_loc="on data",
        frameon=False,
        ncols=3,
    )
    plt.show()
# %%

plot_embedding(
    adata,
    color=["leiden", "Pax6", "Tnc", "Eomes", "brdu_pos", "edu_pos"],
    basis="umap",
    dpi=300,
    figsize=(12,8),
    s=0.2,
    alpha=0.5,
    cmap="CMRmap_r",
)
#%%


# %%
_ap_ml_um = np.asarray(adata.obsm["AP_ML_um"])
adata.obsm["AP_ML_um_jitter"] = _ap_ml_um[:, [1,0]] + np.random.normal(loc=0.0, scale=50.0, size=_ap_ml_um.shape)
#%%
adata.obs['ap'] = adata.obsm['AP_ML_um'][:, 1]
adata.obs['ml'] = adata.obsm['AP_ML_um'][:, 0]

# %%
import scFates as scf

scf.tl.curve(adata,Nodes=30,use_rep="X_pca",ndims_rep=2, epg_lambda=0.1, epg_mu=0.1)
# %%
scf.pl.graph(adata,basis="umap")
# %%
adata.obs_names_make_unique()
scf.tl.root(adata,"Pax6")
# %%
scf.tl.pseudotime(adata,n_jobs=32,n_map=32,seed=42)
# %%
sc.pl.umap(adata,color="t")
# %%
adata.write_h5ad(Path.home() / "nvme" / "pseudotime.h5ad")


# %%

# %% ElPiGraph principal curve on cluster 2 (first ROI of the first dataset only)
import os

import elpigraph

# _numba_cache_dir = Path("/tmp/numba_cache_elpigraph")
# _numba_cache_dir.mkdir(parents=True, exist_ok=True)
# os.environ.setdefault("NUMBA_CACHE_DIR", str(_numba_cache_dir))


elpigraph_cluster_key = "leiden"
elpigraph_cluster_value = "1"
elpigraph_num_nodes = 45
elpigraph_n_cores = 24
elpigraph_lambda = 0.01
elpigraph_mu = 0.1

xy_all = np.asarray(adata.obsm["spatial"])
if xy_all.ndim != 2 or xy_all.shape[1] < 2:
    raise ValueError(f"Expected adata.obsm['spatial'] to be (n,2+); got shape={xy_all.shape}.")
xy_all = xy_all[:, :2]

datasets = sorted(pd.unique(adata.obs["dataset"].astype(str)))
if not datasets:
    raise ValueError("ElPiGraph: no datasets found in `adata.obs['dataset']`.")

dataset = datasets[0]
ds_mask = adata.obs["dataset"].astype(str) == dataset
ds_rois = sorted(pd.unique(adata.obs.loc[ds_mask, "roi"].astype(str)))
if not ds_rois:
    raise ValueError(f"ElPiGraph: dataset={dataset} has no ROIs.")

roi = ds_rois[0]
mask = (
    ds_mask
    & (adata.obs["roi"].astype(str) == roi)
    & (adata.obs[elpigraph_cluster_key].astype(str) == elpigraph_cluster_value)
)
n = int(mask.sum())
if n < 50:
    raise ValueError(f"ElPiGraph: dataset={dataset}, roi={roi} has only n={n} cells in cluster={elpigraph_cluster_value}.")

X = xy_all[mask.to_numpy()]
print(f"ElPiGraph: fitting curve (n={n}) for dataset={dataset}, roi={roi}, cluster={elpigraph_cluster_value}")
res = elpigraph.computeElasticPrincipalCurve(
    X,
    NumNodes=elpigraph_num_nodes,
    Do_PCA=False,
    n_cores=elpigraph_n_cores,
    verbose=False,
    Lambda=elpigraph_lambda,
    Mu=elpigraph_mu,
)[0]

node_pos = np.asarray(res["NodePositions"])
edges = np.asarray(res["Edges"][0], dtype=int)
elpigraph_results: dict[str, dict[str, dict[str, np.ndarray]]] = {
    dataset: {roi: {"node_positions": node_pos, "edges": edges}}
}

adata.uns["elpigraph_cluster2_first_roi_first_dataset_spatial"] = elpigraph_results

fig, ax = plt.subplots(figsize=(6, 6), dpi=200)
ax.scatter(X[:, 0], X[:, 1], s=0.5, alpha=0.25, color="k", rasterized=True)
for i, j in edges:
    a = node_pos[i]
    b = node_pos[j]
    ax.plot([a[0], b[0]], [a[1], b[1]], color="tab:red", lw=1.0, alpha=0.9)
ax.set_title(f"ElPiGraph: dataset={dataset}, roi={roi}, cluster={elpigraph_cluster_value}")
ax.set_aspect("equal")
ax.invert_yaxis()
plt.tight_layout()

# %%

# --- Overlay UMAP: BRDU (green) vs EDU (magenta) -----------------------------------------------

import anndata as ad
import matplotlib


def _percentile_normalize(x: np.ndarray, percs: tuple[float, float] = (1.0, 99.9)) -> np.ndarray:
    """Normalize a 1D array to 0..1 using given percentiles, robust to NaNs/inf."""
    x = np.asarray(x)
    finite = np.isfinite(x)
    if not finite.any():
        return np.zeros_like(x, dtype=float)
    lo = np.nanpercentile(x[finite], percs[0])
    hi = np.nanpercentile(x[finite], percs[1])
    if hi <= lo:
        return np.clip(x - lo, 0, None)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def plot_umap_overlay_green_magenta(
    adata: ad.AnnData,
    *,
    green_key: str = "log_edu_mean",
    magenta_key: str = "log_brdu_mean",
    basis: str = "X_umap",
    percs: tuple[float, float] = (70,99.99),
    s: float = 1.0,
    alpha: float = 0.7,
    background: str = "black",
    fig: plt.Figure | None = None,
    ax: plt.Axes | None = None,
    title: str | None = "UMAP: BrdU (green) vs EdU (magenta)",
) -> tuple[plt.Figure, plt.Axes]:
    """
    Overlay two continuous features on UMAP using additive RGB:
    - Green channel ← `green_key` (e.g., BRDU)
    - Magenta channel (R+B) ← `magenta_key` (e.g., EDU)
    Areas with both high → near-white; exclusive → pure green/magenta.
    """

    xy = adata.obsm[basis]
    if xy.shape[1] != 2:
        raise ValueError("Expected 2D UMAP embedding in adata.obsm['X_umap'].")

    if green_key not in adata.obs or magenta_key not in adata.obs:
        missing = [k for k in [green_key, magenta_key] if k not in adata.obs]
        raise KeyError(f"Missing keys in adata.obs: {missing}")

    g_raw = adata.obs[green_key].to_numpy()
    m_raw = adata.obs[magenta_key].to_numpy()

    g = _percentile_normalize(g_raw, percs)
    m = _percentile_normalize(m_raw, percs)

    # Magenta = R+B; Green stays in G.
    rgb = np.column_stack([m, g, m])  # shape (n, 3)

    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(18, 12), dpi=200, facecolor=background)
    else:
        fig = ax.figure

    ax.set_facecolor(background)
    ax.scatter(xy[:, 0], xy[:, 1], c=rgb, s=s, alpha=alpha, linewidths=0, rasterized=True)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if title:
        ax.set_title(title, color="white" if background == "black" else "black")

    # Text legend
    ax.text(0.02, 0.98, "BrdU", transform=ax.transAxes, color="magenta", va="top", ha="left", fontsize=10)
    ax.text(0.98, 0.98, "EdU", transform=ax.transAxes, color="lime", va="top", ha="right", fontsize=10)

    return fig, ax


fig, ax = plt.subplots(figsize=(18, 12), dpi=300, facecolor="black")
# Render the overlay next to the individual maps
plot_umap_overlay_green_magenta(
    adata, fig=fig, ax=ax, green_key="log_edu_mean", magenta_key="log_brdu_mean", basis="X_umap", s=0.5
)


# %%
plt.scatter(adata.obs['p_edu'], adata.obs['p_brdu'], s=0.1, alpha=0.4)

 # %%
var = "EdU"
fig, ax = plt.subplots(ncols=1, figsize=(8, 6), dpi=200)
# u = adata[adata.obs["leiden"] == "1"]
ax.set_title(f"θ vs mean {var} intensity")
ax.hexbin(
    adata.obs["tricycle"],
    adata.obs[f"p_{var.lower()}"],
    gridsize=200,
    cmap="Greens",
    vmax=100,
)

ax.set_xlabel("θ")
ax.set_ylabel(f"log10(mean {var} intensity)")
