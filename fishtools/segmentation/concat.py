# %%
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from collections import defaultdict
from itertools import product
from pathlib import Path
from typing import NamedTuple

import matplotlib as mpl
import polars as pl
import seaborn as sns

from fishtools.postprocess import (
    normalize_total,
    rotate_rois_in_adata,
)
from fishtools.utils.io import Workspace
from fishtools.utils.utils import create_rotation_matrix

mpl.rcParams["figure.dpi"] = 300
sns.set_theme()
ws = Workspace(Path("/working/20251001_JaxA3_Coro11/analysis/deconv"))
rois = ["2r"]  # ws.rois
seg_codebook = "pi"
codebooks = ["cs_base"]
path = ws.deconved

dfs: dict[tuple[str, str], pl.DataFrame] = {}
for roi, codebook in product(rois, codebooks):
    print(roi, codebook)
    glob_path = path / f"stitch--{roi}+{seg_codebook}" / f"chunks+{codebook}/ident_*.parquet"
    df_roi = (
        pl.scan_parquet(
            glob_path,
            include_file_paths="path",
            missing_columns="insert",
        )
        .with_columns(
            z=pl.col("path").str.extract(r"(\d+)\.parquet").cast(pl.UInt8),
            codebook=pl.col("path").str.extract(r"chunks\+(\w+)"),
            spot_id=pl.col("spot_id").cast(pl.UInt32),
            roi=pl.lit(roi),
        )
        .with_columns(
            roilabel=pl.format("{}|{}", pl.col("roi"), pl.col("label")),
        )
        .sort("z")
        .collect()
    )
    if not df_roi.is_empty():
        dfs[(roi, codebook)] = df_roi

if not dfs:
    raise ValueError(
        f"No ident files found for any ROI in '{rois}' with seg_codebook '{seg_codebook}', or 'rois' list is empty."
    )
df = pl.concat(dfs.values())
# %%
INTENSITY = NamedTuple("INTENSITY", [("roi", str), ("intensity", str)])
channels = ["edu", "brdu", "pi"]
intensities: dict[INTENSITY, pl.DataFrame] = {}
for intensity in channels:
    for roi in rois:
        # Construct path using the singular seg_codebook
        glob_path_intensity = (
            path / f"stitch--{roi}+{seg_codebook}" / f"intensity_{intensity}/intensity-*.parquet"
        )
        print(glob_path_intensity)
        try:
            intensity_roi = (
                pl.scan_parquet(
                    glob_path_intensity,
                    include_file_paths="path",
                    missing_columns="insert",
                )
                .with_columns(
                    z=pl.col("path").str.extract(r"(\d+)\.parquet").cast(pl.UInt16),
                    roi=pl.lit(roi),
                    label=pl.col("label").cast(pl.UInt32),
                )
                .select(["z", "roi", "label", "mean_intensity", "max_intensity", "min_intensity"])
                .collect()
            )
            print(intensity_roi)
            intensity_roi = intensity_roi.rename({
                f"{x}_intensity": f"{intensity}_{x}" for x in ["mean", "max", "min"]
            })
            if not intensity_roi.is_empty():
                intensities[INTENSITY(roi=roi, intensity=intensity)] = intensity_roi
            # else: # Optional:
            # print(f"Warning: No intensity files found for ROI '{roi}' with seg_codebook '{seg_codebook}' at {glob_path_intensity}")
        except Exception as e:
            # print(f"Warning: Could not load intensity files for ROI '{roi}' with seg_codebook '{seg_codebook}' from {glob_path_intensity}. Error: {e}")
            pass


if not intensities:  # Corrected check
    raise ValueError(
        f"No intensity files found for any ROI in '{rois}' with seg_codebook '{seg_codebook}', or 'rois' list is empty."
    )

assert len(intensities) == len(rois) * len(channels), (
    f"Not all ROIs have intensity data. Expected {len(rois)}, got {len(intensities)}."
)

# %%

_spots_accumulator = defaultdict(list)

# Iterate per ROI/codebook to build a joined frame that carries spot coordinates for export.
joineds = []
for (roi, codebook), _df in dfs.items():
    print(roi, codebook)

    spots_file_path = path.parent / "output" / f"{roi}+{codebook}.parquet"
    if not spots_file_path.exists():
        print(f"Info: Spots file not found, skipping: {spots_file_path}")
        continue

    # Ensure a "spot_id" column that matches ident_*.parquet
    spots_df = pl.read_parquet(spots_file_path).with_columns(roi=pl.lit(roi), codebook=pl.lit(codebook))
    if "index" in spots_df.columns:
        spots_df = spots_df.rename({"index": "spot_id"})
    else:
        spots_df = spots_df.with_row_index(name="spot_id")

    # Join by the unique spot identifier to fetch per-spot coordinates and attributes
    joined = _df.join(spots_df, on="spot_id", how="left")
    if joined.is_empty():
        print(f"Info: No relevant ident data found for ROI '{roi}', codebook '{codebook}'.")
    joineds.append(joined)

joined = pl.concat(joineds) if joineds else pl.DataFrame()

# %%
(path / "baysor").mkdir(exist_ok=True)
if not joined.is_empty():
    joined.select(
        x=pl.col("x"),
        y=pl.col("y"),
        z=pl.col("z"),
        gene=pl.col("target").str.split("-").list.get(0),
        cell=pl.col("label").fill_null(0),
    ).filter(pl.col("gene") != "Blank").write_csv(path / "baysor" / "spots.csv")


# %%
def arrange_rois(polygons: pl.DataFrame, max_columns: int = 2, padding: float = 100):
    # Calculate bounding box for each ROI
    roi_bounds = (
        polygons.group_by("roi")
        .agg(
            min_x=pl.col("centroid_x").min(),
            max_x=pl.col("centroid_x").max(),
            min_y=pl.col("centroid_y").min(),
            max_y=pl.col("centroid_y").max(),
        )
        .sort("roi")
    )
    print(roi_bounds)

    # Calculate offsets for each ROI in a grid layout
    roi_offsets = {}
    max_width = 0
    max_height = 0

    for i, roi_row in enumerate(roi_bounds.iter_rows(named=True)):
        row = i // max_columns
        col = i % max_columns
        width = roi_row["max_x"] - roi_row["min_x"]
        height = roi_row["max_y"] - roi_row["min_y"]

        x_offset = col * (max_width + padding) - roi_row["min_x"]
        y_offset = row * (max_height + padding) - roi_row["min_y"]

        roi_offsets[roi_row["roi"]] = (x_offset, y_offset)
        max_width = max(max_width, width)
        max_height = max(max_height, height)

    # Apply offsets to polygons
    return polygons.with_columns([
        pl.col("centroid_x")
        + pl.col("roi").map_elements(lambda r: roi_offsets[r][0], return_dtype=pl.Float64),
        pl.col("centroid_y")
        + pl.col("roi").map_elements(lambda r: roi_offsets[r][1], return_dtype=pl.Float64),
    ]), roi_offsets


polygons = {}
for roi in rois:
    polygons[roi] = (
        pl.scan_parquet(
            path / f"stitch--{roi}+{seg_codebook}" / f"chunks+{codebooks[0]}/polygons_*.parquet",
            include_file_paths="path",
        )
        .with_columns(z=pl.col("path").str.extract(r"(\d+)\.parquet").cast(pl.UInt8), roi=pl.lit(roi))
        .with_columns(
            roilabel=pl.format("{}|{}", pl.col("roi"), pl.col("label")),
            roilabelz=pl.format(
                "{}|{}|{}",
                pl.col("z").cast(pl.Utf8),
                pl.col("roi"),
                pl.col("label").cast(pl.Utf8),
            ),
        )
        .drop("path")
        .sort("z")
        .collect()
    )


# --- Diagnostics: check (z, roi, label) coverage parity before joining intensities ---
def _pair_set(df: pl.DataFrame) -> set[tuple[int, int]]:
    if df.is_empty():
        return set()
    # Cast to Python ints to avoid dtype mismatches when comparing
    return set((int(z), int(lbl)) for z, lbl in df.select(["z", "label"]).unique().iter_rows())


for roi in rois:
    poly_pairs = _pair_set(polygons[roi])
    print(f"[diag] ROI={roi}: polygons unique (z,label)={len(poly_pairs)}")
    for ch in channels:
        key = INTENSITY(roi=roi, intensity=ch)
        if key not in intensities:
            print(f"[diag] ROI={roi}, ch={ch}: intensity shards missing")
            continue
        int_pairs = _pair_set(intensities[key])
        missing = poly_pairs - int_pairs
        extra = int_pairs - poly_pairs
        print(
            f"[diag] ROI={roi}, ch={ch}: missing_in_intensity={len(missing)}, extra_in_intensity={len(extra)}"
        )
        if missing:
            print(f"[diag]   sample missing (z,label): {sorted(list(missing))[:10]}")
        if extra:
            print(f"[diag]   sample extra   (z,label): {sorted(list(extra))[:10]}")

for polygon in list(polygons):
    _df = polygons[polygon].with_columns(
        z=pl.col("z").cast(pl.UInt16),
        label=pl.col("label").cast(pl.UInt32),
    )
    for intensity in ["edu", "brdu", "pi"]:
        if (roi_intensity := INTENSITY(roi=polygon, intensity=intensity)) in intensities:
            _df = _df.join(intensities[roi_intensity], on=["z", "roi", "label"], how="left")

    polygons[polygon] = _df
polygons = pl.concat(polygons.values())
polygons, roi_offsets = arrange_rois(polygons, max_columns=2, padding=100)


def pl_weighted_mean(value_col: str, weight_col: str) -> pl.Expr:
    """
    Generate a Polars aggregation expression to take a weighted mean.

    Note: the previous implementation incorrectly used the literal string
    name for the weight column inside a `then(...)`, which produced a
    string-valued expression instead of referencing the actual column.
    That could propagate nulls and yield zeros/NaNs when aggregating. The
    correct form references the column via `pl.col(weight_col)` and computes
    sum(value * weight) / sum(weight).
    """
    values = pl.col(value_col)
    weights = pl.when(values.is_not_null()).then(pl.col(weight_col)).otherwise(None)
    return (values * weights).sum().truediv(weights.sum()).fill_nan(None)


_agg = dict(
    area=pl.col("area").sum(),
    x=(pl.col("centroid_x") * pl.col("area")).sum() / pl.col("area").sum(),
    y=(pl.col("centroid_y") * pl.col("area")).sum() / pl.col("area").sum(),
    z=(pl.col("z").cast(pl.Float64) * pl.col("area")).sum() / pl.col("area").sum(),
    roi=pl.col("roi").first(),
)

for intensity in ["edu", "brdu", "pi"]:
    _agg.update({
        f"{intensity}_mean": pl_weighted_mean(f"{intensity}_mean", "area"),
        f"{intensity}_max": pl.col(f"{intensity}_max").max(),
        f"{intensity}_min": pl.col(f"{intensity}_min").min(),
    })

# %%
weighted_centroids = polygons.group_by(pl.col("roilabel")).agg(**_agg).sort("roilabel")

# %%
weighted_centroids = weighted_centroids.to_pandas().set_index("roilabel")
weighted_centroids.index = weighted_centroids.index.astype(str)

# %%
# ident = df.join(spots[["index", "target"]], left_on="spot_id", right_on="index", how="left")
# %%
molten = df.group_by([pl.col("roilabel"), pl.col("target")]).agg(pl.len())


# %%
def convert_transcript_to_gene(ts: str, non_unique_targets: list[str] | None = None) -> str:
    if non_unique_targets and any(ts.startswith(nt) for nt in non_unique_targets):
        return ts
    gene, tsidx = ts.rsplit("-", 1)
    return gene


non_unique_targets = (
    molten.filter(~pl.col("target").str.starts_with("Blank"))
    .unique("target")
    .with_columns(gene_name=pl.col("target").map_elements(convert_transcript_to_gene, return_dtype=pl.Utf8))
    .group_by("gene_name")
    .agg(count=pl.len())
    .filter(pl.col("count") > 1)
    .get_column("gene_name")
    .to_list()
)
print("Non-unique genes:", non_unique_targets)

# %%
cbg = (
    molten.filter(~pl.col("target").str.starts_with("Blank"))
    .with_columns(
        gene_name=pl.col("target").map_elements(
            lambda x: convert_transcript_to_gene(x, non_unique_targets), return_dtype=pl.Utf8
        )
    )
    .drop("target")
    .pivot("gene_name", index="roilabel", values="len")
    .fill_null(0)
    .sort("roilabel")
    .to_pandas()
    .set_index("roilabel")
)
cbg.index = cbg.index.astype(str)
# %%
# %%

import anndata as ad
import cmocean  # colormap, do not remove
import colorcet as cc  # colormap, do not remove
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import spaco
import tifffile
from shapely import MultiPolygon, Point, Polygon, STRtree

adata = ad.AnnData(cbg)


adata.obs = weighted_centroids.reindex(adata.obs.index)
n_genes = adata.shape[1]
sc.pp.calculate_qc_metrics(
    adata, inplace=True, percent_top=(n_genes // 10, n_genes // 5, n_genes // 2, n_genes)
)
sc.pp.filter_cells(adata, min_counts=30)
sc.pp.filter_cells(adata, max_counts=1200)
sc.pp.filter_genes(adata, min_cells=10)
# adata = adata[(adata.obs["y"] < 23189.657075200797) | (adata.obs["y"] > 46211.58630310604)]
# adata = adata[~adata.obs["roi"].isin(["bl", "br"])]
adata.obsm["spatial"] = adata.obs[["x", "y"]].to_numpy()
sc.pl.violin(
    adata,
    ["n_genes_by_counts", "total_counts"],
    jitter=0.4,
    multi_panel=True,
)
print(np.median(adata.obs["total_counts"]), len(adata))
# %%
adata.write_h5ad(ws.output / f"1whole+cs_base.h5ad")
# adata.write_h5ad(path / "segmentation_counts.h5ad")
# %%
plt.scatter(adata.obs["x"][::1], adata.obs["y"][::1], s=1, alpha=0.3)
plt.axis("equal")

# %%
# %%

#
adata.layers["raw"] = adata.X.copy()
adata, plot = normalize_total(adata)
adata.obs["edu_mean"] = adata.obs["edu_mean"] * adata.obs["area"]
# adata.obsm["spatial"] = np.dot(create_rotation_matrix(82), adata.obsm["spatial"].T).T
# plot()
# adata, plot = normalize_pearson(adata)

# %%

# adata.X = adata.X / (adata.obs["area"].to_numpy()[:, np.newaxis] / adata.obs["area"].mean())
# sc.pp.log1p(adata)
# %%
#  (adata.obs['total_counts'] / adata.obs["area"]) / np.mean(adata.X.sum(axis=1) / adata.obs["area"])


# %%
import rapids_singlecell as rsc

rsc.tl.pca(adata, n_comps=50)
sc.pl.pca_variance_ratio(adata, log=True)

# %%
# import rapids_singlecell as rsc
import scanpy.external as sce

# sce.pp.scanorama_integrate(adata, "roi", verbose=1)
# adata.write_h5ad(path / "scanorama.h5ad")
# %%
rsc.pp.neighbors(adata, n_neighbors=25, n_pcs=25, metric="cosine", random_state=12)
# %%
sc.tl.leiden(adata, n_iterations=2, resolution=2, flavor="igraph")
# %%
rsc.tl.umap(adata, min_dist=0.1, n_components=2, random_state=0)
# %%
# fig, ax = plt.subplots(figsize=(8, 6))

rotate_rois_in_adata(adata, {"1whole": 60})
# %%
from fishtools.utils.plot import add_scale_bar, plot_embedding

fig, axs = plot_embedding(
    adata,
    color=["edu_max"],
    basis="spatial_rot",
    dpi=300,
    figsize=(10, 6),
    s=2,
    # legend_loc=None,
    # cmap="Blues",
    # vmin=np.percentile(adata.obs["mean_intensity"], 1),
    # vmax=np.percentile(adata.obs["mean_intensity"], 99),
)
axs[0].axis("off")
add_scale_bar(axs[0], 1000 / 0.216, "1000 μm")
# adata.obs["cfsepos"] = adata.obs["mean_intensity"] > 10000

# %%
adata.obs["log_edu_mean"] = np.log10(adata.obs["edu_mean"] + 1)
adata.obs["log_brdu_mean"] = np.log10(adata.obs["brdu_mean"] + 1)


def plot_intensity(gene: str, basis="spatial_rot", **kwargs):
    fig, axs = plot_embedding(
        adata[adata.obs["roi"] == rois[0]],
        color=[f"log_{gene}_mean"],
        basis=basis,
        dpi=300,
        cmap="Blues",
        figsize=(8, 4),
        vmin=np.percentile(adata.obs[f"log_{gene}_mean"], 80),
        vmax=np.percentile(adata.obs[f"log_{gene}_mean"], 99.9),
        colorbar_loc=None,
        # cmap="Blues",
        # vmin=np.percentile(adata.obs["mean_intensity"], 1),
        # vmax=np.percentile(adata.obs["mean_intensity"], 99),
    )
    axs[0].axis("off")
    add_scale_bar(axs[0], 1000 / 0.216, "1000 μm")
    return fig, axs


plot_intensity("brdu")


# %%
def plot_genes(genes: list[str], basis="umap", **kwargs):
    genes = [adata.var_names[adata.var_names.str.startswith(gene)][0] for gene in genes]
    return sc.pl.embedding(adata, basis=basis, color=["leiden", *genes], **kwargs)


# %%

# %%
fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
gene = "leiden"
# sc.pl.embedding(
#     adata,
#     color=gene,
#     basis="spatial",
#     ax=ax,
#     cmap="Blues",
#     groups=["2", "6", "8"],
#     # vmax=np.percentile(adata[:, gene].X, 99.99),
#     # vmin=np.percentile(adata[:, gene].X, 50),
# )  # type: ignore
m = adata  # [adata.obs["mean_intensity"] > 2000]
# ax.scatter(
#     m.obsm["spatial"][:, 0],
#     m.obsm["spatial"][:, 1],
#     s=0.1,
#     alpha=0.3,
#     c=m.obs["mean_intensity"],
#     cmap="RdBu",
# )

fig, axs = plot_embedding(
    m[m.obs["roi"] == "1whole"],
    color=["leiden"],
    basis="spatial",
    dpi=300,
    # cmap="Blues",
    figsize=(8, 4),
    s=2,
    legend_loc=None,
    colorbar_loc=None,
    # cmap="Blues",
    # vmin=np.percentile(adata.obs["mean_intensity"], 1),
    # vmax=np.percentile(adata.obs["mean_intensity"], 99),
)
axs[0].axis("off")
add_scale_bar(axs[0], 1000 / 0.216, "1000 μm")

# ax.set_aspect("equal")
# ax.scatter(
#     m.obsm["spatial"][:, 0],
#     m.obsm["spatial"][:, 1],
#     s=0.1,
#     alpha=1.0,
#     c=list(map(int, m.obs["leiden"])),
# )
# %%
# fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
gene = "leiden"


sc.pl.embedding(
    adata,
    color=["leiden"],
    basis="umap",
    ncols=3,
    cmap="CMRmap_r",
    # ax=ax,
    # vmin=np.percentile(adata.obs["mean_intensity"], 1),
    # vmax=np.percentile(adata.obs["mean_intensity"], 99.9),
)  # type: ignore

# %%
# !! Progenitors are not the same. Separate by dorsoventral

# with sns.axes_style("white"):
fig = sc.pl.embedding(
    adata,
    basis="spatial_rot",
    color=["Pax6", "Cux2", "Neurog2", "Lhx2", "Boc", "Slc1a3", "Neurog2", "Satb2"],
    cmap="Blues",
    ncols=3,
    return_fig=True,
)

for ax in fig.axes:
    if ax.get_label() != "<colorbar>":
        ax.set_aspect("equal")
# fig.set_size_inches(12, 3)

plt.tight_layout()

# %%


trc = pd.read_csv("neuroRef.csv")
shared = sorted(set(trc["symbol"]) & set(adata.var_names))

loadings = trc = (
    trc[trc["symbol"].isin(shared)].set_index("symbol").reindex(shared).reset_index()[["pc1.rot", "pc2.rot"]]
)


# P = Et @ R
# where R represents the o-by-2 reference matrix (o≤500),
# contains the weights of top 2 PCs learned from PCA of GO cell-cycle genes;
# E ~ is a o-by-n matrix, subsetted from E (the log2 transformed expression matrix)
# with genes in the weights matrix and row-means centered.
pls = (adata[:, shared].X - np.mean(adata[:, shared].X, axis=1, keepdims=True)) @ loadings.to_numpy()
offset = [0, 0]

adata.obsm["tricycle"] = pls
adata.obs["tricycle"] = (
    np.arctan2(adata.obsm["tricycle"][:, 1] + offset[0], adata.obsm["tricycle"][:, 0] + offset[1])
) % (2 * np.pi)


sc.pl.embedding(
    adata,
    color=["tricycle"],
    basis="spatial_rot",
    ncols=3,
    # legend_loc="on data",
    frameon=False,
    # cmap="Blues",
    cmap="cet_colorwheel",
    # palette=palette_spaco,
    show=False,
)
fig = plt.gcf()
fig.set_size_inches(8, 10)
for ax in fig.axes:
    if ax.get_label() == "<colorbar>":
        continue
    ax.set_aspect("equal")

# %%
sc.pl.pca(adata, color=["tricycle"], cmap="cet_colorwheel")
# %%
sc.pl.umap(adata, color=["Eomes"], cmap="CMRmap_r", vmax=np.percentile(adata[:, "Eomes"].X, 99.9))
sc.pl.umap(
    adata, color=["log_edu_mean"], cmap="CMRmap_r", vmax=np.percentile(adata.obs["log_edu_mean"], 99.9)
)

# --- Overlay UMAP: BRDU (green) vs EDU (magenta) -----------------------------------------------
from typing import Tuple

import anndata as ad


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
    percs: tuple[float, float] = (40, 99),
    s: float = 2.0,
    alpha: float = 0.7,
    background: str = "black",
    ax: plt.Axes | None = None,
    title: str | None = "UMAP overlay: BRDU (green) vs EDU (magenta)",
) -> tuple[plt.Figure, plt.Axes]:
    """
    Overlay two continuous features on UMAP using additive RGB:
    - Green channel ← `green_key` (e.g., BRDU)
    - Magenta channel (R+B) ← `magenta_key` (e.g., EDU)
    Areas with both high → near-white; exclusive → pure green/magenta.
    """
    if "X_umap" not in adata.obsm:
        raise ValueError("UMAP coordinates not found in adata.obsm['X_umap'].")

    xy = adata.obsm["X_umap"]
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

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.2, 6.0), dpi=200, facecolor=background)
        created_fig = True
    else:
        fig = ax.figure

    ax.set_facecolor(background)
    ax.scatter(xy[:, 0], xy[:, 1], c=rgb, s=s, alpha=alpha, linewidths=0, rasterized=True)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title)

    # Text legend
    ax.text(0.02, 0.98, "BrdU", transform=ax.transAxes, color="lime", va="top", ha="left", fontsize=10)
    ax.text(0.98, 0.98, "EdU", transform=ax.transAxes, color="magenta", va="top", ha="right", fontsize=10)

    return fig, ax


# Render the overlay next to the individual maps
plot_umap_overlay_green_magenta(adata, green_key="log_edu_mean", magenta_key="log_brdu_mean")


# %%


# %%
var = "EdU"
fig, ax = plt.subplots(ncols=1, figsize=(8, 6), dpi=200)
u = adata[adata.obs["leiden"] == "1"]
ax.set_title(f"θ vs mean {var} intensity")
ax.hexbin(
    u.obs["tricycle"], np.log10(u.obs[f"log_{var.lower()}_mean"] + 1), gridsize=200, cmap="Blues", vmax=10
)
ax.set_xlabel("θ")
ax.set_ylabel(f"log10(mean {var} intensity)")
# %%
fig, ax = plt.subplots(ncols=1, figsize=(8, 6), dpi=200)

ax.set_title(f"θ vs mean {var} intensity")
ax.hexbin(
    adata.obs["tricycle"],
    np.log10(adata.obs[f"log_{var.lower()}_mean"] + 1),
    gridsize=200,
    cmap="Blues",
    vmax=10,
)
ax.set_xlabel("θ")
ax.set_ylabel(f"log10(mean {var} intensity)")

# %%

plt.loglog(
    np.mean(adata[adata.obs["roi"] == "tc"].X, axis=0),
    np.mean(adata[adata.obs["roi"] == "tl"].X, axis=0),
    ".",
)

# %%
from fishtools.utils.plot import plot_wheel

# %%
fig, axs = plot_wheel(
    np.nan_to_num(adata.obsm["tricycle"], nan=0),
    # scatter_cmap="Blues",
    c=adata.obs["tricycle"],
    alpha=0.1,
    # c=adata.obs["total_intensity"],
    # scatter_cmap="RdBu_r",
    colorize_background=False,
    fig=fig,
    # vmax=np.percentile(adata.obs["total_intensity"], 99),
    colorbar_label="θ",
)
# %%
fig, axs = plot_wheel(
    np.nan_to_num(m.obsm["tricycle"], nan=0),
    # scatter_cmap="Blues",
    alpha=0.1,
    c=m.obs["log_brdu_mean"],
    scatter_cmap="CMRmap_r",
    colorize_background=False,
    fig=fig,
    # vmax=np.percentile(adata.obs["total_intensity"], 99),
    colorbar_label="mean BrdU intensity",
)


# ! θ vs ventricular distance, count cells
# ! test with EdU
# %%
fig = sc.pl.embedding(
    adata,
    basis="umap",
    color=[
        "leiden",
        "log_edu_mean",
        "log_brdu_mean",
        "Pax6",
        "Sox2",
        "Hes5",
        # "Btg2",
        # "Neurog2",
        # "Neurod1",
        # "Eomes",
        # "Dlx1",
        # "Dlx2",
        # "Lhx6",
        # "Nrp1",
        # "Gad1",
        # "Satb2",
        # "Fezf2",
        # "Slc17a6",
        "Slc17a7",
        "Lhx2",
        "Lhx6",
        "Pdgfra",
        "Gad1",
        "Cux2",
        "Eomes",
        "Foxp2",
    ],
    frameon=False,
    cmap="Blues",
    return_fig=True,
)


# %%

# %%


for ax in fig.axes:
    ax.set_aspect("equal")

# %%
# fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
# ax.set_aspect("equal")
plot_embedding(adata, color="leiden", basis="spatial", ax=ax, dpi=200)

# %%
plt.scatter(adata.obs["x"], adata.obs["y"], s=0.1, alpha=0.3)
# %%
import spaco

color_mapping = spaco.colorize(
    cell_coordinates=adata.obsm["spatial"],
    cell_labels=adata.obs["leiden"],
    radius=0.1,
    n_neighbors=15,
    colorblind_type="none",
)
# %%

color_mapping = {k: color_mapping[k] for k in adata.obs["leiden"].cat.categories}
palette_spaco = list(color_mapping.values())

# %%
# fig, axs = plt.subplots(figsize=(12, 6), dpi=200, ncols=2)
# axs = axs.flatten()
sc.pl.embedding(
    adata,
    color="leiden",
    palette=palette_spaco,
    basis="spatial_rot",
)  # type: ignore
# sc.pl.embedding(
#     adata,
#     color="leiden",
#     palette=palette_spaco,
#     basis="umap",
#     ax=axs[0],
# )  # type: ignore

for ax in plt.gcf().axes:
    ax.set_aspect("equal")
    if not ax.has_data():
        fig.delaxes(ax)
plt.tight_layout()

# %%
fig, ax = plt.subplots(figsize=(8, 6))

ax.set_aspect("equal")


# %%
def compare_genes(adata, genes, ax=None, jitter=0.02, dark=False, quadrant_thresholds=None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6), dpi=200, facecolor="white" if not dark else "black")

    ax.set_aspect("equal")
    rand = np.random.default_rng(0)
    # Ensure gene names are valid and exist in adata
    valid_genes = [gene for gene in genes if gene in adata.var_names]
    if len(valid_genes) < 2:
        raise ValueError(f"At least two valid genes are required. Found: {valid_genes} in {genes}")

    x = adata[:, valid_genes[0]].X.squeeze() + rand.normal(0, jitter, len(adata))
    y = adata[:, valid_genes[1]].X.squeeze() + rand.normal(0, jitter, len(adata))
    print(x)
    if not dark:
        ax.scatter(x, y, s=0.1, alpha=0.1, **kwargs)
    else:
        ax.hexbin(x, y, gridsize=250)
    ax.set_xlabel(valid_genes[0])
    ax.set_ylabel(valid_genes[1])
    ax.set_aspect("equal")
    min_val = min(np.min(x), np.min(y)) - 1
    max_val = max(np.max(x), np.max(y)) + 1
    ax.set_xlim((min_val, max_val))
    ax.set_ylim((min_val, max_val))

    if quadrant_thresholds is not None and len(quadrant_thresholds) == 2:
        x_thresh, y_thresh = quadrant_thresholds
        ax.axhline(y_thresh, color="grey", linestyle="--", lw=1)
        ax.axvline(x_thresh, color="grey", linestyle="--", lw=1)

        total_points = len(x)
        if total_points > 0:
            q_tr = np.sum((x >= x_thresh) & (y >= y_thresh))
            q_tl = np.sum((x < x_thresh) & (y >= y_thresh))
            q_bl = np.sum((x < x_thresh) & (y < y_thresh))
            q_br = np.sum((x >= x_thresh) & (y < y_thresh))

            perc_tr = (q_tr / total_points) * 100
            perc_tl = (q_tl / total_points) * 100
            perc_bl = (q_bl / total_points) * 100
            perc_br = (q_br / total_points) * 100

            # Position text relative to plot limits and thresholds
            text_props = dict(ha="center", va="center", fontsize=12, color="black" if not dark else "white")

            # Adjust text position to be within the plot and quadrant
            x_range = max_val - min_val
            y_range = max_val - min_val

            ax.text(
                x_thresh + 0.5 * (max_val - x_thresh),
                y_thresh + 0.5 * (max_val - y_thresh),
                f"{perc_tr:.1f}%",
                **text_props,
            )  # TR
            ax.text(
                x_thresh - 0.5 * (x_thresh - min_val),
                y_thresh + 0.5 * (max_val - y_thresh),
                f"{perc_tl:.1f}%",
                **text_props,
            )  # TL
            ax.text(
                x_thresh - 0.5 * (x_thresh - min_val),
                y_thresh - 0.5 * (y_thresh - min_val),
                f"{perc_bl:.1f}%",
                **text_props,
            )  # BL
            ax.text(
                x_thresh + 0.5 * (max_val - x_thresh),
                y_thresh - 0.5 * (y_thresh - min_val),
                f"{perc_br:.1f}%",
                **text_props,
            )  # BR
    return ax


sns.set_theme()
with sns.axes_style("white"):
    compare_genes(adata, ["Slc17a6", "Gad1"], dark=False, quadrant_thresholds=(0.15, 0.15))

# %%
sc.pl.embedding(
    adata[adata[:, "Gad2"].X.__gt__(0) & adata[:, "Slc17a7"].X.__gt__(1)],
    basis="spatial",
    color=["Gad2", "Slc17a7"],
    cmap="Blues",
)


# %%

adata.obs["leidencfse"] = adata.obs["leiden"].astype(str) + adata.obs["cfsepos"].astype(str)
# %%
sc.tl.rank_genes_groups(adata, groupby="leiden", method="wilcoxon")

# %%
from fishtools.postprocess import plot_ranked_genes

plot_ranked_genes(adata)

# %%
adata.write_h5ad(path / "org1.h5ad")

# %%

sns.clustermap(adata.X)

# %%
import pandas as pd
import scipy
import seaborn as sns

# Get gene expression matrix
gene_expr = adata.X.toarray() if scipy.sparse.issparse(adata.X) else adata.X

# Create DataFrame with gene names
genes_df = pd.DataFrame(gene_expr, columns=adata.var_names)

# Compute correlation matrix
corr_matrix = genes_df.corr()

# Create clustermap
sns.clustermap(corr_matrix, cmap="RdBu_r", figsize=(12, 12), center=0, vmin=-1, vmax=1)
# %%
