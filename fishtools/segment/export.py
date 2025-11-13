from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Iterable, TYPE_CHECKING

import polars as pl
from loguru import logger

from fishtools.io.workspace import Workspace

if TYPE_CHECKING:  # pragma: no cover - typing only
    import anndata


@dataclass(frozen=True)
class _IntensityKey:
    roi: str
    channel: str


def _discover_channels(stitch_root: Path) -> list[str]:
    """Discover channels by scanning intensity_* subdirectories.

    Returns a sorted list of channel names.
    """
    channels: list[str] = []
    if not stitch_root.exists():
        return channels
    for p in stitch_root.iterdir():
        if p.is_dir() and p.name.startswith("intensity_"):
            channels.append(p.name.removeprefix("intensity_"))
    return sorted(set(channels))


def _scan_ident(path: Path) -> pl.DataFrame:
    return pl.scan_parquet(
        path,
        include_file_paths="path",
        missing_columns="insert",
    ).collect()


def _scan_polygons(path: Path) -> pl.DataFrame:
    return pl.scan_parquet(
        path,
        include_file_paths="path",
        missing_columns="insert",
    ).collect()


def _pl_weighted_mean(value_col: str, weight_col: str) -> pl.Expr:
    values = pl.col(value_col)
    weights = pl.when(values.is_not_null()).then(pl.col(weight_col)).otherwise(None)
    return (values * weights).sum().truediv(weights.sum()).fill_nan(None)


def _gene_name_from_target(target: str) -> str:
    base, sep, _ = target.rpartition("-")
    return base if sep else target


def _resolve_gene_name(target: str, duplicate_genes: set[str]) -> str:
    candidate = _gene_name_from_target(target)
    if candidate in duplicate_genes:
        return target
    return candidate


def _percent_top_tuple(n_genes: int) -> tuple[int, ...]:
    bases = [max(1, n_genes // d) for d in (10, 5, 2)] + [max(1, n_genes)]
    # Preserve input order but drop duplicates while keeping ints >= 1.
    seen: set[int] = set()
    ordered: list[int] = []
    for value in bases:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return tuple(ordered)


def _resolve_rois(ws: Workspace, roi: str | None) -> list[str]:
    rois = [roi] if roi else ws.rois
    if not rois:
        raise ValueError("No ROIs found in workspace; provide ROI explicitly or populate the workspace.")
    return rois


def _resolve_codebooks(codebooks: Iterable[str]) -> list[str]:
    cb_list = list(codebooks)
    if not cb_list:
        raise ValueError("At least one --codebook must be provided.")
    return cb_list


def _prepare_output_dir(deconv: Path, override: Path | None) -> Path:
    out_dir = override or (deconv / "segment_export")
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _resolve_channels(
    deconv: Path,
    rois: Iterable[str],
    seg_codebook: str,
    channels: str,
) -> list[str]:
    candidate = channels.strip()
    if candidate.lower() == "auto":
        discovered: set[str] = set()
        for roi in rois:
            stitched = deconv / f"stitch--{roi}+{seg_codebook}"
            discovered.update(_discover_channels(stitched))
        if not discovered:
            raise ValueError(
                "No intensity_* directories found; run 'segment overlay intensity' first or pass --channels."
            )
        return sorted(discovered)

    channel_list = [entry.strip() for entry in channels.split(",") if entry.strip()]
    if not channel_list:
        raise ValueError("Provided --channels string did not contain any valid entries.")
    return channel_list


def _load_ident_shards(
    deconv: Path,
    rois: Iterable[str],
    codebooks: Iterable[str],
    seg_codebook: str,
) -> dict[tuple[str, str], pl.DataFrame]:
    dfs: dict[tuple[str, str], pl.DataFrame] = {}
    for roi, codebook in product(rois, codebooks):
        root = deconv / f"stitch--{roi}+{seg_codebook}" / f"chunks+{codebook}"
        glob_path = root / "ident_*.parquet"
        if not any(root.glob("ident_*.parquet")):
            logger.warning(f"ROI={roi} codebook={codebook}: no ident shards under {root}")
            continue
        df_roi = _scan_ident(glob_path).with_columns(
            z=pl.col("path").str.extract(r"(\d+)\.parquet").cast(pl.UInt16),
            codebook=pl.lit(codebook),
            roi=pl.lit(roi),
            spot_id=pl.col("spot_id").cast(pl.UInt32),
        ).with_columns(
            roilabel=pl.format("{}|{}", pl.col("roi"), pl.col("label")),
        ).drop("path").sort("z")
        if not df_roi.is_empty():
            dfs[(roi, codebook)] = df_roi
    if not dfs:
        raise ValueError(
            f"No ident files found for ROIs {list(rois)} with seg_codebook '{seg_codebook}'."
        )
    return dfs


def _load_intensity_shards(
    deconv: Path,
    rois: Iterable[str],
    seg_codebook: str,
    channel_list: Iterable[str],
) -> dict[_IntensityKey, pl.DataFrame]:
    intensities: dict[_IntensityKey, pl.DataFrame] = {}
    for channel, roi in product(channel_list, rois):
        channel_dir = deconv / f"stitch--{roi}+{seg_codebook}" / f"intensity_{channel}"
        if not channel_dir.exists():
            logger.warning(f"ROI={roi} channel={channel}: intensity directory missing under {channel_dir}")
            continue
        glob_path = channel_dir / "intensity-*.parquet"
        if not any(channel_dir.glob("intensity-*.parquet")):
            logger.warning(f"ROI={roi} channel={channel}: no intensity shards under {channel_dir}")
            continue
        intensity_df = (
            pl.scan_parquet(glob_path, include_file_paths="path", missing_columns="insert")
            .with_columns(
                z=pl.col("path").str.extract(r"(\d+)\.parquet").cast(pl.UInt16),
                roi=pl.lit(roi),
                label=pl.col("label").cast(pl.UInt32),
            )
            .select(["z", "roi", "label", "mean_intensity", "max_intensity", "min_intensity"])
            .collect()
            .rename({
                "mean_intensity": f"{channel}_mean",
                "max_intensity": f"{channel}_max",
                "min_intensity": f"{channel}_min",
            })
        )
        if not intensity_df.is_empty():
            intensities[_IntensityKey(roi=roi, channel=channel)] = intensity_df
    if not intensities:
        raise ValueError(
            f"No intensity shards found for ROIs {list(rois)} and channels {list(channel_list)}."
        )
    return intensities


def _load_polygon_shards(
    deconv: Path,
    rois: Iterable[str],
    seg_codebook: str,
    primary_codebook: str,
) -> dict[str, pl.DataFrame]:
    polygons_by_roi: dict[str, pl.DataFrame] = {}
    for roi in rois:
        chunks_dir = deconv / f"stitch--{roi}+{seg_codebook}" / f"chunks+{primary_codebook}"
        glob_path = chunks_dir / "polygons_*.parquet"
        if not any(chunks_dir.glob("polygons_*.parquet")):
            logger.warning(f"ROI={roi}: no polygons shards under {chunks_dir}")
            continue
        pdf = _scan_polygons(glob_path).with_columns(
            z=pl.col("path").str.extract(r"(\d+)\.parquet").cast(pl.UInt16),
            roi=pl.lit(roi),
        ).with_columns(
            roilabel=pl.format("{}|{}", pl.col("roi"), pl.col("label")),
        ).drop("path").sort("z")
        if not pdf.is_empty():
            polygons_by_roi[roi] = pdf
    if not polygons_by_roi:
        raise ValueError(
            f"No polygons shards found for ROIs {list(rois)} with primary codebook '{primary_codebook}'."
        )
    return polygons_by_roi


def _emit_pairing_diagnostics(
    polygons_by_roi: dict[str, pl.DataFrame],
    intensities: dict[_IntensityKey, pl.DataFrame],
    channel_list: Iterable[str],
) -> None:
    def _pair_set(df: pl.DataFrame) -> set[tuple[int, int]]:
        if df.is_empty():
            return set()
        return {
            (int(z_value), int(label_value))
            for z_value, label_value in df.select(["z", "label"]).unique().iter_rows()
        }

    for roi, poly_df in polygons_by_roi.items():
        poly_pairs = _pair_set(poly_df)
        logger.info(f"[diag] ROI={roi}: polygons unique (z,label)={len(poly_pairs)}")
        for channel in channel_list:
            key = _IntensityKey(roi=roi, channel=channel)
            if key not in intensities:
                logger.info(f"[diag] ROI={roi}, ch={channel}: intensity shards missing")
                continue
            intensity_pairs = _pair_set(intensities[key])
            missing = poly_pairs - intensity_pairs
            extra = intensity_pairs - poly_pairs
            logger.info(
                f"[diag] ROI={roi}, ch={channel}: missing_in_intensity={len(missing)}, extra_in_intensity={len(extra)}"
            )
            if missing or extra:
                raise ValueError(
                    "Segmentation masks disagree: ROI="
                    f"{roi}, channel={channel}, missing_in_intensity={len(missing)}, extra_in_intensity={len(extra)}"
                )


def _build_cells_dataframe(
    polygons_by_roi: dict[str, pl.DataFrame],
    intensities: dict[_IntensityKey, pl.DataFrame],
    channel_list: Iterable[str],
) -> pl.DataFrame:
    polygons: list[pl.DataFrame] = []
    for roi, poly in polygons_by_roi.items():
        joined = poly.with_columns(label=pl.col("label").cast(pl.UInt32))
        for channel in channel_list:
            key = _IntensityKey(roi=roi, channel=channel)
            if key in intensities:
                joined = joined.join(intensities[key], on=["z", "roi", "label"], how="left")
        polygons.append(joined)
    if not polygons:
        raise ValueError("Failed to assemble polygons with intensities; no data available.")

    polygons_df = pl.concat(polygons)
    agg: dict[str, pl.Expr] = dict(
        area=pl.col("area").sum(),
        x=(pl.col("centroid_x") * pl.col("area")).sum() / pl.col("area").sum(),
        y=(pl.col("centroid_y") * pl.col("area")).sum() / pl.col("area").sum(),
        z=(pl.col("z").cast(pl.Float64) * pl.col("area")).sum() / pl.col("area").sum(),
        roi=pl.col("roi").first(),
    )
    for channel in channel_list:
        agg.update({
            f"{channel}_mean": _pl_weighted_mean(f"{channel}_mean", "area"),
            f"{channel}_max": pl.col(f"{channel}_max").max(),
            f"{channel}_min": pl.col(f"{channel}_min").min(),
        })

    return polygons_df.group_by(pl.col("roilabel")).agg(**agg).sort("roilabel")


def _write_cells_parquet(
    cells: pl.DataFrame,
    out_dir: Path,
    seg_codebook: str,
    primary_codebook: str,
) -> Path:
    out_path = out_dir / f"cells--{seg_codebook}+{primary_codebook}.parquet"
    cells.write_parquet(out_path)
    logger.info(f"Wrote cells parquet to {out_path}")
    return out_path


def _build_counts_matrix(dfs: dict[tuple[str, str], pl.DataFrame]) -> pl.DataFrame:
    ident_concat = pl.concat(list(dfs.values()))
    if ident_concat.is_empty():
        raise ValueError("Spot ident shards contained no rows to aggregate.")

    transcript_counts = (
        ident_concat
        .filter(~pl.col("target").str.starts_with("Blank"))
        .group_by(["roilabel", "target"])
        .agg(pl.len().alias("count"))
    )
    if transcript_counts.is_empty():
        raise ValueError("No informative targets found after filtering Blank controls.")

    duplicate_genes = (
        transcript_counts
        .with_columns(gene=pl.col("target").map_elements(_gene_name_from_target, return_dtype=pl.Utf8))
        .group_by("gene")
        .agg(pl.len().alias("occurrences"))
        .filter(pl.col("occurrences") > 1)
        .get_column("gene")
        .to_list()
    )
    duplicate_set = set(duplicate_genes)

    counts_by_gene = (
        transcript_counts
        .with_columns(
            gene=pl.col("target").map_elements(
                lambda value: _resolve_gene_name(value, duplicate_set),
                return_dtype=pl.Utf8,
            )
        )
        .group_by(["roilabel", "gene"])
        .agg(pl.col("count").sum().alias("transcripts"))
        .pivot(index="roilabel", on="gene", values="transcripts")
        .fill_null(0)
        .sort("roilabel")
    )
    if counts_by_gene.is_empty():
        raise ValueError("Failed to construct gene expression matrix for export.")
    return counts_by_gene


def _build_anndata(counts_by_gene: pl.DataFrame, cells: pl.DataFrame) -> "anndata.AnnData":
    counts_pd = counts_by_gene.to_pandas().set_index("roilabel")
    cells_pd = cells.to_pandas().set_index("roilabel")
    missing_obs = counts_pd.index.difference(cells_pd.index)
    if not missing_obs.empty:
        raise ValueError(
            "Missing centroid entries for roilabels: " + ", ".join(missing_obs.astype(str))
        )
    obs_pd = cells_pd.loc[counts_pd.index]

    import anndata as ad
    import numpy as np
    import pandas as pd
    import scanpy as sc

    adata = ad.AnnData(
        X=counts_pd.to_numpy(dtype=np.float32, copy=True),
        obs=obs_pd.copy(),
        var=pd.DataFrame(index=pd.Index(counts_pd.columns.astype(str), name="gene")),
    )
    adata.obs_names = pd.Index(counts_pd.index.astype(str), name="roilabel")
    if {"x", "y"} - set(adata.obs.columns):
        raise ValueError("Cells dataframe missing centroid columns 'x' and 'y'.")
    adata.obsm["spatial"] = adata.obs[["x", "y"]].to_numpy(dtype=np.float32)

    n_genes = adata.n_vars
    percent_top = _percent_top_tuple(n_genes)
    sc.pp.calculate_qc_metrics(adata, inplace=True, percent_top=percent_top)
    sc.pp.filter_cells(adata, min_counts=30)
    sc.pp.filter_cells(adata, max_counts=1200)
    sc.pp.filter_genes(adata, min_cells=10)
    if adata.n_obs == 0 or adata.n_vars == 0:
        raise ValueError("Filtering removed all cells or genes; adjust thresholds or inputs.")
    adata.obsm["spatial"] = adata.obs[["x", "y"]].to_numpy(dtype=np.float32)
    return adata

def export_cmd(
    path: Path,
    roi: str | None,
    seg_codebook: str,
    codebooks: Iterable[str],
    channels: str,
    out_dir: Path | None,
    diag: bool,
) -> None:
    """Export per-cell intensities and h5ad ready for Scanpy workflows.

    - Writes aggregated per-cell centroids and intensities to
      <out_dir>/cells.parquet (default: <deconv>/segment_export)
    - Emits an AnnData file under <workspace>/analysis/output capturing
      gene counts, QC metrics, and spatial coordinates.
    """

    ws = Workspace(path)
    rois = _resolve_rois(ws, roi)
    cb_list = _resolve_codebooks(codebooks)

    deconv = ws.deconved
    out_root = _prepare_output_dir(deconv, out_dir)

    ident_frames = _load_ident_shards(deconv, rois, cb_list, seg_codebook)
    channel_list = _resolve_channels(deconv, rois, seg_codebook, channels)
    intensities = _load_intensity_shards(deconv, rois, seg_codebook, channel_list)

    primary_cb = cb_list[0]
    polygons_by_roi = _load_polygon_shards(deconv, rois, seg_codebook, primary_cb)

    if diag:
        _emit_pairing_diagnostics(polygons_by_roi, intensities, channel_list)

    cells = _build_cells_dataframe(polygons_by_roi, intensities, channel_list)
    _write_cells_parquet(cells, out_root, seg_codebook, primary_cb)

    counts_by_gene = _build_counts_matrix(ident_frames)
    adata = _build_anndata(counts_by_gene, cells)

    roi_token = rois[0] if len(rois) == 1 else "all"
    cb_token = Workspace.sanitize_codebook_name(primary_cb)
    ws.output.mkdir(parents=True, exist_ok=True)
    out_h5ad = ws.output / f"{roi_token}+{cb_token}.h5ad"
    adata.write_h5ad(out_h5ad)
    logger.info(f"Wrote AnnData export to {out_h5ad}")
