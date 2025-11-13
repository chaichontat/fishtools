from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Iterable

import polars as pl
from loguru import logger

from fishtools.io.workspace import Workspace


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


def export_cmd(
    path: Path,
    roi: str | None,
    seg_codebook: str,
    codebooks: Iterable[str],
    channels: str,
    out_dir: Path | None,
    diag: bool,
) -> None:
    """Export joined spot assignments and per-cell intensities.

    - Builds baysor-compatible spots CSV under <deconv>/baysor/spots.csv
    - Writes aggregated per-cell centroids and intensities to
      <out_dir>/cells.parquet (default: <deconv>/segment_export)
    """

    ws = Workspace(path)
    rois = [roi] if roi else ws.rois
    if not rois:
        raise ValueError("No ROIs found in workspace; provide ROI explicitly or populate the workspace.")

    cb_list = list(codebooks)
    if not cb_list:
        raise ValueError("At least one --codebook must be provided.")

    deconv = ws.deconved
    out_dir = out_dir or (deconv / "segment_export")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load ident shards (spot→label) per ROI/codebook
    dfs: dict[tuple[str, str], pl.DataFrame] = {}
    for r, cb in product(rois, cb_list):
        root = deconv / f"stitch--{r}+{seg_codebook}" / f"chunks+{cb}"
        glob_path = root / "ident_*.parquet"
        if not any(root.glob("ident_*.parquet")):
            logger.warning(f"ROI={r} codebook={cb}: no ident shards under {root}")
            continue
        df_roi = _scan_ident(glob_path).with_columns(
            z=pl.col("path").str.extract(r"(\d+)\.parquet").cast(pl.UInt16),
            codebook=pl.lit(cb),
            roi=pl.lit(r),
            spot_id=pl.col("spot_id").cast(pl.UInt32),
        ).with_columns(
            roilabel=pl.format("{}|{}", pl.col("roi"), pl.col("label")),
        ).drop("path").sort("z")
        if not df_roi.is_empty():
            dfs[(r, cb)] = df_roi
    if not dfs:
        raise ValueError(
            f"No ident files found for ROIs {rois} with seg_codebook '{seg_codebook}'."
        )
    # Keep per-ROI/codebook idents for joins and counts; concatenation is not required here.

    # 2) Load intensity shards per ROI/channel
    intensities: dict[_IntensityKey, pl.DataFrame] = {}
    # channel selection: auto from intensity_* directories when channels == 'auto'
    if channels.strip().lower() == "auto":
        discovered: set[str] = set()
        for r in rois:
            stitched = deconv / f"stitch--{r}+{seg_codebook}"
            discovered.update(_discover_channels(stitched))
        if not discovered:
            raise ValueError(
                "No intensity_* directories found; run overlay_intensity2 first or pass --channels."
            )
        channel_list = sorted(discovered)
    else:
        channel_list = [c.strip() for c in channels.split(",") if c.strip()]

    for ch, r in product(channel_list, rois):
        glob_path = deconv / f"stitch--{r}+{seg_codebook}" / f"intensity_{ch}" / "intensity-*.parquet"
        if not any((glob_path.parent).glob("intensity-*.parquet")):
            logger.warning(f"ROI={r} channel={ch}: no intensity shards under {glob_path.parent}")
            continue
        intensity_df = (
            pl.scan_parquet(glob_path, include_file_paths="path", missing_columns="insert")
            .with_columns(
                z=pl.col("path").str.extract(r"(\d+)\.parquet").cast(pl.UInt16),
                roi=pl.lit(r),
                label=pl.col("label").cast(pl.UInt32),
            )
            .select(["z", "roi", "label", "mean_intensity", "max_intensity", "min_intensity"])
            .collect()
            .rename({
                "mean_intensity": f"{ch}_mean",
                "max_intensity": f"{ch}_max",
                "min_intensity": f"{ch}_min",
            })
        )
        if not intensity_df.is_empty():
            intensities[_IntensityKey(roi=r, channel=ch)] = intensity_df

    if not intensities:
        raise ValueError(
            f"No intensity shards found for ROIs {rois} and channels {channel_list}."
        )

    # 3) Spots export (Baysor CSV)
    baysor_dir = deconv / "baysor"
    baysor_dir.mkdir(exist_ok=True)
    joineds: list[pl.DataFrame] = []
    for (r, cb), idf in dfs.items():
        try:
            spots_path = ws.spots_parquet(r, cb, must_exist=True)
        except FileNotFoundError as exc:
            logger.warning(f"ROI={r} codebook={cb}: {exc}")
            continue
        spots_df = pl.read_parquet(spots_path).with_columns(roi=pl.lit(r), codebook=pl.lit(cb))
        if "index" in spots_df.columns:
            spots_df = spots_df.rename({"index": "spot_id"})
        else:
            spots_df = spots_df.with_row_index(name="spot_id")
        joined = idf.join(spots_df, on="spot_id", how="left")
        joineds.append(joined)

    if joineds:
        joined = pl.concat(joineds)
        (
            joined.select(
                x=pl.col("x"),
                y=pl.col("y"),
                z=pl.col("z"),
                gene=pl.col("target").str.split("-").list.get(0),
                cell=pl.col("label").fill_null(0),
            )
            .filter(pl.col("gene") != "Blank")
            .write_csv(baysor_dir / "spots.csv")
        )
        logger.info(f"Wrote Baysor spots CSV to {baysor_dir / 'spots.csv'}")
    else:
        logger.warning("No joined spots rows to export; skipping Baysor CSV.")

    # 4) Polygons + intensities join and aggregation
    # Use the first codebook for polygons (authoritative per-slice geometry)
    primary_cb = cb_list[0]
    polygons_by_roi: dict[str, pl.DataFrame] = {}
    for r in rois:
        poly_glob = deconv / f"stitch--{r}+{seg_codebook}" / f"chunks+{primary_cb}" / "polygons_*.parquet"
        if not any((poly_glob.parent).glob("polygons_*.parquet")):
            logger.warning(f"ROI={r}: no polygons shards under {poly_glob.parent}")
            continue
        pdf = _scan_polygons(poly_glob).with_columns(
            z=pl.col("path").str.extract(r"(\d+)\.parquet").cast(pl.UInt16),
            roi=pl.lit(r),
        ).with_columns(
            roilabel=pl.format("{}|{}", pl.col("roi"), pl.col("label")),
        ).drop("path").sort("z")
        polygons_by_roi[r] = pdf

    if not polygons_by_roi:
        raise ValueError(
            f"No polygons shards found for ROIs {rois} with primary codebook '{primary_cb}'."
        )

    if diag:
        def _pair_set(df: pl.DataFrame) -> set[tuple[int, int]]:
            if df.is_empty():
                return set()
            return set((int(z), int(lbl)) for z, lbl in df.select(["z", "label"]).unique().iter_rows())

        for r in rois:
            if r not in polygons_by_roi:
                continue
            poly_pairs = _pair_set(polygons_by_roi[r])
            logger.info(f"[diag] ROI={r}: polygons unique (z,label)={len(poly_pairs)}")
            for ch in channel_list:
                key = _IntensityKey(roi=r, channel=ch)
                if key not in intensities:
                    logger.info(f"[diag] ROI={r}, ch={ch}: intensity shards missing")
                    continue
                int_pairs = _pair_set(intensities[key])
                missing = poly_pairs - int_pairs
                extra = int_pairs - poly_pairs
                logger.info(
                    f"[diag] ROI={r}, ch={ch}: missing_in_intensity={len(missing)}, extra_in_intensity={len(extra)}"
                )

    # Join intensities onto polygons by (z, roi, label)
    polygons: list[pl.DataFrame] = []
    for r, pdf in polygons_by_roi.items():
        pdf = pdf.with_columns(label=pl.col("label").cast(pl.UInt32))
        for ch in channel_list:
            key = _IntensityKey(roi=r, channel=ch)
            if key in intensities:
                pdf = pdf.join(intensities[key], on=["z", "roi", "label"], how="left")
        polygons.append(pdf)
    polygons_df = pl.concat(polygons)

    # Aggregate per-cell weighted centroids and intensities
    agg: dict[str, pl.Expr] = dict(
        area=pl.col("area").sum(),
        x=(pl.col("centroid_x") * pl.col("area")).sum() / pl.col("area").sum(),
        y=(pl.col("centroid_y") * pl.col("area")).sum() / pl.col("area").sum(),
        z=(pl.col("z").cast(pl.Float64) * pl.col("area")).sum() / pl.col("area").sum(),
        roi=pl.col("roi").first(),
    )
    for ch in channel_list:
        agg.update({
            f"{ch}_mean": _pl_weighted_mean(f"{ch}_mean", "area"),
            f"{ch}_max": pl.col(f"{ch}_max").max(),
            f"{ch}_min": pl.col(f"{ch}_min").min(),
        })

    cells = polygons_df.group_by(pl.col("roilabel")).agg(**agg).sort("roilabel")
    out_cells = out_dir / f"cells--{seg_codebook}+{primary_cb}.parquet"
    cells.write_parquet(out_cells)
    logger.info(f"Wrote cells parquet to {out_cells}")
