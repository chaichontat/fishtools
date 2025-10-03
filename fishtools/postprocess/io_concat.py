"""Data loading helpers for concat analyses.

This module centralizes parquet discovery for the segmentation concat workflows so
that interactive notebooks and CLI entry-points can share the same I/O logic.
The helpers here intentionally avoid side-effects beyond reading parquet files
and annotating derived columns that downstream code expects (e.g. `roi`,
`codebook`, and composite ROI labels).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, MutableMapping, Sequence

import polars as pl
from loguru import logger

from fishtools.io.workspace import Workspace

__all__ = [
    "ConcatDataError",
    "ConcatDataSpec",
    "load_ident_tables",
    "load_intensity_tables",
    "load_spot_tables",
    "load_polygon_tables",
    "join_ident_with_spots",
    "merge_polygons_with_intensity",
    "arrange_rois",
    "compute_weighted_centroids",
]


class ConcatDataError(RuntimeError):
    """Raised when required concat inputs are missing or malformed."""


@dataclass(slots=True)
class ConcatDataSpec:
    """Container describing where concat parquet assets live.

    Args:
        workspace: Workspace root or object.
        rois: Sequence of ROI identifiers to load. If empty, all workspace ROIs
            are used.
        seg_codebook: Segmentation codebook suffix used in stitched folders.
        analysis_codebooks: Codebooks whose concatenated outputs should be read.
    """

    workspace: Workspace
    rois: tuple[str, ...]
    seg_codebook: str
    analysis_codebooks: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.workspace, Workspace):
            self.workspace = Workspace(self.workspace)

        resolved_rois = tuple(self.rois) if self.rois else tuple(self.workspace.rois)
        if not resolved_rois:
            msg = f"No ROIs discovered in workspace {self.workspace}"
            logger.error(msg)
            raise ConcatDataError(msg)
        self.rois = resolved_rois

        if not self.seg_codebook:
            raise ConcatDataError("seg_codebook must be provided")

        resolved_codebooks = tuple(self.analysis_codebooks)
        if not resolved_codebooks:
            raise ConcatDataError("At least one analysis codebook is required")
        self.analysis_codebooks = resolved_codebooks

    @property
    def stitch_root(self) -> Path:
        return self.workspace.deconved


def _matching_files(directory: Path, pattern: str) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(directory.glob(pattern))


def load_ident_tables(spec: ConcatDataSpec) -> dict[tuple[str, str], pl.DataFrame]:
    """Load stitched identification tables for each ROI/codebook pair."""

    results: MutableMapping[tuple[str, str], pl.DataFrame] = {}
    missing: list[str] = []
    for roi in spec.rois:
        for codebook in spec.analysis_codebooks:
            chunks_dir = spec.stitch_root / f"stitch--{roi}+{spec.seg_codebook}" / f"chunks+{codebook}"
            files = _matching_files(chunks_dir, "ident_*.parquet")
            if not files:
                missing.append(f"{roi}+{codebook}")
                continue

            lazy = pl.scan_parquet(
                str(chunks_dir / "ident_*.parquet"),
                include_file_paths="path",
                missing_columns="insert",
            ).with_columns([
                pl.col("path").str.extract(r"(\d+)\.parquet").cast(pl.UInt16).alias("z"),
                pl.col("path").str.extract(r"chunks\+([^/]+)").alias("codebook"),
                pl.lit(roi).alias("roi"),
                (pl.lit(roi) + pl.col("label").cast(pl.Utf8)).alias("roilabel"),
            ])

            schema_names = lazy.collect_schema().names()

            if "label" not in schema_names:
                raise ConcatDataError(f"Identification parquet missing 'label' column: {chunks_dir}")

            if "spot_id" in schema_names:
                lazy = lazy.with_columns(pl.col("spot_id").cast(pl.UInt32))

            df = lazy.sort("z").collect()
            if df.is_empty():
                missing.append(f"{roi}+{codebook}")
                continue
            results[(roi, codebook)] = df

    if not results:
        joined = ", ".join(missing) if missing else "all ROI/codebook pairs"
        raise ConcatDataError(f"No identification files found for {joined}")

    if missing:
        logger.warning("Missing identification parquet for: {}", ", ".join(missing))

    return dict(results)


def load_intensity_tables(
    spec: ConcatDataSpec,
    *,
    channel: str,
    required: bool = True,
) -> dict[str, pl.DataFrame]:
    """Load per-ROI intensity parquet sets for a given channel."""

    if not channel:
        raise ConcatDataError("intensity channel must be provided")

    results: MutableMapping[str, pl.DataFrame] = {}
    missing: list[str] = []

    for roi in spec.rois:
        intensity_dir = spec.stitch_root / f"stitch--{roi}+{spec.seg_codebook}" / f"intensity_{channel}"
        files = _matching_files(intensity_dir, "intensity-*.parquet")
        if not files:
            missing.append(roi)
            continue

        lazy = pl.scan_parquet(
            str(intensity_dir / "intensity-*.parquet"),
            include_file_paths="path",
            missing_columns="insert",
        ).with_columns([
            pl.col("path").str.extract(r"(\d+)\.parquet").cast(pl.UInt16).alias("z"),
            pl.lit(roi).alias("roi"),
            (pl.lit(roi) + pl.col("label").cast(pl.Utf8)).alias("roilabel"),
        ])

        schema_names = lazy.collect_schema().names()

        if "label" not in schema_names:
            raise ConcatDataError(f"Intensity parquet missing 'label' column: {intensity_dir}")

        lazy = lazy.with_columns([
            (pl.col("z").cast(pl.Utf8) + pl.lit(roi) + pl.col("label").cast(pl.Utf8)).alias("roilabelz"),
        ])

        df = lazy.collect()
        if df.is_empty():
            missing.append(roi)
            continue

        results[roi] = df

    if required and len(results) != len(spec.rois):
        raise ConcatDataError("No intensity files found for: " + ", ".join(sorted(set(missing))))

    if missing and not required:
        logger.info(
            "Skipping intensity channel '{}' for ROIs without data: {}",
            channel,
            ", ".join(sorted(set(missing))),
        )

    return dict(results)


def load_spot_tables(spec: ConcatDataSpec) -> dict[tuple[str, str], pl.DataFrame]:
    """Load per-ROI spot parquet tables keyed by (roi, codebook)."""

    results: MutableMapping[tuple[str, str], pl.DataFrame] = {}
    missing: list[str] = []

    for roi in spec.rois:
        for codebook in spec.analysis_codebooks:
            spots_path = spec.stitch_root / f"{roi}+{codebook}.parquet"
            if not spots_path.exists():
                missing.append(f"{roi}+{codebook}")
                continue

            df = pl.read_parquet(spots_path)
            if "index" in df.columns:
                df = df.rename({"index": "label"})
            if "label" not in df.columns:
                df = df.with_row_index(name="label")

            df = df.with_columns([
                pl.col("label").cast(pl.UInt32),
                pl.lit(roi).alias("roi"),
                pl.lit(codebook).alias("codebook"),
            ])
            results[(roi, codebook)] = df

    if not results:
        joined = ", ".join(missing) if missing else "all ROI/codebook pairs"
        raise ConcatDataError(f"No spot parquet found for {joined}")

    if missing:
        logger.warning("Missing spot parquet for: {}", ", ".join(missing))

    return dict(results)


def load_polygon_tables(
    spec: ConcatDataSpec,
    *,
    codebook: str | None = None,
) -> dict[str, pl.DataFrame]:
    """Load polygon chunk parquet tables per ROI for the requested codebook."""

    target_codebook = codebook or spec.analysis_codebooks[0]
    if codebook and codebook not in spec.analysis_codebooks:
        logger.warning(
            "Requested polygon codebook '{}' not in analysis_codebooks; using anyway.",
            codebook,
        )

    results: MutableMapping[str, pl.DataFrame] = {}
    missing: list[str] = []

    for roi in spec.rois:
        chunks_dir = spec.stitch_root / f"stitch--{roi}+{spec.seg_codebook}" / f"chunks+{target_codebook}"
        files = _matching_files(chunks_dir, "polygons_*.parquet")
        if not files:
            missing.append(roi)
            continue

        lazy = pl.scan_parquet(
            str(chunks_dir / "polygons_*.parquet"),
            include_file_paths="path",
            missing_columns="insert",
        ).with_columns([
            pl.col("path").str.extract(r"(\d+)\.parquet").cast(pl.UInt16).alias("z"),
            pl.lit(roi).alias("roi"),
            (pl.lit(roi) + pl.col("label").cast(pl.Utf8)).alias("roilabel"),
        ])

        schema_names = lazy.collect_schema().names()
        required = {"label", "centroid_x", "centroid_y", "area"}
        missing_cols = sorted(required - set(schema_names))
        if missing_cols:
            raise ConcatDataError(f"Polygon parquet missing required columns {missing_cols}: {chunks_dir}")

        lazy = lazy.with_columns(
            (pl.col("z").cast(pl.Utf8) + pl.lit(roi) + pl.col("label").cast(pl.Utf8)).alias("roilabelz")
        ).drop("path")

        df = lazy.sort("z").collect()
        if df.is_empty():
            missing.append(roi)
            continue

        results[roi] = df

    if not results:
        joined = ", ".join(sorted(set(missing))) if missing else "all rois"
        raise ConcatDataError(f"No polygon parquet found for {joined}")

    if missing:
        logger.warning("Missing polygon parquet for: {}", ", ".join(sorted(set(missing))))

    return dict(results)


def join_ident_with_spots(
    ident_tables: Mapping[tuple[str, str], pl.DataFrame],
    spot_tables: Mapping[tuple[str, str], pl.DataFrame],
) -> pl.DataFrame:
    """Join per-cell identifications with spot coordinates across ROI/codebooks."""

    joined_frames: list[pl.DataFrame] = []
    missing: list[str] = []

    for key, ident_df in ident_tables.items():
        spots_df = spot_tables.get(key)
        if spots_df is None:
            missing.append(f"{key[0]}+{key[1]}")
            continue

        for column in ("label", "codebook"):
            if column not in ident_df.columns:
                raise ConcatDataError(f"Identification table missing '{column}' column for {key}")
            if column not in spots_df.columns:
                raise ConcatDataError(f"Spot table missing '{column}' column for {key}")

        ident_cast = ident_df.with_columns(pl.col("label").cast(pl.UInt32))
        spots_cast = spots_df.with_columns(pl.col("label").cast(pl.UInt32))

        joined = ident_cast.join(spots_cast, on=["label", "codebook"], how="left")
        if joined.is_empty():
            missing.append(f"{key[0]}+{key[1]}")
            continue
        joined_frames.append(joined)

    if not joined_frames:
        joined = ", ".join(missing) if missing else "all ROI/codebook pairs"
        raise ConcatDataError(f"No joined ident/spot data produced for {joined}")

    if missing:
        logger.warning("Missing spot join data for: {}", ", ".join(sorted(set(missing))))

    return pl.concat(joined_frames, how="vertical_relaxed")


def merge_polygons_with_intensity(
    polygons: Mapping[str, pl.DataFrame],
    intensities: Mapping[str, pl.DataFrame],
) -> pl.DataFrame:
    """Attach per-ROI intensity measurements to polygon features."""

    merged_frames: list[pl.DataFrame] = []
    missing: list[str] = []

    for roi, poly_df in polygons.items():
        if "roilabelz" not in poly_df.columns:
            raise ConcatDataError(f"Polygon table missing 'roilabelz' column for {roi}")

        intensity_df = intensities.get(roi)
        if intensity_df is None:
            missing.append(roi)
            continue

        if "roilabelz" not in intensity_df.columns:
            raise ConcatDataError(f"Intensity table missing 'roilabelz' column for {roi}")

        merged = poly_df.join(intensity_df, on="roilabelz", how="left", suffix="_int")
        merged_frames.append(merged)

    if not merged_frames:
        joined = ", ".join(sorted(set(missing))) if missing else "all rois"
        raise ConcatDataError(f"No polygons merged with intensity for {joined}")

    if missing:
        logger.warning(
            "Missing intensity tables for polygons in ROIs: {}",
            ", ".join(sorted(set(missing))),
        )

    return pl.concat(merged_frames, how="vertical_relaxed")


def arrange_rois(
    polygons: pl.DataFrame,
    *,
    max_columns: int = 2,
    padding: float = 100.0,
) -> tuple[pl.DataFrame, dict[str, tuple[float, float]]]:
    """Lay out ROIs on a grid, returning shifted centroids and offsets."""

    required = {"roi", "centroid_x", "centroid_y"}
    missing = sorted(required - set(polygons.columns))
    if missing:
        raise ConcatDataError(f"Polygons missing required columns: {missing}")

    if max_columns < 1:
        raise ValueError("max_columns must be at least 1")

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

    if roi_bounds.is_empty():
        raise ConcatDataError("No polygons available to arrange")

    roi_offsets: dict[str, tuple[float, float]] = {}
    widths: dict[str, float] = {}
    heights: dict[str, float] = {}

    for row in roi_bounds.iter_rows(named=True):
        roi = row["roi"]
        widths[roi] = row["max_x"] - row["min_x"]
        heights[roi] = row["max_y"] - row["min_y"]

    max_width = max(widths.values()) if widths else 0.0
    max_height = max(heights.values()) if heights else 0.0

    for idx, row in enumerate(roi_bounds.iter_rows(named=True)):
        roi = row["roi"]
        grid_row = idx // max_columns
        grid_col = idx % max_columns
        x_offset = grid_col * (max_width + padding) - row["min_x"]
        y_offset = grid_row * (max_height + padding) - row["min_y"]
        roi_offsets[roi] = (float(x_offset), float(y_offset))

    offset_df = pl.DataFrame({
        "roi": list(roi_offsets.keys()),
        "x_offset": [offset[0] for offset in roi_offsets.values()],
        "y_offset": [offset[1] for offset in roi_offsets.values()],
    })

    arranged = (
        polygons.join(offset_df, on="roi", how="left")
        .with_columns(
            (pl.col("centroid_x") + pl.col("x_offset")).alias("centroid_x"),
            (pl.col("centroid_y") + pl.col("y_offset")).alias("centroid_y"),
        )
        .drop("x_offset", "y_offset")
    )

    return arranged, roi_offsets


def compute_weighted_centroids(polygons: pl.DataFrame) -> pl.DataFrame:
    """Compute area-weighted centroids and intensity summaries per ROI label."""

    required = {"roilabel", "roi", "area", "centroid_x", "centroid_y", "z"}
    missing = sorted(required - set(polygons.columns))
    if missing:
        raise ConcatDataError(f"Polygons missing required columns: {missing}")

    weight = pl.col("area")

    aggregations = [
        weight.sum().alias("area"),
        (pl.col("centroid_x") * weight).sum().truediv(weight.sum()).alias("x"),
        (pl.col("centroid_y") * weight).sum().truediv(weight.sum()).alias("y"),
        (pl.col("z").cast(pl.Float64) * weight).sum().truediv(weight.sum()).alias("z"),
        pl.col("roi").first().alias("roi"),
    ]

    if "mean_intensity" in polygons.columns:
        aggregations.append(
            (pl.col("mean_intensity") * weight).sum().truediv(weight.sum()).alias("mean_intensity")
        )
    if "max_intensity" in polygons.columns:
        aggregations.append(pl.col("max_intensity").max().alias("max_intensity"))
    if "min_intensity" in polygons.columns:
        aggregations.append(pl.col("min_intensity").min().alias("min_intensity"))

    return polygons.group_by("roilabel").agg(aggregations).sort("roilabel")
