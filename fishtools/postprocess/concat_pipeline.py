"""
Multi-ROI FISH data concatenation pipeline.

This module implements the core processing pipeline for concatenating multi-ROI FISH data,
including data loading, spatial arrangement, and AnnData construction.
"""

from collections import defaultdict
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import anndata as ad
import numpy as np
import pandas as pd
import polars as pl
import polars.exceptions as pl_exc
import scanpy as sc
from loguru import logger

from fishtools.utils.io import Workspace

from .concat_config import ConcatConfig, validate_config_workspace


class ConcatPipeline:
    """
    Pipeline for concatenating multi-ROI FISH data into a unified AnnData object.

    This class orchestrates the entire concat pipeline, from loading raw parquet files
    to creating the final AnnData object with quality control and spatial arrangement.
    """

    def __init__(self, config: ConcatConfig, validate_workspace: bool = True) -> None:
        """
        Initialize the concat pipeline with configuration.

        Args:
            config: Configuration object containing all pipeline parameters
            validate_workspace: Whether to validate workspace structure exists (default True)
        """
        self.config = config
        self.workspace = config.workspace
        self.rois = config.rois

        # Validate configuration and workspace structure
        if validate_workspace:
            validate_config_workspace(config)

        logger.info(f"Initialized pipeline with {len(self.rois)} ROIs: {self.rois}")

    def run(self) -> ad.AnnData:
        """
        Execute the complete concat pipeline.

        Returns:
            AnnData object containing concatenated multi-ROI data
        """
        logger.info("Starting concat pipeline execution")

        # Load identification data
        logger.info("Loading identification data")
        ident_data = self.load_identification_data()

        # Load intensity data
        logger.info("Loading intensity data")
        intensity_data = self.load_intensity_data()

        # Load and join spots data
        logger.info("Loading and joining spots data")
        joined_data = self.load_and_join_spots_data(ident_data)

        # Export Baysor data if requested
        if self.config.output.baysor_export:
            logger.info("Exporting Baysor-compatible data")
            self.export_baysor_data(joined_data)

        # Load and arrange polygons spatially
        logger.info("Loading polygon data and arranging ROIs spatially")
        arranged_polygons, roi_offsets = self.load_and_arrange_polygons(intensity_data)

        # Calculate weighted centroids
        logger.info("Calculating weighted centroids")
        weighted_centroids = self.calculate_weighted_centroids(arranged_polygons)

        # Create gene expression matrix
        logger.info("Creating gene expression matrix")
        gene_expression_matrix = self.create_gene_expression_matrix(ident_data)

        # Construct AnnData object
        logger.info("Constructing AnnData object")
        adata = self.construct_anndata(gene_expression_matrix, weighted_centroids)

        # Apply quality control filters
        logger.info("Applying quality control filters")
        adata = self.apply_quality_control(adata)

        logger.info(f"Pipeline completed successfully. Final AnnData shape: {adata.shape}")
        return adata

    def load_identification_data(self) -> Dict[Tuple[str, str], pl.DataFrame]:
        """
        Load identification data for all ROI-codebook combinations.

        Returns:
            Dictionary mapping (roi, codebook) tuples to DataFrames
        """
        dfs: Dict[Tuple[str, str], pl.DataFrame] = {}

        for roi, codebook in product(self.rois, self.config.codebooks):
            logger.debug(f"Loading identification data for ROI {roi}, codebook {codebook}")

            glob_path = (
                self.workspace.path
                / f"stitch--{roi}+{self.config.seg_codebook}"
                / f"chunks+{codebook}/ident_*.parquet"
            )

            try:
                df_roi = (
                    pl.scan_parquet(
                        glob_path,
                        include_file_paths="path",
                        allow_missing_columns=True,
                    )
                    .with_columns(
                        z=pl.col("path").str.extract(r"(\d+)\.parquet").cast(pl.UInt8),
                        codebook=pl.col("path").str.extract(r"chunks\+(\w+)"),
                        spot_id=pl.col("spot_id").cast(pl.UInt32),
                        roi=pl.lit(roi),
                    )
                    .with_columns(
                        roilabel=pl.col("roi") + pl.col("label").cast(pl.Utf8),
                    )
                    .sort("z")
                    .collect()
                )

                if not df_roi.is_empty():
                    dfs[(roi, codebook)] = df_roi
                    logger.debug(f"Loaded {len(df_roi)} identification records for {roi}-{codebook}")
                else:
                    logger.warning(f"No identification data found for {roi}-{codebook}")

            except (FileNotFoundError, PermissionError) as e:
                logger.error(f"File access error for {roi}-{codebook}: {e}")
                continue
            except (pl_exc.ComputeError, pl_exc.ColumnNotFoundError) as e:
                logger.error(f"Data processing error for {roi}-{codebook}: {e}")
                continue
            except Exception as e:
                logger.error(f"Unexpected error loading identification data for {roi}-{codebook}: {e}")
                raise

        if not dfs:
            raise ValueError(
                f"No identification files found for any ROI in {self.rois} "
                f"with seg_codebook {self.config.seg_codebook}"
            )

        logger.info(f"Loaded identification data for {len(dfs)} ROI-codebook combinations")
        return dfs

    def load_intensity_data(self) -> Dict[str, pl.DataFrame]:
        """
        Load intensity data for all ROIs.

        Returns:
            Dictionary mapping ROI names to intensity DataFrames
        """
        intensities: Dict[str, pl.DataFrame] = {}

        for roi in self.rois:
            logger.debug(f"Loading intensity data for ROI {roi}")

            glob_path_intensity = (
                self.workspace.path
                / f"stitch--{roi}+{self.config.seg_codebook}"
                / f"intensity_{self.config.intensity.channel}/intensity-*.parquet"
            )

            try:
                intensity_roi = (
                    pl.scan_parquet(
                        glob_path_intensity,
                        include_file_paths="path",
                        allow_missing_columns=True,
                    )
                    .with_columns(
                        z=pl.col("path").str.extract(r"(\d+)\.parquet").cast(pl.UInt8),
                        roi=pl.lit(roi),
                    )
                    .with_columns(
                        roilabel=pl.col("roi") + pl.col("label").cast(pl.Utf8),
                        roilabelz=pl.col("z").cast(pl.Utf8) + pl.col("roi") + pl.col("label").cast(pl.Utf8),
                    )
                    .collect()
                )

                if not intensity_roi.is_empty():
                    intensities[roi] = intensity_roi
                    logger.debug(f"Loaded {len(intensity_roi)} intensity records for {roi}")
                else:
                    logger.warning(f"No intensity data found for ROI {roi}")

            except (FileNotFoundError, PermissionError) as e:
                logger.error(f"File access error for intensity data ROI {roi}: {e}")
                if self.config.intensity.required:
                    raise
                continue
            except (pl_exc.ComputeError, pl_exc.ColumnNotFoundError) as e:
                logger.error(f"Data processing error for intensity data ROI {roi}: {e}")
                if self.config.intensity.required:
                    raise
                continue
            except Exception as e:
                logger.error(f"Unexpected error loading intensity data for ROI {roi}: {e}")
                if self.config.intensity.required:
                    raise
                continue

        if not intensities and self.config.intensity.required:
            raise ValueError(
                f"No intensity files found for any ROI in {self.rois} "
                f"with seg_codebook {self.config.seg_codebook}"
            )

        expected_rois = len(self.rois)
        found_rois = len(intensities)
        if self.config.intensity.required and found_rois != expected_rois:
            raise ValueError(f"Not all ROIs have intensity data. Expected {expected_rois}, got {found_rois}")

        logger.info(f"Loaded intensity data for {len(intensities)} ROIs")
        return intensities

    def load_and_join_spots_data(self, ident_data: Dict[Tuple[str, str], pl.DataFrame]) -> pl.DataFrame:
        """
        Load spots data and join with identification data.

        Args:
            ident_data: Dictionary of identification DataFrames

        Returns:
            Joined DataFrame containing identification and spots data
        """
        joineds: List[pl.DataFrame] = []

        for (roi, codebook), df in ident_data.items():
            logger.debug(f"Processing spots data for ROI {roi}, codebook {codebook}")

            spots_file_path = self.workspace.path / f"{roi}+{codebook}.parquet"

            if not spots_file_path.exists():
                logger.warning(f"Spots file not found, skipping: {spots_file_path}")
                continue

            try:
                current_roi_codebook_spots_df = pl.read_parquet(spots_file_path).with_columns(
                    roi=pl.lit(roi), codebook=pl.lit(codebook)
                )

                # Handle index column naming
                if "index" not in current_roi_codebook_spots_df.columns:
                    current_roi_codebook_spots_df = current_roi_codebook_spots_df.with_row_index(name="label")
                else:
                    current_roi_codebook_spots_df = current_roi_codebook_spots_df.rename({"index": "label"})

                # Join identification and spots data
                joined = df.with_columns(label=pl.col("label").cast(pl.UInt32)).join(
                    current_roi_codebook_spots_df, on=["codebook", "label"], how="left"
                )

                if joined.is_empty():
                    logger.warning(f"No relevant identification data found for {roi}-{codebook}")
                else:
                    joineds.append(joined)
                    logger.debug(f"Joined {len(joined)} records for {roi}-{codebook}")

            except (FileNotFoundError, PermissionError) as e:
                logger.error(f"File access error for spots data {roi}-{codebook}: {e}")
                continue
            except (pl_exc.ComputeError, pl_exc.ColumnNotFoundError) as e:
                logger.error(f"Data processing error for spots data {roi}-{codebook}: {e}")
                continue
            except Exception as e:
                logger.error(f"Unexpected error processing spots data for {roi}-{codebook}: {e}")
                raise

        if not joineds:
            raise ValueError("No spots data could be joined with identification data")

        joined = pl.concat(joineds)
        logger.info(f"Joined spots data: {len(joined)} total records")
        return joined

    def export_baysor_data(self, joined_data: pl.DataFrame) -> None:
        """
        Export data in Baysor-compatible format.

        Args:
            joined_data: Joined DataFrame containing spots and identification data
        """
        baysor_dir = self.workspace.path / "baysor"
        baysor_dir.mkdir(exist_ok=True)

        baysor_output_path = self.workspace.path / self.config.output.baysor_path
        baysor_output_path.parent.mkdir(parents=True, exist_ok=True)

        baysor_data = joined_data.select(
            x="x",
            y="y",
            z="z",
            gene=pl.col("target").str.split("-").list.get(0),
            cell=pl.col("label").fill_null(0),
        ).filter(pl.col("gene") != "Blank")

        baysor_data.write_csv(baysor_output_path)
        logger.info(f"Exported {len(baysor_data)} Baysor spots to {baysor_output_path}")

    def arrange_rois(
        self, polygons: pl.DataFrame, max_columns: int, padding: float
    ) -> Tuple[pl.DataFrame, Dict[str, Tuple[float, float]]]:
        """
        Arrange ROIs in a grid layout with spatial offsets.

        Args:
            polygons: DataFrame containing polygon data
            max_columns: Maximum number of columns in the grid
            padding: Padding between ROIs

        Returns:
            Tuple of arranged polygons DataFrame and ROI offsets dictionary
        """
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

        logger.debug(f"ROI bounds calculated for {len(roi_bounds)} ROIs")

        # Calculate offsets for each ROI in a grid layout
        roi_offsets: Dict[str, Tuple[float, float]] = {}
        max_width = 0.0
        max_height = 0.0

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
        arranged_polygons = polygons.with_columns(
            [
                pl.col("centroid_x")
                + pl.col("roi").map_elements(
                    lambda r: roi_offsets[r][0] if r in roi_offsets else 0.0, return_dtype=pl.Float64
                ),
                pl.col("centroid_y")
                + pl.col("roi").map_elements(
                    lambda r: roi_offsets[r][1] if r in roi_offsets else 0.0, return_dtype=pl.Float64
                ),
            ]
        )

        logger.info(f"Arranged {len(roi_offsets)} ROIs in grid layout")
        return arranged_polygons, roi_offsets

    def load_and_arrange_polygons(
        self, intensity_data: Dict[str, pl.DataFrame]
    ) -> Tuple[pl.DataFrame, Dict[str, Tuple[float, float]]]:
        """
        Load polygon data and arrange ROIs spatially.

        Args:
            intensity_data: Dictionary of intensity DataFrames by ROI

        Returns:
            Tuple of arranged polygons DataFrame and ROI offsets dictionary
        """
        polygons_by_roi: Dict[str, pl.DataFrame] = {}

        for roi in self.rois:
            logger.debug(f"Loading polygon data for ROI {roi}")

            glob_path = (
                self.workspace.path
                / f"stitch--{roi}+{self.config.seg_codebook}"
                / f"chunks+{self.config.codebooks[0]}/polygons_*.parquet"
            )

            try:
                polygon_data = (
                    pl.scan_parquet(glob_path, include_file_paths="path")
                    .with_columns(
                        z=pl.col("path").str.extract(r"(\d+)\.parquet").cast(pl.UInt8), roi=pl.lit(roi)
                    )
                    .with_columns(
                        roilabel=pl.col("roi") + pl.col("label").cast(pl.Utf8),
                        roilabelz=pl.col("z").cast(pl.Utf8) + pl.col("roi") + pl.col("label").cast(pl.Utf8),
                    )
                    .drop("path")
                    .sort("z")
                    .collect()
                )

                if not polygon_data.is_empty():
                    polygons_by_roi[roi] = polygon_data
                    logger.debug(f"Loaded {len(polygon_data)} polygon records for {roi}")
                else:
                    logger.warning(f"No polygon data found for ROI {roi}")

            except (FileNotFoundError, PermissionError) as e:
                logger.error(f"File access error for polygon data ROI {roi}: {e}")
                continue
            except (pl_exc.ComputeError, pl_exc.ColumnNotFoundError) as e:
                logger.error(f"Data processing error for polygon data ROI {roi}: {e}")
                continue
            except Exception as e:
                logger.error(f"Unexpected error loading polygon data for ROI {roi}: {e}")
                raise

        if not polygons_by_roi:
            raise ValueError(f"No polygon data found for any ROI in {self.rois}")

        # Join with intensity data
        for roi in polygons_by_roi:
            if roi in intensity_data:
                polygons_by_roi[roi] = polygons_by_roi[roi].join(intensity_data[roi], on="roilabelz")
            else:
                logger.warning(f"No intensity data available for ROI {roi}")

        # Concatenate all polygon data
        polygons = pl.concat(polygons_by_roi.values())

        # Arrange ROIs spatially
        arranged_polygons, roi_offsets = self.arrange_rois(
            polygons, self.config.spatial.max_columns, self.config.spatial.padding
        )

        logger.info(f"Loaded and arranged polygon data: {len(arranged_polygons)} total records")
        return arranged_polygons, roi_offsets

    def calculate_weighted_centroids(self, polygons: pl.DataFrame) -> pd.DataFrame:
        """
        Calculate area-weighted centroids for each cell.

        Args:
            polygons: DataFrame containing polygon data with areas

        Returns:
            DataFrame with weighted centroids indexed by roilabel
        """
        # Validate input data
        if polygons.is_empty():
            raise ValueError("No polygon data provided for centroid calculation")

        # Check for valid areas and log warnings
        zero_area_count = polygons.filter(pl.col("area") <= 0).height
        if zero_area_count > 0:
            logger.warning(f"Found {zero_area_count} polygons with zero/negative area, filtering them out")

        # Filter out invalid areas before processing
        valid_polygons = polygons.filter(pl.col("area") > 0)

        if valid_polygons.is_empty():
            raise ValueError("No polygons with valid areas found")

        weighted_centroids = (
            valid_polygons.group_by("roilabel")
            .agg(
                area=pl.col("area").sum(),
                # Safe division with fallback to simple mean for zero total area
                x=pl.when(pl.col("area").sum() > 0)
                .then((pl.col("centroid_x") * pl.col("area")).sum() / pl.col("area").sum())
                .otherwise(pl.col("centroid_x").mean()),
                y=pl.when(pl.col("area").sum() > 0)
                .then((pl.col("centroid_y") * pl.col("area")).sum() / pl.col("area").sum())
                .otherwise(pl.col("centroid_y").mean()),
                z=pl.when(pl.col("area").sum() > 0)
                .then((pl.col("z").cast(pl.Float64) * pl.col("area")).sum() / pl.col("area").sum())
                .otherwise(pl.col("z").cast(pl.Float64).mean()),
                mean_intensity=pl.when(pl.col("area").sum() > 0)
                .then((pl.col("mean_intensity") * pl.col("area")).sum() / pl.col("area").sum())
                .otherwise(pl.col("mean_intensity").mean()),
                max_intensity=pl.col("max_intensity").max(),
                min_intensity=pl.col("min_intensity").min(),
                roi=pl.col("roi").first(),
            )
            .sort("roilabel")
        )

        # Convert to pandas and validate results
        weighted_centroids_pd = weighted_centroids.to_pandas().set_index("roilabel")
        weighted_centroids_pd.index = weighted_centroids_pd.index.astype(str)

        # Validate numerical results
        for col in ["x", "y", "z", "mean_intensity"]:
            if col in weighted_centroids_pd.columns:
                non_finite_count = (~np.isfinite(weighted_centroids_pd[col])).sum()
                if non_finite_count > 0:
                    logger.error(f"Found {non_finite_count} non-finite values in column {col}")
                    raise ValueError(f"Non-finite values detected in weighted centroids column: {col}")

                # Check for extreme values that might indicate numerical issues
                extreme_count = (np.abs(weighted_centroids_pd[col]) > 1e15).sum()
                if extreme_count > 0:
                    logger.warning(f"Found {extreme_count} extreme values in column {col}")

        logger.info(f"Calculated weighted centroids for {len(weighted_centroids_pd)} cells")
        return weighted_centroids_pd

    def create_gene_expression_matrix(self, ident_data: Dict[Tuple[str, str], pl.DataFrame]) -> pd.DataFrame:
        """
        Create gene expression count matrix from identification data.

        Args:
            ident_data: Dictionary of identification DataFrames

        Returns:
            Gene expression matrix as pandas DataFrame
        """
        # Concatenate all identification data
        df = pl.concat(ident_data.values())

        # Aggregate counts by cell and target
        molten = df.group_by(["roilabel", "target"]).agg(pl.len())

        # Process gene names and create count matrix
        gene_expression_matrix = (
            molten.with_columns(
                gene_name=pl.when(
                    pl.col("target").str.contains(r"-2\d+").and_(~pl.col("target").str.starts_with("Blank"))
                )
                .then(pl.col("target").str.split("-").list.get(0))
                .otherwise(pl.col("target"))
            )
            .drop("target")
            .pivot(on="gene_name", index="roilabel", values="len")
            .fill_null(0)
            .sort("roilabel")
            .to_pandas()
            .set_index("roilabel")
        )

        gene_expression_matrix.index = gene_expression_matrix.index.astype(str)
        logger.info(f"Created gene expression matrix: {gene_expression_matrix.shape}")
        return gene_expression_matrix

    def construct_anndata(
        self, gene_expression_matrix: pd.DataFrame, weighted_centroids: pd.DataFrame
    ) -> ad.AnnData:
        """
        Construct AnnData object from gene expression matrix and metadata.

        Args:
            gene_expression_matrix: Gene expression count matrix
            weighted_centroids: Cell metadata with spatial coordinates

        Returns:
            AnnData object with expression data and metadata
        """
        adata = ad.AnnData(gene_expression_matrix)
        adata.obs = weighted_centroids.reindex(adata.obs.index)

        # Calculate quality control metrics
        n_genes = adata.shape[1]
        sc.pp.calculate_qc_metrics(
            adata, inplace=True, percent_top=(n_genes // 10, n_genes // 5, n_genes // 2, n_genes)
        )

        # Add spatial coordinates
        adata.obsm["spatial"] = adata.obs[["x", "y"]].to_numpy()

        # Store raw counts
        adata.layers["raw"] = adata.X.copy()

        logger.info(f"Constructed AnnData object: {adata.shape}")
        return adata

    def apply_quality_control(self, adata: ad.AnnData) -> ad.AnnData:
        """
        Apply quality control filters to AnnData object.

        Args:
            adata: Input AnnData object

        Returns:
            Filtered AnnData object
        """
        initial_cells = adata.n_obs
        initial_genes = adata.n_vars

        # Filter cells by count thresholds
        sc.pp.filter_cells(adata, min_counts=self.config.quality_control.min_counts)
        sc.pp.filter_cells(adata, max_counts=self.config.quality_control.max_counts)

        # Filter genes by minimum cells
        sc.pp.filter_genes(adata, min_cells=self.config.quality_control.min_cells)

        # Filter cells by minimum area
        adata = adata[adata.obs["area"] > self.config.quality_control.min_area]

        final_cells = adata.n_obs
        final_genes = adata.n_vars

        logger.info(
            f"Quality control completed: "
            f"cells {initial_cells} → {final_cells} "
            f"({100 * final_cells / initial_cells:.1f}%), "
            f"genes {initial_genes} → {final_genes} "
            f"({100 * final_genes / initial_genes:.1f}%)"
        )

        return adata
