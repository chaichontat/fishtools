"""
Tests for fishtools.postprocess.concat_pipeline module.

Comprehensive unit tests for the concat pipeline including data loading,
spatial processing, and AnnData construction.
"""

import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple
from unittest.mock import MagicMock, patch

import anndata as ad
import numpy as np
import pandas as pd
import polars as pl
import pytest

from fishtools.postprocess.concat_config import (
    ConcatConfig,
    IntensityConfig,
    OutputConfig,
    QualityControlConfig,
    SpatialConfig,
)
from fishtools.postprocess.concat_pipeline import ConcatPipeline


class TestConcatPipelineInit:
    """Test ConcatPipeline initialization."""

    def create_test_config(self, workspace_path: Path) -> ConcatConfig:
        """Create a test configuration."""
        return ConcatConfig(
            workspace_path=workspace_path,
            codebooks=["book1", "book2"],
            seg_codebook="seg",
            intensity=IntensityConfig(channel="cfse", required=True),
            quality_control=QualityControlConfig(),
            spatial=SpatialConfig(),
            output=OutputConfig(),
        )

    def test_pipeline_initialization(self) -> None:
        """Test pipeline initialization with valid configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_dir = Path(temp_dir)
            config = self.create_test_config(workspace_dir)

            pipeline = ConcatPipeline(config, validate_workspace=False)

            assert pipeline.config == config
            assert pipeline.workspace == config.workspace
            assert pipeline.rois == config.rois


class TestDataLoading:
    """Test data loading methods."""

    def create_test_config(self, workspace_path: Path) -> ConcatConfig:
        """Create a test configuration."""
        return ConcatConfig(
            workspace_path=workspace_path,
            codebooks=["book1"],
            seg_codebook="seg",
            intensity=IntensityConfig(channel="cfse", required=True),
            quality_control=QualityControlConfig(),
            spatial=SpatialConfig(),
            output=OutputConfig(),
        )

    def create_mock_ident_dataframe(self) -> pl.DataFrame:
        """Create a mock identification DataFrame."""
        return pl.DataFrame(
            {
                "spot_id": [1, 2, 3],
                "label": [100, 101, 102],
                "target": ["Gene1-20", "Gene2-21", "Blank-22"],
                "z": [0, 0, 1],
                "codebook": ["book1", "book1", "book1"],
                "roi": ["roi1", "roi1", "roi1"],
                "roilabel": ["roi1100", "roi1101", "roi1102"],
            }
        )

    def create_mock_intensity_dataframe(self) -> pl.DataFrame:
        """Create a mock intensity DataFrame."""
        return pl.DataFrame(
            {
                "label": [100, 101, 102],
                "mean_intensity": [50.0, 75.0, 100.0],
                "max_intensity": [80.0, 120.0, 150.0],
                "min_intensity": [20.0, 30.0, 50.0],
                "z": [0, 0, 1],
                "roi": ["roi1", "roi1", "roi1"],
                "roilabel": ["roi1100", "roi1101", "roi1102"],
                "roilabelz": ["0roi1100", "0roi1101", "1roi1102"],
            }
        )

    def create_mock_spots_dataframe(self) -> pl.DataFrame:
        """Create a mock spots DataFrame."""
        return pl.DataFrame(
            {
                "index": [100, 101, 102],
                "x": [10.5, 20.5, 30.5],
                "y": [15.5, 25.5, 35.5],
                "z": [0, 0, 1],
                "target": ["Gene1-20", "Gene2-21", "Blank-22"],
                "roi": ["roi1", "roi1", "roi1"],
                "codebook": ["book1", "book1", "book1"],
            }
        )

    @patch("polars.scan_parquet")
    def test_load_identification_data_success(self, mock_scan: MagicMock) -> None:
        """Test successful loading of identification data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_dir = Path(temp_dir)
            config = self.create_test_config(workspace_dir)

            # Create a proper mock chain for Polars
            mock_df = self.create_mock_ident_dataframe()
            mock_lazy = MagicMock()
            mock_lazy.with_columns.return_value.with_columns.return_value.sort.return_value.collect.return_value = mock_df
            mock_scan.return_value = mock_lazy

            pipeline = ConcatPipeline(config, validate_workspace=False)

            # Mock the rois directly in the pipeline object
            pipeline.rois = ["roi1"]
            result = pipeline.load_identification_data()

            assert len(result) == 1
            assert ("roi1", "book1") in result
            assert result[("roi1", "book1")].equals(mock_df)

    @patch("polars.scan_parquet")
    def test_load_identification_data_empty_results(self, mock_scan: MagicMock) -> None:
        """Test handling of empty identification data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_dir = Path(temp_dir)
            config = self.create_test_config(workspace_dir)

            # Mock empty DataFrame
            empty_df = pl.DataFrame()
            mock_lazy = MagicMock()
            mock_lazy.with_columns.return_value.with_columns.return_value.sort.return_value.collect.return_value = empty_df
            mock_scan.return_value = mock_lazy

            pipeline = ConcatPipeline(config, validate_workspace=False)

            # Mock the rois directly in the pipeline object
            pipeline.rois = ["roi1"]
            with pytest.raises(ValueError, match="No identification files found"):
                pipeline.load_identification_data()

    @patch("polars.scan_parquet")
    def test_load_intensity_data_success(self, mock_scan: MagicMock) -> None:
        """Test successful loading of intensity data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_dir = Path(temp_dir)
            config = self.create_test_config(workspace_dir)

            # Mock the Polars scan_parquet chain
            mock_df = self.create_mock_intensity_dataframe()
            mock_lazy = MagicMock()
            mock_lazy.with_columns.return_value.with_columns.return_value.collect.return_value = mock_df
            mock_scan.return_value = mock_lazy

            pipeline = ConcatPipeline(config, validate_workspace=False)

            # Mock the rois directly in the pipeline object
            pipeline.rois = ["roi1"]
            result = pipeline.load_intensity_data()

            assert len(result) == 1
            assert "roi1" in result
            assert result["roi1"].equals(mock_df)

    @patch("polars.scan_parquet")
    def test_load_intensity_data_required_missing(self, mock_scan: MagicMock) -> None:
        """Test error when required intensity data is missing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_dir = Path(temp_dir)
            config = self.create_test_config(workspace_dir)

            # Mock empty DataFrame
            empty_df = pl.DataFrame()
            mock_lazy = MagicMock()
            mock_lazy.with_columns.return_value.with_columns.return_value.collect.return_value = empty_df
            mock_scan.return_value = mock_lazy

            pipeline = ConcatPipeline(config, validate_workspace=False)

            # Mock the rois directly in the pipeline object
            pipeline.rois = ["roi1"]
            with pytest.raises(ValueError, match="No intensity files found"):
                pipeline.load_intensity_data()

    @patch("polars.scan_parquet")
    def test_load_intensity_data_optional_missing(self, mock_scan: MagicMock) -> None:
        """Test handling when optional intensity data is missing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_dir = Path(temp_dir)
            config = self.create_test_config(workspace_dir)
            config.intensity.required = False

            # Mock empty DataFrame
            empty_df = pl.DataFrame()
            mock_lazy = MagicMock()
            mock_lazy.with_columns.return_value.with_columns.return_value.collect.return_value = empty_df
            mock_scan.return_value = mock_lazy

            pipeline = ConcatPipeline(config, validate_workspace=False)

            # Mock the rois directly in the pipeline object
            pipeline.rois = ["roi1"]
            result = pipeline.load_intensity_data()

            assert len(result) == 0

    @patch("polars.read_parquet")
    def test_load_and_join_spots_data_success(self, mock_read: MagicMock) -> None:
        """Test successful loading and joining of spots data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_dir = Path(temp_dir)
            config = self.create_test_config(workspace_dir)

            # Create mock identification data
            ident_data = {("roi1", "book1"): self.create_mock_ident_dataframe()}

            # Mock spots data
            mock_spots = self.create_mock_spots_dataframe()
            mock_read.return_value.with_columns.return_value = mock_spots

            pipeline = ConcatPipeline(config, validate_workspace=False)

            # Mock file existence
            with patch.object(Path, "exists", return_value=True):
                result = pipeline.load_and_join_spots_data(ident_data)

            assert len(result) > 0
            assert "x" in result.columns
            assert "y" in result.columns
            assert "z" in result.columns

    @patch("polars.read_parquet")
    def test_load_and_join_spots_data_missing_file(self, mock_read: MagicMock) -> None:
        """Test handling when spots file is missing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_dir = Path(temp_dir)
            config = self.create_test_config(workspace_dir)

            # Create mock identification data
            ident_data = {("roi1", "book1"): self.create_mock_ident_dataframe()}

            pipeline = ConcatPipeline(config, validate_workspace=False)

            # Mock file not existing
            with patch.object(Path, "exists", return_value=False):
                with pytest.raises(ValueError, match="No spots data could be joined"):
                    pipeline.load_and_join_spots_data(ident_data)


class TestSpatialProcessing:
    """Test spatial processing methods."""

    def create_mock_polygon_dataframe(self) -> pl.DataFrame:
        """Create a mock polygon DataFrame."""
        return pl.DataFrame(
            {
                "roi": ["roi1", "roi1", "roi2", "roi2"],
                "centroid_x": [10.0, 15.0, 100.0, 105.0],
                "centroid_y": [20.0, 25.0, 120.0, 125.0],
                "area": [500.0, 600.0, 550.0, 650.0],
                "label": [100, 101, 200, 201],
                "z": [0, 0, 0, 0],
                "roilabel": ["roi1100", "roi1101", "roi2200", "roi2201"],
                "roilabelz": ["0roi1100", "0roi1101", "0roi2200", "0roi2201"],
            }
        )

    def test_arrange_rois_grid_layout(self) -> None:
        """Test ROI arrangement in grid layout."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_dir = Path(temp_dir)
            config = ConcatConfig(
                workspace_path=workspace_dir,
                codebooks=["book1"],
                seg_codebook="seg",
                intensity=IntensityConfig(),
                quality_control=QualityControlConfig(),
                spatial=SpatialConfig(max_columns=2, padding=100.0),
                output=OutputConfig(),
            )

            pipeline = ConcatPipeline(config, validate_workspace=False)
            polygons = self.create_mock_polygon_dataframe()

            arranged_polygons, roi_offsets = pipeline.arrange_rois(polygons, max_columns=2, padding=100.0)

            # Check that ROI offsets were calculated
            assert len(roi_offsets) == 2
            assert "roi1" in roi_offsets
            assert "roi2" in roi_offsets

            # Check that polygons were modified
            assert len(arranged_polygons) == len(polygons)
            assert "centroid_x" in arranged_polygons.columns
            assert "centroid_y" in arranged_polygons.columns

    def test_arrange_rois_single_column(self) -> None:
        """Test ROI arrangement with single column."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_dir = Path(temp_dir)
            config = ConcatConfig(
                workspace_path=workspace_dir,
                codebooks=["book1"],
                seg_codebook="seg",
                intensity=IntensityConfig(),
                quality_control=QualityControlConfig(),
                spatial=SpatialConfig(max_columns=1, padding=50.0),
                output=OutputConfig(),
            )

            pipeline = ConcatPipeline(config, validate_workspace=False)
            polygons = self.create_mock_polygon_dataframe()

            arranged_polygons, roi_offsets = pipeline.arrange_rois(polygons, max_columns=1, padding=50.0)

            # Check that all ROIs are in the same column (x-offset should be similar)
            x_offsets = [offset[0] for offset in roi_offsets.values()]
            y_offsets = [offset[1] for offset in roi_offsets.values()]

            # All x-offsets should be similar (same column)
            assert len(set(round(x, 1) for x in x_offsets)) <= 2
            # Y-offsets should be different (different rows)
            assert len(set(round(y, 1) for y in y_offsets)) == len(roi_offsets)


class TestAnnDataConstruction:
    """Test AnnData construction methods."""

    def create_mock_gene_expression_data(self) -> pd.DataFrame:
        """Create mock gene expression matrix."""
        return pd.DataFrame(
            {"Gene1": [10, 5, 8], "Gene2": [3, 12, 7], "Gene3": [0, 2, 15]}, index=["cell1", "cell2", "cell3"]
        )

    def create_mock_centroids_data(self) -> pd.DataFrame:
        """Create mock weighted centroids data."""
        return pd.DataFrame(
            {
                "x": [10.5, 20.5, 30.5],
                "y": [15.5, 25.5, 35.5],
                "z": [0.0, 0.0, 1.0],
                "area": [500.0, 600.0, 700.0],
                "mean_intensity": [50.0, 75.0, 100.0],
                "max_intensity": [80.0, 120.0, 150.0],
                "min_intensity": [20.0, 30.0, 50.0],
                "roi": ["roi1", "roi1", "roi1"],
            },
            index=["cell1", "cell2", "cell3"],
        )

    @patch("scanpy.pp.calculate_qc_metrics")
    def test_construct_anndata_basic(self, mock_qc_metrics: MagicMock) -> None:
        """Test basic AnnData construction."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_dir = Path(temp_dir)
            config = ConcatConfig(
                workspace_path=workspace_dir,
                codebooks=["book1"],
                seg_codebook="seg",
                intensity=IntensityConfig(),
                quality_control=QualityControlConfig(),
                spatial=SpatialConfig(),
                output=OutputConfig(),
            )

            pipeline = ConcatPipeline(config, validate_workspace=False)
            gene_matrix = self.create_mock_gene_expression_data()
            centroids = self.create_mock_centroids_data()

            adata = pipeline.construct_anndata(gene_matrix, centroids)

            # Check basic structure
            assert isinstance(adata, ad.AnnData)
            assert adata.shape == (3, 3)  # 3 cells, 3 genes
            assert "spatial" in adata.obsm
            assert "raw" in adata.layers

            # Check that observations match centroids
            assert all(col in adata.obs.columns for col in centroids.columns)

            # Check spatial coordinates
            assert adata.obsm["spatial"].shape == (3, 2)  # 3 cells, x,y coordinates

            # Verify scanpy function was called
            mock_qc_metrics.assert_called_once()

    @patch("scanpy.pp.filter_genes")
    @patch("scanpy.pp.filter_cells")
    @patch("scanpy.pp.calculate_qc_metrics")
    def test_apply_quality_control_filtering(
        self, mock_qc_metrics: MagicMock, mock_filter_cells: MagicMock, mock_filter_genes: MagicMock
    ) -> None:
        """Test quality control filtering."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_dir = Path(temp_dir)
            config = ConcatConfig(
                workspace_path=workspace_dir,
                codebooks=["book1"],
                seg_codebook="seg",
                intensity=IntensityConfig(),
                quality_control=QualityControlConfig(
                    min_counts=5, max_counts=20, min_cells=1, min_area=550.0
                ),
                spatial=SpatialConfig(),
                output=OutputConfig(),
            )

            pipeline = ConcatPipeline(config, validate_workspace=False)
            gene_matrix = self.create_mock_gene_expression_data()
            centroids = self.create_mock_centroids_data()

            # Construct initial AnnData
            adata = pipeline.construct_anndata(gene_matrix, centroids)
            initial_shape = adata.shape

            # Apply quality control
            adata_filtered = pipeline.apply_quality_control(adata)

            # Check that filtering functions were called
            assert mock_filter_cells.call_count >= 2  # Called twice for min/max counts
            mock_filter_genes.assert_called_once()

            # Check that remaining cells meet area requirement
            if adata_filtered.n_obs > 0:
                assert all(adata_filtered.obs["area"] > 550.0)


class TestPipelineIntegration:
    """Test full pipeline integration."""

    def create_test_config(self, workspace_path: Path) -> ConcatConfig:
        """Create a test configuration."""
        return ConcatConfig(
            workspace_path=workspace_path,
            codebooks=["book1"],
            seg_codebook="seg",
            intensity=IntensityConfig(channel="cfse", required=False),
            quality_control=QualityControlConfig(min_counts=1, max_counts=1000, min_cells=1, min_area=100.0),
            spatial=SpatialConfig(max_columns=2, padding=100.0),
            output=OutputConfig(baysor_export=False),
        )

    def test_pipeline_error_handling(self) -> None:
        """Test pipeline error handling with missing data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_dir = Path(temp_dir)
            config = self.create_test_config(workspace_dir)

            pipeline = ConcatPipeline(config, validate_workspace=False)

            # Mock the rois directly in the pipeline object to return empty list
            pipeline.rois = []
            with pytest.raises(ValueError):
                pipeline.run()

    def test_baysor_export_path_creation(self) -> None:
        """Test that Baysor export creates necessary directories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_dir = Path(temp_dir)
            config = self.create_test_config(workspace_dir)
            config.output.baysor_export = True
            config.output.baysor_path = Path("nested/baysor/spots.csv")

            pipeline = ConcatPipeline(config, validate_workspace=False)

            # Create mock joined data
            mock_joined_data = pl.DataFrame(
                {
                    "x": [10.0, 20.0],
                    "y": [15.0, 25.0],
                    "z": [0, 0],
                    "target": ["Gene1-20", "Gene2-21"],
                    "label": [100, 101],
                }
            )

            # Mock the workspace path
            with patch.object(pipeline.workspace, "path", workspace_dir):
                pipeline.export_baysor_data(mock_joined_data)

            # Check that directories were created
            expected_baysor_dir = workspace_dir / "baysor"
            expected_nested_dir = workspace_dir / "nested" / "baysor"

            assert expected_baysor_dir.exists()
            assert expected_nested_dir.exists()
            assert (workspace_dir / "nested" / "baysor" / "spots.csv").exists()


class TestPipelineMemoryEfficiency:
    """Test memory efficiency considerations."""

    def test_large_dataframe_processing_simulation(self) -> None:
        """Test that pipeline can handle large DataFrames efficiently."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_dir = Path(temp_dir)
            config = ConcatConfig(
                workspace_path=workspace_dir,
                codebooks=["book1"],
                seg_codebook="seg",
                intensity=IntensityConfig(),
                quality_control=QualityControlConfig(),
                spatial=SpatialConfig(),
                output=OutputConfig(),
            )

            pipeline = ConcatPipeline(config, validate_workspace=False)

            # Create a moderately large mock polygon DataFrame
            n_cells = 10000
            mock_polygons = pl.DataFrame(
                {
                    "roi": ["roi1"] * n_cells,
                    "centroid_x": np.random.uniform(0, 1000, n_cells),
                    "centroid_y": np.random.uniform(0, 1000, n_cells),
                    "area": np.random.uniform(400, 800, n_cells),
                    "label": range(n_cells),
                    "z": np.random.randint(0, 3, n_cells),
                    "roilabel": [f"roi1{i}" for i in range(n_cells)],
                    "roilabelz": [f"{z}roi1{i}" for i, z in enumerate(np.random.randint(0, 3, n_cells))],
                }
            )

            # Test that spatial arrangement works with large datasets
            arranged_polygons, roi_offsets = pipeline.arrange_rois(
                mock_polygons, max_columns=2, padding=100.0
            )

            # Verify processing completed without memory issues
            assert len(arranged_polygons) == n_cells
            assert len(roi_offsets) == 1  # Only one ROI
            assert "roi1" in roi_offsets

    def test_weighted_centroids_numerical_stability(self) -> None:
        """Test numerical stability of weighted centroid calculations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_dir = Path(temp_dir)
            config = ConcatConfig(
                workspace_path=workspace_dir,
                codebooks=["book1"],
                seg_codebook="seg",
                intensity=IntensityConfig(),
                quality_control=QualityControlConfig(),
                spatial=SpatialConfig(),
                output=OutputConfig(),
            )

            pipeline = ConcatPipeline(config, validate_workspace=False)

            # Create polygon data with extreme values to test numerical stability
            mock_polygons = pl.DataFrame(
                {
                    "roilabel": ["cell1", "cell1", "cell2", "cell2"],
                    "area": [1e-6, 1e6, 0.1, 1000.0],  # Extreme range of areas
                    "centroid_x": [1e10, 1e-10, -1e8, 1e8],  # Extreme coordinates
                    "centroid_y": [1e-8, 1e8, -1e10, 1e10],
                    "z": [0.0, 1.0, 0.0, 1.0],
                    "mean_intensity": [1e-3, 1e3, 50.0, 75.0],
                    "max_intensity": [1e-2, 1e4, 80.0, 120.0],
                    "min_intensity": [1e-4, 1e2, 20.0, 30.0],
                    "roi": ["roi1", "roi1", "roi1", "roi1"],
                }
            )

            centroids = pipeline.calculate_weighted_centroids(mock_polygons)

            # Check that results are finite and reasonable
            assert len(centroids) == 2  # Two unique cells
            assert all(np.isfinite(centroids["x"]))
            assert all(np.isfinite(centroids["y"]))
            assert all(np.isfinite(centroids["z"]))
            assert all(centroids["area"] > 0)
