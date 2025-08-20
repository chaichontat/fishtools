"""
End-to-end integration tests for the concat pipeline.

These tests demonstrate the complete workflow from configuration to final output.
"""

import tempfile
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import anndata as ad
import numpy as np
import pandas as pd
import polars as pl
import pytest
import toml
from click.testing import CliRunner

from fishtools.postprocess.cli_concat import concat
from fishtools.postprocess.concat_config import (
    ConcatConfig,
    IntensityConfig,
    OutputConfig,
    QualityControlConfig,
    SpatialConfig,
)
from fishtools.postprocess.concat_pipeline import ConcatPipeline


class TestConcatE2E:
    """End-to-end integration tests for the concat pipeline."""

    def create_realistic_config_data(self) -> Dict[str, Any]:
        """Create realistic configuration data for testing."""
        return {
            "workspace_path": "/tmp/test_experiment",
            "codebooks": ["mousecommon", "zachDE"],
            "seg_codebook": "atp",
            "intensity": {
                "channel": "cfse",
                "required": False,  # Make optional for testing
            },
            "quality_control": {"min_counts": 10, "max_counts": 500, "min_cells": 5, "min_area": 100.0},
            "spatial": {"max_columns": 2, "padding": 50.0},
            "output": {
                "h5ad_path": "test_output.h5ad",
                "baysor_export": True,
                "baysor_path": "baysor/test_spots.csv",
            },
        }

    def create_mock_data_pipeline(self) -> Dict[str, Any]:
        """Create comprehensive mock data for pipeline testing."""
        # Create consistent cell labels for joins
        cell_labels = list(range(1, 21))  # 20 cells total

        # Mock identification data - ensure consistent roilabels
        ident_records = []
        for roi_idx, roi in enumerate(["roi1", "roi2"]):
            for codebook in ["mousecommon", "zachDE"]:
                for cell_label in cell_labels:
                    # Create multiple spots per cell
                    for spot_idx in range(5):  # 5 spots per cell per codebook
                        ident_records.append(
                            {
                                "spot_id": len(ident_records),
                                "label": cell_label,
                                "target": f"Gene{(cell_label + spot_idx) % 10}",  # Consistent gene naming
                                "z": np.random.randint(0, 3),
                                "codebook": codebook,
                                "roi": roi,
                                "roilabel": f"{roi}{cell_label}",
                            }
                        )

        ident_df = pl.DataFrame(ident_records)

        # Mock spots data - one record per cell per codebook
        spots_records = []
        for roi in ["roi1", "roi2"]:
            for codebook in ["mousecommon", "zachDE"]:
                for cell_label in cell_labels:
                    spots_records.append(
                        {
                            "index": cell_label,  # Use index as expected by pipeline
                            "x": np.random.uniform(0, 1000),
                            "y": np.random.uniform(0, 1000),
                            "z": np.random.randint(0, 3),
                            "target": f"Gene{cell_label % 10}",
                            "roi": roi,
                            "codebook": codebook,
                        }
                    )

        spots_df = pl.DataFrame(spots_records)

        # Mock polygon data - one record per cell with consistent roilabels
        polygon_records = []
        for roi in ["roi1", "roi2"]:
            for cell_label in cell_labels:
                for z in range(3):  # Multiple z-slices per cell
                    polygon_records.append(
                        {
                            "label": cell_label,
                            "centroid_x": np.random.uniform(100, 900),
                            "centroid_y": np.random.uniform(100, 900),
                            "area": np.random.uniform(200, 800),
                            "z": z,
                            "roi": roi,
                            "roilabel": f"{roi}{cell_label}",
                            "roilabelz": f"{z}{roi}{cell_label}",
                            "mean_intensity": np.random.uniform(50, 150),
                            "max_intensity": np.random.uniform(100, 200),
                            "min_intensity": np.random.uniform(10, 50),
                        }
                    )

        polygon_df = pl.DataFrame(polygon_records)

        return {"ident": ident_df, "spots": spots_df, "polygons": polygon_df}

    def test_complete_pipeline_workflow(self) -> None:
        """Test complete pipeline workflow with realistic file system setup."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_dir = Path(temp_dir) / "test_experiment"
            workspace_dir.mkdir()

            # Create realistic directory structure
            for roi in ["roi1", "roi2"]:
                roi_dir = workspace_dir / f"stitch--{roi}+atp"
                for codebook in ["mousecommon", "zachDE"]:
                    codebook_dir = roi_dir / f"chunks+{codebook}"
                    codebook_dir.mkdir(parents=True)

                    # Create polygon directory for first codebook only (as per original logic)
                    if codebook == "mousecommon":
                        polygon_dir = roi_dir / f"chunks+{codebook}"
                        polygon_dir.mkdir(exist_ok=True)

            config_data = self.create_realistic_config_data()
            config_data["workspace_path"] = str(workspace_dir)
            config = ConcatConfig(**config_data)

            # Create realistic mock data
            mock_data = self.create_mock_data_pipeline()

            # Create actual parquet files
            for roi in ["roi1", "roi2"]:
                for codebook in ["mousecommon", "zachDE"]:
                    # Create ident files
                    ident_dir = workspace_dir / f"stitch--{roi}+atp" / f"chunks+{codebook}"
                    ident_file = ident_dir / "ident_0.parquet"
                    roi_ident_data = mock_data["ident"].filter(
                        (pl.col("roi") == roi) & (pl.col("codebook") == codebook)
                    )
                    roi_ident_data.write_parquet(ident_file)

                    # Create spots files at workspace root
                    spots_file = workspace_dir / f"{roi}+{codebook}.parquet"
                    roi_spots_data = mock_data["spots"].filter(
                        (pl.col("roi") == roi) & (pl.col("codebook") == codebook)
                    )
                    roi_spots_data.write_parquet(spots_file)

                # Create polygon files (only for first codebook as per original logic)
                polygon_dir = workspace_dir / f"stitch--{roi}+atp" / f"chunks+mousecommon"
                polygon_file = polygon_dir / "polygons_0.parquet"
                roi_polygon_data = mock_data["polygons"].filter(pl.col("roi") == roi)
                roi_polygon_data.write_parquet(polygon_file)

            # Test pipeline without mocking
            pipeline = ConcatPipeline(config, validate_workspace=False)
            pipeline.rois = ["roi1", "roi2"]  # Set ROIs directly

            # Test individual pipeline components
            ident_data = pipeline.load_identification_data()
            assert len(ident_data) > 0
            assert len(ident_data) == 4  # 2 ROIs × 2 codebooks

            # Test spots data loading and joining
            joined_data = pipeline.load_and_join_spots_data(ident_data)
            assert len(joined_data) > 0

            # Test spatial arrangement
            arranged_polygons, roi_offsets = pipeline.arrange_rois(
                mock_data["polygons"], max_columns=2, padding=50.0
            )
            assert len(roi_offsets) == 2  # Should have offsets for both ROIs

            # Test weighted centroids calculation
            centroids = pipeline.calculate_weighted_centroids(arranged_polygons)
            assert len(centroids) > 0
            assert all(np.isfinite(centroids["x"]))
            assert all(np.isfinite(centroids["y"]))

            # Test gene expression matrix creation
            gene_matrix = pipeline.create_gene_expression_matrix(ident_data)
            assert gene_matrix.shape[0] > 0  # Should have cells
            assert gene_matrix.shape[1] > 0  # Should have genes

            # Test AnnData construction
            with patch("scanpy.pp.calculate_qc_metrics"):
                adata = pipeline.construct_anndata(gene_matrix, centroids)
                assert isinstance(adata, ad.AnnData)
                assert "spatial" in adata.obsm
                assert "raw" in adata.layers

            # Test quality control
            with patch("scanpy.pp.filter_cells"), patch("scanpy.pp.filter_genes"):
                adata_filtered = pipeline.apply_quality_control(adata)
                assert isinstance(adata_filtered, ad.AnnData)

    def test_cli_integration_dry_run(self) -> None:
        """Test CLI integration with dry run."""
        config_data = self.create_realistic_config_data()

        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_dir = Path(temp_dir) / "test_experiment"
            deconved_dir = workspace_dir / "analysis" / "deconv"
            workspace_dir.mkdir()
            deconved_dir.mkdir(parents=True)

            config_data["workspace_path"] = str(workspace_dir)

            # Create config file
            config_file = Path(temp_dir) / "test_config.toml"
            with open(config_file, "w") as f:
                toml.dump(config_data, f)

            # Test CLI dry run
            runner = CliRunner()
            result = runner.invoke(concat, ["--config", str(config_file), "--dry-run"])

            assert result.exit_code == 0
            assert "Configuration validation passed" in result.output
            assert "Workspace:" in result.output
            assert "Codebooks: ['mousecommon', 'zachDE']" in result.output

    def test_configuration_validation_comprehensive(self) -> None:
        """Test comprehensive configuration validation."""
        config_data = self.create_realistic_config_data()

        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_dir = Path(temp_dir)
            config_data["workspace_path"] = str(workspace_dir)

            # Test valid configuration
            config = ConcatConfig(**config_data)
            assert config.workspace_path == workspace_dir
            assert len(config.codebooks) == 2
            assert config.seg_codebook == "atp"
            assert config.intensity.channel == "cfse"
            assert not config.intensity.required  # Set to False for testing
            assert config.quality_control.min_counts == 10
            assert config.spatial.max_columns == 2
            assert config.output.h5ad_path == Path("test_output.h5ad")
            assert config.output.baysor_export is True

    def test_error_handling_comprehensive(self) -> None:
        """Test comprehensive error handling scenarios."""
        config_data = self.create_realistic_config_data()

        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_dir = Path(temp_dir)
            config_data["workspace_path"] = str(workspace_dir)

            config = ConcatConfig(**config_data)
            pipeline = ConcatPipeline(config, validate_workspace=False)

            # Test empty ROIs handling
            pipeline.rois = []
            with pytest.raises(ValueError, match="No identification files found"):
                pipeline.load_identification_data()

            # Test empty polygons handling
            empty_polygons = pl.DataFrame()
            with pytest.raises(ValueError, match="No polygon data provided"):
                pipeline.calculate_weighted_centroids(empty_polygons)

            # Test invalid areas handling
            invalid_polygons = pl.DataFrame(
                {
                    "roilabel": ["cell1", "cell2"],
                    "area": [-10.0, 0.0],  # Invalid areas
                    "centroid_x": [10.0, 20.0],
                    "centroid_y": [15.0, 25.0],
                    "z": [0.0, 1.0],
                    "mean_intensity": [50.0, 75.0],
                    "max_intensity": [80.0, 120.0],
                    "min_intensity": [20.0, 30.0],
                    "roi": ["roi1", "roi1"],
                }
            )

            with pytest.raises(ValueError, match="No polygons with valid areas found"):
                pipeline.calculate_weighted_centroids(invalid_polygons)

    def test_memory_efficiency_simulation(self) -> None:
        """Test memory efficiency with simulated large datasets."""
        config_data = self.create_realistic_config_data()

        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_dir = Path(temp_dir)
            config_data["workspace_path"] = str(workspace_dir)

            config = ConcatConfig(**config_data)
            pipeline = ConcatPipeline(config, validate_workspace=False)

            # Create large mock polygon data
            n_cells = 5000
            large_polygons = pl.DataFrame(
                {
                    "roilabel": [f"roi1cell{i}" for i in range(n_cells)],
                    "area": np.random.uniform(200, 800, n_cells),
                    "centroid_x": np.random.uniform(0, 2000, n_cells),
                    "centroid_y": np.random.uniform(0, 2000, n_cells),
                    "z": np.random.uniform(0, 10, n_cells),
                    "mean_intensity": np.random.uniform(50, 150, n_cells),
                    "max_intensity": np.random.uniform(100, 200, n_cells),
                    "min_intensity": np.random.uniform(10, 50, n_cells),
                    "roi": ["roi1"] * n_cells,
                }
            )

            # Test that pipeline can handle large datasets efficiently
            centroids = pipeline.calculate_weighted_centroids(large_polygons)

            # Verify results
            assert len(centroids) == n_cells
            assert all(np.isfinite(centroids["x"]))
            assert all(np.isfinite(centroids["y"]))
            assert all(centroids["area"] > 0)

            # Test ROI arrangement with large dataset
            arranged_polygons, roi_offsets = pipeline.arrange_rois(
                large_polygons, max_columns=2, padding=100.0
            )

            assert len(arranged_polygons) == n_cells
            assert len(roi_offsets) == 1  # Only one ROI

    def test_numerical_stability_edge_cases(self) -> None:
        """Test numerical stability with edge case data."""
        config_data = self.create_realistic_config_data()

        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_dir = Path(temp_dir)
            config_data["workspace_path"] = str(workspace_dir)

            config = ConcatConfig(**config_data)
            pipeline = ConcatPipeline(config, validate_workspace=False)

            # Test with extreme coordinate values
            extreme_polygons = pl.DataFrame(
                {
                    "roilabel": ["cell1", "cell2", "cell3"],
                    "area": [1e-6, 1e6, 1.0],  # Extreme range of areas
                    "centroid_x": [1e-10, 1e10, 100.0],  # Extreme coordinates
                    "centroid_y": [1e-10, 1e10, 200.0],
                    "z": [0.0, 1.0, 0.5],
                    "mean_intensity": [1e-3, 1e3, 50.0],
                    "max_intensity": [1e-2, 1e4, 80.0],
                    "min_intensity": [1e-4, 1e2, 20.0],
                    "roi": ["roi1", "roi1", "roi1"],
                }
            )

            centroids = pipeline.calculate_weighted_centroids(extreme_polygons)

            # Check that results are finite and reasonable
            assert len(centroids) == 3
            assert all(np.isfinite(centroids["x"]))
            assert all(np.isfinite(centroids["y"]))
            assert all(np.isfinite(centroids["z"]))
            assert all(centroids["area"] > 0)
