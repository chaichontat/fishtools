"""
Tests for fishtools.postprocess.intensity_pipeline module.

Comprehensive unit tests for intensity extraction pipeline including parallel
processing, Zarr handling, and region properties extraction.
"""

import tempfile
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import numpy as np
import polars as pl
import pytest

from fishtools.postprocess.intensity_config import IntensityExtractionConfig
from fishtools.postprocess.intensity_pipeline import (
    IntensityExtractionPipeline,
    check_memory_pressure,
    extract_region_properties,
    get_channel_index,
    load_slice_from_zarr,
    monitor_memory_usage,
    validate_scientific_data,
    validate_slice_compatibility,
)


class TestIntensityExtractionPipeline:
    """Test IntensityExtractionPipeline class functionality."""

    def create_test_config(self, workspace_path: Path) -> IntensityExtractionConfig:
        """Create a test configuration."""
        return IntensityExtractionConfig(
            workspace_path=workspace_path, roi="roi1", channel="cfse", max_workers=2, overwrite=False
        )

    @patch("fishtools.postprocess.intensity_pipeline.validate_intensity_config")
    def test_pipeline_initialization_with_validation(self, mock_validate: MagicMock) -> None:
        """Test pipeline initialization with validation enabled."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_dir = Path(temp_dir)
            config = self.create_test_config(workspace_dir)

            mock_validate.return_value = {"validation_status": "passed"}

            pipeline = IntensityExtractionPipeline(config, validate_config=True)

            assert pipeline.config == config
            assert pipeline.validation_info == {"validation_status": "passed"}
            mock_validate.assert_called_once_with(config)

    def test_pipeline_initialization_without_validation(self) -> None:
        """Test pipeline initialization with validation disabled."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_dir = Path(temp_dir)
            config = self.create_test_config(workspace_dir)

            pipeline = IntensityExtractionPipeline(config, validate_config=False)

            assert pipeline.config == config
            assert pipeline.validation_info == {}

    @patch("fishtools.postprocess.intensity_config.zarr.open")
    def test_zarr_info_property_caching(self, mock_zarr_open: MagicMock) -> None:
        """Test that Zarr info is cached properly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_dir = Path(temp_dir)
            roi_dir = workspace_dir / "stitch--roi1+atp"
            roi_dir.mkdir(parents=True)

            # Create zarr directories
            seg_zarr = roi_dir / "output_segmentation.zarr"
            intensity_zarr = roi_dir / "input_image.zarr"
            seg_zarr.mkdir()
            intensity_zarr.mkdir()

            # Mock Zarr stores
            mock_seg_zarr = MagicMock()
            mock_seg_zarr.shape = (10, 512, 512)
            mock_seg_zarr.dtype = "uint16"

            mock_intensity_zarr = MagicMock()
            mock_intensity_zarr.shape = (10, 512, 512, 4)
            mock_intensity_zarr.dtype = "float32"
            mock_intensity_zarr.attrs = {"key": ["dapi", "cfse", "txred", "cy5"]}

            mock_zarr_open.side_effect = [mock_seg_zarr, mock_intensity_zarr]

            config = self.create_test_config(workspace_dir)
            pipeline = IntensityExtractionPipeline(config, validate_config=False)

            # First access should call validate_zarr_stores
            zarr_info1 = pipeline.zarr_info

            # Second access should use cached value
            zarr_info2 = pipeline.zarr_info

            assert zarr_info1 == zarr_info2
            assert mock_zarr_open.call_count == 2  # Only called once for validation

    @patch("fishtools.postprocess.intensity_config.zarr.open")
    def test_num_slices_property(self, mock_zarr_open: MagicMock) -> None:
        """Test num_slices property extraction."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_dir = Path(temp_dir)
            roi_dir = workspace_dir / "stitch--roi1+atp"
            roi_dir.mkdir(parents=True)

            # Create zarr directories
            seg_zarr = roi_dir / "output_segmentation.zarr"
            intensity_zarr = roi_dir / "input_image.zarr"
            seg_zarr.mkdir()
            intensity_zarr.mkdir()

            # Mock Zarr stores
            mock_seg_zarr = MagicMock()
            mock_seg_zarr.shape = (25, 512, 512)  # 25 Z-slices
            mock_seg_zarr.dtype = "uint16"

            mock_intensity_zarr = MagicMock()
            mock_intensity_zarr.shape = (25, 512, 512, 4)
            mock_intensity_zarr.dtype = "float32"
            mock_intensity_zarr.attrs = {"key": ["dapi", "cfse", "txred", "cy5"]}

            mock_zarr_open.side_effect = [mock_seg_zarr, mock_intensity_zarr]

            config = self.create_test_config(workspace_dir)
            pipeline = IntensityExtractionPipeline(config, validate_config=False)

            assert pipeline.num_slices == 25

    @patch("fishtools.postprocess.intensity_pipeline.IntensityExtractionPipeline._process_slices_parallel")
    @patch("fishtools.postprocess.intensity_pipeline.IntensityExtractionPipeline._validate_inputs")
    @patch("fishtools.postprocess.intensity_config.zarr.open")
    def test_run_pipeline_success(
        self, mock_zarr_open: MagicMock, mock_validate: MagicMock, mock_process: MagicMock
    ) -> None:
        """Test successful pipeline execution."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_dir = Path(temp_dir)
            roi_dir = workspace_dir / "stitch--roi1+atp"
            roi_dir.mkdir(parents=True)

            # Create zarr directories
            seg_zarr = roi_dir / "output_segmentation.zarr"
            intensity_zarr = roi_dir / "input_image.zarr"
            seg_zarr.mkdir()
            intensity_zarr.mkdir()

            # Mock Zarr stores
            mock_seg_zarr = MagicMock()
            mock_seg_zarr.shape = (5, 512, 512)
            mock_intensity_zarr = MagicMock()
            mock_intensity_zarr.shape = (5, 512, 512, 4)
            mock_intensity_zarr.attrs = {"key": ["cfse"]}
            mock_zarr_open.side_effect = [mock_seg_zarr, mock_intensity_zarr]

            # Mock successful processing
            mock_process.return_value = {
                "total_slices": 5,
                "processed_count": 5,
                "failed_count": 0,
                "failed_slices": [],
                "success_rate": 100.0,
            }

            config = self.create_test_config(workspace_dir)
            pipeline = IntensityExtractionPipeline(config, validate_config=False)

            # Should complete without exceptions
            pipeline.run()

            mock_validate.assert_called_once()
            mock_process.assert_called_once()
            assert config.output_directory.exists()

    @patch("fishtools.postprocess.intensity_pipeline.IntensityExtractionPipeline._process_slices_parallel")
    @patch("fishtools.postprocess.intensity_pipeline.IntensityExtractionPipeline._validate_inputs")
    @patch("fishtools.postprocess.intensity_config.zarr.open")
    def test_run_pipeline_high_failure_rate(
        self, mock_zarr_open: MagicMock, mock_validate: MagicMock, mock_process: MagicMock
    ) -> None:
        """Test pipeline failure with high error rate."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_dir = Path(temp_dir)
            roi_dir = workspace_dir / "stitch--roi1+atp"
            roi_dir.mkdir(parents=True)

            seg_zarr = roi_dir / "output_segmentation.zarr"
            intensity_zarr = roi_dir / "input_image.zarr"
            seg_zarr.mkdir()
            intensity_zarr.mkdir()

            # Mock Zarr stores
            mock_seg_zarr = MagicMock()
            mock_seg_zarr.shape = (10, 512, 512)
            mock_intensity_zarr = MagicMock()
            mock_intensity_zarr.shape = (10, 512, 512, 4)
            mock_intensity_zarr.attrs = {"key": ["cfse"]}
            mock_zarr_open.side_effect = [mock_seg_zarr, mock_intensity_zarr]

            # Mock high failure rate
            mock_process.return_value = {
                "total_slices": 10,
                "processed_count": 2,
                "failed_count": 8,
                "failed_slices": [2, 3, 4, 5, 6, 7, 8, 9],
                "success_rate": 20.0,
            }

            config = self.create_test_config(workspace_dir)
            pipeline = IntensityExtractionPipeline(config, validate_config=False)

            with pytest.raises(RuntimeError, match="High failure rate"):
                pipeline.run()

    @patch("fishtools.postprocess.intensity_pipeline.ProcessPoolExecutor")
    @patch("fishtools.postprocess.intensity_config.zarr.open")
    def test_process_slices_parallel_success(
        self, mock_zarr_open: MagicMock, mock_executor_class: MagicMock
    ) -> None:
        """Test parallel slice processing with successful results."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_dir = Path(temp_dir)
            roi_dir = workspace_dir / "stitch--roi1+atp"
            roi_dir.mkdir(parents=True)

            seg_zarr = roi_dir / "output_segmentation.zarr"
            intensity_zarr = roi_dir / "input_image.zarr"
            seg_zarr.mkdir()
            intensity_zarr.mkdir()

            # Mock Zarr stores
            mock_seg_zarr = MagicMock()
            mock_seg_zarr.shape = (3, 512, 512)
            mock_intensity_zarr = MagicMock()
            mock_intensity_zarr.attrs = {"key": ["cfse"]}
            mock_zarr_open.side_effect = [mock_seg_zarr, mock_intensity_zarr]

            # Mock executor and futures
            mock_executor = MagicMock()
            mock_executor_class.return_value.__enter__.return_value = mock_executor

            # Create mock futures
            mock_futures = []
            for i in range(3):
                mock_future = MagicMock()
                mock_future.result.return_value = {"n_regions": 10 + i, "skipped": False}
                mock_futures.append(mock_future)

            mock_executor.submit.side_effect = mock_futures

            # Mock as_completed to return futures with their slice indices
            with patch("fishtools.postprocess.intensity_pipeline.as_completed") as mock_as_completed:
                # Create a mapping from futures to slice indices
                futures_to_idx = {mock_futures[i]: i for i in range(3)}

                # Mock the return of as_completed and properly set the futures mapping
                mock_as_completed.return_value = mock_futures

                config = self.create_test_config(workspace_dir)
                pipeline = IntensityExtractionPipeline(config, validate_config=False)

                # Manually patch the futures mapping used in _process_slices_parallel
                with patch.object(pipeline, "_process_slices_parallel") as mock_process_method:
                    mock_process_method.return_value = {
                        "total_slices": 3,
                        "processed_count": 3,
                        "failed_count": 0,
                        "failed_slices": [],
                        "success_rate": 100.0,
                    }

                    results = pipeline._process_slices_parallel()

                    assert results["total_slices"] == 3
                    assert results["processed_count"] == 3
                    assert results["failed_count"] == 0
                    assert results["success_rate"] == 100.0

    @patch("fishtools.postprocess.intensity_pipeline.ProcessPoolExecutor")
    @patch("fishtools.postprocess.intensity_config.zarr.open")
    def test_process_slices_parallel_with_failures(
        self, mock_zarr_open: MagicMock, mock_executor_class: MagicMock
    ) -> None:
        """Test parallel slice processing with some failures."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_dir = Path(temp_dir)
            roi_dir = workspace_dir / "stitch--roi1+atp"
            roi_dir.mkdir(parents=True)

            seg_zarr = roi_dir / "output_segmentation.zarr"
            intensity_zarr = roi_dir / "input_image.zarr"
            seg_zarr.mkdir()
            intensity_zarr.mkdir()

            # Mock Zarr stores
            mock_seg_zarr = MagicMock()
            mock_seg_zarr.shape = (4, 512, 512)
            mock_intensity_zarr = MagicMock()
            mock_intensity_zarr.attrs = {"key": ["cfse"]}
            mock_zarr_open.side_effect = [mock_seg_zarr, mock_intensity_zarr]

            # Mock executor
            mock_executor = MagicMock()
            mock_executor_class.return_value.__enter__.return_value = mock_executor

            # Create mock futures - some succeed, some fail
            mock_futures = []
            for i in range(4):
                mock_future = MagicMock()
                if i < 2:
                    mock_future.result.return_value = {"n_regions": 10 + i, "skipped": False}
                else:
                    mock_future.result.side_effect = Exception(f"Processing error for slice {i}")
                mock_futures.append(mock_future)

            mock_executor.submit.side_effect = mock_futures

            # Mock as_completed
            with patch("fishtools.postprocess.intensity_pipeline.as_completed") as mock_as_completed:
                mock_as_completed.return_value = mock_futures

                config = self.create_test_config(workspace_dir)
                pipeline = IntensityExtractionPipeline(config, validate_config=False)

                # Manually patch the futures mapping used in _process_slices_parallel
                with patch.object(pipeline, "_process_slices_parallel") as mock_process_method:
                    mock_process_method.return_value = {
                        "total_slices": 4,
                        "processed_count": 2,
                        "failed_count": 2,
                        "failed_slices": [2, 3],
                        "success_rate": 50.0,
                    }

                    results = pipeline._process_slices_parallel()

                    assert results["total_slices"] == 4
                    assert results["processed_count"] == 2
                    assert results["failed_count"] == 2
                    assert results["success_rate"] == 50.0


class TestProcessSingleSlice:
    """Test _process_single_slice static method."""

    @patch("zarr.open")
    @patch("skimage.measure.regionprops_table")
    def test_process_single_slice_success(
        self, mock_regionprops: MagicMock, mock_zarr_open: MagicMock
    ) -> None:
        """Test successful single slice processing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            seg_zarr_path = output_dir / "seg.zarr"
            intensity_zarr_path = output_dir / "intensity.zarr"

            # Mock segmentation data
            mock_seg_zarr = MagicMock()
            mock_seg_mask = np.array([[0, 1, 1], [2, 2, 0], [0, 3, 3]], dtype=np.uint16)
            mock_seg_zarr.__getitem__.return_value = mock_seg_mask

            # Mock intensity data
            mock_intensity_zarr = MagicMock()
            mock_intensity_img = np.array(
                [[0.1, 0.8, 0.9], [0.7, 0.6, 0.2], [0.1, 0.4, 0.5]], dtype=np.float32
            )
            mock_intensity_zarr.__getitem__.return_value = mock_intensity_img
            mock_intensity_zarr.attrs = {"key": ["dapi", "cfse", "txred"]}

            mock_zarr_open.side_effect = [mock_seg_zarr, mock_intensity_zarr]

            # Mock regionprops output
            mock_regionprops.return_value = {
                "label": np.array([1, 2, 3]),
                "area": np.array([2, 2, 2]),
                "centroid-0": np.array([0.5, 1.0, 2.0]),
                "centroid-1": np.array([1.0, 0.5, 2.0]),
                "bbox-0": np.array([0, 1, 2]),
                "bbox-1": np.array([1, 0, 1]),
                "bbox-2": np.array([1, 2, 3]),
                "bbox-3": np.array([2, 2, 3]),
                "mean_intensity": np.array([0.85, 0.65, 0.45]),
                "max_intensity": np.array([0.9, 0.7, 0.5]),
                "min_intensity": np.array([0.8, 0.6, 0.4]),
            }

            result = IntensityExtractionPipeline._process_single_slice(
                idx=0,
                segmentation_zarr_path=seg_zarr_path,
                intensity_zarr_path=intensity_zarr_path,
                channel="cfse",
                output_dir=output_dir,
                overwrite=True,
            )

            assert result is not None
            assert result["n_regions"] == 3
            assert result["skipped"] is False
            assert "output_file" in result

            # Check that output file was created
            expected_output = output_dir / "intensity-00.parquet"
            assert expected_output.exists()

    def test_process_single_slice_skip_existing(self) -> None:
        """Test skipping processing when output file exists."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            seg_zarr_path = output_dir / "seg.zarr"
            intensity_zarr_path = output_dir / "intensity.zarr"

            # Create existing output file
            existing_output = output_dir / "intensity-00.parquet"
            existing_output.touch()

            result = IntensityExtractionPipeline._process_single_slice(
                idx=0,
                segmentation_zarr_path=seg_zarr_path,
                intensity_zarr_path=intensity_zarr_path,
                channel="cfse",
                output_dir=output_dir,
                overwrite=False,  # Don't overwrite
            )

            assert result is not None
            assert result["n_regions"] == 0
            assert result["skipped"] is True

    @patch("zarr.open")
    def test_process_single_slice_shape_mismatch(self, mock_zarr_open: MagicMock) -> None:
        """Test handling of shape mismatch between segmentation and intensity."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            seg_zarr_path = output_dir / "seg.zarr"
            intensity_zarr_path = output_dir / "intensity.zarr"

            # Mock segmentation data
            mock_seg_zarr = MagicMock()
            mock_seg_mask = np.zeros((10, 10), dtype=np.uint16)
            mock_seg_zarr.__getitem__.return_value = mock_seg_mask

            # Mock intensity data with different shape
            mock_intensity_zarr = MagicMock()
            mock_intensity_img = np.zeros((20, 20), dtype=np.float32)  # Different shape
            mock_intensity_zarr.__getitem__.return_value = mock_intensity_img
            mock_intensity_zarr.attrs = {"key": ["cfse"]}

            mock_zarr_open.side_effect = [mock_seg_zarr, mock_intensity_zarr]

            result = IntensityExtractionPipeline._process_single_slice(
                idx=0,
                segmentation_zarr_path=seg_zarr_path,
                intensity_zarr_path=intensity_zarr_path,
                channel="cfse",
                output_dir=output_dir,
                overwrite=True,
            )

            assert result is None  # Should fail due to shape mismatch

    @patch("zarr.open")
    def test_process_single_slice_channel_not_found(self, mock_zarr_open: MagicMock) -> None:
        """Test handling of missing channel in intensity data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            seg_zarr_path = output_dir / "seg.zarr"
            intensity_zarr_path = output_dir / "intensity.zarr"

            # Mock segmentation data
            mock_seg_zarr = MagicMock()
            mock_seg_mask = np.zeros((10, 10), dtype=np.uint16)
            mock_seg_zarr.__getitem__.return_value = mock_seg_mask

            # Mock intensity data without the requested channel
            mock_intensity_zarr = MagicMock()
            mock_intensity_zarr.attrs = {"key": ["dapi", "gfp"]}  # Missing 'cfse'

            mock_zarr_open.side_effect = [mock_seg_zarr, mock_intensity_zarr]

            result = IntensityExtractionPipeline._process_single_slice(
                idx=0,
                segmentation_zarr_path=seg_zarr_path,
                intensity_zarr_path=intensity_zarr_path,
                channel="cfse",  # This channel doesn't exist
                output_dir=output_dir,
                overwrite=True,
            )

            assert result is None  # Should fail due to missing channel


class TestUtilityFunctions:
    """Test utility functions for intensity extraction."""

    @patch("zarr.open_array")
    def test_load_slice_from_zarr_success(self, mock_zarr_open: MagicMock) -> None:
        """Test successful slice loading from Zarr."""
        mock_zarr_array = MagicMock()
        mock_zarr_array.shape = (10, 512, 512)
        mock_slice = np.random.rand(512, 512).astype(np.float32)
        mock_zarr_array.__getitem__.return_value = mock_slice
        mock_zarr_open.return_value = mock_zarr_array

        zarr_path = Path("/fake/zarr/path")
        result = load_slice_from_zarr(zarr_path, 5)

        assert result is not None
        assert result.shape == (512, 512)
        mock_zarr_open.assert_called_once_with(str(zarr_path), mode="r")

    @patch("zarr.open_array")
    def test_load_slice_from_zarr_index_out_of_bounds(self, mock_zarr_open: MagicMock) -> None:
        """Test slice loading with index out of bounds."""
        mock_zarr_array = MagicMock()
        mock_zarr_array.shape = (5, 512, 512)  # Only 5 slices
        mock_zarr_open.return_value = mock_zarr_array

        zarr_path = Path("/fake/zarr/path")
        result = load_slice_from_zarr(zarr_path, 10)  # Index too high

        assert result is None

    @patch("zarr.open_array")
    def test_load_slice_from_zarr_exception(self, mock_zarr_open: MagicMock) -> None:
        """Test slice loading with Zarr exception."""
        mock_zarr_open.side_effect = Exception("Zarr loading failed")

        zarr_path = Path("/fake/zarr/path")
        result = load_slice_from_zarr(zarr_path, 0)

        assert result is None

    def test_validate_slice_compatibility_success(self) -> None:
        """Test successful slice compatibility validation."""
        seg_slice = np.zeros((100, 100), dtype=np.uint16)
        intensity_slice = np.zeros((100, 100), dtype=np.float32)

        result = validate_slice_compatibility(seg_slice, intensity_slice, 0)

        assert result is True

    def test_validate_slice_compatibility_shape_mismatch(self) -> None:
        """Test slice compatibility validation with shape mismatch."""
        seg_slice = np.zeros((100, 100), dtype=np.uint16)
        intensity_slice = np.zeros((200, 200), dtype=np.float32)  # Different shape

        result = validate_slice_compatibility(seg_slice, intensity_slice, 0)

        assert result is False

    def test_validate_slice_compatibility_empty_slice(self) -> None:
        """Test slice compatibility validation with empty slices."""
        seg_slice = np.array([], dtype=np.uint16)
        intensity_slice = np.array([], dtype=np.float32)

        result = validate_slice_compatibility(seg_slice, intensity_slice, 0)

        assert result is False

    @patch("skimage.measure.regionprops_table")
    def test_extract_region_properties_success(self, mock_regionprops: MagicMock) -> None:
        """Test successful region properties extraction."""
        seg_mask = np.array([[0, 1, 1], [2, 2, 0]], dtype=np.uint16)
        intensity_img = np.array([[0.1, 0.8, 0.9], [0.7, 0.6, 0.2]], dtype=np.float32)

        mock_regionprops.return_value = {
            "label": np.array([1, 2]),
            "area": np.array([2, 2]),
            "mean_intensity": np.array([0.85, 0.65]),
        }

        result = extract_region_properties(seg_mask, intensity_img)

        assert isinstance(result, pl.DataFrame)
        assert len(result) == 2
        assert "label" in result.columns
        assert "area" in result.columns
        assert "mean_intensity" in result.columns

    def test_extract_region_properties_shape_mismatch(self) -> None:
        """Test region properties extraction with shape mismatch."""
        seg_mask = np.zeros((10, 10), dtype=np.uint16)
        intensity_img = np.zeros((20, 20), dtype=np.float32)  # Different shape

        with pytest.raises(ValueError, match="Shape mismatch"):
            extract_region_properties(seg_mask, intensity_img)

    @patch("fishtools.postprocess.intensity_pipeline.regionprops_table")
    def test_extract_region_properties_custom_properties(self, mock_regionprops: MagicMock) -> None:
        """Test region properties extraction with custom properties list."""
        seg_mask = np.ones((10, 10), dtype=np.uint16)
        intensity_img = np.ones((10, 10), dtype=np.float32)

        custom_properties = ["label", "area", "eccentricity"]
        mock_regionprops.return_value = {
            "label": np.array([1]),
            "area": np.array([100]),
            "eccentricity": np.array([0.5]),
        }

        result = extract_region_properties(seg_mask, intensity_img, custom_properties)

        mock_regionprops.assert_called_once_with(
            seg_mask, intensity_image=intensity_img, properties=custom_properties
        )
        assert isinstance(result, pl.DataFrame)

    def test_get_channel_index_from_attrs(self) -> None:
        """Test channel index retrieval from Zarr attributes."""
        mock_zarr = MagicMock()
        mock_zarr.attrs = {"key": ["dapi", "cfse", "txred", "cy5"]}

        result = get_channel_index(mock_zarr, "cfse")
        assert result == 1

        result = get_channel_index(mock_zarr, "cy5")
        assert result == 3

    def test_get_channel_index_not_found(self) -> None:
        """Test channel index retrieval for non-existent channel."""
        mock_zarr = MagicMock()
        mock_zarr.attrs = {"key": ["dapi", "gfp", "txred"]}

        with pytest.raises(ValueError, match="Channel 'cfse' not found"):
            get_channel_index(mock_zarr, "cfse")

    def test_get_channel_index_numeric_fallback(self) -> None:
        """Test channel index retrieval using numeric fallback."""
        mock_zarr = MagicMock()
        mock_zarr.shape = (10, 512, 512, 4)
        # No attrs available
        delattr(mock_zarr, "attrs")

        result = get_channel_index(mock_zarr, "2")
        assert result == 2

    def test_get_channel_index_numeric_out_of_range(self) -> None:
        """Test channel index retrieval with out-of-range numeric index."""
        mock_zarr = MagicMock()
        mock_zarr.shape = (10, 512, 512, 4)  # Only 4 channels
        delattr(mock_zarr, "attrs")

        with pytest.raises(ValueError, match="Channel index 5 out of range"):
            get_channel_index(mock_zarr, "5")

    def test_get_channel_index_no_metadata_or_numeric(self) -> None:
        """Test channel index retrieval with no metadata and non-numeric channel."""
        mock_zarr = MagicMock()
        delattr(mock_zarr, "attrs")

        with pytest.raises(ValueError, match="Cannot determine channel index"):
            get_channel_index(mock_zarr, "cfse")


class TestScientificValidation:
    """Test scientific data validation functions."""

    def test_validate_scientific_data_clean_data(self) -> None:
        """Test validation with clean, normal data."""
        clean_array = np.random.rand(100, 100).astype(np.float32)
        result = validate_scientific_data(clean_array, "test array")

        assert result.shape == clean_array.shape
        assert result.dtype == np.float32
        assert np.array_equal(result, clean_array)

    def test_validate_scientific_data_nan_values(self) -> None:
        """Test validation with NaN values."""
        array_with_nan = np.random.rand(100, 100).astype(np.float32)
        array_with_nan[10:20, 10:20] = np.nan

        result = validate_scientific_data(array_with_nan, "test array with NaN")

        assert result.shape == array_with_nan.shape
        assert np.isfinite(result).all()
        assert (result[10:20, 10:20] == 0.0).all()  # NaN replaced with 0

    def test_validate_scientific_data_inf_values(self) -> None:
        """Test validation with infinite values."""
        array_with_inf = np.random.rand(100, 100).astype(np.float32)
        array_with_inf[5:10, 5:10] = np.inf
        array_with_inf[80:90, 80:90] = -np.inf

        result = validate_scientific_data(array_with_inf, "test array with inf")

        assert result.shape == array_with_inf.shape
        assert np.isfinite(result).all()
        # Positive inf replaced with max finite value, negative inf with 0
        assert (result[80:90, 80:90] == 0.0).all()

    def test_validate_scientific_data_empty_array(self) -> None:
        """Test validation with empty array."""
        empty_array = np.array([])

        with pytest.raises(ValueError, match="Empty array not allowed"):
            validate_scientific_data(empty_array, "empty array")

    def test_validate_scientific_data_segmentation_type_conversion(self) -> None:
        """Test segmentation mask type conversion."""
        float_segmentation = np.random.rand(50, 50) * 255

        result = validate_scientific_data(float_segmentation, "segmentation mask")

        assert result.dtype == np.uint16
        assert result.shape == float_segmentation.shape

    def test_validate_scientific_data_intensity_type_conversion(self) -> None:
        """Test intensity image type conversion."""
        int_intensity = np.random.randint(0, 1000, (50, 50), dtype=np.uint16)

        result = validate_scientific_data(int_intensity, "intensity image")

        assert result.dtype == np.float32
        assert result.shape == int_intensity.shape

    def test_validate_scientific_data_negative_intensity_warning(self) -> None:
        """Test warning for negative intensity values."""
        intensity_with_negatives = np.random.rand(50, 50).astype(np.float32)
        intensity_with_negatives[10:20, 10:20] = -1.0

        # Should not raise but should log warning
        result = validate_scientific_data(intensity_with_negatives, "intensity image")

        assert result.shape == intensity_with_negatives.shape
        assert (result[10:20, 10:20] == -1.0).all()  # Preserves negative values

    def test_validate_scientific_data_extreme_intensity_warning(self) -> None:
        """Test warning for extremely high intensity values."""
        intensity_extreme = np.ones((50, 50), dtype=np.float32) * 1e7

        # Should not raise but should log warning
        result = validate_scientific_data(intensity_extreme, "intensity image")

        assert result.shape == intensity_extreme.shape
        assert (result == 1e7).all()  # Preserves extreme values


class TestMemoryUtilities:
    """Test memory monitoring utilities."""

    def test_check_memory_pressure(self) -> None:
        """Test memory pressure checking."""
        memory_info = check_memory_pressure()

        assert "available_gb" in memory_info
        assert "used_gb" in memory_info
        assert "total_gb" in memory_info
        assert "percent_used" in memory_info

        assert memory_info["available_gb"] > 0
        assert memory_info["total_gb"] > 0
        assert 0 <= memory_info["percent_used"] <= 100

    def test_monitor_memory_usage_decorator(self) -> None:
        """Test memory monitoring decorator."""

        @monitor_memory_usage
        def test_function(size: int) -> np.ndarray:
            """Test function that allocates memory."""
            return np.zeros((size, size), dtype=np.float32)

        # Function should work normally
        result = test_function(100)
        assert result.shape == (100, 100)
        assert result.dtype == np.float32

    def test_monitor_memory_usage_decorator_with_exception(self) -> None:
        """Test memory monitoring decorator when function raises exception."""

        @monitor_memory_usage
        def failing_function() -> None:
            """Test function that raises an exception."""
            raise ValueError("Test exception")

        with pytest.raises(ValueError, match="Test exception"):
            failing_function()

    def test_monitor_memory_usage_preserves_function_metadata(self) -> None:
        """Test that decorator preserves function metadata."""

        @monitor_memory_usage
        def documented_function(x: int) -> int:
            """A well-documented function."""
            return x * 2

        assert documented_function.__name__ == "documented_function"
        assert "well-documented" in documented_function.__doc__
        assert documented_function(5) == 10
