"""
Intensity extraction pipeline for segmentation mask analysis.

This module implements the core processing pipeline for extracting intensity
measurements from segmentation masks using Zarr-format data stores and
scikit-image regionprops.
"""

import functools
import gc
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import polars as pl
import psutil
import zarr
from loguru import logger
from skimage.measure import regionprops_table

from .intensity_config import IntensityExtractionConfig, validate_intensity_config


def monitor_memory_usage(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Decorator to monitor memory usage during function execution.

    Args:
        func: Function to monitor

    Returns:
        Wrapped function with memory monitoring
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        initial_memory = psutil.virtual_memory().available / (1024**3)  # GB
        try:
            result = func(*args, **kwargs)
            final_memory = psutil.virtual_memory().available / (1024**3)  # GB
            used_memory_gb = initial_memory - final_memory

            if used_memory_gb > 0.5:  # Log significant memory usage (>500MB)
                logger.info(f"Function {func.__name__} used {used_memory_gb:.1f}GB memory")
            elif used_memory_gb < -0.1:  # Memory freed
                logger.debug(f"Function {func.__name__} freed {-used_memory_gb:.1f}GB memory")

            return result
        except Exception as e:
            final_memory = psutil.virtual_memory().available / (1024**3)
            used_memory_gb = initial_memory - final_memory
            if used_memory_gb > 0.1:
                logger.warning(f"Function {func.__name__} failed after using {used_memory_gb:.1f}GB memory")
            raise

    return wrapper


def validate_scientific_data(array: np.ndarray, array_name: str) -> np.ndarray:
    """
    Validate and clean scientific imaging data.

    This function checks for common issues in scientific imaging data including
    non-finite values, empty arrays, and data type compatibility.

    Args:
        array: Input array to validate
        array_name: Descriptive name for logging

    Returns:
        Cleaned and validated array

    Raises:
        ValueError: If array is fundamentally invalid
    """
    if array.size == 0:
        raise ValueError(f"{array_name}: Empty array not allowed for analysis")

    # Check for non-finite values (NaN, inf, -inf)
    non_finite_mask = ~np.isfinite(array)
    non_finite_count = non_finite_mask.sum()

    if non_finite_count > 0:
        non_finite_pct = (non_finite_count / array.size) * 100
        logger.warning(
            f"{array_name}: {non_finite_count} ({non_finite_pct:.1f}%) non-finite values detected and cleaned"
        )

        # Replace non-finite values with appropriate defaults
        array = np.nan_to_num(array, nan=0.0, posinf=array[np.isfinite(array)].max(), neginf=0.0)

    # Check for suspicious data ranges in intensity data
    if "intensity" in array_name.lower():
        if array.min() < 0:
            negative_count = (array < 0).sum()
            logger.warning(f"{array_name}: {negative_count} negative intensity values found (unusual)")

        if array.max() > 1e6:
            logger.warning(f"{array_name}: Very high intensity values detected (max: {array.max():.2e})")

    # Ensure appropriate data type for downstream processing
    if "segmentation" in array_name.lower() or "mask" in array_name.lower():
        if not np.issubdtype(array.dtype, np.integer):
            logger.warning(f"{array_name}: Converting from {array.dtype} to uint16 for segmentation")
            array = array.astype(np.uint16)
    elif "intensity" in array_name.lower():
        if array.dtype != np.float32:
            logger.debug(f"{array_name}: Converting from {array.dtype} to float32 for processing")
            array = array.astype(np.float32)

    return array


def check_memory_pressure() -> Dict[str, float]:
    """
    Check current system memory pressure.

    Returns:
        Dictionary with memory information in GB
    """
    memory = psutil.virtual_memory()
    return {
        "available_gb": memory.available / (1024**3),
        "used_gb": memory.used / (1024**3),
        "total_gb": memory.total / (1024**3),
        "percent_used": memory.percent,
    }


class IntensityExtractionPipeline:
    """
    Pipeline for extracting intensity measurements from segmentation masks.

    This class orchestrates the complete intensity extraction workflow, from
    Zarr store validation to parallel processing of Z-slices and parquet output.
    """

    def __init__(self, config: IntensityExtractionConfig, validate_config: bool = True) -> None:
        """
        Initialize the intensity extraction pipeline.

        Args:
            config: Configuration object containing all pipeline parameters
            validate_config: Whether to validate configuration and files (default True)
        """
        self.config = config

        if validate_config:
            self.validation_info = validate_intensity_config(config)
            logger.info("Configuration and system validation completed successfully")
        else:
            self.validation_info = {}
            logger.info("Configuration validation skipped")

        # Cache Zarr information for performance
        self._zarr_info: Optional[Dict[str, Any]] = None
        self._num_slices: Optional[int] = None

        logger.info(
            f"Initialized intensity extraction pipeline for ROI '{config.roi}', channel '{config.channel}'"
        )

    @property
    def zarr_info(self) -> Dict[str, Any]:
        """Get cached Zarr store information."""
        if self._zarr_info is None:
            self._zarr_info = self.config.validate_zarr_stores()
        return self._zarr_info

    @property
    def num_slices(self) -> int:
        """Get the number of Z-slices in the segmentation stack."""
        if self._num_slices is None:
            self._num_slices = self.zarr_info["segmentation_shape"][0]
        return self._num_slices

    @monitor_memory_usage
    def run(self) -> None:
        """
        Execute the complete intensity extraction pipeline.

        This method coordinates the entire workflow from validation to completion.
        """
        logger.info("Starting intensity extraction pipeline")

        # Check initial memory state
        initial_memory = check_memory_pressure()
        logger.info(
            f"Initial memory state: {initial_memory['available_gb']:.1f}GB available ({initial_memory['percent_used']:.1f}% used)"
        )

        # Validate inputs
        logger.info("Validating input files and system requirements")
        self._validate_inputs()

        # Create output directory
        self.config.output_directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created output directory: {self.config.output_directory}")

        # Process slices in parallel
        logger.info(f"Processing {self.num_slices} slices with {self.config.max_workers} workers")
        results = self._process_slices_parallel()

        # Report results
        self._report_results(results)

        # Check final memory state
        final_memory = check_memory_pressure()
        logger.info(
            f"Final memory state: {final_memory['available_gb']:.1f}GB available ({final_memory['percent_used']:.1f}% used)"
        )

        # Force garbage collection for large datasets
        if initial_memory["available_gb"] - final_memory["available_gb"] > 1.0:
            logger.info("Running garbage collection after large dataset processing")
            gc.collect()

        logger.info("Intensity extraction pipeline completed successfully")

    def _validate_inputs(self) -> None:
        """
        Validate inputs and log system information.

        Raises:
            ValueError: If validation fails
        """
        zarr_info = self.zarr_info

        logger.info(f"Segmentation shape: {zarr_info['segmentation_shape']}")
        logger.info(f"Intensity shape: {zarr_info['intensity_shape']}")
        logger.info(f"Available channels: {zarr_info['available_channels']}")
        logger.info(f"Selected channel: {self.config.channel}")

        # Log performance information if available
        if "performance_info" in self.validation_info:
            perf_info = self.validation_info["performance_info"]
            logger.info(f"Estimated peak memory usage: {perf_info['estimated_peak_memory_gb']:.1f} GB")
            logger.info(f"Available memory: {perf_info['available_memory_gb']:.1f} GB")
            logger.info(f"Recommended workers: {perf_info['recommended_workers']}")

    def _process_slices_parallel(self) -> Dict[str, Any]:
        """
        Process all Z-slices in parallel using ProcessPoolExecutor.

        Returns:
            Dictionary containing processing results and statistics
        """
        processed_count = 0
        failed_count = 0
        failed_slices: List[int] = []

        # Use spawn context for process isolation
        with ProcessPoolExecutor(
            max_workers=self.config.max_workers, mp_context=get_context("spawn")
        ) as executor:
            # Submit all slice processing tasks
            futures = {
                executor.submit(
                    self._process_single_slice,
                    idx,
                    self.config.segmentation_zarr_path,
                    self.config.intensity_zarr_path,
                    self.config.channel,
                    self.config.output_directory,
                    self.config.overwrite,
                ): idx
                for idx in range(self.num_slices)
            }

            # Collect results as they complete
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    result = future.result()
                    if result is not None:
                        processed_count += 1
                        logger.info(f"Successfully processed slice {idx}: {result['n_regions']} regions")
                    else:
                        failed_count += 1
                        failed_slices.append(idx)
                        logger.error(f"Slice {idx} processing returned None")
                except Exception as e:
                    failed_count += 1
                    failed_slices.append(idx)
                    logger.error(f"Slice {idx} processing failed: {e}")

        return {
            "total_slices": self.num_slices,
            "processed_count": processed_count,
            "failed_count": failed_count,
            "failed_slices": failed_slices,
            "success_rate": (processed_count / self.num_slices) * 100 if self.num_slices > 0 else 0,
        }

    def _report_results(self, results: Dict[str, Any]) -> None:
        """
        Report processing results and handle errors.

        Args:
            results: Processing results from _process_slices_parallel

        Raises:
            RuntimeError: If processing failure rate is too high
        """
        logger.info("--- Intensity Extraction Summary ---")
        logger.info(f"Total slices: {results['total_slices']}")
        logger.info(f"Successfully processed: {results['processed_count']}")
        logger.info(f"Failed: {results['failed_count']}")
        logger.info(f"Success rate: {results['success_rate']:.1f}%")

        if results["failed_slices"]:
            logger.warning(f"Failed slices: {results['failed_slices']}")

        # Check for unacceptable failure rate
        if results["success_rate"] < 80.0:
            raise RuntimeError(
                f"High failure rate: {results['failed_count']}/{results['total_slices']} slices failed. "
                "Check logs for details."
            )

        if results["failed_count"] > 0:
            logger.warning(f"Some slices failed processing. Output may be incomplete.")

    @staticmethod
    def _process_single_slice(
        idx: int,
        segmentation_zarr_path: Path,
        intensity_zarr_path: Path,
        channel: str,
        output_dir: Path,
        overwrite: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """
        Process a single Z-slice to extract region properties.

        This static method is designed for use with ProcessPoolExecutor and contains
        all the logic from the original overlay_intensity.py process_slice_regionprops function.

        Args:
            idx: Z-slice index
            segmentation_zarr_path: Path to segmentation Zarr store
            intensity_zarr_path: Path to intensity Zarr store
            channel: Channel name for intensity extraction
            output_dir: Output directory for parquet files
            overwrite: Whether to overwrite existing files

        Returns:
            Dictionary with processing results or None if failed
        """
        # Configure logging for subprocess
        from loguru import logger

        # Determine output file path
        props_path = output_dir / f"intensity-{idx:02d}.parquet"

        if not overwrite and props_path.exists():
            logger.info(f"Slice {idx}: Skipping, output file already exists: {props_path}")
            return {"n_regions": 0, "skipped": True}

        try:
            # Load segmentation mask
            seg_zarr = zarr.open(str(segmentation_zarr_path), mode="r")
            seg_mask = seg_zarr[idx]
            logger.debug(f"Slice {idx}: Loaded segmentation with shape {seg_mask.shape}")

            # Load intensity image
            intensity_zarr = zarr.open(str(intensity_zarr_path), mode="r")

            # Extract channel-specific intensity data
            if hasattr(intensity_zarr, "attrs") and "key" in intensity_zarr.attrs:
                channels = intensity_zarr.attrs["key"]
                try:
                    c_idx = channels.index(channel)
                    intensity_img = intensity_zarr[idx, :, :, c_idx]
                except ValueError:
                    raise ValueError(f"Channel '{channel}' not found in {channels}")
            else:
                # Fallback: assume single channel or use channel as index
                if isinstance(channel, str) and channel.isdigit():
                    c_idx = int(channel)
                    intensity_img = intensity_zarr[idx, :, :, c_idx]
                else:
                    raise ValueError(f"Cannot determine channel index for '{channel}'")

            logger.debug(f"Slice {idx}: Loaded intensity with shape {intensity_img.shape}")

            # Validate shapes match
            if seg_mask.shape != intensity_img.shape:
                raise ValueError(
                    f"Slice {idx}: Shape mismatch between segmentation {seg_mask.shape} "
                    f"and intensity {intensity_img.shape}"
                )

            # Apply scientific data validation
            seg_mask = validate_scientific_data(seg_mask, f"Slice {idx} segmentation mask")
            intensity_img = validate_scientific_data(intensity_img, f"Slice {idx} intensity image")

            # Calculate region properties
            logger.debug(f"Slice {idx}: Calculating region properties...")

            # Define properties to extract (matching original overlay_intensity.py)
            properties = [
                "label",
                "area",
                "centroid",
                "bbox",
                "mean_intensity",
                "max_intensity",
                "min_intensity",
            ]

            # Extract region properties using intensity image
            props_table = regionprops_table(seg_mask, intensity_image=intensity_img, properties=properties)

            # Convert to Polars DataFrame
            props_df = pl.DataFrame(props_table)
            n_regions = len(props_df)

            logger.debug(f"Slice {idx}: Found {n_regions} regions")

            # Save results
            props_df.write_parquet(props_path)
            logger.info(f"Slice {idx}: Saved {n_regions} regions to {props_path}")

            return {"n_regions": n_regions, "output_file": str(props_path), "skipped": False}

        except Exception as e:
            logger.error(f"Slice {idx}: Processing failed: {e}")
            return None


def load_slice_from_zarr(zarr_path: Path, idx: int) -> Optional[np.ndarray]:
    """
    Load a specific Z-slice from a Zarr store.

    This function provides a simple interface for loading individual slices
    and includes error handling for common issues.

    Args:
        zarr_path: Path to the Zarr store
        idx: Z-slice index to load

    Returns:
        NumPy array containing the slice data, or None if loading fails
    """
    try:
        logger.debug(f"Loading slice {idx} from {zarr_path}")
        img_stack = zarr.open_array(str(zarr_path), mode="r")

        if idx >= img_stack.shape[0]:
            logger.error(f"Slice index {idx} exceeds stack size {img_stack.shape[0]}")
            return None

        img_slice = img_stack[idx]
        logger.debug(f"Loaded slice {idx} with shape {img_slice.shape}")
        return img_slice

    except Exception as e:
        logger.error(f"Failed to load slice {idx} from {zarr_path}: {e}")
        return None


def validate_slice_compatibility(seg_slice: np.ndarray, intensity_slice: np.ndarray, slice_idx: int) -> bool:
    """
    Validate that segmentation and intensity slices are compatible.

    Args:
        seg_slice: Segmentation mask slice
        intensity_slice: Intensity image slice
        slice_idx: Slice index for error reporting

    Returns:
        True if slices are compatible, False otherwise
    """
    if seg_slice.shape != intensity_slice.shape:
        logger.error(
            f"Slice {slice_idx}: Shape mismatch - "
            f"segmentation: {seg_slice.shape}, intensity: {intensity_slice.shape}"
        )
        return False

    if seg_slice.size == 0 or intensity_slice.size == 0:
        logger.warning(f"Slice {slice_idx}: Empty slice detected")
        return False

    return True


def extract_region_properties(
    seg_mask: np.ndarray, intensity_img: np.ndarray, properties: Optional[List[str]] = None
) -> pl.DataFrame:
    """
    Extract region properties from segmentation mask and intensity image.

    Args:
        seg_mask: Segmentation mask array
        intensity_img: Intensity image array
        properties: List of properties to extract (uses default if None)

    Returns:
        Polars DataFrame containing region properties

    Raises:
        ValueError: If inputs are invalid or processing fails
    """
    if properties is None:
        properties = [
            "label",
            "area",
            "centroid",
            "bbox",
            "mean_intensity",
            "max_intensity",
            "min_intensity",
        ]

    if seg_mask.shape != intensity_img.shape:
        raise ValueError(f"Shape mismatch: segmentation {seg_mask.shape} vs intensity {intensity_img.shape}")

    # Apply scientific validation
    seg_mask = validate_scientific_data(seg_mask, "segmentation mask")
    intensity_img = validate_scientific_data(intensity_img, "intensity image")

    try:
        props_table = regionprops_table(seg_mask, intensity_image=intensity_img, properties=properties)

        props_df = pl.DataFrame(props_table)

        # Validate results
        if props_df.is_empty():
            logger.warning("No regions found in segmentation mask")
        else:
            logger.debug(f"Extracted properties for {len(props_df)} regions")

            # Additional validation for scientific data quality
            if "mean_intensity" in props_df.columns:
                mean_intensities = props_df["mean_intensity"].to_numpy()
                if not np.isfinite(mean_intensities).all():
                    logger.error("Non-finite mean intensities detected in results")
                    raise ValueError("Region properties calculation produced non-finite values")

        return props_df

    except Exception as e:
        raise ValueError(f"Failed to extract region properties: {e}")


def get_channel_index(intensity_zarr: zarr.Array, channel: str) -> int:
    """
    Get the channel index from intensity Zarr store metadata.

    Args:
        intensity_zarr: Opened Zarr array
        channel: Channel name to find

    Returns:
        Channel index

    Raises:
        ValueError: If channel is not found
    """
    if hasattr(intensity_zarr, "attrs") and "key" in intensity_zarr.attrs:
        channels = intensity_zarr.attrs["key"]
        if channel in channels:
            return channels.index(channel)
        else:
            raise ValueError(f"Channel '{channel}' not found in available channels: {channels}")

    # Fallback: try to parse channel as integer index
    if channel.isdigit():
        c_idx = int(channel)
        if hasattr(intensity_zarr, "shape") and len(intensity_zarr.shape) > 3:
            if c_idx < intensity_zarr.shape[-1]:
                return c_idx
        raise ValueError(f"Channel index {c_idx} out of range")

    raise ValueError(f"Cannot determine channel index for '{channel}' - no metadata available")
