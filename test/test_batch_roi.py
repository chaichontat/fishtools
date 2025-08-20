"""Comprehensive tests for @batch_roi decorator using pytest fixtures.

This test suite covers all aspects of the batch_roi decorator functionality
including batch processing, codebook integration, error handling, and filesystem interactions.
"""

import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Any
from collections.abc import Generator

import pytest
from loguru import logger

from fishtools.utils.utils import batch_roi


@pytest.fixture
def temp_workspace() -> Generator[Path, None, None]:
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def basic_roi_structure(temp_workspace: Path) -> Path:
    """Create basic ROI directory structure for testing."""
    roi_dirs: list[str] = [
        "registered--roi1",
        "registered--roi2",
        "registered--cortex",
    ]
    for roi_dir in roi_dirs:
        (temp_workspace / roi_dir).mkdir(parents=True, exist_ok=True)
    return temp_workspace


@pytest.fixture
def codebook_roi_structure(temp_workspace: Path) -> Path:
    """Create ROI directories with codebook suffixes."""
    codebook_dirs: list[str] = [
        "registered--roi1+book1",
        "registered--roi2+book1",
        "registered--cortex+book2",
    ]
    for roi_dir in codebook_dirs:
        (temp_workspace / roi_dir).mkdir(parents=True, exist_ok=True)
    return temp_workspace


@pytest.fixture
def realistic_workspace(temp_workspace: Path) -> tuple[Path, list[str], list[str]]:
    """Create realistic fishtools workspace structure."""
    rois: list[str] = ["cortex", "hippocampus", "striatum"]
    codebooks: list[str] = ["geneA", "geneB"]

    for roi in rois:
        for codebook in codebooks:
            # Create various processing stage directories
            (temp_workspace / f"registered--{roi}+{codebook}").mkdir(parents=True, exist_ok=True)
            (temp_workspace / f"stitch--{roi}+{codebook}").mkdir(parents=True, exist_ok=True)

            # Add some realistic files
            for i in range(3):
                (temp_workspace / f"registered--{roi}+{codebook}" / f"image-{i:03d}.tif").touch()

    return temp_workspace, rois, codebooks


@pytest.fixture
def log_capture() -> Generator[list[str], None, None]:
    """Capture log messages for testing."""
    log_records: list[str] = []

    def capture_logs(message: str) -> None:
        # Extract just the message part from the formatted log
        # loguru format: "timestamp | level | module | message"
        last_pipe: int = message.rfind(" | ")
        if last_pipe != -1:
            msg: str = message[last_pipe + 3 :].strip()  # +3 to skip " | "
            # Further extract just the message part after " - "
            if " - " in msg:
                msg = msg.split(" - ", 1)[1]
            log_records.append(msg)

    handler_id: int = logger.add(capture_logs, level="INFO")

    yield log_records

    logger.remove(handler_id)


class TestBatchRoiBasic:
    """Test basic batch_roi functionality without filesystem dependencies."""

    def test_single_roi_passthrough(self) -> None:
        """Test that non-wildcard ROI passes through unchanged."""

        @batch_roi()
        def mock_func(path: Path, roi: str) -> str:
            return f"processed_{roi}"

        result = mock_func(path=Path("/test"), roi="roi1")
        assert result == "processed_roi1"

    def test_single_roi_with_codebook_passthrough(self) -> None:
        """Test single ROI with codebook passes through unchanged."""

        @batch_roi(include_codebook=True)
        def mock_func(path: Path, roi: str, codebook: str) -> str:
            return f"processed_{roi}_{codebook}"

        result = mock_func(path=Path("/test"), roi="roi1", codebook="book1")
        assert result == "processed_roi1_book1"

    def test_non_wildcard_roi_values(self) -> None:
        """Test various non-wildcard ROI values pass through correctly."""

        @batch_roi()
        def mock_func(path: Path, roi: str) -> str:
            return f"result_{roi}"

        test_cases: list[str] = ["roi1", "cortex", "hippocampus", "roi_123", "test-roi"]
        for roi in test_cases:
            result = mock_func(path=Path("/test"), roi=roi)
            assert result == f"result_{roi}"


class TestBatchRoiWildcardBehavior:
    """Test batch processing with wildcard ROI patterns."""

    def test_wildcard_batch_processing(self, basic_roi_structure: Path) -> None:
        """Test basic wildcard batch processing finds all ROIs."""
        call_log: list[str] = []

        @batch_roi()
        def mock_func(path: Path, roi: str) -> None:
            call_log.append(f"processed_{roi}")

        result = mock_func(path=basic_roi_structure, roi="*")

        # Should return None for batch processing
        assert result is None

        # Should have called function for each ROI
        expected_calls = {"processed_roi1", "processed_roi2", "processed_cortex"}
        assert set(call_log) == expected_calls
        assert len(call_log) == 3

    def test_wildcard_with_custom_pattern(self, basic_roi_structure: Path) -> None:
        """Test wildcard with custom directory pattern."""
        # Create additional directories with different pattern
        (basic_roi_structure / "stitch--roi1").mkdir(exist_ok=True)
        (basic_roi_structure / "stitch--roi2").mkdir(exist_ok=True)

        call_log: list[str] = []

        @batch_roi(look_for="stitch--*")
        def mock_func(path: Path, roi: str) -> None:
            call_log.append(f"processed_{roi}")

        mock_func(path=basic_roi_structure, roi="*")

        expected_calls = {"processed_roi1", "processed_roi2"}
        assert set(call_log) == expected_calls
        assert len(call_log) == 2

    def test_wildcard_no_matching_directories(self, basic_roi_structure: Path) -> None:
        """Test wildcard when no matching directories exist."""
        call_log: list[str] = []

        @batch_roi(look_for="nonexistent--*")
        def mock_func(path: Path, roi: str) -> None:
            call_log.append(f"processed_{roi}")

        result = mock_func(path=basic_roi_structure, roi="*")

        # Should return None but not call function (no ROIs found)
        assert result is None
        assert len(call_log) == 0


class TestBatchRoiCodebookIntegration:
    """Test codebook integration functionality."""

    def test_codebook_string_integration(self, codebook_roi_structure: Path) -> None:
        """Test codebook integration with string codebook."""
        call_log: list[str] = []

        @batch_roi(include_codebook=True)
        def mock_func(path: Path, roi: str, codebook: str) -> None:
            call_log.append(f"processed_{roi}_{codebook}")

        mock_func(path=codebook_roi_structure, roi="*", codebook="book1")

        # Should find roi1 and roi2 with book1
        expected_calls = {"processed_roi1_book1", "processed_roi2_book1"}
        assert set(call_log) == expected_calls
        assert len(call_log) == 2

    def test_codebook_path_integration(self, codebook_roi_structure: Path) -> None:
        """Test codebook integration with Path codebook."""
        call_log: list[str] = []

        @batch_roi(include_codebook=True)
        def mock_func(path: Path, roi: str, codebook: str) -> None:
            call_log.append(f"processed_{roi}_{codebook}")

        codebook_path: Path = Path("/some/path/book1.csv")
        mock_func(path=codebook_roi_structure, roi="*", codebook=codebook_path)  # type: ignore[arg-type]

        # The decorator passes the original Path object as-is to the function
        # Only the file pattern matching uses .stem
        expected_calls = {"processed_roi1_/some/path/book1.csv", "processed_roi2_/some/path/book1.csv"}
        assert set(call_log) == expected_calls

    def test_codebook_split_disabled(self, codebook_roi_structure: Path) -> None:
        """Test codebook behavior with split_codebook=False."""
        call_log: list[str] = []

        @batch_roi(include_codebook=True, split_codebook=False)
        def mock_func(path: Path, roi: str, codebook: str) -> None:
            call_log.append(f"processed_{roi}_{codebook}")

        mock_func(path=codebook_roi_structure, roi="*", codebook="book1")

        # Should include full directory name including codebook
        expected_calls = {"processed_roi1+book1_book1", "processed_roi2+book1_book1"}
        assert set(call_log) == expected_calls

    def test_codebook_missing_parameter_error(self, codebook_roi_structure: Path) -> None:
        """Test error when codebook parameter is required but missing."""

        @batch_roi(include_codebook=True)
        def mock_func(path: Path, roi: str, codebook: str) -> None:
            pass

        with pytest.raises(ValueError, match="batch_roi with include_codebook=True requires codebook"):
            mock_func(path=codebook_roi_structure, roi="*")  # type: ignore[misc]

    def test_codebook_invalid_type_error(self, codebook_roi_structure: Path) -> None:
        """Test error when codebook is invalid type."""

        @batch_roi(include_codebook=True)
        def mock_func(path: Path, roi: str, codebook: str) -> None:
            pass

        with pytest.raises(ValueError, match="codebook must be a string or Path"):
            mock_func(path=codebook_roi_structure, roi="*", codebook=123)  # type: ignore[arg-type]


class TestBatchRoiEdgeCases:
    """Test edge cases and error conditions."""

    def test_complex_roi_names(self, temp_workspace: Path) -> None:
        """Test ROI names with special characters and patterns."""
        complex_dirs: list[str] = [
            "registered--roi_with_underscores",
            "registered--roi-with-dashes",
            "registered--roi123numbers",
            "registered--ROI_CAPS",
        ]
        for roi_dir in complex_dirs:
            (temp_workspace / roi_dir).mkdir(parents=True, exist_ok=True)

        call_log: list[str] = []

        @batch_roi()
        def mock_func(path: Path, roi: str) -> None:
            call_log.append(roi)

        mock_func(path=temp_workspace, roi="*")

        expected_rois = {"roi_with_underscores", "roi-with-dashes", "roi123numbers", "ROI_CAPS"}
        assert set(call_log) == expected_rois

    def test_nested_directory_structure(self, temp_workspace: Path) -> None:
        """Test that decorator only looks at direct children."""
        # Create nested structure
        (temp_workspace / "registered--roi1").mkdir(parents=True, exist_ok=True)
        (temp_workspace / "subdir" / "registered--nested").mkdir(parents=True, exist_ok=True)

        call_log: list[str] = []

        @batch_roi()
        def mock_func(path: Path, roi: str) -> None:
            call_log.append(roi)

        mock_func(path=temp_workspace, roi="*")

        # Should only find direct children, not nested
        assert call_log == ["roi1"]

    def test_function_with_additional_parameters(self, temp_workspace: Path) -> None:
        """Test decorator with functions that have additional parameters."""
        (temp_workspace / "registered--roi1").mkdir(exist_ok=True)

        call_log: list[str] = []

        @batch_roi()
        def mock_func(path: Path, roi: str, extra_param: str = "default", **kwargs: Any) -> None:
            call_log.append(f"{roi}_{extra_param}_{kwargs.get('optional', 'none')}")

        mock_func(path=temp_workspace, roi="*", extra_param="test", optional="value")

        assert call_log == ["roi1_test_value"]

    def test_function_exception_handling(self, temp_workspace: Path) -> None:
        """Test that exceptions in decorated function are propagated."""
        (temp_workspace / "registered--roi1").mkdir(exist_ok=True)

        @batch_roi()
        def failing_func(path: Path, roi: str) -> None:
            raise RuntimeError(f"Error processing {roi}")

        with pytest.raises(RuntimeError, match="Error processing roi1"):
            failing_func(path=temp_workspace, roi="*")

    def test_empty_roi_after_split(self, temp_workspace: Path) -> None:
        """Test handling of directories with malformed names."""
        # Create directory with malformed name (no ROI part after split)
        (temp_workspace / "registered--").mkdir(exist_ok=True)
        (temp_workspace / "registered--roi1").mkdir(exist_ok=True)

        call_log: list[str] = []

        @batch_roi()
        def mock_func(path: Path, roi: str) -> None:
            call_log.append(roi)

        mock_func(path=temp_workspace, roi="*")

        # Should handle empty string ROI gracefully
        expected_rois = {"", "roi1"}
        assert set(call_log) == expected_rois


class TestBatchRoiFilesystemInteraction:
    """Test filesystem interaction patterns and path handling."""

    @pytest.fixture
    def filesystem_structure(self, temp_workspace: Path) -> Path:
        """Create realistic FISH directory structure."""
        roi_structures: dict[str, list[str]] = {
            "registered--cortex": ["image-001.tif", "image-002.tif"],
            "registered--hippocampus": ["image-001.tif", "image-003.tif"],
            "registered--striatum+book1": ["image-001.tif"],
            "stitch--cortex": ["fused.tif"],
        }

        for roi_dir, files in roi_structures.items():
            roi_path = temp_workspace / roi_dir
            roi_path.mkdir(parents=True, exist_ok=True)
            for file in files:
                (roi_path / file).touch()

        return temp_workspace

    def test_path_resolution(self, filesystem_structure: Path) -> None:
        """Test that path resolution works correctly."""
        call_log: list[str] = []

        @batch_roi()
        def mock_func(path: Path, roi: str) -> None:
            # Verify path is correctly passed through
            assert path == filesystem_structure
            assert path.exists()
            call_log.append(roi)

        mock_func(path=filesystem_structure, roi="*")

        expected_rois = {"cortex", "hippocampus", "striatum"}
        assert set(call_log) == expected_rois

    def test_relative_vs_absolute_paths(self, filesystem_structure: Path) -> None:
        """Test behavior with relative vs absolute paths."""
        # Change to temp directory to test relative paths
        original_cwd = os.getcwd()
        os.chdir(filesystem_structure)

        try:
            call_log: list[str] = []

            @batch_roi()
            def mock_func(path: Path, roi: str) -> None:
                call_log.append(roi)

            # Test with relative path
            mock_func(path=Path("."), roi="*")

            expected_rois = {"cortex", "hippocampus", "striatum"}
            assert set(call_log) == expected_rois

        finally:
            os.chdir(original_cwd)

    def test_symlink_handling(self, filesystem_structure: Path) -> None:
        """Test behavior with symbolic links."""
        # Create symlink to one of the ROI directories
        target_dir = filesystem_structure / "registered--cortex"
        symlink_dir = filesystem_structure / "registered--cortex_link"

        try:
            symlink_dir.symlink_to(target_dir)

            call_log: list[str] = []

            @batch_roi()
            def mock_func(path: Path, roi: str) -> None:
                call_log.append(roi)

            mock_func(path=filesystem_structure, roi="*")

            # Should include both original and symlinked directories
            expected_rois = {"cortex", "hippocampus", "striatum", "cortex_link"}
            assert set(call_log) == expected_rois

        except (OSError, NotImplementedError):
            # Skip test if symlinks not supported
            pytest.skip("Symlinks not supported on this system")

    def test_permission_denied_directory(self, filesystem_structure: Path) -> None:
        """Test handling of directories with restricted permissions."""
        # Create directory with restricted permissions
        restricted_dir = filesystem_structure / "registered--restricted"
        restricted_dir.mkdir(exist_ok=True)

        try:
            # Restrict permissions (may not work on all systems)
            restricted_dir.chmod(0o000)

            call_log: list[str] = []

            @batch_roi()
            def mock_func(path: Path, roi: str) -> None:
                call_log.append(roi)

            # Should still find the directory in glob, but function might fail
            mock_func(path=filesystem_structure, roi="*")

            # Verify restricted directory was found
            assert "restricted" in call_log

        except (OSError, PermissionError):
            # Skip if permission changing not supported
            pytest.skip("Permission changing not supported")
        finally:
            # Restore permissions for cleanup
            try:
                restricted_dir.chmod(0o755)
            except (OSError, PermissionError):
                pass


class TestBatchRoiConcurrency:
    """Test concurrent execution and thread safety."""

    @pytest.fixture
    def concurrent_workspace(self, temp_workspace: Path) -> Path:
        """Create test environment for concurrency tests."""
        # Create multiple ROI directories
        for i in range(5):
            (temp_workspace / f"registered--roi{i}").mkdir(parents=True, exist_ok=True)
        return temp_workspace

    def test_concurrent_batch_processing(self, concurrent_workspace: Path) -> None:
        """Test that batch processing doesn't interfere with concurrent calls."""
        call_log: list[str] = []
        call_lock: threading.Lock = threading.Lock()

        @batch_roi()
        def mock_func(path: Path, roi: str) -> None:
            # Simulate some processing time
            time.sleep(0.01)
            with call_lock:
                call_log.append(f"thread_{threading.current_thread().ident}_{roi}")

        # Create multiple threads calling batch processing
        threads: list[threading.Thread] = []
        for i in range(3):
            thread = threading.Thread(target=mock_func, kwargs={"path": concurrent_workspace, "roi": "*"})
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Verify all ROIs were processed by all threads
        assert len(call_log) == 15  # 3 threads × 5 ROIs

        # Verify each thread processed all ROIs
        thread_ids = set()
        for call in call_log:
            thread_id = call.split("_")[1]
            thread_ids.add(thread_id)

        assert len(thread_ids) == 3  # 3 different threads

    def test_kwargs_mutation_safety(self, concurrent_workspace: Path) -> None:
        """Test that kwargs mutations don't affect other calls."""
        original_kwargs_log: list[dict[str, Any]] = []
        modified_kwargs_log: list[dict[str, Any]] = []

        @batch_roi()
        def mock_func(path: Path, roi: str, **kwargs: Any) -> None:
            # Log original kwargs
            original_kwargs_log.append(dict(kwargs))

            # Modify kwargs (should not affect other calls)
            kwargs["modified"] = True
            kwargs["roi"] = f"modified_{roi}"

            # Log modified kwargs
            modified_kwargs_log.append(dict(kwargs))

        # Call with wildcard
        mock_func(path=concurrent_workspace, roi="*", original_param="value")

        # Verify original kwargs were preserved for each call
        assert len(original_kwargs_log) == 5
        for kwargs_dict in original_kwargs_log:
            assert kwargs_dict["original_param"] == "value"
            assert "modified" not in kwargs_dict

        # Verify modifications were isolated to each call
        assert len(modified_kwargs_log) == 5
        for kwargs_dict in modified_kwargs_log:
            assert kwargs_dict["modified"] is True
            assert kwargs_dict["original_param"] == "value"


class TestBatchRoiLogging:
    """Test logging behavior and integration."""

    @pytest.fixture
    def logging_workspace(self, temp_workspace: Path) -> Path:
        """Create test environment with ROI directories for logging tests."""
        for roi in ["roi1", "roi2", "cortex"]:
            (temp_workspace / f"registered--{roi}").mkdir(parents=True, exist_ok=True)
        return temp_workspace

    def test_batch_logging_messages(self, logging_workspace: Path, log_capture: list[str]) -> None:
        """Test that batch processing logs appropriate messages."""

        @batch_roi()
        def mock_func(path: Path, roi: str) -> None:
            pass

        mock_func(path=logging_workspace, roi="*")

        # Should log batching message for each ROI
        expected_messages = {"Batching roi1", "Batching roi2", "Batching cortex"}
        assert set(log_capture) == expected_messages

    def test_single_roi_no_logging(self, logging_workspace: Path, log_capture: list[str]) -> None:
        """Test that single ROI processing doesn't log batch messages."""

        @batch_roi()
        def mock_func(path: Path, roi: str) -> None:
            pass

        mock_func(path=logging_workspace, roi="roi1")

        # Should not log any batching messages
        assert len(log_capture) == 0

    def test_logging_with_codebook(self, temp_workspace: Path, log_capture: list[str]) -> None:
        """Test logging messages include codebook information."""
        # Create codebook directories
        for roi in ["roi1", "roi2"]:
            (temp_workspace / f"registered--{roi}+book1").mkdir(exist_ok=True)

        @batch_roi(include_codebook=True)
        def mock_func(path: Path, roi: str, codebook: str) -> None:
            pass

        mock_func(path=temp_workspace, roi="*", codebook="book1")

        expected_messages = {"Batching roi1", "Batching roi2"}
        assert set(log_capture) == expected_messages


class TestBatchRoiTypeAnnotations:
    """Test type annotation preservation and typing behavior."""

    def test_return_type_preservation(self) -> None:
        """Test that return types are properly preserved."""

        @batch_roi()
        def typed_func(path: Path, roi: str) -> str:
            return f"result_{roi}"

        # Single ROI should return string
        result = typed_func(path=Path("/test"), roi="roi1")
        assert isinstance(result, str)
        assert result == "result_roi1"

    def test_none_return_for_batch(self, temp_workspace: Path) -> None:
        """Test that batch processing returns None."""
        (temp_workspace / "registered--roi1").mkdir(exist_ok=True)

        @batch_roi()
        def typed_func(path: Path, roi: str) -> str:
            return f"result_{roi}"

        # Batch processing should return None
        result = typed_func(path=temp_workspace, roi="*")
        assert result is None

    def test_generic_type_preservation(self) -> None:
        """Test preservation of generic types and complex annotations."""

        @batch_roi()
        def generic_func(path: Path, roi: str) -> list[dict[str, Any]]:
            return [{"roi": roi, "data": [1, 2, 3]}]

        result = generic_func(path=Path("/test"), roi="roi1")
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["roi"] == "roi1"
        assert result[0]["data"] == [1, 2, 3]


class TestBatchRoiRealWorldScenarios:
    """Test real-world usage scenarios mimicking actual fishtools usage."""

    def test_register_command_simulation(
        self, realistic_workspace: tuple[Path, list[str], list[str]]
    ) -> None:
        """Simulate the register command from cli_stitch.py."""
        workspace, rois, codebooks = realistic_workspace
        processed_rois: list[str] = []

        # Add base registered directories without codebook suffixes
        for roi in rois:
            (workspace / f"registered--{roi}").mkdir(parents=True, exist_ok=True)

        @batch_roi()
        def register_simulation(path: Path, roi: str, **kwargs: Any) -> None:
            # Simulate finding registered images
            roi_path = path / f"registered--{roi}"
            if roi_path.exists():
                processed_rois.append(roi)

        register_simulation(path=workspace, roi="*")

        # Should process all base ROIs (without codebook suffix)
        expected_rois = set(rois)
        assert set(processed_rois) == expected_rois

    def test_fuse_command_simulation(self, realistic_workspace: tuple[Path, list[str], list[str]]) -> None:
        """Simulate the fuse command with codebook integration."""
        workspace, rois, codebooks = realistic_workspace
        processed_combinations: list[str] = []

        @batch_roi("registered--*", include_codebook=True, split_codebook=True)
        def fuse_simulation(path: Path, roi: str, codebook: str, **kwargs: Any) -> None:
            # Simulate processing specific ROI+codebook combination
            processed_combinations.append(f"{roi}+{codebook}")

        # Test just the first codebook to avoid state issues
        codebook: str = codebooks[0]  # "geneA"
        fuse_simulation(path=workspace, roi="*", codebook=codebook)

        # Should process all ROIs for this codebook
        expected_combinations = {f"{roi}+{codebook}" for roi in rois}
        assert set(processed_combinations) == expected_combinations

    def test_mixed_directory_patterns(self, realistic_workspace: tuple[Path, list[str], list[str]]) -> None:
        """Test handling of mixed directory patterns in same workspace."""
        workspace, rois, codebooks = realistic_workspace

        # Add some non-matching directories
        (workspace / "other--directory").mkdir(exist_ok=True)
        (workspace / "random_folder").mkdir(exist_ok=True)
        (workspace / "registered--partial").mkdir(exist_ok=True)  # No codebook

        processed_rois: list[str] = []

        @batch_roi("registered--*", include_codebook=True, split_codebook=True)
        def selective_processing(path: Path, roi: str, codebook: str) -> None:
            processed_rois.append(roi)

        selective_processing(path=workspace, roi="*", codebook="geneA")

        # Should only process ROIs that match the pattern and have the codebook
        expected_rois = set(rois)
        assert set(processed_rois) == expected_rois
