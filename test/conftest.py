import random
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import tifffile

# Lightweight stub for heavy optional dependency 'scanpy' used in import-time
# paths (e.g., fishtools.postprocess). This avoids importing the real package
# during tests where it is not exercised.
if "scanpy" not in sys.modules:
    _sc = types.ModuleType("scanpy")
    _sc.pp = types.SimpleNamespace(
        filter_cells=lambda *a, **k: None,
        filter_genes=lambda *a, **k: None,
        normalize_total=lambda *a, **k: {"X": None},
        log1p=lambda *a, **k: None,
        calculate_qc_metrics=lambda *a, **k: None,
    )
    _sc.tl = types.SimpleNamespace(leiden=lambda *a, **k: None)
    _sc.pl = types.SimpleNamespace(
        embedding=lambda *a, **k: types.SimpleNamespace(axes=[]),
        rank_genes_groups=lambda *a, **k: None,
    )
    _sc.get = types.SimpleNamespace(rank_genes_groups_df=lambda *a, **k: types.SimpleNamespace(head=lambda n: {"names": []}))
    _sc.experimental = types.SimpleNamespace(
        pp=types.SimpleNamespace(
            highly_variable_genes=lambda *a, **k: None,
            normalize_pearson_residuals=lambda *a, **k: None,
        )
    )
    sys.modules["scanpy"] = _sc


@pytest.fixture(autouse=True)
def _seed_all() -> None:
    """Deterministic RNG across tests to reduce flakiness."""
    random.seed(0)
    np.random.seed(0)


@pytest.fixture
def mock_workspace(tmp_path: Path) -> Path:
    """Create a mock workspace directory structure with registered TIFFs.

    - Creates analysis/deconv/registered--{roi}+{cb}/ folders
    - Writes 5 small registered TIFFs per ROI/CB with metadata['key']
    - Creates opt_{cb} folders with minimal optimization files
    """
    base = tmp_path / "analysis" / "deconv"

    # Create registered directories with different ROIs and codebooks
    for roi in ["roi1", "roi2"]:
        for cb in ["cb1", "cb2"]:
            reg_dir = base / f"registered--{roi}+{cb}"
            reg_dir.mkdir(parents=True)

            # Create mock registered TIFF files
            for i in range(5):
                tiff_path = reg_dir / f"reg{i:04d}.tif"
                data = np.random.randint(0, 65535, (3, 4, 100, 100), dtype=np.uint16)
                tifffile.imwrite(
                    tiff_path,
                    data,
                    metadata={"key": list(range(1, 37))},
                    compression="zlib",
                )

    # Create optimization directories
    for cb in ["cb1", "cb2"]:
        opt_dir = base / f"opt_{cb}"
        opt_dir.mkdir(parents=True)
        (opt_dir / "global_scale.txt").write_text("1.0 1.1 1.2 1.3")
        (opt_dir / "global_min.txt").write_text("0.1 0.2 0.3 0.4")
        (opt_dir / "percentiles.json").write_text("[]")

    # Codebooks dir (empty, tests may write into it)
    (tmp_path / "codebooks").mkdir(exist_ok=True)

    return tmp_path


@pytest.fixture
def raw_bleed_setup(tmp_path: Path) -> tuple[Path, Path, float, float]:
    """Create paired sample/blank directories with a known bleed-through relationship."""

    analysis_path = tmp_path / "analysis"
    preprocessed_dir = tmp_path / "preprocessed"
    analysis_path.mkdir(parents=True)
    preprocessed_dir.mkdir(parents=True)

    slope = 0.18
    intercept = 75.0

    rng = np.random.default_rng(42)
    for i in range(6):
        blank_slice = rng.uniform(200, 800, size=(64, 64)).astype(np.float32)
        image_slice = blank_slice * slope + intercept + rng.normal(0, 3, size=blank_slice.shape)

        np.savez_compressed(
            preprocessed_dir / f"reg-{i:04d}_key-1.npz",
            image_slice=image_slice,
            blank_slice=blank_slice,
        )

    return analysis_path, preprocessed_dir, slope, intercept


@pytest.fixture
def registered_stack_data(
    tmp_path: Path,
) -> tuple[Path, np.ndarray, dict[str, int], float, float, np.ndarray, np.ndarray]:
    """Create a synthetic registered stack with signal tied to a blank channel."""

    path = tmp_path / "registered_stack.tif"
    slope = 0.22
    intercept = 40.0

    rng = np.random.default_rng(7)
    blank = rng.uniform(300, 900, size=(8, 64, 64)).astype(np.float32)
    signal = blank * slope + intercept + rng.normal(0, 1.5, size=blank.shape)
    other = rng.uniform(50, 120, size=blank.shape).astype(np.float32)
    fid = rng.uniform(0, 1, size=blank.shape).astype(np.float32)

    stack = np.stack([signal, blank, other, fid], axis=1).astype(np.uint16)
    channel_names = ["geneA", "Blank-560", "geneB", "fiducial"]
    channel_index = {name: idx for idx, name in enumerate(channel_names)}

    tifffile.imwrite(path, stack, metadata={"key": channel_names})

    return path, stack, channel_index, slope, intercept, signal, blank
