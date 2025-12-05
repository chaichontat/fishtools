from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fishtools.segmentation.distributed.cache_utils import (
    read_nonempty_cache,
    read_normalization_cache,
    write_nonempty_cache,
    write_normalization_cache,
)


def test_nonempty_cache_round_trip(tmp_path: Path) -> None:
    cache_path = tmp_path / "nonempty.json"
    blocksize = (8, 16, 32)
    idxs = [0, 2, 5]

    write_nonempty_cache(cache_path, blocksize, idxs)
    assert cache_path.exists()

    loaded = read_nonempty_cache(cache_path, blocksize)
    assert loaded == idxs


def test_nonempty_cache_block_mismatch(tmp_path: Path) -> None:
    cache_path = tmp_path / "nonempty.json"
    write_nonempty_cache(cache_path, (4, 4, 4), [1, 3])

    assert read_nonempty_cache(cache_path, (4, 4, 5)) is None


def test_normalization_cache_round_trip(tmp_path: Path) -> None:
    cache_path = tmp_path / "normalization.json"
    normalization = {"lowhigh": [[1.0, 2.0], [3.0, 4.0]]}

    write_normalization_cache(cache_path, normalization)
    assert cache_path.exists()

    loaded = read_normalization_cache(cache_path)
    assert loaded == normalization


def test_normalization_cache_invalid_json(tmp_path: Path) -> None:
    cache_path = tmp_path / "normalization.json"
    cache_path.write_text("not-json")

    assert read_normalization_cache(cache_path) is None


def test_lowhigh_cache_roundtrip() -> None:
    """Test the inlined lowhigh cache pattern."""
    lowhigh = np.array([[20.0, 40.0], [10.0, 30.0]])
    channels = [2, 1]

    # Cache pattern (inlined)
    cached = {str(ch): lh.tolist() for ch, lh in zip(channels, lowhigh)}
    assert cached == {"2": [20.0, 40.0], "1": [10.0, 30.0]}

    # Retrieve in different order
    selected = np.array([cached[str(ch)] for ch in [1, 2]], dtype=np.float64)
    assert np.allclose(selected, np.array([[10.0, 30.0], [20.0, 40.0]]))


def test_lowhigh_cache_missing_key() -> None:
    """Test KeyError when channel missing from cache."""
    cache = {"1": [10.0, 30.0]}

    with pytest.raises(KeyError):
        np.array([cache[str(ch)] for ch in [2]], dtype=np.float64)
