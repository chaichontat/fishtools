from __future__ import annotations

from pathlib import Path

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
