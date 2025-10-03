from pathlib import Path

import numpy as np
import pytest
from tifffile import TiffFile

from fishtools.utils.io import safe_imwrite


def test_safe_imwrite_creates_final_file_without_partial(tmp_path):
    destination = tmp_path / "example.tif"
    array = np.arange(8, dtype=np.uint16).reshape(2, 2, 2)

    safe_imwrite(destination, array)

    assert destination.exists()
    assert not destination.with_name("example.tif.partial").exists()

    with TiffFile(destination) as tif:
        data = tif.asarray()

    assert np.array_equal(data, array)


def test_safe_imwrite_cleans_partial_on_writer_failure(tmp_path):
    destination = tmp_path / "broken.tif"

    class Boom(RuntimeError):
        pass

    def failing_writer(path: Path | str, *_args, **_kwargs):
        raise Boom("write failed")

    with pytest.raises(Boom):
        safe_imwrite(destination, np.zeros((1, 1), dtype=np.uint16), imwrite_func=failing_writer)

    assert not destination.exists()
    assert not destination.with_name("broken.tif.partial").exists()


def test_safe_imwrite_cleans_partial_on_rename_failure(tmp_path, monkeypatch):
    destination = tmp_path / "rename.tif"
    partial_path = destination.with_name("rename.tif.partial")

    def stub_writer(path: Path | str, *_args, **_kwargs):
        Path(path).write_bytes(b"payload")

    def failing_replace(self: Path, target: Path) -> None:  # type: ignore[override]
        raise OSError("rename failed")

    monkeypatch.setattr(Path, "replace", failing_replace, raising=False)

    with pytest.raises(OSError):
        safe_imwrite(destination, np.zeros(1, dtype=np.uint8), imwrite_func=stub_writer)

    assert not destination.exists()
    assert not partial_path.exists()
