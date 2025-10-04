import numpy as np
import tifffile


def test_extract_channel_releases_gpu(monkeypatch, tmp_path):
    # Arrange
    from fishtools.preprocess import cli_stitch as cs

    calls = {"count": 0}

    def _release():
        calls["count"] += 1

    monkeypatch.setattr(cs, "gpu_release_all", _release)

    data = (np.random.default_rng(0).integers(0, 1000, size=(3, 32, 32))).astype(np.uint16)
    in_path = tmp_path / "reg-0001.tif"
    out_path = tmp_path / "out.tif"
    tifffile.imwrite(in_path, data, metadata={"axes": "CYX"})

    # Act
    cs.extract_channel(in_path, out_path, idx=1, trim=0, downsample=2)

    # Assert
    assert out_path.exists()
    assert calls["count"] >= 1, "Expected GPU cleanup to be invoked at least once"


def test_extract_2d_releases_gpu(monkeypatch, tmp_path):
    from fishtools.preprocess import cli_stitch as cs

    calls = {"count": 0}

    def _release():
        calls["count"] += 1

    monkeypatch.setattr(cs, "gpu_release_all", _release)

    data = (np.random.default_rng(1).integers(0, 1000, size=(3, 32, 32))).astype(np.uint16)
    in_path = tmp_path / "reg-0042.tif"
    out_dir = tmp_path / "out"
    out_dir.mkdir(exist_ok=True)
    tifffile.imwrite(in_path, data, metadata={"axes": "CYX", "key": ["a", "b", "c"]})

    cs.extract(in_path, out_dir, is_2d=True, downsample=2)

    # Expect at least one channel file written
    written = list(out_dir.glob("*/0042.tif"))
    assert written, "Expected at least one output image to be written"
    assert calls["count"] >= 1, "Expected GPU cleanup to be invoked at least once"


def test_extract_3d_releases_gpu(monkeypatch, tmp_path):
    from fishtools.preprocess import cli_stitch as cs

    calls = {"count": 0}

    def _release():
        calls["count"] += 1

    monkeypatch.setattr(cs, "gpu_release_all", _release)

    # ZCYX shape -> will be normalized to at least 3D in the function
    data = (np.random.default_rng(2).integers(0, 1000, size=(2, 2, 16, 16))).astype(np.uint16)
    in_path = tmp_path / "reg-0100.tif"
    out_dir = tmp_path / "out3d"
    out_dir.mkdir(exist_ok=True)
    tifffile.imwrite(in_path, data, metadata={"axes": "ZCYX", "key": ["a", "b"]})

    cs.extract(in_path, out_dir, is_2d=False, downsample=2, subsample_z=1, channels=[0])

    # Expect at least one output tile
    written = list(out_dir.glob("*/*/0100.tif"))
    assert written, "Expected at least one output image to be written"
    assert calls["count"] >= 1, "Expected GPU cleanup to be invoked at least once"

