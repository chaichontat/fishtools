import time
from pathlib import Path
from typing import Any

import imagecodecs
import numpy as np
import numpy.typing as npt
import pytest
from click.testing import CliRunner
from tifffile import imread, imwrite

from fishtools.cli import main
from fishtools.compression import compress, dax_reader, decompress

path = Path("__temp.tif")
N_FRAMES = 2


@pytest.fixture(autouse=True)
def img():
    img = np.random.randint(0, 2**16 - 1, size=(N_FRAMES, 2048, 2048), dtype=np.uint16)
    imwrite(path, img)
    yield img
    path.unlink(missing_ok=True)
    path.with_suffix(".jxl").unlink(missing_ok=True)
    path.with_suffix(".dax").unlink(missing_ok=True)


@pytest.fixture(autouse=True)
def runner():
    return CliRunner()


def test_lossless(img: npt.NDArray[np.uint16]):
    size = img.nbytes
    imwrite(path, img)
    compress(path, level=100)
    assert path.with_suffix(".tif").__sizeof__() < size
    assert np.allclose(img, imread(path))


def test_lossy(img: npt.NDArray[np.uint16]):
    size = img.nbytes
    imwrite(path, img)
    compress(path, level=99)
    jxl = path.with_suffix(".jxl")
    imgjxl = imagecodecs.imread(jxl)  # type: ignore

    assert jxl.__sizeof__() < size
    assert np.mean(np.abs(img.astype(np.int32) - imgjxl.astype(np.int32))) < 1000

    decompress(jxl, "tif")
    assert np.allclose(imgjxl, imread(path))
    decompress(jxl, "dax")
    assert np.allclose(imgjxl, dax_reader(path.with_suffix(".dax"), n_frames=N_FRAMES))


def test_dax(img: npt.NDArray[np.uint16]):
    compress(path, level=100)
    decompress(path, "dax")
    assert np.allclose(img, dax_reader(path.with_suffix(".dax"), n_frames=N_FRAMES))


# https://github.com/pallets/click/issues/824
def test_no_wtf_deletes(runner: CliRunner, img: npt.NDArray[np.uint16], capsys: pytest.CaptureFixture[Any]):
    with capsys.disabled():
        # replaces with JPEG-XR
        assert runner.invoke(main, ["compress", "--quality=100", "--delete", path.as_posix()]).exit_code == 0
        assert runner.invoke(main, ["decompress", "--out=tif", path.as_posix()]).exit_code == 0
        time.sleep(0.1)
        assert path.exists()

        assert runner.invoke(main, ["decompress", "--delete", "--out=tif", path.as_posix()]).exit_code == 0
        time.sleep(0.1)
        assert path.exists()

        assert runner.invoke(main, ["compress", "--quality=99", "--delete", path.as_posix()]).exit_code == 0
        time.sleep(0.1)
        assert not path.exists()
        assert path.with_suffix(".jxl").exists()

        assert (
            runner.invoke(
                main, ["decompress", "--delete", "--out=tif", path.with_suffix(".jxl").as_posix()]
            ).exit_code
            == 0
        )
        time.sleep(0.1)
        assert not path.with_suffix(".jxl").exists()
        assert path.exists()


def test_do_not_allow_autodelete_if_quality_below_98(
    runner: CliRunner, img: npt.NDArray[np.uint16], capsys: pytest.CaptureFixture[Any]
):
    with capsys.disabled():
        assert runner.invoke(main, ["compress", "--quality=90", "--delete", path.as_posix()]).exit_code == 0
    time.sleep(0.1)
    assert path.exists()
    assert path.with_suffix(".jxl").exists()
