import logging
from pathlib import Path
from typing import Literal, TypedDict

import numpy as np
import numpy.typing as npt
import tifffile
from imagecodecs import imread, jpegxl_encode
from tifffile import TiffFile, imwrite


class JPEG_XR_KWARGS_(TypedDict):
    photometric: Literal["minisblack"]
    compression: Literal[22610]


JPEG_XR_KWARGS = JPEG_XR_KWARGS_(photometric="minisblack", compression=22610)


def check_jpegxr(path: Path):
    with TiffFile(path) as tif:
        if tif.pages[0].compression == 22610:
            return True, tif.asarray()
        return False, tif.asarray()


def dax_reader(path: Path | str, n_frames: int | None = None):
    path = Path(path)
    if path.suffix != ".dax":
        raise ValueError(f"Unknown file type {path}. Must be .dax")

    def parse_inf(f: str):
        return dict(x.split(" = ") for x in f.split("\n") if x)

    if n_frames is None:
        inf = parse_inf(path.with_suffix(".inf").read_text())
        n_frames = int(inf["number of frames"])

    image_data = np.fromfile(path, dtype=np.uint16, count=2048 * 2048 * n_frames).reshape(
        n_frames, 2048, 2048
    )
    image_data.byteswap(False)

    return image_data


def dax_writer(path: Path | str, img: npt.NDArray[np.uint16]):
    path = Path(path)
    if path.suffix != ".dax":
        raise ValueError(f"Unknown file type {path}. Must be .dax")
    img.tofile(path)


def compress(path: Path, *, level: int, log: logging.Logger = logging.getLogger(__name__)):
    match path.suffix:
        case ".jp2":
            import glymur

            jp2 = glymur.Jp2k(path)
            img = np.moveaxis(jp2[:], 2, 0)  # type: ignore
        case ".tif" | ".tiff":
            is_jpegxr, img = check_jpegxr(path)
            if level == 100 and is_jpegxr:
                return False  # to save time
        case ".dax":
            img = dax_reader(path)
        case _:
            raise NotImplementedError(f"Unknown file type {path.suffix}")
    try:
        if level == 100:
            imwrite(path.with_suffix(".tif"), img, **JPEG_XR_KWARGS)
        else:
            path.with_suffix(".jxl").write_bytes(jpegxl_encode(img, level=level, photometric=2))
    except Exception as e:
        log.error(f"Failed to compress {path} with error {e}")

    return True


def decompress(path: Path, out: Literal["tif", "dax"]):
    if path.suffix == ".jxl":
        img = imread(path)  # type: ignore
    elif path.suffix == ".tif":
        img = tifffile.imread(path)
    else:
        raise ValueError(f"Unknown file type {path}. Must be .jxl or .tif")

    if out == "tif":
        imwrite(path.with_suffix(".tif"), img, **JPEG_XR_KWARGS)
    else:
        dax_writer(path.with_suffix(".dax"), img)

    return True
