import logging
from concurrent.futures import ProcessPoolExecutor
from itertools import chain
from pathlib import Path

import click
import numpy as np
import tifffile
from imagecodecs import jpegxl_encode
from tifffile import imwrite
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

log = logging.getLogger(__name__)


def dax_reader(path: Path):
    def parse_inf(f: str):
        return dict(x.split(" = ") for x in f.split("\n") if x)

    inf = parse_inf(path.with_suffix(".inf").read_text())
    n_frames = int(inf["number of frames"])

    image_data = np.fromfile(path, dtype=np.uint16, count=2048 * 2048 * n_frames).reshape(
        n_frames, 2048, 2048
    )
    image_data.byteswap(False)

    return image_data


def run(path: Path, *, level: int, delete: bool = False):
    match path.suffix:
        case ".jp2":
            import glymur

            jp2 = glymur.Jp2k(path)
            img = np.moveaxis(jp2[:], 2, 0)  # type: ignore
        case ".tif" | ".tiff":
            img = tifffile.imread(path)
        case ".dax":
            img = dax_reader(path)
        case _:
            raise NotImplementedError(f"Unknown file type {path.suffix}")
    try:
        if level == 100:
            imwrite(path.with_suffix(".tif"), img, photometric="minisblack", compression=22610)
        else:
            path.with_suffix(".jxl").write_bytes(jpegxl_encode(img, level=level, photometric=2))
    except Exception as e:
        log.error(f"Failed to compress {path} with error {e}")
    else:
        if delete and path.suffix != ".tif":
            path.unlink()


@click.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=True, file_okay=False))
@click.option("--delete", "-d", is_flag=True, help="Delete original files")
@click.option(
    "--quality",
    "-q",
    default=99,
    type=int,
    help="JPEG XL quality level (-inf-100) 100 = lossless",
)
def main(path: Path, delete: bool = False, quality: int = 99):
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    path = Path(path)
    file_types = [".tif", ".tiff", ".jp2", ".dax"]
    files = list(chain.from_iterable([Path(path).glob(f"**/*{file_type}") for file_type in file_types]))
    log.info(f"Found {len(files)} files")
    with ProcessPoolExecutor() as pool:
        with tqdm(total=len(files)) as progress, logging_redirect_tqdm():

            def submit(file: Path):
                future = pool.submit(run, file, level=quality, delete=delete)
                future.add_done_callback(lambda _: progress.update())
                future.add_done_callback(lambda x: log.info(f"Finished {file.as_posix()}"))
                return future

            [fut.result() for fut in list(map(submit, files))]
            log.info("Done")


if __name__ == "__main__":
    main()
