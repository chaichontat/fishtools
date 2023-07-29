import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Callable

import click
import numpy.typing as npt
from imagecodecs import imread
from tifffile import TiffFile
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

log = logging.getLogger(__name__)


def run(path: Path, *, delete: bool):
    if path.suffix == ".jxl":
        img: npt.NDArray[Any] = imread(path)  # type: ignore
    elif path.suffix == ".tif":
        with TiffFile(path) as tif:
            if tif.pages[0].compression != 22610:
                return False
            img = tif.asarray()
    else:
        raise ValueError(f"Unknown file type {path}. Must be .jxl or .tif")
    img.tofile(path.with_suffix(".dax"))
    if delete:
        path.unlink()

    return True


@click.command()
@click.argument(
    "path",
    type=click.Path(
        exists=True,
        dir_okay=True,
        file_okay=True,
    ),
)
@click.option("--delete", "-d", is_flag=True, help="Delete original files")
def main(path: Path, delete: bool = False):
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    path = Path(path)
    files = list(path.glob("**/*.jxl")) + list(path.glob("**/*.tif")) if path.is_dir() else [path]
    log.info(f"Found {len(files)} files")
    with ProcessPoolExecutor() as pool:
        with tqdm(total=len(files)) as progress, logging_redirect_tqdm():
            futures = []

            def gen(msg: str) -> Callable[..., None]:
                return lambda x: log.info(msg) if x.result() else None

            for file in files:
                future = pool.submit(run, file, delete=delete)
                future.add_done_callback(lambda _: progress.update())
                future.add_done_callback(gen(f"Finished {file.as_posix()}"))
                futures.append(future)
            [future.result() for future in futures]
            log.info("Done")


if __name__ == "__main__":
    main()
