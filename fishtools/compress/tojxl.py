import logging
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from itertools import chain
from pathlib import Path

import click
import numpy as np
import tifffile
from imagecodecs import jpegxl_encode
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")


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


def run(path: Path, *, level: int, remove: bool = False):
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

    path.with_suffix(".jxl").write_bytes(jpegxl_encode(img, level=level))
    if remove:
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
    file_types = [".tif", ".tiff", ".jp2", ".dax"]
    files = list(chain.from_iterable([Path(path).glob(f"**/*{file_type}") for file_type in file_types]))
    click.echo(f"Found {len(files)} files")
    with ProcessPoolExecutor() as pool:
        with tqdm(total=len(files)) as progress, logging_redirect_tqdm():
            futures = []
            for file in files:
                future = pool.submit(run, file, level=quality, remove=delete)
                future.add_done_callback(lambda _: progress.update())
                future.add_done_callback(partial(log.info, f"Finished {file}"))
                futures.append(future)
            [future.result() for future in futures]
            log.info("Done")


if __name__ == "__main__":
    main()
