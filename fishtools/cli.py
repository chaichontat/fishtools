from concurrent.futures import Future, ProcessPoolExecutor
from itertools import chain
from pathlib import Path
from typing import Literal, ParamSpec, Protocol, TypeVar

import rich_click as click
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

import fishtools.compression.compression as lib
from fishtools.io.pretty_print import setup_logging

log = setup_logging()
P, R = ParamSpec("P"), TypeVar("R", covariant=True)


class PathFirst(Protocol[P, R]):
    def __call__(self, path: Path, *args: P.args, **kwargs: P.kwargs) -> R:
        ...


def execute(
    files: list[Path], run: PathFirst[P, R], n_process: int = 16, *args: P.args, **kwargs: P.kwargs
) -> list[R]:
    with ProcessPoolExecutor(n_process) as pool:
        with tqdm(total=len(files)) as progress, logging_redirect_tqdm():

            def submit(file: Path) -> Future[R]:
                future = pool.submit(run, file, *args, **kwargs)
                future.add_done_callback(lambda _: progress.update())
                future.add_done_callback(
                    lambda x: log.info(f"Finished {file.as_posix()}") if x.result() else None
                )
                return future

            res = [fut.result() for fut in list(map(submit, files))]
            log.info("Done")
    return res


@click.group()
def main():
    ...


def process_path(path: Path, file_types: list[str]):
    if path.is_dir():
        files = list(chain.from_iterable([Path(path).glob(f"**/*{file_type}") for file_type in file_types]))
    elif path.suffix in file_types:
        files = [path]
    else:
        raise ValueError(f"Unknown file type {path.suffix}")
    return files


# fmt: off
@main.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=True, file_okay=True, path_type=Path))
@click.option("--delete", "-d", is_flag=True, help="Delete original files")
@click.option("--quality", "-q", default=99, type=int, help="Quality level (-inf-100). 100 = lossless (outputs JPEG-XR TIF)")
@click.option("--n-process", "-n", default=16, type=int, help="Number of processes to use")
# fmt: on
def compress(path: Path, delete: bool = False, quality: int = 99, n_process: int = 16):
    """Converts TIFF, JP2, and DAX image files to JPEG XL (JXL) (lossy) or TIFF-JPEG XR (lossless) files."""
    files = process_path(path, [".tif", ".tiff", ".jp2", ".dax"])
    log.info(f"Found {len(files)} potential file(s).")
    execute(files, lib.compress, level=quality, n_process=n_process)
    if delete and quality <= 97:
        log.warning("Not deleting original files because quality is less than 98. Please delete manually.")
    for file in files:
        if delete and 97 < quality < 100:  # .tif output is only possible when quality == 100.
            file.unlink()


# fmt: off
@main.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=True, file_okay=True, path_type=Path))
@click.option("--delete", "-d", is_flag=True, help="Delete original files")
@click.option("--out", type=click.Choice(("dax", "tif")), default="dax", help="Output format. DAX or JPEG-XR TIF")
@click.option("--n-process", "-n", default=16, type=int, help="Number of processes to use")
# fmt: on
def decompress(path: Path, out: Literal["dax", "tif"], delete: bool = False, n_process: int = 16):
    """Converts JPEG XL (JXL) files to DAX or TIF-JPEG XR files."""
    files = process_path(path, [".tif", ".jxl"])
    n_files = len(files)
    if out == "dax":
        files = list(filter(lambda x: x.with_suffix(".inf").exists(), files))
    if len(files) != n_files:
        log.warning(f"Found {n_files - len(files)} file(s) without corresponding .inf file(s). Skipping.")

    log.info(f"Found {len(files)} file(s).")

    execute(files, lib.decompress, out=out, n_process=n_process)
    for file in files:
        if delete and not (file.suffix == ".tif" and out == "tif"):
            file.unlink()


if __name__ == "__main__":
    main()
