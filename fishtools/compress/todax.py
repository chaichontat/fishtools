import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import click
from imagecodecs import jpegxl_decode
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")


def run(path: Path):
    if path.suffix != ".jxl":
        raise ValueError(f"Unknown file type {path}. Must be .jxl")

    jpegxl_decode(path.read_bytes()).tofile(path.with_suffix(".dax"))


@click.command()
@click.argument(
    "path",
    type=click.Path(
        exists=True,
        dir_okay=True,
        file_okay=True,
    ),
)
def main(path: Path):
    path = Path(path)
    files = list(path.glob("**/*.jxl")) if path.is_dir() else [path]
    with ProcessPoolExecutor() as pool:
        with tqdm(total=len(files)) as progress, logging_redirect_tqdm():
            futures = []
            for file in files:
                curr_file = file
                future = pool.submit(run, file)
                future.add_done_callback(lambda _: progress.update())
                future.add_done_callback(lambda _: log.info(f"Finished {curr_file}"))
                futures.append(future)
            [future.result() for future in futures]
            log.info("Done")


if __name__ == "__main__":
    main()
