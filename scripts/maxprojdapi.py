import signal
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import rich
import rich_click as click
import tifffile

from fishtools.utils.pretty_print import progress_bar


class CustomThreadPoolExecutor(ThreadPoolExecutor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.should_stop = threading.Event()

    def submit(self, fn, *args, **kwargs):
        def wrapped_fn(*args, **kwargs):
            try:
                if self.should_stop.is_set():
                    raise Exception("Execution stopped due to an error in another thread")
                return fn(*args, **kwargs)
            except Exception as e:
                self.should_stop.set()
                raise e

        return super().submit(wrapped_fn, *args, **kwargs)


def run(p: Path, *, start: int = 0, end: int | None = None, downsample: int = 1):
    img = tifffile.imread(p)[:, -3:, ::downsample, ::downsample]  # type: ignore
    out = p.parent / "dapi3"
    maxed = np.max(img, axis=0)
    # try:
    for i in range(maxed.shape[0]):
        (out / str(i)).mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(
            out / str(i) / (p.stem.split("-")[1] + ".tif"),
            maxed[[i]],
            compression=22610,
            metadata={"axes": "CYX", "mode": "composite"},
            compressionargs={"level": 0.8},
        )
    # except:
    # p.with_stem(p.stem + "_max").unlink(missing_ok=True)


console = rich.get_console()


@click.command()
@click.argument("folder", type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=Path))
@click.option("--downsample", "-d", default=1)
# @click.option("--threads", "-t", default=8)
# @click.option("--overwrite", is_flag=True)
def main(
    folder: Path,
    glob: str = "*.tif",
    n_channels: int = 1,
    downsample: int = 1,
    start: int = 0,
    end: int = 0,
    threads: int = 8,
    overwrite: bool = False,
):
    files = sorted(folder.glob(glob))

    if not overwrite:
        before = len(files)
        files = [f for f in files if not (f.parent / "dapi" / f.name).exists()]
        if before != len(files):
            rich.print(
                f"Removed [bold]{before - len(files)}[/bold] maxed files. To overwrite, use `--overwrite`."
            )

    total = len(files)
    with progress_bar(total) as callback, CustomThreadPoolExecutor(threads) as exc:
        futs = [exc.submit(run, p, start=start, end=end, downsample=downsample) for p in files]
        for future in futs:
            future.add_done_callback(callback)
        for future in as_completed(futs):
            try:
                future.result()  # To get exceptions
            except Exception as e:
                console.print(f"[red]Error in thread: {e}[/red]")
                exc.shutdown(wait=False)
                break

        time.sleep(0.5)
    rich.print("\n🎉 Done!")


if __name__ == "__main__":
    main()
