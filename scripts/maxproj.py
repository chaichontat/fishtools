import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import rich
import rich_click as click
import tifffile

from fishtools.utils.pretty_print import progress_bar


def run(p: Path, n_channels: int = 3, *, start: int = 0, end: int | None = None):
    img = tifffile.imread(p)[:-1]  # type: ignore
    nz, rem = divmod(img.shape[0], n_channels)
    if not rem == 0:
        raise ValueError(f"Number of slices is not divisible by {n_channels}")

    img = img.reshape(n_channels, nz, *img.shape[1:])
    end = end or nz

    try:
        tifffile.imwrite(
            p.with_stem(p.stem + "_max"),
            img[:, start:end].max(axis=1),
            compression=22610,
            metadata={"axes": "CYX", "mode": "composite"},
            imagej=True,
        )
    except:
        p.with_stem(p.stem + "_max").unlink(missing_ok=True)


console = rich.get_console()


@click.command()
@click.argument("files", type=click.Path(exists=True, dir_okay=False, file_okay=True), nargs=-1)
@click.option("--n_channels", "-c", default=3)
@click.option("--start", "-s", default=0)
@click.option("--end", "-e", default=None)
@click.option("--threads", "-t", default=8)
@click.option("--overwrite", is_flag=True)
def main(
    files: list[Path] | list[str],
    glob: str = "*.tif",
    n_channels: int = 1,
    start: int = 0,
    end: int = 0,
    threads: int = 8,
    overwrite: bool = False,
):
    console.print("\n✨ Max projector 3000 ✨", style="bold magenta")

    files = [Path(f) for f in files]
    if any(f.suffix != ".tif" for f in files):
        files = [f for f in files if f.suffix == ".tif" and not f.stem.endswith("_max")]

    if not overwrite:
        before = len(files)
        files = [f for f in files if not (f.with_stem(f.stem + "_max")).exists()]
        if before != len(files):
            rich.print(
                f"Removed [bold]{before - len(files)}[/bold] maxed files. To overwrite, use `--overwrite`."
            )

    total = len(files)
    with progress_bar(total) as callback, ThreadPoolExecutor(threads) as exc:
        futs = [exc.submit(run, p, n_channels, start=start, end=end) for p in files]
        for future in futs:
            future.add_done_callback(callback)
        for future in as_completed(futs):
            future.result()  # To get exceptions

        time.sleep(0.5)
    rich.print("\n🎉 Done!")


if __name__ == "__main__":
    main()
