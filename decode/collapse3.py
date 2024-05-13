# %%
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path

import bitmath
import click
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tifffile import TiffFile, imread, imwrite

from fishtools.utils.pretty_print import progress_bar


def run(path: Path):
    with TiffFile(path) as t:
        c = t.asarray()
        if path.name.startswith("dapi_polyA_1_17"):
            fid = c[-1]
            c = c[:-1].reshape(24, 4, 2048, 2048)[:, [0, 2, 3], :, :].reshape(-1, 2048, 2048)
            c = np.concatenate([c, fid[np.newaxis]])
            name = path.name.replace("dapi_polyA_1_17", "dapi_1_17")
        elif path.name.startswith("8_polyA_28"):
            fid = c[-1]
            c = c[:-1].reshape(24, 3, 2048, 2048)[:, [0, 2], :, :].reshape(-1, 2048, 2048)
            c = np.concatenate([c, fid[np.newaxis]])
            name = path.name.replace("8_polyA_28", "8_polyA")

        else:
            name = path.name

        m = t.shaped_metadata
    imwrite(
        path.with_name(name),
        c,
        compression=22610,
        imagej=True,
        compressionargs={"level": 0.65},
        metadata=m[0],
    )
    return path.with_name(name).stat().st_size


@click.command()
@click.argument("input", type=click.Path(exists=True))
def main(input: Path):
    files = list(sorted(Path(input).glob("*.tif")))
    initial_size = 0
    out_size = 0
    with progress_bar(len(files)) as callback, ThreadPoolExecutor(8) as exc:
        futs: list[Future[int]] = []
        for f in files:
            initial_size += f.stat().st_size
            futs.append(fut := exc.submit(run, f))
            fut.add_done_callback(callback)
        for f in as_completed(futs):
            out_size += f.result()

    print(f"Initial size: {bitmath.Byte(initial_size).to_GB()}")
    print(f"Final size: {bitmath.Byte(out_size).to_GB()}")


if __name__ == "__main__":
    main()
