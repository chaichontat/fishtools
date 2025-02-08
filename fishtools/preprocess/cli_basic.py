import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Collection

import matplotlib.pyplot as plt
import numpy as np
import rich_click as click
import seaborn as sns
from basicpy import BaSiC
from loguru import logger
from tifffile import TiffFile

from fishtools.preprocess.basic import fit_basic, plot_basic
from fishtools.preprocess.cli_deconv import scale_deconv

sns.set_theme()


def _run(path: Path, round_: str, *, plot: bool = True, zs: Collection[float] = (0.5,)):
    if "deconv" in path.resolve().as_posix():
        raise ValueError("deconv in path. This script runs prior to deconvolution.")

    (path / "basic").mkdir(exist_ok=True)

    logger.info(f"Running {round_}")

    # TODO: Get channel info.
    nc = round_.split("_").__len__()
    if not all(0 <= z <= 1 for z in zs):
        raise ValueError("zs must be between 0 and 1.")

    if nc == 4:
        channels = [488, 560, 650, 750]
    else:
        channels = [560, 650, 750]

    files = list((path).glob(f"{round_}--*/*.tif"))
    # Put --basic files last
    files.sort(key=lambda x: ("--basic" in str(x), str(x)))

    if len(files) < 100:
        raise ValueError(f"Not enough files ({len(files)}). Need at least 100 to be somewhat reliable.")

    if len(files) < 500:
        logger.warning(f"Not enough files ({len(files)}). Result may be unreliable.")

    n = min(len(files), 500)
    nc = len(round_.split("_"))

    deconv_path = path / "deconv_scaling" / f"{round_}.txt"
    if deconv_path.exists():
        deconv_meta = np.loadtxt(deconv_path)
        logger.info("Using deconvolution scaling.")
    else:
        deconv_meta = None

    n_zs = {}

    # This is the extracted z slices from all files.
    out = np.zeros((n, len(zs), nc, 2048, 2048), dtype=np.float32)
    for i, file in enumerate(files[:n]):
        if i % 100 / len(channels) == 0:
            logger.info("Loaded {}/{}", i, n)
        with TiffFile(file) as tif:
            if file.parent.name not in n_zs:
                n_zs[file.parent.name] = len(tif.pages) // nc
            nz = n_zs[file.parent.name]
            meta = tif.shaped_metadata[0]
            for c in range(nc):
                for k, z in enumerate(zs):
                    try:
                        img = tif.pages[int(z * nz) * nc + c].asarray()
                        if img.ndim == 3:  # single channel
                            img = img[0]
                        out[i, k, c] = (
                            img
                            if deconv_meta is None
                            else scale_deconv(
                                img, c, name=file.name, global_deconv_scaling=deconv_meta, metadata=meta
                            )
                        )

                        if np.sum(out[i, k, c]) == 0:
                            raise ValueError("All zeros.")

                    except Exception as e:
                        raise Exception(f"Error at {file},{i}, {k}, {c}.") from e

    out = np.reshape(out, (n * len(zs), nc, 2048, 2048))

    logger.info(f"Loaded {len(files)} files. {out.shape}")

    def fit_write(c: int):
        basic = fit_basic(out, c)
        logger.info(f"Writing {round_}-{c}.pkl")
        with open(path / "basic" / f"{round_}-{c}.pkl", "wb") as f:
            pickle.dump(basic, f)
            plot_basic(basic)
            plt.savefig(path / "basic" / f"{round_}-{c}.png")
            plt.close()
        return basic

    with ThreadPoolExecutor(3) as exc:
        futs = [exc.submit(fit_write, c) for c in range(out.shape[1])]
        basics: list[BaSiC] = []
        for fut in futs:
            basics.append(fut.result())
            if plot:
                plot_basic(basics[-1])
                plt.show()

        return [fut.result() for fut in futs]


@click.group()
def basic(): ...


@basic.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=Path))
@click.argument("round_", type=str)
@click.option("--zs", type=str, default="0.")
@click.option("--overwrite", is_flag=True)
def run(path: Path, round_: str, *, overwrite: bool = False, zs: str = "0."):
    paths_pickle = [path / "basic" / f"{round_}-{c}.pkl" for c in range(round_.split("_").__len__())]
    if all(p.exists() for p in paths_pickle) and not overwrite:
        logger.info(f"Skipping {round_}. Already exists.")
        return

    return _run(path, round_, plot=False, zs=tuple(map(float, zs.split(","))))


@basic.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=Path))
@click.option("--overwrite", is_flag=True)
@click.option("--threads", "-t", type=int, default=1)
@click.option("--zs", type=str, default="0.")
def batch(path: Path, overwrite: bool = False, threads: int = 1, zs: str = "0."):
    rounds = sorted({p.name.split("--")[0] for p in path.glob("*") if p.is_dir() and "--" in p.name})
    with ThreadPoolExecutor(threads) as exc:
        futs = [exc.submit(run.callback, path, r, overwrite=overwrite, zs=zs) for r in rounds]  # type: ignore
        for fut in as_completed(futs):
            fut.result()


if __name__ == "__main__":
    basic()
