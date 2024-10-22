# %%
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import jax
import matplotlib.pyplot as plt
import numpy as np
import rich_click as click
import seaborn as sns
from basicpy import BaSiC
from loguru import logger
from tifffile import TiffFile

from fishtools.preprocess.deconv import scale_deconv

sns.set_theme()


def plot_basic(basic: BaSiC):
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    im = axes[0].imshow(basic.flatfield)
    fig.colorbar(im, ax=axes[0])
    axes[0].set_title("Flatfield")
    im = axes[1].imshow(basic.darkfield)
    fig.colorbar(im, ax=axes[1])
    axes[1].set_title("Darkfield")
    axes[2].plot(basic.baseline)
    axes[2].set_xlabel("Frame")
    axes[2].set_ylabel("Baseline")
    axes[0].axis("off")
    axes[1].axis("off")
    fig.tight_layout()


channels = [560, 650, 750]


def _run(path: Path, round_: str, plot: bool = True):
    # TODO: Get channel info.
    # if not round_.split("_").__len__() == 3:
    #     raise ValueError("Round must be in the format of 1_9_17")

    files = list((path).glob(f"{round_}*/*.tif"))

    if len(files) < 500:
        logger.warning(f"Not enough files ({len(files)}). Result may be unreliable.")

    n = min(len(files), 500)
    zs = [10, 20]
    c = len(round_.split("_"))
    deconv_meta = np.loadtxt(path / "deconv_scaling" / f"{round_}.txt")

    # This is the extracted z slices from all files.
    out = np.zeros((n * len(zs), c, 2048, 2048), dtype=np.float32)
    for i, file in enumerate(files[:n]):
        if i % 100 / len(channels) == 0:
            logger.info("Loaded {}/{}", i, n)
        with TiffFile(file) as tif:
            meta = tif.shaped_metadata[0]
            for j in range(c):
                for k, z in enumerate(zs):
                    img = tif.pages[z * c + j].asarray()
                    out[i * len(zs) + k, j] = scale_deconv(
                        img, j, name=file.name, global_deconv_scaling=deconv_meta, metadata=meta
                    )

    if np.any(np.sum(out, axis=(1, 2, 3)) == 0):
        logger.error("Some imgs are all zeros.")

    logger.info(f"Loaded {len(files)} files. {out.shape}")

    def run_basic(c: int):
        with jax.default_device(jax.devices("cpu")[0]):
            logger.info("Fitting channel {}", c)
            basic = BaSiC()
            basic.fit(out[:, c])
            return basic

    with ThreadPoolExecutor() as exc:
        futs = [exc.submit(run_basic, c) for c in range(out.shape[1])]
        basics: list[BaSiC] = []
        for fut in futs:
            basics.append(fut.result())
            if plot:
                plot_basic(basics[-1])
                plt.show()

        return [fut.result() for fut in futs]


@click.group()
def cli(): ...


@cli.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=Path))
@click.argument("round_", type=str)
@click.option("--overwrite", is_flag=True)
def run(path: Path, round_: str, overwrite: bool = False):
    (path / "basic").mkdir(exist_ok=True)
    path_pickle = path / "basic" / f"{round_}.pkl"
    if path_pickle.exists() and not overwrite:
        logger.info(f"Skipping {round_}. Already exists.")
        return

    logger.info(f"Running {round_}")
    basics = _run(path, round_)
    with open(path / "basic" / f"{round_}.pkl", "wb") as f:
        pickle.dump(dict(zip(channels, basics)), f)


@cli.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=Path))
@click.option("--overwrite", is_flag=True)
@click.option("--threads", "-t", type=int, default=2)
def batch(path: Path, overwrite: bool = False, threads: int = 2):
    rounds = sorted((path / "deconv_scaling").glob("*.txt"))
    with ThreadPoolExecutor(threads) as exc:
        futs = [exc.submit(run.callback, path, r.stem, overwrite=overwrite) for r in rounds]
        for fut in as_completed(futs):
            fut.result()


if __name__ == "__main__":
    cli()
# %%

# %%
# %%
