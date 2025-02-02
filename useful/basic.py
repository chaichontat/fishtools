# %%
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
from fishtools.preprocess.deconv import scale_deconv

sns.set_theme()


def _run(path: Path, round_: str, *, plot: bool = True, zs: Collection[float] = (0.5,)):
    # if "deconv" in path.resolve().as_posix() and not deconv:
    #     raise ValueError("deconv in path. Make sure to set deconv=True")

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
def cli(): ...


@cli.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=Path))
@click.argument("round_", type=str)
@click.option("--zs", type=str, default="0.5")
@click.option("--overwrite", is_flag=True)
def run(path: Path, round_: str, *, overwrite: bool = False, zs: str = "0.5"):
    paths_pickle = [path / "basic" / f"{round_}-{c}.pkl" for c in range(round_.split("_").__len__())]
    if all(p.exists() for p in paths_pickle) and not overwrite:
        logger.info(f"Skipping {round_}. Already exists.")
        return

    return _run(path, round_, plot=False, zs=tuple(map(float, zs.split(","))))


@cli.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=Path))
@click.option("--overwrite", is_flag=True)
@click.option("--threads", "-t", type=int, default=1)
def batch(path: Path, overwrite: bool = False, threads: int = 1):
    rounds = sorted({p.name.split("--")[0] for p in path.glob("*") if p.is_dir() and "--" in p.name})
    with ThreadPoolExecutor(threads) as exc:
        futs = [exc.submit(run.callback, path, r, overwrite=overwrite) for r in rounds]  # type: ignore
        for fut in as_completed(futs):
            fut.result()


if __name__ == "__main__":
    cli()
# %%

# %%
# # %%
# path = Path("/mnt/working/e155trcdeconv")
# files = list((path).glob(f"registered--/*.tif"))

# if len(files) < 500:
#     logger.warning(f"Not enough files ({len(files)}). Result may be unreliable.")

# n = min(len(files), 500)
# zs = [10, 40]
# c = 3
# deconv_meta = np.loadtxt(path / "deconv_scaling" / f"1_9_17.txt")

# # This is the extracted z slices from all files.
# out = np.zeros((n, len(zs), c, 2048, 2048), dtype=np.float32)
# for i, file in enumerate(files[:n]):
#     if i % 100 / len(channels) == 0:
#         logger.info("Loaded {}/{}", i, n)
#     with TiffFile(file) as tif:
#         meta = tif.shaped_metadata[0]
#         for j in range(c):
#             for k, z in enumerate(zs):
#                 img = tif.pages[z * c + j].asarray()
#                 out[i, k, j] = scale_deconv(
#                     img, j, name=file.name, global_deconv_scaling=deconv_meta, metadata=meta
#                 )
# out = np.reshape(out, (n * len(zs), c, 2048, 2048))

# if np.any(np.sum(out, axis=(1, 2, 3)) == 0):
#     logger.error("Some imgs are all zeros.")

# logger.info(f"Loaded {len(files)} files. {out.shape}")


# def run_basic(c: int):
#     with jax.default_device(jax.devices("cpu")[0]):
#         logger.info("Fitting channel {}", c)
#         basic = BaSiC()
#         basic.fit(out[:, c])
#         return basic


# with ThreadPoolExecutor() as exc:
#     futs = [exc.submit(run_basic, c) for c in range(3)]
#     basics: list[BaSiC] = []
#     for fut in futs:
#         basics.append(fut.result())

#         plot_basic(basics[-1])
#         plt.show()


# # %%
# u = basics[0].transform(np.array(out[::2, 0]))
# # %%
# with jax.default_device(jax.devices("cpu")[0]):
#     logger.info("Fitting channel {}", c)
#     basic = BaSiC()
#     basic.fit(u)

# # %%
# plot_basic(basic)
# # %%

# # %%
# p = []
# with TiffFile(file) as tif:
#     for k, z in enumerate(range(0, 50, 10)):
#         img = tif.pages[z * c + 0].asarray()
#         p.append(scale_deconv(img, 0, name=file.name, global_deconv_scaling=deconv_meta, metadata=meta))

# # %%

# fig, axs = plt.subplots(ncols=2, nrows=3, figsize=(12, 8), dpi=200, facecolor="white")
# axs = axs.flatten()
# for i, ax in enumerate(axs):
#     ax.imshow(p[i], zorder=1, vmax=3000)
# %%
# from tifffile import imread, imwrite

# path = Path("/mnt/working/e155trcdeconv")
# files = list((path).glob(f"registered--left/*.tif"))

# if len(files) < 500:
#     logger.warning(f"Not enough files ({len(files)}). Result may be unreliable.")

# n = min(len(files), 500)
# zs = [0, 1]
# c = imread(files[0]).shape[1]

# # This is the extracted z slices from all files.
# out = np.zeros((n, len(zs), c, 1988, 1988), dtype=np.float32)
# for i, file in enumerate(files[:n]):
#     if i % 100 / len(channels) == 0:
#         logger.info("Loaded {}/{}", i, n)
#     with TiffFile(file) as tif:
#         img = tif.asarray()
#         for j in range(c):
#             for k, z in enumerate(zs):
#                 out[i, k, j] = img[k, j]

# out = np.reshape(out, (n * len(zs), c, 1988, 1988))

# if np.any(np.sum(out, axis=(1, 2, 3)) == 0):
#     logger.error("Some imgs are all zeros.")

# logger.info(f"Loaded {len(files)} files. {out.shape}")


# def run_basic(c: int):
#     with jax.default_device(jax.devices("cpu")[0]):
#         logger.info("Fitting channel {}", c)
#         basic = BaSiC()
#         basic.fit(out[:, c])
#         return basic


# # %%

# with ThreadPoolExecutor(4) as exc:
#     futs = [exc.submit(run_basic, c) for c in range(imread(files[0]).shape[1])]
#     basics: list[BaSiC] = []
#     for fut in futs:
#         basics.append(fut.result())
#         plot_basic(basics[-1])
#         plt.show()


# # %%
# u = basics[0].transform(np.array(out[::2, 0]))
# # %%
# with jax.default_device(jax.devices("cpu")[0]):
#     logger.info("Fitting channel {}", c)
#     basic = BaSiC()
#     basic.fit(u)

# # %%
# plot_basic(basic)
# # %%

# # %%
# p = []
# with TiffFile(file) as tif:
#     for k, z in enumerate(range(0, 50, 10)):
#         img = tif.pages[z * c + 0].asarray()
#         p.append(scale_deconv(img, 0, name=file.name, global_deconv_scaling=deconv_meta, metadata=meta))

# # %%

# fig, axs = plt.subplots(ncols=2, nrows=3, figsize=(12, 8), dpi=200, facecolor="white")
# axs = axs.flatten()
# for i, ax in enumerate(axs):
#     ax.imshow(p[i], zorder=1, vmax=3000)
# # %%
