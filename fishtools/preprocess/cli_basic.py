import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Collection, List, Optional, Protocol

import matplotlib.pyplot as plt
import numpy as np
import rich_click as click
import seaborn as sns
from basicpy import BaSiC
from loguru import logger
from tifffile import TiffFile, imread

from fishtools.preprocess.basic import fit_basic, plot_basic
from fishtools.preprocess.cli_deconv import scale_deconv

sns.set_theme()


def extract_data_from_tiff(
    files: List[Path],
    zs: Collection[float],
    deconv_meta: Optional[np.ndarray] = None,
    max_files: int = 500,
    nc: int | None = None,
) -> np.ndarray:
    """
    Extract data from TIFF files with multiple channels per file.

    Args:
        files: List of TIFF files to process
        nc: Number of channels
        zs: Collection of z-slice positions (between 0 and 1)
        deconv_meta: Optional deconvolution scaling metadata
        max_files: Maximum number of files to process

    Returns:
        Numpy array with shape (n_files * len(zs), nc, height, width)
    """
    n = min(len(files), max_files)
    n_zs = {}  # Track number of z-slices per file parent directory

    if nc is None:
        raise ValueError("nc must be provided.")

    # Initialize output array
    out = np.zeros((n, len(zs), nc, 2048, 2048), dtype=np.float32)

    # Extract z slices from all files
    for i, file in enumerate(files[:n]):
        if i % 100 / nc == 0:
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

                        if deconv_meta is not None:
                            img = scale_deconv(
                                img,
                                c,
                                name=file.name,
                                global_deconv_scaling=deconv_meta,
                                metadata=meta,
                            )

                        out[i, k, c] = img

                        if np.sum(out[i, k, c]) == 0:
                            raise ValueError("All zeros.")

                    except Exception as e:
                        raise Exception(f"Error at {file},{i}, {k}, {c}.") from e

    # Reshape to combine file and z dimensions
    out = np.reshape(out, (n * len(zs), nc, 2048, 2048))
    logger.info(f"Loaded {n} files. Output shape: {out.shape}")

    return out


def fit_and_save_basic(data: np.ndarray, output_dir: Path, round_: str, plot: bool = True) -> List[BaSiC]:
    """
    Fit BaSiC models for each channel and save the results.

    Args:
        data: Input data with shape (n_samples, n_channels, height, width)
        output_dir: Directory to save the results
        round_: Round identifier
        plot: Whether to display plots

    Returns:
        List of fitted BaSiC models
    """
    output_dir.mkdir(exist_ok=True)

    def fit_write(c: int) -> BaSiC:
        basic = fit_basic(data, c)
        logger.info(f"Writing {round_}-{c}.pkl")

        with open(output_dir / f"{round_}-{c}.pkl", "wb") as f:
            pickle.dump(basic, f)

        plot_basic(basic)
        plt.savefig(output_dir / f"{round_}-{c}.png")
        plt.close()

        return basic

    with ThreadPoolExecutor(3) as exc:
        futs = [exc.submit(fit_write, c) for c in range(data.shape[1])]
        basics: List[BaSiC] = []

        for fut in futs:
            basics.append(fut.result())
            if plot:
                plot_basic(basics[-1])
                plt.show()

        return basics


# The original _run function has been removed in favor of the more generic run_with_extractor function


class DataExtractor(Protocol):
    def __call__(
        self,
        files: List[Path],
        zs: Collection[float],
        deconv_meta: Optional[np.ndarray] = None,
        max_files: int = 500,
        nc: int | None = None,
    ) -> np.ndarray: ...


# Alternative data extractor for single-file format
def extract_data_from_registered(
    files: List[Path],
    zs: Collection[float],
    deconv_meta: Optional[np.ndarray] = None,
    max_files: int = 500,
    nc: int | None = None,
    max_proj: bool = True,
) -> np.ndarray:
    """
    Extract data from TIFF files with all channels in one file.

    Args:
        files: List of TIFF files to process
        nc: Number of channels
        zs: Collection of z-slice positions (between 0 and 1)
        deconv_meta: Optional deconvolution scaling metadata
        max_files: Maximum number of files to process

    Returns:
        Numpy array with shape (n_files * len(zs), nc, height, width)
    """
    n = min(len(files), max_files)
    example = imread(files[0])

    if nc is not None:
        logger.warning("nc is not None. Ignoring for registered files.")

    # Initialize output array
    nc = example.shape[1]
    out = np.zeros((n, *example.shape[1:]), dtype=np.float32)

    # Extract z slices from all files
    for i, file in enumerate(files[:n]):
        if i % 20 == 0:
            logger.info("Loaded {}/{}", i, n)

        with TiffFile(file) as tif:
            img = tif.asarray().max(axis=0)
            out[i] = img

    # Reshape to combine file and z dimensions
    logger.info(f"Loaded {n} files. Output shape: {out.shape}")

    return out


def run_with_extractor(
    path: Path,
    round_: str,
    extractor_func: DataExtractor,
    plot: bool = True,
    zs: Collection[float] = (0.5,),
):
    """
    Generic function to run processing with a specific data extractor.

    Args:
        path: Path to the directory containing the data
        round_: Round identifier
        extractor_func: Function to extract data from files
        plot: Whether to display plots
        zs: Z-slice positions (between 0 and 1)
    """
    if "deconv" in path.resolve().as_posix() and round_ != "registered":
        input("deconv in path. This script runs prior to deconvolution. Press Enter to continue.")

    basic_dir = path / "basic"
    basic_dir.mkdir(exist_ok=True)

    logger.info(f"Running {round_}")

    # Validate inputs
    if not all(0 <= z <= 1 for z in zs):
        raise ValueError("zs must be between 0 and 1.")

    # Determine number of channels from round string
    nc = len(round_.split("_"))

    # Find and sort files
    files = list(path.glob(f"{round_}--*/*.tif"))
    # Put --basic files last. These are extra files taken in random places for BaSiC training.
    files.sort(key=lambda x: ("--basic" in str(x), str(x)))

    # Validate file count
    if len(files) < 100:
        raise ValueError(f"Not enough files ({len(files)}). Need at least 100 to be somewhat reliable.")

    if len(files) < 500:
        logger.warning(f"Not enough files ({len(files)}). Result may be unreliable.")

    # Load deconvolution scaling if available
    deconv_path = path / "deconv_scaling" / f"{round_}.txt"
    deconv_meta = np.loadtxt(deconv_path) if deconv_path.exists() else None
    if deconv_meta is not None:
        logger.info("Using deconvolution scaling.")

    # Extract data using the provided extractor function
    data = extractor_func(files, zs, deconv_meta, nc=nc)

    # Fit and save BaSiC models
    return fit_and_save_basic(data, basic_dir, round_, plot)


@click.group()
def basic(): ...


@basic.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=Path))
@click.argument("round_", type=str)
@click.option("--zs", type=str, default="0.")
@click.option("--overwrite", is_flag=True)
def run(path: Path, round_: str, *, overwrite: bool = False, zs: str = "0."):
    paths_pickle = [path / "basic" / f"{round_}-{c}.pkl" for c in range(len(round_.split("_")))]
    if all(p.exists() for p in paths_pickle) and not overwrite:
        logger.info(f"Skipping {round_}. Already exists.")
        return

    z_values = tuple(map(float, zs.split(",")))
    extractor = extract_data_from_registered if round_ == "registered" else extract_data_from_tiff

    return run_with_extractor(path, round_, extractor, plot=False, zs=z_values)


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
