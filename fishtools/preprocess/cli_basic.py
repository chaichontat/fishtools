import pickle
from collections.abc import Collection
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Protocol

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rich_click as click
import seaborn as sns
from basicpy import BaSiC
from loguru import logger
from tifffile import TiffFile, imread

from fishtools.io.workspace import get_channels
from fishtools.preprocess.basic import fit_basic, plot_basic
from fishtools.preprocess.cli_deconv import scale_deconv
from fishtools.preprocess.plot import plot_tile_sizes
from fishtools.preprocess.tileconfig import tiles_at_least_n_steps_from_edges

sns.set_theme()


class DataExtractor(Protocol):
    def __call__(
        self,
        files: list[Path],
        zs: Collection[float],
        deconv_meta: np.ndarray | None = None,
        max_files: int = 800,
        nc: int | None = None,
    ) -> np.ndarray: ...


def extract_data_from_tiff(
    files: list[Path],
    zs: Collection[float],
    deconv_meta: np.ndarray | None = None,
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
            logger.info(f"Loaded {i}/{n}")

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
                        raise Exception(f"Error at {file}, {i}, {k}, {c}.") from e

    # Reshape to combine file and z dimensions
    out = np.reshape(out, (n * len(zs), nc, 2048, 2048))
    logger.info(f"Loaded {n} files. Output shape: {out.shape}")

    return out


def fit_and_save_basic(
    data: np.ndarray,
    output_dir: Path,
    round_: str,
    channels: list[str],
    plot: bool = True,
) -> list[BaSiC]:
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
    lock = Lock()

    def fit_write(idx: int, channel: str) -> BaSiC:
        basic = fit_basic(data, idx)
        logger.info(f"Writing {round_}-{channel}.pkl")

        with open(output_dir / f"{round_}-{channel}.pkl", "wb") as f:
            pickle.dump(
                dict(
                    basic=basic,
                    path=output_dir.resolve().as_posix(),
                    name=round_,
                    created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                ),
                f,
            )

        with lock:
            plot_basic(basic)
            plt.savefig(output_dir / f"{round_}-{channel}.png")
            plt.close()

        return basic

    with ThreadPoolExecutor(3) as exc:
        futs = [exc.submit(fit_write, i, channel) for i, channel in enumerate(channels)]
        basics: list[BaSiC] = []

        for fut in futs:
            basics.append(fut.result())
            if plot:
                plot_basic(basics[-1])
                plt.show()

        return basics


# Alternative data extractor for single-file format
def extract_data_from_registered(
    files: list[Path],
    zs: Collection[float],
    deconv_meta: np.ndarray | None = None,
    max_files: int = 800,
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
    round_: str | None,
    extractor_func: DataExtractor,
    plot: bool = True,
    plot_selected: bool = False,
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

    # Get rounds with 3 channels bits
    files = sorted(path.glob(f"{round_}--*/*.tif" if round_ else "*_*_*--*/*.tif"))

    if not round_:
        import re

        regex = re.compile(r"\d+_\d+_\d+")
        files = [f for f in files if regex.match(f.stem.split("--")[0])]

        already = set(p.parent.name.split("--")[0] for p in files)
        logger.warning(f"No round specified. Sampling from {sorted(already)}.")
        extra_rounds = sorted(
            s
            for s in set(p.name.split("--")[0] for p in path.glob("*--*")) - already
            if not all(c.isdigit() for c in s.split("_")) and len(list(path.glob(f"{s}--*/*.tif")))
        )
        if extra_rounds:
            logger.info(f"Will run {extra_rounds} separately due to an atypical channel layout.")
    else:
        extra_rounds = []

    # Put --basic files last. These are extra files taken in random places for BaSiC training.
    files.sort(key=lambda x: ("--basic" in str(x), str(x)))

    channels = get_channels(files[0])
    if not round_ and channels != ["560", "650", "750"]:
        raise ValueError(f"Expected channels 560, 650, 750 when running all. Found {channels} at {files[0]}")

    # Validate file count
    if len(files) < 100:
        raise ValueError(
            f"Not enough files ({len(files)}) for {path}. Need at least 100 to be somewhat reliable."
        )

    if len(files) < 500:
        logger.warning(f"Not enough files ({len(files)}) for {path}. Result may be unreliable.")
    import random

    # Filter sampling to tiles that are at least 2 steps from edges (per ROI)
    def _parse_tile_index(p: Path) -> int | None:
        stem = p.stem
        # Prefer last hyphen-delimited token if numeric, else whole stem if numeric
        parts = stem.split("-")
        for token in reversed(parts):
            if token.isdigit():
                try:
                    return int(token)
                except ValueError:
                    return None
        # Fallback: no numeric token
        return None

    def _roi_base_from_dirname(dirname: str) -> str | None:
        # Expect pattern: "<round>--<roi>[--suffix]" → extract <roi> (strip any extra suffix)
        if "--" not in dirname:
            return None
        roi_part = dirname.split("--", 1)[1]
        # If extra suffixes present (e.g., "roi--basic"), strip them for CSV lookup
        return roi_part.split("--", 1)[0]

    # Build allowed index sets per ROI (from <roi>.csv when available)
    allowed_by_roi: dict[str, set[int]] = {}

    def _allowed_indices_for_roi(roi_dirname: str) -> set[int] | None:
        roi_base = _roi_base_from_dirname(roi_dirname)
        if roi_base is None:
            return None
        if roi_base in allowed_by_roi:
            return allowed_by_roi[roi_base]

        csv_path = path / f"{roi_base}.csv"
        if not csv_path.exists():
            logger.warning(
                f"CSV for ROI '{roi_base}' not found at {csv_path}. Skipping edge-based filtering for this ROI."
            )
            allowed_by_roi[roi_base] = None  # type: ignore[assignment]
            return None

        try:
            df = pd.read_csv(csv_path, header=None)
            # Use first two columns as x,y
            xy = df.iloc[:, :2].to_numpy()
            idx = tiles_at_least_n_steps_from_edges(xy, n=2)
            allowed = set(int(i) for i in idx.tolist())
            allowed_by_roi[roi_base] = allowed
            return allowed
        except Exception as e:
            logger.warning(
                f"Failed to compute interior tiles for ROI '{roi_base}': {e}. Not filtering this ROI."
            )
            allowed_by_roi[roi_base] = None  # type: ignore[assignment]
            return None

    # Compute 60th percentile size across all candidate files
    sizes = np.array([f.lstat().st_size for f in files], dtype=np.int64)
    pos = sizes[sizes > 0]
    if pos.size:
        size_thresh = float(np.percentile(pos, 60))
    else:
        # No positive sizes detected; disable size-based selection
        size_thresh = float("inf")

    # Apply OR filter: interior (n>=2) OR size >= 60th percentile
    filtered_files: list[Path] = []
    for f in files:
        roi_dirname = f.parent.name
        allowed = _allowed_indices_for_roi(roi_dirname)
        idx = _parse_tile_index(f)
        is_interior = allowed is not None and idx is not None and idx in allowed
        sz = f.lstat().st_size
        # Consider only positive sizes for percentile logic to prevent zero-valued files from
        # collapsing the threshold and letting edge tiles through inadvertently.
        # is_large = (sz > 0) and (sz >= size_thresh)
        if is_interior:
            filtered_files.append(f)

    files = random.sample(filtered_files, k=min(1000, len(filtered_files)))
    logger.info(
        f"Using {len(files)} files. Size p60 (nonzero)={size_thresh:.0f}. Min size: {min(f.lstat().st_size for f in files)}. Max size: {max(f.lstat().st_size for f in files)}"
    )

    # Load deconvolution scaling if available
    deconv_path = path / "deconv_scaling" / f"{round_ or 'all'}.txt"
    deconv_meta = np.loadtxt(deconv_path) if deconv_path.exists() else None
    if deconv_meta is not None:
        logger.warning("Using deconvolution scaling. Should not use for normal workflow.")

    # Diagnostics plot: per-ROI tile grid colored by size with selected samples marked
    try:
        by_roi: dict[str, list[Path]] = {}
        for f in files:
            by_roi.setdefault(f.parent.name, []).append(f)

        for roi_dir, roi_files in by_roi.items():
            roi_base = _roi_base_from_dirname(roi_dir)
            if roi_base is None:
                continue
            csv_path = path / f"{roi_base}.csv"
            if not csv_path.exists():
                continue
            # Build sizes per CSV index, and selected index set for overlay
            df = pd.read_csv(csv_path, header=None)
            xy = df.iloc[:, :2].to_numpy()
            sizes_all = np.zeros(len(df), dtype=np.float64)

            # Gather all candidate files in this ROI to populate size map
            roi_all_files = sorted((path / f"{(round_ or 'all')}--{roi_base}").glob("*.tif"))
            # Fallback: if constructed folder name doesn't exist (legacy), use parent of selected files
            if not roi_all_files:
                roi_all_files = sorted((Path(roi_files[0]).parent).glob("*.tif"))

            for p in roi_all_files:
                idx = _parse_tile_index(p)
                if idx is not None and 0 <= idx < len(sizes_all):
                    try:
                        sizes_all[idx] = p.lstat().st_size
                    except FileNotFoundError:
                        sizes_all[idx] = 0

            selected_idx = set()
            for p in roi_files:
                idx = _parse_tile_index(p)
                if idx is not None and 0 <= idx < len(df):
                    selected_idx.add(idx)

            if plot_selected:
                ax = plot_tile_sizes(
                    xy,
                    sizes_all,
                    selected=selected_idx,
                    title=f"{round_ or 'all'} — {roi_base} (selected={len(selected_idx)})",
                    show_colorbar=True,
                    annotate="selected",
                    fontsize=6,
                )
                (basic_dir / f"{(round_ or 'all')}--{roi_base}-sampling.png").parent.mkdir(exist_ok=True)
                plt.savefig(
                    basic_dir / f"{(round_ or 'all')}--{roi_base}-sampling.png", dpi=150, bbox_inches="tight"
                )
                plt.close(ax.figure)
    except Exception as e:
        logger.warning(f"Sampling plot generation failed: {e}")

    # Extract data using the provided extractor function
    data = extractor_func(files, zs, deconv_meta, nc=len(channels), max_files=1000 if round_ else 1000)

    # Fit and save BaSiC models
    res = fit_and_save_basic(data, basic_dir, round_ or "all", channels, plot)

    for r in extra_rounds:
        logger.info(f"Running {r}")
        run_with_extractor(path, r, extractor_func, plot=plot, zs=zs)

    return res


@click.group()
def basic(): ...


@basic.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=Path))
@click.argument("round_", type=str)
@click.option("--zs", type=str, default="0.5")
@click.option("--overwrite", is_flag=True)
def run(path: Path, round_: str, *, overwrite: bool = False, zs: str = "0.5"):
    if not overwrite and list((path / "basic").glob(f"{round_}*")):
        logger.info(f"Basic already run for {round_} in {path}. Use --overwrite to re-run.")
        return
    z_values = tuple(map(float, zs.split(",")))
    extractor = extract_data_from_registered if round_ == "registered" else extract_data_from_tiff

    return run_with_extractor(path, round_ if round_ != "all" else None, extractor, plot=False, zs=z_values)


@basic.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=Path))
@click.option("--overwrite", is_flag=True)
@click.option("--threads", "-t", type=int, default=1)
@click.option("--zs", type=str, default="0.5")
def batch(path: Path, overwrite: bool = False, threads: int = 1, zs: str = "0.5"):
    rounds = sorted({p.name.split("--")[0] for p in path.glob("*") if p.is_dir() and "--" in p.name})
    with ThreadPoolExecutor(threads) as exc:
        futs = [exc.submit(run.callback, path, r, overwrite=overwrite, zs=zs) for r in rounds]  # type: ignore
        for fut in as_completed(futs):
            fut.result()


if __name__ == "__main__":
    basic()
