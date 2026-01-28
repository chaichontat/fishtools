import json
import pickle
import re
from collections.abc import Collection
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from threading import Lock
from types import FunctionType
from typing import Protocol

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rich_click as click
import seaborn as sns
from basicpy import BaSiC
from loguru import logger
from tifffile import TiffFile, imread

from fishtools.io.workspace import Workspace, get_channels
from fishtools.preprocess.basic import fit_basic, plot_basic
from fishtools.preprocess.deconv.helpers import scale_deconv
from fishtools.preprocess.plot import plot_tile_sizes
from fishtools.preprocess.tileconfig import tiles_at_least_n_steps_from_edges
from fishtools.utils.logging import setup_cli_logging

sns.set_theme()


class DataExtractor(Protocol):
    def __call__(
        self,
        files: list[Path],
        zs: Collection[float],
        deconv_meta: np.ndarray | None = None,
        max_files: int = 800,
        nc: int | None = None,
        *,
        random_flip: bool = False,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray: ...


# ----------------------
# Sampling helpers
# ----------------------

_ROUND_TOKEN = re.compile(r"^[^\-]+--")


def _parse_tile_index(p: Path) -> int | None:
    stem = p.stem
    parts = stem.split("-")
    for token in reversed(parts):
        if token.isdigit():
            try:
                return int(token)
            except ValueError:
                return None
    return None


def _roi_base_from_dirname(dirname: str) -> str | None:
    # Reuse Workspace's robust pattern to parse ROI from directory names
    m = Workspace.ROI_CODEBOOK_PATTERN.match(dirname)
    return m.group(1) if m else None


def _allowed_indices_for_roi(
    base_path: Path, roi_dirname: str, cache: dict[str, set[int] | None]
) -> set[int] | None:
    roi_base = _roi_base_from_dirname(roi_dirname)
    if roi_base is None:
        return None
    if roi_base in cache:
        return cache[roi_base]

    csv_path = base_path / f"{roi_base}.csv"
    if not csv_path.exists():
        logger.warning(
            f"CSV for ROI '{roi_base}' not found at {csv_path}. Skipping edge-based filtering for this ROI."
        )
        cache[roi_base] = None
        return None

    try:
        df = pd.read_csv(csv_path, header=None)
        xy = df.iloc[:, :2].to_numpy()
        idx = tiles_at_least_n_steps_from_edges(xy, n=2)
        allowed = set(int(i) for i in idx.tolist())
        cache[roi_base] = allowed
        return allowed
    except Exception as e:  # pragma: no cover - defensive path
        logger.warning(f"Failed to compute interior tiles for ROI '{roi_base}': {e}. Not filtering this ROI.")
        cache[roi_base] = None
        return None


def _round_name_from_parent(parent_name: str) -> str:
    return parent_name.split("--", 1)[0] if "--" in parent_name else parent_name


def _is_canonical_round(sample_file: Path) -> bool:
    try:
        ch = get_channels(sample_file)
        return len(ch) == 3 and all(c.isdigit() for c in ch)
    except Exception:
        # Fallback: use round name token pattern like 1_9_17
        round_token = _round_name_from_parent(sample_file.parent.name)
        return bool(re.fullmatch(r"\d+_\d+_\d+", round_token))


def sample_canonical_unique_tiles(
    path: Path, *, include_edge_tiles: bool = False
) -> tuple[list[Path], list[str]]:
    """Return interior tiles from canonical rounds with round-rotating selection.

    Canonical rounds are those whose channel names are numeric and exactly 3 long
    (e.g., 560/650/750). For each ROI, we consider the set of interior tile
    indices (n>=2 from edges). For a given (roi, index), we choose the tile from
    one of the canonical rounds in a round-robin fashion across rounds so that
    returned files are mixed across rounds without any index overlap.

    Returns:
        (files, extra_rounds)
        files: interior tile paths with at most one file per (roi, index),
               chosen by rotating across canonical rounds
        extra_rounds: non-canonical rounds encountered (for separate handling)
    """
    rounds = Workspace.discover_rounds(path)
    # Representative tile per round to determine canonicality
    round_to_sample: dict[str, Path] = {}
    for r in rounds:
        first = next(iter(sorted(path.glob(f"{r}--*/*.tif"))), None)
        if first is not None:
            round_to_sample[r] = first

    canonical_rounds = [r for r, f in sorted(round_to_sample.items()) if _is_canonical_round(f)]
    extra_rounds = sorted(set(round_to_sample.keys()) - set(canonical_rounds))

    # Map: roi_base -> index -> {round: path}
    by_roi_index: dict[str, dict[int, dict[str, Path]]] = {}
    allowed_by_roi: dict[str, set[int] | None] = {}

    for r in canonical_rounds:
        for f in sorted(path.glob(f"{r}--*/*.tif")):
            roi_dirname = f.parent.name
            roi_base = _roi_base_from_dirname(roi_dirname)
            if roi_base is None:
                continue
            allowed = _allowed_indices_for_roi(path, roi_dirname, allowed_by_roi)
            idx = _parse_tile_index(f)
            if idx is None:
                continue
            if not include_edge_tiles and (allowed is None or idx not in allowed):
                continue  # enforce interior only when CSV known
            by_roi_index.setdefault(roi_base, {}).setdefault(idx, {})[r] = f

    # Build rotated selection across rounds per ROI
    selected: list[Path] = []
    for roi_base, idx_map in sorted(by_roi_index.items()):
        if not idx_map:
            continue
        # Maintain a rotation pointer per ROI
        order = list(canonical_rounds)
        ptr = 0
        for idx in sorted(idx_map.keys()):
            tried = 0
            chosen: Path | None = None
            while tried < len(order):
                r = order[ptr]
                ptr = (ptr + 1) % len(order)
                tried += 1
                f = idx_map[idx].get(r)
                if f is not None:
                    chosen = f
                    break
            if chosen is not None:
                selected.append(chosen)

    # Prefer stable ordering and non "--basic" sources
    selected.sort(key=lambda p: ("--basic" in p.as_posix(), p.as_posix()))
    return selected, extra_rounds


def extract_data_from_tiff(
    files: list[Path],
    zs: Collection[float],
    deconv_meta: np.ndarray | None = None,
    max_files: int = 500,
    nc: int | None = None,
    *,
    random_flip: bool = False,
    rng: np.random.Generator | None = None,
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

    with TiffFile(files[0]) as tif0:
        first_page = tif0.pages[0].asarray()
        if first_page.ndim == 3:
            first_page = first_page[0]
        if first_page.ndim != 2:
            raise ValueError(f"Unexpected TIFF page shape {first_page.shape} in {files[0]}.")
        height, width = first_page.shape

    # Initialize output array
    out = np.zeros((n, len(zs), nc, height, width), dtype=np.float32)

    # Extract z slices from all files
    for i, file in enumerate(files[:n]):
        if i % 100 / nc == 0:
            logger.info(f"Loaded {i}/{n}")

        flip_y = False
        flip_x = False
        if random_flip:
            if rng is None:
                rng = np.random.default_rng()
            flip_y = bool(rng.random() < 0.5)
            flip_x = bool(rng.random() < 0.5)

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
                        if img.shape != (height, width):
                            raise ValueError(
                                f"Unexpected tile size {img.shape} in {file}; expected {(height, width)}."
                            )

                        if deconv_meta is not None:
                            img = scale_deconv(
                                img,
                                c,
                                name=file.name,
                                global_deconv_scaling=deconv_meta,
                                metadata=meta,
                            )

                        if flip_y:
                            img = img[::-1, :]
                        if flip_x:
                            img = img[:, ::-1]

                        out[i, k, c] = img

                        if np.sum(out[i, k, c]) == 0:
                            raise ValueError("All zeros.")

                    except Exception as e:
                        raise Exception(f"Error at {file}, {i}, {k}, {c}.") from e

    # Reshape to combine file and z dimensions
    out = np.reshape(out, (n * len(zs), nc, height, width))
    logger.info(f"Loaded {n} files. Output shape: {out.shape}")

    return out


def extract_data_from_tiff_channels(
    files: list[Path],
    zs: Collection[float],
    channel_indices: list[int],
    deconv_meta: np.ndarray | None = None,
    max_files: int = 500,
    nc: int | None = None,
    *,
    flips: list[tuple[bool, bool]] | None = None,
) -> np.ndarray:
    """Extract data from TIFF files, loading only selected channels.

    Returns a float32 array with shape (n_files * len(zs), len(channel_indices), height, width).
    """
    if nc is None:
        raise ValueError("nc must be provided.")
    if not channel_indices:
        raise ValueError("channel_indices must be non-empty.")
    if any((c < 0) or (c >= nc) for c in channel_indices):
        raise ValueError(f"channel_indices out of range for nc={nc}: {channel_indices}")

    n = min(len(files), max_files)
    if flips is not None and len(flips) != n:
        raise ValueError(f"flips must be length n_files={n}, got {len(flips)}")

    n_zs: dict[str, int] = {}  # Track number of z-slices per file parent directory

    with TiffFile(files[0]) as tif0:
        first_page = tif0.pages[0].asarray()
        if first_page.ndim == 3:
            first_page = first_page[0]
        if first_page.ndim != 2:
            raise ValueError(f"Unexpected TIFF page shape {first_page.shape} in {files[0]}.")
        height, width = first_page.shape

    out = np.zeros((n, len(zs), len(channel_indices), height, width), dtype=np.float32)

    for i, file in enumerate(files[:n]):
        if i % 100 / max(1, len(channel_indices)) == 0:
            logger.info(f"Loaded {i}/{n}")

        flip_y = False
        flip_x = False
        if flips is not None:
            flip_y, flip_x = flips[i]

        with TiffFile(file) as tif:
            if file.parent.name not in n_zs:
                n_zs[file.parent.name] = len(tif.pages) // nc
            nz = n_zs[file.parent.name]
            meta = tif.shaped_metadata[0]

            for out_c, c in enumerate(channel_indices):
                for k, z in enumerate(zs):
                    try:
                        img = tif.pages[int(z * nz) * nc + c].asarray()
                        if img.ndim == 3:  # single channel
                            img = img[0]
                        if img.shape != (height, width):
                            raise ValueError(
                                f"Unexpected tile size {img.shape} in {file}; expected {(height, width)}."
                            )

                        if deconv_meta is not None:
                            img = scale_deconv(
                                img,
                                c,
                                name=file.name,
                                global_deconv_scaling=deconv_meta,
                                metadata=meta,
                            )

                        if flip_y:
                            img = img[::-1, :]
                        if flip_x:
                            img = img[:, ::-1]

                        out[i, k, out_c] = img

                        if np.sum(out[i, k, out_c]) == 0:
                            raise ValueError("All zeros.")

                    except Exception as e:
                        raise Exception(f"Error at {file}, {i}, {k}, {c}.") from e

    out = np.reshape(out, (n * len(zs), len(channel_indices), height, width))
    logger.info(f"Loaded {n} files. Output shape: {out.shape}")
    return out


def fit_and_save_basic(
    data: np.ndarray,
    output_dir: Path,
    round_: str,
    channels: list[str],
    plot: bool = True,
    *,
    overwrite: bool = False,
    threads: int = 3,
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
    if threads < 1:
        raise ValueError(f"threads must be >= 1, got {threads}")
    output_dir.mkdir(exist_ok=True)
    lock = Lock()

    def fit_write(idx: int, channel: str) -> BaSiC:
        out_pkl = output_dir / f"{round_}-{channel}.pkl"
        out_png = output_dir / f"{round_}-{channel}.png"

        if not overwrite and out_pkl.exists():
            loaded = pickle.loads(out_pkl.read_bytes())
            basic = loaded.get("basic") if isinstance(loaded, dict) else loaded
            if not isinstance(basic, BaSiC):
                raise TypeError(f"Unexpected BaSiC pickle payload at {out_pkl}: {type(basic)}")

            if not out_png.exists():
                with lock:
                    plot_basic(basic)
                    plt.savefig(out_png)
                    plt.close()
            return basic

        basic = fit_basic(data, idx)
        logger.info(f"Writing {round_}-{channel}.pkl")

        with open(out_pkl, "wb") as f:
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
            plt.savefig(out_png)
            plt.close()

        return basic

    max_workers = min(threads, len(channels))
    with ThreadPoolExecutor(max_workers=max_workers) as exc:
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
    *,
    random_flip: bool = False,
    rng: np.random.Generator | None = None,
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

        flip_y = False
        flip_x = False
        if random_flip:
            if rng is None:
                rng = np.random.default_rng()
            flip_y = bool(rng.random() < 0.5)
            flip_x = bool(rng.random() < 0.5)

        with TiffFile(file) as tif:
            img = tif.asarray().max(axis=0)
            if flip_y:
                img = img[:, ::-1, :]
            if flip_x:
                img = img[:, :, ::-1]
            out[i] = img

    # Reshape to combine file and z dimensions
    logger.info(f"Loaded {n} files. Output shape: {out.shape}")

    return out


def extract_data_from_registered_channels(
    files: list[Path],
    channel_indices: list[int],
    max_files: int = 800,
    *,
    flips: list[tuple[bool, bool]] | None = None,
) -> np.ndarray:
    """Extract data from registered TIFFs, loading only selected channels.

    Computes a max projection over Z and keeps only the requested channels.
    Returns a float32 array with shape (n_files, len(channel_indices), height, width).
    """
    if not channel_indices:
        raise ValueError("channel_indices must be non-empty.")

    n = min(len(files), max_files)
    if flips is not None and len(flips) != n:
        raise ValueError(f"flips must be length n_files={n}, got {len(flips)}")

    with TiffFile(files[0]) as tif0:
        series0 = tif0.series[0]
        axes = series0.axes
        shape = series0.shape
        if len(axes) != len(shape):
            raise ValueError(f"Registered TIFF axes/shape mismatch: axes={axes}, shape={shape} in {files[0]}.")
        if not axes.endswith("YX"):
            raise ValueError(f"Expected axes ending with YX, got axes={axes} in {files[0]}.")
        plane_axes = axes[:-2]
        plane_shape = shape[:-2]
        if "C" not in plane_axes:
            raise ValueError(f"Expected a C axis in axes={axes} for {files[0]}.")
        if any(ax not in {"T", "Z", "C"} for ax in plane_axes):
            raise ValueError(
                f"Unsupported registered TIFF plane axes={plane_axes} (full axes={axes}) in {files[0]}."
            )
        if "Z" not in plane_axes:
            raise ValueError(f"Expected a Z axis in axes={axes} for {files[0]}.")

        sizes = dict(zip(plane_axes, plane_shape))
        nz = int(sizes["Z"])
        nc = int(sizes["C"])
        height, width = shape[-2], shape[-1]

        def plane_index(*, t: int = 0, z: int = 0, c: int = 0) -> int:
            coords = {"T": t, "Z": z, "C": c}
            idx = 0
            stride = 1
            for ax, dim in reversed(list(zip(plane_axes, plane_shape))):
                idx += int(coords[ax]) * stride
                stride *= int(dim)
            return idx

        # Validate early that page indexing is sane for the first file.
        if plane_index(z=0, c=0) < 0:
            raise ValueError(f"Invalid plane indexing for axes={axes}, shape={shape} in {files[0]}.")

        if series0.pages is None:
            raise ValueError(f"Missing pages for series in {files[0]}.")
        pages0 = series0.pages
        if len(pages0) != int(np.prod(plane_shape)):
            raise ValueError(
                f"Unexpected page count for axes={axes}, shape={shape}: pages={len(pages0)} in {files[0]}."
            )

        # Read one page to validate plane size without materializing the full stack.
        first_plane = pages0[plane_index(z=0, c=0)].asarray()
        if first_plane.ndim == 3:
            first_plane = first_plane[0]
        if first_plane.shape != (height, width):
            raise ValueError(
                f"Unexpected tile size {first_plane.shape} in {files[0]}; expected {(height, width)}."
            )

    if any((c < 0) or (c >= nc) for c in channel_indices):
        raise ValueError(f"channel_indices out of range for nc={nc}: {channel_indices}")

    out = np.zeros((n, len(channel_indices), height, width), dtype=np.float32)

    for i, file in enumerate(files[:n]):
        if i % 20 == 0:
            logger.info("Loaded {}/{}", i, n)

        flip_y = False
        flip_x = False
        if flips is not None:
            flip_y, flip_x = flips[i]

        with TiffFile(file) as tif:
            series = tif.series[0]
            if series.axes != axes or series.shape[-2:] != (height, width):
                raise ValueError(
                    f"Registered TIFF layout mismatch across files: expected axes={axes}, yx={(height, width)}, "
                    f"got axes={series.axes}, yx={series.shape[-2:]} in {file}."
                )
            pages = series.pages
            if pages is None:
                raise ValueError(f"Missing pages for series in {file}.")

            for out_c, c in enumerate(channel_indices):
                max_img: np.ndarray | None = None
                for z in range(nz):
                    img = pages[plane_index(z=z, c=c)].asarray()
                    if img.ndim == 3:
                        img = img[0]
                    if img.shape != (height, width):
                        raise ValueError(
                            f"Unexpected tile size {img.shape} in {file}; expected {(height, width)}."
                        )
                    max_img = img if max_img is None else np.maximum(max_img, img)

                if max_img is None:
                    raise ValueError(f"Failed to load any pages for channel {c} in {file}.")

                if flip_y:
                    max_img = max_img[::-1, :]
                if flip_x:
                    max_img = max_img[:, ::-1]

                out[i, out_c] = max_img

    logger.info(f"Loaded {n} files. Output shape: {out.shape}")
    return out


def run_with_extractor(
    path: Path,
    round_: str | None,
    extractor_func: DataExtractor,
    plot: bool = True,
    plot_selected: bool = False,
    zs: Collection[float] = (0.5,),
    *,
    max_n_tiles: int = 1000,
    seed: int | None = None,
    random_flip: bool = False,
    threads: int = 3,
    overwrite: bool = False,
    include_edge_tiles: bool = False,
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
    if max_n_tiles < 0:
        raise ValueError("max_n_tiles must be >= 0 (0 means no limit).")

    # Get rounds/tiles. When round_ is None → sample canonical rounds with dedup & 2-step margin.
    if not round_:
        files, extra_rounds = sample_canonical_unique_tiles(path, include_edge_tiles=include_edge_tiles)
        if extra_rounds:
            logger.info(
                f"Will run {extra_rounds} separately due to an atypical/non-canonical channel layout."
            )
        logger.warning(
            f"No round specified. Sampling from {sorted({p.parent.name.split('--')[0] for p in files})}."
        )
    else:
        files = sorted(path.glob(f"{round_}--*/*.tif"))
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

    rng = random.Random(seed) if seed is not None else random

    # Build allowed index sets per ROI (from <roi>.csv when available)
    allowed_by_roi: dict[str, set[int] | None] = {}

    def _allowed_indices_for_roi_local(roi_dirname: str) -> set[int] | None:
        return _allowed_indices_for_roi(path, roi_dirname, allowed_by_roi)

    # Compute 60th percentile size across all candidate files
    sizes = np.array([f.lstat().st_size for f in files], dtype=np.int64)
    pos = sizes[sizes > 0]
    if pos.size:
        size_thresh = float(np.percentile(pos, 60))
    else:
        # No positive sizes detected; disable size-based selection
        size_thresh = float("inf")

    # Apply filter: interior (n>=2) when ROI CSVs are available.
    filtered_files: list[Path] = []
    if include_edge_tiles:
        filtered_files = list(files)
    else:
        for f in files:
            roi_dirname = f.parent.name
            allowed = _allowed_indices_for_roi_local(roi_dirname)
            idx = _parse_tile_index(f)
            is_interior = allowed is not None and idx is not None and idx in allowed
            # sz = f.lstat().st_size
            # Consider only positive sizes for percentile logic to prevent zero-valued files from
            # collapsing the threshold and letting edge tiles through inadvertently.
            # is_large = (sz > 0) and (sz >= size_thresh)
            if is_interior:
                filtered_files.append(f)

    if not filtered_files:
        roi_labels = sorted(
            {_roi_base_from_dirname(f.parent.name) or f.parent.name for f in files}
        )
        raise ValueError(
            "No interior tiles found for BaSiC sampling. "
            "ROI layout CSV files (e.g. '<roi>.csv') appear to be missing for all ROIs "
            f"({', '.join(roi_labels)}). "
            "Please generate ROI CSVs (e.g. via stitching helpers) before running 'preprocess basic run'."
        )

    k = len(filtered_files) if max_n_tiles == 0 else min(max_n_tiles, len(filtered_files))
    files = rng.sample(filtered_files, k=k)
    # Print sample names
    show_n = min(10, len(files))
    if show_n:
        logger.info("Sample tiles ({} of {}): {}", show_n, len(files), [p.name for p in files[:show_n]])

    # Group-by reporting
    counts: dict[str, int] = {}
    if not round_:
        for p in files:
            key = p.parent.name.split("--", 1)[0]
            counts[key] = counts.get(key, 0) + 1
        logger.info("Sampled counts by round: {}", dict(sorted(counts.items())))
    else:
        for p in files:
            key = _roi_base_from_dirname(p.parent.name) or p.parent.name
            counts[key] = counts.get(key, 0) + 1
        logger.info("Sampled counts by ROI: {}", dict(sorted(counts.items())))
    logger.info(
        f"Using {len(files)} files. Size p60 (nonzero)={size_thresh:.0f}. Min size: {min(f.lstat().st_size for f in files)}. Max size: {max(f.lstat().st_size for f in files)}"
    )

    # Load deconvolution scaling if available
    deconv_path = path / "deconv_scaling" / f"{round_ or 'all'}.txt"
    deconv_meta = np.loadtxt(deconv_path) if deconv_path.exists() else None
    if deconv_meta is not None:
        logger.warning("Using deconvolution scaling. Should not use for normal workflow.")

    # Save sampling manifest to JSON
    try:
        sampled_rel: list[str] = []
        for p in files:
            try:
                sampled_rel.append(str(p.relative_to(path)))
            except ValueError:
                sampled_rel.append(p.as_posix())

        sampling_json = {
            "round": round_ or "all",
            "seed": seed,
            "random_flip": bool(random_flip),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "channels": channels,
            "n_candidates": int(len(filtered_files)),
            "n_selected": int(len(files)),
            "sampled_tiles": sampled_rel,
        }
        if not round_:
            sampling_json["extra_rounds"] = extra_rounds
            sampling_json["counts_by_round"] = dict(sorted(counts.items()))
        else:
            sampling_json["counts_by_roi"] = dict(sorted(counts.items()))

        (basic_dir / f"{round_ or 'all'}-sampling.json").write_text(json.dumps(sampling_json, indent=2))
    except Exception as e:
        logger.warning(f"Failed to write sampling manifest: {e}")

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

    if threads < 1:
        raise ValueError(f"threads must be >= 1, got {threads}")

    round_token = round_ or "all"

    is_builtin_extractor = (
        (extractor_func is extract_data_from_tiff and isinstance(extractor_func, FunctionType))
        or (extractor_func is extract_data_from_registered and isinstance(extractor_func, FunctionType))
    )

    # For non-built-in extractors (e.g. tests, patched callables), keep the original behavior.
    if not is_builtin_extractor:
        flip_rng = np.random.default_rng(seed) if random_flip else None
        data = extractor_func(
            files,
            zs,
            deconv_meta,
            nc=len(channels),
            max_files=len(files),
            random_flip=random_flip,
            rng=flip_rng,
        )
        res = fit_and_save_basic(
            data,
            basic_dir,
            round_token,
            channels,
            plot,
            overwrite=overwrite,
            threads=threads,
        )
    else:
        # Precompute per-file flips so channel-batched loading stays consistent across batches.
        flips: list[tuple[bool, bool]] | None = None
        if random_flip:
            flip_rng = np.random.default_rng(seed)
            flips = [(bool(flip_rng.random() < 0.5), bool(flip_rng.random() < 0.5)) for _ in range(len(files))]

        channels_to_fit: list[str] = []
        for channel in channels:
            out_pkl = basic_dir / f"{round_token}-{channel}.pkl"
            out_png = basic_dir / f"{round_token}-{channel}.png"
            if overwrite or not out_pkl.exists():
                channels_to_fit.append(channel)
            elif out_pkl.exists() and not out_png.exists():
                loaded = pickle.loads(out_pkl.read_bytes())
                basic = loaded.get("basic") if isinstance(loaded, dict) else loaded
                if not isinstance(basic, BaSiC):
                    raise TypeError(f"Unexpected BaSiC pickle payload at {out_pkl}: {type(basic)}")
                plot_basic(basic)
                plt.savefig(out_png)
                plt.close()

        channel_to_index = {ch: i for i, ch in enumerate(channels)}
        for start in range(0, len(channels_to_fit), threads):
            chunk = channels_to_fit[start : start + threads]
            if not chunk:
                continue

            for offset, channel in enumerate(chunk):
                idx = start + offset + 1
                logger.info(f"Loading channel {idx}/{len(channels_to_fit)} to-fit ({channel}); total={len(channels)}")
            channel_indices = [channel_to_index[ch] for ch in chunk]
            if extractor_func is extract_data_from_tiff:
                data = extract_data_from_tiff_channels(
                    files,
                    zs,
                    channel_indices,
                    deconv_meta,
                    nc=len(channels),
                    max_files=len(files),
                    flips=flips,
                )
            else:
                data = extract_data_from_registered_channels(
                    files,
                    channel_indices,
                    max_files=len(files),
                    flips=flips,
                )

            fit_and_save_basic(
                data,
                basic_dir,
                round_token,
                chunk,
                plot,
                overwrite=overwrite,
                threads=min(threads, len(chunk)),
            )
            del data

        # Match historical return value: one BaSiC per channel for this round.
        basics_by_channel: dict[str, BaSiC] = {}
        for channel in channels:
            out_pkl = basic_dir / f"{round_token}-{channel}.pkl"
            if not out_pkl.exists():
                raise FileNotFoundError(f"Expected BaSiC output {out_pkl} was not created.")
            loaded = pickle.loads(out_pkl.read_bytes())
            basic = loaded.get("basic") if isinstance(loaded, dict) else loaded
            if not isinstance(basic, BaSiC):
                raise TypeError(f"Unexpected BaSiC pickle payload at {out_pkl}: {type(basic)}")
            basics_by_channel[channel] = basic

        res = [basics_by_channel[ch] for ch in channels]

    for r in extra_rounds:
        logger.info(f"Running {r}")
        run_with_extractor(
            path,
            r,
            extractor_func,
            plot=plot,
            zs=zs,
            seed=seed,
            random_flip=random_flip,
            threads=threads,
            overwrite=overwrite,
            include_edge_tiles=include_edge_tiles,
        )

    return res


@click.group()
def basic(): ...


@basic.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=Path))
@click.argument("round_", type=str)
@click.option("--zs", type=str, default="0.5")
@click.option("--overwrite", is_flag=True)
@click.option("--random-flip", is_flag=True, help="Randomly flip tiles vertically and/or horizontally.")
@click.option(
    "--include-edge-tiles/--no-include-edge-tiles",
    default=False,
    help="Include edge tiles in BaSiC sampling (does not require ROI CSVs).",
)
@click.option(
    "--max-n-tiles",
    type=click.IntRange(min=0),
    default=1000,
    show_default=True,
    help="Maximum number of tiles to load (0 = no limit).",
)
@click.option(
    "--threads",
    "-t",
    type=click.IntRange(min=1),
    default=3,
    show_default=True,
    help="Number of channels to load/fit in parallel.",
)
@click.option("--seed", type=int, default=None, help="Random seed for sampling (optional)")
def run(
    path: Path,
    round_: str,
    *,
    overwrite: bool = False,
    random_flip: bool = False,
    include_edge_tiles: bool = False,
    max_n_tiles: int = 1000,
    threads: int = 3,
    zs: str = "0.5",
    seed: int | None = None,
):
    # Workspace-scoped logging to {workspace}/analysis/logs; compatible with progress bars
    setup_cli_logging(
        path,
        component="preprocess.basic.run",
        file=f"basic-{round_}",
        extra={"round": round_},
    )
    basic_dir = path / "basic"
    if not overwrite:
        expected_channels: list[str]
        if round_ == "all":
            expected_channels = ["560", "650", "750"]
        else:
            sample = next(iter(sorted(path.glob(f"{round_}--*/*.tif"))), None)
            if sample is None:
                raise click.ClickException(f"No TIFFs found for round '{round_}' under {path}.")
            expected_channels = get_channels(sample)

        expected_pkls = [basic_dir / f"{round_}-{ch}.pkl" for ch in expected_channels]
        if expected_pkls and all(p.exists() for p in expected_pkls):
            logger.info(f"BaSiC already complete for {round_} in {path}. Use --overwrite to re-run.")
            return
    z_values = tuple(map(float, zs.split(",")))
    extractor = extract_data_from_registered if round_ == "registered" else extract_data_from_tiff

    return run_with_extractor(
        path,
        round_ if round_ != "all" else None,
        extractor,
        plot=False,
        zs=z_values,
        max_n_tiles=max_n_tiles,
        seed=seed,
        random_flip=random_flip,
        threads=threads,
        overwrite=overwrite,
        include_edge_tiles=include_edge_tiles,
    )


@basic.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=Path))
@click.option("--overwrite", is_flag=True)
@click.option("--zs", type=str, default="0.5")
@click.option(
    "--max-n-tiles",
    type=click.IntRange(min=0),
    default=1000,
    show_default=True,
    help="Maximum number of tiles to load per round (0 = no limit).",
)
@click.option("--seed", type=int, default=None, help="Random seed for sampling (optional)")
@click.option("--random-flip", is_flag=True, help="Randomly flip tiles vertically and/or horizontally.")
def batch(
    path: Path,
    overwrite: bool = False,
    zs: str = "0.5",
    max_n_tiles: int = 1000,
    seed: int | None = None,
    random_flip: bool = False,
):
    setup_cli_logging(
        path,
        component="preprocess.basic.batch",
        file="basic-batch",
        extra={"overwrite": overwrite},
    )
    rounds = sorted({p.name.split("--")[0] for p in path.glob("*") if p.is_dir() and "--" in p.name})
    for r in rounds:
        run.callback(  # type: ignore
            path,
            r,
            overwrite=overwrite,
            zs=zs,
            max_n_tiles=max_n_tiles,
            seed=seed,
            random_flip=random_flip,
        )

if __name__ == "__main__":
    basic()
