import json
import pickle
import re
from collections.abc import Collection, Sequence
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

from fishtools import IMWRITE_KWARGS
from fishtools.io.workspace import Workspace, get_channels, safe_imwrite
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


def sample_canonical_unique_tiles(path: Path) -> tuple[list[Path], list[str]]:
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
            if allowed is None or idx is None or idx not in allowed:
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


def _load_basic_profiles_cpu(paths: Sequence[Path]) -> tuple[np.ndarray, np.ndarray]:
    """Load BaSiC darkfield/flatfield profiles from pickle payloads for CPU usage."""
    darks: list[np.ndarray] = []
    flats: list[np.ndarray] = []
    reference_shape: tuple[int, int] | None = None

    for pkl_path in paths:
        loaded = pickle.loads(pkl_path.read_bytes())
        basic = loaded.get("basic") if isinstance(loaded, dict) else loaded
        if basic is None:
            raise ValueError(f"BaSiC payload at {pkl_path} does not contain a 'basic' model.")

        dark = np.asarray(basic.darkfield, dtype=np.float32)
        flat = np.asarray(basic.flatfield, dtype=np.float32)

        if dark.shape != flat.shape:
            raise ValueError(f"Dark/flat shape mismatch for {pkl_path}: {dark.shape} vs {flat.shape}.")
        if reference_shape is None:
            reference_shape = dark.shape
        elif dark.shape != reference_shape:
            raise ValueError("Inconsistent BaSiC profile geometry; all profiles must match.")

        darks.append(dark)
        flats.append(flat)

    if not darks:
        raise ValueError("No BaSiC profiles provided.")

    darkfield_np = np.stack(darks, axis=0).astype(np.float32, copy=False)
    flatfield_np = np.stack(flats, axis=0).astype(np.float32, copy=False)
    return darkfield_np, flatfield_np


def _apply_basic_cpu(payload: np.ndarray, darkfield: np.ndarray, flatfield: np.ndarray) -> np.ndarray:
    """Apply BaSiC correction on CPU and return float32 array with original shape."""
    if payload.ndim < 3:
        raise ValueError("Expected payload with at least 3 dimensions (planes, height, width).")

    channels, height, width = darkfield.shape
    if flatfield.shape != (channels, height, width):
        raise ValueError("Darkfield and flatfield shapes must match.")

    plane_count = int(np.prod(payload.shape[:-2]))
    if plane_count % channels != 0:
        raise ValueError(
            f"Payload plane count {plane_count} not divisible by channel count {channels}. "
            "Verify --channels or --basic-name selection."
        )

    reshaped = payload.reshape(-1, channels, height, width).astype(np.float32, copy=False)

    if not np.all(flatfield > 0):
        logger.warning("Flatfield contains non-positive entries. Clamping to epsilon for stability.")
        flatfield = np.where(flatfield > 0, flatfield, np.float32(1e-6))

    corrected = reshaped - darkfield[np.newaxis, ...]
    corrected /= flatfield[np.newaxis, ...]
    np.clip(corrected, 0.0, np.finfo(np.float32).max, out=corrected)
    return corrected.reshape(payload.shape)


def _infer_channel_names_for_transform(
    *,
    input_path: Path,
    workspace: Path,
    round_name: str,
    basic_name: str | None,
    explicit_channels: str | None,
) -> list[str]:
    if explicit_channels:
        channels = [token.strip() for token in explicit_channels.split(",") if token.strip()]
        if not channels:
            raise ValueError("--channels provided but no valid entries parsed.")
        return channels

    inferred = get_channels(input_path)
    if inferred:
        return inferred

    basic_dir = workspace / "basic"
    target_prefixes: list[str] = []
    if basic_name:
        target_prefixes.append(basic_name)
    else:
        target_prefixes.extend([round_name, "all"])

    for prefix in target_prefixes:
        prefix_str = f"{prefix}-"
        candidates = sorted(p for p in basic_dir.glob(f"{prefix}-*.pkl") if p.is_file())
        channels: list[str] = []
        for candidate in candidates:
            stem = candidate.stem
            if not stem.startswith(prefix_str):
                continue
            channels.append(stem[len(prefix_str) :])
        if channels:
            return channels

    raise ValueError(
        "Unable to infer channel names from TIFF metadata or existing BaSiC profiles. "
        "Provide --channels (comma-separated)."
    )


def _resolve_basic_paths_cpu(
    workspace: Path,
    *,
    round_name: str,
    channels: Sequence[str],
    basic_name: str | None,
) -> list[Path]:
    paths: list[Path] = []
    basic_dir = workspace / "basic"
    for channel in channels:
        primary: Path | None = None
        if basic_name:
            candidate = basic_dir / f"{basic_name}-{channel}.pkl"
            if candidate.exists():
                primary = candidate
            else:
                raise FileNotFoundError(f"BaSiC profile not found: {candidate} (explicit --basic-name).")
        else:
            preferred = basic_dir / f"{round_name}-{channel}.pkl"
            fallback_all = basic_dir / f"all-{channel}.pkl"
            if preferred.exists():
                primary = preferred
            elif fallback_all.exists():
                primary = fallback_all
            else:
                raise FileNotFoundError(
                    "Missing BaSiC profile for channel "
                    f"{channel}: tried {preferred.name} then {fallback_all.name}."
                )
        paths.append(primary)
        logger.debug(f"Using BaSiC profile for channel {channel} at {primary}")
    return paths


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
    *,
    seed: int | None = None,
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

    # Get rounds/tiles. When round_ is None → sample canonical rounds with dedup & 2-step margin.
    if not round_:
        files, extra_rounds = sample_canonical_unique_tiles(path)
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

    # Apply OR filter: interior (n>=2) OR size >= 60th percentile
    filtered_files: list[Path] = []
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

    files = rng.sample(filtered_files, k=min(1000, len(filtered_files)))
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
@click.option("--seed", type=int, default=None, help="Random seed for sampling (optional)")
def run(path: Path, round_: str, *, overwrite: bool = False, zs: str = "0.5", seed: int | None = None):
    # Workspace-scoped logging to {workspace}/analysis/logs; compatible with progress bars
    setup_cli_logging(
        path,
        component="preprocess.basic.run",
        file=f"basic-{round_}",
        extra={"round": round_},
    )
    if not overwrite and list((path / "basic").glob(f"{round_}*")):
        logger.info(f"Basic already run for {round_} in {path}. Use --overwrite to re-run.")
        return
    z_values = tuple(map(float, zs.split(",")))
    extractor = extract_data_from_registered if round_ == "registered" else extract_data_from_tiff

    return run_with_extractor(
        path,
        round_ if round_ != "all" else None,
        extractor,
        plot=False,
        zs=z_values,
        seed=seed,
    )


@basic.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=Path))
@click.option("--overwrite", is_flag=True)
@click.option("--threads", "-t", type=int, default=1)
@click.option("--zs", type=str, default="0.5")
@click.option("--seed", type=int, default=None, help="Random seed for sampling (optional)")
def batch(path: Path, overwrite: bool = False, threads: int = 1, zs: str = "0.5", seed: int | None = None):
    setup_cli_logging(
        path,
        component="preprocess.basic.batch",
        file="basic-batch",
        extra={"threads": threads, "overwrite": overwrite},
    )
    rounds = sorted({p.name.split("--")[0] for p in path.glob("*") if p.is_dir() and "--" in p.name})
    with ThreadPoolExecutor(threads) as exc:
        futs = [
            exc.submit(run.callback, path, r, overwrite=overwrite, zs=zs, seed=seed)
            for r in rounds  # type: ignore
        ]
        for fut in as_completed(futs):
            fut.result()


def _transform_single_tile(
    tile_path: Path,
    *,
    workspace_root: Path,
    base_output_root: Path | None,
    default_root: Path,
    darkfield: np.ndarray,
    flatfield: np.ndarray,
    channel_names: list[str],
    n_fids: int,
    overwrite: bool,
) -> None:
    if base_output_root is not None:
        dest_dir = base_output_root
    else:
        try:
            rel_parent = tile_path.parent.relative_to(workspace_root)
            dest_dir = default_root / rel_parent
        except ValueError:
            dest_dir = default_root

    dest_dir.mkdir(parents=True, exist_ok=True)
    output_path = dest_dir / tile_path.name

    if output_path.exists() and not overwrite:
        logger.info(f"{output_path} exists; skipping. Use --overwrite to replace.")
        return

    with TiffFile(tile_path) as tif:
        try:
            metadata = tif.shaped_metadata[0]  # type: ignore[index]
        except (TypeError, IndexError):
            metadata = tif.imagej_metadata or {}
        if not isinstance(metadata, dict):
            metadata = dict(metadata) if metadata is not None else {}
        image = tif.asarray()

    if image.ndim < 3:
        raise click.ClickException(
            f"Unsupported TIFF shape {image.shape}; expected at least (channels, H, W)."
        )

    if n_fids:
        if image.ndim != 3:
            raise click.ClickException("--n-fids currently supports 3D stacks with channel axis first.")
        if image.shape[0] <= n_fids:
            raise click.ClickException(
                f"{tile_path.name} has {image.shape[0]} plane(s); cannot remove {n_fids} fid slice(s)."
            )
        fid = image[-n_fids:]
        payload = image[:-n_fids]
    else:
        fid = None
        payload = image

    corrected = _apply_basic_cpu(payload, darkfield, flatfield)
    corrected_u16 = np.clip(np.rint(corrected), 0, np.iinfo(np.uint16).max).astype(np.uint16, copy=False)

    if fid is not None:
        deliverable = np.concatenate([corrected_u16, fid.astype(np.uint16, copy=False)], axis=0)
    else:
        deliverable = corrected_u16.astype(np.uint16, copy=False)

    metadata_out = dict(metadata)
    metadata_out["basic_corrected"] = True
    metadata_out["basic_channels"] = channel_names

    safe_imwrite(
        output_path,
        deliverable,
        metadata=metadata_out,
        **IMWRITE_KWARGS,
    )
    logger.info(f"Wrote BaSiC-corrected TIFF to {output_path}")


@basic.command()
@click.argument("workspace", type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=Path))
@click.argument("roi", type=str)
@click.argument("idx", type=int, required=False)
@click.option("--round", "round_name", required=True, help="Round identifier used to locate the input tiles.")
@click.option("--output-dir", type=click.Path(dir_okay=True, file_okay=False, path_type=Path), default=None)
@click.option(
    "--basic-name",
    type=str,
    default=None,
    help="Override BaSiC profile prefix (default: round, fallback to 'all').",
)
@click.option(
    "--channels", type=str, default=None, help="Comma-separated channel names if metadata lookup fails."
)
@click.option(
    "--n-fids", type=int, default=0, show_default=True, help="Number of fiducial slices appended in the TIFF."
)
@click.option("--overwrite", is_flag=True, help="Overwrite existing outputs.")
def transform(
    workspace: Path,
    roi: str,
    idx: int | None,
    *,
    round_name: str,
    output_dir: Path | None,
    basic_name: str | None,
    channels: str | None,
    n_fids: int,
    overwrite: bool,
) -> None:
    """Apply BaSiC correction to one or more TIFF tiles identified by ROI/index."""
    setup_cli_logging(
        workspace,
        component="preprocess.basic.transform",
        file=f"basic-transform-{round_name}",
        extra={"round": round_name, "roi": roi, "idx": idx},
    )

    workspace_path = workspace.resolve()
    try:
        ws = Workspace(workspace_path)
        workspace_root = ws.path
        try:
            roi = ws.resolve_rois([roi])[0]
        except ValueError as exc:
            raise click.ClickException(str(exc)) from exc
    except Exception:
        ws = None
        workspace_root = workspace_path

    round_dir = workspace_root / f"{round_name}--{roi}"
    if not round_dir.exists():
        raise click.ClickException(f"Round directory not found: {round_dir}")

    if idx is not None:
        idx_token = f"{idx:04d}"
        candidate = round_dir / f"{round_name}-{idx_token}.tif"
        if not candidate.exists():
            raise click.ClickException(f"Tile not found: {candidate}")
        tile_paths = [candidate]
    else:
        tile_paths = sorted(
            p for p in round_dir.glob(f"{round_name}-*.tif") if _parse_tile_index(p) is not None
        )
        if not tile_paths:
            raise click.ClickException(f"No tiles found under {round_dir} matching {round_name}-*.tif.")

    try:
        channel_names = _infer_channel_names_for_transform(
            input_path=tile_paths[0],
            workspace=workspace_root,
            round_name=round_name,
            basic_name=basic_name,
            explicit_channels=channels,
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    basic_paths = _resolve_basic_paths_cpu(
        workspace_root,
        round_name=round_name,
        channels=channel_names,
        basic_name=basic_name,
    )

    darkfield_np, flatfield_np = _load_basic_profiles_cpu(basic_paths)

    default_root = workspace_root / "analysis" / "basic_transform"
    if output_dir is not None:
        base_output_root = (
            output_dir if output_dir.is_absolute() else (workspace_root / output_dir)
        ).resolve()
    else:
        base_output_root = None

    processed = 0
    for tile_path in tile_paths:
        try:
            _transform_single_tile(
                tile_path,
                workspace_root=workspace_root,
                base_output_root=base_output_root,
                default_root=default_root,
                darkfield=darkfield_np,
                flatfield=flatfield_np,
                channel_names=channel_names,
                n_fids=n_fids,
                overwrite=overwrite,
            )
        except click.ClickException:
            raise
        except Exception as exc:  # pragma: no cover - defensive
            raise click.ClickException(f"Failed to transform {tile_path.name}: {exc}") from exc
        processed += 1

    logger.info(f"Finished BaSiC correction for {processed} tile(s).")


if __name__ == "__main__":
    basic()
