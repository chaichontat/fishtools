"""Blank subtraction utilities and CLI helpers.

This module combines the production bleed-through constant discovery CLI that
previously lived in ``scripts/sub.py`` with the exploratory regression helpers
from ``fishtools/preprocess/subtraction.py``. The CLI exposes Click
commands such as ``preprocess`` and ``fit`` which pool blank/sample image pairs and
fit robust linear models per imaging channel. The regression utilities are
retained so ad-hoc scripts (e.g. ``scripts/subtract_blank.py``) can continue to
inspect the underlying relationships.
"""

from __future__ import annotations

import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
import click
from loguru import logger
from matplotlib.figure import Figure
from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    track,
)
from skimage.filters import gaussian, threshold_otsu
from sklearn.linear_model import RANSACRegressor

from fishtools.io.workspace import get_metadata

# CLI application -----------------------------------------------------------------

app = click.Group(
    name="parallel-bleed-through-analyzer", help="Two-step pipeline for calculating bleed-through parameters."
)

# Shared constants kept in sync with production subtraction logic.
KEYS_560NM = set(range(0, 9)) | {25, 28, 31, 34}
KEYS_650NM = set(range(9, 17)) | {26, 29, 32, 35}
KEYS_750NM = set(range(17, 25)) | {27, 30, 33, 36}
GAUSS_SIGMA = (2, 2, 2)


def get_blank_channel_info(key: str) -> tuple[int, int]:
    """Map an image channel key to its wavelength and blank channel index."""

    key_int = int(key)
    if key_int in KEYS_560NM:
        return 560, 0
    if key_int in KEYS_650NM:
        return 650, 1
    if key_int in KEYS_750NM:
        return 750, 2
    raise ValueError(f"Channel key '{key}' has no defined wavelength mapping.")


def preprocess_channel(channel_data: np.ndarray) -> np.ndarray:
    """Apply Gaussian blur then subtract the minimum to remove background."""

    blurred = gaussian(channel_data, sigma=GAUSS_SIGMA, preserve_range=True)
    subtracted = channel_data - np.minimum(blurred, channel_data)
    return subtracted.astype(np.float32)


def process_single_file(image_path: Path, blank_path: Path, output_dir: Path, z_slice: int) -> str:
    """Worker function that preprocesses one image/blank pair."""

    try:
        shaped_metadata = get_metadata(image_path)
        channel_keys = shaped_metadata["key"]

        sample_img = tifffile.imread(image_path)
        blank_img = tifffile.imread(blank_path)

        for i, key in enumerate(channel_keys):
            _, blank_idx = get_blank_channel_info(key)

            sample_channel = preprocess_channel(sample_img[:, i])
            blank_channel = preprocess_channel(blank_img[:, blank_idx])

            output_npz = output_dir / f"{image_path.stem}_key-{key}.npz"
            np.savez_compressed(
                output_npz,
                image_slice=sample_channel[z_slice],
                blank_slice=blank_channel[z_slice],
            )
        return f"SUCCESS: {image_path.name}"
    except Exception as exc:  # pragma: no cover - defensive logging
        return f"ERROR: {image_path.name} - {exc}"


@app.command("preprocess")
@logger.catch
@click.argument("analysis_path", type=click.Path(path_type=Path, file_okay=False, dir_okay=True, exists=True))
@click.argument("preprocessed_path", type=click.Path(path_type=Path, file_okay=False, dir_okay=True))
@click.option(
    "--data-dir",
    default="registered--hippo+10xhuman",
    show_default=True,
    help="Directory name for sample images.",
)
@click.option(
    "--blank-dir",
    default="registered--hippo+blank",
    show_default=True,
    help="Directory name for blank images.",
)
@click.option(
    "--num-sample-images",
    default=10,
    show_default=True,
    help="Number of images to sample. -1 for all.",
    type=int,
)
@click.option("--z-slice", default=5, show_default=True, help="Z-slice index to use.", type=int)
@click.option("--log-level", default="INFO", show_default=True, help="Logging level.")
def run_preprocess(
    analysis_path: Path,
    preprocessed_path: Path,
    data_dir: str,
    blank_dir: str,
    num_sample_images: int,
    z_slice: int,
    log_level: str,
) -> None:
    """Step 1: preprocess images in parallel and save representative slices."""

    logger.remove()
    logger.add(sys.stderr, level=log_level.upper())

    data_path = analysis_path / data_dir
    blank_path = analysis_path.parent / blank_dir
    preprocessed_path.mkdir(exist_ok=True)

    logger.info(f"Starting parallel preprocessing. Output will be saved to: {preprocessed_path}")

    image_files = sorted(data_path.glob("reg-*.tif"))
    if not image_files:
        logger.critical(f"No images found in {data_path}. Aborting.")
        raise click.ClickException(f"No images found in {data_path}.")

    if num_sample_images > 0 and len(image_files) > num_sample_images:
        sampled_files = random.sample(image_files, num_sample_images)
    else:
        sampled_files = image_files
    logger.info(f"Selected {len(sampled_files)} images to preprocess.")

    tasks = []
    for img_path in sampled_files:
        blk_path = blank_path / img_path.name
        if blk_path.exists():
            tasks.append((img_path, blk_path, preprocessed_path, z_slice))
        else:
            logger.warning(f"Blank for {img_path.name} not found. Skipping.")

    progress_columns = [
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[bold blue]{task.completed} of {task.total}"),
        TimeRemainingColumn(),
        TimeElapsedColumn(),
    ]
    with Progress(*progress_columns, transient=False) as progress:
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(process_single_file, *task) for task in tasks]
            task_id = progress.add_task("Preprocessing Images", total=len(futures))
            for future in as_completed(futures):
                result = future.result()
                if "ERROR" in result:
                    logger.warning(result)
                progress.update(task_id, advance=1)

    logger.success("Preprocessing step complete.")


@app.command("fit")
@logger.catch
@click.argument("analysis_path", type=click.Path(path_type=Path, file_okay=False, dir_okay=True))
@click.argument(
    "preprocessed_path", type=click.Path(path_type=Path, file_okay=False, dir_okay=True, exists=True)
)
@click.option(
    "--output-filename",
    default="robust_bleedthrough_params.csv",
    show_default=True,
    help="Name for the final output CSV file.",
)
@click.option(
    "--fit-threshold", default=120.0, show_default=True, type=float, help="Intensity threshold for fitting."
)
@click.option(
    "--percentile",
    default=20.0,
    show_default=True,
    type=float,
    help="Percentile of signal per bin used for regression.",
)
@click.option(
    "--max-percentile",
    default=99.99,
    show_default=True,
    type=float,
    help="Upper percentile bound for blank bins.",
)
@click.option(
    "--num-bins", default=50, show_default=True, type=int, help="Number of evenly spaced blank bins."
)
@click.option(
    "--min-bin-count",
    default=100,
    show_default=True,
    type=int,
    help="Minimum pixels per bin to keep it in the fit.",
)
@click.option("--log-level", default="INFO", show_default=True, help="Logging level.")
def run_fit(
    analysis_path: Path,
    preprocessed_path: Path,
    output_filename: str,
    fit_threshold: float,
    percentile: float,
    max_percentile: float,
    num_bins: int,
    min_bin_count: int,
    log_level: str,
) -> None:
    """Step 2: fit robust blank-to-signal relationships per channel."""

    logger.remove()
    logger.add(sys.stderr, level=log_level.upper())

    logger.info(f"Starting fitting step from data in: {preprocessed_path}")

    npz_files = list(preprocessed_path.glob("*.npz"))
    if not npz_files:
        logger.critical(
            f"No .npz files found in {preprocessed_path}. Did you run the 'preprocess' step? Aborting."
        )
        raise click.ClickException(f"No .npz files found in {preprocessed_path}.")

    files_by_key: dict[str, list[Path]] = {}
    for file in npz_files:
        key = file.stem.split("_key-")[-1]
        files_by_key.setdefault(key, []).append(file)
    channel_keys = sorted(files_by_key.keys(), key=int)

    all_results = []
    logger.info(f"Found data for {len(channel_keys)} channels. Pooling and fitting...")

    for key in track(channel_keys, description="Fitting Channels..."):
        image_pixels: list[np.ndarray] = []
        blank_pixels: list[np.ndarray] = []
        for npz_path in files_by_key[key]:
            data = np.load(npz_path)
            image_pixels.append(data["image_slice"])
            blank_pixels.append(data["blank_slice"])

        if not image_pixels:
            logger.warning(f"No data loaded for channel key {key}. Skipping.")
            continue

        image_pixels_all = np.concatenate(image_pixels)
        blank_pixels_all = np.concatenate(blank_pixels)

        wavelength, _ = get_blank_channel_info(key)

        mask = blank_pixels_all > fit_threshold
        if np.sum(mask) < 100:
            logger.warning(f"Insufficient data for key {key}. Skipping fit.")
            slope, intercept = None, None
        else:
            try:
                linreg, _, _ = fit_blank_linear(
                    blank_pixels_all[mask],
                    image_pixels_all[mask],
                    percentile=percentile,
                    max_percentile=max_percentile,
                    num_bins=num_bins,
                    min_bin_count=min_bin_count,
                )
            except ValueError as exc:
                logger.warning(f"Channel {key}: {exc}. Skipping fit.")
                slope, intercept = None, None
            else:
                estimator = linreg.estimator_
                slope = float(estimator.coef_[0])
                intercept = float(estimator.intercept_)

        all_results.append({
            "channel_key": key,
            "wavelength_nm": wavelength,
            "slope": slope,
            "intercept": intercept,
            "num_images_sampled": len(files_by_key[key]),
        })

    results_df = pd.DataFrame(all_results)
    output_path = analysis_path / output_filename
    results_df.to_csv(output_path, index=False, float_format="%.6f")

    logger.success(f"Fitting complete. Robust parameters saved to: {output_path}")
    logger.info(f"\nResults Summary:\n{results_df.to_string()}")


# ---------------------------------------------------------------------------
# Registered image workflow


def _load_registered_stack(path: Path) -> tuple[np.ndarray, list[str], dict[str, int], dict]:
    with tifffile.TiffFile(path) as tif:
        data = tif.asarray()
        metadata = tif.shaped_metadata[0] if getattr(tif, "shaped_metadata", None) else {}

    if data.ndim != 4:
        raise ValueError(f"Expected ZCYX registered stack, got shape {data.shape}")

    keys = metadata.get("key")
    if keys is None:
        raise ValueError("Registered image metadata missing 'key' entries for channels")

    names = [str(k) for k in keys]
    index = {name: i for i, name in enumerate(names)}
    if len(index) != len(names):
        raise ValueError("Channel names contain duplicates; cannot disambiguate indices")
    return data, names, index, metadata


def _select_registered_channels(
    stack: np.ndarray,
    channel_index: dict[str, int],
    signal_channel: str,
    blank_channel: str,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        signal_idx = channel_index[signal_channel]
    except KeyError as exc:
        raise ValueError(f"Signal channel '{signal_channel}' not found in metadata") from exc
    try:
        blank_idx = channel_index[blank_channel]
    except KeyError as exc:
        raise ValueError(f"Blank channel '{blank_channel}' not found in metadata") from exc

    signal = stack[:, signal_idx].astype(np.float32, copy=False)
    blank = stack[:, blank_idx].astype(np.float32, copy=False)
    return signal, blank


def _subtract_registered_channels(
    stack: np.ndarray,
    channel_index: dict[str, int],
    signal_channel: str,
    blank_channel: str,
    slope: float,
    intercept: float,
    *,
    clip_min: float = 0.0,
) -> np.ndarray:
    signal_idx = channel_index[signal_channel]
    blank_idx = channel_index[blank_channel]

    signal = stack[:, signal_idx].astype(np.float32)
    blank = stack[:, blank_idx].astype(np.float32)

    corrected = signal - (blank * slope + intercept)
    if clip_min is not None:
        np.maximum(corrected, clip_min, out=corrected)
    if np.issubdtype(stack.dtype, np.integer):
        high = np.iinfo(stack.dtype).max
        np.minimum(corrected, high, out=corrected)

    out_stack = stack.copy()
    out_stack[:, signal_idx] = corrected.astype(stack.dtype, copy=False)
    return out_stack


@app.command("fit-registered")
@logger.catch
@click.argument(
    "registered_path", type=click.Path(path_type=Path, file_okay=True, dir_okay=False, exists=True)
)
@click.argument("output_csv", type=click.Path(path_type=Path, file_okay=True, dir_okay=False))
@click.option("--signal-channel", required=True, help="Channel name to correct.")
@click.option("--blank-channel", required=True, help="Reference blank channel name.")
@click.option(
    "--fit-threshold",
    default=120.0,
    show_default=True,
    type=float,
    help="Ignore blank pixels at or below this intensity.",
)
@click.option(
    "--percentile",
    default=20.0,
    show_default=True,
    type=float,
    help="Percentile of signal per bin used for regression.",
)
@click.option(
    "--max-percentile",
    default=99.99,
    show_default=True,
    type=float,
    help="Upper percentile bound for blank bins.",
)
@click.option(
    "--num-bins", default=50, show_default=True, type=int, help="Number of evenly spaced blank bins."
)
@click.option(
    "--min-bin-count",
    default=100,
    show_default=True,
    type=int,
    help="Minimum pixels per bin to keep it in the fit.",
)
@click.option("--log-level", default="INFO", show_default=True, help="Logging level.")
def run_fit_registered(
    registered_path: Path,
    output_csv: Path,
    signal_channel: str,
    blank_channel: str,
    fit_threshold: float,
    percentile: float,
    max_percentile: float,
    num_bins: int,
    min_bin_count: int,
    log_level: str,
) -> None:
    logger.remove()
    logger.add(sys.stderr, level=log_level.upper())

    stack, names, index, _ = _load_registered_stack(registered_path)
    logger.info("Loaded registered stack %s with channels: %s", registered_path, ", ".join(names))

    signal, blank = _select_registered_channels(stack, index, signal_channel, blank_channel)

    flat_blank = blank.reshape(-1)
    flat_signal = signal.reshape(-1)
    mask = flat_blank > fit_threshold
    if np.sum(mask) < min_bin_count:
        raise click.ClickException(
            f"Not enough pixels above threshold {fit_threshold} to fit regression (found {np.sum(mask)})."
        )

    linreg, bin_centers, percentiles = fit_blank_linear(
        flat_blank[mask],
        flat_signal[mask],
        percentile=percentile,
        max_percentile=max_percentile,
        num_bins=num_bins,
        min_bin_count=min_bin_count,
    )

    slope = float(linreg.estimator_.coef_[0])
    intercept = float(linreg.estimator_.intercept_)
    logger.success(
        "Fitted slope %.6f and intercept %.2f using %d bins (median blank %.2f)",
        slope,
        intercept,
        len(bin_centers),
        float(np.median(flat_blank[mask])),
    )

    results_df = pd.DataFrame([
        dict(
            signal_channel=signal_channel,
            blank_channel=blank_channel,
            slope=slope,
            intercept=intercept,
            num_bins=len(bin_centers),
            percentile=percentile,
            max_percentile=max_percentile,
            fit_threshold=fit_threshold,
        )
    ])
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_csv, index=False, float_format="%.6f")
    logger.info("Saved parameters to %s", output_csv)


@app.command("subtract-registered")
@logger.catch
@click.argument(
    "registered_path", type=click.Path(path_type=Path, file_okay=True, dir_okay=False, exists=True)
)
@click.argument("output_path", type=click.Path(path_type=Path, file_okay=True, dir_okay=False))
@click.option("--signal-channel", required=True, help="Channel to correct.")
@click.option("--blank-channel", required=True, help="Reference blank channel.")
@click.option(
    "--params-csv",
    type=click.Path(path_type=Path, dir_okay=False, file_okay=True),
    help="CSV produced by fit commands containing slope/intercept.",
)
@click.option("--slope", type=float, help="Slope to apply (overrides CSV if provided).")
@click.option("--intercept", type=float, help="Intercept to apply (overrides CSV if provided).")
@click.option(
    "--clip-min",
    default=0.0,
    show_default=True,
    type=float,
    help="Lower bound after subtraction; defaults to 0.",
)
@click.option("--log-level", default="INFO", show_default=True, help="Logging level.")
def run_subtract_registered(
    registered_path: Path,
    output_path: Path,
    signal_channel: str,
    blank_channel: str,
    params_csv: Path | None,
    slope: float | None,
    intercept: float | None,
    clip_min: float,
    log_level: str,
) -> None:
    logger.remove()
    logger.add(sys.stderr, level=log_level.upper())

    if slope is None or intercept is None:
        if params_csv is None:
            raise click.BadParameter("Provide either --params-csv or both --slope and --intercept.")
        df = pd.read_csv(params_csv)
        candidates = df[df.get("signal_channel", df.get("channel_key")) == signal_channel]
        if "blank_channel" in df.columns:
            candidates = candidates[candidates["blank_channel"] == blank_channel]
        if candidates.empty:
            raise ValueError(
                f"Could not find parameters for signal '{signal_channel}' (blank '{blank_channel}') in {params_csv}"
            )
        row = candidates.iloc[0]
        slope = float(row["slope"])
        intercept = float(row["intercept"])
        logger.info(
            "Loaded slope %.6f and intercept %.2f from %s",
            slope,
            intercept,
            params_csv,
        )

    assert slope is not None and intercept is not None  # for mypy

    stack, _, index, metadata = _load_registered_stack(registered_path)
    corrected_stack = _subtract_registered_channels(
        stack,
        index,
        signal_channel,
        blank_channel,
        slope,
        intercept,
        clip_min=clip_min,
    )

    metadata_out = dict(metadata)
    subtract_meta = metadata_out.get("subtract", {})
    subtract_meta.update({
        "signal_channel": signal_channel,
        "blank_channel": blank_channel,
        "slope": slope,
        "intercept": intercept,
    })
    metadata_out["subtract"] = subtract_meta

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(output_path, corrected_stack, metadata=metadata_out)
    logger.success("Wrote corrected stack to %s", output_path)


# Regression helpers -----------------------------------------------------------------


def fit_blank_linear(
    blank_values: np.ndarray,
    signal_values: np.ndarray,
    *,
    percentile: float = 20,
    max_percentile: float = 99.99,
    num_bins: int = 50,
    min_bin_count: int = 100,
) -> tuple[RANSACRegressor, np.ndarray, np.ndarray]:
    """Fit a robust linear model describing blank→signal bleed-through.

    This is the canonical implementation used by both the CLI and exploratory
    notebooks. The procedure mirrors :func:`regress` but accepts flattened
    vectors so large aggregations (thousands of tiles) can be pooled before the
    fit. The heuristics are:

    * Use ``threshold_otsu`` on the blank distribution to seed ``num_bins``
      linearly spaced bins up to ``max_percentile``.
    * Drop bins with fewer than ``min_bin_count`` pixels to avoid unstable
      percentiles.
    * Measure the ``percentile``th signal within each valid bin, emphasising the
      baseline rather than extreme bleed-through spikes.
    * Fit a ``RANSACRegressor`` to the (bin centre, percentile) pairs so obvious
      outlier bins are rejected.
    """

    flat_blank = np.asarray(blank_values).reshape(-1)
    flat_signal = np.asarray(signal_values).reshape(-1)
    if flat_blank.size != flat_signal.size:
        raise ValueError("blank and signal arrays must have the same length")

    bin_centers, percentiles = _compute_binned_percentiles(
        flat_blank,
        flat_signal,
        percentile=percentile,
        max_percentile=max_percentile,
        num_bins=num_bins,
        min_bin_count=min_bin_count,
    )

    linreg = RANSACRegressor()
    linreg.fit(bin_centers.reshape(-1, 1), percentiles)
    return linreg, bin_centers, percentiles


def regress(
    img: np.ndarray,
    blank: np.ndarray,
    *,
    percentile: float = 20,
    max_percentile: float = 99.99,
    num_bins: int = 50,
    min_bin_count: int = 100,
) -> tuple[RANSACRegressor, np.ndarray, np.ndarray]:
    """Estimate a robust blank→signal transfer curve for scientific inspection.

    Delegates to :func:`fit_blank_linear` after flattening the paired image
    volumes. The scientific heuristics are identical to the production CLI:

    * ``threshold_otsu`` sets the lower bound for ``num_bins`` evenly spaced
      bins. This suppresses the dark-noise mode and targets intensities
      dominated by bleed-through.
    * Bins with fewer than ``min_bin_count`` pixels are removed; surviving bins
      contribute their ``percentile``th blank-conditioned signal. The default
      (20th percentile) emphasizes baseline signal rather than hot pixels.
    * ``RANSACRegressor`` fits the ``(bin_center, percentile)`` pairs, rejecting
      outlier bins and yielding a slope/intercept consistent with the CSV stored
      by :func:`run_fit`.
    """

    flat_blank = blank.flatten()
    flat_img = img.flatten()
    return fit_blank_linear(
        flat_blank,
        flat_img,
        percentile=percentile,
        max_percentile=max_percentile,
        num_bins=num_bins,
        min_bin_count=min_bin_count,
    )


def _compute_binned_percentiles(
    flat_blank: np.ndarray,
    flat_signal: np.ndarray,
    *,
    percentile: float,
    max_percentile: float,
    num_bins: int,
    min_bin_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    if num_bins < 2:
        raise ValueError("num_bins must be at least 2")
    if min_bin_count < 1:
        raise ValueError("min_bin_count must be positive")

    bins = np.linspace(
        threshold_otsu(flat_blank),
        np.percentile(flat_blank, max_percentile),
        num_bins,
    )

    digitized = np.digitize(flat_blank, bins)
    bin_counts = np.bincount(digitized)[1:-1]

    valid_bins = bin_counts >= min_bin_count
    if not np.any(valid_bins):
        raise ValueError("No valid bins found")

    binned_data = np.full((valid_bins.sum(), len(flat_blank)), np.nan)
    valid_bin_idx = 0
    for idx, count in enumerate(bin_counts):
        if count >= 100:
            mask = digitized == idx + 1
            binned_data[valid_bin_idx, : len(flat_signal[mask])] = flat_signal[mask]
            valid_bin_idx += 1

    percentiles = np.nanpercentile(binned_data, percentile, axis=1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_centers = bin_centers[valid_bins]
    return bin_centers, percentiles


def plot_regression(
    img: np.ndarray,
    blank: np.ndarray,
    bin_centers: np.ndarray,
    percentiles: np.ndarray,
    slope: float,
    intercept: float,
    *,
    sample_rate: int = 16,
) -> tuple[Figure, plt.Axes]:
    """Plot raw scatter, bin percentiles, and a regression line."""

    fig, ax = plt.subplots()

    ax.scatter(blank.flatten()[::sample_rate], img.flatten()[::sample_rate], alpha=0.01)

    x_line = np.linspace(0, blank.max(), 10)
    y_line = x_line * slope + intercept
    ax.plot(x_line, y_line, "r-", label=f"y = {slope:.2f}x + {intercept:.2f}")

    ax.scatter(bin_centers, percentiles, label="Bin percentiles")

    ax.legend()
    ax.set_xlabel("Blank intensity")
    ax.set_ylabel("Image intensity")

    return fig, ax


if __name__ == "__main__":  # pragma: no cover
    app()
