import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import tifffile
import typer
from loguru import logger
from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    track,
)
from skimage.filters import gaussian
from sklearn.linear_model import HuberRegressor

from fishtools.utils.io import get_metadata

# Define the main CLI app
app = typer.Typer(
    name="Parallel Bleed-through Analyzer",
    help="A high-performance, two-step pipeline for calculating bleed-through parameters.",
    add_completion=False,
)

# --- Shared Constants and Functions ---
KEYS_560NM = set(range(0, 9)) | {25, 28, 31, 34}
KEYS_650NM = set(range(9, 17)) | {26, 29, 32, 35}
KEYS_750NM = set(range(17, 25)) | {27, 30, 33, 36}
GAUSS_SIGMA = (2, 2, 2)


def get_blank_channel_info(key: str) -> tuple[int, int]:
    """Maps an image channel key to its corresponding wavelength and blank channel index."""
    key_int = int(key)
    if key_int in KEYS_560NM:
        return 560, 0
    if key_int in KEYS_650NM:
        return 650, 1
    if key_int in KEYS_750NM:
        return 750, 2
    raise ValueError(f"Channel key '{key}' has no defined wavelength mapping.")


def preprocess_channel(channel_data: np.ndarray) -> np.ndarray:
    """Applies Gaussian blur and minimum subtraction for background removal."""
    blurred = gaussian(channel_data, sigma=GAUSS_SIGMA, preserve_range=True)
    subtracted = channel_data - np.minimum(blurred, channel_data)
    return subtracted.astype(np.float32)


def process_single_file(image_path: Path, blank_path: Path, output_dir: Path, z_slice: int) -> str:
    """Worker function to be run in parallel. Processes one image/blank pair."""
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
    except Exception as e:
        return f"ERROR: {image_path.name} - {e}"


# --- CLI Commands ---


@app.command("preprocess")
@logger.catch
def run_preprocess(
    analysis_path: Annotated[Path, typer.Argument(help="Path to the analysis project directory.")],
    preprocessed_path: Annotated[Path, typer.Argument(help="Directory to save intermediate .npz files.")],
    data_dir: Annotated[
        str, typer.Option(help="Directory name for sample images.")
    ] = "registered--hippo+10xhuman",
    blank_dir: Annotated[
        str, typer.Option(help="Directory name for blank images.")
    ] = "registered--hippo+blank",
    num_sample_images: Annotated[int, typer.Option(help="Number of images to sample. -1 for all.")] = 10,
    z_slice: Annotated[int, typer.Option(help="Z-slice index to use.")] = 5,
    log_level: Annotated[str, typer.Option(help="Logging level.")] = "INFO",
):
    """Step 1: Preprocess all images in parallel and save Z-slices to disk."""
    logger.remove()
    logger.add(sys.stderr, level=log_level.upper())

    data_path = analysis_path / data_dir
    blank_path = analysis_path.parent / blank_dir
    preprocessed_path.mkdir(exist_ok=True)

    logger.info(f"Starting parallel preprocessing. Output will be saved to: {preprocessed_path}")

    image_files = sorted(list(data_path.glob("reg-*.tif")))
    if not image_files:
        logger.critical(f"No images found in {data_path}. Aborting.")
        raise typer.Exit(code=1)

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
def run_fit(
    analysis_path: Annotated[Path, typer.Argument(help="Path to the analysis project directory.")],
    preprocessed_path: Annotated[Path, typer.Argument(help="Directory containing intermediate .npz files.")],
    output_filename: Annotated[
        str, typer.Option(help="Name for the final output CSV file.")
    ] = "robust_bleedthrough_params.csv",
    fit_threshold: Annotated[float, typer.Option(help="Intensity threshold for fitting.")] = 120.0,
    log_level: Annotated[str, typer.Option(help="Logging level.")] = "INFO",
):
    """Step 2: Fit models using the preprocessed data."""
    logger.remove()
    logger.add(sys.stderr, level=log_level.upper())

    logger.info(f"Starting fitting step from data in: {preprocessed_path}")

    npz_files = list(preprocessed_path.glob("*.npz"))
    if not npz_files:
        logger.critical(
            f"No .npz files found in {preprocessed_path}. Did you run the 'preprocess' step? Aborting."
        )
        raise typer.Exit(code=1)

    files_by_key = {}
    for f in npz_files:
        key = f.stem.split("_key-")[-1]
        files_by_key.setdefault(key, []).append(f)
    channel_keys = sorted(files_by_key.keys(), key=int)

    all_results = []
    logger.info(f"Found data for {len(channel_keys)} channels. Pooling and fitting...")

    for key in track(channel_keys, description="Fitting Channels..."):
        image_pixels, blank_pixels = [], []
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

        linreg = HuberRegressor(epsilon=1.1)
        mask = blank_pixels_all > fit_threshold
        if np.sum(mask) < 100:
            logger.warning(f"Insufficient data for key {key}. Skipping fit.")
            slope, intercept = None, None
        else:
            linreg.fit(blank_pixels_all[mask].reshape(-1, 1), image_pixels_all[mask])
            slope, intercept = float(linreg.coef_[0]), float(linreg.intercept_)

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


if __name__ == "__main__":
    app()
