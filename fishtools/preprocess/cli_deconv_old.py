import json
import pickle
import queue
import sys
import threading
import time
from collections.abc import Iterable
from itertools import chain
from pathlib import Path
from typing import Any

import cupy as cp
import numpy as np
import numpy.typing as npt
import pyfiglet
import rich_click as click
import tifffile
from basicpy import BaSiC
from loguru import logger
from rich.console import Console
from rich.logging import RichHandler

from fishtools.utils.io import Workspace, get_channels, get_metadata
from fishtools.utils.pretty_print import progress_bar
from fishtools.preprocess.deconv.range_utils import (
    DEFAULT_PERCENTILE_OVERRIDES as RANGE_DEFAULT_PERCENTILE_OVERRIDES,
    compute_range_for_round,
    get_percentiles_for_round,
)
from fishtools.preprocess.deconv.core import (
    load_projectors_cached,
    deconvolve_lucyrichardson_guo,
    rescale as core_rescale,
)


def scale_deconv(
    img: np.ndarray,
    idx: int,
    *,
    global_deconv_scaling: np.ndarray,
    metadata: dict[str, Any],
    name: str | None = None,
    debug: bool = False,
):
    """Scale back deconvolved image using global scaling.

    Args:
        img: Original image, e.g. 1_9_17
        idx: Channel index in original image e.g. 0, 1, 2, ...
    """
    global_deconv_scaling = global_deconv_scaling.reshape((2, -1))
    m_ = global_deconv_scaling[0, idx]
    s_ = global_deconv_scaling[1, idx]

    # Same as:
    # scaled = s_ * ((img / self.metadata["deconv_scale"][idx] + self.metadata["deconv_min"][idx]) - m_)
    scale_factor = np.float32(s_ / metadata["deconv_scale"][idx])
    offset = np.float32(s_ * (metadata["deconv_min"][idx] - m_))
    scaled = scale_factor * img.astype(np.float32) + offset

    if debug:
        logger.debug(f"Deconvolution scaling: {scale_factor}. Offset: {offset}")
    if name and scaled.max() > 65535:
        logger.debug(f"Scaled image {name} has max > 65535.")

    if np.all(scaled < 0):
        logger.warning(f"Scaled image {name} has all negative values.")

    return np.clip(scaled, 0, 65534)


def _compute_range(
    path: Path,
    round_: str,
    *,
    perc_min: float | list[float] = 0.1,
    perc_scale: float | list[float] = 0.1,
):
    """Backward-compatible wrapper around range_utils.compute_range_for_round."""
    compute_range_for_round(
        path,
        round_,
        perc_min=perc_min,
        perc_scale=perc_scale,
    )


logger.remove()
logger.configure(handlers=[{"sink": RichHandler(), "format": "{message}", "level": "INFO"}])
console = Console()


def high_pass_filter(
    img: npt.NDArray[Any],
    σ: tuple[float, ...] = (3.0, 3.0, 3.0),
    dtype: npt.DTypeLike = np.float32,
) -> npt.NDArray:
    """
    Args:
        image: the input image to be filtered
        windowSize: the size of the Gaussian kernel to use.
        sigma: the sigma of the Gaussian.

    Returns:
        the high pass filtered image. The returned image is the same type
        as the input image.
    """
    from cupyx.scipy.ndimage import gaussian_filter

    img = cp.asarray(img)
    lowpass = cp.zeros_like(img)
    for c in range(img.shape[1]):
        lowpass[:, c] = gaussian_filter(img[:, c], sigma=σ, cval=0, truncate=4.0)

    # img = img.astype(dtype)
    # window_size = int(2 * np.ceil(2 * σ) + 1)
    # lowpass = cv2.GaussianBlur(img, (window_size, window_size), σ, borderType=cv2.BORDER_REPLICATE)
    img -= lowpass
    del lowpass
    img = cp.clip(img, 0, None)

    return img


projectors = load_projectors_cached
rescale = core_rescale



def _run(
    paths: list[Path],
    out: Path,
    basics: dict[str, list[BaSiC]],
    overwrite: bool,
    n_fids: int,
    step: int = 6,
    debug: bool = False,
):
    if not overwrite:
        paths = [f for f in paths if not (out / f.parent.name / f.name).exists()]

    q_write = queue.Queue(maxsize=3)
    q_img: queue.Queue[tuple[Path, np.ndarray, Iterable[np.ndarray], dict[str, Any]]] = queue.Queue(maxsize=1)
    profiles = []

    def f_read(files: list[Path]):
        for file in files:
            t0_file = time.perf_counter()
            logger.debug(f"[{file.name}] START Read/Pre-process")
            round_ = file.name.split("-")[0]
            bits = round_.split("_")
            if file.name.startswith("fid"):
                continue

            try:
                t_read_start = time.perf_counter()
                with tifffile.TiffFile(file) as tif:
                    try:
                        metadata = tif.shaped_metadata[0]  # type: ignore
                    except (TypeError, IndexError):
                        metadata = tif.imagej_metadata or {}
                    img = tif.asarray()
                t_read_end = time.perf_counter()
                logger.debug(f"[{file.name}] Disk read took: {t_read_end - t_read_start:.4f}s")

                fid = np.atleast_3d(img[-n_fids:])
                nofid = img[:-n_fids].reshape(-1, len(bits), 2048, 2048).astype(cp.float32)
                print(nofid.shape)
                nofid = cp.asarray(nofid)
            except ValueError as e:
                raise Exception(f"File {file.resolve()} is corrupted. Please check the file.") from e

            t_basic_start = time.perf_counter()
            if not profiles:
                darkfield = cp.asarray(
                    np.stack([basic.darkfield for basic in basics[round_]], axis=0)[np.newaxis, ...]
                )
                flatfield = cp.asarray(
                    np.stack([basic.flatfield for basic in basics[round_]], axis=0)[np.newaxis, ...]
                )
                profiles.append((darkfield, flatfield))

            nofid -= profiles[0][0]
            nofid /= profiles[0][1]
            cp.clip(nofid, 0, None, out=nofid)

            t_basic_end = time.perf_counter()
            logger.debug(f"[{file.name}] BaSiC correction took: {t_basic_end - t_basic_start:.4f}s")

            total_preprocess_time = time.perf_counter() - t0_file
            logger.debug(f"[{file.name}] Total preprocess time: {total_preprocess_time:.4f}s")

            t_put_start = time.perf_counter()
            q_img.put((file, nofid, fid, metadata))
            t_put_end = time.perf_counter()
            put_wait_time = t_put_end - t_put_start
            if put_wait_time > 0.01:
                logger.warning(
                    f"[{file.name}] Waited {put_wait_time:.4f}s to put in q_img (GPU stage is likely the bottleneck)"
                )
        q_img.put((Path("STOP"), np.array([]), np.array([]), {}))  # Sentinel value

    def f_write():
        while True:
            t_get_write_start = time.perf_counter()
            gotten = q_write.get()
            t_get_write_end = time.perf_counter()

            if gotten is None:
                break

            file, towrite, fid, metadata = gotten
            if file.name == "STOP":
                break

            get_wait_time = t_get_write_end - t_get_write_start
            if get_wait_time > 0.01:
                logger.debug(f"[{file.name}] Waited {get_wait_time:.4f}s to get from q_write")

            logger.debug(f"[{file.name}] START Write")
            t_write_start = time.perf_counter()
            sub = out / file.parent.name
            sub.mkdir(exist_ok=True, parents=True)

            (sub / file.name).with_suffix(".deconv.json").write_text(json.dumps(metadata, indent=2))

            tifffile.imwrite(
                sub / file.name,
                np.concatenate([towrite, fid], axis=0),
                compression=22610,
                compressionargs={"level": 0.75},
                metadata=metadata,
            )
            t_write_end = time.perf_counter()
            logger.debug(f"[{file.name}] Disk write took: {t_write_end - t_write_start:.4f}s")
            q_write.task_done()
        logger.debug("Write thread finished.")

    try:
        threading.current_thread().name = "MainGPUThread"
        thread = threading.Thread(target=f_read, args=(paths,), daemon=True, name="ReadThread")
        thread.start()

        thread_write = threading.Thread(target=f_write, args=(), daemon=True, name="WriteThread")
        thread_write.start()

        with progress_bar(len(paths)) as callback:
            while True:
                t_get_start = time.perf_counter()
                start, img, fid, metadata = q_img.get()
                t_get_end = time.perf_counter()

                if start.name == "STOP":
                    logger.debug("Stop signal received in main thread.")
                    q_img.task_done()
                    break

                logger.debug(f"[{start.name}] START GPU Stage")
                get_wait_time = t_get_end - t_get_start
                if get_wait_time > 0.01:
                    logger.warning(
                        f"[{start.name}] Waited {get_wait_time:.4f}s to get from q_img (Read stage is likely the bottleneck)"
                    )

                t_gpu_stage_start = time.perf_counter()

                img_gpu = cp.asarray(img)
                if debug:
                    cp.cuda.runtime.deviceSynchronize()

                # Deconvolution
                t_deconv_start = time.perf_counter()
                res = deconvolve_lucyrichardson_guo(img_gpu, projectors(step), iters=1)
                if debug:
                    cp.cuda.runtime.deviceSynchronize()
                t_deconv_end = time.perf_counter()
                logger.debug(
                    f"[{start.name}] Deconvolution kernel took: {t_deconv_end - t_deconv_start:.4f}s"
                )

                single_chan = res.shape[1] == 1
                if single_chan:
                    mins, maxs = cp.percentile(res.squeeze(), (0.1, 99.999))
                else:
                    mins, maxs = cp.percentile(res, (0.1, 99.999), axis=(0, 2, 3), keepdims=True)

                if ((maxs - mins) < 1e-20).any():
                    logger.warning(f"[{start.name}] Dynamic range is very low.")

                scale = 65534 / (maxs - mins + 1e-20)
                if debug:
                    cp.cuda.runtime.deviceSynchronize()

                towrite = (
                    cp.clip((res - mins) * scale, 0.0, 65534.0)
                    .astype(np.uint16)
                    .get()
                    .reshape(-1, 2048, 2048)
                )
                del res, img_gpu

                scaling = {
                    "deconv_min": list(map(float, mins.flatten())),
                    "deconv_scale": list(map(float, scale.flatten())),
                }
                logger.info(f"[{start.name}] Scaling: {scaling['deconv_min']}, {scaling['deconv_scale']}")
                total_gpu_time = time.perf_counter() - t_gpu_stage_start
                logger.info(f"{start.name}: GPU took {total_gpu_time:.2f}s")

                t_put_write_start = time.perf_counter()
                q_write.put((start, towrite, fid, metadata | scaling))
                t_put_write_end = time.perf_counter()
                put_write_wait_time = t_put_write_end - t_put_write_start
                if put_write_wait_time > 0.01:
                    logger.warning(
                        f"[{start.name}] Waited {put_write_wait_time:.4f}s to put in q_write (Write stage is likely the bottleneck)"
                    )

                q_img.task_done()
                callback()

    except Exception as e:
        logger.exception(f"Error in processing pipeline: {e}")
        q_write.put(None)
        raise e

    q_write.put(None)
    thread.join()
    thread_write.join()


def _safe_delete_origin_dirs(files: list[Path], out: Path) -> None:
    """Delete input round folders when all inputs have corresponding outputs.

    This function groups the processed input files by their parent directory
    (e.g., `<workspace>/<round>--<roi>`). For each such folder, it verifies
    that the corresponding output directory exists at `out/<round>--<roi>` and
    that every input basename has a matching output file. Only then is the
    input folder removed.

    Args:
        files: List of input `.tif` files that were processed.
        out:   Base output directory (typically `path/analysis/deconv`).
    """
    import shutil

    if not files:
        return

    by_parent: dict[Path, list[Path]] = {}
    for f in files:
        by_parent.setdefault(f.parent, []).append(f)

    for src_dir, src_files in by_parent.items():
        dst_dir = out / src_dir.name
        if not dst_dir.exists():
            logger.warning(f"Skip delete: destination {dst_dir} does not exist for {src_dir}.")
            continue

        # Ensure every input has a corresponding output file
        missing = [s for s in src_files if not (dst_dir / s.name).exists()]
        if missing:
            logger.warning(
                f"Skip delete: {src_dir} has {len(missing)} inputs without outputs (e.g., {missing[0].name})."
            )
            continue

        # Extra safety: ensure we are not about to delete within 'analysis'
        if "analysis" in src_dir.parts:
            logger.error(f"Refusing to delete source in analysis tree: {src_dir}")
            continue

        logger.info(f"Deleting origin folder: {src_dir}")
    shutil.rmtree(src_dir)


@deconv.command()
@click.argument("path", type=click.Path(path_type=Path))
@click.argument("name", type=str)
@click.option("--ref", type=click.Path(path_type=Path), default=None)
@click.option("--limit", type=int, default=None)
@click.option("--overwrite", is_flag=True)
@click.option("--n-fids", type=int, default=2)
@click.option("--basic-name", type=str, default=None)
@click.option("--debug", is_flag=True)
def run(
    path: Path,
    name: str,
    *,
    ref: Path | None,
    limit: int | None,
    overwrite: bool,
    basic_name: str | None,
    n_fids: int,
    debug: bool = False,
):
    """GPU-accelerated very accelerated 3D deconvolution.

    Separate read and write threadsin order to have an image ready for the GPU at all times.
    Outputs in `path/analysis/deconv`.

    Args:
        path: Path to the workspace.
        name: Name of the round to deconvolve. E.g. 1_9_17.
        ref: Path of folder that have the same image indices as those we want to deconvolve.
            Used when skipping blank areas in round >1.
            We don't want to deconvolve the blanks in round 1.
        limit: Limit the total number of images to deconvolve. Mainly for debugging.
        overwrite: Overwrite existing deconvolved images.
        n_fid: Number of fiducial frames.
    """
    if debug:
        logger.remove()
        log_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{thread.name: <13}</cyan> | "
            "<level>{message}</level>"
        )
        logger.add(sys.stderr, level="DEBUG", format=log_format)
        logger.add("profiling_{time}.log", level="DEBUG", format=log_format, enqueue=True)
        logger.info("DEBUG mode enabled. Detailed profiling logs will be generated.")
    else:
        logger.remove()
        logger.add(sys.stderr, level="INFO", format="{message}")

    console.print(f"[magenta]{pyfiglet.figlet_format('3D Deconv', font='slant')}[/magenta]")

    out = path / "analysis" / "deconv"
    out.mkdir(exist_ok=True, parents=True)
    if ref is not None:
        files = []
        for roi in [r.name.split("--")[1] for r in path.glob(f"{ref}--*")]:
            ok_idxs = {
                int(f.stem.split("-")[1]) for f in sorted((path / f"{ref}--{roi}").glob(f"{ref}-*.tif"))
            }

            files.extend([
                f
                for f in sorted(path.glob(f"{name}--{roi}/{name}-*.tif"))
                if "analysis/deconv" not in str(f)
                and not f.parent.name.endswith("basic")
                and int(f.stem.split("-")[1]) in ok_idxs
            ])

    else:
        files = [
            f
            for f in sorted(path.glob(f"{name}--*/{name}-*.tif"))
            if "analysis/deconv" not in str(f) and not f.parent.name.endswith("basic")
        ]

    files = files[:limit] if limit is not None else files
    logger.info(
        f"Total: {len(files)} at {path}/{name}*" + (f" Limited to {limit}" if limit is not None else "")
    )

    if not files:
        logger.warning("No files found to process. Exiting.")
        return

    basic_name = basic_name or name
    logger.info(f"Using {path / 'basic'}/{basic_name}-*.pkl for BaSiC")

    channels = get_channels(files[0])

    basic_list: list[BaSiC] = []
    for c in channels:
        try:
            loaded = pickle.loads((path / "basic" / f"{basic_name}-{c}.pkl").read_bytes())
        except FileNotFoundError:
            try:
                loaded = pickle.loads((path / "basic" / f"{name}-{c}.pkl").read_bytes())
                logger.warning(
                    f"Could not find basic file for {basic_name}-{c}.pkl. Using {name}-{c}.pkl instead."
                )
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"Could not find basic file for {name}-{c}.pkl. Please run `preprocess basic run` first."
                )
        basic_list.append(loaded["basic"] if isinstance(loaded, dict) else loaded)

        basics = {name: basic_list}
        _run(
            files,
            path / "analysis" / "deconv",
            basics,
            overwrite=overwrite,
            n_fids=n_fids,
            debug=debug,
        )


@deconv.command()
@click.argument("path", type=click.Path(path_type=Path))
@click.option("--roi", type=str, default=None)
@click.option("--ref", type=click.Path(path_type=Path), default=None)
@click.option("--overwrite", is_flag=True)
@click.option("--n-fids", type=int, default=2)
@click.option("--basic-name", type=str)
@click.option(
    "--delete-origin", is_flag=True, help="Delete source round folders after successful deconvolution."
)
@click.option("--debug", is_flag=True)
def batch(
    path: Path,
    *,
    roi: str | None,
    ref: Path | None,
    overwrite: bool,
    n_fids: int,
    basic_name: str | None = "all",
    delete_origin: bool = False,
    debug: bool = False,
):
    if debug:
        logger.remove()
        log_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{thread.name: <13}</cyan> | "
            "<level>{message}</level>"
        )
        logger.add(sys.stderr, level="DEBUG", format=log_format)
        logger.add("profiling_{time}.log", level="DEBUG", format=log_format, enqueue=True)
        logger.info("DEBUG mode enabled. Detailed profiling logs will be generated.")
    else:
        logger.remove()
        logger.add(sys.stderr, level="INFO", format="{message}")

    ws = Workspace(path)
    rois = ws.rois
    logger.info(f"Found ROIs: {rois}")
    if not rois:
        raise ValueError("No ROIs found.")

    out = ws.deconved
    out.mkdir(exist_ok=True, parents=True)

    FORBIDDEN = ["10x", "analysis", "shifts", "fid", "registered", "old", "basic"]

    rounds = sorted(
        {
            p.name.split("--")[0]
            for p in path.iterdir()
            if "--" in p.name and p.is_dir() and not any(p.name.startswith(f) for f in FORBIDDEN)
        },
        key=lambda x: (
            x.split("_")[0].isdigit(),
            f"{int(x.split('_')[0]):02d}" if x.split("_")[0].isdigit() else x,
        ),
    )
    logger.info(f"Rounds: {rounds}")

    for r in rounds:
        logger.info(f"Processing round: {r}")
        if ref is not None:
            files = []
            for roi_ in rois:
                logger.info(f"Processing {r}--{roi_}")
                ok_idxs = {
                    int(f.stem.split("-")[1]) for f in sorted((path / f"{ref}--{roi_}").glob(f"{ref}-*.tif"))
                }
                if not ok_idxs:
                    logger.warning(f"No reference files found for {ref}--{roi_}. Skipping ROI.")
                    continue

                files.extend([
                    f
                    for f in sorted(path.glob(f"{r}--{roi_}/{r}-*.tif"))
                    if "analysis/deconv" not in str(f)
                    and not f.parent.name.endswith("basic")
                    and int(f.stem.split("-")[1]) in ok_idxs
                ])

        else:
            files = [
                f
                for f in sorted(path.glob(f"{r}--*/{r}-*.tif"))
                if "analysis/deconv" not in str(f) and not f.parent.name.endswith("basic")
            ]

        if not overwrite:
            n_before = len(files)
            files = [f for f in files if not (out / f.parent.name / f.name).exists()]
            logger.info(f"Skipping {n_before - len(files)} already processed files.")

        if not files:
            logger.warning(
                f"No files found to process for round {r}, skipping. Use --overwrite to reprocess."
            )
            continue

        meta = get_metadata(files[0])
        channels = get_channels(files[0])
        try:
            waveform = json.loads(meta["waveform"])
            step = int(waveform["params"]["step"] * 10)
            logger.info(f"Using PSF step={step} from waveform metadata.")
        except (KeyError, json.JSONDecodeError):
            step = 6
            logger.warning("Could not determine step from metadata, using default step=6.")

        basic_list: list[BaSiC] = []
        for c in channels:
            try:
                basic_path = path / "basic" / f"{basic_name}-{c}.pkl"
                loaded = pickle.loads(basic_path.read_bytes())
                logger.debug(f"Loaded BaSiC profile from {basic_path}")
            except FileNotFoundError:
                try:
                    basic_path = path / "basic" / f"{r}-{c}.pkl"
                    loaded = pickle.loads(basic_path.read_bytes())
                    logger.warning(
                        f"Could not find basic file for {basic_name}-{c}.pkl. Using {r}-{c}.pkl instead."
                    )
                except FileNotFoundError:
                    raise FileNotFoundError(
                        f"Could not find basic file for {r}-{c}.pkl. Please run `preprocess basic run` first."
                    )
            basic_list.append(loaded["basic"] if isinstance(loaded, dict) else loaded)

        basics = {r: basic_list}

        _run(
            files,
            path / "analysis" / "deconv",
            basics,
            overwrite=overwrite,
            n_fids=n_fids,
            step=step,
            debug=debug,
        )

        # Optionally delete origin folders once outputs exist for all inputs
        if delete_origin:
            try:
                _safe_delete_origin_dirs(files, out)
            except Exception as e:
                logger.error(f"Failed to delete origin folders for round {r}: {e}")
                # Do not raise; continue with other rounds


if __name__ == "__main__":
    deconv()
