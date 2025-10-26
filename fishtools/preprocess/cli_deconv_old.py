import json
import pickle
import queue
import re
import threading
import time
from collections.abc import Iterable
from itertools import chain
from pathlib import Path
from typing import Any

import cupy as cp
import numpy as np
import pyfiglet
import rich_click as click
import tifffile
from basicpy import BaSiC
from loguru import logger
from rich.console import Console

from fishtools.preprocess.deconv.core import deconvolve_lucyrichardson_guo, projectors
from fishtools.preprocess.deconv.helpers import safe_delete_origin_dirs
from fishtools.utils.io import Workspace, get_channels, get_metadata, safe_imwrite
from fishtools.utils.logging import setup_cli_logging
from fishtools.utils.pretty_print import progress_bar

DATA = Path(__file__).parent.parent.parent / "data"


def _setup_cli_logging(
    workspace: Path,
    *,
    component: str,
    file_tag: str,
    debug: bool = False,
    extra: dict[str, Any] | None = None,
) -> Path | None:
    return setup_cli_logging(
        workspace,
        component=component,
        file=file_tag,
        debug=debug,
        extra=extra,
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
    """
    Find the min and scale of the deconvolution for all files in a directory.
    The reverse scaling equation is:
                 s_global
        scaled = -------- * img + s_global * (min - m_global)
                  scale
    Hence we want scale_global to be as low as possible
    and min_global to be as high as possible.
    """
    ws = Workspace(path)
    files = sorted(ws.deconved.glob(f"{round_}*/*.tif"))
    n_c = len(round_.split("_"))

    files = [u for p in files if (u := p.with_suffix(".deconv.json")).exists()]
    n = len(files)

    deconv_min = np.zeros((n, n_c))
    deconv_scale = np.zeros((n, n_c))
    logger.info(f"Found {n} files")
    if not files:
        logger.warning(f"No files found in {ws.deconved / f'{round_}*'}. Skipping.")
        return

    with progress_bar(len(files)) as pbar:
        for i, f in enumerate(files):
            meta = json.loads(f.read_text())

            try:
                deconv_min[i, :] = meta["deconv_min"]
                deconv_scale[i, :] = meta["deconv_scale"]
            except KeyError:
                raise AttributeError("No deconv metadata found.")
            pbar()

    logger.info(f"Calculating percentiles: {perc_min, perc_scale}")

    if isinstance(perc_min, float) and isinstance(perc_scale, float):
        m_ = np.percentile(deconv_min, perc_min, axis=0)
        s_ = np.percentile(deconv_scale, perc_scale, axis=0)
    elif isinstance(perc_min, list) and isinstance(perc_scale, list):
        m_, s_ = np.zeros(n_c), np.zeros(n_c)
        for i in range(n_c):
            m_[i] = np.percentile(deconv_min[:, i], perc_min[i])
            s_[i] = np.percentile(deconv_scale[:, i], perc_scale[i])
    else:
        raise ValueError("perc_min and perc_scale must be either float or list of floats.")

    if np.any(m_ == 0) or np.any(s_ == 0):
        raise ValueError("Found a channel with min=0. This is not allowed.")

    ws.deconv_scaling().mkdir(exist_ok=True)
    np.savetxt(ws.deconv_scaling(round_), np.vstack([m_, s_]))
    logger.info(f"Saved to {ws.deconv_scaling(round_)}")


console = Console()


@click.group()
def deconv(): ...


PROTEIN_PERC_MIN = 50
PROTEIN_PERC_SCALE = 50


def _get_percentiles_for_round(
    round_name: str,
    default_perc_min: float,
    default_perc_scale: float,
    max_rna_bit: int,
    override: dict[str, tuple[float, float]] | None = None,
) -> tuple[list[float], list[float]]:
    """Determines the percentile values for min and scale based on the round name components."""
    percentile_mins: list[float] = []
    percentile_scales: list[float] = []
    override = override or {}

    bits = round_name.split("_")
    for bit in bits:
        if bit in override:
            min_val, scale_val = override[bit]
            logger.info(f"Using override for bit '{bit}' to {override[bit]} in round '{round_name}'.")
            percentile_mins.append(min_val)
            percentile_scales.append(scale_val)
            continue

        is_rna_bit = (bit.isdigit() and int(bit) <= max_rna_bit) or re.match(r"^b\d{3}$", bit)
        if not is_rna_bit:
            # Protein staining often has bright flare spots.
            logger.info(
                f"Bit '{bit}' in round '{round_name}' is non-numeric or exceeds max_rna_bit ({max_rna_bit}). "
                f"Using protein percentiles: min={PROTEIN_PERC_MIN}, scale={PROTEIN_PERC_SCALE}."
            )
        percentile_mins.append(PROTEIN_PERC_MIN if not is_rna_bit else default_perc_min)
        percentile_scales.append(PROTEIN_PERC_SCALE if not is_rna_bit else default_perc_scale)
    return percentile_mins, percentile_scales


@deconv.command()
@click.argument("path", type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path))
@click.option("--perc_min", type=float, default=0.1, help="Percentile of the min")
@click.option("--perc_scale", type=float, default=0.1, help="Percentile of the scale")
@click.option("--overwrite", is_flag=True)
@click.option("--override", type=str, multiple=True)
@click.option("--max-rna-bit", type=int, default=36)
def compute_range(
    path: Path,
    perc_min: float = 0.1,
    perc_scale: float = 0.1,
    overwrite: bool = False,
    max_rna_bit: int = 36,
    override: list[str] | None = None,
):
    """
    Find the scaling factors of all images in the children sub-folder of `path`.
    Run this on the entire workspace. See `_compute_range` for more details.

    Args:
        path: The root path containing round subdirectories (e.g., 'path/analysis/deconv').
        perc_min: Default percentile for calculating the minimum intensity value for RNA bits.
        perc_scale: Default percentile for calculating the scaling factor for RNA bits.
        overwrite: If True, recompute and overwrite existing scaling files.
        max_rna_bit: The maximum integer value considered an RNA bit. Others are treated differently (e.g., protein).
    """
    _setup_cli_logging(
        path,
        component="preprocess.deconv.compute_range",
        file_tag="compute_range",
        extra={
            "perc_min": perc_min,
            "perc_scale": perc_scale,
            "overwrite": overwrite,
        },
    )

    if "deconv" not in path.resolve().as_posix():
        raise ValueError(f"This command must be run within a 'deconv' directory. Path provided: {path}")

    ws = Workspace(path)
    override = override or []
    rounds = ws.rounds
    try:
        override_dict = {(parts := v.split(","))[0]: (float(parts[1]), float(parts[2])) for v in override}
    except ValueError:
        raise ValueError("Override must be in the format 'bit,min,scale'.")
    del override

    if not len(set(override_dict) & set(chain.from_iterable(map(lambda x: x.split("_"), rounds)))) == len(
        override_dict
    ):
        raise ValueError("Overridden bits must exist.")

    logger.info(f"Found rounds: {rounds}")
    scaling_dir = path / "deconv_scaling"
    scaling_dir.mkdir(exist_ok=True)

    for round_name in rounds:
        scaling_file = scaling_dir / f"{round_name}.txt"
        if not overwrite and scaling_file.exists():
            logger.info(f"Scaling file for {round_name} already exists. Skipping.")
            continue

        try:
            logger.info(f"Processing {round_name}")
            percentile_mins, percentile_scales = _get_percentiles_for_round(
                round_name, perc_min, perc_scale, max_rna_bit, override=override_dict
            )

            _compute_range(path, round_name, perc_min=percentile_mins, perc_scale=percentile_scales)

        except FileNotFoundError:
            logger.warning(f"No .tif files found for round {round_name} in {path}. Skipping.")
        except AttributeError as e:
            logger.error(f"Metadata error processing {round_name}: {e}. Skipping.")
        except ValueError as e:
            logger.error(f"Value error processing {round_name}: {e}. Skipping.")
        except Exception as e:
            logger.exception(f"An unexpected error occurred while processing {round_name}: {e}. Skipping.")


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

            # print(nofid[5, 0, 1024:1026, 1024:1026])
            # print(profiles[0][0][0, 0, 1024:1026, 1024:1026])
            # print(profiles[0][1][0, 0, 1024:1026, 1024:1026])
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

            safe_imwrite(
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
    _setup_cli_logging(
        path,
        component="preprocess.deconv.run",
        file_tag=f"run-{name}",
        debug=debug,
        extra={
            "round": name,
            "ref": ref.name if isinstance(ref, Path) else None,
            "overwrite": overwrite,
        },
    )

    if debug:
        log_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{thread.name: <13}</cyan> | "
            "<level>{message}</level>"
        )
        logger.add("profiling_{time}.log", level="DEBUG", format=log_format, enqueue=True)
        logger.debug("DEBUG mode enabled. Detailed profiling logs will be generated.")

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

    meta = get_metadata(files[0])
    channels = get_channels(files[0])
    try:
        waveform = json.loads(meta["waveform"])
        step = int(waveform["params"]["step"] * 10)
        logger.info(f"Using PSF step={step} from waveform metadata.")
    except (KeyError, json.JSONDecodeError):
        step = 6
        logger.warning("Could not determine step from metadata, using default step=6.")

    basics = {name: basic_list}
    _run(
        files,
        path / "analysis" / "deconv",
        basics,
        overwrite=overwrite,
        n_fids=n_fids,
        debug=debug,
        step=step,
    )


@deconv.command()
@click.argument("path", type=click.Path(path_type=Path))
@click.option("--roi", type=str, default=None)
@click.option("--ref", type=click.Path(path_type=Path), default=None)
@click.option("--overwrite", is_flag=True)
@click.option("--n-fids", type=int, default=2)
@click.option("--basic-name", type=str)
@click.option("--debug", is_flag=True)
def batch(
    path: Path,
    *,
    roi: str | None,
    ref: Path | None,
    overwrite: bool,
    n_fids: int,
    basic_name: str | None = "all",
    debug: bool = False,
):
    _setup_cli_logging(
        path,
        component="preprocess.deconv.batch",
        file_tag="batch",
        debug=debug,
        extra={
            "roi": roi or "all",
            "ref": ref.name if isinstance(ref, Path) else None,
            "overwrite": overwrite,
        },
    )

    if debug:
        log_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{thread.name: <13}</cyan> | "
            "<level>{message}</level>"
        )
        logger.add("profiling_{time}.log", level="DEBUG", format=log_format, enqueue=True)
        logger.debug("DEBUG mode enabled. Detailed profiling logs will be generated.")

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

        all_round_files = sorted(path.glob(f"{r}--*/{r}-*.tif"))
        try:
            safe_delete_origin_dirs(all_round_files, path / "analysis" / "deconv")
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Failed to delete origin directories safely: {exc}")


if __name__ == "__main__":
    deconv()
