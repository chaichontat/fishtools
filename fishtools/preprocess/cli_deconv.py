import json
import pickle
import queue
import re
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

from fishtools.preprocess.deconv.normalize import quantize_global
from fishtools.preprocess.deconv.logging_utils import configure_logging
from fishtools.preprocess.deconv.discovery import infer_psf_step
from fishtools.preprocess.deconv.hist import sample_histograms
from fishtools.preprocess.deconv.core import (
    load_projectors_cached,
    deconvolve_lucyrichardson_guo,
    rescale as core_rescale,
)
from fishtools.utils.io import Workspace, get_channels, get_metadata
from fishtools.utils.pretty_print import progress_bar


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


## NOTE: legacy compute_range has been retired from this CLI.
## It remains available in cli_deconv_old for backward compatibility.


logger.remove()
logger.configure(
    handlers=[{"sink": RichHandler(), "format": "{message}", "level": "INFO"}]
)
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


@click.group()
def deconv() -> None:
    """3D deconvolution workflows."""


def _run(
    paths: list[Path],
    out: Path,
    basics: dict[str, list[BaSiC]],
    overwrite: bool,
    n_fids: int,
    step: int = 6,
    debug: bool = False,
    *,
    # NEW OPTIONS:
    save_float32: bool = False,
    save_histograms: bool = False,
    histogram_bins: int = 8192,
    m_glob: np.ndarray | None = None,
    s_glob: np.ndarray | None = None,
):
    """
    Core pipeline:
      - Read -> BaSiC -> 3D deconvolution (Guo) on GPU
      - Write 16-bit JPEG-XR deliverable (with fiducials appended)
      - (optional) Write raw float32 deconvolved non-fid stack to analysis/deconv32
      - (optional) Write per-tile deconvolved histograms to analysis/deconv32

    Float32 + histograms are intended for estimating global quantization; they are *not*
    re-scaled. Histograms are per-channel with tile-specific edges (min/max of the tile).
    """
    # Filter worklist based on requested artifacts
    original_count = len(paths)
    if not overwrite:
        if save_float32 or save_histograms:
            # Only skip when the requested artifacts already exist in deconv32
            def needs_artifacts(f: Path) -> bool:
                sub32 = (out.parent / "deconv32") / f.parent.name
                targets: list[Path] = []
                if save_float32:
                    targets.append(sub32 / f.name)
                if save_histograms:
                    targets.append((sub32 / f.name).with_suffix(".histogram.txt"))
                return any(not t.exists() for t in targets)

            paths = [f for f in paths if needs_artifacts(f)]
        else:
            # Legacy behavior: skip when 16-bit deliverable exists
            paths = [f for f in paths if not (out / f.parent.name / f.name).exists()]

    if not paths:
        logger.info(
            "No files to process after filtering: %d sampled, 0 pending%s.",
            original_count,
            " (overwrite off)" if not overwrite else "",
        )
        return
    else:
        logger.info(
            "Processing %d/%d tiles%s …",
            len(paths),
            original_count,
            " (overwrite)" if overwrite else "",
        )

    # Where the float32 + histogram artifacts go
    out32 = out.parent / "deconv32"
    if save_float32 or save_histograms:
        out32.mkdir(exist_ok=True, parents=True)

    # Writer queue carries all artifacts
    q_write: queue.Queue[
        tuple[
            Path,
            np.ndarray,  # uint16 deliverable (ZC,Y,X)
            Iterable[np.ndarray],  # fid frames (unchanged)
            dict[str, Any],  # metadata (extended with deconv_min/scale as before)
            np.ndarray | None,  # optional float32 deconv (ZC,Y,X) on CPU
            dict | None,  # optional histogram payload
        ]
        | None
    ] = queue.Queue(maxsize=3)

    q_img: queue.Queue[
        tuple[Path, np.ndarray, Iterable[np.ndarray], dict[str, Any]]
    ] = queue.Queue(maxsize=1)
    profiles: list[tuple[cp.ndarray, cp.ndarray]] = []

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
                logger.debug(
                    f"[{file.name}] Disk read took: {t_read_end - t_read_start:.4f}s"
                )

                fid = np.atleast_3d(img[-n_fids:])
                nofid = (
                    img[:-n_fids].reshape(-1, len(bits), 2048, 2048).astype(cp.float32)
                )
                nofid = cp.asarray(nofid)
            except ValueError as e:
                raise Exception(
                    f"File {file.resolve()} is corrupted. Please check the file."
                ) from e

            # BaSiC shading correction (GPU)
            t_basic_start = time.perf_counter()
            if not profiles:
                darkfield = cp.asarray(
                    np.stack([basic.darkfield for basic in basics[round_]], axis=0)[
                        np.newaxis, ...
                    ]
                )
                flatfield = cp.asarray(
                    np.stack([basic.flatfield for basic in basics[round_]], axis=0)[
                        np.newaxis, ...
                    ]
                )
                profiles.append((darkfield, flatfield))
            nofid -= profiles[0][0]
            nofid /= profiles[0][1]
            cp.clip(nofid, 0, None, out=nofid)
            t_basic_end = time.perf_counter()
            logger.debug(
                f"[{file.name}] BaSiC correction took: {t_basic_end - t_basic_start:.4f}s"
            )

            total_preprocess_time = time.perf_counter() - t0_file
            logger.debug(
                f"[{file.name}] Total preprocess time: {total_preprocess_time:.4f}s"
            )

            q_img.put((file, nofid, fid, metadata))
        q_img.put((Path("STOP"), np.array([]), np.array([]), {}))  # sentinel

    def f_write():
        while True:
            gotten = q_write.get()
            if gotten is None:
                break
            file, towrite_u16, fid, metadata, f32_payload, hist_payload = gotten
            if file.name == "STOP":
                break

            sub16 = out / file.parent.name
            sub16.mkdir(exist_ok=True, parents=True)

            # 2) write 16-bit deliverable (with fid appended)
            if towrite_u16 is not None:
                (sub16 / file.name).with_suffix(".deconv.json").write_text(
                    json.dumps(metadata, indent=2)
                )
                tifffile.imwrite(
                    sub16 / file.name,
                    np.concatenate([towrite_u16, fid], axis=0),
                    compression=22610,
                    compressionargs={"level": 0.75},
                    metadata=metadata,
                )

            # 3) optional float32 deconvolved (non-fid) + histogram
            if (save_float32 or save_histograms) and f32_payload is not None:
                sub32 = out32 / file.parent.name
                sub32.mkdir(exist_ok=True, parents=True)

                if save_float32:
                    # pure float32, uncompressed; change compression if you want zlib
                    tifffile.imwrite(
                        sub32 / file.name,
                        f32_payload,  # shape (-1, 2048, 2048), dtype float32
                        dtype=np.float32,
                        metadata={"dtype": "float32", **metadata},
                        compression="zlib",
                    )

                if save_histograms and hist_payload is not None:
                    import csv

                    hist_path = (sub32 / file.name).with_suffix(".histogram.csv")
                    with open(hist_path, "w", newline="") as fh:
                        writer = csv.writer(fh)
                        writer.writerow(["channel", "bin_left", "bin_right", "count"])
                        counts_list: list[np.ndarray] = hist_payload["counts"]  # type: ignore[index]
                        edges_list: list[np.ndarray] = hist_payload["edges"]  # type: ignore[index]
                        for channel_idx, (edges, counts) in enumerate(
                            zip(edges_list, counts_list, strict=True)
                        ):
                            for left, right, count in zip(
                                edges[:-1], edges[1:], counts, strict=True
                            ):
                                writer.writerow(
                                    [
                                        channel_idx,
                                        f"{float(left):.8g}",
                                        f"{float(right):.8g}",
                                        int(count),
                                    ]
                                )

            q_write.task_done()
        logger.debug("Write thread finished.")

    try:
        threading.current_thread().name = "MainGPUThread"
        thread = threading.Thread(
            target=f_read, args=(paths,), daemon=True, name="ReadThread"
        )
        thread.start()

        thread_write = threading.Thread(
            target=f_write, args=(), daemon=True, name="WriteThread"
        )
        thread_write.start()

        with progress_bar(len(paths)) as callback:
            while True:
                t_get_start = time.perf_counter()
                start, img, fid, metadata = q_img.get()
                t_get_end = time.perf_counter()

                if start.name == "STOP":
                    q_img.task_done()
                    break

                if (t_get_end - t_get_start) > 0.01:
                    logger.warning(
                        f"[{start.name}] Waited {(t_get_end - t_get_start):.4f}s to get from q_img"
                    )

                logger.debug(f"[{start.name}] START GPU Stage")
                t_gpu_stage_start = time.perf_counter()

                img_gpu = cp.asarray(img)
                if debug:
                    cp.cuda.runtime.deviceSynchronize()

                # --- 3D deconvolution (Guo) ---
                t_deconv_start = time.perf_counter()
                res = deconvolve_lucyrichardson_guo(img_gpu, projectors(step), iters=1)
                if debug:
                    cp.cuda.runtime.deviceSynchronize()
                t_deconv_end = time.perf_counter()
                logger.debug(
                    f"[{start.name}] Deconvolution kernel took: {t_deconv_end - t_deconv_start:.4f}s"
                )

                towrite_u16 = None
                if not save_float32:
                    # single_chan = res.shape[1] == 1
                    # if single_chan:
                    #     mins, maxs = cp.percentile(res.squeeze(), (0.1, 99.999))
                    # else:
                    #     mins, maxs = cp.percentile(
                    #         res, (0.1, 99.999), axis=(0, 2, 3), keepdims=True
                    #     )

                    towrite_u16 = quantize_global(
                        res,
                        m_glob,
                        s_glob,
                        i_max=65535,
                        return_stats=False,
                        as_numpy=True,
                    )

                    # if ((maxs - mins) < 1e-20).any():
                    #     logger.warning(f"[{start.name}] Dynamic range is very low.")

                    # legacy per-tile uint16 (we'll keep writing this unless you switch to global mapping)
                    # scale = 65534 / (maxs - mins + 1e-20)

                    # arr_u16 = cp.clip((res - mins) * scale, 0.0, 65534.0)
                    # arr_u16 = cp.rint(arr_u16).astype(np.uint16)  # unbiased rounding
                    # towrite_u16 = arr_u16.get().reshape(-1, 2048, 2048)

                f32_payload = None
                hist_payload = None

                if save_float32 or save_histograms:
                    # (Z,C,Y,X) -> (Z*C,Y,X) on CPU
                    f32_cpu = (
                        res.reshape(-1, 2048, 2048).get().astype(np.float32, copy=False)
                    )
                    if save_float32:
                        f32_payload = f32_cpu

                    if save_histograms:
                        seed = int(float(f32_cpu[0, 0, 0]) * 1e6) % (2**32)
                        hist_payload = sample_histograms(
                            res,
                            bins=int(histogram_bins),
                            strides=(4, 2, 2),
                            seed=int(seed),
                        )

                # free GPU intermediates
                try:
                    del arr_u16
                except UnboundLocalError:
                    ...

                del img_gpu
                total_gpu_time = time.perf_counter() - t_gpu_stage_start
                logger.info(f"{start.name}: GPU took {total_gpu_time:.2f}s")

                # extend metadata with legacy deconv_min/scale (unchanged behavior)
                if towrite_u16 is not None:
                    assert m_glob is not None and s_glob is not None
                    scaling = {
                        "deconv_min": list(map(float, m_glob.flatten())),
                        "deconv_scale": list(map(float, s_glob.flatten())),
                        "prenormalized": True,
                    }
                    metadata = metadata | scaling

                # enqueue writer job
                q_write.put((
                    start,
                    towrite_u16,
                    fid,
                    metadata,
                    f32_payload,
                    hist_payload,
                ))

                # drop references
                del res
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
            logger.warning(
                f"Skip delete: destination {dst_dir} does not exist for {src_dir}."
            )
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
@click.option("--basic-name", type=str, default="all")
@click.option("--debug", is_flag=True)
@click.option("--multi-gpu", is_flag=True, help="Use experimental multi-GPU scheduler.")
@click.option(
    "--devices",
    type=str,
    default="auto",
    show_default=True,
    help="Comma-separated CUDA device indices when using --multi-gpu.",
)
@click.option(
    "--queue-depth",
    type=int,
    default=5,
    show_default=True,
    help="Maximum in-flight tiles per GPU in multi-GPU mode.",
)
@click.option(
    "--stop-on-error/--continue-on-error",
    default=True,
    show_default=True,
    help="Stop all workers after the first failure in multi-GPU mode.",
)
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
    multi_gpu: bool = False,
    devices: str = "auto",
    queue_depth: int = 5,
    stop_on_error: bool = True,
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
    if multi_gpu:
        from fishtools.preprocess.cli_deconv_multi import (
            DEFAULT_QUEUE_DEPTH,
            multi_run,
            parse_device_spec,
        )

        configure_logging(debug)

        try:
            device_list = parse_device_spec(devices)
        except Exception as exc:  # noqa: BLE001
            raise click.ClickException(str(exc)) from exc

        depth = queue_depth if queue_depth > 0 else DEFAULT_QUEUE_DEPTH

        multi_run(
            path,
            name,
            ref=ref,
            limit=limit,
            overwrite=overwrite,
            n_fids=n_fids,
            basic_name=basic_name,
            debug=debug,
            devices=device_list,
            queue_depth=depth,
            stop_on_error=stop_on_error,
        )
        return

    configure_logging(debug)

    console.print(
        f"[magenta]{pyfiglet.figlet_format('3D Deconv', font='slant')}[/magenta]"
    )

    out = path / "analysis" / "deconv"
    out.mkdir(exist_ok=True, parents=True)
    if ref is not None:
        files = []
        for roi in [r.name.split("--")[1] for r in path.glob(f"{ref}--*")]:
            ok_idxs = {
                int(f.stem.split("-")[1])
                for f in sorted((path / f"{ref}--{roi}").glob(f"{ref}-*.tif"))
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
        f"Total: {len(files)} at {path}/{name}*"
        + (f" Limited to {limit}" if limit is not None else "")
    )

    if not files:
        logger.warning("No files found to process. Exiting.")
        return

    basic_name = basic_name or name
    logger.info(f"Using {path / 'basic'}/{basic_name}-*.pkl for BaSiC")

    channels = get_channels(files[0])

    basic_list: list[BaSiC] = []

    global_scaling = (
        np.loadtxt(path / "analysis" / "deconv32" / "deconv_scaling" / f"{name}.txt")
        .astype(np.float32)
        .reshape((2, -1))
    )

    for c in channels:
        try:
            loaded = pickle.loads(
                (path / "basic" / f"{basic_name}-{c}.pkl").read_bytes()
            )
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
            m_glob=global_scaling[0],
            s_glob=global_scaling[1],
        )


@deconv.command()
@click.argument("path", type=click.Path(path_type=Path))
@click.option(
    "-n",
    "num_tiles",
    type=int,
    default=100,
    show_default=True,
    help="Number of tiles to sample.",
)
@click.option(
    "--percent",
    type=float,
    default=0.1,
    show_default=True,
    help="Percent of total tiles to sample.",
)
@click.option(
    "--roi",
    type=str,
    multiple=True,
    help="Restrict to one or more ROI names (repeatable).",
)
@click.option(
    "--round",
    "round_names",
    type=str,
    multiple=True,
    help="Restrict to specific round names (repeatable).",
)
@click.option(
    "--seed",
    type=int,
    default=0,
    show_default=True,
    help="Random seed for reproducible sampling.",
)
@click.option("--histogram-bins", type=int, default=8192, show_default=True)
@click.option(
    "--overwrite", is_flag=True, help="Reprocess and overwrite outputs if exist."
)
@click.option("--n-fids", type=int, default=2, show_default=True)
@click.option("--basic-name", type=str, default="all", show_default=True)
@click.option("--debug", is_flag=True)
@click.option("--multi-gpu", is_flag=True, help="Use experimental multi-GPU scheduler.")
@click.option(
    "--devices",
    type=str,
    default="auto",
    show_default=True,
    help="Comma-separated CUDA device indices when using --multi-gpu.",
)
@click.option(
    "--queue-depth",
    type=int,
    default=5,
    show_default=True,
    help="Maximum in-flight tiles per GPU in multi-GPU mode.",
)
@click.option(
    "--stop-on-error/--continue-on-error",
    default=True,
    show_default=True,
    help="Stop all workers after the first failure in multi-GPU mode.",
)
def prepare(
    path: Path,
    *,
    num_tiles: int,
    percent: float,
    roi: tuple[str, ...],
    round_names: tuple[str, ...],
    seed: int,
    histogram_bins: int,
    overwrite: bool,
    n_fids: int,
    basic_name: str,
    debug: bool,
    multi_gpu: bool,
    devices: str,
    queue_depth: int,
    stop_on_error: bool,
):
    """Sample tiles and emit float32 + histogram artifacts for global scaling prep.

    - Samples up to `num_tiles` tiles across the workspace (optionally restricted by ROI/round).
    - Runs GPU 3D deconvolution and writes:
        - 16-bit JPEG‑XR deliverables (legacy)
        - Float32 deconvolved stacks to `analysis/deconv32`
        - Per-channel histogram text files next to float32 output
    """
    configure_logging(debug)

    if multi_gpu:
        from fishtools.preprocess.cli_deconv_multi import (
            DEFAULT_QUEUE_DEPTH,
            multi_prepare,
            parse_device_spec,
        )

        try:
            device_list = parse_device_spec(devices)
        except Exception as exc:  # noqa: BLE001
            raise click.ClickException(str(exc)) from exc

        depth = queue_depth if queue_depth > 0 else DEFAULT_QUEUE_DEPTH

        multi_prepare(
            path,
            num_tiles=num_tiles,
            percent=percent,
            roi=roi,
            round_names=round_names,
            seed=seed,
            histogram_bins=histogram_bins,
            overwrite=overwrite,
            n_fids=n_fids,
            basic_name=basic_name,
            debug=debug,
            devices=device_list,
            queue_depth=depth,
            stop_on_error=stop_on_error,
        )
        return

    ws = Workspace(path)
    all_rois = ws.rois
    rois = list(all_rois) if not roi else [r for r in roi if r in all_rois]
    if roi and not rois:
        raise ValueError("No matching ROIs found for the provided --roi filters.")

    # Discover rounds (mirror logic from batch())
    FORBIDDEN = ["10x", "analysis", "shifts", "fid", "registered", "old", "basic"]
    rounds_all = sorted(
        {
            p.name.split("--")[0]
            for p in path.iterdir()
            if "--" in p.name
            and p.is_dir()
            and not any(p.name.startswith(f) for f in FORBIDDEN)
        },
        key=lambda x: (
            x.split("_")[0].isdigit(),
            f"{int(x.split('_')[0]):02d}" if x.split("_")[0].isdigit() else x,
        ),
    )

    rounds = (
        rounds_all if not round_names else [r for r in round_names if r in rounds_all]
    )
    if round_names and not rounds:
        raise ValueError("No matching rounds found for the provided --round filters.")

    # Collect candidate files
    candidates: list[Path] = []
    for r in rounds:
        for roi_ in rois:
            candidates.extend(
                f
                for f in sorted(path.glob(f"{r}--{roi_}/{r}-*.tif"))
                if "analysis/deconv" not in str(f)
                and not f.parent.name.endswith("basic")
                and not f.name.startswith("fid")
            )
    if not candidates:
        logger.warning("No candidate tiles found for sampling. Exiting.")
        return

    # Sample tiles
    rng = np.random.default_rng(seed)
    k = max(min(num_tiles, len(candidates)), int(len(candidates) * percent))
    idx = rng.choice(len(candidates), size=k, replace=False)
    sampled = [candidates[i] for i in idx]

    # Group by round to respect BaSiC profiles in _run
    by_round: dict[str, list[Path]] = {}
    for f in sampled:
        r = f.name.split("-")[0]
        by_round.setdefault(r, []).append(f)

    logger.info(
        f"Preparing {k} tiles across {len(by_round)} rounds with float32+histograms …"
    )

    out = path / "analysis" / "deconv"
    out.mkdir(exist_ok=True, parents=True)

    for r, files in by_round.items():
        if not files:
            continue

        # Determine step per round
        try:
            meta = get_metadata(files[0])
            waveform = json.loads(meta["waveform"])
            step_val = int(waveform["params"]["step"] * 10)
            logger.info(f"[{r}] Using PSF step={step_val} from waveform metadata.")
        except (KeyError, json.JSONDecodeError):
            step_val = 6
            logger.warning(
                f"[{r}] Could not determine step from metadata, using default step=6."
            )

        # Load BaSiC profiles for this round
        channels = get_channels(files[0])
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
                        f"[{r}] Could not find basic file for {basic_name}-{c}.pkl. Using {r}-{c}.pkl instead."
                    )
                except FileNotFoundError:
                    raise FileNotFoundError(
                        f"[{r}] Could not find basic file for {r}-{c}.pkl. Please run `preprocess basic run` first."
                    )
            basic_list.append(loaded["basic"] if isinstance(loaded, dict) else loaded)

        basics = {r: basic_list}
        logger.info(f"[{r}] Selected {len(files)} sampled tiles (pre-filter) …")
        _run(
            files,
            out,
            basics,
            overwrite=overwrite,
            n_fids=n_fids,
            step=step_val,
            debug=debug,
            save_float32=True,
            save_histograms=True,
            histogram_bins=histogram_bins,
        )

    logger.info(
        "Completed deconv prepare: float32 stacks and histograms written under analysis/deconv32."
    )


@deconv.command()
@click.argument("path", type=click.Path(path_type=Path))
@click.option("--ref", type=click.Path(path_type=Path), default=None)
@click.option("--overwrite", is_flag=True)
@click.option("--n-fids", type=int, default=2)
@click.option("--basic-name", type=str)
@click.option(
    "--delete-origin",
    is_flag=True,
    help="Delete source round folders after successful deconvolution.",
)
@click.option("--debug", is_flag=True)
@click.option("--multi-gpu", is_flag=True, help="Use experimental multi-GPU scheduler.")
@click.option(
    "--devices",
    type=str,
    default="auto",
    show_default=True,
    help="Comma-separated CUDA device indices when using --multi-gpu.",
)
@click.option(
    "--queue-depth",
    type=int,
    default=5,
    show_default=True,
    help="Maximum in-flight tiles per GPU in multi-GPU mode.",
)
@click.option(
    "--stop-on-error/--continue-on-error",
    default=True,
    show_default=True,
    help="Stop all workers after the first failure in multi-GPU mode.",
)
def batch(
    path: Path,
    *,
    ref: Path | None,
    overwrite: bool,
    n_fids: int,
    basic_name: str | None = "all",
    delete_origin: bool = False,
    debug: bool = False,
    multi_gpu: bool = False,
    devices: str = "auto",
    queue_depth: int = 5,
    stop_on_error: bool = True,
):
    configure_logging(debug)

    if multi_gpu:
        from fishtools.preprocess.cli_deconv_multi import (
            DEFAULT_QUEUE_DEPTH,
            multi_batch,
            parse_device_spec,
        )

        try:
            device_list = parse_device_spec(devices)
        except Exception as exc:  # noqa: BLE001
            raise click.ClickException(str(exc)) from exc

        depth = queue_depth if queue_depth > 0 else DEFAULT_QUEUE_DEPTH

        multi_batch(
            path,
            ref=ref,
            overwrite=overwrite,
            n_fids=n_fids,
            basic_name=basic_name,
            delete_origin=delete_origin,
            debug=debug,
            devices=device_list,
            queue_depth=depth,
            stop_on_error=stop_on_error,
        )
        return

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
            if "--" in p.name
            and p.is_dir()
            and not any(p.name.startswith(f) for f in FORBIDDEN)
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
                    int(f.stem.split("-")[1])
                    for f in sorted((path / f"{ref}--{roi_}").glob(f"{ref}-*.tif"))
                }
                if not ok_idxs:
                    logger.warning(
                        f"No reference files found for {ref}--{roi_}. Skipping ROI."
                    )
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
                if "analysis/deconv" not in str(f)
                and not f.parent.name.endswith("basic")
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
        step, inferred = infer_psf_step(files[0])
        if inferred:
            logger.info(f"Using PSF step={step} from waveform metadata.")
        else:
            logger.warning(f"Could not determine step from metadata, using default step={step}.")

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
