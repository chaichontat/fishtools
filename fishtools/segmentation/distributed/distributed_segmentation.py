import datetime
import json
import logging
import os
import pathlib
import shutil
from collections.abc import Callable
from importlib.metadata import version
from pathlib import Path
from typing import Annotated, Any, cast

import cellpose.io
import dask_jobqueue
import distributed
import imagecodecs
import numpy as np
import tifffile
import typer
import zarr
from cellpose import transforms as cp_transforms
from numpy.typing import NDArray
from rich.logging import RichHandler

from fishtools.preprocess.config import NumpyEncoder
from fishtools.preprocess.segmentation import unsharp_all
from fishtools.segment.normalize import sample_percentile
from fishtools.segment.train import IS_CELLPOSE_SAM, plan_path_for_device
from fishtools.segmentation.distributed.cache_utils import (
    read_nonempty_cache,
    read_normalization_cache,
    write_nonempty_cache,
    write_normalization_cache,
)
from fishtools.segmentation.distributed.gpu_cluster import cluster, myLocalCluster
from fishtools.segmentation.distributed.merge_utils import (
    block_faces,
    bounding_boxes_in_global_coordinates,
    determine_merge_relabeling,
    get_block_crops,
    get_nblocks,
    global_segment_ids,
    merge_all_boxes,
    relabel_and_write,
    remove_overlaps,
)
from fishtools.utils.pretty_print import progress_bar

# logger.remove()
# console = Console()
# logger.add(RichHandler(console=console, rich_tracebacks=True), format="{message}", level="INFO")


logging.basicConfig(level="INFO", handlers=[RichHandler(level="INFO")])
logging.getLogger("cellpose").setLevel(logging.WARNING)

logger = logging.getLogger("rich")

# Per-worker cache for PackedCellpose models (keyed by backend and model path).
_MODEL_CACHE: dict[tuple[str, str], Any] = {}

# Per-worker initialization flag to avoid repeated setup overhead.
_WORKER_INITIALIZED: bool = False
_WORKER_LOGS_DIR: str | None = None


######################## Checkpoint/Resume Functions ###########################


def compute_model_md5(model_path: Path) -> str:
    """MD5 hash of model file. Matches `md5sum <file>` CLI output."""
    import hashlib

    h = hashlib.md5()
    with open(model_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def save_run_config(
    path: Path,
    model_kwargs: dict[str, Any],
    eval_kwargs: dict[str, Any],
    blocksize: tuple[int, ...],
    input_shape: tuple[int, ...],
    overlap: int,
    preprocessing_steps: list[tuple[Callable[..., NDArray[Any]], dict[str, Any]]],
) -> None:
    """Save run configuration for resume validation."""
    model_path = Path(model_kwargs["pretrained_model"])
    config = {
        "model_md5": compute_model_md5(model_path),
        "model_path": str(model_path),
        "model_kwargs": model_kwargs,
        "eval_kwargs": eval_kwargs,
        "blocksize": list(blocksize),
        "overlap": overlap,
        "input_shape": list(input_shape),
        "preprocessing_steps": [f[0].__name__ for f in preprocessing_steps],
        "created_at": datetime.datetime.now().isoformat(),
    }
    path.write_text(json.dumps(config, indent=2, cls=NumpyEncoder))


def _normalize_for_comparison(obj: Any) -> Any:
    """Normalize objects for comparison (convert numpy arrays to lists)."""
    return json.loads(json.dumps(obj, cls=NumpyEncoder))


def validate_run_config(
    path: Path,
    model_kwargs: dict[str, Any],
    eval_kwargs: dict[str, Any],
    blocksize: tuple[int, ...],
    input_shape: tuple[int, ...],
    overlap: int,
    preprocessing_steps: list[tuple[Callable[..., NDArray[Any]], dict[str, Any]]],
) -> None:
    """Validate current config matches saved config. Raises on mismatch."""
    saved = json.loads(path.read_text())
    current_md5 = compute_model_md5(Path(model_kwargs["pretrained_model"]))

    # Normalize current values for comparison (numpy arrays -> lists)
    current_model_kwargs = _normalize_for_comparison(model_kwargs)
    current_eval_kwargs = _normalize_for_comparison(eval_kwargs)

    errors = []
    if saved["model_md5"] != current_md5:
        errors.append(f"model_md5: {saved['model_md5']} != {current_md5}")
    if saved["model_kwargs"] != current_model_kwargs:
        errors.append(f"model_kwargs differ: {saved['model_kwargs']} != {current_model_kwargs}")
    if saved["eval_kwargs"] != current_eval_kwargs:
        errors.append(f"eval_kwargs differ: {saved['eval_kwargs']} != {current_eval_kwargs}")
    if saved["blocksize"] != list(blocksize):
        errors.append(f"blocksize: {saved['blocksize']} != {list(blocksize)}")
    if saved["overlap"] != overlap:
        errors.append(f"overlap: {saved['overlap']} != {overlap}")
    if saved["input_shape"] != list(input_shape):
        errors.append(f"input_shape: {saved['input_shape']} != {list(input_shape)}")

    current_pp = [f[0].__name__ for f in preprocessing_steps]
    if saved["preprocessing_steps"] != current_pp:
        errors.append(f"preprocessing_steps: {saved['preprocessing_steps']} != {current_pp}")

    if errors:
        raise ValueError("Cannot resume - config mismatch:\n" + "\n".join(errors))

    logger.info(f"Config validated - model MD5 matches (verify: md5sum {model_kwargs['pretrained_model']})")


def load_checkpoint(path: Path) -> set[tuple[int, ...]]:
    """Load completed block indices from checkpoint file."""
    completed: set[tuple[int, ...]] = set()
    if not path.exists():
        return completed
    with open(path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                completed.add(tuple(entry["index"]))
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping corrupted checkpoint line {i}: {e}")
    return completed


def append_checkpoint(
    checkpoint_path: Path,
    block_index: tuple[int, ...],
    worker_name: str,
    duration_s: float,
    n_masks: int,
) -> None:
    """Atomically append checkpoint entry."""

    entry = {
        "index": list(block_index),
        "ts": datetime.datetime.now().isoformat(),
        "worker": worker_name,
        "duration_s": round(duration_s, 2),
        "n_masks": n_masks,
    }
    with open(checkpoint_path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def _gpu_probe() -> dict[str, Any]:
    """Executed on workers to report CUDA visibility and current device."""
    import distributed as _dist  # local import to avoid serialization surprises
    import torch  # type: ignore

    info: dict[str, Any] = {}
    try:
        info["worker"] = getattr(_dist.get_worker(), "name", "unknown")
    except Exception:
        info["worker"] = "unknown"
    try:
        info["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    except Exception:
        info["cuda_visible_devices"] = ""
    try:
        if torch.cuda.is_available():
            info["torch_current_device"] = int(torch.cuda.current_device())
        else:
            info["torch_current_device"] = None
    except Exception:
        info["torch_current_device"] = None
    return info


######################## File format functions ################################
def numpy_array_to_zarr(write_path: Path | str, array: NDArray[Any], chunks: tuple[int, ...]) -> zarr.Array:
    """
    Store an in memory numpy array to disk as a chunked Zarr array

    Parameters
    ----------
    write_path : string
        Filepath where Zarr array will be created

    array : numpy.ndarray
        The already loaded in-memory numpy array to store as zarr

    chunks : tuple, must be array.ndim length
        How the array will be chunked in the Zarr array

    Returns
    -------
    zarr.core.Array
        A read+write reference to the zarr array on disk
    """

    zarr.config.set({"array.target_shard_size_bytes": "10MB"})
    zarr_array = zarr.open(
        write_path,
        mode="w",
        shape=array.shape,
        chunks=chunks,
        dtype=array.dtype,
        codecs=[
            zarr.codecs.BytesCodec(),
            zarr.codecs.BloscCodec(
                cname="zstd",
                clevel=4,
                shuffle=zarr.codecs.BloscShuffle.shuffle,
                typesize=2,
            ),
        ],
    )
    zarr_array[...] = array
    return zarr_array


def wrap_folder_of_tiffs(
    filename_pattern: str,
    block_index_pattern: str = r"_(Z)(\d+)(Y)(\d+)(X)(\d+)",
) -> zarr.Array:
    """
    Wrap a folder of tiff files with a zarr array without duplicating data.
    Tiff files must all contain images with the same shape and data type.
    Tiff file names must contain a pattern indicating where individual files
    lie in the block grid.

    Distributed computing requires parallel access to small regions of your
    image from different processes. This is best accomplished with chunked
    file formats like Zarr and N5. This function can accommodate a folder of
    tiff files, but it is not equivalent to reformating your data as Zarr or N5.
    If your individual tiff files/tiles are huge, distributed performance will
    be poor or not work at all.

    It does not make sense to use this function if you have only one tiff file.
    That tiff file will become the only chunk in the zarr array, which means all
    workers will have to load the entire image to fetch their crop of data anyway.
    If you have a single tiff image, you should just reformat it with the
    numpy_array_to_zarr function. Single tiff files too large to fit into system
    memory are not be supported.

    Parameters
    ----------
    filename_pattern : string
        A glob pattern that will match all needed tif files

    block_index_pattern : regular expression string
        A regular expression pattern that indicates how to parse tiff filenames
        to determine where each tiff file lies in the overall block grid
        The default pattern assumes filenames like the following:
            {any_prefix}_Z000Y000X000{any_suffix}
            {any_prefix}_Z000Y000X001{any_suffix}
            ... and so on

    Returns
    -------
    zarr.core.Array
    """

    # define function to read individual files
    def imread(fname: str) -> NDArray[Any]:
        with open(fname, "rb") as fh:
            return imagecodecs.tiff_decode(fh.read(), index=None)

    # create zarr store, open it as zarr array and return
    store = tifffile.imread(
        filename_pattern,
        aszarr=True,
        imread=imread,
        pattern=block_index_pattern,
        axestiled={x: x for x in range(3)},
    )
    return zarr.open(store=store)


######################## Cluster related functions ############################


def format_slice(s: slice | tuple[slice, ...]) -> str:
    """
    Format a slice or tuple of slices as a human-readable string.

    This function converts slice objects into string representation
    suitable for logging and debugging purposes.

    Examples
    --------
    >>> format_slice(slice(1, 10, 2))
    '1:10:2'
    >>> format_slice((slice(0, 5), slice(10, 20)))
    '0:5,10:20'
    """
    if isinstance(s, tuple):
        return ",".join(format_slice(item) for item in s)
    if not isinstance(s, slice):  # type: ignore
        return str(s)

    start, stop, step = s.start, s.stop, s.step
    parts = [
        "" if start in (None, 0) else str(start),
        "" if stop is None else str(stop),
        "" if step in (None, 1) else str(step),
    ]

    return ":".join(parts).rstrip(":")


######################## the function to run on each block ####################


# ----------------------- The main function -----------------------------------#
def process_block(
    block_index: tuple[int, ...],
    crop: tuple[slice, ...],
    input_zarr: zarr.Array,
    model_kwargs: dict[str, Any],
    eval_kwargs: dict[str, Any],
    blocksize: tuple[int, ...],
    overlap: int,
    output_zarr: zarr.Array,
    preprocessing_steps: list[tuple[Callable[..., NDArray[Any]], dict[str, Any]]] = [],
    worker_logs_directory: str | None = None,
    test_mode: bool = False,
    checkpoint_path: Path | None = None,
) -> (
    tuple[NDArray[np.uint32], list[tuple[slice, ...]], NDArray[np.uint32]]
    | tuple[list[NDArray[Any]], list[tuple[slice, ...]], NDArray[np.uint32]]
):
    """
    Preprocess and segment one block, of many, with eventual merger
    of all blocks in mind. The block is processed as follows:

    (1) Read block from disk, preprocess, and segment.
    (2) Remove overlaps.
    (3) Get bounding boxes for every segment.
    (4) Remap segment IDs to globally unique values.
    (5) Write segments to disk.
    (6) Get segmented block faces.

    A user may want to test this function on one block before running
    the distributed function. When test_mode=True, steps (5) and (6)
    are omitted and replaced with:

    (5) return remapped segments as a numpy array, boxes, and box_ids

    **Absolutely cannot use `logger` in mapped functions since it cannot be pickled**

    Parameters
    ----------
    block_index : tuple
        The (i, j, k, ...) index of the block in the overall block grid

    crop : tuple of slice objects
        The bounding box of the data to read from the input_zarr array

    input_zarr : zarr.core.Array
        The image data we want to segment

    preprocessing_steps : list of tuples (default: the empty list)
        Optionally apply an arbitrary pipeline of preprocessing steps
        to the image block before running cellpose.

        Must be in the following format:
        [(f, {'arg1':val1, ...}), ...]
        That is, each tuple must contain only two elements, a function
        and a dictionary. The function must have the following signature:
        def F(image, ..., crop=None)
        That is, the first argument must be a numpy array, which will later
        be populated by the image data. The function must also take a keyword
        argument called crop, even if it is not used in the function itself.
        All other arguments to the function are passed using the dictionary.
        Here is an example:

        def F(image, sigma, crop=None):
            return gaussian_filter(image, sigma)
        def G(image, radius, crop=None):
            return median_filter(image, radius)
        preprocessing_steps = [(F, {'sigma':2.0}), (G, {'radius':4})]

    model_kwargs : dict
        Arguments passed to PackedCellposeModel
        This is how you select and parameterize a model.

    eval_kwargs : dict
        Arguments passed to the eval function of the PackedCellpose model
        This is how you parameterize model evaluation.

    blocksize : iterable (list, tuple, np.ndarray)
        The number of voxels (the shape) of blocks without overlaps

    overlap : int
        The number of voxels added to the blocksize to provide context
        at the edges

    output_zarr : zarr.core.Array
        A location where segments can be stored temporarily before
        merger is complete

    worker_logs_directory : string (default: None)
        A directory path where log files for each worker can be created
        The directory must exist

    test_mode : bool (default: False)
        The primary use case of this function is to be called by
        distributed_eval (defined later in this same module). However
        you may want to call this function manually to test what
        happens to an individual block; this is a good idea before
        ramping up to process big data and also useful for debugging.

        When test_mode is False (default) this function stores
        the segments and returns objects needed for merging between
        blocks.

        When test_mode is True this function does not store the
        segments, and instead returns them to the caller as a numpy
        array. The boxes and box IDs are also returned. When test_mode
        is True, you can supply dummy values for many of the inputs,
        such as:

        block_index = (0, 0, 0)
        output_zarr=None

    Returns
    -------
    If test_mode == False (the default), three things are returned:
        faces : a list of numpy arrays - the faces of the block segments
        boxes : a list of crops (tuples of slices), bounding boxes of segments
        box_ids : 1D numpy array, parallel to boxes, the segment IDs of the
                  boxes

    If test_mode == True, three things are returned:
        segments : np.ndarray containing the segments with globally unique IDs
        boxes : a list of crops (tuples of slices), bounding boxes of segments
        box_ids : 1D numpy array, parallel to boxes, the segment IDs of the
                  boxes
    """
    import time

    start_time = time.perf_counter()
    logger.info(f"RUNNING BLOCK: {block_index}\tREGION: [{format_slice(crop)}]")
    segmentation_3d = read_preprocess_and_segment(
        input_zarr,
        crop,
        preprocessing_steps,
        model_kwargs,
        eval_kwargs,
        worker_logs_directory,
    )
    logger.info(f"Block {block_index}: {np.max(segmentation_3d)} masks found.")
    # print(f"Block {block_index}: Segmentation result shape (3D): {segmentation_3d.shape}")

    # --- Define 3D spatial parameters for overlap removal and writing ---
    spatial_crop_slices = crop[:-1]  # Get ZYX slices from the 4D crop
    spatial_blocksize = blocksize[:-1]  # Get ZYX blocksize
    # print(f"Block {block_index}: Spatial crop (3D): {spatial_crop_slices}")
    # print(f"Block {block_index}: Spatial blocksize (3D): {spatial_blocksize}")

    # --- Remove overlaps on the 3D segmentation ---
    # Pass 3D segmentation, 3D crop slices, 3D blocksize to remove_overlaps
    segmentation_trimmed_3d, crop_trimmed_3d = remove_overlaps(
        segmentation_3d,  # Operate on 3D data
        spatial_crop_slices,  # Use 3D crop info
        overlap,
        spatial_blocksize,  # Use 3D blocksize info
    )
    crop_trimmed_3d = tuple(crop_trimmed_3d)  # Convert to tuple
    # print(f"Block {block_index}: Trimmed segmentation shape (3D): {segmentation_trimmed_3d.shape}")
    # print(f"Block {block_index}: Trimmed crop (3D): {crop_trimmed_3d}")

    # --- Calculate bounding boxes (3D) ---
    # Pass 3D trimmed segmentation and 3D trimmed crop
    boxes = bounding_boxes_in_global_coordinates(segmentation_trimmed_3d, crop_trimmed_3d)

    # --- Calculate global IDs (operates on labels, dimensions not critical here) ---
    # Need nblocks in 3D for correct mapping if using ravel_multi_index on 3D indices
    nblocks_3d = get_nblocks(input_zarr.shape[:-1], spatial_blocksize)  # Use 3D shapes
    # Assuming block_index is ZYX index (needs verification how it's generated/used)
    # If block_index is ZYXC, need to adapt. Let's assume we only need ZYX index part
    # If block_indices were generated using 4D nblocks, adjust here:
    block_index_3d = block_index[:-1]  # Assume we only need ZYX index part

    segmentation_global_3d, remap = global_segment_ids(segmentation_trimmed_3d, block_index_3d, nblocks_3d)

    # Extract non-zero IDs for this block from the *final* global 3D segmentation
    final_unique_ids_3d = np.unique(segmentation_global_3d)
    box_ids_for_this_block = final_unique_ids_3d[final_unique_ids_3d != 0]

    if test_mode:
        return (
            segmentation_global_3d,
            boxes,
            box_ids_for_this_block,
        )  # Return 3D results

    # --- Write 3D results to 3D output Zarr ---
    # print(f"Block {block_index}: Writing to output Zarr at {crop_trimmed_3d}")
    output_zarr[crop_trimmed_3d] = segmentation_global_3d  # 3D[3D_slice] = 3D array (This should work)

    # --- Calculate faces (3D) ---
    faces = block_faces(segmentation_global_3d)  # Faces are 3D

    # --- Write checkpoint ---
    if checkpoint_path is not None:
        append_checkpoint(
            checkpoint_path,
            block_index,
            distributed.get_worker().name,
            time.perf_counter() - start_time,
            int(np.max(segmentation_global_3d)),
        )

    # Return 3D faces, 3D boxes, and corresponding IDs
    return faces, boxes, box_ids_for_this_block


# ----------------------- component functions ---------------------------------#


def _build_packed_cellpose_model(model_kwargs: dict[str, Any]):
    """Instantiate a PackedCellpose model, preferring TensorRT plans when available."""
    import torch
    from cellpose.contrib.packed_infer import (
        PackedCellposeModel,
        PackedCellposeModelTRT,
        PackedCellposeUNetModel,
        PackedCellposeUNetModelTRT,
    )

    resolved_kwargs = dict(model_kwargs)
    backend = resolved_kwargs.pop("backend", "sam").lower()
    if backend not in {"sam", "unet"}:
        raise ValueError("backend must be either 'sam' or 'unet'.")

    pretrained_model = resolved_kwargs.get("pretrained_model")
    if pretrained_model is None:
        raise ValueError("model_kwargs must include 'pretrained_model'.")

    pretrained_path = Path(pretrained_model)
    resolved_kwargs["pretrained_model"] = str(pretrained_path)

    plan_selection: tuple[Path, str] | None = None
    if torch is not None and getattr(torch.cuda, "is_available", lambda: False)():
        if getattr(torch.cuda, "device_count", lambda: 0)() > 0:
            device_index = 0
            device_name = torch.cuda.get_device_name(device_index)
            plan_candidate = plan_path_for_device(pretrained_path, device_name)
            if plan_candidate.is_file():
                plan_selection = (plan_candidate, device_name)
    plan_candidate = plan_candidate if "plan_candidate" in locals() else plan_path_for_device(pretrained_path, "cuda")

    backend_to_classes = {
        "sam": (PackedCellposeModel, PackedCellposeModelTRT),
        "unet": (PackedCellposeUNetModel, PackedCellposeUNetModelTRT),
    }
    base_cls, trt_cls = backend_to_classes[backend]

    if plan_selection is not None:
        plan_path, device_name = plan_selection
        logger.info(f"Using TensorRT plan {plan_path.name} for CUDA device '{device_name}'")
        trt_kwargs = dict(resolved_kwargs)
        trt_kwargs["pretrained_model"] = str(plan_path)
        trt_kwargs.setdefault("gpu", True)
        return trt_cls(**trt_kwargs)

    raise FileNotFoundError(
        f"TensorRT plan required. Expected plan at {plan_candidate} for the current GPU."
    )

    if backend == "sam" and IS_CELLPOSE_SAM:
        resolved_kwargs.pop("pretrained_model_ortho", None)

    resolved_kwargs.setdefault("gpu", True)
    return base_cls(**resolved_kwargs)


def read_preprocess_and_segment(
    input_zarr: zarr.Array,
    crop: tuple[slice, ...],
    preprocessing_steps: list[tuple[Callable[..., NDArray[Any]], dict[str, Any]]],
    model_kwargs: dict[str, Any],
    eval_kwargs: dict[str, Any],
    worker_logs_directory: str | None,
) -> Annotated[NDArray[np.uint32], "Segmentation masks"]:
    """
    Read image block from zarr array, apply preprocessing pipeline, and run Cellpose segmentation.

    This function forms the core of the distributed segmentation pipeline. It handles loading
    a specific block of image data, applies user-defined preprocessing steps, and runs
    Cellpose segmentation on the processed block.

    Parameters
    ----------
    input_zarr : zarr.Array
        Input zarr array containing image data
    crop : tuple of slice
        Crop coordinates defining the block to process
    preprocessing_steps : list of tuple
        List of (function, kwargs) pairs for preprocessing pipeline
    model_kwargs : dict
        Arguments passed to PackedCellposeModel constructor
    eval_kwargs : dict
        Arguments passed to CellposeModel.eval method
    worker_logs_directory : str, optional
        Directory for worker log files, by default None

    Returns
    -------
    NDArray[np.uint32]
        Segmentation masks with uint32 labels

    Notes
    -----
    The preprocessing pipeline allows arbitrary image processing before segmentation.
    Each preprocessing function must accept an image array and a 'crop' keyword argument.
    """
    if preprocessing_steps is None:
        preprocessing_steps = []

    # Minimal per-worker GPU probe for visibility during bring-up
    try:
        import torch

        vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        dev = torch.cuda.current_device() if torch.cuda.is_available() else None
        logger.info(
            "Worker %s visible CUDA=%s torch_dev=%s",
            getattr(distributed.get_worker(), "name", "unknown"),
            vis,
            None if dev is None else int(dev),
        )
    except Exception:
        pass

    image = input_zarr[crop]
    for pp_step in preprocessing_steps:
        pp_step[1]["crop"] = crop
        image = pp_step[0](image, **pp_step[1])
    log_file = None
    if worker_logs_directory is not None:
        log_file = f"dask_worker_{distributed.get_worker().name}.log"
        log_file = pathlib.Path(worker_logs_directory).joinpath(log_file)
    cellpose.io.logger_setup(stdout_file_replacement=log_file)

    model = _build_packed_cellpose_model(model_kwargs)
    backend = model_kwargs.get("backend", "sam")
    filtered_eval_kwargs = dict(eval_kwargs)
    if backend == "unet":
        filtered_eval_kwargs.pop("ortho_weights", None)
    return model.eval(image, **filtered_eval_kwargs)[0].astype(np.uint32)


def _padded_from_raw_y(Ly_raw: int, *, bsize: int) -> int:
    """
    Compute padded Y size seen by Cellpose tiler for a given raw Ly.

    Uses the same padding rule as core.run_net via transforms.get_pad_yx.
    """
    ypad1, ypad2, _, _ = cp_transforms.get_pad_yx(Ly_raw, 0, min_size=(bsize, bsize))
    return Ly_raw + ypad1 + ypad2


def _padded_from_raw_x(Lx_raw: int, *, bsize: int) -> int:
    """
    Compute padded X size seen by Cellpose tiler for a given raw Lx.

    Uses the same padding rule as core.run_net via transforms.get_pad_yx.
    """
    _, _, xpad1, xpad2 = cp_transforms.get_pad_yx(0, Lx_raw, min_size=(bsize, bsize))
    return Lx_raw + xpad1 + xpad2


def _find_raw_for_target_tiles(
    n_target: int,
    *,
    bsize: int = 256,
    tile_overlap: float = 0.1,
    axis: str,
    search_margin: int = 512,
) -> tuple[int, int]:
    """
    Find the largest raw size whose padded size yields the desired tile count.

    Returns (raw_size, padded_size) along the requested axis.
    """
    if n_target <= 0:
        raise ValueError("n_target must be positive")

    alpha = 1.0 + 2.0 * tile_overlap

    if n_target == 1:
        padded_min = 0
        padded_max = bsize
    else:
        padded_min = int(np.floor((n_target - 1) * bsize / alpha)) + 1
        padded_max = int(np.floor(n_target * bsize / alpha))
        if padded_min <= bsize:
            padded_min = bsize + 1

    if padded_max < padded_min:
        raise ValueError(f"No padded size range for n_target={n_target}")

    if axis == "y":
        mapper = _padded_from_raw_y
    elif axis == "x":
        mapper = _padded_from_raw_x
    else:
        raise ValueError("axis must be 'y' or 'x'")

    raw_upper = max(bsize, padded_max + search_margin)
    best_raw: int | None = None
    best_padded: int | None = None

    for raw in range(1, raw_upper + 1):
        padded = mapper(raw, bsize=bsize)
        if padded_min <= padded <= padded_max:
            if best_raw is None or raw > best_raw:
                best_raw = raw
                best_padded = padded

    if best_raw is None or best_padded is None:
        raise ValueError(f"Could not find raw size for n_target={n_target} along axis={axis}")

    return best_raw, best_padded


def solve_internal_xy_for_tiles(
    ny_target: int,
    nx_target: int,
    *,
    bsize: int = 256,
    tile_overlap: float = 0.1,
) -> tuple[int, int]:
    """
    Solve for effective internal (Ly, Lx) that yield desired (ny, nx) tiles.

    The returned Ly/Lx correspond to the dimensions seen by Cellpose's tiler
    after padding and any diameter rescaling. This is exactly the geometry
    that run_net operates on.
    """
    if ny_target <= 0 or nx_target <= 0:
        raise ValueError("ny_target and nx_target must be positive")

    Ly_raw, _ = _find_raw_for_target_tiles(
        ny_target,
        bsize=bsize,
        tile_overlap=tile_overlap,
        axis="y",
    )
    Lx_raw, _ = _find_raw_for_target_tiles(
        nx_target,
        bsize=bsize,
        tile_overlap=tile_overlap,
        axis="x",
    )
    return Ly_raw, Lx_raw


######################## Distributed Cellpose #################################


# ----------------------- The main function -----------------------------------#
@cluster
def distributed_eval(
    input_zarr: zarr.Array,
    blocksize: tuple[int, ...],
    write_path: Path | str,
    mask: NDArray[Any] | None = None,
    preprocessing_steps: list[tuple[Callable[..., NDArray[Any]], dict[str, Any]]] | None = None,
    model_kwargs: dict[str, Any] | None = None,
    eval_kwargs: dict[str, Any] | None = None,
    cluster: myLocalCluster | None = None,
    cluster_kwargs: dict[str, Any] | None = None,
    temporary_directory: Path | None = None,
) -> tuple[zarr.Array, list[tuple[slice, ...]]]:
    """
    Evaluate a cellpose model on overlapping blocks of a big image.
    Distributed over workstation or cluster resources with Dask.
    Optionally run preprocessing steps on the blocks before running cellpose.
    Optionally use a mask to ignore background regions in image.
    Either cluster or cluster_kwargs parameter must be set to a
    non-default value; please read these parameter descriptions below.
    If using cluster_kwargs, the workstation and Janelia LSF cluster cases
    are distinguished by the arguments present in the dictionary.

    PC/Mac/Linux workstations and the Janelia LSF cluster are supported;
    running on a different institute cluster will require implementing your
    own dask cluster class. Look at the JaneliaLSFCluster class in this
    module as an example, also look at the dask_jobqueue library. A PR with
    a solid start is the right way to get help running this on your own
    institute cluster.

    If running on a workstation, please read the docstring for the
    LocalCluster class defined in this module. That will tell you what to
    put in the cluster_kwargs dictionary. If using the Janelia cluster,
    please read the docstring for the janeliaLSFCluster class in this module.

    Parameters
    ----------
    input_zarr : zarr.core.Array
        A zarr.core.Array instance containing the image data you want to
        segment.

    blocksize : iterable
        The size of blocks in voxels. E.g. [128, 256, 256]

    write_path : string
        The location of a zarr file on disk where you'd like to write your results

    mask : numpy.ndarray (default: None)
        A foreground mask for the image data; may be at a different resolution
        (e.g. lower) than the image data. If given, only blocks that contain
        foreground will be processed. This can save considerable time and
        expense. It is assumed that the domain of the input_zarr image data
        and the mask is the same in physical units, but they may be on
        different sampling/voxel grids.

    preprocessing_steps : list of tuples (default: the empty list)
        Optionally apply an arbitrary pipeline of preprocessing steps
        to the image blocks before running cellpose.

        Must be in the following format:
        [(f, {'arg1':val1, ...}), ...]
        That is, each tuple must contain only two elements, a function
        and a dictionary. The function must have the following signature:
        def F(image, ..., crop=None)
        That is, the first argument must be a numpy array, which will later
        be populated by the image data. The function must also take a keyword
        argument called crop, even if it is not used in the function itself.
        All other arguments to the function are passed using the dictionary.
        Here is an example:

        def F(image, sigma, crop=None):
            return gaussian_filter(image, sigma)
        def G(image, radius, crop=None):
            return median_filter(image, radius)
        preprocessing_steps = [(F, {'sigma':2.0}), (G, {'radius':4})]

    model_kwargs : dict (default: {})
        Arguments passed to PackedCellposeModel

    eval_kwargs : dict (default: {})
        Arguments passed to PackedCellposeModel.eval

    cluster : A dask cluster object (default: None)
        Only set if you have constructed your own static cluster. The default
        behavior is to construct a dask cluster for the duration of this function,
        then close it when the function is finished.

    cluster_kwargs : dict (default: {})
        Arguments used to parameterize your cluster.
        If you are running locally, see the docstring for the myLocalCluster
        class in this module. If you are running on the Janelia LSF cluster, see
        the docstring for the janeliaLSFCluster class in this module. If you are
        running on a different institute cluster, you may need to implement
        a dask cluster object that conforms to the requirements of your cluster.

    temporary_directory : string (default: None)
        Temporary files are created during segmentation. The temporary files
        will be in their own folder within the temporary_directory. The default
        is the current directory. Temporary files are removed if the function
        completes successfully.

    Returns
    -------
    Two values are returned:
    (1) A reference to the zarr array on disk containing the stitched cellpose
        segments for your entire image
    (2) Bounding boxes for every segment. This is a list of tuples of slices:
        [(slice(z1, z2), slice(y1, y2), slice(x1, x2)), ...]
        The list is sorted according to segment ID. That is the smallest segment
        ID is the first tuple in the list, the largest segment ID is the last
        tuple in the list.
    """
    # Handle default parameters
    if preprocessing_steps is None:
        preprocessing_steps = []
    if model_kwargs is None:
        model_kwargs = {}
    if eval_kwargs is None:
        eval_kwargs = {}
    if cluster_kwargs is None:
        cluster_kwargs = {}

    # Derive a stable base directory for artifacts/logs
    if temporary_directory is None:
        base_parent = Path(write_path).parent if isinstance(write_path, (str, Path)) else Path.cwd()
        temporary_directory = base_parent / "cellpose_temp"

    timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    worker_logs_dirname = f"dask_worker_logs_{timestamp}"
    base_dir = Path(temporary_directory).parent
    worker_logs_dir = base_dir / worker_logs_dirname
    worker_logs_dir.mkdir(parents=True, exist_ok=True)

    if "diameter" not in eval_kwargs.keys():
        raise ValueError("Diameter must be set in eval_kwargs")

    overlap = eval_kwargs["diameter"] * 2
    block_indices, block_crops = get_block_crops(input_zarr.shape, blocksize, overlap, mask)
    assert cluster is not None

    def check_block_has_data(crop: tuple[slice, ...], zarr_array: zarr.Array, threshold: int = 0) -> bool:
        """
        Checks if a given crop in a Zarr array contains any data above a threshold.
        Returns True if data exists, False otherwise.
        """
        try:
            data_slice = zarr_array[crop]
            if threshold == 0:
                return data_slice.any()
            else:
                return (data_slice > threshold).any()
        except Exception as e:
            print(f"Error checking block crop {crop}: {e}")
            return False

    # Optional: GPU preflight probe to confirm worker pinning
    try:
        probe = cluster.client.run(_gpu_probe)
        logger.info("GPU probe results: %s", probe)
    except Exception as e:
        logger.warning("GPU probe failed: %s", e)


    offset = 0
    n = None

    path_nonempty = Path(write_path).parent / "nonempty.json"
    idxs = read_nonempty_cache(path_nonempty, blocksize)
    if idxs is not None:
        logger.info("Loaded cached non-empty block indices (%d entries).", len(idxs))
    else:
        check_futures = cluster.client.map(
            check_block_has_data,
            block_crops[offset : None if n is None else offset + n],
            zarr_array=input_zarr,
            threshold=1,
        )

        total_tiles = len(check_futures)
        logger.info("Checking non-zero blocks: 0/%d", total_tiles)
        try:
            with progress_bar(total_tiles) as submit:
                [fut.add_done_callback(submit) for fut in check_futures]

                non_zero_results = cluster.client.gather(check_futures, errors="raise")
            logger.info("Checked non-zero blocks: %d/%d", total_tiles, total_tiles)
        except Exception as e:
            logger.critical(f"Error gathering input check results: {e}")
            logger.critical("Skipping zero-input block filtering due to error.")
            raise e

        idxs = [i for i, is_non_zero in enumerate(non_zero_results, offset) if is_non_zero]
        write_nonempty_cache(path_nonempty, blocksize, idxs)
        logger.info("Persisted %d non-empty block indices to %s", len(idxs), path_nonempty)

    # final_block_indices, final_block_crops = [], []
    final_block_indices, final_block_crops = (
        [block_indices[i] for i in idxs],
        [block_crops[i] for i in idxs],
    )
    del block_indices, block_crops

    print(f"Selected {len(final_block_indices)} blocks with non-zero input data.")

    output_shape = input_zarr.shape[:-1]  # Get ZYX dimensions
    output_blocksize = blocksize[:-1]

    Path(temporary_directory).mkdir(parents=True, exist_ok=True)
    assert temporary_directory.exists()
    temp_zarr_path = Path(temporary_directory) / "segmentation_unstitched.zarr"
    checkpoint_path = Path(temporary_directory) / "checkpoint.jsonl"
    run_config_path = Path(temporary_directory) / "run_config.json"

    # Detect resume vs fresh start
    is_resume = run_config_path.exists()

    if is_resume:
        if not temp_zarr_path.exists():
            raise RuntimeError(f"Cannot resume: temp_zarr missing at {temp_zarr_path}")
        validate_run_config(
            run_config_path,
            model_kwargs,
            eval_kwargs,
            blocksize,
            input_zarr.shape,
            overlap,
            preprocessing_steps,
        )
        completed_indices = load_checkpoint(checkpoint_path)
        logger.info(f"Resuming: {len(completed_indices)} of {len(final_block_indices)} blocks already completed")
    else:
        save_run_config(
            run_config_path,
            model_kwargs,
            eval_kwargs,
            blocksize,
            input_zarr.shape,
            overlap,
            preprocessing_steps,
        )
        completed_indices = set()
        logger.info(f"Fresh run - saving config to {run_config_path}")

    # Filter to remaining blocks
    remaining_block_indices = []
    remaining_block_crops = []
    for idx, crop in zip(final_block_indices, final_block_crops):
        if tuple(idx) not in completed_indices:
            remaining_block_indices.append(idx)
            remaining_block_crops.append(crop)

    logger.info(f"Blocks to process: {len(remaining_block_indices)} (skipped {len(completed_indices)} already completed)")

    zarr.config.set({"array.target_shard_size_bytes": "10MB"})
    temp_zarr = zarr.open(
        temp_zarr_path,
        mode="r+" if is_resume else "w",
        shape=output_shape,  # Use 3D shape
        chunks=output_blocksize,  # Use 3D chunks
        dtype=np.uint32,
    )

    if not remaining_block_indices:
        logger.info("All blocks already completed, proceeding to merge")
    else:
        futures = cluster.client.map(
            process_block,
            remaining_block_indices,
            remaining_block_crops,
            input_zarr=input_zarr,
            preprocessing_steps=preprocessing_steps,
            model_kwargs=model_kwargs,
            eval_kwargs=eval_kwargs,
            blocksize=blocksize,
            overlap=overlap,
            output_zarr=temp_zarr,
            worker_logs_directory=str(worker_logs_dir),
            checkpoint_path=checkpoint_path,
        )

        with progress_bar(len(remaining_block_indices)) as submit:
            [fut.add_done_callback(submit) for fut in futures]
            cluster.client.gather(futures)

    # Reconstruct faces/boxes from temp_zarr for ALL blocks (handles resume case)
    logger.info("Computing faces and bounding boxes from temp_zarr...")
    results = []
    for block_idx, block_crop in zip(final_block_indices, final_block_crops):
        # Get the trimmed crop (without overlap)
        spatial_crop = block_crop[:-1]  # ZYX slices
        spatial_blocksize = blocksize[:-1]
        trimmed_crop = []
        for axis, (slc, bs) in enumerate(zip(spatial_crop, spatial_blocksize)):
            start = slc.start if slc.start == 0 else slc.start + overlap
            stop = min(start + bs, slc.stop)
            trimmed_crop.append(slice(start, stop))
        trimmed_crop = tuple(trimmed_crop)

        seg_block = temp_zarr[trimmed_crop]
        faces = block_faces(seg_block)
        boxes = bounding_boxes_in_global_coordinates(seg_block, trimmed_crop)
        unique_ids = np.unique(seg_block)
        box_ids = unique_ids[unique_ids != 0]
        results.append((faces, boxes, box_ids))

    if isinstance(cluster, dask_jobqueue.core.JobQueueCluster):
        cluster.scale(0)

    faces, boxes_, box_ids_ = list(zip(*results))
    boxes = [box for sublist in boxes_ for box in sublist]
    box_ids = np.concatenate(box_ids_).astype(int)  # unsure how but without cast these are float64

    print(f"Box IDs: {len(box_ids)}")
    new_labeling = determine_merge_relabeling(
        [(bi[0], bi[1], bi[2]) for bi in final_block_indices], faces, box_ids
    )
    new_labeling_path = Path(temporary_directory) / "new_labeling.npy"
    np.save(new_labeling_path, new_labeling)

    # stitching step is cheap, we should release gpus and use small workers
    if isinstance(cluster, dask_jobqueue.core.JobQueueCluster):
        cluster.change_worker_attributes(
            min_workers=cluster.locals_store["min_workers"],
            max_workers=12,
            ncpus=1,
            memory="32GB",
            mem=int(32e9),
            queue=None,
            job_extra_directives=[],
        )
        cluster.scale(32)

    print("Relabeling blocks...")
    relabel_and_write(temp_zarr, new_labeling_path, write_path)

    print("Merging boxes...")
    merged_boxes = merge_all_boxes(boxes, new_labeling[box_ids])
    return zarr.open(write_path, mode="r"), merged_boxes


app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def main(
    path: Path = typer.Argument(..., help="Path to the folder containing the zarr file."),
    channels: str | None = typer.Option(None, help="Comma-separated list of channel names to use."),
    overwrite: bool = typer.Option(False, help="Overwrite existing segmentation."),
    config_path: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Explicit path to config.json. Defaults to <path>/../config.json when omitted.",
    ),
    workers_per_gpu: int = typer.Option(
        2, help="Number of workers to spawn per GPU (>=2 enables multi-worker SpecCluster)"
    ),
    threads_per_worker: int = typer.Option(1, help="Threads per worker (GPU-bound work typically uses 1)"),
    use_localcuda: bool = typer.Option(
        False, help="If true and workers_per_gpu<=1, use dask-cuda LocalCUDACluster"
    ),
    n_workers: int | None = typer.Option(
        None, help="For LocalCUDACluster: number of workers (defaults to #GPUs)"
    ),
    target_ny: int | None = typer.Option(
        None,
        help="Desired internal Cellpose ny tiles (SAM backend only; overrides default 4).",
    ),
    target_nx: int | None = typer.Option(
        None,
        help="Desired internal Cellpose nx tiles (SAM backend only; overrides default 6).",
    ),
) -> None:
    """
    Main CLI entry point for distributed Cellpose segmentation.

    Orchestrates the complete distributed segmentation workflow including:
    - Loading and validating input data
    - Setting up distributed computing cluster
    - Running preprocessing and segmentation
    - Post-processing and result saving

    Parameters
    ----------
    path : Path
        Path to folder containing 'fused.zarr' input file and 'config.json'
    channels : str
        Comma-separated channel names to use for segmentation
    overwrite : bool, optional
        Whether to overwrite existing segmentation results, by default False

    Notes
    -----
    Expected folder structure:
    - path/fused.zarr (input image data)
    - path/config.json (segmentation parameters)
    - path/normalization.json (auto-generated if missing)

    Output files:
    - path/output_segmentation-sam.zarr or path/output_segmentation-unet.zarr (segmentation results, backend-dependent)
    - path/segmentation.done (completion marker)
    - path/cellpose_temp/ (temporary processing files)
    """
    # Normalize input paths:
    # - If given a directory named '*.zarr' → treat as the Zarr store
    # - If given a parent directory → expect 'fused.zarr' inside it
    input_path = Path(path)
    if input_path.suffix == ".zarr" and input_path.exists():
        zarr_input_path = input_path
        base_dir = input_path.parent
    elif input_path.is_dir():
        base_dir = input_path
        zarr_input_path = base_dir / "fused.zarr"
        if not zarr_input_path.exists():
            raise FileNotFoundError(f"Expected 'fused.zarr' in {base_dir} but it was not found.")
    else:
        raise FileNotFoundError(f"Path {input_path} must be a directory or a '.zarr' store.")

    # Short-circuit if results already present (unless overwrite)
    if not overwrite and (base_dir / "segmentation.done").exists():
        logger.warning("Segmentation already exists. Exiting.")
        exit()

    IS_CELLPOSE_SAM = version("cellpose").startswith("4.")
    if not IS_CELLPOSE_SAM:
        raise RuntimeError("This script requires Cellpose version 4.x for SAM backend support.")

    # ---- 1. Configuration ----
    if config_path is None:
        config_path = base_dir.parent / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found at {config_path}")

    config = json.loads(config_path.read_text())
    backend = config.get("backend", "sam").lower()
    if backend not in {"sam", "unet"}:
        raise ValueError("backend must be one of {'sam', 'unet'}.")
    using_sam_backend = backend == "sam"

    zarr_output_path = base_dir / f"output_segmentation-{backend}.zarr"
    temporary_directory = base_dir / "cellpose_temp"

    # Clear temp directory if --overwrite is set (forces fresh start, no resume)
    if overwrite and temporary_directory.exists():
        logger.info(f"--overwrite: removing existing temp directory {temporary_directory}")
        shutil.rmtree(temporary_directory)

    (base_dir / "segmentation.done").unlink(missing_ok=True)
    ortho_weights = config.get("ortho_weights", [4, 1.0, 1.0])
    diameter = config.get("diameter", 25)
    cellpose_model_kwargs = {
        "pretrained_model": config["pretrained_model"],  # Or 'nuclei', 'cyto', or path to custom model
        "gpu": True,
        "backend": backend,
    }
    if using_sam_backend and IS_CELLPOSE_SAM:
        ...
    else:
        cellpose_model_kwargs["pretrained_model_ortho"] = config.get("pretrained_model_ortho", None)

    # Dask Cluster Configuration (always GPU-backed)
    local_cluster_kwargs = {
        "workers_per_gpu": int(8 if backend == "unet" else workers_per_gpu),
        "threads_per_worker": int(threads_per_worker),
    }
    if use_localcuda and workers_per_gpu <= 1:
        local_cluster_kwargs.update({
            "use_localcuda": True,
            "n_workers": n_workers,
        })

    preprocessing_pipeline = [(unsharp_all, {})]

    mask_path = None
    foreground_mask = None
    if mask_path and os.path.exists(mask_path):
        print(f"Loading mask from {mask_path}")
        foreground_mask = tifffile.imread(mask_path) > 1.0

    input_zarr_array = zarr.open_array(zarr_input_path, mode="r")

    key = cast(str, input_zarr_array.attrs["key"])
    if channels is None:
        channels_list = list(range(1, input_zarr_array.shape[3] + 1))  # All channels
        logger.info(f"No channels specified. Using all channels: {channels_list}")
    else:
        try:
            channels_list = [key.index(c) + 1 for c in channels.split(",")]
            logger.info(f"Using channels: {channels}")
            if len(channels_list) != len(key):
                logger.warning(
                    f"Selected channels {channels_list} do not match total channels {list(range(1, input_zarr_array.shape[3] + 1))}"
                )
                input("Do you want to continue? Press Enter to continue or Ctrl+C to abort.")
        except ValueError:
            raise ValueError(f"Channel names {channels} not found in {key}")
    del channels

    if backend == "unet":
        processing_blocksize = (
            input_zarr_array.shape[0],
            224,
            224 * 4,
            len(channels_list),
        )  # ZYXc tuned for UNet
    else:
        # Determine effective internal tiling size for SAM backend.
        ny_target = target_ny if target_ny is not None else 2
        nx_target = target_nx if target_nx is not None else 6
        Ly_internal, Lx_internal = solve_internal_xy_for_tiles(
            ny_target,
            nx_target,
            bsize=256,
            tile_overlap=0.1,
        )
        # Map internal (post-diameter-rescale) back to raw block size.
        scale_back = float(diameter) / 30.0
        by = int(Ly_internal * scale_back)
        bx = int(Lx_internal * scale_back)
        processing_blocksize = (
            input_zarr_array.shape[0],
            by,
            bx,
            len(channels_list),
        )  # ZYXc tuned for SAM
        logger.info(f"SAM backend: target tiles (ny={ny_target}, nx={nx_target}) → internal size ({Ly_internal}x{Lx_internal}) → blocksize ({by}x{bx})")
    normalization_path = base_dir / "normalization.json"
    normalization = read_normalization_cache(normalization_path)
    if normalization is not None:
        logger.info("Loaded cached normalization thresholds from %s", normalization_path)
    else:
        logger.info("Calculating normalization percentiles (cache miss).")
        perc, _ = sample_percentile(
            input_zarr_array,
            channels=channels_list,
            block=(256, 1024),
            n=30,
            low=1,
            high=99.9,
        )
        normalization = {"lowhigh": perc}
        write_normalization_cache(normalization_path, normalization)
        logger.info("Saved normalization thresholds to %s", normalization_path)

    if len(channels_list) == 1:
        channels_list = [channels_list[0], channels_list[0]]
        normalization = {"lowhigh": [normalization["lowhigh"], normalization["lowhigh"]]}

    if using_sam_backend and IS_CELLPOSE_SAM and len(channels_list) < 3:
        print(np.repeat([[0], [1]], repeats=3 - len(channels_list), axis=0).shape)
        normalization = {
            "lowhigh": np.concatenate(
                [
                    normalization["lowhigh"],
                    np.repeat([[0, 1]], repeats=3 - len(channels_list), axis=0),
                ],
                axis=0,
            )
        }

    logger.info(f"Normalization params: {normalization}")

    # Cellpose Evaluation Configuration
    cellpose_eval_kwargs = {
        "diameter": config.get("diameter", 25),  # MUST BE INT
        "batch_size": 16 if backend == "unet" else 4,
        "normalize": normalization,
        "flow_threshold": 0,  # Ignored in 3D
        "cellprob_threshold": 0,  # Default is 0.0, adjust if needed
        "anisotropy": 2.0,
        "resample": False,
        "flow3D_smooth": 0.5,
        "niter": 1000,
        "do_3D": True,
        "min_size": 200,
        "channel_axis": 3,
    }
    if using_sam_backend:
        cellpose_eval_kwargs["z_axis"] = 0
        if not IS_CELLPOSE_SAM:
            cellpose_eval_kwargs["channels"] = channels_list
    else:
        cellpose_eval_kwargs["channels"] = channels_list
    if ortho_weights is not None:
        cellpose_eval_kwargs["ortho_weights"] = ortho_weights

    # ---- 3. Run Distributed Evaluation ----
    logger.info("Starting distributed Cellpose evaluation…")
    try:
        # The @cluster decorator handles cluster creation/management
        final_segmentation_zarr, final_bounding_boxes = distributed_eval(
            input_zarr=input_zarr_array,
            blocksize=processing_blocksize,
            write_path=zarr_output_path,
            mask=foreground_mask,
            preprocessing_steps=preprocessing_pipeline,
            model_kwargs=cellpose_model_kwargs,
            eval_kwargs=cellpose_eval_kwargs,
            cluster_kwargs=local_cluster_kwargs,
            temporary_directory=temporary_directory,
        )
        (zarr_output_path.parent / "segmentation.done").touch()
        shutil.rmtree(temporary_directory)
        logger.info("Run Finished")
        logger.info(f"Final segmentation saved to: {zarr_output_path}")
        logger.info(
            f"Output Zarr shape: {final_segmentation_zarr.shape}, dtype: {final_segmentation_zarr.dtype}"
        )
        logger.info(f"Number of segmented objects found: {len(final_bounding_boxes)}")

    except Exception as e:
        logger.exception("Error during distributed evaluation")
        raise e


if __name__ == "__main__":
    app()
