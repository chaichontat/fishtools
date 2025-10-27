# %%
import json
import multiprocessing as mp
import os
import pickle
import random
import re
import shutil
import subprocess
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import timedelta
from itertools import chain, groupby
from pathlib import Path
from typing import Any, Literal, Mapping, Optional, Sequence, cast

import cupy as cp
import matplotlib

# disable memory pool for predictable memory usage
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import rich_click as click
import starfish
import tifffile
import xarray as xr
from loguru import logger
from numpy.typing import NDArray  # noqa: F401
from pydantic import BaseModel, TypeAdapter, field_validator
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from starfish import ImageStack, IntensityTable
from starfish.core.intensity_table.decoded_intensity_table import DecodedIntensityTable
from starfish.core.intensity_table.intensity_table_coordinates import (
    transfer_physical_coords_to_intensity_table,
)
from starfish.core.spots.DetectPixels.combine_adjacent_features import (
    CombineAdjacentFeatures,
)
from starfish.core.types import Axes, Coordinates, CoordinateValue
from starfish.experiment.builder import FetchedTile, TileFetcher
from starfish.image import Filter
from starfish.spots import DecodeSpots, FindSpots
from starfish.types import Features, Levels
from starfish.util.plot import imshow_plane, intensity_histogram
from tifffile import TiffFile

matplotlib.use("Agg", force=True)
if hasattr(cp, "cuda") and hasattr(cp.cuda, "set_allocator"):
    cp.cuda.set_allocator(None)
else:  # CPU-only environments expose a stub without allocator control
    logger.warning("CuPy CUDA allocator unavailable; continuing without GPU allocator reset.")

from fishtools.gpu.memory import release_all as _gpu_release_all
from fishtools.io.workspace import CorruptedTiffError, Workspace, safe_imwrite
from fishtools.preprocess.addition import ElementWiseAddition
from fishtools.preprocess.cli_spotlook import threshold
from fishtools.preprocess.config import (
    OPTIMIZE_DECODE,
    SpotDecodeConfig,
)
from fishtools.preprocess.config_loader import load_config
from fishtools.preprocess.spots.align_batchoptimize import optimize
from fishtools.preprocess.spots.illum_field_correction import (
    FieldContext,
    correct_channel_with_field,
)
from fishtools.preprocess.spots.illum_field_correction import (
    slice_field_ds_for_tile as _slice_field_ds_for_tile,
)
from fishtools.preprocess.spots.overlay_spots import overlay
from fishtools.preprocess.spots.stitch_spot_prod import stitch
from fishtools.utils.logging import setup_cli_logging
from fishtools.utils.plot import plot_all_genes
from fishtools.utils.pretty_print import progress_bar_threadpool
from fishtools.utils.utils import git_hash

GPU = os.environ.get("GPU", "1") == "1"
if GPU:
    logger.info("Using GPU")
    from fishtools.gpu.codebook import Codebook
else:
    from starfish import Codebook


os.environ["TQDM_DISABLE"] = "1"


def _field_store_path(ws: Workspace, roi: str, codebook_label: str) -> Path:
    slug = Workspace.sanitize_codebook_name(codebook_label)
    base = ws.path / "analysis" / "deconv" / f"fields+{slug}"
    return base / f"field--{roi}+{slug}.zarr"


def _discover_field_store(ws: Workspace, roi: str, codebook_label: str) -> Path:
    candidate = _field_store_path(ws, roi, codebook_label)
    if candidate.is_dir():
        return candidate
    raise FileNotFoundError(
        f"Illumination field store not found for ROI '{roi}' and codebook '{codebook_label}'. "
        "Run 'preprocess correct-illum export-field' to generate it."
    )


def _ensure_field_stores(ws: Workspace, rois: Sequence[str], codebook_label: str) -> None:
    missing: list[str] = []
    for roi in rois:
        try:
            _discover_field_store(ws, roi, codebook_label)
        except FileNotFoundError:
            missing.append(roi)
    if missing:
        joined = ", ".join(sorted(set(missing)))
        raise click.ClickException(
            f"Missing illumination field store(s) for ROI(s): {joined}. "
            "Run 'preprocess correct-illum export-field' before using --field-correct."
        )


def _build_field_context(
    path: Path,
    img: np.ndarray,
    *,
    workspace_root: Optional[Path] = None,
    x_off: int = 0,
    y_off: int = 0,
) -> FieldContext:
    m = re.match(r"^registered--(.+)\+(.*)$", path.parent.name)
    if m is None:
        raise click.ClickException(
            f"Path {path.parent.name} does not match expected format 'registered--<roi>+<codebook>'"
        )
    roi, codebook = m.groups()

    ws = Workspace(workspace_root) if workspace_root is not None else Workspace(path.parent.parent.parent)
    tc = ws.tileconfig(roi)

    def _tile_index_from_path(tile_path: Path) -> int:
        stem = tile_path.stem
        try:
            return int(stem.split("-")[1])
        except Exception:
            return int(re.sub(r"\D+", "", stem))

    tile_index = _tile_index_from_path(path)
    row = tc.df.filter(pl.col("index") == int(tile_index))
    if row.height == 0:
        raise click.ClickException(f"Tile index {tile_index} not found for ROI '{roi}' in TileConfiguration")
    tile_x0 = float(row[0, "x"])  # type: ignore[index]
    tile_y0 = float(row[0, "y"])  # type: ignore[index]

    try:
        with TiffFile(path) as tif_handle:
            metadata = tif_handle.shaped_metadata[0]
            maybe_key = metadata.get("key") if isinstance(metadata, dict) else None
            if isinstance(maybe_key, (list, tuple)):
                channel_labels = [str(ch) for ch in maybe_key]
            elif isinstance(maybe_key, str):
                channel_labels = [maybe_key]
            else:
                channel_labels = [str(i) for i in range(img.shape[1])]
    except Exception:
        channel_labels = [str(i) for i in range(img.shape[1])]

    try:
        store_path = _discover_field_store(ws, roi, codebook)
    except FileNotFoundError as exc:
        raise click.ClickException(str(exc))

    try:
        import zarr  # local import
    except Exception as exc:  # pragma: no cover
        raise click.ClickException(f"zarr is required for --field-correct: {exc}")

    za = zarr.open_array(str(store_path), mode="r")
    attrs = getattr(za, "attrs", {})
    axes_attr = attrs.get("axes")
    if za.ndim != 4:
        raise click.ClickException("Discovered field store must be TCYX (axes='TCYX').")
    t_labels = attrs.get("t_labels")
    try:
        t_low = 0
        t_range = 1
    except ValueError:
        raise click.ClickException("TCYX store must contain 'low' and 'range' planes in t_labels")
    model_meta = dict(attrs.get("model_meta", {})) if hasattr(attrs, "get") else {}
    ds = 4  # int(model_meta.get("downsample", 1) or 1)
    fx0 = float(model_meta["x0"])
    fy0 = float(model_meta["y0"])
    field_channels = model_meta["channels"]
    if isinstance(field_channels, (list, tuple)):
        c_index_map = {
            i: (field_channels.index(channel_labels[i]) if channel_labels[i] in field_channels else i)
            for i in range(len(channel_labels))
        }
    else:
        c_index_map = {i: i for i in range(len(channel_labels))}

    low_scale = 65535.0

    return FieldContext(
        enabled=True,
        field_arr=za,
        t_low=t_low,
        t_range=t_range,
        ds=int(ds),
        fx0=float(fx0),
        fy0=float(fy0),
        tile_x0=float(tile_x0),
        tile_y0=float(tile_y0),
        x_off=int(x_off),
        y_off=int(y_off),
        tile_h=int(img.shape[-2]),
        tile_w=int(img.shape[-1]),
        c_index_map={int(k): int(v) for k, v in c_index_map.items()},
        use_gpu=False,
        low_scale=float(low_scale),
    )


def make_fetcher(
    path: Path,
    raw: np.ndarray,
    sl: slice | tuple | list[int] = np.s_[:],
    *,
    field_correct: bool = False,
    workspace_root: Optional[Path] = None,
) -> ImageStack:
    """Create an ImageStack backed by a demo tile fetcher.

    Behavior
    --------
    - Loads the tile TIFF and exposes it to Starfish via a lightweight
      FetchedTile implementation (DemoFetchedTile).
    - When ``field_correct`` is True, discovers the TCYX illumination field
      for the ROI/codebook inside the workspace and applies the correction
      to each returned YX plane before Starfish sees it:
        corrected = max(plane - low, 0) * range
      where (low, range) are sliced from the TCYX store at the tile origin,
      padded outside with identity=1.0, then upsampled and applied on GPU.

    Coordinate alignment
    --------------------
    - Uses the registered tile origin (TileConfiguration) and optional local
      XY offsets derived from ``sl`` (e.g., quadrant slicing) to index into
      the field store.
    - Relies on field metadata (model_meta.x0/y0, downsample) stored in the
      TCYX Zarr attributes for correct alignment.

    Requirements
    ------------
    - TCYX stores are expected under ``analysis/deconv/fields+{codebook}``
      with the filename ``field--{roi}+{codebook}.zarr`` produced by
      ``preprocess correct-illum export-field --what both``.
    """

    # Resolve XY slice offsets from sl relative to the native tile
    def _norm_slice(s: slice, dim: int) -> tuple[int, int]:
        start = 0 if s.start is None else (dim + s.start if s.start < 0 else s.start)
        stop = dim if s.stop is None else (dim + s.stop if s.stop < 0 else s.stop)
        if start < 0:
            start = 0
        if stop > dim:
            stop = dim
        if stop < start:
            stop = start
        return int(start), int(stop - start)

    y_off = 0
    x_off = 0
    if isinstance(sl, tuple) and len(sl) >= 4:
        y_slice = sl[-2] if isinstance(sl[-2], slice) else slice(None)
        x_slice = sl[-1] if isinstance(sl[-1], slice) else slice(None)
        y_off, _ = _norm_slice(y_slice, raw.shape[-2])
        x_off, _ = _norm_slice(x_slice, raw.shape[-1])

    # Prepare field correction context if enabled
    field_ctx: FieldContext | None = None
    if field_correct:
        field_ctx = _build_field_context(
            path,
            raw,
            workspace_root=workspace_root,
            x_off=int(x_off),
            y_off=int(y_off),
        )

        low_ds, rng_ds = _slice_field_ds_for_tile(
            field_ctx.field_arr,
            field_ctx.t_low,
            field_ctx.t_range,
            int(field_ctx.ds),
            int(field_ctx.tile_w),
            int(field_ctx.tile_h),
            float(field_ctx.tile_x0) + float(field_ctx.x_off),
            float(field_ctx.tile_y0) + float(field_ctx.y_off),
            float(field_ctx.fx0),
            float(field_ctx.fy0),
        )

        logger.info("Applying field correction on all channels")
        raw = raw[sl]

        img = correct_channel_with_field(
            raw,
            low_ds,
            rng_ds,
            int(field_ctx.ds),
            use_gpu=False,
            normalize=True,
            sl=sl,
        )
        logger.info("Field correction applied.")
        # return np.clip(corrected_stack, 0.0, 1.0).astype(np.float32, copy=False)

    else:
        # Apply the actual slice, normalize to [0,1] float32
        img = (raw[sl] / 65535.0).astype(np.float32, copy=False)
        img = np.clip(img, 0.0, 1.0)

    del raw

    # print(type(img), img.shape, img.dtype)

    class DemoFetchedTile(FetchedTile):
        def __init__(
            self,
            img: np.ndarray,
            z: int,
            chs: int,
            *,
            field_ctx: Optional[FieldContext] = None,
        ):
            self.img = img
            self.z = z
            self.c = chs
            self._field_ctx = field_ctx
            self._corrected_stack: np.ndarray | None = None

        @property
        def shape(self) -> Mapping[Axes, int]:
            return {
                Axes.Y: self.img.shape[2],
                Axes.X: self.img.shape[3],
            }

        @property
        def coordinates(self) -> Mapping[str | Coordinates, CoordinateValue]:
            return {
                Coordinates.X: (0, 0.001),
                Coordinates.Y: (0, 0.001),
                Coordinates.Z: (0.001 * self.z, 0.001 * (self.z + 1)),
            }

        def tile_data(self) -> np.ndarray:
            return self.img[self.z, self.c]

    class DemoTileFetcher(TileFetcher):
        def __init__(self, field_ctx: Optional[FieldContext] = None):
            self._field_ctx = field_ctx

        def get_tile(self, fov_id: int, round_label: int, ch_label: int, zplane_label: int) -> FetchedTile:
            return DemoFetchedTile(
                img,
                zplane_label,
                ch_label,
                field_ctx=self._field_ctx,
            )

    return ImageStack.from_tilefetcher(
        DemoTileFetcher(field_ctx),
        {
            Axes.X: img.shape[3],
            Axes.Y: img.shape[2],
        },
        fov=0,
        rounds=range(1),
        chs=range(img.shape[1]),
        zplanes=range(img.shape[0]),
        group_by=(Axes.CH, Axes.ZPLANE),
    )


# Define some useful functions for viewing multiple images and histograms
def imshow_3channels(stack: starfish.ImageStack, r: int):
    fig = plt.figure(dpi=150)
    ax1 = fig.add_subplot(131, title="ch: 0")
    ax2 = fig.add_subplot(132, title="ch: 1")
    ax3 = fig.add_subplot(133, title="ch: 2")
    imshow_plane(stack, sel={Axes.ROUND: 0, Axes.CH: 0}, ax=ax1)
    imshow_plane(stack, sel={Axes.ROUND: 0, Axes.CH: 1}, ax=ax2)
    imshow_plane(stack, sel={Axes.ROUND: 0, Axes.CH: 2}, ax=ax3)


def plot_intensity_histograms(stack: starfish.ImageStack, r: int):
    fig = plt.figure(dpi=150)
    ax1 = fig.add_subplot(131, title="ch: 0")
    ax2 = fig.add_subplot(132, title="ch: 1", sharex=ax1, sharey=ax1)
    ax3 = fig.add_subplot(133, title="ch: 2", sharex=ax1, sharey=ax1)
    intensity_histogram(stack, sel={Axes.ROUND: r, Axes.CH: 0}, log=True, bins=50, ax=ax1)
    intensity_histogram(stack, sel={Axes.ROUND: r, Axes.CH: 1}, log=True, bins=50, ax=ax2)
    intensity_histogram(stack, sel={Axes.ROUND: r, Axes.CH: 2}, log=True, bins=50, ax=ax3)
    fig.tight_layout()


def scale(
    img: ImageStack,
    scale: np.ndarray[np.float32, Any],
    mins: np.ndarray[np.float32, Any] | None = None,
):
    if mins is not None:
        ElementWiseAddition(
            xr.DataArray(
                np.nan_to_num(-mins, nan=1).reshape(-1, 1, 1, 1, 1),
                dims=("c", "x", "y", "z", "r"),
            )
        ).run(img, in_place=True)

    Filter.ElementWiseMultiply(
        xr.DataArray(
            np.nan_to_num(scale, nan=1).reshape(-1, 1, 1, 1, 1),
            dims=("c", "x", "y", "z", "r"),
        )
    ).run(img, in_place=True)


def load_codebook(
    path: Path,
    bit_mapping: dict[str, int],
    exclude: set[str] | None = None,
    simple: bool = False,
):
    cb_json: dict[str, list[int]] = json.loads(path.read_text())
    for k in exclude or set():
        try:
            cb_json.pop(k)
        except KeyError:
            logger.warning(f"Codebook does not contain {k}")

    if simple:
        cb_json = {k: [v[0]] for k, v in cb_json.items()}
        logger.warning("Running in simple mode. Only using the first bit for each gene.")

    # Remove any genes that are not imaged.
    available_bits = sorted(bit_mapping)
    assert len(available_bits) == len(set(available_bits))
    cb_json = {k: v for k, v in cb_json.items() if all(str(bit) in available_bits for bit in v)}

    if not cb_json:
        raise ValueError("No genes in codebook are imaged. Please check your codebook and bit mapping.")

    used_bits = sorted(set(chain.from_iterable(cb_json.values())))
    names = np.array(list(cb_json.keys()))

    # mapping from bit name to index
    bit_map = np.ones(max(used_bits) + 1, dtype=int) * 5000
    for i, bit in enumerate(used_bits):
        bit_map[bit] = i

    arr = np.zeros((len(cb_json), len(used_bits)), dtype=bool)
    for i, bits_gene in enumerate(cb_json.values()):
        for bit in bits_gene:
            assert bit > 0
            arr[i, bit_map[bit]] = 1

    is_blank = np.array([n.startswith("Blank") for n in names])[:, None]  # [:-1, None]
    arr_zeroblank = arr * ~is_blank

    return (
        Codebook.from_numpy(names, n_round=1, n_channel=len(used_bits), data=arr[:, np.newaxis]),
        used_bits,
        names,
        arr_zeroblank,
    )


@click.group()
def spots(): ...


def _sanitize_tag(part: str | None) -> str:
    if part is None:
        return ""
    return str(part).replace("/", "_").replace(" ", "-")


def _make_tag(*parts: str | None) -> str:
    cleaned = [p for p in (_sanitize_tag(part) for part in parts) if p]
    return "-".join(cleaned) if cleaned else "spots"


def _setup_command_logging(
    path: Path | None,
    *,
    component: str,
    file_tag: str,
    debug: bool = False,
    extra: dict[str, Any] | None = None,
) -> None:
    payload = {k: v for k, v in (extra or {}).items() if v is not None}
    setup_cli_logging(path, component=component, file=file_tag, debug=debug, extra=payload)


def _batch(
    paths: list[Path],
    mode: str,
    args: list[str],
    *,
    threads: int = 13,
    split: bool | list[int | None] = False,
    stagger: float = 5.0,
    stagger_jitter: float = 0.0,
):
    if isinstance(split, list):
        split_list = split
        if any(x is None for x in split_list):
            raise ValueError("Cannot use None in split as list")
    elif split:
        split_list = list(range(4))
    else:
        split_list = [None]

    if not len(paths):
        raise ValueError("No files found.")

    def _run_with_stagger(cmd: list[str], delay: float = 0.0):
        if delay > 0:
            time.sleep(delay)
        return subprocess.run(cmd, check=True, capture_output=False)

    with progress_bar_threadpool(
        len(paths) * len(split_list), threads=threads, stop_on_exception=False
    ) as submit:
        idx = 0
        for path in paths:
            for s in split_list:
                slot = idx % max(1, threads)
                base = float(stagger) * slot if stagger else 0.0
                jitter = random.uniform(0.0, float(stagger_jitter)) if stagger_jitter > 0 else 0.0
                delay = base + jitter
                cmd = ["python", __file__, mode, str(path), *[a for a in args if a]] + (
                    ["--split", str(s)] if s is not None else []
                )
                submit(_run_with_stagger, cmd, delay)
                idx += 1


def sample_imgs(
    path: Path,
    codebook: str,
    round_num: int,
    *,
    roi: str | None = None,
    batch_size: int = 50,
):
    rand = np.random.default_rng(round_num)
    if roi is None:
        roi = "*"
    paths = sorted(
        (p for p in path.glob(f"registered--{roi}+{codebook}/reg*.tif") if not p.name.endswith(".hp.tif"))
    )
    if batch_size > len(paths):
        logger.info(f"Batch size {batch_size} is larger than {len(paths)}. Returning all images.")
        return paths
    return [paths[i] for i in sorted(rand.choice(range(len(paths)), size=batch_size, replace=False))]


@spots.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=Path))
@click.argument("roi", type=str)
@click.option("--round", "round_num", type=int)
@click.option(
    "--codebook",
    "codebook_path",
    type=click.Path(exists=True, dir_okay=False, file_okay=True, path_type=Path),
)
@click.option("--batch-size", "-n", type=int, default=40)
@click.option("--threads", "-t", type=int, default=8)
@click.option("--overwrite", is_flag=True)
@click.option("--split", type=int, default=None)
@click.option("--blank", type=str, default=None)
@click.option(
    "--field-correct/--no-field-correct",
    default=False,
    help="Apply illumination field correction using discovered TCYX field stores.",
)
@click.option(
    "--config",
    "json_config",
    type=click.Path(exists=True, dir_okay=False, file_okay=True, path_type=Path),
    default=None,
    help="Optional project config (JSON). CLI flags override JSON.",
)
def step_optimize(
    path: Path,
    roi: str,
    round_num: int,
    codebook_path: Path,
    batch_size: int = 40,
    threads: int = 8,
    overwrite: bool = False,
    split: int = 0,
    blank: str | None = None,
    json_config: Path | None = None,
    field_correct: bool = False,
):
    _setup_command_logging(
        path,
        component="preprocess.spots.step_optimize",
        file_tag=_make_tag(
            "step-optimize",
            roi,
            codebook_path.stem,
            f"r{round_num}" if round_num else None,
        ),
        extra={
            "roi": roi,
            "round": round_num,
            "codebook": codebook_path.stem,
            "overwrite": overwrite,
        },
    )
    wd = path / f"opt_{codebook_path.stem}{f'+{roi}' if roi != '*' else ''}"
    if round_num > 0 and not (wd / "percentiles.json").exists():
        raise Exception("Please run `fishtools find-threshold` first.")

    selected = sample_imgs(path, codebook_path.stem, round_num, roi=roi, batch_size=batch_size)

    group_counts = {key: len(list(group)) for key, group in groupby(selected, key=lambda x: x.parent.name)}
    logger.info(f"Group counts: {json.dumps(group_counts, indent=2)}")

    # Copy codebook to the same folder as the images for reproducibility.
    (path / "codebooks").mkdir(exist_ok=True)
    if not (new_cb_path := path / "codebooks" / codebook_path.name).exists():
        shutil.copy(codebook_path, new_cb_path)

    ws = Workspace(path)
    if field_correct:
        rois_needed = {
            p.parent.name.split("--", 1)[1].split("+", 1)[0] for p in selected if "--" in p.parent.name
        }
        if not rois_needed:
            rois_needed = {roi} if roi != "*" else set(ws.rois)
        _ensure_field_stores(ws, sorted(rois_needed), codebook_path.stem)

    return _batch(
        selected,
        "run",
        [
            "--calc-deviations",
            "--global-scale",
            str(wd / "global_scale.txt"),
            "--codebook",
            codebook_path.as_posix(),
            f"--round={round_num}",
            *(["--overwrite"] if overwrite else []),
            f"--split={split}",
            *(["--roi", roi] if roi else []),
            *(["--blank", blank] if blank else []),
            *(["--config", json_config.as_posix()] if json_config else []),
            *(["--field-correct"] if field_correct else []),
        ],
        threads=threads,
    )


def _channel_labels_from_metadata(metadata: Mapping[str, Any] | None, channel_count: int) -> list[str]:
    keys: Any = None
    if metadata is not None:
        keys = metadata.get("keys") or metadata.get("key")

    if isinstance(keys, (list, tuple)):
        labels = [str(k) for k in keys]
    elif keys is not None:
        labels = [str(keys)]
    else:
        labels = []

    if len(labels) != channel_count:
        labels = [str(i) for i in range(channel_count)]

    return labels


@spots.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=Path))
@click.argument("roi", type=str)
@click.option(
    "--codebook",
    type=click.Path(exists=True, dir_okay=False, file_okay=True, path_type=Path),
)
@click.option("--overwrite", is_flag=True)
@click.option("--round", "round_num", type=int, default=0)
@click.option("--blank", type=str, default=None)
@click.option(
    "--field-correct/--no-field-correct",
    default=False,
    help="Apply illumination field correction using discovered TCYX field stores.",
)
@click.option(
    "--config",
    "json_config",
    type=click.Path(exists=True, dir_okay=False, file_okay=True, path_type=Path),
    default=None,
    help="Optional project config (JSON). CLI flags override JSON.",
)
def find_threshold(
    path: Path,
    roi: str,
    codebook: Path,
    overwrite: bool = False,
    round_num: int = 0,
    blank: str | None = None,
    json_config: Path | None = None,
    field_correct: bool = False,
):
    _setup_command_logging(
        path,
        component="preprocess.spots.find_threshold",
        file_tag=_make_tag("find-threshold", roi, codebook.stem, f"r{round_num}"),
        extra={
            "roi": roi,
            "round": round_num,
            "codebook": codebook.stem,
            "overwrite": overwrite,
        },
    )
    SUBFOLDER = "_highpassed"
    paths = sorted(path.glob(f"registered--{roi}+{codebook.stem}/reg*.tif"))
    path_out = path / (f"opt_{codebook.stem}" + (f"+{roi}" if roi != "*" else ""))
    jsonfile = path_out / "percentiles.json"

    ws = Workspace(path)
    if field_correct:
        if roi == "*":
            rois_needed = set(ws.rois)
        else:
            rois_needed = {roi}
        _ensure_field_stores(ws, sorted(rois_needed), codebook.stem)

    rand = np.random.default_rng(0)
    if len(paths) > 50:
        paths = rand.choice(paths, size=50, replace=False)  # type: ignore
    paths = sorted(
        p for p in paths if not (p.parent / SUBFOLDER / f"{p.stem}_{codebook.stem}.hp.tif").exists()
    )

    # --- plotall command ---

    if not (path_out / "global_scale.txt").exists():
        raise ValueError("Please run align_prod optimize first.")

    if round_num > 0 and not jsonfile.exists():
        raise ValueError(
            "Round_num > 0 but no existing percentiles file found. Please run with round_num=0 first."
        )

    if paths:
        logger.info(f"Creating {len(paths)} highpassed images")
        _batch(
            paths,
            "run",
            [
                "--codebook",
                str(codebook),
                "--highpass-only",
                *(["--overwrite"] if overwrite else []),
                *(["--blank", blank] if blank else []),
                *(["--config", json_config.as_posix()] if json_config else []),
                *(["--field-correct"] if field_correct else []),
            ],
            threads=4,
            split=[0],
        )

    highpasses = list(path.glob(f"registered--{roi}+{codebook.stem}/{SUBFOLDER}/*_{codebook.stem}.hp.tif"))
    logger.info(f"Found {len(highpasses)} images to get percentiles from.")

    norms = {}

    # Load percentile from config (SpotDecodeConfig.clip_percentile)
    try:
        cfg = load_config(json_config) if json_config is not None else load_config()
        percentile = float(cfg.spot_decode.clip_percentile) if cfg.spot_decode is not None else 40.0
        logger.info(f"Using clip_percentile={percentile}")
    except Exception as e:
        logger.warning(f"Falling back to default percentile due to config error: {e}")
        percentile = 40.0

    try:
        mins_np = np.loadtxt(path_out / "global_min.txt", dtype=np.float32).reshape(1, -1, 1, 1)
        global_scale_np = (
            np.atleast_3d(np.loadtxt(path_out / "global_scale.txt", dtype=np.float32))[round_num]
            .reshape(1, -1, 1, 1)
            .astype(np.float32, copy=False)
        )
        logger.info(f"Using global scale from round {round_num}")
    except IndexError:
        raise ValueError(
            f"Round {round_num} not found. Please run align_prod optimize with the round_number first."
        )

    use_gpu = False  # GPU and _cupy_available()
    if use_gpu:
        logger.info("find-threshold: Using GPU for percentile computation")
        mins_dev = cp.asarray(mins_np, dtype=cp.float32)
        scale_dev = cp.asarray(global_scale_np, dtype=cp.float32)
    else:
        mins_dev = None  # type: ignore[assignment]
        scale_dev = None  # type: ignore[assignment]

    skipped_mismatched: list[str] = []
    try:
        for p in sorted(highpasses[:50]):
            logger.info(f"Processing highpass {p.parent.name}/{p.name}")

            # Load and lightly downsample spatially to control memory; keep float32
            with TiffFile(p) as tif:
                hp_np = tif.asarray()
                metadata = tif.shaped_metadata[0] if tif.shaped_metadata else None

            channel_labels = _channel_labels_from_metadata(metadata, hp_np.shape[1])
            channel_count_img = hp_np.shape[1]
            expected_channels = mins_np.shape[1]
            if channel_count_img != expected_channels:
                if channel_count_img < expected_channels:
                    problematic = [str(i) for i in range(channel_count_img, expected_channels)]
                    issue = "missing"
                else:
                    problematic = channel_labels[expected_channels:]
                    issue = "extra"

                logger.error(
                    "Skipping %s: expected %d channels (indices %s) but found %d (labels %s). %s indices: %s",
                    p.name,
                    expected_channels,
                    [str(i) for i in range(expected_channels)],
                    channel_count_img,
                    channel_labels,
                    issue.capitalize(),
                    problematic,
                )
                skipped_mismatched.append(p.name)
                continue

            img_np = hp_np.astype(np.float32)[:, :, :1024:2, :1024:2]  # ZCYX

            if use_gpu:
                img_dev = cp.asarray(img_np, dtype=cp.float32)
                img_dev = img_dev - mins_dev  # broadcast
                img_dev = img_dev * scale_dev
                norm_dev = cp.linalg.norm(img_dev, axis=1)
                val = float(cp.asnumpy(cp.percentile(norm_dev, percentile)))
                del img_dev, norm_dev  # explicit free before next iteration
            else:
                img_np = img_np - mins_np
                img_np = img_np * global_scale_np
                norm_np = np.linalg.norm(img_np, axis=1)
                val = float(np.percentile(norm_np, percentile))
                del img_np, norm_np

            norms[p.parent.parent.name + "-" + p.name] = val
    finally:
        if use_gpu:
            # Best‑effort: free pools between large batches
            _gpu_release_all()

    if skipped_mismatched:
        logger.warning(
            "Skipped %d highpass file(s) with channel mismatches: %s",
            len(skipped_mismatched),
            ", ".join(skipped_mismatched),
        )

    path_out.mkdir(exist_ok=True)

    logger.info(f"Writing to {jsonfile}")
    if not jsonfile.exists():
        jsonfile.write_text(json.dumps([norms], indent=2))
    else:
        prev_norms = json.loads(jsonfile.read_text())
        prev_norms.append(norms)
        jsonfile.write_text(json.dumps(prev_norms, indent=2))


def _sanitize_codebook_name(codebook: str) -> str:
    return codebook.replace("-", "_").replace(" ", "_")


def _resolve_spots_parquet(base: Path, roi: str, codebook: str) -> Path:
    cb_s = _sanitize_codebook_name(codebook)
    candidates = [
        base / "analysis" / "output" / f"{roi}+{cb_s}.parquet",
        base / "analysis" / "output" / f"{roi}+{codebook}.parquet",
        base / "analysis" / "deconv" / f"{roi}+{cb_s}.parquet",
        base / "analysis" / "deconv" / f"{roi}+{codebook}.parquet",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Could not find spots parquet for ROI '{roi}', codebook '{codebook}'.\n"
        f"Searched: {', '.join(map(str, candidates))}"
    )


def _render_one_roi_plot(
    base: Path,
    roi: str,
    codebook: str,
    outdir: Path,
    *,
    dark: bool = False,
    only_blank: bool = False,
    overwrite: bool = False,
    max_per_plot: int | None = None,
    figure_sizes: list[tuple[float, float]] | None = None,
    cmap: str | None = None,
) -> list[Path]:
    """Render and save plot(s) for one ROI.

    - When max_per_plot is None, produce a single figure.
    - When max_per_plot is a positive integer and the gene count exceeds it,
      split across multiple figures and append ``--part{n}`` to filenames.
    Returns a list of output paths created or found.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    cb_s = _sanitize_codebook_name(codebook)
    base_name = f"plotall--{roi}+{cb_s}{'--blank' if only_blank else ''}{'--dark' if dark else ''}"

    spots_path = _resolve_spots_parquet(base, roi, codebook)
    spots_df = pl.read_parquet(spots_path)
    genes = spots_df.get_column("target").unique().to_list()  # type: ignore[attr-defined]
    genes = sorted(g for g in genes if isinstance(g, str))
    if only_blank:
        genes = [g for g in genes if g.startswith("Blank")]

    outputs: list[Path] = []

    if (max_per_plot is None) or (len(genes) <= max_per_plot):
        out_png = outdir / f"{base_name}.png"
        if not out_png.exists() or overwrite:
            size0 = figure_sizes[0] if figure_sizes else None
            fig, _ = plot_all_genes(spots_df, dark=dark, only_blank=only_blank, figsize=size0, cmap=cmap)
            fig.tight_layout()
            fig.savefig(out_png.as_posix(), dpi=200, bbox_inches="tight")
            plt.close(fig)
        outputs.append(out_png)
        return outputs

    # Multi-part rendering
    n_parts = (len(genes) + max_per_plot - 1) // max_per_plot  # type: ignore[operator]
    for idx in range(n_parts):
        g_chunk = genes[idx * max_per_plot : (idx + 1) * max_per_plot]  # type: ignore[operator]
        df_chunk = spots_df.filter(pl.col("target").is_in(g_chunk))
        out_png = outdir / f"{base_name}.{idx + 1}.png"
        if not out_png.exists() or overwrite:
            size_i = (
                figure_sizes[idx]
                if (figure_sizes and idx < len(figure_sizes))
                else (figure_sizes[-1] if figure_sizes else None)
            )
            fig, _ = plot_all_genes(df_chunk, dark=dark, only_blank=False, figsize=size_i, cmap=cmap)
            fig.tight_layout()
            fig.savefig(out_png.as_posix(), dpi=200, bbox_inches="tight")
            plt.close(fig)
        outputs.append(out_png)

    return outputs


@spots.command("plotall")
@click.argument("path", type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=Path))
@click.argument("roi", type=str, default="*")
@click.option(
    "--codebook",
    "codebook_label",
    type=str,
    default="cs-base",
    help="Codebook label or path; if a path is given, the stem is used.",
)
@click.option("--threads", "-t", type=int, default=8, help="Max processes to render plots.")
@click.option(
    "--only-blank",
    is_flag=True,
    help="Plot only genes with names starting with 'Blank'.",
)
@click.option("--dark", is_flag=True, help="Use dark background + magma colormap.")
@click.option(
    "--outdir",
    type=click.Path(dir_okay=True, file_okay=False, path_type=Path),
    default=None,
    help="Output directory for saved figures (default: analysis/output/plots).",
)
@click.option("--overwrite", is_flag=True, help="Overwrite existing figures if present.")
@click.option(
    "--max-per-plot",
    "max_per_plot_opt",
    type=str,
    default=None,
    help='Maximum genes per figure; integer or "none" for unlimited.',
)
@click.option(
    "--figure-size",
    "figure_size_opt",
    type=str,
    default=None,
    help="Per-plot figure size in inches. Accepts a single size (e.g., '72x12') or a semicolon-separated list for multi-part outputs (e.g., '60x10;48x8').",
)
@click.option(
    "--cmap",
    "cmap",
    type=str,
    default=None,
    help="Colormap name for hexbin plots (e.g., 'magma', 'viridis').",
)
def plot_all_genes_cli(
    path: Path,
    roi: str,
    codebook_label: str = "cs-base",
    threads: int = 8,
    only_blank: bool = False,
    dark: bool = False,
    outdir: Path | None = None,
    overwrite: bool = False,
    max_per_plot_opt: str | None = None,
    figure_size_opt: str | None = None,
    cmap: str | None = None,
):
    """Plot per-gene spot distributions for one or many ROIs.

    Examples:
    - Single ROI: preprocess spots plotall /workspace cortex --codebook cs-base
    - All ROIs:   preprocess spots plotall /workspace --codebook cs-base   (ROI omitted)
    - Wildcard:   preprocess spots plotall /workspace * --codebook cs-base
    """
    cb_stem = Path(codebook_label).stem if Path(codebook_label).exists() else codebook_label
    _setup_command_logging(
        path,
        component="preprocess.spots.plotall",
        file_tag=_make_tag("plotall", roi, cb_stem),
        extra={
            "roi": roi,
            "codebook": cb_stem,
            "threads": threads,
            "overwrite": overwrite,
        },
    )
    ws = Workspace(path)
    # Derive codebook name even if a full path is passed
    cb = cb_stem
    click.echo(cb)

    # Resolve ROIs
    if roi in {"*", "all"}:
        rois = ws.rois
    else:
        rois = ws.resolve_rois([roi])
    if not rois:
        raise click.ClickException("No ROIs found in workspace")

    # Output directory
    if outdir is None:
        outdir = ws.path / "analysis" / "output" / "plots"
    outdir.mkdir(parents=True, exist_ok=True)

    # Dispatch per-ROI rendering jobs

    # Session start log
    t0 = time.perf_counter()
    logger.info(
        "[spots.plotall] started | path={} | rois={} | codebook={} | outdir={} | threads={} | only_blank={} | dark={} | overwrite={}",
        ws.path,
        rois,
        cb,
        outdir,
        threads,
        only_blank,
        dark,
        overwrite,
    )

    errors: list[str] = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]spots.plotall"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task("render", total=len(rois))

        # Parse figure_size(s): allow single or semicolon-separated list
        def _parse_figsize_one(s: str) -> tuple[float, float]:
            s = s.strip().lower().replace("x", ",").replace(" ", ",")
            parts = [p for p in s.split(",") if p]
            if len(parts) != 2:
                raise click.BadParameter("--figure-size expects WIDTHxHEIGHT (inches)")
            try:
                return float(parts[0]), float(parts[1])
            except Exception:
                raise click.BadParameter("--figure-size expects numeric WIDTH and HEIGHT")

        def _parse_figsizes(opt: str | None) -> list[tuple[float, float]] | None:
            if opt is None:
                return None
            if ";" in opt:
                return [_parse_figsize_one(x) for x in opt.split(";") if x.strip()]
            return [_parse_figsize_one(opt)]

        fig_sizes = _parse_figsizes(figure_size_opt)
        if int(threads) <= 1:
            # Sandbox-friendly sequential path
            for r in rois:
                try:
                    progress.console.log(f"[bold cyan]ROI '{r}' started")
                    # Parse max_per_plot locally per run
                    if max_per_plot_opt is None:
                        mpp = None
                    else:
                        vv = max_per_plot_opt.strip().lower()
                        if vv in {"none", "null", ""}:
                            mpp = None
                        else:
                            try:
                                mpp = int(vv)
                                if mpp <= 0:
                                    raise ValueError
                            except Exception:
                                raise click.BadParameter("--max-per-plot must be positive int or 'none'")

                    saved = _render_one_roi_plot(
                        ws.path,
                        r,
                        cb,
                        outdir,
                        dark=dark,
                        only_blank=only_blank,
                        overwrite=overwrite,
                        max_per_plot=mpp,
                        figure_sizes=fig_sizes,
                        cmap=cmap,
                    )
                    progress.advance(task)
                    progress.console.log(f"[bold green]ROI '{r}' completed -> {saved}")
                except Exception as e:  # pragma: no cover
                    progress.console.log(f"[bold red]ROI '{r}' failed: {e}")
                    errors.append(str(e))
        else:
            ctx = mp.get_context("spawn")
            # Parse once and pass into workers
            if max_per_plot_opt is None:
                mpp = None
            else:
                vv = max_per_plot_opt.strip().lower()
                if vv in {"none", "null", ""}:
                    mpp = None
                else:
                    try:
                        mpp = int(vv)
                        if mpp <= 0:
                            raise ValueError
                    except Exception:
                        raise click.BadParameter("--max-per-plot must be positive int or 'none'")

            with ProcessPoolExecutor(max_workers=max(1, int(threads)), mp_context=ctx) as ex:
                futures = {}
                for r in rois:
                    progress.console.log(f"[bold cyan]ROI '{r}' submitted")
                    fut = ex.submit(
                        _render_one_roi_plot,
                        ws.path,
                        r,
                        cb,
                        outdir,
                        dark=dark,
                        only_blank=only_blank,
                        overwrite=overwrite,
                        max_per_plot=mpp,
                        figure_sizes=fig_sizes,
                        cmap=cmap,
                    )
                    futures[fut] = r

                pending = set(futures.keys())
                while pending:
                    done, pending = wait(pending, timeout=0.5, return_when=FIRST_COMPLETED)
                    for fut in done:
                        r = futures.get(fut, "?")
                        try:
                            saved = fut.result()
                            progress.advance(task)
                            progress.console.log(f"[bold green]ROI '{r}' completed -> {saved}")
                        except Exception as e:  # pragma: no cover
                            progress.console.log(f"[bold red]ROI '{r}' failed: {e}")
                            errors.append(str(e))

    if errors:
        raise click.ClickException("; ".join(errors))

    dt = time.perf_counter() - t0
    logger.success(
        "[spots.plotall] completed | rois_processed={} | duration={:.2f}s | outdir={}",
        len(rois),
        dt,
        outdir,
    )


def load_2d(path: Path | str, *args, **kwargs):
    return np.atleast_2d(np.loadtxt(path, *args, **kwargs))


def create_opt_path(
    *,
    codebook_path: Path,
    mode: Literal["json", "pkl", "folder"],
    round_num: int | None = None,
    path_img: Path | None = None,
    path_folder: Path | None = None,
    roi: str | None = None,
):
    if roi is None:
        roi = "*"
    if not ((path_img is None) ^ (path_folder is None)):
        raise ValueError("Must provide only path_img or path_folder")

    if path_img is not None:
        base = path_img.parent.parent / f"opt_{codebook_path.stem}{f'+{roi}' if roi != '*' else ''}"
        name = f"{path_img.stem}--{path_img.resolve().parent.name.split('--')[1]}"
    elif path_folder is not None:
        base = path_folder / f"opt_{codebook_path.stem}{f'+{roi}' if roi != '*' else ''}"
        if mode == "folder":
            return base
        raise ValueError("Unknown mode")
    else:
        raise Exception("This should never happen.")

    base.mkdir(exist_ok=True)
    if mode == "json":
        return base / f"{name}.json"
    if mode == "pkl":
        return base / f"{name}_opt{round_num:02d}.pkl"
    if mode == "folder":
        return base
    raise ValueError(f"Unknown mode {mode}")


@spots.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=Path))
@click.argument("roi", type=str, default="*")
@click.option(
    "--codebook",
    "codebook_path",
    type=click.Path(exists=True, dir_okay=False, file_okay=True, path_type=Path),
)
@click.option("--batch-size", "-n", type=int, default=50)
@click.option("--round", "round_num", type=int)
def combine(path: Path, roi: str, codebook_path: Path, batch_size: int, round_num: int):
    """Create global scaling factors from `optimize`.

    Part of the channel optimization process.
    This is to be run after `optimize` has been run.

    Round 0: 5th percentile of all scaling factors (lower indicates brighter spots).
    Round n:
        Balance the scaling factors such that each positive spot
        has about the same intensity in all bit channels.
        Get deviations from all images during `optimize` and average them.
        This average deviation is a vector of length n.
        Divide this vector by its mean to get a scaling factor for each bit channel.
        This gets applied to the previous round's scaling factors.
        Also calculates the variance of the deviations to track the convergence.

    Args:
        path: registered images folder
        codebook_path
        round_num: starts from 0.
    """
    _setup_command_logging(
        path,
        component="preprocess.spots.combine",
        file_tag=_make_tag("combine", roi, codebook_path.stem, f"r{round_num}"),
        extra={
            "roi": roi,
            "round": round_num,
            "codebook": codebook_path.stem,
            "batch_size": batch_size,
        },
    )

    selected = sample_imgs(
        path,
        codebook_path.stem,
        round_num,
        batch_size=batch_size * 2 if round_num == 0 else batch_size,
        roi=roi,
    )
    path_opt = create_opt_path(
        path_folder=path,
        codebook_path=codebook_path,
        mode="folder",
        round_num=round_num,
        roi=roi,
    )
    paths = [
        create_opt_path(
            path_img=p,
            codebook_path=codebook_path,
            mode="json",
            round_num=round_num,
            roi=roi,
        )
        for p in selected
    ]

    # paths = list((path / codebook_path.stem).glob("reg*.json"))
    curr = []
    mins = []
    n = 0
    for p in paths:
        if not p.exists():
            logger.warning(f"Path {p} does not exist. Skipping.")
            continue
        sf = Deviations.validate_json(p.read_text())
        # if (round_num + 1) > len(sf):
        #     raise ValueError(f"Round number {round_num} exceeds what's available ({len(sf)}).")

        if round_num == 0:
            curr.append([cast(InitialScale, s).initial_scale for s in sf if s.round_num == 0][0])
            mins.append([cast(InitialScale, s).mins for s in sf if s.round_num == 0][0])
        else:
            try:
                want = cast(Deviation, [s for s in sf if s.round_num == round_num][0])
            except IndexError:
                logger.warning(f"No deviation for round {round_num} in {p}. Skipping.")
                continue
            if want.n < 150:
                logger.debug(f"Skipping {p} at round {round_num} because n={want.n} < 150.")
                continue
            curr.append(np.nan_to_num(np.array(want.deviation, dtype=float), nan=1) * want.n)
            n += want.n
    curr, mins = np.array(curr), np.array(mins)
    # pl.DataFrame(curr).write_csv(path / name)

    global_scale_file = path_opt / "global_scale.txt"
    global_min_file = path_opt / "global_min.txt"
    if round_num == 0:
        # ! The result after scaling must be in [0, 1].
        curr = np.mean(np.array(curr), axis=0, keepdims=True)
        overallmean = np.nanmean(curr)
        curr /= overallmean
        curr = np.clip(curr, 0, 2)
        curr /= np.nanmean(curr)
        curr = np.clip(curr, 0.5, None)
        curr /= np.nanmean(curr)
        np.savetxt(global_scale_file, curr)

        np.savetxt(global_min_file, np.mean(mins, axis=0))
        return

    if not global_scale_file.exists() or not global_min_file.exists():
        raise ValueError("Round > 0 requires global scale file and global min file.")

    deviation = curr.sum(axis=0) / n
    # Normalize per channel
    deviation = deviation / np.nanmean(deviation)
    cv = np.sqrt(np.mean(np.square(deviation - 1)))
    logger.info(f"Round {round_num}. CV: {cv:04f}.")

    previouses = load_2d(global_scale_file)
    old = previouses[round_num - 1]
    new = old / deviation
    # if round_num > 3:
    #     new = (old + new) / 2
    # β = 0.1
    # velocity = old - previouses[round_num - 2]
    # velocity = β * velocity + (1 - β) * grad
    # new = old + velocity

    np.savetxt(
        global_scale_file,
        np.concatenate([np.atleast_2d(previouses[:round_num]), np.atleast_2d(new)], axis=0),
    )
    (path_opt / "mse.txt").open("a").write(f"{round_num:02d}\t{cv:04f}\n")


def initial(img: ImageStack, percentiles: tuple[float, float] = (1, 99.99)):
    """
    Create initial scaling factors.
    Returns the max intensity of each channel.
    Seems to work well enough.
    """
    maxed = img.reduce({Axes.ROUND, Axes.ZPLANE}, func="max")
    res = np.percentile(np.array(maxed.xarray).squeeze(), percentiles, axis=(1, 2))
    # res = np.array(maxed.xarray).squeeze()
    if np.isnan(res).any() or (res[:, 1] == 0).any():
        raise ValueError("NaNs or zeros found in initial scaling factor.")
    return res


class Deviation(BaseModel):
    n: int
    deviation: list[float | None]
    percent_blanks: float | None = None
    round_num: int

    @field_validator("round_num")
    @classmethod
    def must_be_positive(cls, v: int) -> int:
        if v > 0:
            return v
        raise ValueError("Round number must be positive for Deviation.")


class InitialScale(BaseModel):
    initial_scale: list[float]
    round_num: int
    mins: list[float]

    @field_validator("round_num")
    @classmethod
    def must_be_zero(cls, v: int) -> int:
        if v == 0:
            return v
        raise ValueError("Round number must be zero for InitialScale.")


Deviations = TypeAdapter(list[InitialScale | Deviation])


def append_json(
    path: Path,
    round_num: int,
    *,
    n: int | None = None,
    deviation: np.ndarray | None = None,
    initial_scale: np.ndarray | None = None,
    mins: np.ndarray | None = None,
    percent_blanks: float | None = None,
):
    existing = Deviations.validate_json(path.read_text()) if path.exists() else Deviations.validate_json("[]")
    if initial_scale is not None and mins is not None:
        if round_num > 0:
            raise ValueError("Cannot set initial scale for round > 0")
        existing.append(
            InitialScale(
                initial_scale=initial_scale.tolist(),
                round_num=round_num,
                mins=mins.tolist(),
            )
        )
    elif (initial_scale is None) ^ (mins is None):
        raise ValueError("Must provide both initial_scale and mins or not")
    elif deviation is not None and n is not None:
        if round_num == 0:
            raise ValueError("Cannot set deviation for round 0")
        # if existing.__len__() < round_num - 1:
        # raise ValueError("Round number exceeds number of existing rounds.")

        existing = [e for e in existing if e.round_num < round_num]
        existing.append(
            Deviation(
                n=n,
                deviation=deviation.tolist(),
                round_num=round_num,
                percent_blanks=percent_blanks,
            )
        )
    else:
        raise ValueError("Must provide either initial_scale or deviation and n.")

    path.write_bytes(Deviations.dump_json(existing))


def pixel_decoding(imgs: ImageStack, config: SpotDecodeConfig, codebook: Codebook, *, gpu: bool = False):
    pixel_intensities = IntensityTable.from_image_stack(imgs)
    logger.info("Decoding")
    decoded_intensities = codebook.decode_metric(
        pixel_intensities,
        max_distance=config.max_distance,
        min_intensity=config.min_intensity,
        norm_order=2,
        metric="euclidean",
        return_original_intensities=True,
    )

    logger.info("Combining")
    caf = CombineAdjacentFeatures(
        min_area=config.min_area, max_area=config.max_area, mask_filtered_features=True
    )
    decoded_spots, image_decoding_results = caf.run(
        intensities=decoded_intensities, n_processes=int(config.threads)
    )
    transfer_physical_coords_to_intensity_table(image_stack=imgs, intensity_table=decoded_spots)
    return decoded_spots, image_decoding_results


def pixel_decoding_gpu(imgs: ImageStack, config: SpotDecodeConfig, codebook: Codebook):
    pixel_intensities = IntensityTable.from_image_stack(imgs)
    # Derive shape robustly for mocks; fall back when unavailable
    try:
        shape_raw = imgs.xarray.squeeze().shape  # CZYX
        shape = (shape_raw[1], shape_raw[0], *shape_raw[2:])
    except Exception:
        shape = None
    logger.info("Decoding")
    kwargs = dict(
        max_distance=config.max_distance,
        min_intensity=config.min_intensity,
        norm_order=2,
        metric="euclidean",
        return_original_intensities=True,
    )
    if shape is not None:
        kwargs["shape"] = shape
    decoded_intensities = codebook.decode_metric(pixel_intensities, **kwargs)

    logger.info("Combining")
    caf = CombineAdjacentFeatures(
        min_area=config.min_area, max_area=config.max_area, mask_filtered_features=True
    )
    try:
        decoded_spots, image_decoding_results = caf.run(
            intensities=decoded_intensities, n_processes=int(config.threads)
        )
    except ValueError as e:
        raise ValueError("No spots found. Skipping.") from e
    transfer_physical_coords_to_intensity_table(image_stack=imgs, intensity_table=decoded_spots)
    return decoded_spots, image_decoding_results


def spot_decoding(
    imgs: ImageStack,
    codebook: Codebook,
    spot_diameter: int = 7,
    min_mass: float = 0.1,
    max_size: int = 2,
    separation: int = 8,
    noise_size: float = 0.65,
    percentile: int = 0,
) -> DecodedIntensityTable:
    # z project
    # max_imgs = imgs.reduce({Axes.ZPLANE}, func="max")
    # tlmpf = starfish.spots.FindSpots.LocalMaxPeakFinder(
    #     spot_diameter=spot_diameter,
    #     min_mass=min_mass,
    #     max_size=max_size,
    #     separation=separation,
    #     noise_size=noise_size,
    #     percentile=percentile,
    #     verbose=True,
    # )

    # run LocalMaxPeakFinder on max projected image
    bd = FindSpots.BlobDetector(
        min_sigma=1,
        max_sigma=4,
        num_sigma=10,
        threshold=0.03,
        is_volume=True,
    )
    # bd = FindSpots.BlobDetector(
    #     min_sigma=2, max_sigma=5, num_sigma=10, threshold=0.01, is_volume=True, measurement_type="mean"
    # )
    spots = bd.run(image_stack=imgs)  # , reference_image=dots)
    # spots = lmp.run(max_imgs)

    decoder = DecodeSpots.SimpleLookupDecoder(codebook=codebook)
    return decoder.run(spots=spots)


KEYS_560NM = set(range(1, 9)) | {25, 28, 31, 34}
KEYS_650NM = set(range(9, 17)) | {26, 29, 32, 35}
KEYS_750NM = set(range(17, 25)) | {27, 30, 33, 36}
GAUSS_SIGMA = (2, 2, 2)


def get_blank_channel_info(key: str) -> tuple[int, int]:
    """
    Maps an image channel key to its corresponding wavelength and blank channel index.

    Args:
        key: The channel key (string representation of an integer).

    Returns:
        A tuple containing (wavelength_nm, blank_channel_index).
    """
    key_int = int(key)
    if key_int in KEYS_560NM:
        return 560, 0
    if key_int in KEYS_650NM:
        return 650, 1
    if key_int in KEYS_750NM:
        return 750, 2
    raise ValueError(f"Channel key '{key}' has no defined wavelength mapping.")


def generate_subtraction_matrix(blanks: xr.DataArray, coefs: pl.DataFrame, keys: list[str]) -> xr.DataArray:
    keys_df = pl.DataFrame({"channel_key": [int(k) for k in keys], "output_key": keys})
    params_df = keys_df.join(coefs, on="channel_key", how="left", maintain_order="left")
    if params_df["slope"].is_null().any():
        missing_key = params_df.filter(pl.col("slope").is_null())["output_key"][0]
        raise ValueError(f"No parameters found for channel key '{missing_key}'.")

    # Extract slopes and intercepts as NumPy arrays
    slopes = params_df["slope"].to_numpy()
    intercepts = params_df["intercept"].to_numpy()

    # G# Get the corresponding source blank index for each output key

    # --- 2. Vectorized Selection ---

    # Select all necessary source blank channels at once using the list of indices.
    # If blanks.dims is (r,c,z,y,x) and len(keys) is N,
    # this creates a DataArray with dims (r, c, z, y, x) and shape (1, N, z, y, x).
    # The new 'c' dimension corresponds to our N output channels.
    source_indices = [get_blank_channel_info(key)[1] for key in keys]
    selected_blanks = blanks.isel(c=source_indices)
    # print(selected_blanks.max({Axes.X, Axes.Y, Axes.ZPLANE}))

    slope_da = xr.DataArray(slopes, dims=[str(Axes.CH)])
    intercept_da = xr.DataArray(intercepts, dims=[str(Axes.CH)])
    # print(slope_da)
    # print(intercept_da)

    scaled_channels = (selected_blanks * slope_da) + intercept_da / 65535
    # print(scaled_channels.max({Axes.X, Axes.Y, Axes.ZPLANE}))
    floored_channels = xr.where(scaled_channels < 0, 0, scaled_channels)

    # Ensure the dimension order is what we expect (though it should be already)
    return -floored_channels.transpose(
        str(Axes.ROUND), str(Axes.CH), str(Axes.ZPLANE), str(Axes.Y), str(Axes.X)
    )


@spots.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=False, file_okay=True, path_type=Path))
@click.option("--overwrite", is_flag=True)
# @click.option("--roi", type=str, default=None)
@click.option("--global-scale", type=click.Path(dir_okay=False, file_okay=True, path_type=Path))
@click.option("--round", "round_num", type=int, default=None)
@click.option("--calc-deviations", is_flag=True)
@click.option("--debug", is_flag=True)
@click.option("--split", type=int, default=None, help="Quadrant split index (0-3)")
@click.option("--highpass-only", is_flag=True)
@click.option(
    "--codebook",
    "codebook_path",
    type=click.Path(exists=True, dir_okay=False, file_okay=True, path_type=Path),
)
@click.option(
    "--field-correct/--no-field-correct",
    default=False,
    help="Apply illumination field correction using discovered TCYX field stores.",
)
@click.option("--simple", is_flag=True)
@click.option("--roi", type=str, default=None)
@click.option("--blank", type=str, default=None)
@click.option(
    "--config",
    "json_config",
    type=click.Path(exists=True, dir_okay=False, file_okay=True, path_type=Path),
    default=None,
    help="Optional project config (JSON). CLI flags override JSON.",
)
def run(
    path: Path,
    *,
    global_scale: Path | None,
    codebook_path: Path,
    round_num: int | None = None,
    overwrite: bool = False,
    debug: bool = False,
    calc_deviations: bool = False,
    split: int | None = None,
    highpass_only: bool = False,
    simple: bool = False,
    decode: SpotDecodeConfig | None = None,
    roi: str | None = None,
    blank: str | None = None,
    json_config: Path | None = None,
    field_correct: bool = False,
):
    """
    Run spot calling.
    Used for actual spot calling and channel optimization.

    For channel optimization, start by calling with `--calc-deviations` and `--round=0`.
    This will get the initial scaling factors for each bit channel (max).
    Returns the average intensity of all non-blank spots including the total number of spots.
    These info from multiple images are used to calculate the scaling factors for each bit channel.

    Intermediate files are saved in path / codebook_path.stem since in the current design,
    different codebooks use different channels.

    For actual spot calling, call with `--global-scale`.
    This will use the latest scaling factors calculated from the previous step to decode.
    """
    debug = debug or (os.environ.get("DEBUG", "0") == "1")
    workspace_hint = path.parent.parent.parent if path.is_file() else path
    _setup_command_logging(
        workspace_hint,
        component="preprocess.spots.run",
        file_tag=_make_tag("run", path.stem, codebook_path.stem, roi or "all"),
        debug=debug,
        extra={
            "roi": roi,
            "round": round_num,
            "split": split,
            "calc_deviations": calc_deviations,
            "blank": blank,
            "highpass_only": highpass_only,
        },
    )

    # Defaults for parameters now typically provided via JSON
    subsample_z: int = 1
    limit_z: int | None = None

    # Merge project config if provided
    config = None
    if json_config is not None:
        try:
            config = load_config(json_config)
        except Exception as e:
            logger.warning(f"Failed to load JSON config {json_config}: {e}")

        # Project config does not override runtime flags here

    if calc_deviations and round_num is None:
        raise ValueError("Round must be provided for calculating deviations.")

    # only for optimize
    path_json = create_opt_path(
        path_img=path,
        codebook_path=codebook_path,
        mode="json",
        round_num=round_num,
        roi=roi,
    )
    if calc_deviations:
        decode = OPTIMIZE_DECODE
        logger.info("Optimizing. Using optimize config.")
        path_pickle = create_opt_path(
            path_img=path,
            codebook_path=codebook_path,
            mode="pkl",
            round_num=round_num,
            roi=roi,
        )
        if path_json.exists() and any(
            round_num == v.round_num for v in Deviations.validate_json(path_json.read_text())
        ):
            logger.info(f"Skipping {path.name}. Already done.")
            return

    else:
        (_path_out := path.parent / ("decoded-" + codebook_path.stem)).mkdir(exist_ok=True)
        path_pickle = _path_out / f"{path.stem}{f'-{split}' if split is not None else ''}.pkl"

    if not highpass_only and not calc_deviations and path_pickle.exists() and not overwrite:
        logger.info(f"Skipping {path.name}. Already exists.")
        return

    # Default decode config when not optimizing and not provided via JSON
    if decode is None:
        if config is not None and hasattr(config, "spot_decode"):
            decode = config.spot_decode  # type: ignore[attr-defined]
        else:
            decode = SpotDecodeConfig()

    logger.info(
        f"Running {path.parent.name}/{path.name} {f'split {split}' if split is not None else ''} with {limit_z} z-slices and {subsample_z}x subsampling."
    )
    with TiffFile(path) as tif:
        img_keys = tif.shaped_metadata[0]["key"]
        raw = tif.asarray()

    bit_mapping = {str(k): i for i, k in enumerate(img_keys)}

    codebook, used_bits, names, arr_zeroblank = load_codebook(
        codebook_path,
        exclude={
            "Malat1-201",
            "tdTomato-Ai9-extra",
        },  # , "Nfib-201", "Stmn1-201", "Ywhae-201", "Sox11-201", "Neurod6-201"},
        simple=simple,
        bit_mapping=bit_mapping,
    )

    used_bits = list(map(str, used_bits))

    # img = tif.asarray()[:, [bit_mapping[k] for k in used_bits]]
    # blurred = gaussian(path, sigma=8)
    # blurred = levels(blurred)  # clip negative values to 0.
    # filtered = image - blurred
    cut = 1024
    # 1998 - 1024 = 974
    split = int(split) if split is not None else None
    if split is None:
        split_slice = np.s_[:, :]
    elif split == 0:
        split_slice = np.s_[:cut, :cut]  # top-left
    elif split == 1:
        split_slice = np.s_[:cut, -cut:]  # top-right
    elif split == 2:
        split_slice = np.s_[-cut:, :cut]  # bottom-left
    elif split == 3:
        split_slice = np.s_[-cut:, -cut:]  # bottom-right
    else:
        raise ValueError(f"Unknown split {split}")

    slc = tuple(
        np.s_[
            ::subsample_z,
            [bit_mapping[k] for k in used_bits if k in bit_mapping],
        ]
    ) + tuple(split_slice)
    stack = make_fetcher(
        path,
        raw,
        slc,
        field_correct=field_correct,
        workspace_root=workspace_hint if not path.is_dir() else path,
        # if limit_z is None
        # else np.s_[
        #     :limit_z:subsample_z,
        #     [bit_mapping[k] for k in used_bits],
        # ],
    )
    # print(stack.xarray)

    match = re.match(r"^registered--(.+)\+(.*)$", path.parent.name)
    if match is None:
        raise ValueError(
            f"Path {path.parent.name} does not match expected format 'registered--<roi>+<codebook>'"
        )
    _roi, _codebook = match.groups()

    _slc_blank = tuple(np.s_[::subsample_z, :]) + tuple(split_slice)
    _stack_blank = (
        make_fetcher(
            path.parent.parent / f"registered--{_roi}+{blank}" / path.name,
            _slc_blank,
            max_proj=max_proj,
            field_correct=field_correct,
            workspace_root=workspace_hint if not path.is_dir() else path,
        )
        if blank is not None
        else None
    )
    del _roi, _codebook, _slc_blank

    # In all modes, data below 0 is set to 0.
    # We probably wouldn't need SATURATED_BY_IMAGE here since this is a subtraction operation.
    # But it's there as a reference.

    ghp = Filter.GaussianHighPass(
        sigma=decode.sigma[2],
        is_volume=True,
        level_method=Levels.SCALE_SATURATED_BY_IMAGE,
    )
    ghp.sigma = (
        decode.sigma[0],
        decode.sigma[1],
        decode.sigma[2],
    )  # z,y,x
    logger.debug(f"Running GHP with sigma {ghp.sigma}")

    imgs: ImageStack = ghp.run(stack)
    # blanks: ImageStack | None = ghp.run(_stack_blank) if _stack_blank is not None else None

    # --- Blank subtraction ---
    # if blanks is not None:
    # import polars as pl

    # bleed_path = (
    #     Path(cfg.bleedthrough_params_csv)
    #     if ("cfg" in locals() and cfg is not None and cfg.common.bleedthrough_params_csv)
    #     else (path.parent.parent / "robust_bleedthrough_params.csv")
    # )
    # df = pl.read_csv(bleed_path)
    # logger.info("Generating subtraction matrix for blanks.")
    # sub_mtx = generate_subtraction_matrix(blanks.xarray, df, used_bits)
    # # print(imgs.xarray)
    # # print(sub_mtx)
    # del blanks
    # logger.debug("Subtracting")
    # # Will automatically clip at 0.
    # if debug:
    #     logger.info(f"Keys: {used_bits}")
    #     tifffile.imwrite(
    #         path.parent.parent / f"{path.stem}_{codebook_path.stem}.hp.tif",
    #         imgs.xarray.to_numpy().squeeze().swapaxes(0, 1),  # ZCYX
    #         compression="zlib",
    #         metadata={"keys": img_keys},
    #     )
    #     tifffile.imwrite(
    #         path.parent.parent / f"{path.stem}_{codebook_path.stem}.blank.tif",
    #         sub_mtx.to_numpy().squeeze().swapaxes(0, 1),  # ZCYX
    #         compression="zlib",
    #         metadata={"keys": img_keys},
    #     )
    # ElementWiseAddition(sub_mtx).run(imgs, in_place=True)
    # if debug:
    #     tifffile.imwrite(
    #         path.parent.parent / f"{path.stem}_{codebook_path.stem}.hpsub.tif",
    #         imgs.xarray.to_numpy().squeeze().swapaxes(0, 1),  # ZCYX
    #         compression="zlib",
    #         metadata={"keys": img_keys},
    #     )
    # del sub_mtx

    if round_num == 0 and calc_deviations:
        logger.debug("Making scale file.")
        mins, base = initial(imgs, (1, 99.9))
        path_json.write_bytes(
            Deviations.dump_json(
                Deviations.validate_python([
                    {
                        "initial_scale": (1 / (base - mins)).tolist(),
                        "mins": mins.tolist(),
                        "round_num": 0,
                    }
                ])
            )
        )

    if highpass_only or (round_num == 0 and calc_deviations):
        (path.parent / "_highpassed").mkdir(exist_ok=True)

        out_path = path.parent / "_highpassed" / f"{path.stem}_{codebook_path.stem}.hp.tif"

        if overwrite or not out_path.exists():
            logger.debug(f"Writing to {out_path}")
            safe_imwrite(
                path.parent / "_highpassed" / f"{path.stem}_{codebook_path.stem}.hp.tif",
                imgs.xarray.to_numpy().squeeze().swapaxes(0, 1),  # ZCYX
                compression="zstd",
                metadata={"keys": img_keys},
            )
        return

    if not global_scale or not global_scale.exists():
        raise ValueError("Production run or round > 0 requires a global scale file.")

    # Scale factors
    scale_all = np.loadtxt(global_scale)
    if len(scale_all.shape) == 1:
        scale_all = scale_all[np.newaxis, :]

    scale_factor = (
        np.array(scale_all[round_num - 1], copy=True)
        if round_num is not None and calc_deviations
        else np.array(scale_all[-1], copy=True)
    )

    mins = np.loadtxt(global_scale.parent / "global_min.txt")

    try:
        scale(imgs, scale_factor, mins=mins)
    except ValueError:
        # global_scale.unlink()
        raise ValueError("Scale factor dim mismatch. Deleted. Please rerun.")

    # Zero out low norm
    try:
        perc = np.mean(
            list(
                json.loads(
                    (
                        path.parent.parent
                        / f"opt_{codebook_path.stem}{f'+{roi}' if roi and roi != '*' else ''}/percentiles.json"
                    ).read_text()
                )[(round_num - 1) if round_num is not None and calc_deviations else -1].values()
            )
        )
    except FileNotFoundError as e:
        raise Exception("Please run `fishtools find-threshold` first.") from e

    z_filt = Filter.ZeroByChannelMagnitude(perc, normalize=False)
    imgs = z_filt.run(imgs)

    # (path.parent / "_debug").mkdir(exist_ok=True)
    # tifffile.imwrite(
    #     path.parent / "_debug" / f"{path.stem}_{codebook_path.stem}-{split}.tif",
    #     imgs.xarray.to_numpy().squeeze().swapaxes(0, 1),  # ZCYX
    #     compression="zlib",
    #     metadata={"keys": img_keys},
    # )

    # Decode
    # if GPU and lock is None:
    #     raise Exception("GPU and lock are both None.")
    if not simple:
        decoded_spots, image_decoding_results = (
            pixel_decoding(imgs, decode, codebook) if not GPU else pixel_decoding_gpu(imgs, decode, codebook)
        )
        decoded_spots = decoded_spots.loc[decoded_spots[Features.PASSES_THRESHOLDS]]
    else:
        decoded_spots, image_decoding_results = spot_decoding(imgs, codebook), None

    spot_intensities = decoded_spots  # .loc[decoded_spots[Features.PASSES_THRESHOLDS]]
    genes, counts = np.unique(
        spot_intensities[Features.AXIS][Features.TARGET],
        return_counts=True,
    )
    gc = dict(zip(genes, counts))
    percent_blanks = sum([v for k, v in gc.items() if k.startswith("Blank")]) / counts.sum()
    logger.info(f"{percent_blanks} blank, Total: {counts.sum()}")

    # Deviations
    if calc_deviations and round_num is not None:
        # Round num already checked above.
        names_l = {n: i for i, n in enumerate(names)}
        try:
            if not len(spot_intensities):
                logger.warning("No spots found. Skipping.")
                append_json(
                    path_json,
                    round_num,
                    deviation=np.ones(len(scale_all)),
                    n=0,
                    percent_blanks=0,
                )
                return

            spot_intensities = spot_intensities[
                (
                    np.linalg.norm(spot_intensities.squeeze(), axis=1)
                    * (1 - spot_intensities.coords["distance"])
                )
                > decode.min_intensity
            ]
            idxs = list(map(names_l.get, spot_intensities.coords["target"].to_index().values))

            deviations = np.nanmean(
                spot_intensities.squeeze() * np.where(arr_zeroblank[idxs], 1, np.nan),
                axis=0,
            )
            logger.debug(f"Deviations: {np.round(deviations / np.nanmean(deviations), 4)}")
        except (np.exceptions.AxisError, ValueError):
            append_json(
                path_json,
                round_num,
                deviation=np.ones(len(scale_factor)),
                n=0,
                percent_blanks=0,
            )
        else:
            """Concatenate with previous rounds. Will overwrite rounds beyond the current one."""
            append_json(
                path_json,
                round_num,
                deviation=deviations,
                n=len(spot_intensities),
                percent_blanks=percent_blanks,
            )

    morph = (
        [
            {"area": prop.area, "centroid": prop.centroid}
            for prop in np.array(image_decoding_results.region_properties)
        ]
        if image_decoding_results
        else []
    )
    meta = {
        "fishtools_commit": git_hash(),
        "config": decode.model_dump(),
    }

    with path_pickle.open("wb") as f:
        pickle.dump((decoded_spots, morph, meta), f)

    return decoded_spots, image_decoding_results


def parse_duration(duration_str: str) -> timedelta:
    """Parses a duration string like '30m', '2h', '5d' into a timedelta."""
    match = re.fullmatch(r"(\d+)([mhd])", duration_str)
    if not match:
        raise ValueError(f"Invalid duration format: {duration_str}. Use 'Nm', 'Nh', or 'Nd'.")
    value, unit = match.groups()
    value = int(value)
    if unit == "m":
        return timedelta(minutes=value)
    elif unit == "h":
        return timedelta(hours=value)
    elif unit == "d":
        return timedelta(days=value)
    else:
        # This should not happen due to regex matching
        raise ValueError(f"Unknown time unit: {unit}")


@spots.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=Path))
@click.argument("roi", type=str, default="*")
@click.option(
    "--codebook",
    "codebook_path",
    type=click.Path(exists=True, dir_okay=False, file_okay=True, path_type=Path),
)
@click.option("--threads", "-t", type=int, default=13)
@click.option("--overwrite", is_flag=True)
@click.option("--simple", is_flag=True)
@click.option(
    "--since",
    type=str,
    default=None,
    help="Only process files modified since this duration (e.g., '30m', '2h', '1d').",
)
@click.option("--delete-corrupted", is_flag=True)
@click.option("--split", is_flag=True, help="Whether to split into quadrants. True regardless.")
@click.option("--local-opt", is_flag=True)
@click.option("--blank", type=str, default=None)
@click.option(
    "--stagger",
    type=float,
    default=5.0,
    help="Seconds to stagger starts per worker slot",
)
@click.option(
    "--stagger-jitter",
    type=float,
    default=0.0,
    help="Add random jitter in [0..x] seconds to each start",
)
@click.option(
    "--config",
    "json_config",
    type=click.Path(exists=True, dir_okay=False, file_okay=True, path_type=Path),
    default=None,
    help="Optional project config (JSON). CLI flags override JSON.",
)
@click.option(
    "--field-correct/--no-field-correct",
    default=False,
    help="Apply illumination field correction using discovered TCYX field stores.",
)
def batch(
    path: Path,
    roi: str,
    codebook_path: Path,
    threads: int = 13,
    overwrite: bool = False,
    simple: bool = False,
    split: bool = False,
    since: str | None = None,
    delete_corrupted: bool = False,
    local_opt: bool = False,
    blank: str | None = None,
    json_config: Path | None = None,
    stagger: float = 0.0,
    stagger_jitter: float = 0.0,
    field_correct: bool = False,
):
    _setup_command_logging(
        path,
        component="preprocess.spots.batch",
        file_tag=_make_tag("batch", roi, codebook_path.stem),
        extra={
            "roi": roi,
            "codebook": codebook_path.stem,
            "threads": threads,
            "overwrite": overwrite,
            "simple": simple,
            "local_opt": local_opt,
        },
    )
    split = True  # Intentional override. --split kept for backward compatibility
    workspace = Workspace(path)
    roi_filter = None if roi == "*" else [roi]
    file_map, _ = workspace.registered_file_map(codebook_path.stem, rois=roi_filter)
    all_paths = {p for paths in file_map.values() for p in paths}
    paths_in_scope = all_paths  # Start with all paths

    if delete_corrupted:
        logger.info("Checking all files for corruption.")
        for i, p in enumerate(sorted(all_paths)):
            if i % 100 == 0:
                logger.info(f"Checked {i}/{len(all_paths)} files.")
            try:
                Workspace.ensure_tiff_readable(p)
            except CorruptedTiffError as error:
                logger.warning(f"Error reading {p}: {error}. Deleting.")
                p.unlink()
                all_paths.discard(p)
                continue

    # Filter by modification time if --since is provided
    if since:
        try:
            duration = parse_duration(since)
        except ValueError as e:
            logger.error(f"Error parsing --since value: {e}")
            return  # Or raise click.BadParameter

        current_time = time.time()
        cutoff_time = current_time - duration.total_seconds()
        recent_paths = {p for p in all_paths if p.stat().st_mtime > cutoff_time}
        logger.info(
            f"Found {len(recent_paths)} files modified since {since} (out of {len(all_paths)} total)."
        )
        paths_in_scope = recent_paths  # Update paths_in_scope to only include recent ones
    else:
        logger.info(f"Found {len(paths_in_scope)} files matching pattern (no --since filter).")

    already_done = set()
    # Check which of the files *in scope* are already done
    for p in paths_in_scope:
        # Check if 4 corresponding pkl files exist for this tif file if split is True
        num_expected_pkl = 4 if split else 1
        glob_pattern = f"decoded-{codebook_path.stem}/{p.stem}{'-*' if split else ''}.pkl"
        # Use parent.parent because the decoded files are one level up
        if len(list(p.parent.glob(glob_pattern))) == num_expected_pkl:
            already_done.add(p)

    if overwrite:
        # Overwrite only applies to files within the scope (all or recent)
        paths_to_process = sorted(list(paths_in_scope))
        logger.info(f"Processing {len(paths_to_process)} files in scope (overwrite enabled).")
        # Log how many of these were already done but will be overwritten
        overwritten_count = len(already_done)
        if overwritten_count > 0:
            logger.info(f"Overwriting {overwritten_count} already processed files within the scope.")
    else:
        # Process only files in scope that are not already done
        paths_to_process = sorted(list(paths_in_scope - already_done))
        skipped_count = len(paths_in_scope) - len(paths_to_process)
        if skipped_count > 0:
            logger.info(f"Skipping {skipped_count} already processed files within the scope.")
        logger.info(f"Processing {len(paths_to_process)} new files in scope.")

    if field_correct:
        rois_needed = {p.parent.name.split("--", 1)[1].split("+", 1)[0] for p in paths_to_process}
        _ensure_field_stores(workspace, sorted(rois_needed), codebook_path.stem)

    if not paths_to_process:
        logger.warning("No files found to process based on current filters and overwrite status.")
        return

    if local_opt:
        return _batch(
            paths_to_process,
            "run",
            [
                "--global-scale",
                (path / f"opt_{codebook_path.stem}" / "global_scale.txt").as_posix(),
                "--codebook",
                codebook_path.as_posix(),
                "--overwrite" if overwrite else "",
                "--simple" if simple else "",
                f"--roi={roi}" if roi else "",
                f"--blank={blank}" if blank else "",
                *(["--config", json_config.as_posix()] if json_config else []),
                *(["--field-correct"] if field_correct else []),
            ],
            threads=threads,
            split=split,
            stagger=stagger,
            stagger_jitter=stagger_jitter,
        )

    for _roi, _paths in groupby(paths_to_process, lambda x: x.parent.name.split("--")[1].split("+")[0]):
        _batch(
            sorted(_paths),
            "run",
            [
                "--global-scale",
                (path / f"opt_{codebook_path.stem}" / "global_scale.txt").as_posix(),
                "--codebook",
                codebook_path.as_posix(),
                "--overwrite" if overwrite else "",
                "--simple" if simple else "",
                f"--blank={blank}" if blank else "",
                *(["--config", json_config.as_posix()] if json_config else []),
                *(["--field-correct"] if field_correct else []),
            ],
            threads=threads,
            split=split,
            stagger=stagger,
            stagger_jitter=stagger_jitter,
        )


spots.add_command(optimize)
spots.add_command(stitch)
spots.add_command(overlay)
spots.add_command(threshold)

if __name__ == "__main__":
    spots()
# %%


def plot_decoded(spots):
    plt.scatter(spots.coords["x"], spots.coords["y"], s=0.1, c="red")
    # plt.imshow(prop_results.label_image[0])
    plt.title("PixelSpotDecoder Labeled Image")
    plt.axis("off")


# .where(spot_intensities.target == "Gad2-201", drop=True)
# plot_decoded(spot_intensities)


# %%

# %%


def plot_bits(spots, bits: Sequence[int], codebook: Mapping[str, Sequence[int]]):
    reference = np.zeros((len(spots), max(bits) + 1))
    for i, target in enumerate(spots.target.values):
        arr = codebook.get(target)
        if arr is None:
            continue
        for a in arr:
            reference[i, a - 1] = 1

    fig, axs = plt.subplots(figsize=(2, 10), ncols=2, dpi=200)
    axs = axs.flatten()
    axs[0].imshow(spots.squeeze())
    axs[0].axis("off")
    axs[1].imshow(reference)
    axs[1].axis("off")
    return fig, axs


# plot_bits(spot_intensities.where(spot_intensities.target == "Neurod6-201", drop=True))
# plot_bits(spot_intensities[:200])
# %%


# %%

# %%


# df = pd.DataFrame(
#     {
#         "target": spot_intensities.coords["target"],
#         "x": spot_intensities.coords["x"],
#         "y": spot_intensities.coords["y"],
#     }
# )


# sns.scatterplot(data=df[df["target"] == "Neurog2-201"], x="x", y="y", hue="target", s=10, legend=True)

# %%

# # Example of how to access the spot attributes
# print(f"The area of the first spot is {prop_results.region_properties[0].area}")

# # View labeled image after connected componenet analysis
# # View decoded spots overlaid on max intensity projected image
# single_plane_max = imgs.reduce({Axes.ROUND, Axes.CH, Axes.ZPLANE}, func="max")
# fig, axs = plt.subplots(ncols=2, figsize=(10, 5), dpi=200)
# axs[1].imshow(prop_results.label_image[0], vmax=1)
# axs[1].set_title("Decoded", loc="left")


# axs[0].imshow(single_plane_max.xarray[0].squeeze(), cmap="gray")
# axs[0].set_title("Raw", loc="left")
# for ax in axs:
#     ax.axis("off")
# fig.tight_layout()
# Uncomment code below to view spots
# %gui qt
# viewer = display(stack=single_plane_max, spots=spot_intensities)


# %%
# %%
# import seaborn as sns
# from starfish import IntensityTable


# def compute_magnitudes(stack, norm_order=2):
#     pixel_intensities = IntensityTable.from_image_stack(stack)
#     feature_traces = pixel_intensities.stack(traces=(Axes.CH.value, Axes.ROUND.value))
#     norm = np.linalg.norm(feature_traces.values, ord=norm_order, axis=1)

#     return norm


# mags = compute_magnitudes(imgs)

# plt.hist(mags, bins=200)
# sns.despine(offset=3)
# plt.xlabel("Barcode magnitude")
# plt.ylabel("Number of pixels")
# plt.yscale("log")
# plt.xscale("log")
# %%
# %%
# from starfish.core.intensity_table.decoded_intensity_table import DecodedIntensityTable
# from starfish.core.intensity_table.intensity_table import IntensityTable

# pixel_intensities = IntensityTable.from_image_stack(clipped_both_scaled.reduce("z", func="max"))
