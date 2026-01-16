from __future__ import annotations

import re
import time
from datetime import timedelta
from pathlib import Path
from typing import Literal

import numpy as np
import rich_click as click
from loguru import logger
from pydantic import BaseModel
from starfish import Codebook, ImageStack
from starfish.core.intensity_table.decoded_intensity_table import DecodedIntensityTable
from starfish.core.intensity_table.intensity_table import IntensityTable
from starfish.core.types import Axes, Features, SpotAttributes, SpotFindingResults
from starfish.image import Filter
from starfish.spots import FindSpots
from tifffile import TiffFile

from fishtools.utils.pretty_print import progress_bar, progress_bar_threadpool, run_subprocess_streaming
from fishtools.utils.logging import setup_cli_logging
from fishtools.utils.tiff import read_metadata_from_tif
from fishtools.utils.utils import git_hash
from fishtools.preprocess.stitching import spot_split_cut_px

_REGISTERED_DIR = re.compile(r"^registered--(.+)\+(.+)$")


class BlobDetectorParams(BaseModel):
    min_sigma: float = 1.0
    max_sigma: float = 2.0
    num_sigma: int = 4
    threshold: float = 0.05
    overlap: float = 0.5
    measurement_type: Literal["mean", "max", "median", "min"] = "mean"
    is_volume: bool = True
    detector_method: str = "blob_log"
    exclude_border: bool | int | tuple[int, ...] = False


DEFAULT_BLOB_DETECTOR_JSON = BlobDetectorParams().model_dump_json()


class NoSpotsFoundError(RuntimeError):
    def __init__(self, path: Path, *, split: int) -> None:
        super().__init__(f"No spots found for {path.name} split={split}")
        self.path = path
        self.split = split


def _parse_duration(duration_str: str) -> timedelta:
    m = re.fullmatch(r"(\d+)([mhd])", duration_str)
    if m is None:
        raise ValueError(f"Invalid duration format: {duration_str}. Use 'Nm', 'Nh', or 'Nd'.")
    n, unit = int(m.group(1)), m.group(2)
    match unit:
        case "m":
            return timedelta(minutes=n)
        case "h":
            return timedelta(hours=n)
        case "d":
            return timedelta(days=n)
    raise AssertionError(f"Unexpected duration unit: {unit!r}")


def _split_slices(*, split: int, cut: int) -> tuple[slice, slice]:
    splits: list[tuple[slice, slice]] = [
        (slice(None, cut), slice(None, cut)),
        (slice(None, cut), slice(-cut, None)),
        (slice(-cut, None), slice(None, cut)),
        (slice(-cut, None), slice(-cut, None)),
    ]
    if split < 0 or split >= len(splits):
        raise ValueError(f"Unknown split={split}, expected 0-3.")
    return splits[split]


def _channel_names(raw: np.ndarray, tif: TiffFile, *, path_tif: Path) -> list[str]:
    if raw.ndim != 4:
        raise ValueError(f"Expected a ZCYX array, got shape={raw.shape}")
    metadata = read_metadata_from_tif(tif)
    n_ch = int(raw.shape[1])
    raw_names = metadata.get("key") or metadata.get("channel_names") or metadata.get("channels")
    if raw_names is None:
        raise ValueError(
            f"Missing channel names in TIFF metadata for {path_tif}. "
            f"Expected a list of length {n_ch} under metadata['key']."
        )
    if isinstance(raw_names, str):
        names = [raw_names]
    elif isinstance(raw_names, (list, tuple)):
        names = list(raw_names)
    elif isinstance(raw_names, np.ndarray):
        names = raw_names.tolist()
    else:
        raise ValueError(
            f"Unsupported channel name format in TIFF metadata for {path_tif}: {type(raw_names).__name__}"
        )
    if len(names) != n_ch:
        raise ValueError(
            f"Channel name count mismatch for {path_tif}: expected {n_ch} names, got {len(names)}."
        )
    return [str(name) for name in names]


def _identity_codebook(channel_names: list[str]) -> Codebook:
    if len(channel_names) != len(set(channel_names)):
        dupes = sorted({name for name in channel_names if channel_names.count(name) > 1})
        raise ValueError(f"Channel names must be unique to infer targets; duplicates: {', '.join(dupes)}")
    n_ch = len(channel_names)
    data = np.zeros((n_ch, 1, n_ch), dtype=bool)
    for i in range(n_ch):
        data[i, 0, i] = True
    return Codebook.from_numpy(np.array(channel_names), n_round=1, n_channel=n_ch, data=data)


def _simple_lookup_decode(spots: SpotFindingResults, *, codebook: Codebook) -> DecodedIntensityTable:
    # Starfish's SimpleLookupDecoder + build_traces_sequential, but filters
    # empty/all-NA frames before pd.concat (Pandas FutureWarning).
    import pandas as pd

    lookup_table: dict[tuple[int, int], str] = {}
    for target in codebook[Features.TARGET]:
        for ch_label in codebook[Axes.CH.value]:
            for round_label in codebook[Axes.ROUND.value]:
                if codebook.loc[target, round_label, ch_label]:
                    lookup_table[(int(round_label), int(ch_label))] = str(target.values)

    for r_ch_index, results in spots.items():
        results.spot_attrs.data[Features.TARGET] = lookup_table[r_ch_index] if r_ch_index in lookup_table else "nan"

    frames: list[pd.DataFrame] = []
    for per in spots.values():
        df = per.spot_attrs.data
        if df.empty or not df.notna().to_numpy().any():
            continue
        frames.append(df)

    if frames:
        all_spots = pd.concat(frames, ignore_index=True, sort=True)
    else:
        # Keep the expected columns/dtypes for SpotAttributes validation.
        sample = next(iter(spots.values())).spot_attrs.data
        all_spots = sample.iloc[0:0].copy()

    all_spots["spot_id"] = all_spots.index

    intensities = IntensityTable.zeros(
        spot_attributes=SpotAttributes(all_spots),
        ch_labels=spots.ch_labels,
        round_labels=spots.round_labels,
    )

    i = 0
    for (r, c), per in spots.items():
        for _, row in per.spot_attrs.data.iterrows():
            intensities.loc[dict(features=i, c=c, r=r)] = row[Features.INTENSITY]
            i += 1

    return DecodedIntensityTable(intensities)


def _morph_from_decoded(
    decoded: DecodedIntensityTable,
    *,
    area_radius_scale: float = 2.5,
) -> list[dict[str, object]]:
    spot_ids = np.asarray(decoded.coords["spot_id"].values, dtype=int)
    xs = np.asarray(decoded.coords["x"].values, dtype=float)
    ys = np.asarray(decoded.coords["y"].values, dtype=float)
    zs = (
        np.asarray(decoded.coords["z"].values, dtype=float) if "z" in decoded.coords else np.zeros_like(xs, dtype=float)
    )
    radii = np.asarray(decoded.coords["radius"].values) if "radius" in decoded.coords else np.ones_like(xs)

    n = int(decoded.sizes.get("features", len(xs)))
    morph: list[dict[str, object]] = [{} for _ in range(n)]
    for spot_id, x, y, z, radius in zip(spot_ids, xs, ys, zs, radii, strict=True):
        area = float(np.pi * float(area_radius_scale * radius) ** 2)
        morph[int(spot_id)] = {"area": area, "centroid": (float(z), float(y), float(x))}
    return morph


def _run_single_split(
    path_tif: Path,
    *,
    split: int,
    codebook_label: str,
    blob_detector_json: str,
    overwrite: bool,
    tophat_radius: int,
    area_radius_scale: float,
) -> tuple[Path, int]:
    match = _REGISTERED_DIR.match(path_tif.parent.name)
    if match is None:
        raise ValueError(
            f"Tile parent directory must match 'registered--<roi>+<codebook>', got {path_tif.parent.name}"
        )
    _roi, codebook_from_dir = match.groups()
    if codebook_from_dir != codebook_label:
        raise ValueError(
            f"Tile {path_tif} belongs to codebook '{codebook_from_dir}' but CLI requested '{codebook_label}'"
        )

    out_dir = path_tif.parent / f"decoded-{codebook_label}"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{path_tif.stem}-{split}.pkl"

    if out_path.exists() and not overwrite:
        return out_path, -1

    with TiffFile(path_tif) as tif:
        raw_u16 = tif.asarray()
        channel_names = _channel_names(raw_u16, tif, path_tif=path_tif)

    cut = spot_split_cut_px(int(raw_u16.shape[-1]))
    y_slc, x_slc = _split_slices(split=split, cut=cut)
    cropped = raw_u16[:, :, y_slc, x_slc]
    raw = (cropped.astype(np.float32, copy=False) / 65535.0).clip(0.0, 1.0)

    # Starfish expects RCZYX
    stack = ImageStack.from_numpy(raw[np.newaxis, ...].transpose(0, 2, 1, 3, 4))

    # Spot_finder.py default preprocessing
    if tophat_radius > 0:
        Filter.WhiteTophat(int(tophat_radius), is_volume=False).run(stack, in_place=True)

    blob = BlobDetectorParams.model_validate_json(blob_detector_json)
    blob_detector_config = blob.model_dump()
    spots = FindSpots.BlobDetector(**blob_detector_config).run(image_stack=stack)

    decoded = _simple_lookup_decode(spots, codebook=_identity_codebook(channel_names))

    n_features = int(decoded.sizes.get("features", 0))

    decoded = decoded.assign_coords(
        distance=("features", np.zeros(n_features, dtype=np.float32)),
        passes_thresholds=("features", np.ones(n_features, dtype=bool)),
    )

    morph = _morph_from_decoded(decoded, area_radius_scale=area_radius_scale)
    meta: dict[str, object] = {
        "fishtools_commit": git_hash(),
        "config": {
            "blob_detector": blob_detector_config,
            "tophat_radius": int(tophat_radius),
            "area_radius_scale": float(area_radius_scale),
        },
    }

    import pickle

    with out_path.open("wb") as f:
        pickle.dump((decoded, morph, meta), f)

    return out_path, n_features


@click.command("simple-batch")
@click.argument("path", type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=Path))
@click.argument("roi", type=str, default="*")
@click.option(
    "--codebook",
    "codebook_path",
    type=click.Path(exists=True, dir_okay=False, file_okay=True, path_type=Path),
    required=True,
)
@click.option("--threads", "-t", type=int, default=8)
@click.option("--overwrite", is_flag=True)
@click.option(
    "--since",
    type=str,
    default=None,
    help="Only process tiles modified since this duration (e.g., '30m', '2h', '1d').",
)
@click.option(
    "--blob-detector",
    "blob_detector_json",
    type=str,
    default=DEFAULT_BLOB_DETECTOR_JSON,
    show_default=True,
    help="JSON dict of Starfish BlobDetector parameters.",
)
@click.option(
    "--tophat-radius",
    type=int,
    default=15,
    show_default=True,
    help="WhiteTophat radius (set to 0 to disable).",
)
@click.option(
    "--area-radius-scale",
    type=float,
    default=2.5,
    show_default=True,
    help="Area = pi*(area-radius-scale*radius)^2 for downstream spotlook filtering.",
)
def simple_batch(
    path: Path,
    roi: str,
    codebook_path: Path,
    *,
    threads: int,
    overwrite: bool,
    since: str | None,
    blob_detector_json: str,
    tophat_radius: int,
    area_radius_scale: float,
) -> None:
    """Call spots using Starfish BlobDetector and infer targets from channel names.

    Writes per-tile split pickles in the same schema as `preprocess spots batch`,
    so `preprocess spots stitch` can consume them as a drop-in replacement.
    """

    setup_cli_logging(
        path,
        component="preprocess.spots.simple_batch",
        file=f"simple-batch-{roi}-{codebook_path.stem}",
        extra={
            "roi": roi,
            "codebook": codebook_path.stem,
            "threads": threads,
            "overwrite": overwrite,
        },
    )

    from fishtools.io.workspace import Workspace

    ws = Workspace(path)
    roi_filter = None if roi == "*" else [roi]
    file_map, _missing = ws.registered_file_map(codebook_path.stem, rois=roi_filter)
    all_paths = sorted({p for paths in file_map.values() for p in paths})

    if since:
        cutoff_time = time.time() - _parse_duration(since).total_seconds()
        all_paths = [p for p in all_paths if p.stat().st_mtime > cutoff_time]

    if not all_paths:
        logger.warning("No registered tiles found to process.")
        return

    split_list = [0, 1, 2, 3]

    tiles_to_process: list[Path] = []
    for tile in all_paths:
        out_dir = tile.parent / f"decoded-{codebook_path.stem}"
        expected = [out_dir / f"{tile.stem}-{s}.pkl" for s in split_list]
        if overwrite or any(not p.exists() for p in expected):
            tiles_to_process.append(tile)

    if not tiles_to_process:
        logger.info("All tiles already processed (use --overwrite to rerun).")
        return

    logger.info(
        f"Processing {len(tiles_to_process)} tiles across {len(all_paths)} registered tiles "
        f"({len(tiles_to_process) * len(split_list)} split tasks)."
    )

    script = Path(__file__).resolve()

    with progress_bar_threadpool(
        len(tiles_to_process) * len(split_list),
        threads=int(threads),
        stop_on_exception=False,
    ) as submit:
        idx = 0
        for tile in tiles_to_process:
            for split in split_list:
                slot = idx % max(1, int(threads))
                cmd = [
                    "python",
                    str(script),
                    "simple",
                    str(tile),
                    "--codebook",
                    codebook_path.as_posix(),
                    "--split",
                    str(split),
                    "--blob-detector",
                    blob_detector_json,
                    "--tophat-radius",
                    str(tophat_radius),
                    "--area-radius-scale",
                    str(area_radius_scale),
                    *([] if not overwrite else ["--overwrite"]),
                ]
                submit(run_subprocess_streaming, cmd, thread_index=slot + 1, check=True, emit_to_console=True)
                idx += 1


@click.command("simple")
@click.argument("path", type=click.Path(exists=True, dir_okay=False, file_okay=True, path_type=Path))
@click.option(
    "--codebook",
    "codebook_path",
    type=click.Path(exists=True, dir_okay=False, file_okay=True, path_type=Path),
    required=True,
)
@click.option("--overwrite", is_flag=True)
@click.option(
    "--split",
    type=int,
    default=None,
    help="Quadrant split index (0-3). If omitted, runs all splits.",
)
@click.option(
    "--blob-detector",
    "blob_detector_json",
    type=str,
    default=DEFAULT_BLOB_DETECTOR_JSON,
    show_default=True,
    help="JSON dict of Starfish BlobDetector parameters.",
)
@click.option(
    "--tophat-radius",
    type=int,
    default=15,
    show_default=True,
    help="WhiteTophat radius (set to 0 to disable).",
)
@click.option(
    "--area-radius-scale",
    type=float,
    default=2.5,
    show_default=True,
    help="Area = pi*(area-radius-scale*radius)^2 for downstream spotlook filtering.",
)
def simple(
    path: Path,
    *,
    codebook_path: Path,
    overwrite: bool,
    split: int | None,
    blob_detector_json: str,
    tophat_radius: int,
    area_radius_scale: float,
) -> None:
    """Run BlobDetector spot calling for a single registered tile."""

    setup_cli_logging(
        path.parent.parent.parent,
        component="preprocess.spots.simple",
        file=f"simple-{path.stem}-{codebook_path.stem}",
        extra={
            "codebook": codebook_path.stem,
            "tile": path.stem,
            "overwrite": overwrite,
            "split": split,
        },
    )

    if split is not None and split not in {0, 1, 2, 3}:
        raise click.BadParameter("Expected --split in {0,1,2,3} or omitted.", param_hint="--split")

    split_list = [split] if split is not None else [0, 1, 2, 3]

    def run_split(s: int) -> None:
        out, n_spots = _run_single_split(
            path,
            split=s,
            codebook_label=codebook_path.stem,
            blob_detector_json=blob_detector_json,
            overwrite=overwrite,
            tophat_radius=tophat_radius,
            area_radius_scale=area_radius_scale,
        )

        if n_spots < 0:
            logger.info(f"{path.name} split={s}: skipped (already decoded)")
        else:
            logger.info(f"{path.name} split={s}: discovered_spots={n_spots}")
            logger.debug(f"Wrote {out}")

    if split is not None:
        run_split(split)
        return

    with progress_bar(len(split_list)) as advance:
        for s in split_list:
            try:
                run_split(s)
            finally:
                advance()


@click.group()
def _main() -> None:
    """Standalone entrypoint for subprocess execution."""


_main.add_command(simple_batch)
_main.add_command(simple)


if __name__ == "__main__":
    _main()
