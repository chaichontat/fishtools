from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Iterable as TypingIterable

import polars as pl

import numpy as np
import zarr
from loguru import logger
from skimage.measure import regionprops_table

from fishtools.io.workspace import Workspace

DEFAULT_REGIONPROPS: Final[tuple[str, ...]] = (
    "label",
    "area",
    "centroid",
    "bbox",
    "mean_intensity",
    "max_intensity",
    "min_intensity",
)


def load_slice(zarr_path: Path, idx: int) -> np.ndarray | None:
    """Load a specific Z-slice from a Zarr store.

    Args:
        zarr_path: Path to a 3D/4D Zarr array where axis 0 is Z.
        idx: Z index to read.

    Returns:
        The requested slice as a NumPy array, or None on failure (error logged).
    """
    logger.info(f"Slice {idx}: Loading from {zarr_path}...")
    try:
        img_stack = zarr.open_array(str(zarr_path), mode="r")
        img_slice = img_stack[idx]
        logger.info(f"Slice {idx}: Loaded slice with shape {img_slice.shape}.")
        return img_slice
    except Exception as e:  # pragma: no cover - IO failure path
        logger.error(f"Slice {idx}: Failed to load Zarr slice: {e}")
        return None


def _normalize_channel_names(keys: TypingIterable[str] | None) -> list[str]:
    if keys is None:
        raise ValueError("Intensity Zarr is missing attrs['key'] for channel names")
    if isinstance(keys, (str, bytes)):
        raise ValueError("attrs['key'] must be an iterable of channel names, not a string")
    return [str(k) for k in keys]


def slice_intensity_channel(arr: zarr.Array, z_index: int, channel_name: str) -> np.ndarray:
    """Return a 2D intensity slice for the requested channel from a Zarr array."""

    keys = _normalize_channel_names(arr.attrs.get("key"))
    if channel_name not in keys:
        raise ValueError(f"Channel {channel_name!r} not found in available keys: {keys}")

    c_idx = int(keys.index(channel_name))

    if arr.ndim == 3:
        if len(keys) > 1 and c_idx != 0:
            raise ValueError(
                f"3D intensity with multiple channel names {keys}; cannot select {channel_name!r}"
            )
        return arr[z_index]

    if arr.ndim != 4:
        raise ValueError(f"Expected 3D or 4D intensity array, got shape {arr.shape}")

    if arr.shape[-1] == len(keys):
        return arr[z_index, :, :, c_idx]
    if arr.shape[1] == len(keys):
        return arr[z_index, c_idx, :, :]

    raise ValueError(f"Cannot infer channel axis from shape {arr.shape} and keys length {len(keys)}")


def compute_regionprops_table(
    seg_mask: np.ndarray,
    *,
    intensity_image: np.ndarray | None = None,
    properties: Sequence[str] = DEFAULT_REGIONPROPS,
) -> pl.DataFrame:
    """Compute a Polars DataFrame of region properties for a single slice."""

    props_table = regionprops_table(seg_mask, intensity_image=intensity_image, properties=properties)
    df = pl.DataFrame(props_table)
    if "label" in df.columns:
        df = df.with_columns(pl.col("label").cast(pl.UInt32))
    return df


def write_regionprops_parquet(
    df: pl.DataFrame,
    output_dir: Path,
    channel: str,
    idx: int,
    *,
    overwrite: bool = False,
) -> Path:
    """Persist regionprops DataFrame to `<output_dir>/intensity_{channel}/intensity-XX.parquet`."""

    props_path = output_dir / f"intensity_{channel}" / f"intensity-{idx:02d}.parquet"
    props_path.parent.mkdir(exist_ok=True, parents=True)

    if not overwrite and props_path.exists():
        logger.info(
            f"Slice {idx} [channel={channel}]: Skipping existing {props_path}"
        )
        return props_path

    df.write_parquet(props_path)
    logger.info(
        f"Slice {idx} [channel={channel}]: Saved region properties ({len(df)} regions) "
        f"to {props_path}"
    )
    return props_path


def process_slice_regionprops(
    idx: int,
    segmentation_zarr_path: Path,
    intensity_zarr_path: Path,
    channel: str,
    output_dir: Path,
    overwrite: bool = False,
) -> None:
    """Compute per-label intensity properties for a single Z slice and save parquet."""

    props_path = output_dir / f"intensity_{channel}" / f"intensity-{idx:02d}.parquet"
    if not overwrite and props_path.exists():
        logger.info(f"Slice {idx} [channel={channel}]: Skipping, output files already exist.")
        return

    intensity_arr = zarr.open_array(str(intensity_zarr_path), mode="r")
    intensity_img = slice_intensity_channel(intensity_arr, idx, channel)

    seg_mask = load_slice(segmentation_zarr_path, idx)
    if seg_mask is None:
        raise RuntimeError(f"Slice {idx} [channel={channel}]: Failed to load segmentation slice.")

    if seg_mask.shape != intensity_img.shape:
        raise ValueError(
            f"Slice {idx}: Shape mismatch between segmentation ({seg_mask.shape}) "
            f"and intensity ({intensity_img.shape}). Cannot calculate intensity props."
        )

    logger.info(f"Slice {idx} [channel={channel}]: Calculating region properties...")
    df = compute_regionprops_table(seg_mask, intensity_image=intensity_img)
    write_regionprops_parquet(df, output_dir, channel, idx, overwrite=overwrite)


@dataclass(slots=True)
class StitchPaths:
    """Normalized paths for a stitched ROI+segmentation codebook pair."""

    workspace: Workspace
    roi: str
    seg_codebook: str

    @property
    def stitch_root(self) -> Path:
        return self.workspace.deconved / f"stitch--{self.roi}+{self.seg_codebook}"

    def segmentation(self, name: str = "output_segmentation.zarr") -> Path:
        return self.stitch_root / name

    def intensity_store(self, name: str = "fused.zarr") -> Path:
        return self.stitch_root / name

    def intensity_dir(self, channel: str) -> Path:
        return self.stitch_root / f"intensity_{channel}"

    def chunks_dir(self, codebook: str) -> Path:
        return self.stitch_root / f"chunks+{codebook}"

    @classmethod
    def from_workspace(cls, workspace: Path | Workspace, roi: str, seg_codebook: str) -> "StitchPaths":
        ws = workspace if isinstance(workspace, Workspace) else Workspace(workspace)
        return cls(workspace=ws, roi=roi, seg_codebook=seg_codebook)

    @classmethod
    def from_stitch_root(
        cls,
        stitch_root: Path,
        *,
        workspace: Path | Workspace | None = None,
    ) -> "StitchPaths":
        name = stitch_root.name
        if not name.startswith("stitch--") or "+" not in name:
            raise ValueError(
                f"Expected stitch root in form 'stitch--<roi>+<seg_codebook>', got {stitch_root}"
            )
        roi, seg_codebook = name.removeprefix("stitch--").split("+", 1)
        if workspace is not None:
            return cls.from_workspace(workspace, roi, seg_codebook)

        errors: list[Exception] = []
        candidate_roots = [stitch_root.parent, *stitch_root.parents[1:3]]
        for candidate in candidate_roots:
            if candidate is None:
                continue
            try:
                return cls.from_workspace(candidate, roi, seg_codebook)
            except ValueError as err:
                errors.append(err)

        error_hint = errors[-1] if errors else None
        raise ValueError(
            f"Unable to resolve workspace for stitch directory {stitch_root}."
        ) from error_hint


def resolve_intensity_store(
    stitch: "StitchPaths",
    intensity_name: str,
    *,
    store_name: str = "fused.zarr",
) -> Path:
    """Resolve an intensity Zarr path from a codebook label."""

    workspace = stitch.workspace
    roi = stitch.roi
    search_labels = [intensity_name]
    sanitized = workspace.sanitize_codebook_name(intensity_name)
    if sanitized not in search_labels:
        search_labels.append(sanitized)

    searched_paths: list[Path] = []
    for label in search_labels:
        candidate = workspace.stitch(roi, label) / store_name
        searched_paths.append(candidate)
        if candidate.exists():
            return candidate

    searched = ", ".join(str(p) for p in searched_paths)
    raise FileNotFoundError(
        f"Could not resolve intensity store for codebook '{intensity_name}'. "
        f"Searched: {searched}"
    )


def load_intensity_parquet(stitch: StitchPaths, channel: str) -> pl.DataFrame:
    """Collect all intensity parquet shards for the given channel."""

    intensity_dir = stitch.intensity_dir(channel)
    if not intensity_dir.exists():
        raise FileNotFoundError(f"Intensity directory not found: {intensity_dir}")

    scan = pl.scan_parquet(
        (intensity_dir / "intensity-*.parquet").as_posix(),
        include_file_paths="path",
        missing_columns="insert",
    )
    return scan.collect()
