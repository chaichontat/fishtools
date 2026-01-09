from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

import numpy as np
import polars as pl
from loguru import logger

from fishtools.io.workspace import Workspace

if TYPE_CHECKING:  # pragma: no cover - typing only
    import anndata


def _list_segmentation_artifacts(seg_zarr_path: Path) -> tuple[list[str], list[str]]:
    """List segmentation export sidecar directories.

    - `chunks+<codebook>`: created by `segment overlay spots`, holds `ident_*.parquet` and `polygons_*.parquet`.
    - `intensity_<channel>`: created by `segment overlay intensity`, holds `intensity-*.parquet`.
    """

    chunks: list[str] = []
    intensities: list[str] = []
    if not seg_zarr_path.exists():
        return (chunks, intensities)

    for entry in seg_zarr_path.iterdir():
        if not entry.is_dir():
            continue
        if entry.name.startswith("chunks+"):
            chunks.append(entry.name.removeprefix("chunks+"))
            continue
        if entry.name.startswith("intensity_"):
            intensities.append(entry.name.removeprefix("intensity_"))
            continue

    return (sorted(set(chunks)), sorted(set(intensities)))


def _explain_export_artifacts() -> str:
    return (
        "Artifacts:\n"
        "- chunks+<codebook>/ident_*.parquet: spot→cell assignments used for gene counts.\n"
        "- chunks+<codebook>/polygons_*.parquet: per-cell geometry (centroids/area).\n"
        "- intensity_<channel>/intensity-*.parquet: per-cell intensity stats.\n"
    )


@dataclass(frozen=True)
class _IntensityKey:
    roi: str
    channel: str


def _discover_channels(seg_zarr_path: Path) -> list[str]:
    """Discover channels by scanning intensity_* subdirectories inside segmentation zarr.

    Returns a sorted list of channel names.
    """
    channels: list[str] = []
    if not seg_zarr_path.exists():
        return channels
    for p in seg_zarr_path.iterdir():
        if p.is_dir() and p.name.startswith("intensity_"):
            channels.append(p.name.removeprefix("intensity_"))
    return sorted(set(channels))


def _scan_ident(path: Path) -> pl.DataFrame:
    return pl.scan_parquet(
        path,
        include_file_paths="path",
        missing_columns="insert",
    ).collect()


def _scan_polygons(path: Path) -> pl.DataFrame:
    return pl.scan_parquet(
        path,
        include_file_paths="path",
        missing_columns="insert",
    ).collect()


def _pl_weighted_mean(value_col: str, weight_col: str) -> pl.Expr:
    values = pl.col(value_col)
    weights = pl.when(values.is_not_null()).then(pl.col(weight_col)).otherwise(None)
    return (values * weights).sum().truediv(weights.sum()).fill_nan(None)


def _gene_name_from_target(target: str) -> str:
    base, sep, _ = target.rpartition("-")
    return base if sep else target


def _resolve_gene_name(target: str, duplicate_genes: set[str]) -> str:
    candidate = _gene_name_from_target(target)
    if candidate in duplicate_genes:
        return target
    return candidate


def _percent_top_tuple(n_genes: int) -> tuple[int, ...]:
    bases = [max(1, n_genes // d) for d in (10, 5, 2)] + [max(1, n_genes)]
    # Preserve input order but drop duplicates while keeping ints >= 1.
    seen: set[int] = set()
    ordered: list[int] = []
    for value in bases:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return tuple(ordered)


def _resolve_rois(ws: Workspace, roi: str | None) -> list[str]:
    rois = [roi] if roi else ws.rois
    if not rois:
        raise ValueError("No ROIs found in workspace; provide ROI explicitly or populate the workspace.")
    return rois


def _filter_rois_with_segmentation(
    ws: Workspace,
    rois: list[str],
    seg_codebook: str,
    segmentation_name: str,
) -> list[str]:
    """Filter ROIs to only those that have the segmentation output directory."""
    valid_rois: list[str] = []
    for roi in rois:
        seg_path = ws.stitch(roi, seg_codebook) / segmentation_name
        if seg_path.exists():
            valid_rois.append(roi)
        else:
            logger.warning(f"Skipping ROI={roi}: segmentation not found at {seg_path}")
    return valid_rois


def _resolve_codebooks(codebooks: Iterable[str]) -> list[str]:
    cb_list = list(codebooks)
    if not cb_list:
        raise ValueError("At least one --codebook must be provided.")
    return cb_list


def _resolve_channels(
    ws: Workspace,
    rois: Iterable[str],
    seg_codebook: str,
    segmentation_name: str,
    channels: str,
) -> list[str]:
    candidate = channels.strip()
    if candidate.lower() == "auto":
        discovered: set[str] = set()
        for roi in rois:
            seg_zarr = ws.stitch(roi, seg_codebook) / segmentation_name
            discovered.update(_discover_channels(seg_zarr))
        if not discovered:
            logger.warning("No intensity_* directories found; exporting counts/cells without intensity columns.")
            return []
        return sorted(discovered)

    channel_list = [entry.strip() for entry in channels.split(",") if entry.strip()]
    if not channel_list:
        raise ValueError("Provided --channels string did not contain any valid entries.")
    return channel_list


def _load_ident_shards(
    ws: Workspace,
    rois: Iterable[str],
    codebooks: Iterable[str],
    seg_codebook: str,
    segmentation_name: str,
) -> dict[tuple[str, str], pl.DataFrame]:
    dfs: dict[tuple[str, str], pl.DataFrame] = {}
    available_chunks_by_roi: dict[str, list[str]] = {}
    for roi, codebook in product(rois, codebooks):
        seg_root = ws.stitch(roi, seg_codebook) / segmentation_name
        if roi not in available_chunks_by_roi:
            available_chunks_by_roi[roi], _ = _list_segmentation_artifacts(seg_root)
        root = seg_root / f"chunks+{codebook}"
        glob_path = root / "ident_*.parquet"
        if not root.exists():
            logger.warning(
                f"ROI={roi} codebook={codebook}: missing chunks dir {root} "
                f"(available: {available_chunks_by_roi.get(roi, [])})"
            )
            continue
        if not any(root.glob("ident_*.parquet")):
            logger.warning(f"ROI={roi} codebook={codebook}: no ident shards under {root}")
            continue
        df_roi = (
            _scan_ident(glob_path)
            .with_columns(
                z=pl.col("path").str.extract(r"(\d+)\.parquet").cast(pl.UInt16),
                codebook=pl.lit(codebook),
                roi=pl.lit(roi),
                spot_id=pl.col("spot_id").cast(pl.UInt32),
            )
            .with_columns(
                roilabel=pl.format("{}|{}", pl.col("roi"), pl.col("label")),
            )
            .drop("path")
            .sort("z")
        )
        if not df_roi.is_empty():
            dfs[(roi, codebook)] = df_roi
    if not dfs:
        roi_list = list(rois)
        requested_codebooks = list(codebooks)
        chunks_summary = "; ".join(
            [f"{roi}: {available_chunks_by_roi.get(roi, [])}" for roi in sorted(set(roi_list))]
        )
        raise ValueError(
            "\n".join(
                [
                    "Missing ident shards needed for counts export.",
                    f"- ROIs: {roi_list}",
                    f"- seg_codebook: {seg_codebook}",
                    f"- segmentation_name: {segmentation_name}",
                    f"- codebooks: {requested_codebooks}",
                    "- expected: <workspace>/analysis/deconv/stitch--<roi>+<seg_codebook>/<segmentation_name>/"
                    "chunks+<codebook>/ident_*.parquet",
                    f"- available chunks per ROI: {chunks_summary or '(none)'}",
                    "",
                    _explain_export_artifacts().rstrip(),
                    "",
                    "Fix:",
                    "- Run `segment overlay spots ...` (or `segment overlay all ...`) to create chunks.",
                    "- Pass `--codebooks` matching an existing `chunks+...` directory.",
                    "- Ensure `--segmentation-name` points at the segmentation Zarr that contains the chunks.",
                ]
            )
        )
    return dfs


def _load_intensity_shards(
    ws: Workspace,
    rois: Iterable[str],
    seg_codebook: str,
    segmentation_name: str,
    channel_list: Iterable[str],
) -> dict[_IntensityKey, pl.DataFrame]:
    intensities: dict[_IntensityKey, pl.DataFrame] = {}
    for channel, roi in product(channel_list, rois):
        # Intensity outputs are inside the segmentation zarr folder
        channel_dir = ws.stitch(roi, seg_codebook) / segmentation_name / f"intensity_{channel}"
        if not channel_dir.exists():
            logger.warning(f"ROI={roi} channel={channel}: intensity directory missing under {channel_dir}")
            continue
        glob_path = channel_dir / "intensity-*.parquet"
        if not any(channel_dir.glob("intensity-*.parquet")):
            logger.warning(f"ROI={roi} channel={channel}: no intensity shards under {channel_dir}")
            continue
        intensity_df = (
            pl.scan_parquet(glob_path, include_file_paths="path", missing_columns="insert")
            .with_columns(
                z=pl.col("path").str.extract(r"(\d+)\.parquet").cast(pl.UInt16),
                roi=pl.lit(roi),
                label=pl.col("label").cast(pl.UInt32),
            )
            .select(["z", "roi", "label", "mean_intensity", "max_intensity", "min_intensity"])
            .collect()
            .rename({
                "mean_intensity": f"{channel}_mean",
                "max_intensity": f"{channel}_max",
                "min_intensity": f"{channel}_min",
            })
            .with_columns(
                pl.col(f"{channel}_mean").cast(pl.Float32),
                pl.col(f"{channel}_max").cast(pl.Float32),
                pl.col(f"{channel}_min").cast(pl.Float32),
            )
        )
        if not intensity_df.is_empty():
            intensities[_IntensityKey(roi=roi, channel=channel)] = intensity_df
    return intensities


def _load_polygon_shards(
    ws: Workspace,
    rois: Iterable[str],
    seg_codebook: str,
    segmentation_name: str,
    primary_codebook: str,
) -> dict[str, pl.DataFrame]:
    polygons_by_roi: dict[str, pl.DataFrame] = {}
    available_chunks_by_roi: dict[str, list[str]] = {}
    for roi in rois:
        seg_root = ws.stitch(roi, seg_codebook) / segmentation_name
        available_chunks_by_roi[roi], _ = _list_segmentation_artifacts(seg_root)
        chunks_dir = seg_root / f"chunks+{primary_codebook}"
        glob_path = chunks_dir / "polygons_*.parquet"
        if not chunks_dir.exists():
            logger.warning(
                f"ROI={roi}: missing chunks dir {chunks_dir} (available: {available_chunks_by_roi.get(roi, [])})"
            )
            continue
        if not any(chunks_dir.glob("polygons_*.parquet")):
            logger.warning(f"ROI={roi}: no polygons shards under {chunks_dir}")
            continue
        pdf = (
            _scan_polygons(glob_path)
            .with_columns(
                z=pl.col("path").str.extract(r"(\d+)\.parquet").cast(pl.UInt16),
                roi=pl.lit(roi),
            )
            .with_columns(
                roilabel=pl.format("{}|{}", pl.col("roi"), pl.col("label")),
            )
            .drop("path")
            .sort("z")
        )
        if not pdf.is_empty():
            polygons_by_roi[roi] = pdf
    if not polygons_by_roi:
        roi_list = list(rois)
        chunks_summary = "; ".join(
            [f"{roi}: {available_chunks_by_roi.get(roi, [])}" for roi in sorted(set(roi_list))]
        )
        raise ValueError(
            "\n".join(
                [
                    "Missing polygons shards needed for cells export.",
                    f"- ROIs: {roi_list}",
                    f"- seg_codebook: {seg_codebook}",
                    f"- segmentation_name: {segmentation_name}",
                    f"- primary codebook: {primary_codebook}",
                    "- expected: <workspace>/analysis/deconv/stitch--<roi>+<seg_codebook>/<segmentation_name>/"
                    f"chunks+{primary_codebook}/polygons_*.parquet",
                    f"- available chunks per ROI: {chunks_summary or '(none)'}",
                    "",
                    _explain_export_artifacts().rstrip(),
                    "",
                    "Fix:",
                    "- Run `segment overlay spots ...` (or `segment overlay all ...`) to create chunks.",
                    "- Pass `--codebooks` matching an existing `chunks+...` directory.",
                    "- Ensure `--segmentation-name` points at the segmentation Zarr that contains the chunks.",
                ]
            )
        )
    return polygons_by_roi


def _emit_pairing_diagnostics(
    polygons_by_roi: dict[str, pl.DataFrame],
    intensities: dict[_IntensityKey, pl.DataFrame],
    channel_list: Iterable[str],
) -> None:
    def _pair_set(df: pl.DataFrame) -> set[tuple[int, int]]:
        if df.is_empty():
            return set()
        return {
            (int(z_value), int(label_value))
            for z_value, label_value in df.select(["z", "label"]).unique().iter_rows()
        }

    for roi, poly_df in polygons_by_roi.items():
        poly_pairs = _pair_set(poly_df)
        logger.info(f"[diag] ROI={roi}: polygons unique (z,label)={len(poly_pairs)}")
        for channel in channel_list:
            key = _IntensityKey(roi=roi, channel=channel)
            if key not in intensities:
                logger.info(f"[diag] ROI={roi}, ch={channel}: intensity shards missing")
                continue
            intensity_pairs = _pair_set(intensities[key])
            missing = poly_pairs - intensity_pairs
            extra = intensity_pairs - poly_pairs
            logger.info(
                f"[diag] ROI={roi}, ch={channel}: missing_in_intensity={len(missing)}, extra_in_intensity={len(extra)}"
            )
            if missing or extra:
                raise ValueError(
                    "Segmentation masks disagree: ROI="
                    f"{roi}, channel={channel}, missing_in_intensity={len(missing)}, extra_in_intensity={len(extra)}"
                )


def _build_cells_dataframe(
    polygons_by_roi: dict[str, pl.DataFrame],
    intensities: dict[_IntensityKey, pl.DataFrame],
    channel_list: Iterable[str],
) -> pl.DataFrame:
    polygons: list[pl.DataFrame] = []
    for roi, poly in polygons_by_roi.items():
        joined = poly.with_columns(label=pl.col("label").cast(pl.UInt32))
        for channel in channel_list:
            key = _IntensityKey(roi=roi, channel=channel)
            if key in intensities:
                joined = joined.join(intensities[key], on=["z", "roi", "label"], how="left")
            else:
                joined = joined.with_columns(
                    pl.lit(None, dtype=pl.Float32).alias(f"{channel}_mean"),
                    pl.lit(None, dtype=pl.Float32).alias(f"{channel}_max"),
                    pl.lit(None, dtype=pl.Float32).alias(f"{channel}_min"),
                )
        polygons.append(joined)
    if not polygons:
        raise ValueError("Failed to assemble polygons with intensities; no data available.")

    polygons_df = pl.concat(polygons)
    agg: dict[str, pl.Expr] = dict(
        area=pl.col("area").sum().cast(pl.Float32),
        x=((pl.col("centroid_x") * pl.col("area")).sum() / pl.col("area").sum()).cast(pl.Float32),
        y=((pl.col("centroid_y") * pl.col("area")).sum() / pl.col("area").sum()).cast(pl.Float32),
        z=((pl.col("z").cast(pl.Float64) * pl.col("area")).sum() / pl.col("area").sum()).cast(pl.Float32),
        roi=pl.col("roi").first(),
    )
    for channel in channel_list:
        agg.update({
            f"{channel}_mean": _pl_weighted_mean(f"{channel}_mean", "area").cast(pl.Float32),
            f"{channel}_max": pl.col(f"{channel}_max").max().cast(pl.Float32),
            f"{channel}_min": pl.col(f"{channel}_min").min().cast(pl.Float32),
        })

    return polygons_df.group_by(pl.col("roilabel")).agg(**agg).sort("roilabel")


def _write_cells_parquet(cells: pl.DataFrame, target_path: Path) -> Path:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    cells.write_parquet(target_path)
    logger.info(f"Wrote cells parquet to {target_path}")
    return target_path


def _build_counts_matrix(dfs: dict[tuple[str, str], pl.DataFrame]) -> pl.DataFrame:
    ident_concat = pl.concat(list(dfs.values()))
    if ident_concat.is_empty():
        raise ValueError("Spot ident shards contained no rows to aggregate.")

    transcript_counts = (
        ident_concat.filter(~pl.col("target").str.starts_with("Blank"))
        .group_by(["roilabel", "target"])
        .agg(pl.len().alias("count"))
    )
    if transcript_counts.is_empty():
        raise ValueError("No informative targets found after filtering Blank controls.")

    duplicate_genes = (
        ident_concat.filter(~pl.col("target").str.starts_with("Blank"))
        .select("target")
        .unique()
        .with_columns(gene=pl.col("target").map_elements(_gene_name_from_target, return_dtype=pl.Utf8))
        .group_by("gene")
        .agg(pl.len().alias("occurrences"))
        .filter(pl.col("occurrences") > 1)
        .get_column("gene")
        .to_list()
    )
    duplicate_set = set(duplicate_genes)
    if duplicate_set:
        logger.debug(
            f"[export] duplicate gene bases require full transcript labels ({len(duplicate_set)}): "
            f"{sorted(duplicate_set)}"
        )

    counts_by_gene = (
        transcript_counts.with_columns(
            gene=pl.col("target").map_elements(
                lambda value: _resolve_gene_name(value, duplicate_set),
                return_dtype=pl.Utf8,
            )
        )
        .group_by(["roilabel", "gene"])
        .agg(pl.col("count").sum().alias("transcripts"))
        .pivot(index="roilabel", on="gene", values="transcripts")
        .fill_null(0)
        .sort("roilabel")
    )
    if counts_by_gene.is_empty():
        raise ValueError("Failed to construct gene expression matrix for export.")
    gene_cols = [col for col in counts_by_gene.columns if col != "roilabel"]
    logger.debug(f"[export] counts matrix gene columns ({len(gene_cols)}): {gene_cols}")
    return counts_by_gene


@dataclass(slots=True)
class _RoiPolygon:
    name: str
    polygon_xy: Any


@dataclass(slots=True)
class _RoiLine:
    name: str
    p0_xy: tuple[float, float]
    p1_xy: tuple[float, float]


def load_roi_polygons(roi_path: Path, *, scale: float = 8.0) -> list[_RoiPolygon]:
    """Load ImageJ ROI polygons from a `.roi` or `.zip` bundle."""

    import zipfile

    import numpy as np
    import roifile
    from matplotlib.path import Path as MplPath

    if scale <= 0:
        raise ValueError("scale must be positive")

    rois: list[tuple[str, roifile.ImagejRoi]] = []
    if roi_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(roi_path) as zf:
            for name in sorted(zf.namelist()):
                if not name.lower().endswith(".roi"):
                    continue
                roi = roifile.ImagejRoi.frombytes(zf.read(name))
                roi_name = roi.name or Path(name).stem
                rois.append((roi_name, roi))
    else:
        roi = roifile.ImagejRoi.fromfile(roi_path)
        roi_name = roi.name or roi_path.stem
        rois.append((roi_name, roi))

    polygons: list[_RoiPolygon] = []
    for roi_name, roi in rois:
        coords_rc = roi.coordinates()
        if coords_rc.size == 0:
            continue
        coords_xy = np.asarray(coords_rc, dtype=np.float32)[:, ::-1] * float(scale)
        polygons.append(_RoiPolygon(name=str(roi_name), polygon_xy=MplPath(coords_xy)))

    return polygons


def _load_roiset_line_and_polygons(
    roi_path: Path,
    *,
    scale: float,
) -> tuple[_RoiLine | None, list[_RoiPolygon]]:
    import zipfile

    import numpy as np
    import roifile
    from matplotlib.path import Path as MplPath

    if scale <= 0:
        raise ValueError("scale must be positive")

    rois: list[tuple[str, roifile.ImagejRoi]] = []
    if roi_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(roi_path) as zf:
            for name in sorted(zf.namelist()):
                if not name.lower().endswith(".roi"):
                    continue
                roi = roifile.ImagejRoi.frombytes(zf.read(name))
                roi_name = roi.name or Path(name).stem
                rois.append((roi_name, roi))
    else:
        roi = roifile.ImagejRoi.fromfile(roi_path)
        roi_name = roi.name or roi_path.stem
        rois.append((roi_name, roi))

    line_entries: list[_RoiLine] = []
    polygons: list[_RoiPolygon] = []
    for roi_name, roi in rois:
        if roi.roitype == roifile.ROI_TYPE.LINE:
            p0 = (float(roi.x1) * scale, float(roi.y1) * scale)
            p1 = (float(roi.x2) * scale, float(roi.y2) * scale)
            line_entries.append(_RoiLine(name=str(roi_name), p0_xy=p0, p1_xy=p1))
            continue

        coords_rc = roi.coordinates()
        if coords_rc.size == 0:
            continue
        coords_xy = np.asarray(coords_rc, dtype=np.float32)[:, ::-1] * float(scale)
        polygons.append(_RoiPolygon(name=str(roi_name), polygon_xy=MplPath(coords_xy)))

    if len(line_entries) > 1:
        names = ", ".join([entry.name for entry in line_entries])
        raise ValueError(f"Expected exactly one line ROI in {roi_path}, found {len(line_entries)}: {names}")

    line = line_entries[0] if line_entries else None
    return line, polygons


def _thumbnail_size(thumbnail_dir: Path) -> tuple[int, int] | None:
    from PIL import Image

    candidates = sorted(thumbnail_dir.glob("thumbnail_z*.png"))
    if not candidates:
        return None
    with Image.open(candidates[0]) as img:
        width, height = img.size
    return width, height


def _segmentation_xy_shape(seg_zarr_path: Path) -> tuple[int, int]:
    import zarr

    arr = zarr.open_array(seg_zarr_path, mode="r")
    if arr.ndim < 2:
        raise ValueError(f"Segmentation zarr {seg_zarr_path} has unexpected shape {arr.shape}")
    height = int(arr.shape[-2])
    width = int(arr.shape[-1])
    return width, height


def _validate_thumbnail_scale(
    thumbnail_dir: Path,
    seg_zarr_path: Path,
    *,
    scale: float,
) -> None:
    if scale <= 0:
        raise ValueError("thumbnail scale must be positive")

    size = _thumbnail_size(thumbnail_dir)
    if size is None:
        raise ValueError(f"No thumbnail_z*.png files found under {thumbnail_dir}")
    thumb_w, thumb_h = size
    seg_w, seg_h = _segmentation_xy_shape(seg_zarr_path)
    if seg_w <= 0 or seg_h <= 0:
        raise ValueError(f"Segmentation zarr {seg_zarr_path} has invalid shape ({seg_h}, {seg_w})")

    scale_int = int(round(scale))
    if not np.isclose(scale, scale_int):
        raise ValueError(f"thumbnail scale must be an integer downsampling factor, got {scale}")

    def _bounds(thumb_dim: int) -> tuple[int, int]:
        if thumb_dim <= 0:
            return (0, 0)
        lo = (thumb_dim - 1) * scale_int + 1
        hi = thumb_dim * scale_int
        return lo, hi

    w_lo, w_hi = _bounds(thumb_w)
    h_lo, h_hi = _bounds(thumb_h)

    if not (w_lo <= seg_w <= w_hi and h_lo <= seg_h <= h_hi):
        raise ValueError(
            "Thumbnail scale mismatch: "
            f"thumb=({thumb_w},{thumb_h}) scale={scale_int} -> "
            f"expected_w=[{w_lo},{w_hi}], expected_h=[{h_lo},{h_hi}], seg=({seg_w},{seg_h})"
        )


def _assign_subroi_labels(
    coords_xy: np.ndarray,
    polygons: list[_RoiPolygon],
) -> list[str]:
    labels = np.full(coords_xy.shape[0], "", dtype=object)
    for poly in polygons:
        mask = poly.polygon_xy.contains_points(coords_xy)
        labels[(labels == "") & mask] = poly.name
    return labels.tolist()


def _apply_roiset_annotations(
    adata: "anndata.AnnData",
    *,
    ws: Workspace,
    roi: str,
    seg_codebook: str,
    segmentation_name: str,
    scale: float,
) -> None:
    import numpy as np

    from fishtools.utils.spatial_transform import rotate_points

    thumbnail_dir = ws.output / "thumbnails" / f"{roi}+{seg_codebook}"
    roi_path = thumbnail_dir / "RoiSet.zip"
    if not roi_path.exists():
        logger.warning(f"Skipping ROI {roi}: RoiSet.zip not found at {roi_path}")
        return

    line, polygons = _load_roiset_line_and_polygons(roi_path, scale=scale)
    seg_zarr_path = ws.stitch(roi, seg_codebook) / segmentation_name
    _validate_thumbnail_scale(thumbnail_dir, seg_zarr_path, scale=scale)

    roi_mask = adata.obs["roi"] == roi
    if not roi_mask.any():
        logger.warning(f"Skipping ROI {roi}: no cells found in export.")
        return

    coords = adata.obs.loc[roi_mask, ["x", "y"]].to_numpy(dtype=np.float64)
    if polygons:
        labels = _assign_subroi_labels(coords, polygons)
        adata.obs.loc[roi_mask, "subroi"] = labels
    else:
        adata.obs.loc[roi_mask, "subroi"] = ""

    if line is None:
        logger.warning(f"Skipping ROI {roi}: no line ROI found in {roi_path}")
        return

    dx = line.p1_xy[0] - line.p0_xy[0]
    dy = line.p1_xy[1] - line.p0_xy[1]
    angle_deg = float(np.rad2deg(np.arctan2(dy, dx)))
    center = ((line.p0_xy[0] + line.p1_xy[0]) / 2.0, (line.p0_xy[1] + line.p1_xy[1]) / 2.0)
    rotated = rotate_points(coords, -angle_deg, center=center).astype(np.float32, copy=False)

    adata.obsm["spatial"][roi_mask.to_numpy()] = rotated
    adata.obs.loc[roi_mask, "x"] = rotated[:, 0]
    adata.obs.loc[roi_mask, "y"] = rotated[:, 1]


def annotate_cells_with_roi(
    adata: "anndata.AnnData",
    roi_path: Path,
    *,
    scale: float = 8.0,
    column_name: str = "roi",
) -> None:
    """Annotate `adata.obs[column_name]` with ROI names for each cell centroid."""

    polygons = load_roi_polygons(roi_path, scale=scale)
    if "x" not in adata.obs.columns or "y" not in adata.obs.columns:
        raise ValueError("Expected adata.obs to contain 'x' and 'y' centroid columns.")

    labels: list[str] = []
    xs = adata.obs["x"].to_numpy()
    ys = adata.obs["y"].to_numpy()
    for x, y in zip(xs, ys):
        assigned = ""
        for poly in polygons:
            if poly.polygon_xy.contains_point((float(x), float(y))):
                assigned = poly.name
                break
        labels.append(assigned)

    adata.obs[column_name] = labels


def _build_anndata(counts_by_gene: pl.DataFrame, cells: pl.DataFrame) -> "anndata.AnnData":
    counts_pd = counts_by_gene.to_pandas().set_index("roilabel")
    cells_pd = cells.to_pandas().set_index("roilabel")
    missing_obs = counts_pd.index.difference(cells_pd.index)
    if not missing_obs.empty:
        raise ValueError("Missing centroid entries for roilabels: " + ", ".join(missing_obs.astype(str)))

    counts_pd = counts_pd.reindex(cells_pd.index, fill_value=0)
    obs_pd = cells_pd.loc[counts_pd.index]

    import anndata as ad
    import numpy as np
    import pandas as pd
    import scanpy as sc

    qc_filtering: dict[str, Any] = {
        "filter_cells": {"max_counts": 1200},
        "filter_genes": {"min_cells": 10},
        "pre_filter": {"n_obs": int(obs_pd.shape[0]), "n_vars": int(counts_pd.shape[1])},
    }

    adata = ad.AnnData(
        X=counts_pd.to_numpy(dtype=np.float32, copy=True),
        obs=obs_pd.copy(),
        var=pd.DataFrame(index=pd.Index(counts_pd.columns.astype(str), name="gene")),
    )
    adata.obs_names = pd.Index(counts_pd.index.astype(str), name="roilabel")
    if {"x", "y"} - set(adata.obs.columns):
        raise ValueError("Cells dataframe missing centroid columns 'x' and 'y'.")
    adata.obsm["spatial"] = adata.obs[["x", "y"]].to_numpy(dtype=np.float32)

    n_genes = adata.n_vars
    percent_top = _percent_top_tuple(n_genes)
    sc.pp.calculate_qc_metrics(adata, inplace=True, percent_top=percent_top)
    # sc.pp.filter_cells(adata, min_counts=30)
    sc.pp.filter_cells(adata, max_counts=1200)
    sc.pp.filter_genes(adata, min_cells=10)
    if adata.n_obs == 0 or adata.n_vars == 0:
        raise ValueError("Filtering removed all cells or genes; adjust thresholds or inputs.")
    adata.obsm["spatial"] = adata.obs[["x", "y"]].to_numpy(dtype=np.float32)
    qc_filtering["calculate_qc_metrics"] = {"percent_top": list(percent_top)}
    qc_filtering["post_filter"] = {"n_obs": int(adata.n_obs), "n_vars": int(adata.n_vars)}
    adata.uns.setdefault("fishtools", {})["qc_filtering"] = qc_filtering
    return adata


def export_cmd(
    path: Path,
    roi: str | None,
    seg_codebook: str,
    codebooks: Iterable[str],
    segmentation_name: str,
    channels: str,
    thumbnail_scale: float = 8.0,
    diag: bool = False,
) -> None:
    """Export per-cell intensities and h5ad ready for Scanpy workflows.

    - Writes aggregated per-cell centroids and intensities to the workspace output directory.
    - Emits an AnnData file under <workspace>/analysis/output capturing gene counts, QC metrics, and spatial coordinates.
    """

    ws = Workspace(path)
    rois = _resolve_rois(ws, roi)
    rois = _filter_rois_with_segmentation(ws, rois, seg_codebook, segmentation_name)
    if not rois:
        raise ValueError("No ROIs have segmentation output; run segmentation first.")
    cb_list = _resolve_codebooks(codebooks)

    ident_frames = _load_ident_shards(ws, rois, cb_list, seg_codebook, segmentation_name)
    channel_list = _resolve_channels(ws, rois, seg_codebook, segmentation_name, channels)
    intensities = _load_intensity_shards(ws, rois, seg_codebook, segmentation_name, channel_list)
    if channel_list and not intensities:
        logger.warning(
            f"No intensity shards found for ROIs={rois}, channels={channel_list}; exporting without intensity values."
        )

    primary_cb = cb_list[0]
    polygons_by_roi = _load_polygon_shards(ws, rois, seg_codebook, segmentation_name, primary_cb)

    if diag:
        _emit_pairing_diagnostics(polygons_by_roi, intensities, channel_list)

    cells = _build_cells_dataframe(polygons_by_roi, intensities, channel_list)
    counts_by_gene = _build_counts_matrix(ident_frames)
    adata = _build_anndata(counts_by_gene, cells)

    adata.obs["subroi"] = ""
    for roi_name in rois:
        _apply_roiset_annotations(
            adata,
            ws=ws,
            roi=roi_name,
            seg_codebook=seg_codebook,
            segmentation_name=segmentation_name,
            scale=thumbnail_scale,
        )

    cb_token = Workspace.sanitize_codebook_name(primary_cb)
    h5ad_token = cb_token
    if len(cb_list) > 1:
        extra = "__".join(Workspace.sanitize_codebook_name(value) for value in cb_list[1:])
        h5ad_token = f"{cb_token}__{extra}"
    seg_stem = Path(segmentation_name).stem

    ws.output.mkdir(parents=True, exist_ok=True)
    cells_path = ws.output / f"polygons+{cb_token}+{seg_stem}.parquet"
    out_h5ad = ws.output / f"all+{h5ad_token}+{seg_stem}.h5ad"

    from importlib.metadata import PackageNotFoundError, version as pkg_version
    import platform
    import sys

    def _safe_pkg_version(name: str) -> str | None:
        try:
            return pkg_version(name)
        except PackageNotFoundError:
            return None

    artifacts_by_roi: dict[str, dict[str, Any]] = {}
    for roi_name in rois:
        seg_zarr_path = ws.stitch(roi_name, seg_codebook) / segmentation_name
        chunks, intensity_channels = _list_segmentation_artifacts(seg_zarr_path)
        roiset_zip_path = ws.output / "thumbnails" / f"{roi_name}+{seg_codebook}" / "RoiSet.zip"
        artifacts_by_roi[roi_name] = {
            "segmentation_path": str(seg_zarr_path),
            "available_chunks": chunks,
            "available_intensity_channels": intensity_channels,
            "roiset_zip_path": str(roiset_zip_path),
            "roiset_zip_exists": roiset_zip_path.exists(),
        }

    adata.uns.setdefault("fishtools", {})["segment_export"] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "workspace_path": str(ws.path),
        "export_invocation_path": str(path),
        "args": {
            "roi": roi,
            "seg_codebook": seg_codebook,
            "codebooks": list(cb_list),
            "segmentation_name": segmentation_name,
            "channels": channels,
            "thumbnail_scale": float(thumbnail_scale),
            "diag": bool(diag),
        },
        "resolved": {
            "rois": list(rois),
            "channels": list(channel_list),
            "primary_codebook": primary_cb,
        },
        "outputs": {"cells_parquet_path": str(cells_path), "h5ad_path": str(out_h5ad)},
        "artifacts_by_roi": artifacts_by_roi,
        "versions": {
            "python": platform.python_version(),
            "fishtools": _safe_pkg_version("fishtools"),
            "anndata": _safe_pkg_version("anndata"),
            "scanpy": _safe_pkg_version("scanpy"),
            "polars": _safe_pkg_version("polars"),
            "numpy": _safe_pkg_version("numpy"),
            "platform": platform.platform(),
            "executable": sys.executable,
        },
    }

    _write_cells_parquet(cells, cells_path)
    adata.write_h5ad(out_h5ad)
    logger.info(f"Wrote AnnData export to {out_h5ad}")
