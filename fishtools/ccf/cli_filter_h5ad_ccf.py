from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Literal, cast

import anndata as ad
import matplotlib as mpl
import numpy as np
import pandas as pd
import rich_click as click
from brainglobe_atlasapi import BrainGlobeAtlas
from scipy.spatial import cKDTree

# Force a non-interactive backend to avoid GUI/event-loop hangs in headless runs
mpl.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402

from fishtools.ccf.ontology import CCFTermKind, mask_ccf_subtree
from fishtools.ccf.landmark import LandmarkRegistrationOutputs
from fishtools.ccf.ndimage_geometry import fused_xy_to_rotated_crop_xy
from fishtools.io.workspace import Workspace
from fishtools.postprocess.roi_polygons import load_roi_polygons
from fishtools.utils.logging import setup_cli_logging


click.rich_click.SHOW_ARGUMENTS = True
click.rich_click.GROUP_ARGUMENTS_OPTIONS = True
click.rich_click.USE_MARKDOWN = True
click.rich_click.STYLE_HELPTEXT = ""


CombineMode = Literal["any", "all"]
MatchMode = Literal["subtree", "exact"]
CoordUnits = Literal["px", "um", "mm"]
SpatialOrder = Literal["xy", "yx"]
CoordSpace = Literal["crop", "full"]
InputSpace = Literal["crop", "full", "fused"]


def _write_qc_mask_overlay_plot(
    *,
    coords_xy: np.ndarray,
    keep_mask: np.ndarray,
    output_png: Path,
    title: str,
    units: str,
    max_points: int = 200_000,
) -> None:
    coords_xy = np.asarray(coords_xy, dtype=np.float64)
    if coords_xy.ndim != 2 or coords_xy.shape[1] != 2:
        raise ValueError(f"Expected coords_xy with shape (N,2), got {coords_xy.shape}.")

    keep_mask = np.asarray(keep_mask, dtype=bool)
    if keep_mask.ndim != 1 or keep_mask.shape[0] != coords_xy.shape[0]:
        raise ValueError(f"keep_mask shape mismatch: {keep_mask.shape} vs coords {coords_xy.shape}.")

    n = int(coords_xy.shape[0])
    stride = max(1, int(math.ceil(n / int(max_points)))) if max_points > 0 else 1
    idx = np.arange(0, n, stride, dtype=np.int64)

    coords_p = coords_xy[idx]
    keep_p = keep_mask[idx]
    kept = coords_p[keep_p]
    dropped = coords_p[~keep_p]

    output_png.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(7.5, 7.5), dpi=180)
    ax.set_aspect("equal")

    s_drop = float(np.clip(2000.0 / max(1, dropped.shape[0]), 0.02, 1.2))
    s_keep = float(np.clip(2000.0 / max(1, kept.shape[0]), 0.05, 3.0))

    if dropped.shape[0] > 0:
        ax.scatter(
            dropped[:, 0],
            dropped[:, 1],
            s=s_drop,
            alpha=0.08,
            linewidths=0,
            color="#666666",
            rasterized=True,
            label=f"dropped ({int((~keep_mask).sum())})",
        )
    if kept.shape[0] > 0:
        ax.scatter(
            kept[:, 0],
            kept[:, 1],
            s=s_keep,
            alpha=0.35,
            linewidths=0,
            color="tab:blue",
            rasterized=True,
            label=f"kept ({int(keep_mask.sum())})",
        )

    ax.set_title(title)
    ax.set_xlabel(f"X ({units})")
    ax.set_ylabel(f"Y ({units})")
    ax.invert_yaxis()
    ax.legend(loc="best", markerscale=5)

    fig.tight_layout()
    fig.savefig(output_png.as_posix(), dpi=200, bbox_inches="tight")
    plt.close(fig)


def _mask_ccf_exact(adata: ad.AnnData, term: int | str, *, kind: CCFTermKind, obsm_key: str) -> np.ndarray:
    if obsm_key not in adata.obsm:
        raise KeyError(f"Missing adata.obsm[{obsm_key!r}].")
    table = adata.obsm[obsm_key]
    if not isinstance(table, pd.DataFrame):
        raise TypeError(f"Expected adata.obsm[{obsm_key!r}] to be a DataFrame, got {type(table).__name__}.")

    kind_norm = cast(CCFTermKind, str(kind).lower())
    if kind_norm not in {"auto", "id", "acronym", "name"}:
        raise ValueError(f"Invalid kind={kind!r}. Expected 'auto'|'id'|'acronym'|'name'.")

    if kind_norm == "auto":
        if isinstance(term, int):
            kind_norm = "id"
        else:
            s = str(term).strip()
            if s.isdigit():
                kind_norm = "id"
            else:
                mask_acr = _mask_ccf_exact(adata, s, kind="acronym", obsm_key=obsm_key)
                mask_name = _mask_ccf_exact(adata, s, kind="name", obsm_key=obsm_key)
                return np.asarray(mask_acr, dtype=bool) | np.asarray(mask_name, dtype=bool)

    if kind_norm == "id":
        if "id" not in table.columns:
            raise KeyError(f"Missing 'id' column in adata.obsm[{obsm_key!r}].")
        q = int(term)
        return table["id"].to_numpy(dtype=np.int64, copy=False) == q

    if kind_norm == "acronym":
        if "acronym" not in table.columns:
            raise KeyError(f"Missing 'acronym' column in adata.obsm[{obsm_key!r}].")
        q = str(term).strip().lower()
        col = table["acronym"].astype("string")
        return col.str.lower().to_numpy() == q

    if "name" not in table.columns:
        raise KeyError(f"Missing 'name' column in adata.obsm[{obsm_key!r}].")
    q = str(term).strip().lower()
    col = table["name"].astype("string")
    return col.str.lower().to_numpy() == q


def _mask_for_term(adata: ad.AnnData, term: str, *, kind: CCFTermKind, match: MatchMode, obsm_key: str) -> np.ndarray:
    if match == "subtree":
        return mask_ccf_subtree(adata, term, kind=kind, obsm_key=obsm_key)
    return _mask_ccf_exact(adata, term, kind=kind, obsm_key=obsm_key)


def _coords_to_um(
    coords: np.ndarray,
    *,
    units: CoordUnits,
    spacing_um: float | None,
) -> np.ndarray:
    coords_f = np.asarray(coords, dtype=np.float64)
    if coords_f.ndim != 2 or coords_f.shape[1] != 2:
        raise ValueError(f"Expected coords with shape (N,2), got {coords_f.shape}.")

    if units == "um":
        return coords_f
    if units == "mm":
        return coords_f * 1000.0
    if units == "px":
        if spacing_um is None:
            raise ValueError("coords spacing_um is required when coords_units='px'.")
        return coords_f * float(spacing_um)
    raise ValueError(f"Unsupported coords units: {units!r}.")


def _coords_to_px(
    coords: np.ndarray,
    *,
    units: CoordUnits,
    spacing_um: float,
) -> np.ndarray:
    coords_f = np.asarray(coords, dtype=np.float64)
    if coords_f.ndim != 2 or coords_f.shape[1] != 2:
        raise ValueError(f"Expected coords with shape (N,2), got {coords_f.shape}.")

    if units == "px":
        return coords_f
    if units == "um":
        return coords_f / float(spacing_um)
    if units == "mm":
        return (coords_f * 1000.0) / float(spacing_um)
    raise ValueError(f"Unsupported coords units: {units!r}.")


def _input_h5ad_candidate(ws: Workspace, *, roi: str, h5ad_name: str | None) -> Path:
    base = ws.output.ccf_transforms / roi
    if h5ad_name is None:
        return base / f"{roi}.syn.h5ad"
    else:
        name = str(h5ad_name).strip()
        if not name:
            raise click.BadParameter("--h5ad-name must be non-empty.")
        if Path(name).name != name:
            raise click.BadParameter("--h5ad-name must be a filename only (no directories).")
        if not name.endswith(".h5ad"):
            name = f"{name}.h5ad"
        return base / name


def _resolve_input_h5ad(ws: Workspace, *, roi: str, h5ad_name: str | None) -> Path:
    candidate = _input_h5ad_candidate(ws, roi=roi, h5ad_name=h5ad_name)
    if not candidate.exists():
        raise click.ClickException(
            f"Missing warped h5ad for roi={roi!r}: {candidate}. "
            "Run `ccf-warp-h5ad-spatial <workspace> <roi> ...` first (or pass --h5ad-name)."
        )
    return candidate


def _default_outputs(
    ws: Workspace,
    *,
    roi: str,
    input_h5ad: Path,
    out_name: str | None,
) -> tuple[str, Path, Path]:
    out_dir = ws.output.ccf_transforms / roi
    if out_name is None:
        out_name = f"{input_h5ad.stem}.annotated.h5ad"
    output_h5ad = out_dir / out_name
    plot_png = output_h5ad.with_suffix(".qc.png")
    return (str(roi), output_h5ad, plot_png)


def _ensure_ccf_obsm(
    adata: ad.AnnData,
    *,
    input_h5ad: Path,
    ccf_obsm_key: str,
    coords_key: str,
    coords_units: CoordUnits,
    coords_space: CoordSpace,
    spatial_order: SpatialOrder,
    roi: str,
    workspace: Path,
) -> None:
    if ccf_obsm_key in adata.obsm:
        return
    if coords_key not in adata.obsm:
        raise click.ClickException(
            f"Missing adata.obsm[{ccf_obsm_key!r}] in {input_h5ad}, and cannot auto-annotate because "
            f"adata.obsm[{coords_key!r}] is missing."
        )

    ws = Workspace(workspace)
    out_root = ws.ccf_transforms(roi)
    out_contract = LandmarkRegistrationOutputs(out_root)
    p1 = out_contract.read_p1_landmarks()

    atlas_voxel_um = float(p1.atlas_voxel_um) if p1.atlas_voxel_um is not None else 20.0
    atlas_name = p1.atlas_name or "kim_dev_mouse_e15-5_lsfm_20um"
    atlas_plane = p1.atlas_plane or ("sagittal" if "Sag" in str(ws.path) else "coronal")
    if p1.atlas_slice_idx is None:
        raise click.ClickException(f"p1_landmarks.json is missing atlas_slice_idx under {out_root}.")
    atlas_slice_idx = int(p1.atlas_slice_idx)

    ar0, ar1, ac0, ac1 = p1.atlas_crop_bbox
    crop_offset_xy = (float(ac0), float(ar0))

    atlas = BrainGlobeAtlas(atlas_name)
    ann_vol = atlas.annotation if atlas_plane == "coronal" else atlas.annotation.transpose(2, 1, 0)
    ann_slice = np.asarray(ann_vol[atlas_slice_idx, :, :], dtype=np.uint32)
    if ann_slice.ndim != 2:
        raise click.ClickException(f"Expected 2D annotation slice, got shape={ann_slice.shape}.")

    coords = np.asarray(adata.obsm[coords_key])
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise click.ClickException(f"Expected obsm[{coords_key!r}] to have shape (N,2), got {coords.shape}.")
    coords = coords.astype(np.float64, copy=False)
    if spatial_order == "yx":
        coords = coords[:, ::-1]

    coords_px = _coords_to_px(coords, units=coords_units, spacing_um=atlas_voxel_um)
    x_px = coords_px[:, 0]
    y_px = coords_px[:, 1]

    if coords_space == "crop":
        x_full = x_px + float(crop_offset_xy[0])
        y_full = y_px + float(crop_offset_xy[1])
    else:
        x_full = x_px
        y_full = y_px

    xi = np.round(x_full).astype(np.int64, copy=False)
    yi = np.round(y_full).astype(np.int64, copy=False)

    h, w = ann_slice.shape
    in_bounds = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
    ids = np.zeros(int(adata.n_obs), dtype=np.uint32)
    ids[in_bounds] = ann_slice[yi[in_bounds], xi[in_bounds]]

    structures = atlas.structures
    tree = structures.tree

    uniq, inv = np.unique(ids.astype(np.int64), return_inverse=True)

    def _info_for_id(structure_id: int) -> tuple[str, str, int, str, str, str, bool]:
        if structure_id == 0:
            return ("background", "background", -1, "", "", "", False)
        st = structures[int(structure_id)]
        acronym = str(st["acronym"])
        name = str(st["name"])
        path_ids = [int(v) for v in st["structure_id_path"]]
        parent_id = int(path_ids[-2]) if len(path_ids) >= 2 else int(structure_id)
        path_ids_json = json.dumps(path_ids, separators=(",", ":"))
        path_acronyms = "/".join(str(structures[i]["acronym"]) for i in path_ids)
        path_names = "/".join(str(structures[i]["name"]) for i in path_ids)
        is_leaf = len(tree.children(int(structure_id))) == 0
        return (acronym, name, parent_id, path_ids_json, path_acronyms, path_names, is_leaf)

    uniq_acronym: list[str] = []
    uniq_name: list[str] = []
    uniq_parent: list[int] = []
    uniq_path_ids: list[str] = []
    uniq_path_acr: list[str] = []
    uniq_path_name: list[str] = []
    uniq_is_leaf: list[bool] = []
    for sid in uniq.tolist():
        a, n, pid, pids, pacr, pname, leaf = _info_for_id(int(sid))
        uniq_acronym.append(a)
        uniq_name.append(n)
        uniq_parent.append(pid)
        uniq_path_ids.append(pids)
        uniq_path_acr.append(pacr)
        uniq_path_name.append(pname)
        uniq_is_leaf.append(leaf)

    ccf_df = pd.DataFrame(
        {
            "id": ids.astype(np.int64, copy=False),
            "acronym": pd.Categorical(np.asarray(uniq_acronym, dtype=object)[inv]),
            "name": pd.Categorical(np.asarray(uniq_name, dtype=object)[inv]),
            "parent_id": np.asarray(uniq_parent, dtype=np.int64)[inv],
            "path_ids": pd.Categorical(np.asarray(uniq_path_ids, dtype=object)[inv]),
            "path_acronyms": pd.Categorical(np.asarray(uniq_path_acr, dtype=object)[inv]),
            "path_names": pd.Categorical(np.asarray(uniq_path_name, dtype=object)[inv]),
            "is_leaf": np.asarray(uniq_is_leaf, dtype=bool)[inv],
            "in_bounds": in_bounds,
        },
        index=adata.obs_names,
    )
    adata.obsm[ccf_obsm_key] = ccf_df
    adata.uns.setdefault(f"{ccf_obsm_key}_atlas", {})
    adata.uns[f"{ccf_obsm_key}_atlas"] = {
        "atlas_name": atlas_name,
        "atlas_plane": atlas_plane,
        "atlas_slice_idx": atlas_slice_idx,
        "atlas_voxel_um": atlas_voxel_um,
        "coords_key": coords_key,
        "coords_space": coords_space,
        "coords_units": coords_units,
        "spatial_order": spatial_order,
        "atlas_crop_bbox": [int(ar0), int(ar1), int(ac0), int(ac1)],
    }


def _dilate_mask_by_radius_um(
    coords_um_xy: np.ndarray,
    *,
    base_mask: np.ndarray,
    radius_um: float,
) -> np.ndarray:
    base_mask = np.asarray(base_mask, dtype=bool)
    if base_mask.ndim != 1 or base_mask.shape[0] != coords_um_xy.shape[0]:
        raise ValueError(f"base_mask shape mismatch: {base_mask.shape} vs coords {coords_um_xy.shape}.")

    r = float(radius_um)
    if r <= 0:
        return base_mask

    seeds = coords_um_xy[base_mask]
    if seeds.shape[0] == 0:
        return np.zeros_like(base_mask, dtype=bool)

    tree = cKDTree(seeds)
    dist, _ = tree.query(coords_um_xy, k=1, distance_upper_bound=r, workers=-1)
    return np.isfinite(dist)


def _find_imagej_roi_file(mask_edit_dir: Path) -> Path | None:
    preferred = mask_edit_dir / "RoiSet.zip"
    if preferred.exists():
        return preferred

    rois = sorted(mask_edit_dir.glob("*.roi"))
    if len(rois) == 1:
        return rois[0]
    if len(rois) > 1:
        raise click.ClickException(
            "Multiple ImageJ ROI files found under "
            f"{mask_edit_dir} ({', '.join(p.name for p in rois)}). "
            "Pass --imagej-roi-path to disambiguate."
        )
    return None


def _to_moving_thumbnail_px_xy(
    coords_xy: np.ndarray,
    *,
    ws: Workspace,
    roi: str,
    input_space: InputSpace,
    input_units: CoordUnits,
    spatial_order: SpatialOrder,
    stitch_codebook: str,
    target_spacing_um: float,
) -> np.ndarray:
    coords = np.asarray(coords_xy, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"Expected coords_xy with shape (N,2), got {coords.shape}.")

    if spatial_order == "yx":
        coords = coords[:, ::-1]

    out_root = ws.ccf_transforms(roi)
    out_contract = LandmarkRegistrationOutputs(out_root)
    p1 = out_contract.read_p1_landmarks()

    if input_units != "px":
        raise click.ClickException("--imagej-input-units currently only supports 'px'.")

    sample_voxel_um = float(p1.sample_voxel_xy_um) if p1.sample_voxel_xy_um is not None else 0.216
    crop_bbox = p1.sample_rotated_crop_bbox
    sr0, sr1, sc0, sc1 = (int(crop_bbox[0]), int(crop_bbox[1]), int(crop_bbox[2]), int(crop_bbox[3]))
    if sr1 <= sr0 or sc1 <= sc0:
        raise ValueError(f"Invalid sample_rotated_crop_bbox={crop_bbox}.")

    x_in = coords[:, 0]
    y_in = coords[:, 1]

    if input_space == "crop":
        x_crop = x_in
        y_crop = y_in
    elif input_space == "full":
        x_crop = x_in - float(sc0)
        y_crop = y_in - float(sr0)
    else:
        fused_zarr = ws.stitch(roi, stitch_codebook) / "fused.zarr"
        if not fused_zarr.exists():
            raise FileNotFoundError(f"Missing fused.zarr at {fused_zarr} (needed for --imagej-input-space=fused).")
        import zarr

        arr = zarr.open(str(fused_zarr), mode="r")
        fused_shape_yx = (int(arr.shape[1]), int(arr.shape[2]))
        x_crop, y_crop, _ = fused_xy_to_rotated_crop_xy(
            x_fused=x_in,
            y_fused=y_in,
            fused_shape_yx=fused_shape_yx,
            prior_flip_x=bool(p1.prior_flip_x),
            prior_rotation_deg=float(p1.prior_rotation_deg),
            rotated_crop_bbox=(sr0, sr1, sc0, sc1),
        )

    target_um = float(target_spacing_um)
    if not target_um > 0:
        raise click.ClickException("--imagej-target-spacing-um must be > 0.")
    scale = float(sample_voxel_um) / target_um
    return np.stack([x_crop * scale, y_crop * scale], axis=1)


def _mask_from_imagej_roi(
    *,
    ws: Workspace,
    roi: str,
    adata: ad.AnnData,
    roi_path: Path,
    coords_key: str,
    input_space: InputSpace,
    input_units: CoordUnits,
    spatial_order: SpatialOrder,
    stitch_codebook: str,
    target_spacing_um: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if coords_key not in adata.obsm:
        raise click.ClickException(f"Missing adata.obsm[{coords_key!r}] in input h5ad.")

    coords_xy = np.asarray(adata.obsm[coords_key])
    coords_thumb_xy = _to_moving_thumbnail_px_xy(
        coords_xy,
        ws=ws,
        roi=roi,
        input_space=input_space,
        input_units=input_units,
        spatial_order=spatial_order,
        stitch_codebook=stitch_codebook,
        target_spacing_um=target_spacing_um,
    )

    roi_polys = load_roi_polygons(roi_path, scale=1.0)
    geometries = [p.geometry for p in roi_polys]

    from matplotlib.path import Path as MplPath
    from shapely.geometry import MultiPolygon, Polygon

    keep = np.zeros(int(coords_thumb_xy.shape[0]), dtype=bool)
    labels = np.empty(int(coords_thumb_xy.shape[0]), dtype=object)
    labels[:] = ""
    points = np.asarray(coords_thumb_xy, dtype=np.float64, order="C")
    for entry, geom in zip(roi_polys, geometries, strict=True):
        if isinstance(geom, Polygon):
            polys = [geom]
        elif isinstance(geom, MultiPolygon):
            polys = list(geom.geoms)
        else:  # pragma: no cover - load_roi_polygons filters these out
            continue

        for poly in polys:
            ext = np.asarray(poly.exterior.coords, dtype=np.float64)
            if ext.ndim != 2 or ext.shape[1] != 2:
                continue
            # radius>0 includes boundary points (similar to shapely 'covers').
            path = MplPath(ext, closed=True)
            inside = path.contains_points(points, radius=1e-9)
            keep |= inside
            new = inside & (labels == "")
            if np.any(new):
                labels[new] = entry.name

    return keep, coords_thumb_xy, labels


@click.command("filter-h5ad-ccf")
@click.argument(
    "workspace",
    type=click.Path(
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
        path_type=Path,
    ),
)
@click.argument("rois", nargs=-1, type=str)
@click.option(
    "--h5ad-name",
    default=None,
    show_default=False,
    help=(
        "Input h5ad filename under <workspace>/analysis/output/ccf-transforms/<roi>/. "
        "Default: <roi>.syn.h5ad"
    ),
)
@click.option(
    "--out-name",
    default=None,
    show_default=False,
    help=(
        "Output filename (written under <workspace>/analysis/output/ccf-transforms/<roi>/). "
        "Default: <input_stem>.annotated.h5ad."
    ),
)
@click.option(
    "--ccf-col",
    default="ccf",
    show_default=True,
    help="obs column name to write term/atlas-derived selection labels into (empty string if not selected).",
)
@click.option(
    "--ccf-adjusted-col",
    default="ccf_adjusted",
    show_default=True,
    help="obs column name to write ImageJ ROI override labels into (empty string if not selected).",
)
@click.option(
    "--term",
    "terms",
    multiple=True,
    required=False,
    help="Ontology term(s) to select (name/acronym/id). Repeatable.",
)
@click.option(
    "--imagej-roi",
    is_flag=True,
    default=False,
    show_default=True,
    help=(
        "Use ImageJ ROI filtering by auto-detecting a .roi/RoiSet.zip under mask_edit/. "
        "If used, overrides --term."
    ),
)
@click.option(
    "--imagej-roi-path",
    default=None,
    show_default=False,
    type=click.Path(
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        path_type=Path,
    ),
    help="Explicit ImageJ annotation path (.roi or RoiSet.zip). Overrides --term.",
)
@click.option(
    "--ignore-imagej-roi/--no-ignore-imagej-roi",
    default=False,
    show_default=True,
    help="Ignore any auto-detected ImageJ ROI and fall back to ontology term filtering.",
)
@click.option(
    "--run-dirname",
    default="landmark_syn_mi",
    show_default=True,
    help="Subdirectory under <workspace>/analysis/output/ccf-transforms/<roi>/ containing mask_edit/.",
)
@click.option(
    "--imagej-coords-key",
    default="spatial",
    show_default=True,
    help="obsm key for coordinates to test against the ImageJ ROI (typically the original 'spatial').",
)
@click.option(
    "--imagej-input-space",
    type=click.Choice(["fused", "full", "crop"], case_sensitive=False),
    default="fused",
    show_default=True,
    help=(
        "Coordinate frame for obsm[imagej_coords_key] when using ImageJ ROI filtering. "
        "'fused' = unrotated fused.zarr pixel space; "
        "'full' = rotated full-res slice pixel space; "
        "'crop' = rotated crop-local pixel space."
    ),
)
@click.option(
    "--imagej-input-units",
    type=click.Choice(["px", "um", "mm"], case_sensitive=False),
    default="px",
    show_default=True,
    help="Units for obsm[imagej_coords_key] when using ImageJ ROI filtering.",
)
@click.option(
    "--imagej-target-spacing-um",
    type=float,
    default=2.0,
    show_default=True,
    help="Pixel size (microns) of the moving thumbnail used for ImageJ ROI annotation.",
)
@click.option(
    "--stitch-codebook",
    default="pi",
    show_default=True,
    help="Stitch codebook name used to locate fused.zarr when --imagej-input-space=fused.",
)
@click.option(
    "--kind",
    type=click.Choice(["auto", "id", "acronym", "name"], case_sensitive=False),
    default="auto",
    show_default=True,
    help="Interpretation of each --term value.",
)
@click.option(
    "--match",
    type=click.Choice(["subtree", "exact"], case_sensitive=False),
    default="subtree",
    show_default=True,
    help="Whether to match the entire ontology subtree (includes descendants) or exact region labels.",
)
@click.option(
    "--combine",
    type=click.Choice(["any", "all"], case_sensitive=False),
    default="any",
    show_default=True,
    help="How to combine multiple --term masks.",
)
@click.option("--invert/--no-invert", default=False, show_default=True, help="Invert the final selection.")
@click.option("--ccf-obsm-key", default="ccf", show_default=True, help="obsm key containing CCF columns.")
@click.option(
    "--dilate-um",
    type=float,
    default=0.0,
    show_default=True,
    help=(
        "Optional dilation radius in microns applied to the selected cell mask in atlas space "
        "(keeps any cell within this distance of the selected cells; uses obsm[coords_key])."
    ),
)
@click.option("--coords-key", default="spatial_ccf", show_default=True, help="obsm key containing atlas-space coords.")
@click.option(
    "--coords-units",
    type=click.Choice(["px", "um", "mm"], case_sensitive=False),
    default="px",
    show_default=True,
    help="Units of obsm[coords_key] (used for dilation distance calculations).",
)
@click.option(
    "--coords-space",
    type=click.Choice(["crop", "full"], case_sensitive=False),
    default="crop",
    show_default=True,
    help="Whether obsm[coords_key] is relative to atlas crop bbox (crop) or full atlas slice (full).",
)
@click.option(
    "--coords-spacing-um",
    type=float,
    default=None,
    show_default=False,
    help="Pixel size in microns if --coords-units=px (default: inferred from adata.uns['ccf']['atlas_voxel_um']).",
)
@click.option(
    "--spatial-order",
    type=click.Choice(["xy", "yx"], case_sensitive=False),
    default="xy",
    show_default=True,
    help="Order of columns in obsm[coords_key].",
)
@click.option(
    "--filter-roi/--no-filter-roi",
    default=True,
    show_default=True,
    help="Restrict selection to observations with adata.obs[roi_col] == ROI.",
)
@click.option("--roi-col", default="roi", show_default=True, help="obs column name used for ROI filtering.")
@click.option(
    "--skip-missing/--no-skip-missing",
    default=True,
    show_default=True,
    help="Skip ROIs that are missing the warped input h5ad under ccf-transforms/<roi>/.",
)
@click.option("--overwrite/--no-overwrite", default=False, show_default=True, help="Overwrite output_h5ad if it exists.")
@click.option("--qc-plot/--no-qc-plot", default=True, show_default=True, help="Write a QC PNG overlay plot.")
@click.option(
    "--qc-plot-png",
    default=None,
    show_default=False,
    type=click.Path(
        exists=False,
        file_okay=True,
        dir_okay=False,
        writable=True,
        resolve_path=True,
        path_type=Path,
    ),
    help="Optional output PNG path for the QC overlay (default: derived from output_h5ad).",
)
@click.option(
    "--debug/--no-debug",
    default=False,
    show_default=True,
    help="Enable verbose logging to <workspace>/analysis/logs/.",
)
def main(  # noqa: PLR0913
    workspace: Path,
    rois: tuple[str, ...],
    *,
    h5ad_name: str | None,
    out_name: str | None,
    ccf_col: str,
    ccf_adjusted_col: str,
    terms: tuple[str, ...],
    imagej_roi: bool,
    imagej_roi_path: Path | None,
    ignore_imagej_roi: bool,
    run_dirname: str,
    imagej_coords_key: str,
    imagej_input_space: str,
    imagej_input_units: str,
    imagej_target_spacing_um: float,
    stitch_codebook: str,
    kind: str,
    match: str,
    combine: str,
    invert: bool,
    ccf_obsm_key: str,
    dilate_um: float,
    coords_key: str,
    coords_units: str,
    coords_space: str,
    coords_spacing_um: float | None,
    spatial_order: str,
    filter_roi: bool,
    roi_col: str,
    skip_missing: bool,
    overwrite: bool,
    qc_plot: bool,
    qc_plot_png: Path | None,
    debug: bool,
) -> None:
    ws = Workspace(workspace)
    try:
        resolved_rois = ws.resolve_rois(rois)
    except ValueError as exc:
        raise click.BadParameter(str(exc), param_hint="rois") from exc

    if qc_plot_png is not None and len(resolved_rois) > 1:
        raise click.BadParameter("--qc-plot-png cannot be used when running multiple ROIs.")

    ccf_col = str(ccf_col).strip()
    if not ccf_col:
        raise click.BadParameter("--ccf-col must be non-empty.")
    ccf_adjusted_col = str(ccf_adjusted_col).strip()
    if not ccf_adjusted_col:
        raise click.BadParameter("--ccf-adjusted-col must be non-empty.")

    for roi_resolved in resolved_rois:
        input_h5ad_candidate = _input_h5ad_candidate(ws, roi=str(roi_resolved), h5ad_name=h5ad_name)
        if not input_h5ad_candidate.exists():
            if skip_missing:
                click.echo(f"Skipping roi={roi_resolved!r} (missing warped h5ad): {input_h5ad_candidate}")
                continue
            input_h5ad = _resolve_input_h5ad(ws, roi=str(roi_resolved), h5ad_name=h5ad_name)
        else:
            input_h5ad = input_h5ad_candidate
        roi_resolved, output_h5ad, plot_png_default = _default_outputs(
            ws,
            roi=str(roi_resolved),
            input_h5ad=input_h5ad,
            out_name=out_name,
        )
        plot_png = qc_plot_png if qc_plot_png is not None else plot_png_default

        output_h5ad.parent.mkdir(parents=True, exist_ok=True)
        if not overwrite:
            existing: list[Path] = []
            if output_h5ad.exists():
                existing.append(output_h5ad)
            if qc_plot and plot_png.exists():
                existing.append(plot_png)
            if existing:
                click.echo(
                    "Skipping (output already exists; pass --overwrite to replace): " + ", ".join(str(p) for p in existing)
                )
                continue

        try:
            setup_cli_logging(
                workspace,
                component="ccf.filter_h5ad_ccf",
                file=f"filter-h5ad-ccf-{roi_resolved}",
                debug=debug,
                extra={"roi": str(roi_resolved)},
            )
        except PermissionError as exc:
            click.echo(
                f"Warning: cannot write logs under {workspace}/analysis/logs (permission denied); continuing without file logging. ({exc})",
                err=True,
            )

        adata = ad.read_h5ad(input_h5ad)
        n_total = int(adata.n_obs)

        roi_mask: np.ndarray | None = None
        adata_work = adata
        if filter_roi:
            if roi_col not in adata.obs.columns:
                raise click.ClickException(f"Missing obs column {roi_col!r} in {input_h5ad}.")
            roi_mask = (adata.obs[roi_col].astype(str) == str(roi_resolved)).to_numpy()
            adata_work = adata[roi_mask].copy()
            click.echo(
                f"Scoped ROI for selection: {adata_work.n_obs}/{n_total} obs (roi_col={roi_col!r}, roi={roi_resolved!r})"
            )

        order_t = cast(SpatialOrder, str(spatial_order).lower())
        qc_units: str
        qc_coords: np.ndarray
        title: str

        roi_path: Path | None = None
        mask_edit_dir = ws.ccf_transforms(str(roi_resolved)) / str(run_dirname) / "mask_edit"
        if not ignore_imagej_roi:
            if imagej_roi_path is not None:
                roi_path = imagej_roi_path
            elif imagej_roi:
                if mask_edit_dir.exists():
                    roi_path = _find_imagej_roi_file(mask_edit_dir)
                if roi_path is None:
                    raise click.ClickException(f"--imagej-roi was requested but no ROI was found under {mask_edit_dir}.")

        imagej_labels: np.ndarray | None = None
        if roi_path is not None:
            click.echo(f"Using ImageJ ROI override: {roi_path}")
            imagej_space_t = cast(InputSpace, str(imagej_input_space).lower())
            imagej_units_t = cast(CoordUnits, str(imagej_input_units).lower())
            mask, qc_coords, imagej_labels = _mask_from_imagej_roi(
                ws=ws,
                roi=str(roi_resolved),
                adata=adata_work,
                roi_path=roi_path,
                coords_key=str(imagej_coords_key),
                input_space=imagej_space_t,
                input_units=imagej_units_t,
                spatial_order=order_t,
                stitch_codebook=str(stitch_codebook),
                target_spacing_um=float(imagej_target_spacing_um),
            )
            qc_units = "thumb_px"
            title = (
                f"ImageJ ROI selection overlay | selected={int(mask.sum())}/{int(adata_work.n_obs)} | roi={roi_resolved!r}"
            )
        else:
            if not terms:
                raise click.ClickException(
                    "At least one --term is required unless an ImageJ ROI is available (or enabled via --imagej-roi/--imagej-roi-path)."
                )

            coords_units_t = cast(CoordUnits, str(coords_units).lower())
            coords_space_t = cast(CoordSpace, str(coords_space).lower())
            _ensure_ccf_obsm(
                adata_work,
                input_h5ad=input_h5ad,
                ccf_obsm_key=ccf_obsm_key,
                coords_key=coords_key,
                coords_units=coords_units_t,
                coords_space=coords_space_t,
                spatial_order=order_t,
                roi=str(roi_resolved),
                workspace=workspace,
            )

            kind_t = cast(CCFTermKind, str(kind).lower())
            match_t = cast(MatchMode, str(match).lower())
            combine_t = cast(CombineMode, str(combine).lower())

            n_work = int(adata_work.n_obs)
            term_masks: list[np.ndarray] = []
            term_labels = np.empty(n_work, dtype=object)
            term_labels[:] = ""
            if combine_t == "all":
                for t in terms:
                    term_masks.append(
                        _mask_for_term(adata_work, str(t), kind=kind_t, match=match_t, obsm_key=ccf_obsm_key)
                    )
                mask = np.ones(n_work, dtype=bool) if term_masks else np.zeros(n_work, dtype=bool)
                for m in term_masks:
                    mask &= m
                if term_masks:
                    joined = "|".join(str(t) for t in terms)
                    term_labels[mask] = joined
            else:
                for t in terms:
                    m = _mask_for_term(adata_work, str(t), kind=kind_t, match=match_t, obsm_key=ccf_obsm_key)
                    term_masks.append(m)
                    new = m & (term_labels == "")
                    if np.any(new):
                        term_labels[new] = str(t)
                mask = np.zeros(n_work, dtype=bool)
                for m in term_masks:
                    mask |= m

            dilate = float(dilate_um)
            if dilate > 0:
                if coords_key not in adata_work.obsm:
                    raise click.ClickException(
                        f"Missing adata.obsm[{coords_key!r}] in {input_h5ad} (needed for --dilate-um)."
                    )
                coords = np.asarray(adata_work.obsm[coords_key])
                if coords.ndim != 2 or coords.shape[1] != 2:
                    raise click.ClickException(
                        f"Expected obsm[{coords_key!r}] to have shape (N,2), got {coords.shape}."
                    )
                if order_t == "yx":
                    coords = coords[:, ::-1]

                spacing_um = coords_spacing_um
                if coords_units_t == "px" and spacing_um is None:
                    ccf_meta = adata_work.uns.get("ccf", {})
                    ccf_atlas_meta = adata_work.uns.get("ccf_atlas", {})
                    if isinstance(ccf_meta, dict) and "atlas_voxel_um" in ccf_meta:
                        spacing_um = float(ccf_meta["atlas_voxel_um"])
                    elif isinstance(ccf_atlas_meta, dict) and "atlas_voxel_um" in ccf_atlas_meta:
                        spacing_um = float(ccf_atlas_meta["atlas_voxel_um"])
                coords_um = _coords_to_um(coords, units=coords_units_t, spacing_um=spacing_um)
                mask = _dilate_mask_by_radius_um(coords_um, base_mask=mask, radius_um=dilate)
                if term_masks:
                    joined = "|".join(str(t) for t in terms)
                    new = mask & (term_labels == "")
                    if np.any(new):
                        term_labels[new] = joined

            if coords_key not in adata_work.obsm:
                raise click.ClickException(f"Missing adata.obsm[{coords_key!r}] in {input_h5ad} (needed for QC plot).")
            qc_coords = np.asarray(adata_work.obsm[coords_key])
            if qc_coords.ndim != 2 or qc_coords.shape[1] != 2:
                raise click.ClickException(f"Expected obsm[{coords_key!r}] to have shape (N,2), got {qc_coords.shape}.")
            if order_t == "yx":
                qc_coords = qc_coords[:, ::-1]

            qc_units = str(coords_units_t)
            title = (
                f"CCF selection overlay | selected={int(mask.sum())}/{int(adata_work.n_obs)} | "
                f"match={match_t}, combine={combine_t}, invert={invert}, dilate_um={float(dilate_um)}"
            )

        if invert:
            mask = ~mask

        labels = imagej_labels if imagej_labels is not None else term_labels
        if invert:
            criteria = roi_path.name if roi_path is not None else "|".join(str(t) for t in terms)
            invert_label = f"not({criteria})"
            labels = np.where(mask, np.where(labels != "", labels, invert_label), "")
        else:
            labels = np.where(mask, labels, "")

        labels_full = np.empty(n_total, dtype=object)
        labels_full[:] = ""
        if roi_mask is not None:
            labels_full[roi_mask] = labels
        else:
            labels_full = labels

        if qc_plot:
            _write_qc_mask_overlay_plot(
                coords_xy=qc_coords,
                keep_mask=mask,
                output_png=plot_png,
                title=title,
                units=qc_units,
            )
            click.echo(f"Wrote QC plot: {plot_png}")

        out = adata.copy()
        out.obs[ccf_col] = pd.Series([""] * n_total, index=out.obs_names)
        out.obs[ccf_adjusted_col] = pd.Series([""] * n_total, index=out.obs_names)
        if imagej_labels is not None:
            out.obs[ccf_adjusted_col] = pd.Series(labels_full, index=out.obs_names)
        else:
            out.obs[ccf_col] = pd.Series(labels_full, index=out.obs_names)
        out.write_h5ad(output_h5ad)
        n_labeled = int(np.sum(labels_full != ""))
        target_col = ccf_adjusted_col if imagej_labels is not None else ccf_col
        click.echo(f"Wrote: {output_h5ad} (annotated {n_labeled}/{out.n_obs} obs in {target_col!r})")


if __name__ == "__main__":
    main()
