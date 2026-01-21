from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, cast

import anndata as ad
import numpy as np
import pandas as pd
import rich_click as click
from brainglobe_atlasapi import BrainGlobeAtlas
from scipy.spatial import cKDTree

from fishtools.ccf.ontology import CCFTermKind, mask_ccf_subtree
from fishtools.ccf.landmark import LandmarkRegistrationOutputs
from fishtools.io.workspace import Workspace


click.rich_click.SHOW_ARGUMENTS = True
click.rich_click.GROUP_ARGUMENTS_OPTIONS = True
click.rich_click.USE_MARKDOWN = True
click.rich_click.STYLE_HELPTEXT = ""


CombineMode = Literal["any", "all"]
MatchMode = Literal["subtree", "exact"]
CoordUnits = Literal["px", "um", "mm"]
SpatialOrder = Literal["xy", "yx"]
CoordSpace = Literal["crop", "full"]


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


def _infer_workspace_path(*, input_h5ad: Path, adata: ad.AnnData) -> Path:
    meta = adata.uns.get("fishtools", {}).get("segment_export")
    if isinstance(meta, dict):
        workspace_path = meta.get("workspace_path")
        if isinstance(workspace_path, str) and workspace_path:
            return Path(workspace_path)
    return input_h5ad.parent


def _infer_roi_for_ccf(*, adata: ad.AnnData, roi: str | None, roi_col: str, input_h5ad: Path) -> str:
    if roi is not None:
        return str(roi)
    if roi_col not in adata.obs.columns:
        raise click.ClickException(f"Missing obs column {roi_col!r} in {input_h5ad}; pass --roi.")
    values = adata.obs[roi_col].astype(str).unique().tolist()
    if len(values) != 1:
        raise click.ClickException(f"Expected a single ROI in obs[{roi_col!r}] but found {values}; pass --roi.")
    return str(values[0])


def _ensure_ccf_obsm(
    adata: ad.AnnData,
    *,
    input_h5ad: Path,
    ccf_obsm_key: str,
    coords_key: str,
    coords_units: CoordUnits,
    coords_space: CoordSpace,
    spatial_order: SpatialOrder,
    roi: str | None,
    roi_col: str,
    workspace: Path | None,
) -> None:
    if ccf_obsm_key in adata.obsm:
        return
    if coords_key not in adata.obsm:
        raise click.ClickException(
            f"Missing adata.obsm[{ccf_obsm_key!r}] in {input_h5ad}, and cannot auto-annotate because "
            f"adata.obsm[{coords_key!r}] is missing."
        )

    ws_path = workspace if workspace is not None else _infer_workspace_path(input_h5ad=input_h5ad, adata=adata)
    ws = Workspace(ws_path)
    roi_for_ccf = _infer_roi_for_ccf(adata=adata, roi=roi, roi_col=roi_col, input_h5ad=input_h5ad)

    out_root = ws.ccf_transforms(roi_for_ccf)
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


@click.command("filter-h5ad-ccf")
@click.argument(
    "input_h5ad",
    type=click.Path(
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        path_type=Path,
    ),
)
@click.argument(
    "output_h5ad",
    type=click.Path(
        exists=False,
        file_okay=True,
        dir_okay=False,
        writable=True,
        resolve_path=True,
        path_type=Path,
    ),
)
@click.option(
    "--term",
    "terms",
    multiple=True,
    required=True,
    help="Ontology term(s) to select (name/acronym/id). Repeatable.",
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
    "--workspace",
    default=None,
    type=click.Path(
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
        path_type=Path,
    ),
    help=(
        "Optional workspace root used to auto-annotate when obsm[ccf_obsm_key] is missing "
        "(default: inferred from adata.uns['fishtools']['segment_export']['workspace_path'] or input_h5ad location)."
    ),
)
@click.option("--roi", default=None, show_default=False, help="Optional ROI value to pre-filter (obs[roi_col] == roi).")
@click.option("--roi-col", default="roi", show_default=True, help="obs column name used for ROI filtering.")
@click.option("--overwrite/--no-overwrite", default=False, show_default=True, help="Overwrite output_h5ad if it exists.")
def main(  # noqa: PLR0913
    input_h5ad: Path,
    output_h5ad: Path,
    *,
    terms: tuple[str, ...],
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
    workspace: Path | None,
    roi: str | None,
    roi_col: str,
    overwrite: bool,
) -> None:
    if output_h5ad.exists() and not overwrite:
        raise click.ClickException(f"Refusing to overwrite existing file: {output_h5ad} (pass --overwrite).")

    adata = ad.read_h5ad(input_h5ad)
    n0 = int(adata.n_obs)

    if roi is not None:
        if roi_col not in adata.obs.columns:
            raise click.ClickException(f"Missing obs column {roi_col!r} in {input_h5ad}.")
        roi_mask = (adata.obs[roi_col].astype(str) == str(roi)).to_numpy()
        adata = adata[roi_mask].copy()
        click.echo(f"Filtered ROI: {adata.n_obs}/{n0} obs (roi_col={roi_col!r}, roi={roi!r})")
        n0 = int(adata.n_obs)

    coords_units_t = cast(CoordUnits, str(coords_units).lower())
    coords_space_t = cast(CoordSpace, str(coords_space).lower())
    order_t = cast(SpatialOrder, str(spatial_order).lower())
    _ensure_ccf_obsm(
        adata,
        input_h5ad=input_h5ad,
        ccf_obsm_key=ccf_obsm_key,
        coords_key=coords_key,
        coords_units=coords_units_t,
        coords_space=coords_space_t,
        spatial_order=order_t,
        roi=roi,
        roi_col=roi_col,
        workspace=workspace,
    )

    kind_t = cast(CCFTermKind, str(kind).lower())
    match_t = cast(MatchMode, str(match).lower())
    combine_t = cast(CombineMode, str(combine).lower())

    if combine_t == "all":
        mask = np.ones(n0, dtype=bool)
        for t in terms:
            mask &= _mask_for_term(adata, str(t), kind=kind_t, match=match_t, obsm_key=ccf_obsm_key)
    else:
        mask = np.zeros(n0, dtype=bool)
        for t in terms:
            mask |= _mask_for_term(adata, str(t), kind=kind_t, match=match_t, obsm_key=ccf_obsm_key)

    dilate = float(dilate_um)
    if dilate > 0:
        if coords_key not in adata.obsm:
            raise click.ClickException(f"Missing adata.obsm[{coords_key!r}] in {input_h5ad} (needed for --dilate-um).")
        coords = np.asarray(adata.obsm[coords_key])
        if coords.ndim != 2 or coords.shape[1] != 2:
            raise click.ClickException(f"Expected obsm[{coords_key!r}] to have shape (N,2), got {coords.shape}.")
        if order_t == "yx":
            coords = coords[:, ::-1]

        spacing_um = coords_spacing_um
        if coords_units_t == "px" and spacing_um is None:
            ccf_meta = adata.uns.get("ccf", {})
            ccf_atlas_meta = adata.uns.get("ccf_atlas", {})
            if isinstance(ccf_meta, dict) and "atlas_voxel_um" in ccf_meta:
                spacing_um = float(ccf_meta["atlas_voxel_um"])
            elif isinstance(ccf_atlas_meta, dict) and "atlas_voxel_um" in ccf_atlas_meta:
                spacing_um = float(ccf_atlas_meta["atlas_voxel_um"])
        coords_um = _coords_to_um(coords, units=coords_units_t, spacing_um=spacing_um)
        mask = _dilate_mask_by_radius_um(coords_um, base_mask=mask, radius_um=dilate)

    if invert:
        mask = ~mask

    out = adata[mask].copy()
    out.write_h5ad(output_h5ad)
    click.echo(f"Wrote: {output_h5ad} ({out.n_obs}/{n0} obs kept)")


if __name__ == "__main__":
    main()
