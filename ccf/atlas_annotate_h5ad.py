from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, cast

import anndata as ad
import numpy as np
import pandas as pd
from brainglobe_atlasapi import BrainGlobeAtlas
import rich_click as click

from fishtools.ccf.landmark import LandmarkRegistrationOutputs
from fishtools.ccf.sitk_utils import UM_TO_MM
from fishtools.io.workspace import Workspace
from fishtools.utils.logging import setup_cli_logging


click.rich_click.SHOW_ARGUMENTS = True
click.rich_click.GROUP_ARGUMENTS_OPTIONS = True
click.rich_click.USE_MARKDOWN = True
click.rich_click.STYLE_HELPTEXT = ""


Units = Literal["px", "um", "mm"]
Space = Literal["crop", "full"]
SpatialOrder = Literal["xy", "yx"]


def _xy_to_px(*, x: np.ndarray, y: np.ndarray, units: Units, spacing_um: float) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if units == "px":
        return x, y
    if units == "um":
        return x / float(spacing_um), y / float(spacing_um)
    if units == "mm":
        return x / (float(spacing_um) * UM_TO_MM), y / (float(spacing_um) * UM_TO_MM)
    raise ValueError(f"Unsupported units: {units!r}.")


@click.command("atlas-annotate-h5ad")
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
@click.argument("roi", type=str)
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
    "--coords-key",
    default="spatial_ccf",
    show_default=True,
    help="obsm key containing atlas-space coordinates to annotate.",
)
@click.option(
    "--coords-space",
    type=click.Choice(["crop", "full"], case_sensitive=False),
    default="crop",
    show_default=True,
    help="Whether coords are relative to the atlas crop bbox (crop) or full atlas slice (full).",
)
@click.option(
    "--coords-units",
    type=click.Choice(["px", "um", "mm"], case_sensitive=False),
    default="px",
    show_default=True,
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
    help="Filter observations by adata.obs[roi_col] == ROI before annotating.",
)
@click.option("--roi-col", default="roi", show_default=True, help="obs column name for ROI filtering.")
@click.option(
    "--prefix",
    default="ccf",
    show_default=True,
    help="Key used for output in obsm/uns (e.g. obsm['ccf'], uns['ccf']).",
)
@click.option(
    "--metrics-json",
    default=None,
    type=click.Path(
        exists=False,
        file_okay=True,
        dir_okay=False,
        writable=True,
        resolve_path=True,
        path_type=Path,
    ),
    help="Optional path to write a JSON summary (counts per region, etc.).",
)
@click.option(
    "--debug/--no-debug",
    default=False,
    show_default=True,
    help="Enable verbose logging to <workspace>/analysis/logs/ when writable.",
)
def main(  # noqa: PLR0913
    workspace: Path,
    roi: str,
    input_h5ad: Path,
    output_h5ad: Path,
    *,
    coords_key: str,
    coords_space: str,
    coords_units: str,
    spatial_order: str,
    filter_roi: bool,
    roi_col: str,
    prefix: str,
    metrics_json: Path | None,
    debug: bool,
) -> None:
    ws = Workspace(workspace)
    try:
        (roi_resolved,) = tuple(ws.resolve_rois((roi,)))
    except ValueError as exc:
        raise click.BadParameter(str(exc), param_hint="roi") from exc

    try:
        setup_cli_logging(
            workspace,
            component="ccf.atlas_annotate_h5ad",
            file=f"atlas-annotate-h5ad-{roi_resolved}",
            debug=debug,
            extra={"roi": str(roi_resolved)},
        )
    except PermissionError as exc:
        click.echo(
            f"Warning: cannot write logs under {workspace}/analysis/logs (permission denied); continuing without file logging. ({exc})",
            err=True,
        )

    out_root = ws.ccf_transforms(roi_resolved)
    out_contract = LandmarkRegistrationOutputs(out_root)
    p1 = out_contract.read_p1_landmarks()

    atlas_voxel_um = float(p1.atlas_voxel_um) if p1.atlas_voxel_um is not None else 20.0
    atlas_name = p1.atlas_name or "kim_dev_mouse_e15-5_lsfm_20um"
    atlas_plane = p1.atlas_plane or ("sagittal" if "Sag" in Path(workspace).name else "coronal")
    if p1.atlas_slice_idx is None:
        raise ValueError(f"p1_landmarks.json is missing atlas_slice_idx under {out_root}.")
    atlas_slice_idx = int(p1.atlas_slice_idx)

    ar0, ar1, ac0, ac1 = p1.atlas_crop_bbox
    crop_offset_xy = (float(ac0), float(ar0))

    atlas = BrainGlobeAtlas(atlas_name)
    ann_vol = atlas.annotation if atlas_plane == "coronal" else atlas.annotation.transpose(2, 1, 0)
    ann_slice = np.asarray(ann_vol[atlas_slice_idx, :, :], dtype=np.uint32)
    if ann_slice.ndim != 2:
        raise ValueError(f"Expected 2D annotation slice, got shape={ann_slice.shape}.")

    adata = ad.read_h5ad(input_h5ad)
    if filter_roi:
        if roi_col not in adata.obs.columns:
            raise click.BadParameter(f"Missing obs column {roi_col!r} for ROI filtering in {input_h5ad}.")
        n_before = int(adata.n_obs)
        mask = adata.obs[roi_col].astype(str) == str(roi_resolved)
        adata = adata[mask].copy()
        if adata.n_obs == 0:
            raise ValueError(f"No observations left after filtering {roi_col}={roi_resolved!r} in {input_h5ad}.")
        click.echo(f"Filtered {input_h5ad} to {roi_col}={roi_resolved}: {adata.n_obs}/{n_before} obs")

    if coords_key not in adata.obsm:
        raise click.BadParameter(f"Missing obsm[{coords_key!r}] in {input_h5ad}.")

    coords = np.asarray(adata.obsm[coords_key])
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"Expected obsm[{coords_key!r}] to have shape (N,2), got {coords.shape}.")
    coords = coords.astype(np.float64, copy=False)

    spatial_order_t = cast(SpatialOrder, str(spatial_order).lower())
    if spatial_order_t == "xy":
        x = coords[:, 0]
        y = coords[:, 1]
    else:
        x = coords[:, 1]
        y = coords[:, 0]

    coords_space_t = cast(Space, str(coords_space).lower())
    coords_units_t = cast(Units, str(coords_units).lower())

    x_px, y_px = _xy_to_px(x=x, y=y, units=coords_units_t, spacing_um=atlas_voxel_um)

    if coords_space_t == "crop":
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

    acronym_arr = np.asarray(uniq_acronym, dtype=object)[inv]
    name_arr = np.asarray(uniq_name, dtype=object)[inv]
    parent_id_arr = np.asarray(uniq_parent, dtype=np.int64)[inv]
    path_ids_arr = np.asarray(uniq_path_ids, dtype=object)[inv]
    path_acr_arr = np.asarray(uniq_path_acr, dtype=object)[inv]
    path_name_arr = np.asarray(uniq_path_name, dtype=object)[inv]
    is_leaf_arr = np.asarray(uniq_is_leaf, dtype=bool)[inv]

    pref = str(prefix).strip()
    if not pref:
        raise click.BadParameter("prefix must be non-empty.")

    ccf_df = pd.DataFrame(
        {
            "id": ids.astype(np.int64, copy=False),
            "acronym": pd.Categorical(acronym_arr),
            "name": pd.Categorical(name_arr),
            "parent_id": parent_id_arr,
            "path_ids": pd.Categorical(path_ids_arr),
            "path_acronyms": pd.Categorical(path_acr_arr),
            "path_names": pd.Categorical(path_name_arr),
            "is_leaf": is_leaf_arr,
            "in_bounds": in_bounds,
        },
        index=adata.obs_names,
    )
    adata.obsm[pref] = ccf_df

    atlas_meta = {
        "atlas_name": atlas_name,
        "atlas_plane": atlas_plane,
        "atlas_slice_idx": atlas_slice_idx,
        "atlas_voxel_um": atlas_voxel_um,
        "coords_key": coords_key,
        "coords_space": coords_space_t,
        "coords_units": coords_units_t,
        "spatial_order": spatial_order_t,
        "atlas_crop_bbox": [int(ar0), int(ar1), int(ac0), int(ac1)],
    }
    adata.uns[f"{pref}_atlas"] = atlas_meta

    # Store ontology info in a JSON-friendly form.
    used: dict[str, dict[str, object]] = {}
    for sid in uniq.tolist():
        sid_int = int(sid)
        if sid_int == 0:
            used["0"] = {
                "id": 0,
                "acronym": "background",
                "name": "background",
                "structure_id_path": [],
                "parent_id": -1,
                "rgb_triplet": [0, 0, 0],
                "is_leaf": False,
            }
            continue
        st = structures[sid_int]
        path_ids = [int(v) for v in st["structure_id_path"]]
        parent_id = int(path_ids[-2]) if len(path_ids) >= 2 else sid_int
        used[str(sid_int)] = {
            "id": sid_int,
            "acronym": str(st["acronym"]),
            "name": str(st["name"]),
            "structure_id_path": path_ids,
            "parent_id": parent_id,
            "rgb_triplet": [int(v) for v in st["rgb_triplet"]],
            "is_leaf": len(tree.children(sid_int)) == 0,
        }

    adata.uns[pref] = {
        "atlas_name": atlas_name,
        "atlas_plane": atlas_plane,
        "atlas_slice_idx": atlas_slice_idx,
        "atlas_voxel_um": atlas_voxel_um,
        "atlas": atlas_meta,
        "obsm_key": pref,
        "obsm_columns": list(ccf_df.columns),
        "structures_used": used,
    }

    adata.write_h5ad(output_h5ad)

    if metrics_json is not None:
        counts = pd.Series(ids.astype(np.int64)).value_counts().sort_values(ascending=False)
        top = counts.head(25)
        top_payload: list[dict[str, object]] = []
        for sid, n in top.items():
            sid_int = int(sid)
            if sid_int == 0:
                top_payload.append({"id": 0, "acronym": "background", "name": "background", "n": int(n)})
            else:
                st = structures[sid_int]
                top_payload.append(
                    {"id": sid_int, "acronym": str(st["acronym"]), "name": str(st["name"]), "n": int(n)}
                )
        payload = {
            "workspace": str(workspace),
            "roi": str(roi_resolved),
            "input_h5ad": str(input_h5ad),
            "output_h5ad": str(output_h5ad),
            "atlas_name": atlas_name,
            "atlas_plane": atlas_plane,
            "atlas_slice_idx": atlas_slice_idx,
            "atlas_voxel_um": atlas_voxel_um,
            "coords_key": coords_key,
            "coords_space": coords_space_t,
            "coords_units": coords_units_t,
            "spatial_order": spatial_order_t,
            "n_obs": int(adata.n_obs),
            "frac_in_bounds": float(in_bounds.mean()) if in_bounds.size else float("nan"),
            "n_unique_ids": int(uniq.size),
            "top_regions": top_payload,
        }
        metrics_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    click.echo(f"Wrote annotated h5ad: {output_h5ad}")
    if metrics_json is not None:
        click.echo(f"Wrote metrics: {metrics_json}")


if __name__ == "__main__":
    main()
