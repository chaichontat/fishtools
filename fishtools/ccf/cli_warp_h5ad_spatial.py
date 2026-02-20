from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, TypedDict, cast

import anndata as ad
import ants
import matplotlib as mpl
from loguru import logger

# Force a non-interactive backend to avoid GUI/event-loop hangs in headless runs
mpl.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
import rich_click as click
import SimpleITK as sitk
import zarr
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import rotate as ndimage_rotate

from fishtools.ccf.landmark import LandmarkRegistrationOutputs
from fishtools.ccf.ndimage_geometry import (
    fused_xy_to_rotated_crop_xy,
    ndimage_rotate_input_to_output_yx,
    ndimage_rotate_output_to_input_yx,
)
from fishtools.ccf.sitk_utils import UM_TO_MM, normalize_robust
from fishtools.io.workspace import Workspace
from fishtools.utils.logging import setup_cli_logging


click.rich_click.SHOW_ARGUMENTS = True
click.rich_click.GROUP_ARGUMENTS_OPTIONS = True
click.rich_click.USE_MARKDOWN = True
click.rich_click.STYLE_HELPTEXT = ""


class _SummaryPaths(TypedDict, total=False):
    fixed_nifti: str
    fixed_mask_orig_nifti: str
    overlap_mask_final_nifti: str


class _Summary(TypedDict, total=False):
    fwdtransforms: list[str] | str
    invtransforms: list[str] | str
    landmark_error_mm: dict[str, float]
    paths: _SummaryPaths


Units = Literal["px", "um", "mm"]
Direction = Literal["moving-to-fixed", "fixed-to-moving"]
InputSpace = Literal["crop", "full", "fused"]
OutputSpace = Literal["crop", "full"]
SpatialOrder = Literal["xy", "yx"]


def _load_json_object(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid JSON payload in {path}: expected an object, got {type(payload).__name__}.")
    return cast(dict[str, object], payload)


def _as_list_of_str(value: object, *, key: str, path: Path) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(v, str) for v in value):
        return list(value)
    raise ValueError(f"Invalid {key} in {path}: expected string or list[str], got {type(value).__name__}.")


def _resolve_paths(paths: list[str], *, base_dir: Path) -> list[str]:
    resolved: list[str] = []
    for p in paths:
        pp = Path(p)
        if not pp.is_absolute():
            pp = base_dir / pp
        resolved.append(str(pp))
    return resolved


def _xy_to_mm(*, x: np.ndarray, y: np.ndarray, units: Units, spacing_um: float) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if units == "mm":
        return x, y
    if units == "um":
        return x * UM_TO_MM, y * UM_TO_MM
    if units == "px":
        sp_mm = float(spacing_um) * UM_TO_MM
        return x * sp_mm, y * sp_mm
    raise ValueError(f"Unsupported units: {units!r}.")


def _xy_from_mm(*, x_mm: np.ndarray, y_mm: np.ndarray, units: Units, spacing_um: float) -> tuple[np.ndarray, np.ndarray]:
    x_mm = np.asarray(x_mm, dtype=np.float64)
    y_mm = np.asarray(y_mm, dtype=np.float64)
    if units == "mm":
        return x_mm, y_mm
    if units == "um":
        return x_mm / UM_TO_MM, y_mm / UM_TO_MM
    if units == "px":
        sp_mm = float(spacing_um) * UM_TO_MM
        return x_mm / sp_mm, y_mm / sp_mm
    raise ValueError(f"Unsupported units: {units!r}.")


def _apply_transforms_to_points_mm(
    *, x_mm: np.ndarray, y_mm: np.ndarray, transformlist: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    if x_mm.shape != y_mm.shape:
        raise ValueError(f"x_mm/y_mm shape mismatch: {x_mm.shape} vs {y_mm.shape}.")
    df = pd.DataFrame({"x": x_mm, "y": y_mm, "z": 0.0, "t": 0.0})
    out = ants.apply_transforms_to_points(dim=2, points=df, transformlist=transformlist)
    return (
        out["x"].to_numpy(dtype=np.float64, copy=False),
        out["y"].to_numpy(dtype=np.float64, copy=False),
    )


def _rmse_mm(
    *, pred_x_mm: np.ndarray, pred_y_mm: np.ndarray, true_x_mm: np.ndarray, true_y_mm: np.ndarray
) -> dict[str, float]:
    pred = np.stack([pred_x_mm, pred_y_mm], axis=1)
    true = np.stack([true_x_mm, true_y_mm], axis=1)
    err = np.linalg.norm(pred - true, axis=1)
    return {
        "rmse_mm": float(np.sqrt(np.mean(err**2))),
        "mean_mm": float(err.mean()),
        "max_mm": float(err.max()) if err.size else 0.0,
    }


def _mask_metrics(
    *,
    x_px: np.ndarray,
    y_px: np.ndarray,
    mask_yx: np.ndarray,
    spacing_um: float,
) -> dict[str, float]:
    x_px = np.asarray(x_px, dtype=np.float64)
    y_px = np.asarray(y_px, dtype=np.float64)
    if x_px.shape != y_px.shape:
        raise ValueError(f"x_px/y_px shape mismatch: {x_px.shape} vs {y_px.shape}.")

    mask = np.asarray(mask_yx, dtype=bool)
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape={mask.shape}.")

    xi = np.round(x_px).astype(np.int64, copy=False)
    yi = np.round(y_px).astype(np.int64, copy=False)
    h, w = mask.shape
    in_bounds = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
    inside = np.zeros_like(in_bounds, dtype=bool)
    inside[in_bounds] = mask[yi[in_bounds], xi[in_bounds]]

    n = int(xi.size)
    n_in = int(inside.sum())
    n_oob = int((~in_bounds).sum())
    n_outside_inb = int((in_bounds & (~inside)).sum())

    out: dict[str, float] = {
        "n_points": float(n),
        "frac_in_bounds": float(in_bounds.mean()) if n else float("nan"),
        "frac_oob": float(n_oob / n) if n else float("nan"),
        "frac_inside_mask": float(n_in / n) if n else float("nan"),
    }

    if n_outside_inb:
        dt_px = distance_transform_edt(~mask).astype(np.float32)
        dt_um = dt_px * float(spacing_um)
        outside_dt = dt_um[yi[in_bounds & (~inside)], xi[in_bounds & (~inside)]]
        out["outside_dist_mean_um"] = float(outside_dt.mean())
        out["outside_dist_max_um"] = float(outside_dt.max()) if outside_dt.size else 0.0
    else:
        out["outside_dist_mean_um"] = float("nan")
        out["outside_dist_max_um"] = float("nan")
    return out


def _ants_from_numpy_yx(*, arr_yx: np.ndarray, spacing_um: float) -> ants.ANTsImage:
    arr = np.asarray(arr_yx, dtype=np.float32)
    spacing_mm = float(spacing_um) * UM_TO_MM
    return ants.from_numpy(
        arr.T,
        origin=(0.0, 0.0),
        spacing=(spacing_mm, spacing_mm),
        direction=np.eye(2, dtype=np.float64),
    )


def _ants_numpy_yx(img: ants.ANTsImage) -> np.ndarray:
    return np.asarray(img.numpy(), dtype=np.float32).T


def _default_outputs(
    ws: Workspace,
    *,
    roi: str,
    input_h5ad: Path,
    out_name: str | None,
) -> tuple[str, Path, Path, Path]:
    try:
        (roi_resolved,) = tuple(ws.resolve_rois((roi,)))
    except ValueError as exc:
        raise click.BadParameter(str(exc), param_hint="roi") from exc

    out_dir = ws.output.ccf_transforms / roi_resolved
    if out_name is None:
        out_name = f"{roi_resolved}.syn.h5ad"
    output_h5ad = out_dir / out_name
    plot_png = output_h5ad.with_suffix(".qc.png")
    metrics_json = output_h5ad.with_suffix(".metrics.json")
    return (str(roi_resolved), output_h5ad, plot_png, metrics_json)


def _resolve_input_h5ad(ws: Workspace, *, roi: str, h5ad_name: str | None) -> Path:
    base = ws.output / "h5ads"
    if h5ad_name is None:
        candidate = base / f"{roi}.h5ad"
    else:
        name = str(h5ad_name).strip()
        if not name:
            raise click.BadParameter("--h5ad-name must be non-empty.")
        if Path(name).name != name:
            raise click.BadParameter("--h5ad-name must be a filename only (no directories).")
        if not name.endswith(".h5ad"):
            name = f"{name}.h5ad"
        candidate = base / name

    if not candidate.exists():
        raise click.ClickException(
            f"Missing export h5ad for roi={roi!r}: {candidate}. "
            "Run `segment export <workspace> <roi> ...` first."
        )
    return candidate


def _format_mtime(path: Path) -> str:
    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).astimezone()
    return mtime.isoformat(timespec="seconds")


@click.command("warp-h5ad-spatial")
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
@click.argument("roi", type=str, required=False)
@click.option(
    "--h5ad-name",
    default=None,
    show_default=False,
    help=(
        "Input h5ad filename under <workspace>/analysis/output/h5ads/. "
        "Default: <roi>.h5ad"
    ),
)
@click.option(
    "--out-name",
    default=None,
    show_default=False,
    help=(
        "Output filename (written under <workspace>/analysis/output/ccf-transforms/<roi>/). "
        "Default: <roi>.syn.h5ad."
    ),
)
@click.option(
    "--overwrite/--no-overwrite",
    default=False,
    show_default=True,
    help="Allow overwriting an existing output h5ad/metrics/qc plot.",
)
@click.option(
    "--run-dirname",
    default="landmark_syn_mi",
    show_default=True,
    help="Subdirectory under <workspace>/analysis/output/ccf-transforms/<roi>/ containing similarity_plus_syn_summary.json.",
)
@click.option(
    "--direction",
    type=click.Choice(["moving-to-fixed", "fixed-to-moving"], case_sensitive=False),
    default="moving-to-fixed",
    show_default=True,
)
@click.option(
    "--input-space",
    type=click.Choice(["fused", "full", "crop"], case_sensitive=False),
    default="fused",
    show_default=True,
    help=(
        "Coordinate frame for input coords. "
        "'fused' = unrotated fused.zarr pixel space (pre prior flip/rotation/crop); "
        "'full' = rotated full-res slice pixel space; "
        "'crop' = rotated crop-local pixel space."
    ),
)
@click.option(
    "--output-space",
    type=click.Choice(["crop", "full"], case_sensitive=False),
    default="crop",
    show_default=True,
    help="Whether output coords are relative to the atlas crop bbox (crop) or full atlas slice (full).",
)
@click.option(
    "--input-units",
    type=click.Choice(["px", "um", "mm"], case_sensitive=False),
    default="px",
    show_default=True,
)
@click.option(
    "--output-units",
    type=click.Choice(["px", "um", "mm"], case_sensitive=False),
    default="px",
    show_default=True,
)
@click.option("--in-key", default="spatial", show_default=True, help="obsm key for input coords.")
@click.option("--out-key", default="spatial_ccf", show_default=True, help="obsm key for warped output coords.")
@click.option(
    "--spatial-order",
    type=click.Choice(["xy", "yx"], case_sensitive=False),
    default="xy",
    show_default=True,
    help="Order of columns in obsm[in_key].",
)
@click.option(
    "--filter-roi/--no-filter-roi",
    default=True,
    show_default=True,
    help="Filter observations by adata.obs[roi_col] == ROI before warping.",
)
@click.option("--roi-col", default="roi", show_default=True, help="obs column name for ROI filtering.")
@click.option(
    "--keep-input/--no-keep-input",
    default=False,
    show_default=True,
    help="If set, copy original coords to obsm[out_key + '_in'].",
)
@click.option(
    "--stitch-codebook",
    default="pi",
    show_default=True,
    help="Stitch codebook name used for fused.zarr (analysis/deconv/stitch--{ROI}+{CODEBOOK}/fused.zarr).",
)
@click.option(
    "--thumbnail-downsample",
    type=int,
    default=32,
    show_default=True,
    help="Downsample factor for fused thumbnail QC (1 = full resolution; recommended >= 16).",
)
@click.option(
    "--max-points-plot",
    type=int,
    default=200_000,
    show_default=True,
    help="Max points to plot in the QC scatter panels (random subset).",
)
@click.option(
    "--metrics-max-points",
    type=int,
    default=100_000,
    show_default=True,
    help="Max points to use for round-trip metrics (random subset).",
)
@click.option(
    "--random-seed",
    type=int,
    default=0,
    show_default=True,
)
@click.option(
    "--debug/--no-debug",
    default=False,
    show_default=True,
    help="Enable verbose logging to <workspace>/analysis/logs/.",
)
def main(  # noqa: PLR0913
    workspace: Path,
    roi: str | None,
    *,
    h5ad_name: str | None,
    out_name: str | None,
    overwrite: bool,
    run_dirname: str,
    direction: str,
    input_space: str,
    output_space: str,
    input_units: str,
    output_units: str,
    in_key: str,
    out_key: str,
    spatial_order: str,
    filter_roi: bool,
    roi_col: str,
    keep_input: bool,
    stitch_codebook: str,
    thumbnail_downsample: int,
    max_points_plot: int,
    metrics_max_points: int,
    random_seed: int,
    debug: bool,
) -> None:
    ws = Workspace(workspace)
    try:
        rois_resolved = ws.resolve_rois((roi,)) if roi is not None else ws.resolve_rois(None)
    except ValueError as exc:
        raise click.BadParameter(str(exc), param_hint="roi") from exc

    processed = 0
    skipped_missing_h5ad = 0
    process_all_rois = roi is None
    for roi_resolved in rois_resolved:
        try:
            _run_one_roi(
                ws=ws,
                workspace=workspace,
                roi_resolved=str(roi_resolved),
                h5ad_name=h5ad_name,
                out_name=out_name,
                overwrite=overwrite,
                run_dirname=run_dirname,
                direction=direction,
                input_space=input_space,
                output_space=output_space,
                input_units=input_units,
                output_units=output_units,
                in_key=in_key,
                out_key=out_key,
                spatial_order=spatial_order,
                filter_roi=filter_roi,
                roi_col=roi_col,
                keep_input=keep_input,
                stitch_codebook=stitch_codebook,
                thumbnail_downsample=thumbnail_downsample,
                max_points_plot=max_points_plot,
                metrics_max_points=metrics_max_points,
                random_seed=random_seed,
                debug=debug,
            )
            processed += 1
        except FileNotFoundError as exc:
            click.echo(f"[{roi_resolved}] Skipping: {exc}")
            continue
        except click.ClickException as exc:
            msg = getattr(exc, "message", str(exc))
            if process_all_rois and str(msg).startswith("Missing export h5ad"):
                click.echo(f"[{roi_resolved}] Skipping: {msg}")
                skipped_missing_h5ad += 1
                continue
            if process_all_rois and str(msg).startswith("Refusing to overwrite existing output(s):"):
                click.echo(f"[{roi_resolved}] Warning: output already exists, skipping. {msg}")
                continue
            raise click.ClickException(f"ROI {roi_resolved!r}: {msg}") from exc
        except Exception as exc:
            raise click.ClickException(f"ROI {roi_resolved!r}: {exc}") from exc

    if process_all_rois and processed == 0 and skipped_missing_h5ad == len(rois_resolved):
        click.echo("No ROIs processed (all missing input h5ads).")


def _run_one_roi(  # noqa: PLR0913
    *,
    ws: Workspace,
    workspace: Path,
    roi_resolved: str,
    h5ad_name: str | None,
    out_name: str | None,
    overwrite: bool,
    run_dirname: str,
    direction: str,
    input_space: str,
    output_space: str,
    input_units: str,
    output_units: str,
    in_key: str,
    out_key: str,
    spatial_order: str,
    filter_roi: bool,
    roi_col: str,
    keep_input: bool,
    stitch_codebook: str,
    thumbnail_downsample: int,
    max_points_plot: int,
    metrics_max_points: int,
    random_seed: int,
    debug: bool,
) -> None:
    input_h5ad_resolved = _resolve_input_h5ad(ws, roi=str(roi_resolved), h5ad_name=h5ad_name)
    roi_resolved, output_h5ad, plot_png, metrics_json = _default_outputs(
        ws,
        roi=str(roi_resolved),
        input_h5ad=input_h5ad_resolved,
        out_name=out_name,
    )

    output_h5ad.parent.mkdir(parents=True, exist_ok=True)
    if not overwrite:
        existing = [p for p in (output_h5ad, plot_png, metrics_json) if p.exists()]
        if existing:
            raise click.ClickException(
                "Refusing to overwrite existing output(s): " + ", ".join(str(p) for p in existing)
            )

    try:
        setup_cli_logging(
            workspace,
            component="ccf.warp_h5ad_spatial",
            file=f"warp-h5ad-spatial-{roi_resolved}",
            debug=debug,
            extra={"roi": str(roi_resolved)},
        )
    except PermissionError as exc:
        click.echo(
            f"Warning: cannot write logs under {workspace}/analysis/logs (permission denied); continuing without file logging. ({exc})",
            err=True,
        )

    out_root = ws.ccf_transforms(roi_resolved)
    run_dir = out_root / str(run_dirname)
    summary_path = run_dir / "similarity_plus_syn_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing {summary_path}. Did you run ccf/ants_landmark_syn_init.py?")

    summary_raw = _load_json_object(summary_path)
    summary = cast(_Summary, summary_raw)
    if "invtransforms" not in summary or "fwdtransforms" not in summary:
        raise ValueError(f"Invalid summary at {summary_path}: missing fwdtransforms/invtransforms.")

    inv_list = _resolve_paths(
        _as_list_of_str(summary["invtransforms"], key="invtransforms", path=summary_path),
        base_dir=run_dir,
    )
    fwd_list = _resolve_paths(
        _as_list_of_str(summary["fwdtransforms"], key="fwdtransforms", path=summary_path),
        base_dir=run_dir,
    )
    for p in [*inv_list, *fwd_list]:
        if not Path(p).exists():
            raise FileNotFoundError(f"Transform file not found: {p} (referenced by {summary_path})")
    for p in inv_list:
        pp = Path(p)
        logger.info(f"ANTs invtransform: {pp} (mtime={_format_mtime(pp)})")
    for p in fwd_list:
        pp = Path(p)
        logger.info(f"ANTs fwdtransform: {pp} (mtime={_format_mtime(pp)})")

    direction_t = cast(Direction, str(direction).lower())
    input_space_t = cast(InputSpace, str(input_space).lower())
    output_space_t = cast(OutputSpace, str(output_space).lower())
    input_units_t = cast(Units, str(input_units).lower())
    output_units_t = cast(Units, str(output_units).lower())
    spatial_order_t = cast(SpatialOrder, str(spatial_order).lower())

    out_contract = LandmarkRegistrationOutputs(out_root)
    p1 = out_contract.read_p1_landmarks()

    atlas_voxel_um = float(p1.atlas_voxel_um) if p1.atlas_voxel_um is not None else 20.0
    sample_voxel_um = float(p1.sample_voxel_xy_um) if p1.sample_voxel_xy_um is not None else 0.216

    ar0, _, ac0, _ = p1.atlas_crop_bbox
    sr0, sr1, sc0, sc1 = p1.sample_rotated_crop_bbox
    atlas_crop_offset_xy = (float(ac0), float(ar0))
    sample_crop_offset_xy = (float(sc0), float(sr0))

    # ---- Load coords ----
    adata = ad.read_h5ad(input_h5ad_resolved)
    if filter_roi:
        if roi_col not in adata.obs.columns:
            raise click.BadParameter(
                f"Missing obs column {roi_col!r} for ROI filtering in {input_h5ad_resolved}."
            )
        n_before = int(adata.n_obs)
        mask = adata.obs[roi_col].astype(str) == str(roi_resolved)
        adata = adata[mask].copy()
        if adata.n_obs == 0:
            raise ValueError(
                f"No observations left after filtering {roi_col}={roi_resolved!r} in {input_h5ad_resolved}."
            )
        click.echo(f"Filtered {input_h5ad_resolved} to {roi_col}={roi_resolved}: {adata.n_obs}/{n_before} obs")
    if in_key not in adata.obsm:
        raise click.BadParameter(f"Missing obsm[{in_key!r}] in {input_h5ad_resolved}.")
    coords = np.asarray(adata.obsm[in_key])
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"Expected obsm[{in_key!r}] to have shape (N,2), got {coords.shape}.")
    coords = coords.astype(np.float64, copy=False)

    if spatial_order_t == "xy":
        x_in = coords[:, 0]
        y_in = coords[:, 1]
    else:
        x_in = coords[:, 1]
        y_in = coords[:, 0]

    if direction_t == "moving-to-fixed":
        in_spacing_um = sample_voxel_um
        out_spacing_um = atlas_voxel_um
        transformlist = inv_list
        reverse_transformlist = fwd_list

        if input_space_t == "crop":
            x_in_crop = x_in
            y_in_crop = y_in
        elif input_space_t == "full":
            x_in_crop = x_in - float(sample_crop_offset_xy[0])
            y_in_crop = y_in - float(sample_crop_offset_xy[1])
        elif input_space_t == "fused":
            if input_units_t != "px":
                raise click.BadParameter("input_units must be 'px' when input_space='fused'.", param_hint="input_units")
            fused_zarr = ws.stitch(roi_resolved, stitch_codebook) / "fused.zarr"
            if not fused_zarr.exists():
                raise FileNotFoundError(f"Missing fused.zarr at {fused_zarr}")
            arr = zarr.open(str(fused_zarr), mode="r")
            fused_shape_yx = (int(arr.shape[1]), int(arr.shape[2]))
            x_in_crop, y_in_crop, _ = fused_xy_to_rotated_crop_xy(
                x_fused=x_in,
                y_fused=y_in,
                fused_shape_yx=fused_shape_yx,
                prior_flip_x=bool(p1.prior_flip_x),
                prior_rotation_deg=float(p1.prior_rotation_deg),
                rotated_crop_bbox=(int(sr0), int(sr1), int(sc0), int(sc1)),
            )
        else:
            raise click.BadParameter(f"Unsupported input_space={input_space_t!r}.", param_hint="input_space")

        out_offset_x, out_offset_y = atlas_crop_offset_xy if output_space_t == "full" else (0.0, 0.0)
    else:
        in_spacing_um = atlas_voxel_um
        out_spacing_um = sample_voxel_um
        transformlist = fwd_list
        reverse_transformlist = inv_list

        if input_space_t == "crop":
            x_in_crop = x_in
            y_in_crop = y_in
        elif input_space_t == "full":
            x_in_crop = x_in - float(atlas_crop_offset_xy[0])
            y_in_crop = y_in - float(atlas_crop_offset_xy[1])
        elif input_space_t == "fused":
            raise click.BadParameter("input_space='fused' is only valid for direction='moving-to-fixed'.")
        else:
            raise click.BadParameter(f"Unsupported input_space={input_space_t!r}.", param_hint="input_space")

        out_offset_x, out_offset_y = sample_crop_offset_xy if output_space_t == "full" else (0.0, 0.0)

    x_in_mm, y_in_mm = _xy_to_mm(x=x_in_crop, y=y_in_crop, units=input_units_t, spacing_um=in_spacing_um)
    x_out_mm, y_out_mm = _apply_transforms_to_points_mm(x_mm=x_in_mm, y_mm=y_in_mm, transformlist=transformlist)
    x_out, y_out = _xy_from_mm(x_mm=x_out_mm, y_mm=y_out_mm, units=output_units_t, spacing_um=out_spacing_um)
    x_out = x_out + float(out_offset_x)
    y_out = y_out + float(out_offset_y)

    # ---- Write h5ad ----
    if keep_input:
        coords_f32 = coords.astype(np.float32, copy=False)
        adata.obsm["spatial_ccf_in"] = coords_f32
        if out_key != "spatial_ccf":
            adata.obsm[f"{out_key}_in"] = coords_f32

    out_coords = np.stack([x_out, y_out], axis=1).astype(np.float32, copy=False)
    if spatial_order_t == "yx":
        out_coords = out_coords[:, ::-1]
    adata.obsm["spatial_ccf"] = out_coords
    if out_key != "spatial_ccf":
        adata.obsm[out_key] = out_coords
    adata.write_h5ad(output_h5ad)

    # ---- Metrics ----
    metrics: dict[str, object] = {
        "workspace": str(workspace),
        "roi": str(roi_resolved),
        "summary_json": str(summary_path),
        "input_h5ad": str(input_h5ad_resolved),
        "output_h5ad": str(output_h5ad),
        "in_key": in_key,
        "out_key": out_key,
        "direction": direction_t,
        "input_space": input_space_t,
        "output_space": output_space_t,
        "input_units": input_units_t,
        "output_units": output_units_t,
        "spatial_order": spatial_order_t,
        "n_points": int(coords.shape[0]),
    }

    crop_h = int(sr1 - sr0)
    crop_w = int(sc1 - sc0)
    in_crop_bounds = (x_in_crop >= 0) & (x_in_crop < float(crop_w)) & (y_in_crop >= 0) & (y_in_crop < float(crop_h))
    metrics["input_frac_in_moving_crop"] = float(in_crop_bounds.mean())

    fixed_pts_mm = np.array(
        [(x * atlas_voxel_um * UM_TO_MM, y * atlas_voxel_um * UM_TO_MM) for x, y in p1.fixed_points_cropped_xy],
        dtype=np.float64,
    )
    moving_pts_mm = np.array(
        [
            (x * sample_voxel_um * UM_TO_MM, y * sample_voxel_um * UM_TO_MM)
            for x, y in p1.moving_points_fullres_xy_in_rotated_crop
        ],
        dtype=np.float64,
    )
    if fixed_pts_mm.shape != moving_pts_mm.shape:
        raise ValueError(f"Landmark shape mismatch: fixed={fixed_pts_mm.shape}, moving={moving_pts_mm.shape}.")

    if direction_t == "moving-to-fixed":
        pred_x_mm, pred_y_mm = _apply_transforms_to_points_mm(
            x_mm=moving_pts_mm[:, 0], y_mm=moving_pts_mm[:, 1], transformlist=inv_list
        )
        lm_err = _rmse_mm(
            pred_x_mm=pred_x_mm,
            pred_y_mm=pred_y_mm,
            true_x_mm=fixed_pts_mm[:, 0],
            true_y_mm=fixed_pts_mm[:, 1],
        )
        metrics["landmark_rmse_um"] = float(lm_err["rmse_mm"] / UM_TO_MM)
        metrics["landmark_error_mm_recomputed"] = lm_err
        if "landmark_error_mm" in summary and isinstance(summary["landmark_error_mm"], dict):
            metrics["landmark_error_mm_from_summary"] = summary["landmark_error_mm"]
            if "rmse_mm" in summary["landmark_error_mm"]:
                rmse_delta_mm = float(lm_err["rmse_mm"]) - float(summary["landmark_error_mm"]["rmse_mm"])
                metrics["landmark_rmse_delta_um_vs_summary"] = float(rmse_delta_mm / UM_TO_MM)
    else:
        pred_x_mm, pred_y_mm = _apply_transforms_to_points_mm(
            x_mm=fixed_pts_mm[:, 0], y_mm=fixed_pts_mm[:, 1], transformlist=fwd_list
        )
        lm_err = _rmse_mm(
            pred_x_mm=pred_x_mm,
            pred_y_mm=pred_y_mm,
            true_x_mm=moving_pts_mm[:, 0],
            true_y_mm=moving_pts_mm[:, 1],
        )
        metrics["landmark_rmse_um"] = float(lm_err["rmse_mm"] / UM_TO_MM)
        metrics["landmark_error_mm_recomputed"] = lm_err

    # Round-trip metric on a subset (helps catch direction/unit mistakes).
    rng = np.random.default_rng(int(random_seed))
    n = int(coords.shape[0])
    n_rt = min(int(metrics_max_points), n)
    idx_rt = rng.choice(n, size=n_rt, replace=False) if n_rt and n_rt < n else np.arange(n, dtype=np.int64)
    rt_x_mm, rt_y_mm = _apply_transforms_to_points_mm(
        x_mm=x_out_mm[idx_rt],
        y_mm=y_out_mm[idx_rt],
        transformlist=reverse_transformlist,
    )
    rt_err = _rmse_mm(
        pred_x_mm=rt_x_mm,
        pred_y_mm=rt_y_mm,
        true_x_mm=x_in_mm[idx_rt],
        true_y_mm=y_in_mm[idx_rt],
    )
    metrics["roundtrip_rmse_um"] = float(rt_err["rmse_mm"] / UM_TO_MM)
    metrics["roundtrip_error_mm"] = rt_err
    metrics["roundtrip_n_points"] = int(idx_rt.size)

    # Mask containment metric (moving->fixed, evaluated in fixed crop pixel coords).
    if direction_t == "moving-to-fixed":
        fixed_mask_nifti = run_dir / "fixed_mask_crop_orig.nii.gz"
        overlap_nifti = run_dir / "overlap_mask_final.nii.gz"
        paths = summary.get("paths", {})
        if isinstance(paths, dict) and "fixed_mask_orig_nifti" in paths:
            fixed_mask_nifti = Path(str(paths["fixed_mask_orig_nifti"]))
        if isinstance(paths, dict) and "overlap_mask_final_nifti" in paths:
            overlap_nifti = Path(str(paths["overlap_mask_final_nifti"]))

        # Evaluate all points (cheap).
        x_out_px_crop, y_out_px_crop = _xy_from_mm(x_mm=x_out_mm, y_mm=y_out_mm, units="px", spacing_um=atlas_voxel_um)

        if fixed_mask_nifti.exists():
            m = sitk.ReadImage(str(fixed_mask_nifti))
            fixed_mask_arr = sitk.GetArrayFromImage(m) > 0
            metrics["fixed_mask_metrics"] = _mask_metrics(
                x_px=x_out_px_crop,
                y_px=y_out_px_crop,
                mask_yx=fixed_mask_arr,
                spacing_um=atlas_voxel_um,
            )
            metrics["fixed_mask_nifti"] = str(fixed_mask_nifti)

        if overlap_nifti.exists():
            m = sitk.ReadImage(str(overlap_nifti))
            overlap_arr = sitk.GetArrayFromImage(m) > 0
            metrics["overlap_mask_metrics"] = _mask_metrics(
                x_px=x_out_px_crop,
                y_px=y_out_px_crop,
                mask_yx=overlap_arr,
                spacing_um=atlas_voxel_um,
            )
            metrics["overlap_mask_nifti"] = str(overlap_nifti)

    metrics_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # ---- QC figure (2x2) ----
    ds = int(thumbnail_downsample)
    if ds <= 0:
        raise click.BadParameter(f"thumbnail_downsample must be > 0, got {ds}.")

    fused_zarr = ws.stitch(roi_resolved, stitch_codebook) / "fused.zarr"
    if not fused_zarr.exists():
        raise FileNotFoundError(f"Missing fused.zarr at {fused_zarr}")
    arr = zarr.open(str(fused_zarr), mode="r")
    keys = list(arr.attrs.get("key", []))
    channel = p1.sample_channel or stitch_codebook
    ch_idx = keys.index(channel) if channel in keys else 0
    z_idx = int(p1.sample_z_idx) if p1.sample_z_idx is not None else 5

    moving_ds = np.asarray(arr[z_idx, ::ds, ::ds, ch_idx], dtype=np.float32)
    if p1.prior_flip_x:
        moving_ds = moving_ds[:, ::-1]
    if p1.prior_rotation_deg != 0:
        moving_ds = ndimage_rotate(moving_ds, p1.prior_rotation_deg, reshape=True, order=1)

    # Crop bbox is specified in the rotated full-res grid; scale down to match ds thumbnail.
    sr0_ds = int(np.floor(float(sr0) / float(ds)))
    sr1_ds = int(np.ceil(float(sr1) / float(ds)))
    sc0_ds = int(np.floor(float(sc0) / float(ds)))
    sc1_ds = int(np.ceil(float(sc1) / float(ds)))
    sr0_ds = max(0, sr0_ds)
    sc0_ds = max(0, sc0_ds)
    sr1_ds = min(int(moving_ds.shape[0]), sr1_ds)
    sc1_ds = min(int(moving_ds.shape[1]), sc1_ds)
    if sr1_ds <= sr0_ds or sc1_ds <= sc0_ds:
        raise ValueError(
            f"Invalid scaled crop bbox: {(sr0_ds, sr1_ds, sc0_ds, sc1_ds)} for thumbnail shape {moving_ds.shape}."
        )
    moving_crop_thumb = moving_ds[sr0_ds:sr1_ds, sc0_ds:sc1_ds]
    moving_crop_thumb = normalize_robust(moving_crop_thumb)

    fixed_img_path = run_dir / "fixed_atlas_crop.nii.gz"
    paths = summary.get("paths", {})
    if isinstance(paths, dict) and "fixed_nifti" in paths:
        fixed_img_path = Path(str(paths["fixed_nifti"]))
    fixed_ants = ants.image_read(str(fixed_img_path))
    moving_thumb_ants = _ants_from_numpy_yx(arr_yx=moving_crop_thumb, spacing_um=sample_voxel_um * float(ds))
    warped_thumb = ants.apply_transforms(
        fixed=fixed_ants,
        moving=moving_thumb_ants,
        transformlist=fwd_list,
        interpolator="linear",
    )
    warped_thumb_yx = normalize_robust(_ants_numpy_yx(warped_thumb))

    n_plot = int(coords.shape[0])
    k_plot = min(int(max_points_plot), n_plot)
    idx_plot = rng.choice(n_plot, size=k_plot, replace=False) if k_plot and k_plot < n_plot else np.arange(n_plot)

    # Plot coords in the same downsampled+flip+rotate grid used for `moving_crop_thumb`.
    # `moving_crop_thumb` is computed by:
    #   (1) downsample fused (stride ::ds)
    #   (2) optional flip-x in the downsampled grid
    #   (3) scipy.ndimage.rotate(..., reshape=True) in the downsampled grid
    #   (4) crop with bbox scaled into ds grid
    #
    # So for overlays, we must transform points in that same order.
    fused_shape_yx = (int(arr.shape[1]), int(arr.shape[2]))
    if direction_t == "moving-to-fixed" and input_space_t in {"fused", "full", "crop"} and input_units_t == "px":
        if input_space_t == "fused":
            x_fused = x_in
            y_fused = y_in
        elif input_space_t == "full":
            y_pose, x_pose, _ = ndimage_rotate_output_to_input_yx(
                y_out=y_in,
                x_out=x_in,
                in_shape_yx=fused_shape_yx,
                angle_deg=float(p1.prior_rotation_deg),
            )
            x_fused = (float(fused_shape_yx[1]) - 1.0) - x_pose if bool(p1.prior_flip_x) else x_pose
            y_fused = y_pose
        else:  # crop
            x_full = x_in + float(sample_crop_offset_xy[0])
            y_full = y_in + float(sample_crop_offset_xy[1])
            y_pose, x_pose, _ = ndimage_rotate_output_to_input_yx(
                y_out=y_full,
                x_out=x_full,
                in_shape_yx=fused_shape_yx,
                angle_deg=float(p1.prior_rotation_deg),
            )
            x_fused = (float(fused_shape_yx[1]) - 1.0) - x_pose if bool(p1.prior_flip_x) else x_pose
            y_fused = y_pose

        h_ds, w_ds = (int(arr[z_idx, ::ds, ::ds, ch_idx].shape[0]), int(arr[z_idx, ::ds, ::ds, ch_idx].shape[1]))
        x_ds = x_fused[idx_plot] / float(ds)
        y_ds = y_fused[idx_plot] / float(ds)
        if p1.prior_flip_x:
            x_ds = (float(w_ds) - 1.0) - x_ds
        if float(p1.prior_rotation_deg) != 0.0:
            y_full_ds, x_full_ds, _ = ndimage_rotate_input_to_output_yx(
                y_in=y_ds,
                x_in=x_ds,
                in_shape_yx=(h_ds, w_ds),
                angle_deg=float(p1.prior_rotation_deg),
            )
        else:
            x_full_ds = x_ds
            y_full_ds = y_ds
        x_in_plot = x_full_ds - float(sc0_ds)
        y_in_plot = y_full_ds - float(sr0_ds)
    else:
        # Fallback: best-effort in crop frame.
        x_in_plot = x_in_crop[idx_plot] / float(ds)
        y_in_plot = y_in_crop[idx_plot] / float(ds)

    x_out_px_crop, y_out_px_crop = _xy_from_mm(
        x_mm=x_out_mm[idx_plot], y_mm=y_out_mm[idx_plot], units="px", spacing_um=atlas_voxel_um
    )

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    ax = axes[0, 0]
    ax.imshow(moving_crop_thumb, cmap="gray", interpolation="nearest")
    ax.set_title(f"Fused thumbnail (moving crop, ds={ds})")
    ax.axis("off")

    ax = axes[0, 1]
    ax.imshow(warped_thumb_yx, cmap="gray", interpolation="nearest")
    ax.set_title("Fused thumbnail warped (fixed crop)")
    ax.axis("off")

    ax = axes[1, 0]
    ax.scatter(x_in_plot, y_in_plot, s=0.25, alpha=0.25, linewidths=0, color="tab:blue", rasterized=True)
    ax.set_title("Point cloud (moving crop)")
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xlim(0, moving_crop_thumb.shape[1])
    ax.set_ylim(moving_crop_thumb.shape[0], 0)

    ax = axes[1, 1]
    ax.scatter(x_out_px_crop, y_out_px_crop, s=0.6, alpha=0.25, linewidths=0, color="tab:green", rasterized=True)
    ax.set_title("Point cloud warped (fixed crop)")
    ax.set_aspect("equal")
    ax.invert_yaxis()

    # fixed crop shape in yx = (y,x)
    fixed_yx = _ants_numpy_yx(fixed_ants)
    ax.set_xlim(0, fixed_yx.shape[1])
    ax.set_ylim(fixed_yx.shape[0], 0)

    lm_rmse = float(metrics.get("landmark_rmse_um", float("nan")))
    rt_rmse = float(metrics.get("roundtrip_rmse_um", float("nan")))
    inside_pct = None
    fixed_mask_metrics = metrics.get("fixed_mask_metrics")
    if isinstance(fixed_mask_metrics, dict) and "frac_inside_mask" in fixed_mask_metrics:
        inside_pct = float(fixed_mask_metrics["frac_inside_mask"]) * 100.0
    title = f"{roi_resolved} | lm_rmse={lm_rmse:.2f} µm | rt_rmse={rt_rmse:.2f} µm"
    if inside_pct is not None:
        title += f" | inside_fixed={inside_pct:.1f}%"
    fig.suptitle(title)

    plt.tight_layout()
    plt.savefig(plot_png, dpi=200)
    plt.close(fig)

    click.echo(f"Wrote h5ad: {output_h5ad}")
    click.echo(f"Wrote metrics: {metrics_json}")
    click.echo(f"Wrote plot: {plot_png}")


if __name__ == "__main__":
    main()
