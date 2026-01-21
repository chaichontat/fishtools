from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, TypedDict, cast

import ants
import numpy as np
import pandas as pd
import rich_click as click
import SimpleITK as sitk
from scipy.ndimage import distance_transform_edt

from fishtools.ccf.landmark import LandmarkRegistrationOutputs
from fishtools.ccf.sitk_utils import UM_TO_MM
from fishtools.io.workspace import Workspace
from fishtools.utils.logging import setup_cli_logging


click.rich_click.SHOW_ARGUMENTS = True
click.rich_click.GROUP_ARGUMENTS_OPTIONS = True
click.rich_click.USE_MARKDOWN = True
click.rich_click.STYLE_HELPTEXT = ""


class _SummaryPaths(TypedDict, total=False):
    fixed_mask_orig_nifti: str
    overlap_mask_final_nifti: str
    moving_mask_reg_full_nifti: str


class _Summary(TypedDict, total=False):
    fwdtransforms: list[str] | str
    invtransforms: list[str] | str
    landmark_error_mm: dict[str, float]
    paths: _SummaryPaths


Units = Literal["px", "um", "mm"]
Direction = Literal["moving-to-fixed", "fixed-to-moving"]
Space = Literal["crop", "full"]


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


def _read_pointcloud_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return pd.DataFrame(payload)
        if isinstance(payload, dict) and "points" in payload:
            points = payload["points"]
            if isinstance(points, list):
                return pd.DataFrame(points)
        raise ValueError(
            f"Unsupported JSON structure in {path}. Expected a list of records or an object with a 'points' list."
        )
    if suffix == ".npy":
        arr = np.load(path)
        if not isinstance(arr, np.ndarray) or arr.ndim != 2 or arr.shape[1] < 2:
            raise ValueError(f"Expected (N,>=2) numpy array in {path}, got shape={getattr(arr, 'shape', None)}.")
        return pd.DataFrame({"x": arr[:, 0], "y": arr[:, 1]})
    raise ValueError(f"Unsupported pointcloud file type: {path} (expected .csv/.tsv/.parquet/.json/.npy).")


def _write_pointcloud_table(df: pd.DataFrame, path: Path) -> None:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df.to_csv(path, index=False)
        return
    if suffix == ".parquet":
        df.to_parquet(path, index=False)
        return
    if suffix == ".json":
        path.write_text(df.to_json(orient="records", indent=2), encoding="utf-8")
        return
    if suffix == ".npy":
        arr = df[["x", "y"]].to_numpy(dtype=np.float64, copy=False)
        np.save(path, arr)
        return
    raise ValueError(f"Unsupported output type: {path} (expected .csv/.parquet/.json/.npy).")


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


def _rmse_mm(*, pred_x_mm: np.ndarray, pred_y_mm: np.ndarray, true_x_mm: np.ndarray, true_y_mm: np.ndarray) -> dict[str, float]:
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


@click.command("ants-warp-pointcloud")
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
    "points",
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
    "output",
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
    help="Direction to map points. moving-to-fixed uses inverse transforms (per ANTs point-mapping convention).",
)
@click.option(
    "--input-space",
    type=click.Choice(["crop", "full"], case_sensitive=False),
    default="crop",
    show_default=True,
    help="Whether input point coordinates are relative to the crop bbox (crop) or the full slice (full).",
)
@click.option(
    "--output-space",
    type=click.Choice(["crop", "full"], case_sensitive=False),
    default="crop",
    show_default=True,
    help="Whether output point coordinates are written relative to the crop bbox (crop) or the full slice (full).",
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
@click.option("--x-col", default="x", show_default=True, help="X column name in the input table.")
@click.option("--y-col", default="y", show_default=True, help="Y column name in the input table.")
@click.option(
    "--keep-input/--no-keep-input",
    default=False,
    show_default=True,
    help="If set, copy original x/y to x_in/y_in before overwriting.",
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
    help="Optional path to write a JSON metrics report (useful for iterating on registrations).",
)
@click.option(
    "--debug/--no-debug",
    default=False,
    show_default=True,
    help="Enable verbose logging to <workspace>/analysis/logs/.",
)
def main(  # noqa: PLR0913
    workspace: Path,
    roi: str,
    points: Path,
    output: Path,
    *,
    run_dirname: str,
    direction: str,
    input_space: str,
    output_space: str,
    input_units: str,
    output_units: str,
    x_col: str,
    y_col: str,
    keep_input: bool,
    metrics_json: Path | None,
    debug: bool,
) -> None:
    ws = Workspace(workspace)
    try:
        (roi_resolved,) = tuple(ws.resolve_rois((roi,)))
    except ValueError as exc:
        raise click.BadParameter(str(exc), param_hint="roi") from exc

    setup_cli_logging(
        workspace,
        component="ccf.ants_warp_pointcloud",
        file=f"ants-warp-pointcloud-{roi_resolved}",
        debug=debug,
        extra={"roi": str(roi_resolved)},
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

    direction_t = cast(Direction, str(direction).lower())
    input_space_t = cast(Space, str(input_space).lower())
    output_space_t = cast(Space, str(output_space).lower())
    input_units_t = cast(Units, str(input_units).lower())
    output_units_t = cast(Units, str(output_units).lower())

    out_contract = LandmarkRegistrationOutputs(out_root)
    p1 = out_contract.read_p1_landmarks()

    atlas_voxel_um = float(p1.atlas_voxel_um) if p1.atlas_voxel_um is not None else 20.0
    sample_voxel_um = float(p1.sample_voxel_xy_um) if p1.sample_voxel_xy_um is not None else 0.216

    ar0, _, ac0, _ = p1.atlas_crop_bbox
    sr0, _, sc0, _ = p1.sample_rotated_crop_bbox
    atlas_crop_offset_xy = (float(ac0), float(ar0))
    sample_crop_offset_xy = (float(sc0), float(sr0))

    # ---- Load point cloud ----
    df_in = _read_pointcloud_table(points)
    if x_col not in df_in.columns:
        raise click.BadParameter(f"Missing x column {x_col!r} in {points}.")
    if y_col not in df_in.columns:
        raise click.BadParameter(f"Missing y column {y_col!r} in {points}.")

    x_in = df_in[x_col].to_numpy(dtype=np.float64, copy=False)
    y_in = df_in[y_col].to_numpy(dtype=np.float64, copy=False)

    # Normalize to crop coordinates for the registration domain.
    if direction_t == "moving-to-fixed":
        in_offset_x, in_offset_y = sample_crop_offset_xy if input_space_t == "full" else (0.0, 0.0)
        out_offset_x, out_offset_y = atlas_crop_offset_xy if output_space_t == "full" else (0.0, 0.0)
        in_spacing_um = sample_voxel_um
        out_spacing_um = atlas_voxel_um
        transformlist = inv_list  # moving->fixed point mapping
        reverse_transformlist = fwd_list
    else:
        in_offset_x, in_offset_y = atlas_crop_offset_xy if input_space_t == "full" else (0.0, 0.0)
        out_offset_x, out_offset_y = sample_crop_offset_xy if output_space_t == "full" else (0.0, 0.0)
        in_spacing_um = atlas_voxel_um
        out_spacing_um = sample_voxel_um
        transformlist = fwd_list  # fixed->moving point mapping
        reverse_transformlist = inv_list

    x_in_crop = x_in - float(in_offset_x)
    y_in_crop = y_in - float(in_offset_y)

    x_in_mm, y_in_mm = _xy_to_mm(x=x_in_crop, y=y_in_crop, units=input_units_t, spacing_um=in_spacing_um)
    x_out_mm, y_out_mm = _apply_transforms_to_points_mm(x_mm=x_in_mm, y_mm=y_in_mm, transformlist=transformlist)
    x_out, y_out = _xy_from_mm(x_mm=x_out_mm, y_mm=y_out_mm, units=output_units_t, spacing_um=out_spacing_um)
    x_out = x_out + float(out_offset_x)
    y_out = y_out + float(out_offset_y)

    df_out = df_in.copy()
    if keep_input:
        df_out["x_in"] = x_in
        df_out["y_in"] = y_in
    df_out[x_col] = x_out
    df_out[y_col] = y_out

    _write_pointcloud_table(df_out, output)

    # ---- Metrics ----
    metrics: dict[str, object] = {
        "workspace": str(workspace),
        "roi": str(roi_resolved),
        "summary_json": str(summary_path),
        "points_in": str(points),
        "points_out": str(output),
        "direction": direction_t,
        "input_space": input_space_t,
        "output_space": output_space_t,
        "input_units": input_units_t,
        "output_units": output_units_t,
        "n_points": int(len(df_in)),
    }

    # Landmark metric (verifiable against the run that produced the summary).
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
        if "landmark_error_mm" in summary:
            metrics["landmark_error_mm_from_summary"] = summary["landmark_error_mm"]
            if isinstance(summary["landmark_error_mm"], dict) and "rmse_mm" in summary["landmark_error_mm"]:
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

    # Round-trip consistency metric on the input points (helps catch direction/unit mistakes).
    rt_x_mm, rt_y_mm = _apply_transforms_to_points_mm(
        x_mm=x_out_mm,
        y_mm=y_out_mm,
        transformlist=reverse_transformlist,
    )
    rt_err = _rmse_mm(pred_x_mm=rt_x_mm, pred_y_mm=rt_y_mm, true_x_mm=x_in_mm, true_y_mm=y_in_mm)
    metrics["roundtrip_rmse_um"] = float(rt_err["rmse_mm"] / UM_TO_MM)
    metrics["roundtrip_error_mm"] = rt_err

    # Mask containment metric (only meaningful for moving->fixed, output evaluated in fixed crop grid).
    if direction_t == "moving-to-fixed":
        paths = summary.get("paths", {})
        fixed_mask_path = None
        overlap_mask_path = None
        if isinstance(paths, dict):
            fixed_mask_path = paths.get("fixed_mask_orig_nifti")
            overlap_mask_path = paths.get("overlap_mask_final_nifti")

        fixed_mask_nifti = Path(str(fixed_mask_path)) if fixed_mask_path else (run_dir / "fixed_mask_crop_orig.nii.gz")
        if fixed_mask_nifti.exists():
            mask = sitk.ReadImage(str(fixed_mask_nifti))
            mask_arr = sitk.GetArrayFromImage(mask) > 0
            # Evaluate in fixed-crop pixel coordinates regardless of requested output_space/units.
            x_out_px_crop, y_out_px_crop = _xy_from_mm(
                x_mm=x_out_mm, y_mm=y_out_mm, units="px", spacing_um=atlas_voxel_um
            )
            metrics["fixed_mask_metrics"] = _mask_metrics(
                x_px=x_out_px_crop,
                y_px=y_out_px_crop,
                mask_yx=mask_arr,
                spacing_um=atlas_voxel_um,
            )
            metrics["fixed_mask_nifti"] = str(fixed_mask_nifti)

        overlap_nifti = Path(str(overlap_mask_path)) if overlap_mask_path else (run_dir / "overlap_mask_final.nii.gz")
        if overlap_nifti.exists():
            mask = sitk.ReadImage(str(overlap_nifti))
            mask_arr = sitk.GetArrayFromImage(mask) > 0
            x_out_px_crop, y_out_px_crop = _xy_from_mm(
                x_mm=x_out_mm, y_mm=y_out_mm, units="px", spacing_um=atlas_voxel_um
            )
            metrics["overlap_mask_metrics"] = _mask_metrics(
                x_px=x_out_px_crop,
                y_px=y_out_px_crop,
                mask_yx=mask_arr,
                spacing_um=atlas_voxel_um,
            )
            metrics["overlap_mask_nifti"] = str(overlap_nifti)

    if metrics_json is not None:
        metrics_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        click.echo(f"Wrote metrics: {metrics_json}")

    click.echo(f"Wrote points: {output}")
    click.echo(
        f"Landmark RMSE: {float(metrics['landmark_rmse_um']):.2f} µm | "
        f"Roundtrip RMSE: {float(metrics['roundtrip_rmse_um']):.2f} µm"
    )
    if "landmark_rmse_delta_um_vs_summary" in metrics:
        click.echo(f"Landmark RMSE Δ vs summary: {float(metrics['landmark_rmse_delta_um_vs_summary']):.3f} µm")
    fixed_mask_metrics = metrics.get("fixed_mask_metrics")
    if isinstance(fixed_mask_metrics, dict) and "frac_inside_mask" in fixed_mask_metrics:
        click.echo(
            f"Fixed-mask inside: {float(fixed_mask_metrics['frac_inside_mask']) * 100:.1f}% | "
            f"OOB: {float(fixed_mask_metrics['frac_oob']) * 100:.1f}%"
        )
    overlap_mask_metrics = metrics.get("overlap_mask_metrics")
    if isinstance(overlap_mask_metrics, dict) and "frac_inside_mask" in overlap_mask_metrics:
        click.echo(f"Overlap-mask inside: {float(overlap_mask_metrics['frac_inside_mask']) * 100:.1f}%")


if __name__ == "__main__":
    main()
