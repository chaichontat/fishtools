from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import click
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk

from fishtools.ccf.landmark import LandmarkRegistrationOutputs
from fishtools.ccf.ndimage_geometry import fused_xy_to_rotated_crop_xy
from fishtools.io.workspace import Workspace


def _normalize_robust(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    finite = np.isfinite(x)
    if not finite.any():
        return np.zeros_like(x)
    lo, hi = np.quantile(x[finite], [0.01, 0.99])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.nanmin(x))
        hi = float(np.nanmax(x))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return np.zeros_like(x)
    y = (x - lo) / (hi - lo)
    return np.clip(y, 0.0, 1.0)


def _roi_mask(adata: ad.AnnData, *, roi: str, roi_col: str) -> np.ndarray:
    if roi_col not in adata.obs.columns:
        raise click.ClickException(f"Missing obs column {roi_col!r} in {roi_col}.")
    return (adata.obs[roi_col].astype(str) == str(roi)).to_numpy()


def _subset_and_align(
    *,
    input_h5ad: Path,
    warped_h5ad: Path,
    roi: str,
    roi_col: str,
    input_key: str,
    warped_key: str,
    spatial_order: str,
) -> tuple[np.ndarray, np.ndarray]:
    a_in = ad.read_h5ad(input_h5ad)
    a_w = ad.read_h5ad(warped_h5ad)

    mask_in = _roi_mask(a_in, roi=roi, roi_col=roi_col)
    mask_w = _roi_mask(a_w, roi=roi, roi_col=roi_col)
    a_in = a_in[mask_in]
    a_w = a_w[mask_w]
    if a_in.n_obs == 0:
        raise click.ClickException(f"No observations left after filtering roi={roi} in {input_h5ad}")
    if a_w.n_obs == 0:
        raise click.ClickException(f"No observations left after filtering roi={roi} in {warped_h5ad}")

    if input_key not in a_in.obsm:
        raise click.ClickException(f"Missing obsm[{input_key!r}] in {input_h5ad}")
    if warped_key not in a_w.obsm:
        raise click.ClickException(f"Missing obsm[{warped_key!r}] in {warped_h5ad}")

    # Align by obs_names if needed.
    if not np.array_equal(a_in.obs_names.to_numpy(), a_w.obs_names.to_numpy()):
        common = a_in.obs_names.intersection(a_w.obs_names)
        if common.empty:
            raise click.ClickException("input_h5ad and warped_h5ad have no overlapping obs_names after ROI filter.")
        a_in = a_in[common]
        a_w = a_w[common]
        a_in = a_in[common]  # keep stable order
        a_w = a_w[common]

    coords_in = np.asarray(a_in.obsm[input_key], dtype=np.float64)
    coords_w = np.asarray(a_w.obsm[warped_key], dtype=np.float64)
    if coords_in.ndim != 2 or coords_in.shape[1] != 2:
        raise click.ClickException(f"Expected obsm[{input_key!r}] to have shape (N,2), got {coords_in.shape}")
    if coords_w.ndim != 2 or coords_w.shape[1] != 2:
        raise click.ClickException(f"Expected obsm[{warped_key!r}] to have shape (N,2), got {coords_w.shape}")

    order = str(spatial_order).lower()
    if order not in {"xy", "yx"}:
        raise click.ClickException("--spatial-order must be xy or yx.")
    if order == "yx":
        coords_in = coords_in[:, ::-1]
        coords_w = coords_w[:, ::-1]

    return coords_in, coords_w


@click.command()
@click.argument("workspace", type=click.Path(exists=True, file_okay=False, resolve_path=True, path_type=Path))
@click.argument("roi", type=str)
@click.argument("input_h5ad", type=click.Path(exists=True, dir_okay=False, resolve_path=True, path_type=Path))
@click.argument("warped_h5ad", type=click.Path(exists=True, dir_okay=False, resolve_path=True, path_type=Path))
@click.argument("output_png", type=click.Path(exists=False, dir_okay=False, resolve_path=True, path_type=Path))
@click.option(
    "--run-dirname",
    default="landmark_syn_mi",
    show_default=True,
    help="Subdirectory under <workspace>/analysis/output/ccf-transforms/<roi>/ containing similarity_plus_syn_summary.json.",
)
@click.option("--roi-col", default="roi", show_default=True, help="obs column name for ROI filtering.")
@click.option("--input-key", default="spatial", show_default=True, help="obsm key for input coords.")
@click.option("--warped-key", default="spatial_ccf", show_default=True, help="obsm key for warped coords.")
@click.option(
    "--input-space",
    type=click.Choice(["fused", "full", "crop"], case_sensitive=False),
    default="fused",
    show_default=True,
    help=(
        "Coordinate frame for input coords. "
        "'fused' = unrotated fused.zarr pixel space; "
        "'full' = rotated full-res slice pixel space; "
        "'crop' = rotated crop-local pixel space (moving_sample_crop)."
    ),
)
@click.option(
    "--stitch-codebook",
    default="pi",
    show_default=True,
    help="Stitch codebook name used for fused.zarr (analysis/deconv/stitch--{ROI}+{CODEBOOK}/fused.zarr).",
)
@click.option(
    "--spatial-order",
    type=click.Choice(["xy", "yx"], case_sensitive=False),
    default="xy",
    show_default=True,
    help="Order of columns in obsm coord arrays.",
)
@click.option(
    "--moving-downsample",
    type=int,
    default=64,
    show_default=True,
    help="Downsample factor for the (large) moving_sample_crop image panel.",
)
@click.option("--max-points", type=int, default=200_000, show_default=True, help="Max points to plot.")
@click.option("--random-seed", type=int, default=0, show_default=True)
def main(
    workspace: Path,
    roi: str,
    input_h5ad: Path,
    warped_h5ad: Path,
    output_png: Path,
    run_dirname: str,
    roi_col: str,
    input_key: str,
    warped_key: str,
    input_space: str,
    stitch_codebook: str,
    spatial_order: str,
    moving_downsample: int,
    max_points: int,
    random_seed: int,
) -> None:
    matplotlib.use("Agg")

    ws = Workspace(workspace)
    roi_resolved = str(ws.resolve_rois([str(roi)])[0])

    coords_in, coords_w = _subset_and_align(
        input_h5ad=input_h5ad,
        warped_h5ad=warped_h5ad,
        roi=roi_resolved,
        roi_col=roi_col,
        input_key=input_key,
        warped_key=warped_key,
        spatial_order=spatial_order,
    )

    # Map input coords into moving_sample_crop pixel coordinates if needed.
    out_root = ws.ccf_transforms(roi_resolved)
    p1 = LandmarkRegistrationOutputs(out_root).read_p1_landmarks()
    sr0, sr1, sc0, sc1 = p1.sample_rotated_crop_bbox

    input_space_t = str(input_space).lower()
    if input_space_t == "crop":
        coords_in_crop = coords_in
    elif input_space_t == "full":
        coords_in_crop = coords_in.copy()
        coords_in_crop[:, 0] = coords_in_crop[:, 0] - float(sc0)
        coords_in_crop[:, 1] = coords_in_crop[:, 1] - float(sr0)
    elif input_space_t == "fused":
        fused_zarr = ws.stitch(roi_resolved, stitch_codebook) / "fused.zarr"
        if not fused_zarr.exists():
            raise click.ClickException(f"Missing fused.zarr at {fused_zarr}")
        import zarr  # local import to keep startup fast

        arr = zarr.open(str(fused_zarr), mode="r")
        fused_shape_yx = (int(arr.shape[1]), int(arr.shape[2]))
        x_crop, y_crop, _ = fused_xy_to_rotated_crop_xy(
            x_fused=coords_in[:, 0],
            y_fused=coords_in[:, 1],
            fused_shape_yx=fused_shape_yx,
            prior_flip_x=bool(p1.prior_flip_x),
            prior_rotation_deg=float(p1.prior_rotation_deg),
            rotated_crop_bbox=(int(sr0), int(sr1), int(sc0), int(sc1)),
        )
        coords_in_crop = np.stack([x_crop, y_crop], axis=1)
    else:
        raise click.ClickException(f"Unsupported --input-space={input_space!r}")

    rng = np.random.default_rng(int(random_seed))
    n = int(coords_in_crop.shape[0])
    k = min(int(max_points), n)
    idx = rng.choice(n, size=k, replace=False) if k < n else np.arange(n)
    pts_in = coords_in_crop[idx]
    pts_w = coords_w[idx]

    run_dir = workspace / "analysis/output/ccf-transforms" / str(roi_resolved) / str(run_dirname)
    summary_path = run_dir / "similarity_plus_syn_summary.json"
    if not summary_path.exists():
        raise click.ClickException(f"Missing {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    paths = summary.get("paths")
    if not isinstance(paths, dict):
        raise click.ClickException(f"Invalid summary JSON: missing paths dict in {summary_path}")

    # Non-warped (moving) image in crop pixel coordinates.
    moving_path = Path(str(paths["moving_nifti"]))
    moving = np.asarray(sitk.GetArrayFromImage(sitk.ReadImage(str(moving_path))), dtype=np.float32)
    if moving.ndim != 2:
        raise click.ClickException(f"Expected 2D moving image at {moving_path}, got shape={moving.shape}")

    ds = int(moving_downsample)
    if ds <= 0:
        raise click.ClickException("--moving-downsample must be > 0")
    moving_thumb = _normalize_robust(moving[::ds, ::ds])

    # Zoom to the point cloud extent in moving space (crop coords).
    x0 = int(np.floor(float(np.min(pts_in[:, 0] / ds))))
    x1 = int(np.ceil(float(np.max(pts_in[:, 0] / ds))))
    y0 = int(np.floor(float(np.min(pts_in[:, 1] / ds))))
    y1 = int(np.ceil(float(np.max(pts_in[:, 1] / ds))))
    pad = 10
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(int(moving_thumb.shape[1]), x1 + pad)
    y1 = min(int(moving_thumb.shape[0]), y1 + pad)
    if x1 <= x0 or y1 <= y0:
        raise click.ClickException("Invalid bbox computed from input points; check coordinate system.")

    # Warped moving & fixed for QC-style overlay (fixed crop pixel coordinates).
    fixed_path = Path(str(paths["fixed_nifti"]))
    warped_path = Path(str(paths["warped_after_nifti"]))
    overlap_path = Path(str(paths["overlap_mask_final_nifti"]))

    fixed = np.asarray(sitk.GetArrayFromImage(sitk.ReadImage(str(fixed_path))), dtype=np.float32)
    warped = np.asarray(sitk.GetArrayFromImage(sitk.ReadImage(str(warped_path))), dtype=np.float32)
    overlap = np.asarray(sitk.GetArrayFromImage(sitk.ReadImage(str(overlap_path))), dtype=np.uint8) > 0
    if fixed.shape != warped.shape:
        raise click.ClickException(f"fixed shape {fixed.shape} != warped shape {warped.shape}")
    if fixed.shape != overlap.shape:
        raise click.ClickException(f"fixed shape {fixed.shape} != overlap shape {overlap.shape}")

    fixed_n = _normalize_robust(fixed)
    warped_n = _normalize_robust(warped)
    rgb = np.zeros((fixed.shape[0], fixed.shape[1], 3), dtype=np.float32)
    rgb[..., 0] = fixed_n
    rgb[..., 2] = fixed_n
    rgb[..., 1] = warped_n

    ys, xs = np.nonzero(overlap)
    fy0, fy1, fx0, fx1 = int(ys.min()), int(ys.max() + 1), int(xs.min()), int(xs.max() + 1)
    fy0 = max(0, fy0 - 10)
    fx0 = max(0, fx0 - 10)
    fy1 = min(int(rgb.shape[0]), fy1 + 10)
    fx1 = min(int(rgb.shape[1]), fx1 + 10)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7), constrained_layout=True)

    ax = axes[0]
    ax.imshow(moving_thumb[y0:y1, x0:x1], cmap="gray", interpolation="nearest")
    ax.scatter(
        (pts_in[:, 0] / ds) - x0,
        (pts_in[:, 1] / ds) - y0,
        s=0.8,
        alpha=0.18,
        linewidths=0,
        color="tab:cyan",
        rasterized=True,
    )
    ax.set_title(f"Non-warped moving (moving_sample_crop) + points\\n(ds={ds}, roi={roi}, n={n:,})")
    ax.set_aspect("equal")
    ax.axis("off")

    ax = axes[1]
    ax.imshow(rgb[fy0:fy1, fx0:fx1], interpolation="nearest")
    ax.scatter(
        pts_w[:, 0] - fx0,
        pts_w[:, 1] - fy0,
        s=0.8,
        alpha=0.18,
        linewidths=0,
        color="cyan",
        rasterized=True,
    )
    ax.set_title("QC overlay (magenta=fixed, green=warped moving) + warped points")
    ax.set_aspect("equal")
    ax.axis("off")

    fig.savefig(output_png, dpi=200)
    plt.close(fig)
    click.echo(f"Wrote plot: {output_png}")


if __name__ == "__main__":
    main()
