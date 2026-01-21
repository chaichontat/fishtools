from __future__ import annotations

from pathlib import Path
from typing import cast

import anndata as ad
import click
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import zarr
from scipy.ndimage import rotate as ndimage_rotate

from fishtools.ccf.ndimage_geometry import (
    ndimage_rotate_input_to_output_yx,
    ndimage_rotate_output_to_input_yx,
    fused_xy_to_rotated_crop_xy,
    fused_xy_to_rotated_full_xy,
)
from fishtools.ccf.landmark import LandmarkRegistrationOutputs
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


@click.command()
@click.argument("workspace", type=click.Path(exists=True, file_okay=False, resolve_path=True, path_type=Path))
@click.argument("roi", type=str)
@click.argument("input_h5ad", type=click.Path(exists=True, dir_okay=False, resolve_path=True, path_type=Path))
@click.argument("output_png", type=click.Path(exists=False, dir_okay=False, resolve_path=True, path_type=Path))
@click.option("--roi-col", default="roi", show_default=True, help="obs column name for ROI filtering.")
@click.option(
    "--filter-roi/--no-filter-roi",
    default=True,
    show_default=True,
    help="Filter observations by adata.obs[roi_col] == ROI before plotting.",
)
@click.option("--in-key", default="spatial", show_default=True, help="obsm key for input coords.")
@click.option(
    "--input-space",
    type=click.Choice(["fused", "full", "crop"], case_sensitive=False),
    default="fused",
    show_default=True,
    help=(
        "Coordinate frame for obsm[in_key]. "
        "'fused' = unrotated fused.zarr pixel space; "
        "'full' = rotated full-res slice pixel space; "
        "'crop' = rotated crop-local pixel space."
    ),
)
@click.option(
    "--spatial-order",
    type=click.Choice(["xy", "yx"], case_sensitive=False),
    default="xy",
    show_default=True,
    help="Order of columns in obsm[in_key].",
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
    help="Downsample factor for fused thumbnail overlay (1 = full resolution; recommended >= 16).",
)
@click.option(
    "--channel",
    default=None,
    show_default=False,
    help="Optional fused.zarr channel name override; defaults to p1_landmarks.sample_channel if present, else 0th channel.",
)
@click.option(
    "--z-idx",
    type=int,
    default=None,
    show_default=False,
    help="Optional fused.zarr z index override; defaults to p1_landmarks.sample_z_idx if present, else 5.",
)
@click.option("--max-points", type=int, default=200_000, show_default=True, help="Max points to plot.")
@click.option("--random-seed", type=int, default=0, show_default=True)
def main(
    workspace: Path,
    roi: str,
    input_h5ad: Path,
    output_png: Path,
    roi_col: str,
    filter_roi: bool,
    in_key: str,
    input_space: str,
    spatial_order: str,
    stitch_codebook: str,
    thumbnail_downsample: int,
    channel: str | None,
    z_idx: int | None,
    max_points: int,
    random_seed: int,
) -> None:
    matplotlib.use("Agg")
    ws = Workspace(workspace)
    roi_resolved = str(ws.resolve_rois([str(roi)])[0])

    out_root = ws.ccf_transforms(roi_resolved)
    p1 = LandmarkRegistrationOutputs(out_root).read_p1_landmarks()

    if p1.sample_channel is None and channel is None:
        channel_name = None
    else:
        channel_name = str(channel) if channel is not None else str(p1.sample_channel)
    z = int(z_idx) if z_idx is not None else int(p1.sample_z_idx) if p1.sample_z_idx is not None else 5

    sr0, sr1, sc0, sc1 = p1.sample_rotated_crop_bbox
    if sr1 <= sr0 or sc1 <= sc0:
        raise click.ClickException(f"Invalid sample_rotated_crop_bbox in p1_landmarks.json: {(sr0, sr1, sc0, sc1)}")

    # ---- Load coords ----
    adata = ad.read_h5ad(input_h5ad)
    if filter_roi:
        if roi_col not in adata.obs.columns:
            raise click.BadParameter(f"Missing obs column {roi_col!r} for ROI filtering in {input_h5ad}.")
        n_before = int(adata.n_obs)
        mask = adata.obs[roi_col].astype(str) == roi_resolved
        adata = adata[mask].copy()
        if adata.n_obs == 0:
            raise click.ClickException(f"No observations left after filtering {roi_col}={roi_resolved!r} in {input_h5ad}.")
        click.echo(f"Filtered {input_h5ad} to {roi_col}={roi_resolved}: {adata.n_obs}/{n_before} obs")

    if in_key not in adata.obsm:
        raise click.ClickException(f"Missing obsm[{in_key!r}] in {input_h5ad}.")
    coords = np.asarray(adata.obsm[in_key])
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise click.ClickException(f"Expected obsm[{in_key!r}] to have shape (N,2), got {coords.shape}.")
    coords = coords.astype(np.float64, copy=False)

    order = cast(str, spatial_order).lower()
    if order == "xy":
        x_in = coords[:, 0]
        y_in = coords[:, 1]
    else:
        x_in = coords[:, 1]
        y_in = coords[:, 0]

    rng = np.random.default_rng(int(random_seed))
    n = int(x_in.size)
    k = min(int(max_points), n)
    idx = rng.choice(n, size=k, replace=False) if k < n else np.arange(n)
    x_in = x_in[idx]
    y_in = y_in[idx]

    # ---- Load fused thumbnail ----
    ds = int(thumbnail_downsample)
    if ds <= 0:
        raise click.BadParameter(f"thumbnail_downsample must be > 0, got {ds}.")

    fused_zarr = ws.stitch(roi_resolved, stitch_codebook) / "fused.zarr"
    if not fused_zarr.exists():
        raise click.ClickException(f"Missing fused.zarr at {fused_zarr}")
    arr = zarr.open(str(fused_zarr), mode="r")
    keys = list(arr.attrs.get("key", []))
    if not keys:
        raise click.ClickException(f"Missing fused.zarr attrs['key'] channels list at {fused_zarr}")

    if channel_name is None:
        ch_idx = 0
        channel_name = str(keys[ch_idx])
    else:
        if channel_name not in keys:
            raise click.ClickException(f"Channel {channel_name!r} not found in fused.zarr keys={keys}")
        ch_idx = keys.index(channel_name)

    if z < 0 or z >= int(arr.shape[0]):
        raise click.ClickException(f"z_idx={z} out of range for fused.zarr shape={arr.shape}")

    fused_shape_yx = (int(arr.shape[1]), int(arr.shape[2]))

    moving_unrot_ds = np.asarray(arr[z, ::ds, ::ds, ch_idx], dtype=np.float32)

    # Apply prior transforms to match the rotated sample frame used for landmarks/crop bbox.
    moving_ds = moving_unrot_ds
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
        raise click.ClickException(
            f"Invalid scaled crop bbox: {(sr0_ds, sr1_ds, sc0_ds, sc1_ds)} for thumbnail shape {moving_ds.shape}."
        )

    moving_unrot_thumb = _normalize_robust(moving_unrot_ds)
    moving_full_thumb = _normalize_robust(moving_ds)
    moving_crop_thumb = moving_full_thumb[sr0_ds:sr1_ds, sc0_ds:sc1_ds]

    # ---- Map coords into all frames for plotting ----
    input_space_t = cast(str, input_space).lower()
    if input_space_t == "fused":
        x_fused = x_in
        y_fused = y_in
        x_full, y_full, _ = fused_xy_to_rotated_full_xy(
            x_fused=x_fused,
            y_fused=y_fused,
            fused_shape_yx=fused_shape_yx,
            prior_flip_x=bool(p1.prior_flip_x),
            prior_rotation_deg=float(p1.prior_rotation_deg),
        )
        x_crop, y_crop, _ = fused_xy_to_rotated_crop_xy(
            x_fused=x_fused,
            y_fused=y_fused,
            fused_shape_yx=fused_shape_yx,
            prior_flip_x=bool(p1.prior_flip_x),
            prior_rotation_deg=float(p1.prior_rotation_deg),
            rotated_crop_bbox=(int(sr0), int(sr1), int(sc0), int(sc1)),
        )
    elif input_space_t == "full":
        x_full = x_in.astype(np.float64, copy=False)
        y_full = y_in.astype(np.float64, copy=False)
        x_crop = x_full - float(sc0)
        y_crop = y_full - float(sr0)
        y_pose, x_pose, _ = ndimage_rotate_output_to_input_yx(
            y_out=y_full,
            x_out=x_full,
            in_shape_yx=fused_shape_yx,
            angle_deg=float(p1.prior_rotation_deg),
        )
        if p1.prior_flip_x:
            x_fused = (float(fused_shape_yx[1]) - 1.0) - x_pose
        else:
            x_fused = x_pose
        y_fused = y_pose
    elif input_space_t == "crop":
        x_crop = x_in.astype(np.float64, copy=False)
        y_crop = y_in.astype(np.float64, copy=False)
        x_full = x_crop + float(sc0)
        y_full = y_crop + float(sr0)
        y_pose, x_pose, _ = ndimage_rotate_output_to_input_yx(
            y_out=y_full,
            x_out=x_full,
            in_shape_yx=fused_shape_yx,
            angle_deg=float(p1.prior_rotation_deg),
        )
        if p1.prior_flip_x:
            x_fused = (float(fused_shape_yx[1]) - 1.0) - x_pose
        else:
            x_fused = x_pose
        y_fused = y_pose
    else:
        raise click.ClickException(f"Unsupported input_space={input_space!r}")

    # ---- Plot ----
    # IMPORTANT: `moving_full_thumb` is computed by:
    #   (1) downsample fused (stride ::ds)
    #   (2) optional flip-x in the downsampled grid
    #   (3) scipy.ndimage.rotate(..., reshape=True) in the downsampled grid
    # So for overlays, we must transform points in the *same* order in the ds grid.
    h_ds, w_ds = (int(moving_unrot_ds.shape[0]), int(moving_unrot_ds.shape[1]))
    x_ds = x_fused / float(ds)
    y_ds = y_fused / float(ds)
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
    x_crop_ds = x_full_ds - float(sc0_ds)
    y_crop_ds = y_full_ds - float(sr0_ds)

    fig, axes = plt.subplots(1, 3, figsize=(24, 7), constrained_layout=True)

    ax = axes[0]
    ax.imshow(moving_unrot_thumb, cmap="gray", interpolation="nearest")
    ax.scatter(
        x_fused / float(ds),
        y_fused / float(ds),
        s=0.6,
        alpha=0.18,
        linewidths=0,
        color="tab:cyan",
        rasterized=True,
    )
    ax.set_title(f"Fused.zarr thumbnail (unrotated) + raw points\\n(ch={channel_name}, z={z}, ds={ds})")
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.axis("off")

    ax = axes[1]
    ax.imshow(moving_full_thumb, cmap="gray", interpolation="nearest")
    ax.scatter(
        x_full_ds,
        y_full_ds,
        s=0.6,
        alpha=0.18,
        linewidths=0,
        color="tab:cyan",
        rasterized=True,
    )
    # Crop bbox overlay on full panel
    rect = plt.Rectangle(
        (float(sc0_ds), float(sr0_ds)),
        float(sc1_ds - sc0_ds),
        float(sr1_ds - sr0_ds),
        fill=False,
        edgecolor="tab:orange",
        linewidth=1.0,
    )
    ax.add_patch(rect)
    ax.set_title(f"Fused.zarr rotated thumbnail + points (full)\\n(ch={channel_name}, z={z}, ds={ds})")
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.axis("off")

    ax = axes[2]
    ax.imshow(moving_crop_thumb, cmap="gray", interpolation="nearest")
    ax.scatter(
        x_crop_ds,
        y_crop_ds,
        s=0.6,
        alpha=0.18,
        linewidths=0,
        color="tab:cyan",
        rasterized=True,
    )
    ax.set_title(
        f"Fused.zarr rotated thumbnail + points (crop)\\n(input_space={input_space_t}, roi={roi_resolved}, n={n:,})"
    )
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.axis("off")

    fig.savefig(output_png, dpi=200)
    plt.close(fig)
    click.echo(f"Wrote plot: {output_png}")


if __name__ == "__main__":
    main()
