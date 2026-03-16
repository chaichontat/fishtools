from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import rich_click as click
import zarr
from scipy.ndimage import rotate as ndimage_rotate

from fishtools.ccf.landmark import LandmarkRegistrationOutputs
from fishtools.ccf.mask_edit_thumbnail import load_moving_crop_yxc, render_mask_edit_thumbnail_rgb01, save_mask_edit_thumbnail_png
from fishtools.io.workspace import Workspace
from fishtools.utils.logging import setup_cli_logging


click.rich_click.SHOW_ARGUMENTS = True
click.rich_click.GROUP_ARGUMENTS_OPTIONS = True
click.rich_click.USE_MARKDOWN = True
click.rich_click.STYLE_HELPTEXT = ""


def _parse_channel_indices(spec: str | None) -> tuple[int, ...] | None:
    if spec is None:
        return None
    raw = spec.strip()
    if not raw:
        return None
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        return None
    out: list[int] = []
    for p in parts:
        try:
            idx = int(p)
        except ValueError as exc:
            raise click.BadParameter("--channels must be comma-separated integer indices like '0,1,2'.") from exc
        if idx < 0:
            raise click.BadParameter("--channels indices must be >= 0.")
        if idx in out:
            raise click.BadParameter("--channels must not contain duplicates.")
        out.append(idx)
    return tuple(out)


def _read_similarity2d_angle_rad(tfm_path: Path) -> float:
    if not tfm_path.exists():
        raise FileNotFoundError(f"Missing similarity transform: {tfm_path}")
    for line in tfm_path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("Parameters:"):
            continue
        parts = line.split(":", 1)[1].strip().split()
        if len(parts) < 2:
            raise ValueError(f"Invalid Similarity2DTransform parameters line in {tfm_path}: {line!r}")
        return float(parts[1])
    raise ValueError(f"Missing 'Parameters:' line in Similarity2DTransform file {tfm_path}")


def _try_read_similarity_rotation_deg(out_contract: LandmarkRegistrationOutputs) -> float | None:
    if not out_contract.p1_similarity_tfm.exists():
        return None
    theta_rad = _read_similarity2d_angle_rad(out_contract.p1_similarity_tfm)
    return float(math.degrees(theta_rad))


def _normalize_fused_name(name: str) -> str:
    raw = str(name).strip()
    if not raw:
        raise click.BadParameter("--fused-name must not be empty.")
    # Historical naming in this repo uses underscores (fused_highpassed.zarr), but users often type hyphens.
    if raw == "fused-highpassed.zarr":
        return "fused_highpassed.zarr"
    return raw


def _fused_label_for_filename(fused_name: str) -> str:
    n = _normalize_fused_name(fused_name)
    if n == "fused.zarr":
        return "raw"
    if n in {"fused_highpassed.zarr", "fused-highpassed.zarr"}:
        return "highpass"
    stem = Path(n).stem
    cleaned = "".join(ch if ch.isalnum() else "-" for ch in stem).strip("-").lower()
    return cleaned or "custom"


def _load_moving_fullslice_yxc(
    *,
    ws: Workspace,
    roi: str,
    stitch_codebook: str,
    fused_name: str,
    z_idx: int,
    rotation_deg: float,
    flip_x: bool,
) -> np.ndarray:
    fused_zarr = ws.stitch(roi, stitch_codebook) / str(fused_name)
    if not fused_zarr.exists():
        raise FileNotFoundError(f"Missing {fused_name} at {fused_zarr}")
    arr = zarr.open(str(fused_zarr), mode="r")

    if z_idx < 0 or z_idx >= int(arr.shape[0]):
        raise ValueError(f"z_idx out of range for fused.zarr: z_idx={z_idx}, shape[0]={int(arr.shape[0])}.")

    sample_slice_full_raw = np.asarray(arr[z_idx, :, :, :], dtype=np.float32)
    if sample_slice_full_raw.ndim != 3:
        raise ValueError(f"Expected fused slice shape (y,x,c), got {sample_slice_full_raw.shape}.")

    out = sample_slice_full_raw
    if flip_x:
        out = out[:, ::-1, :]
    if rotation_deg != 0:
        rotated: list[np.ndarray] = []
        for c in range(int(out.shape[2])):
            rotated.append(ndimage_rotate(out[:, :, c], float(rotation_deg), reshape=True, order=1))
        out = np.stack(rotated, axis=2)
    return out


@click.command("export-moving-thumbnail")
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
    "--run-dirname",
    default="landmark_syn_mi",
    show_default=True,
    help="Subdirectory under <workspace>/analysis/output/ccf-transforms/<roi>/ to write mask_edit/ outputs under.",
)
@click.option(
    "--stitch-codebook",
    default="pi",
    show_default=True,
    help="Stitch codebook name used for fused.zarr (analysis/deconv/stitch--{ROI}+{CODEBOOK}/fused.zarr).",
)
@click.option(
    "--fused-name",
    default="fused.zarr",
    show_default=True,
    help="Name of the intensity Zarr inside stitch--ROI+<codebook> (e.g., fused.zarr, fused_highpassed.zarr).",
)
@click.option(
    "--z-idx",
    type=int,
    default=None,
    show_default=False,
    help="Z index in fused.zarr to export (default: p1_landmarks.json sample_z_idx, else 5).",
)
@click.option(
    "--moving-pre-downsample",
    type=int,
    default=8,
    show_default=True,
    help="Downsample factor applied to the moving crop BEFORE resampling (8 matches the mask_edit thumbnail convention).",
)
@click.option(
    "--target-spacing-um",
    type=float,
    default=2.0,
    show_default=True,
    help="Target pixel size (µm/px) for the PNG output.",
)
@click.option(
    "--channels",
    default=None,
    show_default=False,
    help="Optional comma-separated channel indices to map into RGB before resampling (default: first 3 channels).",
)
@click.option(
    "--thumbnail-png",
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
    help="Output PNG path for the moving thumbnail (default: under <run_dirname>/mask_edit/).",
)
@click.option(
    "--debug/--no-debug",
    default=False,
    show_default=True,
    help="Enable verbose logging to <workspace>/analysis/logs/.",
)
def main(
    workspace: Path,
    roi: str | None,
    *,
    run_dirname: str,
    stitch_codebook: str,
    fused_name: str,
    z_idx: int | None,
    moving_pre_downsample: int,
    target_spacing_um: float,
    channels: str | None,
    thumbnail_png: Path | None,
    debug: bool,
) -> None:
    """Export a single moving-crop thumbnail PNG in the same format as `ccf export-mask-edit-pack`."""

    ws = Workspace(workspace)
    process_all_rois = roi is None
    if roi is None and thumbnail_png is not None:
        raise click.BadParameter(
            "When ROI is omitted (process all ROIs), --thumbnail-png must be omitted so per-ROI default filenames can be used."
        )
    try:
        rois_resolved = ws.resolve_rois((roi,)) if roi is not None else ws.resolve_rois(None)
    except ValueError as exc:
        raise click.BadParameter(str(exc), param_hint="roi") from exc

    channel_indices = _parse_channel_indices(channels)
    fused_name_norm = _normalize_fused_name(fused_name)
    fused_label = _fused_label_for_filename(fused_name_norm)

    for roi_resolved in rois_resolved:
        setup_cli_logging(
            workspace,
            component="ccf.export_moving_thumbnail",
            file=f"export-moving-thumbnail-{roi_resolved}",
            debug=debug,
            extra={"roi": str(roi_resolved)},
        )
        try:
            _run_one_roi(
                ws=ws,
                roi_resolved=str(roi_resolved),
                run_dirname=str(run_dirname),
                stitch_codebook=str(stitch_codebook),
                fused_name=fused_name_norm,
                fused_label=fused_label,
                z_idx=z_idx,
                moving_pre_downsample=int(moving_pre_downsample),
                target_spacing_um=float(target_spacing_um),
                channel_indices=channel_indices,
                thumbnail_png=thumbnail_png,
            )
        except FileNotFoundError as exc:
            if process_all_rois:
                click.echo(f"[{roi_resolved}] skipping: {exc}")
                continue
            raise click.ClickException(f"ROI {roi_resolved!r}: {exc}") from exc
        except Exception as exc:
            raise click.ClickException(f"ROI {roi_resolved!r}: {exc}") from exc


def _run_one_roi(
    *,
    ws: Workspace,
    roi_resolved: str,
    run_dirname: str,
    stitch_codebook: str,
    fused_name: str,
    fused_label: str,
    z_idx: int | None,
    moving_pre_downsample: int,
    target_spacing_um: float,
    channel_indices: tuple[int, ...] | None,
    thumbnail_png: Path | None,
) -> None:
    t_start = time.perf_counter()

    click.echo(f"[{roi_resolved}] export-moving-thumbnail: start")
    out_root = ws.ccf_transforms(roi_resolved)
    out_contract = LandmarkRegistrationOutputs(out_root)
    p1 = out_contract.try_read_p1_landmarks()

    run_dir = out_root / str(run_dirname)
    out_dir = run_dir / "mask_edit"
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = int(moving_pre_downsample)
    if ds <= 0:
        raise click.BadParameter(f"moving_pre_downsample must be > 0, got {ds}.")
    target_um = float(target_spacing_um)
    if not target_um > 0:
        raise click.BadParameter(f"target_spacing_um must be > 0, got {target_um}.")

    sample_voxel_um = 0.216
    if p1 is not None and p1.sample_voxel_xy_um is not None:
        sample_voxel_um = float(p1.sample_voxel_xy_um)

    z_idx_eff: int
    if z_idx is not None:
        z_idx_eff = int(z_idx)
    elif p1 is not None and p1.sample_z_idx is not None:
        z_idx_eff = int(p1.sample_z_idx)
    else:
        # Match legacy default while avoiding out-of-range errors on shallow stacks.
        fused_zarr = ws.stitch(str(roi_resolved), str(stitch_codebook)) / str(fused_name)
        if not fused_zarr.exists():
            raise FileNotFoundError(f"Missing {fused_name} at {fused_zarr}")
        arr = zarr.open(str(fused_zarr), mode="r")
        z_idx_eff = min(5, int(arr.shape[0]) - 1)
        z_idx_eff = max(0, z_idx_eff)

    tag = f"z{z_idx_eff}_ds{ds}_target{target_um:g}um"
    if thumbnail_png is None:
        thumbnail_png = out_dir / f"moving_thumbnail_{stitch_codebook}_{fused_label}_{tag}.png"

    if p1 is not None:
        moving_crop_yxc, _ = load_moving_crop_yxc(
            ws=ws,
            roi=str(roi_resolved),
            stitch_codebook=str(stitch_codebook),
            p1=p1,
            z_idx=z_idx_eff,
            fused_name=fused_name,
        )
    else:
        rotation_deg = _try_read_similarity_rotation_deg(out_contract) or 0.0
        moving_crop_yxc = _load_moving_fullslice_yxc(
            ws=ws,
            roi=str(roi_resolved),
            stitch_codebook=str(stitch_codebook),
            fused_name=fused_name,
            z_idx=z_idx_eff,
            rotation_deg=float(rotation_deg),
            flip_x=False,
        )

    rgb, _ = render_mask_edit_thumbnail_rgb01(
        moving_crop_yxc=moving_crop_yxc,
        sample_voxel_xy_um=sample_voxel_um,
        moving_pre_downsample=ds,
        target_spacing_um=target_um,
        channel_indices=channel_indices,
    )
    save_mask_edit_thumbnail_png(path=thumbnail_png, rgb01=rgb)

    click.echo(f"Wrote: {thumbnail_png}")
    click.echo(f"[{roi_resolved}] total={time.perf_counter() - t_start:.3f}s")


if __name__ == "__main__":
    main()
