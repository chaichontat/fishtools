from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Literal, cast

import ants
import matplotlib as mpl
import numpy as np
import rich_click as click
import SimpleITK as sitk
import tifffile
import zarr
from brainglobe_atlasapi import BrainGlobeAtlas
from scipy.ndimage import rotate as ndimage_rotate

# Force a non-interactive backend to avoid GUI/event-loop hangs in headless runs
mpl.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402

from fishtools.ccf.landmark import LandmarkRegistrationOutputs, P1Landmarks
from fishtools.ccf.sitk_utils import UM_TO_MM, normalize_robust, resample_sitk_to_spacing, sitk_from_numpy_2d
from fishtools.io.workspace import Workspace
from fishtools.utils.logging import setup_cli_logging


click.rich_click.SHOW_ARGUMENTS = True
click.rich_click.GROUP_ARGUMENTS_OPTIONS = True
click.rich_click.USE_MARKDOWN = True
click.rich_click.STYLE_HELPTEXT = ""

CCFTermKind = Literal["auto", "id", "acronym", "name"]
CombineMode = Literal["any", "all"]


def _load_json_object(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}, got {type(payload).__name__}.")
    return payload


def _as_list_of_str(value: object, *, key: str, path: Path) -> list[str]:
    if isinstance(value, str):
        return [value]
    if not isinstance(value, list | tuple):
        raise ValueError(f"Expected {key} to be a list of strings in {path}, got {type(value).__name__}.")
    out: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError(f"Expected {key} entries to be strings in {path}, got {type(item).__name__}.")
        out.append(item)
    return out


def _resolve_paths(paths: list[str], *, base_dir: Path) -> list[str]:
    out: list[str] = []
    for p in paths:
        pp = Path(p)
        if not pp.is_absolute():
            pp = base_dir / pp
        out.append(str(pp))
    return out


def _ants_numpy_yx(img: ants.ANTsImage) -> np.ndarray:
    arr_xy = np.asarray(img.numpy())
    if arr_xy.ndim != 2:
        raise ValueError(f"Expected 2D ANTsImage, got shape={arr_xy.shape}.")
    return arr_xy.T


def _atlas_slice_yx(*, atlas: BrainGlobeAtlas, plane: str, slice_idx: int) -> tuple[np.ndarray, np.ndarray]:
    if plane == "coronal":
        ref = np.asarray(atlas.reference[slice_idx, :, :])
        ann = np.asarray(atlas.annotation[slice_idx, :, :])
        return (ref, ann)
    if plane == "sagittal":
        # Avoid transposing the full 3D atlas volume.
        ref = np.asarray(atlas.reference[:, :, slice_idx]).T
        ann = np.asarray(atlas.annotation[:, :, slice_idx]).T
        return (ref, ann)
    raise ValueError(f"Unsupported atlas plane={plane!r}. Expected 'coronal' or 'sagittal'.")


def _moving_crop_yx(
    *,
    ws: Workspace,
    roi: str,
    stitch_codebook: str,
    p1: P1Landmarks,
    z_idx: int,
) -> np.ndarray:
    fused_zarr = ws.stitch(roi, stitch_codebook) / "fused.zarr"
    if not fused_zarr.exists():
        raise FileNotFoundError(f"Missing fused.zarr at {fused_zarr}")
    arr = zarr.open(str(fused_zarr), mode="r")

    ch_idx = 0

    if z_idx < 0 or z_idx >= int(arr.shape[0]):
        raise ValueError(f"z_idx out of range for fused.zarr: z_idx={z_idx}, shape[0]={int(arr.shape[0])}.")

    sample_slice_full_raw = np.asarray(arr[z_idx, :, :, ch_idx], dtype=np.float32)

    if p1.prior_flip_x:
        sample_slice_full_raw = sample_slice_full_raw[:, ::-1]
    if p1.prior_rotation_deg != 0:
        sample_slice_full_raw = ndimage_rotate(sample_slice_full_raw, int(p1.prior_rotation_deg), reshape=True, order=1)

    sr0, sr1, sc0, sc1 = p1.sample_rotated_crop_bbox
    moving_yx = sample_slice_full_raw[int(sr0) : int(sr1), int(sc0) : int(sc1)]
    if moving_yx.ndim != 2:
        raise ValueError(f"Expected moving crop to be 2D, got shape={moving_yx.shape}.")
    return moving_yx


def _moving_crop_yxc(
    *,
    ws: Workspace,
    roi: str,
    stitch_codebook: str,
    p1: P1Landmarks,
    z_idx: int,
) -> tuple[np.ndarray, list[str]]:
    fused_zarr = ws.stitch(roi, stitch_codebook) / "fused.zarr"
    if not fused_zarr.exists():
        raise FileNotFoundError(f"Missing fused.zarr at {fused_zarr}")
    arr = zarr.open(str(fused_zarr), mode="r")

    keys = list(arr.attrs.get("key", []))
    keys = [str(k) for k in keys] if keys else [stitch_codebook]

    if z_idx < 0 or z_idx >= int(arr.shape[0]):
        raise ValueError(f"z_idx out of range for fused.zarr: z_idx={z_idx}, shape[0]={int(arr.shape[0])}.")

    sample_slice_full_raw = np.asarray(arr[z_idx, :, :, :], dtype=np.float32)
    if sample_slice_full_raw.ndim != 3:
        raise ValueError(f"Expected fused slice shape (y,x,c), got {sample_slice_full_raw.shape}.")

    if p1.prior_flip_x:
        sample_slice_full_raw = sample_slice_full_raw[:, ::-1, :]
    if p1.prior_rotation_deg != 0:
        rotated: list[np.ndarray] = []
        for c in range(int(sample_slice_full_raw.shape[2])):
            rotated.append(ndimage_rotate(sample_slice_full_raw[:, :, c], int(p1.prior_rotation_deg), reshape=True, order=1))
        sample_slice_full_raw = np.stack(rotated, axis=2)

    sr0, sr1, sc0, sc1 = p1.sample_rotated_crop_bbox
    moving_yxc = sample_slice_full_raw[int(sr0) : int(sr1), int(sc0) : int(sc1), :]
    if moving_yxc.ndim != 3:
        raise ValueError(f"Expected moving crop to be 3D (y,x,c), got shape={moving_yxc.shape}.")

    if len(keys) != int(moving_yxc.shape[2]):
        keys = [f"ch{c}" for c in range(int(moving_yxc.shape[2]))]
    return moving_yxc, keys


def _term_mask_from_annotation_yx(
    *,
    annotation_yx: np.ndarray,
    terms: tuple[str, ...],
    kind: CCFTermKind,
    combine: CombineMode,
    invert: bool,
    atlas: BrainGlobeAtlas,
) -> np.ndarray:
    ann = np.asarray(annotation_yx)
    if ann.ndim != 2:
        raise ValueError(f"Expected 2D annotation, got shape={ann.shape}.")

    kind_t = cast(CCFTermKind, str(kind).lower())
    combine_t = cast(CombineMode, str(combine).lower())

    structures = atlas.structures

    ids_present = np.unique(ann.astype(np.int64, copy=False))
    ids_present = ids_present[ids_present > 0]

    def iter_structure_ids() -> list[int]:
        # BrainGlobeAtlas.structures is dict-like but not always a plain dict.
        # We only assume `__iter__` and `__getitem__` are supported.
        keys = getattr(structures, "keys", None)
        if callable(keys):
            return [int(v) for v in keys()]
        return [int(v) for v in structures]

    def resolve_term_ids(term: str) -> set[int]:
        s = str(term).strip()
        if kind_t == "id":
            return {int(s)}
        if kind_t == "auto" and s.isdigit():
            return {int(s)}

        q = s.lower()
        ids: set[int] = set()
        for sid in iter_structure_ids():
            st = structures[int(sid)]
            if kind_t in {"acronym", "auto"} and str(st.get("acronym", "")).lower() == q:
                ids.add(int(sid))
            if kind_t in {"name", "auto"} and str(st.get("name", "")).lower() == q:
                ids.add(int(sid))
        if not ids:
            raise ValueError(
                f"Could not resolve term={term!r} (kind={kind_t}) to any atlas structure id."
            )
        return ids

    def subtree_ids(root_ids: set[int]) -> set[int]:
        roots = {int(v) for v in root_ids}
        out: set[int] = set()
        for sid in ids_present.tolist():
            st = structures[int(sid)]
            path = st.get("structure_id_path", [])
            try:
                path_ids = {int(v) for v in path}
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid structure_id_path for structure id={int(sid)}") from exc
            if roots & path_ids:
                out.add(int(sid))
        return out

    per_term: list[np.ndarray] = []
    for t in terms:
        ids = resolve_term_ids(t)
        ids = subtree_ids(ids)
        per_term.append(np.isin(ann.astype(np.int64, copy=False), np.asarray(sorted(ids), dtype=np.int64)))

    if not per_term:
        raise ValueError("At least one --term is required.")

    if combine_t == "all":
        mask = np.logical_and.reduce(per_term)
    else:
        mask = np.logical_or.reduce(per_term)
    return ~mask if invert else mask


def _to_ants_2d(*, arr_yx: np.ndarray, spacing_um: float) -> ants.ANTsImage:
    a = np.asarray(arr_yx, dtype=np.float32)
    if a.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape={a.shape}.")
    sp_mm = float(spacing_um) * UM_TO_MM
    return ants.from_numpy(
        a.T,
        origin=[0.0, 0.0],
        spacing=[sp_mm, sp_mm],
    )


@click.command("export-mask-edit-pack")
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
    help="Subdirectory under <workspace>/analysis/output/ccf-transforms/<roi>/ containing similarity_plus_syn_summary.json.",
)
@click.option(
    "--stitch-codebook",
    default="pi",
    show_default=True,
    help="Stitch codebook name used for fused.zarr (analysis/deconv/stitch--{ROI}+{CODEBOOK}/fused.zarr).",
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
    help="Downsample factor applied to the moving crop BEFORE warping (8 matches the legacy thumbnail request).",
)
@click.option(
    "--target-spacing-um",
    type=float,
    default=2.0,
    show_default=True,
    help="Target pixel size (µm/px) for both outputs (moving thumbnail + mask).",
)
@click.option(
    "--term",
    "terms",
    multiple=True,
    required=True,
    help="Ontology term(s) to render as a mask (name/acronym/id). Repeatable.",
)
@click.option(
    "--kind",
    type=click.Choice(["auto", "id", "acronym", "name"], case_sensitive=False),
    default="auto",
    show_default=True,
    help="Interpretation of each --term value.",
)
@click.option(
    "--combine",
    type=click.Choice(["any", "all"], case_sensitive=False),
    default="any",
    show_default=True,
    help="How to combine multiple --term masks.",
)
@click.option("--invert/--no-invert", default=False, show_default=True, help="Invert the final mask.")
@click.option(
    "--warped-moving-png",
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
    help="Output PNG path for the moving thumbnail (RGB, first 3 fused channels; default: under run dir).",
)
@click.option(
    "--mask-tif",
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
    help="Output TIFF path for the atlas-region mask (default: under run dir).",
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
    z_idx: int | None,
    moving_pre_downsample: int,
    target_spacing_um: float,
    terms: tuple[str, ...],
    kind: str,
    combine: str,
    invert: bool,
    warped_moving_png: Path | None,
    mask_tif: Path | None,
    debug: bool,
) -> None:
    ws = Workspace(workspace)
    process_all_rois = roi is None
    if roi is None and (warped_moving_png is not None or mask_tif is not None):
        raise click.BadParameter(
            "When ROI is omitted (process all ROIs), --warped-moving-png/--mask-tif must be omitted "
            "so per-ROI default filenames can be used."
        )
    try:
        rois_resolved = ws.resolve_rois((roi,)) if roi is not None else ws.resolve_rois(None)
    except ValueError as exc:
        raise click.BadParameter(str(exc), param_hint="roi") from exc

    for roi_resolved in rois_resolved:
        setup_cli_logging(
            workspace,
            component="ccf.export_mask_edit_pack",
            file=f"export-mask-edit-pack-{roi_resolved}",
            debug=debug,
            extra={"roi": str(roi_resolved)},
        )

        try:
            _run_one_roi(
                ws=ws,
                roi_resolved=roi_resolved,
                run_dirname=str(run_dirname),
                stitch_codebook=str(stitch_codebook),
                z_idx=z_idx,
                moving_pre_downsample=int(moving_pre_downsample),
                target_spacing_um=float(target_spacing_um),
                terms=terms,
                kind=str(kind),
                combine=str(combine),
                invert=bool(invert),
                warped_moving_png=warped_moving_png,
                mask_tif=mask_tif,
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
    z_idx: int | None,
    moving_pre_downsample: int,
    target_spacing_um: float,
    terms: tuple[str, ...],
    kind: str,
    combine: str,
    invert: bool,
    warped_moving_png: Path | None,
    mask_tif: Path | None,
) -> None:
    t_start = time.perf_counter()
    timings_s: dict[str, float] = {}

    def mark(step: str, t0: float) -> float:
        t1 = time.perf_counter()
        timings_s[step] = t1 - t0
        return t1

    click.echo(f"[{roi_resolved}] export-mask-edit-pack: start")
    out_root = ws.ccf_transforms(roi_resolved)
    run_dir = out_root / str(run_dirname)
    summary_path = run_dir / "similarity_plus_syn_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing {summary_path}. Did you run ccf/ants_landmark_syn_init.py?")

    t0 = time.perf_counter()
    summary = _load_json_object(summary_path)
    if "invtransforms" not in summary:
        raise ValueError(f"Invalid summary at {summary_path}: missing invtransforms.")

    inv_list = _resolve_paths(
        _as_list_of_str(summary["invtransforms"], key="invtransforms", path=summary_path), base_dir=run_dir
    )
    for p in inv_list:
        if not Path(p).exists():
            raise FileNotFoundError(f"Transform file not found: {p} (referenced by {summary_path})")
    t0 = mark("load_summary_and_transforms", t0)

    out_contract = LandmarkRegistrationOutputs(out_root)
    t1 = time.perf_counter()
    p1 = out_contract.read_p1_landmarks()
    t0 = mark("load_p1_landmarks", t1)

    atlas_voxel_um = float(p1.atlas_voxel_um) if p1.atlas_voxel_um is not None else 20.0
    sample_voxel_um = float(p1.sample_voxel_xy_um) if p1.sample_voxel_xy_um is not None else 0.216

    atlas_name = p1.atlas_name or "kim_dev_mouse_e15-5_lsfm_20um"
    atlas_plane = str(p1.atlas_plane or ("sagittal" if "Sag" in str(ws.path) else "coronal")).lower()
    if p1.atlas_slice_idx is None:
        raise click.ClickException(f"p1_landmarks.json is missing atlas_slice_idx under {out_root}.")
    atlas_slice_idx = int(p1.atlas_slice_idx)

    out_dir = run_dir / "mask_edit"
    out_dir.mkdir(parents=True, exist_ok=True)
    z_idx_eff = int(z_idx) if z_idx is not None else int(p1.sample_z_idx) if p1.sample_z_idx is not None else 5

    ds = int(moving_pre_downsample)
    if ds <= 0:
        raise click.BadParameter(f"moving_pre_downsample must be > 0, got {ds}.")
    target_um = float(target_spacing_um)
    if not target_um > 0:
        raise click.BadParameter(f"target_spacing_um must be > 0, got {target_um}.")

    tag = f"z{z_idx_eff}_ds{ds}_target{target_um:g}um"
    if warped_moving_png is None:
        warped_moving_png = out_dir / f"warped_moving_{tag}.png"
    if mask_tif is None:
        mask_tif = out_dir / f"mask_{tag}.tif"

    # --- Fixed atlas annotation crop (mask source, fixed space) ---
    t1 = time.perf_counter()
    atlas = BrainGlobeAtlas(atlas_name)
    _, atlas_ann_full_yx = _atlas_slice_yx(atlas=atlas, plane=atlas_plane, slice_idx=atlas_slice_idx)
    ar0, ar1, ac0, ac1 = p1.atlas_crop_bbox
    click.echo(
        f"[{roi_resolved}] atlas={atlas_name}, plane={atlas_plane}, slice={atlas_slice_idx}, "
        f"atlas_voxel_um={atlas_voxel_um:g}, sample_voxel_um={sample_voxel_um:g}, invtransforms={len(inv_list)}"
    )

    fixed_ann_crop = atlas_ann_full_yx[int(ar0) : int(ar1), int(ac0) : int(ac1)]
    t0 = mark("load_atlas_annotation_crop", t1)

    # --- Term mask in fixed crop domain ---
    t1 = time.perf_counter()
    term_mask_yx = _term_mask_from_annotation_yx(
        annotation_yx=fixed_ann_crop,
        terms=terms,
        kind=cast(CCFTermKind, str(kind).lower()),
        combine=cast(CombineMode, str(combine).lower()),
        invert=bool(invert),
        atlas=atlas,
    )
    if not bool(np.any(term_mask_yx)):
        click.echo(f"[{roi_resolved}] empty mask for terms={terms}; skipping")
        return
    fixed_mask_yx = term_mask_yx
    fixed_mask_ants = _to_ants_2d(arr_yx=(fixed_mask_yx.astype(np.float32, copy=False) * 255.0), spacing_um=atlas_voxel_um)
    t0 = mark("term_mask_and_to_ants", t1)

    # --- Moving crop thumbnail in moving space, resampled to target spacing ---
    t1 = time.perf_counter()
    moving_crop_yxc, _ = _moving_crop_yxc(
        ws=ws,
        roi=str(roi_resolved),
        stitch_codebook=str(stitch_codebook),
        p1=p1,
        z_idx=z_idx_eff,
    )
    moving_crop_ds_yxc = moving_crop_yxc[::ds, ::ds, :].astype(np.float32, copy=False)
    moving_spacing_um = sample_voxel_um * float(ds)

    n_in = int(moving_crop_ds_yxc.shape[2])
    n_rgb = min(3, n_in)
    click.echo(
        f"[{roi_resolved}] moving crop yxc={moving_crop_yxc.shape}, ds={ds} -> {moving_crop_ds_yxc.shape}, "
        f"moving_spacing_um={moving_spacing_um:g}, target_um={target_um:g}, rgb_channels={n_rgb}"
    )
    outs_yx: list[np.ndarray] = []
    norms_yx: list[np.ndarray] = []
    for c in range(n_rgb):
        moving_sitk = sitk_from_numpy_2d(moving_crop_ds_yxc[:, :, c], spacing_um=moving_spacing_um)
        moving_out_sitk = resample_sitk_to_spacing(moving_sitk, target_spacing_um=target_um, interp=sitk.sitkLinear)
        out_yx = sitk.GetArrayFromImage(moving_out_sitk).astype(np.float32, copy=False)
        outs_yx.append(out_yx)
        norms_yx.append(normalize_robust(out_yx, 1.0, 99.0))
    t0 = mark("load_moving_and_resample", t1)

    if not outs_yx:
        raise ValueError("Fused image has no channels.")

    if len(norms_yx) == 1:
        rgb = np.repeat(norms_yx[0][:, :, None], 3, axis=2)
    elif len(norms_yx) == 2:
        rgb = np.stack([norms_yx[0], norms_yx[1], np.zeros_like(norms_yx[0])], axis=2)
    else:
        rgb = np.stack(norms_yx[:3], axis=2)

    moving_out_ants = _to_ants_2d(arr_yx=outs_yx[0], spacing_um=target_um)

    t1 = time.perf_counter()
    warped_mask = ants.apply_transforms(
        fixed=moving_out_ants,
        moving=fixed_mask_ants,
        transformlist=inv_list,
        interpolator="bSpline",
        defaultvalue=0,
    )
    t0 = mark("ants_apply_transforms", t1)
    warped_mask_yx = np.clip(_ants_numpy_yx(warped_mask).astype(np.float32, copy=False), 0.0, 255.0)
    mask_bin = warped_mask_yx >= 128.0
    if not bool(np.any(mask_bin)):
        click.echo(f"[{roi_resolved}] warped mask is empty; skipping")
        return

    if rgb.shape[:2] != mask_bin.shape:
        raise ValueError(f"Output shape mismatch: moving={rgb.shape[:2]} vs mask={mask_bin.shape}.")

    t1 = time.perf_counter()
    warped_moving_png.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(warped_moving_png.as_posix(), rgb, vmin=0.0, vmax=1.0)

    mask_tif.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(
        mask_tif.as_posix(),
        (mask_bin.astype(np.uint8, copy=False) * 255),
        photometric="minisblack",
    )
    t0 = mark("write_outputs", t1)

    click.echo(f"Wrote: {warped_moving_png}")
    click.echo(f"Wrote: {mask_tif}")
    timings_s["total"] = time.perf_counter() - t_start
    click.echo(
        f"[{roi_resolved}] timings (s): "
        + ", ".join(f"{k}={timings_s[k]:.3f}" for k in sorted(timings_s.keys()) if k != "total")
        + f", total={timings_s['total']:.3f}"
    )


if __name__ == "__main__":
    main()
