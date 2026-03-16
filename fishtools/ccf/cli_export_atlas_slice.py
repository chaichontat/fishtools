from __future__ import annotations

import math
from pathlib import Path

import matplotlib.image as mpimg
import numpy as np
import rich_click as click
from brainglobe_atlasapi import BrainGlobeAtlas

from fishtools.ccf.landmark import LandmarkRegistrationOutputs, P1Landmarks
from fishtools.io.workspace import Workspace
from fishtools.utils.logging import setup_cli_logging


click.rich_click.SHOW_ARGUMENTS = True
click.rich_click.GROUP_ARGUMENTS_OPTIONS = True
click.rich_click.USE_MARKDOWN = True
click.rich_click.STYLE_HELPTEXT = ""


def _normalize_for_display(
    img_yx: np.ndarray, *, mode: str, lo_percentile: float, hi_percentile: float
) -> np.ndarray:
    arr = np.asarray(img_yx, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros_like(arr, dtype=np.float32)

    mode_norm = str(mode).strip().lower()
    if mode_norm == "auto":
        vmin = float(np.min(finite))
        vmax = float(np.max(finite))
    elif mode_norm == "robust":
        lo = float(lo_percentile)
        hi = float(hi_percentile)
        if not (0.0 <= lo < hi <= 100.0):
            raise ValueError(f"Invalid percentiles: lo={lo} hi={hi} (expected 0 <= lo < hi <= 100).")
        vmin, vmax = (float(v) for v in np.percentile(finite, [lo, hi]))
    else:
        raise ValueError(f"Unknown contrast mode {mode!r}.")

    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return np.zeros_like(arr, dtype=np.float32)
    if math.isclose(vmin, vmax):
        return np.zeros_like(arr, dtype=np.float32)

    norm = (arr - vmin) / (vmax - vmin)
    return np.clip(norm, 0.0, 1.0).astype(np.float32, copy=False)


def _pallium_subtree_ids(atlas: BrainGlobeAtlas) -> set[int]:
    df = atlas.lookup_df
    if "name" not in df.columns or "acronym" not in df.columns or "id" not in df.columns:
        raise ValueError(f"Unexpected atlas.lookup_df schema: columns={list(df.columns)}")

    candidates = df[
        (df["name"].astype(str).str.lower() == "pallium")
        | (df["acronym"].astype(str).str.lower().isin({"pal", "pallium"}))
    ]
    if candidates.empty:
        candidates = df[df["name"].astype(str).str.lower().str.contains("pallium", na=False)]
    if candidates.empty:
        raise ValueError("Could not find a 'pallium' structure in atlas lookup table.")

    candidate_ids = [int(v) for v in candidates["id"].tolist()]
    best_id = min(candidate_ids, key=lambda rid: len(atlas.structures[rid]["structure_id_path"]))

    subtree_ids: set[int] = set()
    for struct in atlas.structures_list:
        path = struct.get("structure_id_path", [])
        if isinstance(path, list) and best_id in {int(v) for v in path}:
            subtree_ids.add(int(struct["id"]))
    return subtree_ids


def _ventricles_subtree_ids(atlas: BrainGlobeAtlas) -> set[int]:
    df = atlas.lookup_df
    if "name" not in df.columns or "acronym" not in df.columns or "id" not in df.columns:
        raise ValueError(f"Unexpected atlas.lookup_df schema: columns={list(df.columns)}")

    candidates = df[
        (df["name"].astype(str).str.lower() == "ventricles")
        | (df["acronym"].astype(str).str.lower().isin({"ventricles"}))
    ]
    if candidates.empty:
        candidates = df[df["name"].astype(str).str.lower().str.contains("ventric", na=False)]
    if candidates.empty:
        raise ValueError("Could not find a 'ventricles' structure in atlas lookup table.")

    candidate_ids = [int(v) for v in candidates["id"].tolist()]
    best_id = min(candidate_ids, key=lambda rid: len(atlas.structures[rid]["structure_id_path"]))

    subtree_ids: set[int] = set()
    for struct in atlas.structures_list:
        path = struct.get("structure_id_path", [])
        if isinstance(path, list) and best_id in {int(v) for v in path}:
            subtree_ids.add(int(struct["id"]))
    return subtree_ids


def _atlas_slices(*, atlas: BrainGlobeAtlas, plane: str) -> tuple[np.ndarray, np.ndarray]:
    plane_norm = str(plane).strip().lower()
    if plane_norm not in {"coronal", "sagittal"}:
        raise ValueError(f"Invalid atlas plane {plane!r}; expected 'coronal' or 'sagittal'.")
    if plane_norm == "coronal":
        return np.asarray(atlas.reference), np.asarray(atlas.annotation)
    # Match `ccf/register_partial_section.py` convention.
    return np.asarray(atlas.reference).transpose(2, 1, 0), np.asarray(atlas.annotation).transpose(2, 1, 0)


def _validate_bbox(*, bbox: tuple[int, int, int, int], shape_yx: tuple[int, int]) -> None:
    r0, r1, c0, c1 = (int(v) for v in bbox)
    h, w = (int(shape_yx[0]), int(shape_yx[1]))
    if not (0 <= r0 < r1 <= h and 0 <= c0 < c1 <= w):
        raise ValueError(f"Invalid atlas_crop_bbox={bbox} for atlas slice shape (y,x)=({h},{w}).")


def _compose_fixed_rgb01(
    *,
    atlas_ref_crop_yx: np.ndarray,
    atlas_ann_crop_yx: np.ndarray,
    atlas: BrainGlobeAtlas,
    contrast: str,
    lo_percentile: float,
    hi_percentile: float,
    overlay_pallium: bool,
    overlay_ventricles: bool,
) -> np.ndarray:
    gray01 = _normalize_for_display(
        atlas_ref_crop_yx, mode=contrast, lo_percentile=lo_percentile, hi_percentile=hi_percentile
    )
    rgb01 = np.repeat(gray01[:, :, None], 3, axis=2).astype(np.float32, copy=False)

    def _apply_overlay(mask_yx: np.ndarray, *, color_rgb: tuple[float, float, float], alpha: float) -> None:
        m = np.asarray(mask_yx, dtype=bool)
        if m.shape != rgb01.shape[:2]:
            raise ValueError(f"Overlay mask shape mismatch: {m.shape} vs fixed {rgb01.shape[:2]}.")
        a = float(alpha)
        if a <= 0:
            return
        w = (m.astype(np.float32) * a)[:, :, None]
        color = np.array(color_rgb, dtype=np.float32)[None, None, :]
        rgb01[:] = rgb01 * (1.0 - w) + color * w

    if overlay_pallium:
        try:
            pallium_ids = _pallium_subtree_ids(atlas)
            _apply_overlay(np.isin(atlas_ann_crop_yx, list(pallium_ids)), color_rgb=(0.1, 0.4, 1.0), alpha=0.30)
        except Exception as exc:
            click.echo(f"WARNING: Pallium overlay disabled: {exc}")
    if overlay_ventricles:
        try:
            vent_ids = _ventricles_subtree_ids(atlas)
            _apply_overlay(np.isin(atlas_ann_crop_yx, list(vent_ids)), color_rgb=(0.5, 1.0, 0.5), alpha=0.25)
        except Exception as exc:
            click.echo(f"WARNING: Ventricles overlay disabled: {exc}")

    if rgb01.ndim != 3 or rgb01.shape[2] != 3:
        raise ValueError(f"Expected RGB image, got shape={rgb01.shape}.")
    return np.clip(rgb01, 0.0, 1.0).astype(np.float32, copy=False)


def _infer_p1_path(*, ws: Workspace, roi: str, p1_landmarks_json: Path | None) -> Path:
    if p1_landmarks_json is not None:
        return Path(p1_landmarks_json)
    return LandmarkRegistrationOutputs(ws.ccf_transforms(roi)).p1_landmarks_json


def _load_p1(*, p1_path: Path) -> P1Landmarks:
    if not p1_path.exists():
        raise FileNotFoundError(f"Missing p1_landmarks.json at {p1_path}")
    return P1Landmarks.from_json(p1_path)


def _required_str(name: str | None, *, field: str) -> str:
    if name is None or not str(name).strip():
        raise click.ClickException(
            f"Missing {field} in p1_landmarks.json; re-save landmarks in register_partial_section.py, "
            f"or pass --{field.replace('_', '-')}."
        )
    return str(name)


def _required_int(value: int | None, *, field: str) -> int:
    if value is None:
        raise click.ClickException(
            f"Missing {field} in p1_landmarks.json; re-save landmarks in register_partial_section.py, "
            f"or pass --{field.replace('_', '-')}."
        )
    return int(value)


@click.command("export-atlas-slice")
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
    "--p1-landmarks-json",
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
    help="Optional path to `p1_landmarks.json` (default: <workspace>/analysis/output/ccf-transforms/<roi>/p1_landmarks.json).",
)
@click.option("--atlas-name", default=None, show_default=False, help="Override atlas name (fallback if missing in JSON).")
@click.option(
    "--atlas-plane",
    type=click.Choice(["coronal", "sagittal"], case_sensitive=False),
    default=None,
    show_default=False,
    help="Override atlas plane (fallback if missing in JSON).",
)
@click.option("--atlas-slice-idx", type=int, default=None, show_default=False, help="Override atlas slice index.")
@click.option(
    "--contrast",
    type=click.Choice(["auto", "robust"], case_sensitive=False),
    default="auto",
    show_default=True,
    help="Intensity scaling for the exported PNG.",
)
@click.option(
    "--lo-percentile",
    type=float,
    default=1.0,
    show_default=True,
    help="Low percentile for --contrast=robust.",
)
@click.option(
    "--hi-percentile",
    type=float,
    default=99.8,
    show_default=True,
    help="High percentile for --contrast=robust.",
)
@click.option(
    "--overlays/--no-overlays",
    default=True,
    show_default=True,
    help="Overlay pallium (blue) + ventricles (green) like `ccf/register_partial_section.py`.",
)
@click.option(
    "--out-png",
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
    help="Output PNG path (default: <ccf-transforms>/<roi>/p1_atlas_slice.png).",
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
    p1_landmarks_json: Path | None,
    atlas_name: str | None,
    atlas_plane: str | None,
    atlas_slice_idx: int | None,
    contrast: str,
    lo_percentile: float,
    hi_percentile: float,
    overlays: bool,
    out_png: Path | None,
    debug: bool,
) -> None:
    """Export the (cropped) CCF atlas reference slice implied by `p1_landmarks.json` to a PNG.

    This corresponds to the "FIXED (atlas)" panel shown in `ccf/register_partial_section.py`.
    """

    ws = Workspace(workspace)
    process_all_rois = roi is None
    if roi is None and (out_png is not None or p1_landmarks_json is not None):
        raise click.BadParameter(
            "When ROI is omitted (process all ROIs), --out-png and --p1-landmarks-json must be omitted."
        )
    try:
        rois_resolved = ws.resolve_rois((roi,)) if roi is not None else ws.resolve_rois(None)
    except ValueError as exc:
        raise click.BadParameter(str(exc), param_hint="roi") from exc

    for roi_resolved in rois_resolved:
        setup_cli_logging(
            workspace,
            component="ccf.export_atlas_slice",
            file=f"export-atlas-slice-{roi_resolved}",
            debug=debug,
            extra={"roi": str(roi_resolved)},
        )
        try:
            _run_one_roi(
                ws=ws,
                roi=str(roi_resolved),
                p1_landmarks_json=p1_landmarks_json,
                atlas_name_override=atlas_name,
                atlas_plane_override=atlas_plane,
                atlas_slice_idx_override=atlas_slice_idx,
                contrast=str(contrast),
                lo_percentile=float(lo_percentile),
                hi_percentile=float(hi_percentile),
                overlays=bool(overlays),
                out_png=out_png,
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
    roi: str,
    p1_landmarks_json: Path | None,
    atlas_name_override: str | None,
    atlas_plane_override: str | None,
    atlas_slice_idx_override: int | None,
    contrast: str,
    lo_percentile: float,
    hi_percentile: float,
    overlays: bool,
    out_png: Path | None,
) -> None:
    p1_path = _infer_p1_path(ws=ws, roi=roi, p1_landmarks_json=p1_landmarks_json)
    p1 = _load_p1(p1_path=p1_path)

    atlas_name_ = _required_str(atlas_name_override or p1.atlas_name, field="atlas_name")
    atlas_plane_ = _required_str(atlas_plane_override or p1.atlas_plane, field="atlas_plane")
    atlas_slice_idx_ = _required_int(
        atlas_slice_idx_override if atlas_slice_idx_override is not None else p1.atlas_slice_idx,
        field="atlas_slice_idx",
    )

    out_root = ws.ccf_transforms(roi)
    out_path = out_png if out_png is not None else (out_root / "p1_atlas_slice.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    atlas = BrainGlobeAtlas(atlas_name_)
    ref_zyx, ann_zyx = _atlas_slices(atlas=atlas, plane=atlas_plane_)

    if atlas_slice_idx_ < 0 or atlas_slice_idx_ >= int(ref_zyx.shape[0]):
        raise ValueError(f"atlas_slice_idx out of range: idx={atlas_slice_idx_}, available_z={int(ref_zyx.shape[0])}.")

    ref_full_yx = np.asarray(ref_zyx[int(atlas_slice_idx_), :, :], dtype=np.float32)
    ann_full_yx = np.asarray(ann_zyx[int(atlas_slice_idx_), :, :], dtype=np.int32)
    if ref_full_yx.shape != ann_full_yx.shape:
        raise ValueError(f"Atlas reference/annotation shape mismatch: ref={ref_full_yx.shape} ann={ann_full_yx.shape}.")

    brain_mask_full = ann_full_yx > 0
    ref_full_masked = ref_full_yx.copy()
    ref_full_masked[~brain_mask_full] = 0.0

    bbox = p1.atlas_crop_bbox
    _validate_bbox(bbox=bbox, shape_yx=ref_full_masked.shape)
    r0, r1, c0, c1 = (int(v) for v in bbox)

    ref_crop = ref_full_masked[r0:r1, c0:c1]
    ann_crop = ann_full_yx[r0:r1, c0:c1]

    rgb01 = _compose_fixed_rgb01(
        atlas_ref_crop_yx=ref_crop,
        atlas_ann_crop_yx=ann_crop,
        atlas=atlas,
        contrast=contrast,
        lo_percentile=lo_percentile,
        hi_percentile=hi_percentile,
        overlay_pallium=bool(overlays),
        overlay_ventricles=bool(overlays),
    )

    mpimg.imsave(out_path.as_posix(), rgb01, vmin=0.0, vmax=1.0)
    click.echo(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
