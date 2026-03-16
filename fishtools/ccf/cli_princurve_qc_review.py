from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import numpy as np
import rich_click as click

from fishtools.ccf.cli_filter_h5ad_ccf import _find_imagej_roi_file, _write_ccf_user_mask_overlay_png
from fishtools.io.workspace import Workspace

# Force a non-interactive backend for headless CLI runs.
mpl.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402


@dataclass(frozen=True, slots=True)
class _ReviewRow:
    roi: str
    subroi: str
    axis3_png: Path | None
    curve_qc_png: Path | None


def _find_princurve_script_path() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts" / "princurve" / "find_princurve.py"
    if not script.exists():
        raise FileNotFoundError(f"Could not find find_princurve.py at {script}.")
    return script


def _resolve_workspace_input_path(*, ws: Workspace, roi: str) -> Path:
    roi_clean = str(roi).strip()
    roi_dir = ws.ccf_transforms(roi_clean)
    expected = roi_dir / f"{roi_clean}.syn.annotated.h5ad"
    if expected.exists():
        return expected

    if not roi_dir.exists():
        raise FileNotFoundError(f"Input not found: {expected}")

    candidates = sorted(p for p in roi_dir.glob("*.syn.annotated.h5ad") if p.is_file())
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        names = "\n".join([f"  - {p.name}" for p in candidates])
        raise FileNotFoundError(
            f"Input not found: {expected}\n"
            "Found multiple candidate inputs in ROI directory; choose one explicitly:\n"
            f"{names}"
        )
    raise FileNotFoundError(f"Input not found: {expected}")


def _curve_qc_subroi_from_name(*, input_stem: str, qc_png: Path) -> str:
    suffix = ".princurve.curve.qc.png"
    name = qc_png.name
    if not name.endswith(suffix):
        return "unknown"
    prefix = name[: -len(suffix)]
    if prefix == input_stem:
        return "(all)"
    stem_prefix = f"{input_stem}."
    if prefix.startswith(stem_prefix):
        out = prefix[len(stem_prefix) :]
        return out if out else "(all)"
    return prefix


def _discover_curve_qc_pngs(*, ws: Workspace, roi: str) -> list[tuple[str, Path]]:
    in_path = _resolve_workspace_input_path(ws=ws, roi=roi)
    roi_dir = ws.ccf_transforms(str(roi))
    matches = sorted(p for p in roi_dir.glob(f"{in_path.stem}*.princurve.curve.qc.png") if p.is_file())
    return [(_curve_qc_subroi_from_name(input_stem=in_path.stem, qc_png=p), p) for p in matches]


def _run_find_princurve_for_roi(*, workspace: Path, roi: str) -> None:
    script = _find_princurve_script_path()
    cmd = [sys.executable, str(script), str(workspace), str(roi)]
    proc = subprocess.run(cmd, cwd=str(script.parent), check=False, capture_output=True, text=True)
    if proc.stdout.strip():
        click.echo(proc.stdout.rstrip())
    if proc.returncode != 0:
        stderr = proc.stderr.strip()
        stdout = proc.stdout.strip()
        detail = stderr if stderr else stdout
        raise RuntimeError(detail if detail else f"find_princurve.py exited with code {proc.returncode}.")


def _resolve_review_rois(*, ws: Workspace, rois: tuple[str, ...]) -> list[str]:
    explicit = [str(r).strip() for r in rois if str(r).strip()]
    if explicit:
        return explicit

    ccf_root = ws.output.ccf_transforms
    if ccf_root.exists():
        ccf_rois = sorted(path.name for path in ccf_root.iterdir() if path.is_dir())
        if ccf_rois:
            return ccf_rois

    try:
        resolved = ws.resolve_rois(None)
    except ValueError as exc:
        raise click.BadParameter(str(exc), param_hint="rois") from exc
    return [str(r) for r in resolved]


def _load_png_rgb(path: Path) -> np.ndarray:
    arr = np.asarray(plt.imread(path))
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.ndim != 3 or arr.shape[2] not in (3, 4):
        raise ValueError(f"Unexpected image shape for {path}: {arr.shape}")
    arr = arr.astype(np.float32, copy=False)
    if arr.shape[2] == 4:
        alpha = np.clip(arr[..., 3:4], 0.0, 1.0)
        arr = arr[..., :3] * alpha + (1.0 - alpha)
    elif arr.max() > 1.0:
        arr = arr / 255.0
    return np.clip(arr[..., :3], 0.0, 1.0)


def _render_review_png(
    *,
    rows: list[_ReviewRow],
    missing_notes: list[str],
    output_png: Path,
    title: str,
) -> None:
    n_data_rows = max(1, len(rows))
    include_summary = len(missing_notes) > 0
    total_rows = n_data_rows + (1 if include_summary else 0)

    fig, axs = plt.subplots(total_rows, 2, figsize=(14.0, 3.8 * float(total_rows)), squeeze=False)
    data_rows = rows if rows else [_ReviewRow(roi="(none)", subroi="(none)", axis3_png=None, curve_qc_png=None)]

    for r_idx, row in enumerate(data_rows):
        ax_l = axs[r_idx, 0]
        ax_r = axs[r_idx, 1]

        ax_l.axis("off")
        ax_r.axis("off")
        ax_l.set_title(f"roi={row.roi} subroi={row.subroi} | similarity+syn axis 3")
        ax_r.set_title(f"roi={row.roi} subroi={row.subroi} | princurve curve QC")

        if row.axis3_png is not None and row.axis3_png.exists():
            ax_l.imshow(_load_png_rgb(row.axis3_png), interpolation="nearest")
        else:
            ax_l.text(0.02, 0.5, "Missing axis 3 panel", ha="left", va="center", fontsize=10)

        if row.curve_qc_png is not None and row.curve_qc_png.exists():
            ax_r.imshow(_load_png_rgb(row.curve_qc_png), interpolation="nearest")
        else:
            ax_r.text(0.02, 0.5, "Missing princurve curve QC", ha="left", va="center", fontsize=10)

    if include_summary:
        ax_summary = axs[-1, 0]
        ax_blank = axs[-1, 1]
        ax_summary.axis("off")
        ax_blank.axis("off")
        max_lines = 60
        clipped = missing_notes[:max_lines]
        if len(missing_notes) > max_lines:
            clipped.append(f"... +{len(missing_notes) - max_lines} more")
        summary_text = "Skipped/missing summary\n" + "\n".join(clipped)
        ax_summary.text(0.01, 0.99, summary_text, ha="left", va="top", fontsize=9, family="monospace")

    fig.suptitle(title)
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=200)
    plt.close(fig)


@click.command("princurve-qc-review")
@click.argument(
    "workspace",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True, resolve_path=True, path_type=Path),
)
@click.argument("rois", nargs=-1)
@click.option(
    "--output-png",
    default=None,
    type=click.Path(exists=False, file_okay=True, dir_okay=False, writable=True, resolve_path=True, path_type=Path),
    help="Output PNG path for the single review sheet (default: <workspace>/analysis/output/ccf-transforms/princurve_qc_review.png).",
)
@click.option(
    "--run-dirname",
    default="landmark_syn_mi",
    show_default=True,
    help="Subdirectory under <workspace>/analysis/output/ccf-transforms/<roi>/ used for mask_edit and SyN summary artifacts.",
)
@click.option(
    "--imagej-target-spacing-um",
    type=float,
    default=2.0,
    show_default=True,
    help="Pixel size (microns) of the moving thumbnail used for user ROI rasterization.",
)
@click.option(
    "--overwrite/--no-overwrite",
    default=True,
    show_default=True,
    help="Recompute per-ROI artifacts even when existing files are present.",
)
@click.option(
    "--skip-missing/--no-skip-missing",
    default=True,
    show_default=True,
    help="Continue when per-ROI inputs are missing and summarize skips at the end.",
)
def main(
    workspace: Path,
    rois: tuple[str, ...],
    *,
    output_png: Path | None,
    run_dirname: str,
    imagej_target_spacing_um: float,
    overwrite: bool,
    skip_missing: bool,
) -> None:
    """Build a single review PNG for all ROI/subROI curve QCs.

    For each ROI, this command:
    1) Rebuilds the CCF user-mask overlay and writes the axis-3-only panel.
    2) Re-runs `scripts/princurve/find_princurve.py` for that ROI.
    3) Collects all generated `*.princurve.curve.qc.png` files and stitches one row per subROI.
    """

    ws = Workspace(workspace)
    out_png = output_png if output_png is not None else (ws.output.ccf_transforms / "princurve_qc_review.png")
    resolved_rois = _resolve_review_rois(ws=ws, rois=rois)
    if not resolved_rois:
        raise click.ClickException("No ROIs resolved from workspace.")

    rows: list[_ReviewRow] = []
    missing_notes: list[str] = []

    for roi in resolved_rois:
        roi_str = str(roi)
        axis3_png: Path | None = None
        run_dir = ws.ccf_transforms(roi_str) / str(run_dirname)
        mask_edit_dir = run_dir / "mask_edit"

        roi_path: Path | None = None
        if mask_edit_dir.exists():
            try:
                roi_path = _find_imagej_roi_file(mask_edit_dir)
            except click.ClickException as exc:
                msg = f"roi={roi_str}: cannot resolve ImageJ ROI in {mask_edit_dir}: {exc}"
                if skip_missing:
                    missing_notes.append(msg)
                else:
                    raise click.ClickException(msg) from exc
        else:
            missing_notes.append(f"roi={roi_str}: missing mask_edit directory: {mask_edit_dir}")

        if roi_path is not None:
            try:
                outputs = _write_ccf_user_mask_overlay_png(
                    ws=ws,
                    roi=roi_str,
                    run_dirname=str(run_dirname),
                    roi_path=roi_path,
                    imagej_target_spacing_um=float(imagej_target_spacing_um),
                    overwrite=bool(overwrite),
                )
                if outputs is not None:
                    maybe_axis3 = outputs.get("overlay_axis3_png")
                    if maybe_axis3 is not None:
                        axis3_png = maybe_axis3
            except (FileNotFoundError, KeyError, TypeError, ValueError, OSError, RuntimeError) as exc:
                msg = f"roi={roi_str}: failed to generate similarity+syn axis 3 panel: {exc}"
                if skip_missing:
                    missing_notes.append(msg)
                else:
                    raise click.ClickException(msg) from exc

        try:
            _run_find_princurve_for_roi(workspace=ws.path, roi=roi_str)
        except RuntimeError as exc:
            msg = f"roi={roi_str}: find_princurve failed: {exc}"
            if skip_missing:
                missing_notes.append(msg)
                rows.append(_ReviewRow(roi=roi_str, subroi="(failed)", axis3_png=axis3_png, curve_qc_png=None))
                continue
            raise click.ClickException(msg) from exc

        curve_qcs = _discover_curve_qc_pngs(ws=ws, roi=roi_str)
        if not curve_qcs:
            missing_notes.append(f"roi={roi_str}: no princurve curve QC pngs found after recompute.")
            rows.append(_ReviewRow(roi=roi_str, subroi="(none)", axis3_png=axis3_png, curve_qc_png=None))
            continue

        for subroi, curve_png in curve_qcs:
            rows.append(_ReviewRow(roi=roi_str, subroi=subroi, axis3_png=axis3_png, curve_qc_png=curve_png))

    _render_review_png(
        rows=rows,
        missing_notes=missing_notes,
        output_png=out_png,
        title=f"Princurve QC review | workspace={ws.path}",
    )
    click.echo(f"Wrote review PNG: {out_png}")
    if missing_notes:
        click.echo(f"Completed with {len(missing_notes)} skipped/missing notes.", err=True)
