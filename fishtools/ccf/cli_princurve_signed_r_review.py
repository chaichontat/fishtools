from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import matplotlib as mpl
import numpy as np
import rich_click as click

from fishtools.ccf.princurve import fit_anchor_curve, project_to_polyline_arclength
from fishtools.io.workspace import Workspace

# Force a non-interactive backend for headless CLI runs.
mpl.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402


@dataclass(frozen=True, slots=True)
class _AnchorJob:
    roi: str
    subroi: str
    anchor_json: Path


@dataclass(frozen=True, slots=True)
class _SignedRPanel:
    roi: str
    subroi: str
    reverse_r_sign: bool
    points_xy: np.ndarray
    points_r: np.ndarray
    curve_xy: np.ndarray
    anchor_xy: np.ndarray
    lim: float


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


def _anchor_subroi_from_name(*, input_stem: str, anchor_json: Path) -> str | None:
    suffix = ".anchors.json"
    name = anchor_json.name
    if not name.endswith(suffix):
        return None
    prefix = name[: -len(suffix)]
    if prefix == input_stem:
        return "(all)"
    stem_prefix = f"{input_stem}."
    if prefix.startswith(stem_prefix):
        out = prefix[len(stem_prefix) :]
        return out if out else "(all)"
    return None


def _discover_anchor_jobs(*, ws: Workspace, roi: str) -> list[_AnchorJob]:
    in_path = _resolve_workspace_input_path(ws=ws, roi=roi)
    roi_dir = ws.ccf_transforms(str(roi))
    out: list[_AnchorJob] = []
    for path in sorted(p for p in roi_dir.glob(f"{in_path.stem}*.anchors.json") if p.is_file()):
        subroi = _anchor_subroi_from_name(input_stem=in_path.stem, anchor_json=path)
        if subroi is None:
            continue
        out.append(_AnchorJob(roi=str(roi), subroi=subroi, anchor_json=path))
    return out


def _list_obs_subrois(*, adata: ad.AnnData, roi_obs_key: str) -> list[str]:
    if roi_obs_key not in adata.obs.columns:
        raise ValueError(f"Missing obs column {roi_obs_key!r} required for subROI mode.")
    vals = adata.obs[roi_obs_key].astype(str).to_numpy()
    out = sorted({str(v).strip() for v in vals if str(v).strip() != "" and str(v).lower() not in {"nan", "none"}})
    return out


def _resolve_subroi_jobs(
    *,
    jobs: list[_AnchorJob],
    adata: ad.AnnData,
    roi_obs_key: str,
) -> tuple[list[_AnchorJob], list[str]]:
    if not jobs:
        return [], []

    subrois = _list_obs_subrois(adata=adata, roi_obs_key=roi_obs_key)
    by_subroi = {job.subroi: job for job in jobs if job.subroi != "(all)"}
    base = next((job for job in jobs if job.subroi == "(all)"), None)

    resolved: list[_AnchorJob] = []
    notes: list[str] = []
    roi = jobs[0].roi
    for subroi in subrois:
        explicit = by_subroi.get(subroi)
        if explicit is not None:
            resolved.append(explicit)
            continue
        if base is not None:
            resolved.append(_AnchorJob(roi=base.roi, subroi=subroi, anchor_json=base.anchor_json))
            continue
        notes.append(f"roi={roi} subroi={subroi}: no matching anchors JSON and no unsuffixed fallback anchors JSON.")
    return resolved, notes


def _extract_anchor_ids_from_payload(payload: dict[str, object]) -> list[str]:
    raw = payload.get("anchors")
    if isinstance(raw, list) and raw:
        out: list[str] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            cell_id = item.get("cell_id")
            if isinstance(cell_id, str) and cell_id != "":
                out.append(cell_id)
        if len(out) >= 2:
            return out

    start = payload.get("start")
    end = payload.get("end")
    out2: list[str] = []
    for item in (start, end):
        if not isinstance(item, dict):
            continue
        cell_id = item.get("cell_id")
        if isinstance(cell_id, str) and cell_id != "":
            out2.append(cell_id)
    return out2


def _subset_adata_for_subroi(*, adata: ad.AnnData, subroi: str, roi_obs_key: str) -> ad.AnnData:
    if subroi == "(all)":
        return adata
    if roi_obs_key not in adata.obs.columns:
        raise ValueError(f"Missing obs column {roi_obs_key!r} required for subROI {subroi!r}.")
    mask = adata.obs[roi_obs_key].astype(str) == str(subroi)
    n_keep = int(mask.sum())
    if n_keep == 0:
        raise ValueError(f"No cells in obs[{roi_obs_key!r}] == {subroi!r}.")
    return adata[mask].copy()


def _compute_signed_r_panel(
    *,
    adata: ad.AnnData,
    anchor_json: Path,
    roi: str,
    subroi: str,
    n_dense: int,
    smoothing: float,
    max_points: int,
    endpoint_extrapolation: float,
) -> _SignedRPanel:
    payload = json.loads(anchor_json.read_text())
    reverse_r_sign = payload.get("reverse_r_sign", False)
    if reverse_r_sign is None:
        reverse_r_sign = False
    if not isinstance(reverse_r_sign, bool):
        raise ValueError(f"Invalid reverse_r_sign in anchors JSON (expected bool): {anchor_json}")

    anchor_ids = _extract_anchor_ids_from_payload(payload)
    if len(anchor_ids) < 2:
        raise ValueError(f"Need at least 2 anchors in JSON: {anchor_json}")

    if "spatial" not in adata.obsm:
        raise ValueError("Missing adata.obsm['spatial'].")

    xy = np.asarray(adata.obsm["spatial"], dtype=float)
    if xy.ndim != 2 or xy.shape[1] < 2:
        raise ValueError(f"Unexpected spatial shape: {xy.shape}")
    xy = xy[:, :2]

    cell_ids = adata.obs_names.astype(str).to_numpy()
    cell_to_index = {cell_id: idx for idx, cell_id in enumerate(cell_ids)}
    missing = [cell_id for cell_id in anchor_ids if cell_id not in cell_to_index]
    if missing:
        preview = ", ".join(missing[:5])
        suffix = "..." if len(missing) > 5 else ""
        raise ValueError(f"Anchors missing in current view: {preview}{suffix}")

    anchor_idx = np.asarray([cell_to_index[cell_id] for cell_id in anchor_ids], dtype=int)
    anchor_xy = xy[anchor_idx, :]

    curve_xy = fit_anchor_curve(
        anchor_xy=anchor_xy,
        n_dense=int(n_dense),
        smoothing=float(smoothing),
    )
    _, r_signed, _ = project_to_polyline_arclength(
        xy=xy,
        line=curve_xy,
        k=50,
        endpoint_extrapolation=float(endpoint_extrapolation),
    )
    if reverse_r_sign:
        r_signed = -np.asarray(r_signed, dtype=float)

    rng = np.random.default_rng(0)
    plot_idx = np.arange(xy.shape[0])
    if plot_idx.size > int(max_points):
        plot_idx = rng.choice(plot_idx, size=int(max_points), replace=False)

    finite_abs = np.abs(r_signed[np.isfinite(r_signed)])
    lim = float(np.quantile(finite_abs, 0.99)) if finite_abs.size else 1.0
    if not np.isfinite(lim) or lim <= 0:
        lim = 1.0

    return _SignedRPanel(
        roi=str(roi),
        subroi=str(subroi),
        reverse_r_sign=bool(reverse_r_sign),
        points_xy=xy[plot_idx, :],
        points_r=np.asarray(r_signed[plot_idx], dtype=float),
        curve_xy=np.asarray(curve_xy, dtype=float),
        anchor_xy=np.asarray(anchor_xy, dtype=float),
        lim=float(lim),
    )


def _render_signed_r_review_png(
    *,
    panels: list[_SignedRPanel],
    missing_notes: list[str],
    output_png: Path,
    title: str,
    ncols: int,
) -> None:
    if not panels:
        raise click.ClickException("No signed-r panels available to render.")

    cols = max(1, int(ncols))
    rows = int(np.ceil(float(len(panels)) / float(cols)))
    fig, axs = plt.subplots(rows, cols, figsize=(5.0 * float(cols), 4.2 * float(rows)), squeeze=False)
    flat_axes = [ax for ax_row in axs for ax in ax_row]

    for idx, panel in enumerate(panels):
        ax = flat_axes[idx]
        sc = ax.scatter(
            panel.points_xy[:, 0],
            panel.points_xy[:, 1],
            c=panel.points_r,
            s=2,
            alpha=0.35,
            cmap="coolwarm",
            vmin=-panel.lim,
            vmax=panel.lim,
            linewidths=0,
            zorder=2,
        )
        ax.plot(panel.curve_xy[:, 0], panel.curve_xy[:, 1], color="black", linewidth=1.5, zorder=3)
        ax.scatter(
            panel.anchor_xy[:, 0],
            panel.anchor_xy[:, 1],
            c="yellow",
            s=30,
            edgecolors="black",
            linewidths=0.5,
            zorder=4,
        )
        ax.scatter(
            [panel.anchor_xy[0, 0]],
            [panel.anchor_xy[0, 1]],
            c="lime",
            s=70,
            edgecolors="black",
            linewidths=0.5,
            zorder=5,
        )
        ax.scatter(
            [panel.anchor_xy[-1, 0]],
            [panel.anchor_xy[-1, 1]],
            c="red",
            s=70,
            edgecolors="black",
            linewidths=0.5,
            zorder=5,
        )
        ax.set_title(f"roi={panel.roi} subroi={panel.subroi} reverse={panel.reverse_r_sign}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")
        fig.colorbar(sc, ax=ax, label="signed r")

    for idx in range(len(panels), len(flat_axes)):
        flat_axes[idx].axis("off")

    fig.suptitle(title)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.98))
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=200)
    plt.close(fig)

    if missing_notes:
        click.echo(f"Skipped/missing notes ({len(missing_notes)}):", err=True)
        for note in missing_notes:
            click.echo(f"  - {note}", err=True)


@click.command("princurve-signed-r-review")
@click.argument(
    "workspace",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True, resolve_path=True, path_type=Path),
)
@click.argument("rois", nargs=-1)
@click.option(
    "--output-png",
    default=None,
    type=click.Path(exists=False, file_okay=True, dir_okay=False, writable=True, resolve_path=True, path_type=Path),
    help="Output PNG path (default: <workspace>/analysis/output/ccf-transforms/princurve_signed_r_review.png).",
)
@click.option("--max-points", type=int, default=100_000, show_default=True, help="Max points per panel.")
@click.option("--n-dense", type=int, default=5_000, show_default=True, help="Dense spline samples for anchor curve.")
@click.option("--smoothing", type=float, default=0.5, show_default=True, help="Anchor curve smoothing.")
@click.option(
    "--endpoint-extrapolation",
    type=float,
    default=0.25,
    show_default=True,
    help="Endpoint extrapolation for signed-r projection (fraction of arclength domain).",
)
@click.option("--roi-obs-key", type=str, default="ccf_adjusted", show_default=True, help="SubROI obs column key.")
@click.option("--ncols", type=int, default=4, show_default=True, help="Number of columns in output grid.")
@click.option(
    "--include-all/--subroi-only",
    default=False,
    show_default=True,
    help="Include unsuffixed '(all)' anchors. Default is --subroi-only.",
)
@click.option(
    "--skip-missing/--no-skip-missing",
    default=True,
    show_default=True,
    help="Continue when per-ROI/subROI inputs are missing; otherwise fail fast.",
)
def main(
    workspace: Path,
    rois: tuple[str, ...],
    *,
    output_png: Path | None,
    max_points: int,
    n_dense: int,
    smoothing: float,
    endpoint_extrapolation: float,
    roi_obs_key: str,
    ncols: int,
    include_all: bool,
    skip_missing: bool,
) -> None:
    """Build a workspace PNG where each axis is one ROI/subROI signed-r review panel."""
    ws = Workspace(workspace)
    out_png = output_png if output_png is not None else (ws.output.ccf_transforms / "princurve_signed_r_review.png")
    resolved_rois = _resolve_review_rois(ws=ws, rois=rois)
    if not resolved_rois:
        raise click.ClickException("No ROIs resolved from workspace.")

    panels: list[_SignedRPanel] = []
    missing_notes: list[str] = []

    for roi in resolved_rois:
        roi_str = str(roi)
        try:
            in_path = _resolve_workspace_input_path(ws=ws, roi=roi_str)
        except FileNotFoundError as exc:
            msg = f"roi={roi_str}: {exc}"
            if skip_missing:
                missing_notes.append(msg)
                continue
            raise click.ClickException(msg) from exc

        adata = ad.read_h5ad(in_path)
        try:
            jobs = _discover_anchor_jobs(ws=ws, roi=roi_str)
            if not jobs:
                msg = f"roi={roi_str}: no anchors JSON files found for input stem {in_path.stem}."
                if skip_missing:
                    missing_notes.append(msg)
                    continue
                raise click.ClickException(msg)

            if not include_all:
                try:
                    jobs, resolve_notes = _resolve_subroi_jobs(jobs=jobs, adata=adata, roi_obs_key=roi_obs_key)
                except ValueError as exc:
                    msg = f"roi={roi_str}: {exc}"
                    if skip_missing:
                        missing_notes.append(msg)
                        continue
                    raise click.ClickException(msg) from exc
                missing_notes.extend(resolve_notes)

            if not jobs:
                mode_label = "anchors JSON files" if include_all else "subROI jobs resolved from obs values"
                msg = f"roi={roi_str}: no {mode_label} found for input stem {in_path.stem}."
                if skip_missing:
                    missing_notes.append(msg)
                    continue
                raise click.ClickException(msg)

            for job in jobs:
                try:
                    adata_job = _subset_adata_for_subroi(adata=adata, subroi=job.subroi, roi_obs_key=roi_obs_key)
                    panel = _compute_signed_r_panel(
                        adata=adata_job,
                        anchor_json=job.anchor_json,
                        roi=job.roi,
                        subroi=job.subroi,
                        n_dense=int(n_dense),
                        smoothing=float(smoothing),
                        max_points=int(max_points),
                        endpoint_extrapolation=float(endpoint_extrapolation),
                    )
                    panels.append(panel)
                except (KeyError, TypeError, ValueError, OSError, RuntimeError, SystemExit) as exc:
                    msg = f"roi={job.roi} subroi={job.subroi}: failed to build panel from {job.anchor_json.name}: {exc}"
                    if skip_missing:
                        missing_notes.append(msg)
                        continue
                    raise click.ClickException(msg) from exc
        finally:
            if getattr(adata, "isbacked", False) and getattr(adata, "file", None) is not None:
                adata.file.close()

    if not panels:
        raise click.ClickException("No signed-r panels were generated.")

    _render_signed_r_review_png(
        panels=panels,
        missing_notes=missing_notes,
        output_png=out_png,
        title=f"Princurve signed-r review | workspace={ws.path}",
        ncols=int(ncols),
    )
    click.echo(f"Wrote review PNG: {out_png}")
    if missing_notes:
        click.echo(f"Completed with {len(missing_notes)} skipped/missing notes.", err=True)
