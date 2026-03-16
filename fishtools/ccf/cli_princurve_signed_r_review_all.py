from __future__ import annotations

from pathlib import Path

import rich_click as click

from fishtools.ccf import cli_princurve_signed_r_review as review
from fishtools.io.workspace import Workspace


def _render_signed_r_review_png_fixed_grid(
    *,
    panels: list[review._SignedRPanel],
    missing_notes: list[str],
    output_png: Path,
    title: str,
    nrows: int,
    ncols: int,
) -> None:
    if not panels:
        raise click.ClickException("No signed-r panels available to render.")
    if nrows <= 0 or ncols <= 0:
        raise click.ClickException("nrows and ncols must be positive integers.")

    max_panels = int(nrows) * int(ncols)
    if len(panels) > max_panels:
        raise click.ClickException(
            f"Got {len(panels)} panels but grid capacity is {max_panels} ({nrows}x{ncols}). "
            "Increase grid size or reduce inputs."
        )

    fig, axs = review.plt.subplots(
        int(nrows),
        int(ncols),
        figsize=(5.0 * float(ncols), 4.2 * float(nrows)),
        squeeze=False,
    )
    flat_axes = [ax for row in axs for ax in row]

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
    review.plt.close(fig)

    if missing_notes:
        click.echo(f"Skipped/missing notes ({len(missing_notes)}):", err=True)
        for note in missing_notes:
            click.echo(f"  - {note}", err=True)


@click.command("princurve-signed-r-review-all")
@click.argument(
    "workspaces",
    nargs=-1,
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True, resolve_path=True, path_type=Path),
)
@click.option(
    "--output-png",
    required=True,
    type=click.Path(exists=False, file_okay=True, dir_okay=False, writable=True, resolve_path=True, path_type=Path),
    help="Output PNG path for the combined multi-dataset review.",
)
@click.option("--roi", "rois", multiple=True, help="Optional ROI filter repeated across all workspaces.")
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
@click.option("--nrows", type=int, default=10, show_default=True, help="Grid rows.")
@click.option("--ncols", type=int, default=10, show_default=True, help="Grid columns.")
def main(
    workspaces: tuple[Path, ...],
    *,
    output_png: Path,
    rois: tuple[str, ...],
    max_points: int,
    n_dense: int,
    smoothing: float,
    endpoint_extrapolation: float,
    roi_obs_key: str,
    include_all: bool,
    skip_missing: bool,
    nrows: int,
    ncols: int,
) -> None:
    """Build one fixed-grid signed-r review PNG across multiple workspaces."""
    if not workspaces:
        raise click.ClickException("Provide at least one workspace path.")

    all_panels: list[review._SignedRPanel] = []
    all_notes: list[str] = []
    for workspace_path in workspaces:
        ws = Workspace(Path(workspace_path))
        panels, notes = review._collect_signed_r_panels_for_workspace(
            ws=ws,
            rois=rois,
            max_points=int(max_points),
            n_dense=int(n_dense),
            smoothing=float(smoothing),
            endpoint_extrapolation=float(endpoint_extrapolation),
            roi_obs_key=str(roi_obs_key),
            include_all=bool(include_all),
            skip_missing=bool(skip_missing),
            roi_label_prefix=ws.path.name,
        )
        all_panels.extend(panels)
        all_notes.extend(notes)

    if not all_panels:
        raise click.ClickException("No signed-r panels were generated from provided workspaces.")

    _render_signed_r_review_png_fixed_grid(
        panels=all_panels,
        missing_notes=all_notes,
        output_png=output_png,
        title=f"Princurve signed-r review | datasets={len(workspaces)}",
        nrows=int(nrows),
        ncols=int(ncols),
    )
    click.echo(f"Wrote review PNG: {output_png}")
    if all_notes:
        click.echo(f"Completed with {len(all_notes)} skipped/missing notes.", err=True)
