#!/usr/bin/env python3
"""Plot transposed x-vs-r_um composition heatmaps from an h5ad.

Outputs four PNGs:
- <basename>_heatmap.png (unranked, count normalized by total cells in plot, turbo)
- <basename>_ranked_heatmap.png (ranked composition)
- <basename>_heatmap_faceted.png (unranked, faceted by dataset; each panel normalized by its total cells)
- <basename>_ranked_heatmap_faceted.png (ranked, faceted by dataset)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import colorcet as cc
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

THETA_MAX = 2.0 * np.pi
THETA_TICKS = [0, np.pi / 2, np.pi, 3 * np.pi / 2, THETA_MAX]
THETA_TICKLABELS = ["0", "pi/2", "pi", "3pi/2", "2pi"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot transposed x-vs-r_um composition heatmaps from h5ad.")
    parser.add_argument("h5ad", type=Path, help="Input h5ad (e.g. ~/nvme/wip.h5ad)")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for PNG files")
    parser.add_argument("--theta-col", type=str, default="tricycle", help="obs column for x-axis values")
    parser.add_argument("--r-col", type=str, default="r_um", help="obs column for radial coordinate in um")
    parser.add_argument("--dataset-col", type=str, default="dataset", help="obs column used to facet panels")
    parser.add_argument(
        "--facet-brdu-edu",
        action="store_true",
        help="Facet by 4-state combination of brdu_pos/edu_pos (none, brdu_only, edu_only, dual); ignores --dataset-col",
    )
    parser.add_argument("--brdu-col", type=str, default="brdu_pos", help="obs column for BrdU positivity (bool)")
    parser.add_argument("--edu-col", type=str, default="edu_pos", help="obs column for EdU positivity (bool)")
    parser.add_argument("--r-min", type=float, default=0.0, help="Minimum r_um to keep")
    parser.add_argument("--r-max", type=float, default=300.0, help="Upper bound for r_um (exclusive)")
    parser.add_argument("--n-r", type=int, default=140, help="Number of r bins")
    parser.add_argument("--n-theta", type=int, default=84, help="Number of theta bins")
    parser.add_argument(
        "--basename",
        type=str,
        default="theta_by_rum_lt300",
        help="Output basename; writes combined and faceted PNGs",
    )
    return parser.parse_args()


def _is_theta_axis(column_name: str) -> bool:
    return column_name in {"tricycle", "theta"}


def _brdu_edu_state(*, brdu_pos: np.ndarray, edu_pos: np.ndarray) -> np.ndarray:
    if brdu_pos.shape != edu_pos.shape:
        raise ValueError(f"brdu_pos and edu_pos must have the same shape; got {brdu_pos.shape} vs {edu_pos.shape}")
    out = np.empty(brdu_pos.shape, dtype=object)
    out[(~brdu_pos) & (~edu_pos)] = "none"
    out[brdu_pos & (~edu_pos)] = "brdu_only"
    out[(~brdu_pos) & edu_pos] = "edu_only"
    out[brdu_pos & edu_pos] = "dual"
    return out.astype(str, copy=False)


def load_filtered_obs(
    *,
    h5ad: Path,
    r_col: str,
    theta_col: str,
    dataset_col: str,
    r_min: float,
    r_max: float,
    wrap_theta: bool,
    facet_brdu_edu: bool,
    brdu_col: str,
    edu_col: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    adata = ad.read_h5ad(h5ad.expanduser(), backed="r")
    try:
        obs = adata.obs

        if r_col not in obs.columns:
            raise KeyError(f"Missing obs column: {r_col}")
        if theta_col not in obs.columns:
            raise KeyError(f"Missing obs column: {theta_col}")
        if facet_brdu_edu:
            if brdu_col not in obs.columns:
                raise KeyError(f"Missing obs column: {brdu_col}")
            if edu_col not in obs.columns:
                raise KeyError(f"Missing obs column: {edu_col}")
        else:
            if dataset_col not in obs.columns:
                raise KeyError(f"Missing obs column: {dataset_col}")

        r_um = np.asarray(obs[r_col], dtype=np.float64)
        theta = np.asarray(obs[theta_col], dtype=np.float64)
        if wrap_theta:
            theta = theta % THETA_MAX
        if facet_brdu_edu:
            brdu_pos = np.asarray(obs[brdu_col], dtype=bool)
            edu_pos = np.asarray(obs[edu_col], dtype=bool)
            dataset = _brdu_edu_state(brdu_pos=brdu_pos, edu_pos=edu_pos)
        else:
            dataset = np.asarray(obs[dataset_col], dtype=str)

        keep = np.isfinite(r_um) & np.isfinite(theta) & (r_um >= r_min) & (r_um < r_max)
        r_um = r_um[keep]
        theta = theta[keep]
        dataset = dataset[keep]
    finally:
        if adata.file is not None:
            adata.file.close()

    if r_um.size == 0:
        raise ValueError(f"No cells remain after filtering with {r_col} in [{r_min}, {r_max}).")

    return r_um, theta, dataset


def theta_binned_counts(
    *, r_um: np.ndarray, theta: np.ndarray, r_min: float, r_max: float, n_r: int, n_theta: int, theta_min: float, theta_max: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r_edges = np.linspace(r_min, r_max, n_r + 1)
    th_edges = np.linspace(theta_min, theta_max, n_theta + 1)
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
    th_centers = 0.5 * (th_edges[:-1] + th_edges[1:])

    counts, _, _ = np.histogram2d(r_um, theta, bins=(r_edges, th_edges))
    counts = counts.astype(np.float64, copy=False)

    theta_counts = counts.sum(axis=0)
    r_weighted = (counts * r_centers[:, None]).sum(axis=0)
    r_mean_by_theta = np.divide(
        r_weighted,
        theta_counts,
        out=np.full_like(theta_counts, np.nan, dtype=np.float64),
        where=theta_counts > 0,
    )

    return counts, r_edges, th_edges, th_centers, r_mean_by_theta


def _facet_grid(n_panels: int) -> tuple[int, int, tuple[float, float]]:
    if n_panels <= 0:
        raise ValueError("n_panels must be > 0")
    if n_panels == 1:
        return 1, 1, (12.0, 7.5)
    ncols = n_panels if n_panels <= 3 else 4
    nrows = int(np.ceil(n_panels / ncols))
    return nrows, ncols, (4.0 * ncols, 3.0 * nrows)


def _draw_ranked_rows(ax: mpl.axes.Axes, *, counts: np.ndarray, r_edges: np.ndarray, theta_colors: np.ndarray) -> None:
    n_r, n_theta = counts.shape
    row_sums = counts.sum(axis=1)
    prop = np.divide(counts, row_sums[:, None], out=np.zeros_like(counts), where=row_sums[:, None] > 0)

    for i in range(n_r):
        if row_sums[i] <= 0:
            continue
        y0 = r_edges[i]
        y1 = r_edges[i + 1]
        x0 = 0.0
        for k in range(n_theta):
            pk = prop[i, k]
            if pk <= 0:
                continue
            x1 = x0 + pk
            ax.fill([x0, x1, x1, x0], [y0, y0, y1, y1], color=theta_colors[k], linewidth=0)
            x0 = x1
            if x0 >= 0.999999:
                break


def plot_unranked_transposed(
    *,
    out_png: Path,
    counts: np.ndarray,
    r_edges: np.ndarray,
    th_edges: np.ndarray,
    n_cells: int,
    r_label: str,
    x_label: str,
    x_ticks: list[float] | None,
    x_ticklabels: list[str] | None,
) -> None:
    fig = plt.figure(figsize=(12, 7.5), dpi=200, constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[5, 1])
    ax = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])

    z = counts / float(n_cells)
    mesh = ax.pcolormesh(th_edges, r_edges, z, cmap="turbo", shading="auto")
    ax.set_xlabel(x_label)
    ax.set_ylabel(r_label)
    if x_ticks is not None:
        ax.set_xticks(x_ticks)
        if x_ticklabels is not None:
            ax.set_xticklabels(x_ticklabels)
    ax.set_title(f"Cell composition: {r_label} vs {x_label} (n={n_cells})")

    cbar = fig.colorbar(mesh, cax=cax)
    cbar.set_label("fraction of cells in plot")
    fig.savefig(out_png)
    plt.close(fig)


def plot_unranked_transposed_by_dataset(
    *,
    out_png: Path,
    r_edges: np.ndarray,
    th_edges: np.ndarray,
    per_dataset: list[tuple[str, np.ndarray, int]],
    n_cells_total: int,
    frac_vmax: float,
    r_label: str,
    x_label: str,
    x_ticks: list[float] | None,
    x_ticklabels: list[str] | None,
) -> None:
    nrows, ncols, figsize = _facet_grid(len(per_dataset))
    fig, axs = plt.subplots(nrows, ncols, figsize=figsize, dpi=200, sharex=True, sharey=True, constrained_layout=True)
    axs = np.asarray(axs, dtype=object)
    if axs.ndim == 0:
        axs = axs.reshape(1, 1)
    elif axs.ndim == 1:
        axs = axs.reshape(nrows, ncols)

    mesh = None
    for ax, (dataset, counts, n_cells) in zip(axs.flat, per_dataset):
        z = counts / float(n_cells)
        mesh = ax.pcolormesh(th_edges, r_edges, z, cmap="turbo", shading="auto", vmin=0.0, vmax=frac_vmax)
        ax.set_title(f"{dataset} (n={n_cells})")
        ax.set_xlabel(x_label)
        ax.set_ylabel(r_label)
        if x_ticks is not None:
            ax.set_xticks(x_ticks)
            if x_ticklabels is not None:
                ax.set_xticklabels(x_ticklabels)

    for ax in axs.flat[len(per_dataset):]:
        ax.set_visible(False)

    if mesh is None:
        raise ValueError("No datasets to plot")

    visible_axes = [ax for ax in axs.flat if ax.get_visible()]
    for ax in visible_axes:
        ax.label_outer()

    cbar = fig.colorbar(mesh, ax=visible_axes, pad=0.01)
    cbar.set_label("fraction of cells in plot")
    fig.suptitle(f"Cell composition: {r_label} vs {x_label} by dataset (n={n_cells_total})")
    fig.savefig(out_png)
    plt.close(fig)


def plot_ranked_transposed_by_dataset(
    *,
    out_png: Path,
    r_edges: np.ndarray,
    th_centers: np.ndarray,
    per_dataset: list[tuple[str, np.ndarray, int]],
    n_cells_total: int,
    r_label: str,
    x_label: str,
    theta_mode: bool,
) -> None:
    if theta_mode:
        cmap_theta = cc.cm["colorwheel"]
        norm_theta = mpl.colors.Normalize(vmin=0.0, vmax=THETA_MAX)
        theta_colors = cmap_theta((th_centers % THETA_MAX) / THETA_MAX)
        cbar_label = "theta (radians)"
        cbar_ticks = THETA_TICKS
        cbar_ticklabels = THETA_TICKLABELS
    else:
        cmap_theta = mpl.colormaps["viridis"]
        norm_theta = mpl.colors.Normalize(vmin=float(np.min(th_centers)), vmax=float(np.max(th_centers)))
        theta_colors = cmap_theta(norm_theta(th_centers))
        cbar_label = x_label
        cbar_ticks = None
        cbar_ticklabels = None

    nrows, ncols, figsize = _facet_grid(len(per_dataset))
    fig, axs = plt.subplots(nrows, ncols, figsize=figsize, dpi=200, sharex=True, sharey=True, constrained_layout=True)
    axs = np.asarray(axs, dtype=object)
    if axs.ndim == 0:
        axs = axs.reshape(1, 1)
    elif axs.ndim == 1:
        axs = axs.reshape(nrows, ncols)

    for ax, (dataset, counts, n_cells) in zip(axs.flat, per_dataset):
        _draw_ranked_rows(ax, counts=counts, r_edges=r_edges, theta_colors=theta_colors)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(float(r_edges[0]), float(r_edges[-1]))
        ax.set_xlabel(f"proportion ({x_label} bins, sorted)")
        ax.set_ylabel(r_label)
        ax.set_title(f"{dataset} (n={n_cells})")

    for ax in axs.flat[len(per_dataset):]:
        ax.set_visible(False)

    visible_axes = [ax for ax in axs.flat if ax.get_visible()]
    for ax in visible_axes:
        ax.label_outer()

    sm = mpl.cm.ScalarMappable(norm=norm_theta, cmap=cmap_theta)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=visible_axes, pad=0.01)
    cbar.set_label(cbar_label)
    if cbar_ticks is not None:
        cbar.set_ticks(cbar_ticks)
        if cbar_ticklabels is not None:
            cbar.set_ticklabels(cbar_ticklabels)

    fig.suptitle(f"Composition (sorted {x_label}) vs {r_label} by dataset (n={n_cells_total})")
    fig.savefig(out_png)
    plt.close(fig)


def plot_ranked_transposed(
    *,
    out_png: Path,
    counts: np.ndarray,
    r_edges: np.ndarray,
    th_centers: np.ndarray,
    n_cells: int,
    r_label: str,
    x_label: str,
    theta_mode: bool,
) -> None:
    if theta_mode:
        cmap_theta = cc.cm["colorwheel"]
        norm_theta = mpl.colors.Normalize(vmin=0.0, vmax=THETA_MAX)
        theta_colors = cmap_theta((th_centers % THETA_MAX) / THETA_MAX)
        cbar_label = "theta (radians)"
        cbar_ticks = THETA_TICKS
        cbar_ticklabels = THETA_TICKLABELS
    else:
        cmap_theta = mpl.colormaps["viridis"]
        norm_theta = mpl.colors.Normalize(vmin=float(np.min(th_centers)), vmax=float(np.max(th_centers)))
        theta_colors = cmap_theta(norm_theta(th_centers))
        cbar_label = x_label
        cbar_ticks = None
        cbar_ticklabels = None

    fig = plt.figure(figsize=(12, 7.5), dpi=200, constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[5, 1])
    ax = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])

    _draw_ranked_rows(ax, counts=counts, r_edges=r_edges, theta_colors=theta_colors)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(float(r_edges[0]), float(r_edges[-1]))
    ax.set_xlabel(f"proportion ({x_label} bins, sorted)")
    ax.set_ylabel(r_label)
    ax.set_title(f"Composition (sorted {x_label}) vs {r_label} (n={n_cells})")

    sm = mpl.cm.ScalarMappable(norm=norm_theta, cmap=cmap_theta)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label(cbar_label)
    if cbar_ticks is not None:
        cbar.set_ticks(cbar_ticks)
        if cbar_ticklabels is not None:
            cbar.set_ticklabels(cbar_ticklabels)

    fig.savefig(out_png)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    theta_mode = _is_theta_axis(args.theta_col)
    x_label = "theta (radians)" if theta_mode else args.theta_col

    out_dir = args.out_dir.expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    r_um, theta, dataset = load_filtered_obs(
        h5ad=args.h5ad,
        r_col=args.r_col,
        theta_col=args.theta_col,
        dataset_col=args.dataset_col,
        r_min=args.r_min,
        r_max=args.r_max,
        wrap_theta=theta_mode,
        facet_brdu_edu=args.facet_brdu_edu,
        brdu_col=args.brdu_col,
        edu_col=args.edu_col,
    )
    if theta_mode:
        theta_min = 0.0
        theta_max = THETA_MAX
        x_ticks = THETA_TICKS
        x_ticklabels = THETA_TICKLABELS
    else:
        theta_min = float(np.min(theta))
        theta_max = float(np.max(theta))
        if not np.isfinite(theta_min) or not np.isfinite(theta_max):
            raise ValueError(f"{args.theta_col} contains non-finite values after filtering")
        if theta_max <= theta_min:
            theta_max = theta_min + 1e-6
        x_ticks = None
        x_ticklabels = None

    datasets = np.unique(dataset)
    per_dataset: list[tuple[str, np.ndarray, int]] = []
    counts_total: np.ndarray | None = None
    r_edges = None
    th_edges = None
    th_centers = None
    max_fraction = 0.0

    for ds in datasets:
        mask = dataset == ds
        n_cells = int(mask.sum())
        counts, r_edges_i, th_edges_i, th_centers_i, _ = theta_binned_counts(
            r_um=r_um[mask],
            theta=theta[mask],
            r_min=args.r_min,
            r_max=args.r_max,
            n_r=args.n_r,
            n_theta=args.n_theta,
            theta_min=theta_min,
            theta_max=theta_max,
        )

        if r_edges is None:
            r_edges = r_edges_i
            th_edges = th_edges_i
            th_centers = th_centers_i

        if counts_total is None:
            counts_total = np.zeros_like(counts)
        counts_total += counts
        per_dataset.append((ds, counts, n_cells))
        max_fraction = max(max_fraction, float(counts.max() / float(n_cells)))

    if r_edges is None or th_edges is None or th_centers is None or counts_total is None:
        raise ValueError("No datasets to plot")

    max_fraction = max(max_fraction, float(counts_total.max() / float(r_um.size)))
    heatmap_png = out_dir / f"{args.basename}_heatmap.png"
    heatmap_faceted_png = out_dir / f"{args.basename}_heatmap_faceted.png"
    ranked_png = out_dir / f"{args.basename}_ranked_heatmap.png"
    ranked_faceted_png = out_dir / f"{args.basename}_ranked_heatmap_faceted.png"

    r_label = f"{args.r_col} ({args.r_min:g}-{args.r_max:g})"
    plot_unranked_transposed(
        out_png=heatmap_png,
        counts=counts_total,
        r_edges=r_edges,
        th_edges=th_edges,
        n_cells=r_um.size,
        r_label=r_label,
        x_label=x_label,
        x_ticks=x_ticks,
        x_ticklabels=x_ticklabels,
    )
    plot_unranked_transposed_by_dataset(
        out_png=heatmap_faceted_png,
        r_edges=r_edges,
        th_edges=th_edges,
        per_dataset=per_dataset,
        n_cells_total=r_um.size,
        frac_vmax=max_fraction,
        r_label=r_label,
        x_label=x_label,
        x_ticks=x_ticks,
        x_ticklabels=x_ticklabels,
    )
    plot_ranked_transposed(
        out_png=ranked_png,
        counts=counts_total,
        r_edges=r_edges,
        th_centers=th_centers,
        n_cells=r_um.size,
        r_label=r_label,
        x_label=x_label,
        theta_mode=theta_mode,
    )
    plot_ranked_transposed_by_dataset(
        out_png=ranked_faceted_png,
        r_edges=r_edges,
        th_centers=th_centers,
        per_dataset=per_dataset,
        n_cells_total=r_um.size,
        r_label=r_label,
        x_label=x_label,
        theta_mode=theta_mode,
    )

    print(f"n_kept={r_um.size}")
    print(heatmap_png)
    print(heatmap_faceted_png)
    print(ranked_png)
    print(ranked_faceted_png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
