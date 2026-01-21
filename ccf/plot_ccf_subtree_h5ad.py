from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import click
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk

from fishtools.ccf.ontology import mask_ccf_subtree


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


def _load_fixed_paths(metrics_json: Path | None, fixed_nifti: Path | None, mask_nifti: Path | None) -> tuple[Path, Path]:
    if fixed_nifti is not None and mask_nifti is not None:
        return (fixed_nifti, mask_nifti)

    if metrics_json is None:
        raise click.ClickException("Provide either --metrics-json, or both --fixed-nifti and --mask-nifti.")

    metrics = json.loads(metrics_json.read_text(encoding="utf-8"))
    if "fixed_mask_nifti" not in metrics:
        raise click.ClickException(f"metrics_json is missing fixed_mask_nifti: {metrics_json}")
    if "summary_json" not in metrics:
        raise click.ClickException(f"metrics_json is missing summary_json: {metrics_json}")

    mask_path = Path(str(metrics["fixed_mask_nifti"]))
    summary_path = Path(str(metrics["summary_json"]))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    run_dir = summary_path.parent

    fixed_path = run_dir / "fixed_atlas_crop.nii.gz"
    paths = summary.get("paths")
    if isinstance(paths, dict) and "fixed_nifti" in paths:
        fixed_path = Path(str(paths["fixed_nifti"]))

    return (fixed_path, mask_path)


def _mask_inside_fraction(coords_xy: np.ndarray, mask_yx: np.ndarray) -> float:
    coords_xy = np.asarray(coords_xy, dtype=np.float32)
    if coords_xy.ndim != 2 or coords_xy.shape[1] != 2:
        raise ValueError(f"Expected coords to have shape (N,2), got {coords_xy.shape}.")

    h, w = mask_yx.shape
    x = coords_xy[:, 0]
    y = coords_xy[:, 1]
    xi = np.round(x).astype(np.int64, copy=False)
    yi = np.round(y).astype(np.int64, copy=False)
    in_bounds = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
    inside = np.zeros(coords_xy.shape[0], dtype=bool)
    inside[in_bounds] = mask_yx[yi[in_bounds], xi[in_bounds]]
    return float(inside.mean())


@click.command()
@click.argument(
    "annot_h5ad",
    type=click.Path(exists=True, dir_okay=False, readable=True, resolve_path=True, path_type=Path),
)
@click.argument("term", type=str)
@click.argument(
    "output_png",
    type=click.Path(exists=False, dir_okay=False, writable=True, resolve_path=True, path_type=Path),
)
@click.option("--coords-key", default="spatial_ccf", show_default=True, help="obsm key for coords to plot.")
@click.option("--ccf-obsm-key", default="ccf", show_default=True, help="obsm key with CCF annotation columns.")
@click.option(
    "--metrics-json",
    type=click.Path(exists=True, dir_okay=False, readable=True, resolve_path=True, path_type=Path),
    default=None,
    show_default=False,
    help="Optional metrics JSON from ccf/ants_warp_h5ad_spatial.py (used to locate atlas image + brain mask).",
)
@click.option(
    "--fixed-nifti",
    type=click.Path(exists=True, dir_okay=False, readable=True, resolve_path=True, path_type=Path),
    default=None,
    show_default=False,
    help="Optional fixed atlas crop NIfTI (2D) to use as background image.",
)
@click.option(
    "--mask-nifti",
    type=click.Path(exists=True, dir_okay=False, readable=True, resolve_path=True, path_type=Path),
    default=None,
    show_default=False,
    help="Optional fixed brain mask crop NIfTI (2D) to use as contour and inside-mask metric.",
)
@click.option(
    "--also-term",
    "also_terms",
    multiple=True,
    default=(),
    show_default=False,
    help="Additional ontology term(s) to plot in separate panels (includes descendants).",
)
@click.option("--max-points", type=int, default=200_000, show_default=True, help="Max points to plot per panel.")
@click.option("--random-seed", type=int, default=0, show_default=True)
def main(
    annot_h5ad: Path,
    term: str,
    output_png: Path,
    coords_key: str,
    ccf_obsm_key: str,
    metrics_json: Path | None,
    fixed_nifti: Path | None,
    mask_nifti: Path | None,
    also_terms: tuple[str, ...],
    max_points: int,
    random_seed: int,
) -> None:
    adata = ad.read_h5ad(annot_h5ad)
    if coords_key not in adata.obsm:
        raise click.ClickException(f"Missing adata.obsm[{coords_key!r}] in {annot_h5ad}")
    if ccf_obsm_key not in adata.obsm:
        raise click.ClickException(f"Missing adata.obsm[{ccf_obsm_key!r}] in {annot_h5ad}")

    coords = np.asarray(adata.obsm[coords_key], dtype=np.float32)
    terms = (str(term),) + tuple(str(t) for t in also_terms)
    masks = [mask_ccf_subtree(adata, t, obsm_key=ccf_obsm_key) for t in terms]

    fixed_path, mask_path = _load_fixed_paths(metrics_json=metrics_json, fixed_nifti=fixed_nifti, mask_nifti=mask_nifti)
    fixed_img = sitk.GetArrayFromImage(sitk.ReadImage(str(fixed_path))).astype(np.float32)
    fixed_mask = sitk.GetArrayFromImage(sitk.ReadImage(str(mask_path))) > 0
    if fixed_img.ndim != 2:
        raise click.ClickException(f"Expected fixed image to be 2D, got shape={fixed_img.shape} from {fixed_path}")
    if fixed_mask.ndim != 2:
        raise click.ClickException(f"Expected fixed mask to be 2D, got shape={fixed_mask.shape} from {mask_path}")
    if fixed_img.shape != fixed_mask.shape:
        raise click.ClickException(f"fixed image shape={fixed_img.shape} != fixed mask shape={fixed_mask.shape}")

    inside_all = _mask_inside_fraction(coords_xy=coords, mask_yx=fixed_mask)
    inside_terms = [
        _mask_inside_fraction(coords_xy=coords[m], mask_yx=fixed_mask) if m.any() else float("nan") for m in masks
    ]

    rng = np.random.default_rng(int(random_seed))
    n = int(coords.shape[0])
    k = min(int(max_points), n)
    idx = rng.choice(n, size=k, replace=False) if k < n else np.arange(n)
    coords_plot = coords[idx]
    masks_plot = [m[idx] for m in masks]

    bg = _normalize_robust(fixed_img)
    h, w = bg.shape

    ncols = 1 + len(terms)
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 7), constrained_layout=True)
    if ncols == 1:
        axes = np.asarray([axes])
    for ax in axes:
        ax.imshow(bg, cmap="gray", interpolation="nearest")
        ax.contour(fixed_mask.astype(np.uint8), levels=[0.5], colors="tab:red", linewidths=0.6)
        ax.set_aspect("equal")
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
        ax.axis("off")

    axes[0].scatter(
        coords_plot[:, 0],
        coords_plot[:, 1],
        s=0.25,
        alpha=0.12,
        linewidths=0,
        color="tab:blue",
        rasterized=True,
    )
    axes[0].set_title(f"All cells (n={n:,})\\ninside mask={inside_all*100:.1f}%")

    colors = ["tab:green", "tab:orange", "tab:purple", "tab:cyan", "tab:pink", "tab:brown"]
    for j, (t, m, inside_t) in enumerate(zip(terms, masks_plot, inside_terms, strict=True), start=1):
        color = colors[(j - 1) % len(colors)]
        axes[j].scatter(
            coords_plot[~m, 0],
            coords_plot[~m, 1],
            s=0.15,
            alpha=0.04,
            linewidths=0,
            color="gray",
            rasterized=True,
        )
        axes[j].scatter(
            coords_plot[m, 0],
            coords_plot[m, 1],
            s=0.35,
            alpha=0.25,
            linewidths=0,
            color=color,
            rasterized=True,
        )
        axes[j].set_title(f"{t} subtree (n={int(masks[j - 1].sum()):,})\\ninside mask={inside_t*100:.1f}%")

    ccf_meta = adata.uns.get("ccf", {})
    atlas_name = ccf_meta.get("atlas_name", "unknown")
    atlas_plane = ccf_meta.get("atlas_plane", "unknown")
    atlas_slice = ccf_meta.get("atlas_slice_idx", "unknown")
    fig.suptitle(f"{annot_h5ad.name} | {atlas_name} | {atlas_plane} slice={atlas_slice}")

    fig.savefig(output_png, dpi=200)
    plt.close(fig)
    click.echo(f"Wrote plot: {output_png}")


if __name__ == "__main__":
    main()
