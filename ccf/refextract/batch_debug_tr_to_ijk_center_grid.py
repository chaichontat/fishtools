from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from ccf.refextract import debug_tr_to_ijk_plot as dbg
from ccf.refextract.midsurface_coords import (
    MIDLINE_INSLICE_ANCHOR_DT,
    _reject_midline_anchor_outliers,
    evaluate_midline_normal_bundle,
    load_coronal_midline_columns,
    load_sagittal_midline_columns,
    nearest_midline_slice_key,
)

AXIS = Literal["coronal", "sagittal"]


@dataclass(frozen=True)
class LutCache:
    axis: AXIS
    source_slice_keys: np.ndarray
    t_grid: np.ndarray
    low_ijk: np.ndarray
    high_ijk: np.ndarray


def _load_lut_cache(*, outdir: Path, axis: AXIS) -> LutCache:
    path = outdir / ("chart_map_coronal_ijk_from_tr.npz" if axis == "coronal" else "chart_map_sagittal_ijk_from_tr.npz")
    d = np.load(path)
    try:
        t_domain = str(np.asarray(d["t_domain"]).reshape(-1)[0])
    except (KeyError, IndexError, ValueError):
        t_domain = ""
    if t_domain != "t_all":
        raise ValueError(f"{path.name} has t_domain={t_domain!r}; expected 't_all'.")
    return LutCache(
        axis=axis,
        source_slice_keys=np.asarray(d["source_slice_keys"], dtype=np.int32).reshape(-1),
        t_grid=np.asarray(d["t_grid"], dtype=np.float64).reshape(-1),
        low_ijk=np.asarray(d["low_ijk"], dtype=np.float64),
        high_ijk=np.asarray(d["high_ijk"], dtype=np.float64),
    )


def _sample_rows(rng: np.random.Generator, arr: np.ndarray, n: int) -> np.ndarray:
    x = np.asarray(arr)
    if x.ndim != 1:
        raise ValueError(f"Expected 1D array, got shape={x.shape}.")
    if x.size <= n:
        return x
    idx = rng.choice(np.arange(x.size), size=int(n), replace=False)
    return x[idx]


def _render_center_panel(
    *,
    ax,
    h5ad_path: Path,
    lut_coronal: LutCache,
    lut_sagittal: LutCache,
    midline_coronal: dict[int, object],
    midline_sagittal: dict[int, object],
    mapping_mode: str,
    interp: str,
    smooth_mid: float,
    smooth_iso: float,
    n_t: int,
    r_bins: int,
    r_roll_half_window: int,
    r_source_um_per_px: float,
    r_target_um_per_px: float,
    r_iso_fracs: np.ndarray,
    max_points_floor: int,
    max_points_scatter: int,
) -> str | None:
    import anndata as ad

    axis, atlas_slice_idx = dbg._read_axis_and_slice_from_p1(h5ad_path=h5ad_path)
    lut = lut_coronal if axis == "coronal" else lut_sagittal
    midline_by_slice = midline_coronal if axis == "coronal" else midline_sagittal
    if mapping_mode == "lut":
        row, used_slice = dbg._nearest_slice_row(
            source_slice_keys=lut.source_slice_keys, atlas_slice_idx=int(atlas_slice_idx)
        )
        t_grid = lut.t_grid
        lo_row = np.asarray(lut.low_ijk[row], dtype=np.float64)
        hi_row = np.asarray(lut.high_ijk[row], dtype=np.float64)
        valid = np.isfinite(lo_row).all(axis=1) & np.isfinite(hi_row).all(axis=1) & np.isfinite(t_grid)
        if not np.any(valid):
            return f"no LUT samples (axis={axis} slice={used_slice})"
        t_sup = t_grid[valid]
        lo_sup = lo_row[valid]
        hi_sup = hi_row[valid]
        t_support_lo = float(np.min(t_sup))
        t_support_hi = float(np.max(t_sup))
        if int(used_slice) in midline_by_slice:
            cols_mid = midline_by_slice[int(used_slice)]
        else:
            nearest_mid = nearest_midline_slice_key(columns_by_slice=midline_by_slice, atlas_slice_idx=int(used_slice))
            cols_mid = midline_by_slice[int(nearest_mid)]
    elif mapping_mode == "midline-normal":
        used_slice = nearest_midline_slice_key(columns_by_slice=midline_by_slice, atlas_slice_idx=int(atlas_slice_idx))
        cols_mid = midline_by_slice[int(used_slice)]
        t_cols = np.asarray(getattr(cols_mid, "t"), dtype=np.float64).reshape(-1)
        finite_t_cols = np.isfinite(t_cols)
        if int(np.count_nonzero(finite_t_cols)) < 2:
            return f"insufficient finite midline t support (axis={axis} slice={used_slice})"
        t_support_lo = float(np.min(t_cols[finite_t_cols]))
        t_support_hi = float(np.max(t_cols[finite_t_cols]))
        t_sup = None
        lo_sup = None
        hi_sup = None
    else:
        return f"unsupported mapping_mode={mapping_mode!r}"

    adata = ad.read_h5ad(str(h5ad_path), backed="r")
    try:
        if "t_all" not in adata.obs.columns:
            return "missing obs[t_all]"
        if "principal_r_signed" not in adata.obsm:
            return "missing obsm[principal_r_signed]"
        t_lookup_all = np.asarray(adata.obs["t_all"].to_numpy(dtype=np.float64, copy=False), dtype=np.float64).reshape(-1)
        r_signed_all = np.asarray(adata.obsm["principal_r_signed"], dtype=np.float64).reshape(-1)
    finally:
        if getattr(adata, "isbacked", False) and getattr(adata, "file", None) is not None:
            adata.file.close()

    finite = np.isfinite(t_lookup_all) & np.isfinite(r_signed_all)
    t_lookup = t_lookup_all[finite]
    r_signed = r_signed_all[finite]
    if t_lookup.size == 0:
        return "no finite (t_all,r_signed)"

    # Deterministic per-panel sampling for speed.
    seed = int(abs(hash(str(h5ad_path))) % (2**32 - 1))
    rng = np.random.default_rng(seed)
    if t_lookup.size > int(max_points_floor):
        idx = rng.choice(np.arange(t_lookup.size), size=int(max_points_floor), replace=False)
        t_floor = t_lookup[idx]
        r_floor = r_signed[idx]
    else:
        t_floor = t_lookup
        r_floor = r_signed

    t_lookup_clip, _t_bins, _r_floor_bin, _r_ceil_bin, (r_floor_at_t, _r_ceil_at_t) = dbg._rolling_floor_ceil(
        t_lookup=t_floor,
        r_signed=r_floor,
        n_bins=int(r_bins),
        roll_half_window=int(r_roll_half_window),
    )
    r_um = (r_floor - r_floor_at_t) * float(r_source_um_per_px)
    r_um = np.clip(np.asarray(r_um, dtype=np.float64), 0.0, np.inf)
    r_hi = float(np.nanpercentile(r_um[np.isfinite(r_um)], 99.0))
    if not np.isfinite(r_hi) or r_hi <= 1.0e-12:
        return "invalid r_hi"

    r_iso_um = (np.asarray(r_iso_fracs, dtype=np.float64).reshape(-1) * r_hi).astype(np.float64, copy=False)

    t_mid = np.asarray(getattr(cols_mid, "t"), dtype=np.float64).reshape(-1)
    y_mid = np.asarray(getattr(cols_mid, "y"), dtype=np.float64).reshape(-1)
    x_mid = np.asarray(getattr(cols_mid, "x"), dtype=np.float64).reshape(-1)
    finite_mid = np.isfinite(t_mid) & np.isfinite(y_mid) & np.isfinite(x_mid)
    if int(np.count_nonzero(finite_mid)) < 2:
        return f"invalid midline columns (axis={axis} slice={used_slice})"
    t_mid_f = t_mid[finite_mid]
    y_mid_f = y_mid[finite_mid]
    x_mid_f = x_mid[finite_mid]
    order_mid = np.argsort(t_mid_f)
    t_mid_f = t_mid_f[order_mid]
    y_mid_f = y_mid_f[order_mid]
    x_mid_f = x_mid_f[order_mid]
    t_mid_u, keep_mid = np.unique(t_mid_f, return_index=True)
    y_mid_u = y_mid_f[keep_mid]
    x_mid_u = x_mid_f[keep_mid]
    if t_mid_u.size < 2:
        return f"invalid unique midline t support (axis={axis} slice={used_slice})"

    # Dense t grid for rendering curves.
    n_t_plot = int(max(64, n_t))
    t_plot = np.linspace(t_support_lo, t_support_hi, n_t_plot, dtype=np.float64)
    if axis == "coronal":
        xlabel, ylabel = "k", "j"
    else:
        xlabel, ylabel = "j", "i"

    lo_xy: np.ndarray | None = None
    hi_xy: np.ndarray | None = None
    if mapping_mode == "lut":
        if t_sup is None or lo_sup is None or hi_sup is None:
            return "internal error: lut support missing"
        lo_plot = dbg._interp_curve_by_t(t_support=t_sup, y_support=lo_sup, t_query=t_plot, mode=str(interp))
        hi_plot = dbg._interp_curve_by_t(t_support=t_sup, y_support=hi_sup, t_query=t_plot, mode=str(interp))
        mid_plot = 0.5 * (lo_plot + hi_plot)
        if axis == "coronal":
            lo_xy = np.column_stack([lo_plot[:, 2], lo_plot[:, 1]]).astype(np.float64, copy=False)
            hi_xy = np.column_stack([hi_plot[:, 2], hi_plot[:, 1]]).astype(np.float64, copy=False)
            mid_xy = np.column_stack([mid_plot[:, 2], mid_plot[:, 1]]).astype(np.float64, copy=False)
        else:
            lo_xy = np.column_stack([lo_plot[:, 1], lo_plot[:, 0]]).astype(np.float64, copy=False)
            hi_xy = np.column_stack([hi_plot[:, 1], hi_plot[:, 0]]).astype(np.float64, copy=False)
            mid_xy = np.column_stack([mid_plot[:, 1], mid_plot[:, 0]]).astype(np.float64, copy=False)
        radial_vec = hi_xy - lo_xy
        radial_norm = np.linalg.norm(radial_vec, axis=1)
        radial_unit = radial_vec / np.maximum(radial_norm[:, None], 1.0e-12)
        anchor_curve_xy = dbg._gauss_smooth_xy(mid_xy, sigma=float(smooth_mid))
        base_curve_xy = lo_xy
    else:
        mid_plot_ijk, normal_plot_ijk = evaluate_midline_normal_bundle(columns=cols_mid, axis=axis, t_query=t_plot)
        if axis == "coronal":
            mid_xy = np.column_stack([mid_plot_ijk[:, 2], mid_plot_ijk[:, 1]]).astype(np.float64, copy=False)
            radial_unit = np.column_stack([normal_plot_ijk[:, 2], normal_plot_ijk[:, 1]]).astype(np.float64, copy=False)
        else:
            mid_xy = np.column_stack([mid_plot_ijk[:, 1], mid_plot_ijk[:, 0]]).astype(np.float64, copy=False)
            radial_unit = np.column_stack([normal_plot_ijk[:, 1], normal_plot_ijk[:, 0]]).astype(np.float64, copy=False)
        radial_unit /= np.maximum(np.linalg.norm(radial_unit, axis=1, keepdims=True), 1.0e-12)
        anchor_curve_xy = dbg._gauss_smooth_xy(mid_xy, sigma=float(smooth_mid))
        base_curve_xy = mid_xy

    # Midline spline anchor points (derived from saved midline columns for this slice).
    dt_anchor = float(MIDLINE_INSLICE_ANCHOR_DT)
    n_anchor = int(max(2, np.rint(1.0 / dt_anchor) + 1))
    t_anchor = np.linspace(0.0, 1.0, int(n_anchor), dtype=np.float64)
    y_anchor = np.interp(t_anchor, t_mid_u, y_mid_u).astype(np.float64, copy=False)
    x_anchor = np.interp(t_anchor, t_mid_u, x_mid_u).astype(np.float64, copy=False)
    anchor_yx = np.column_stack([y_anchor, x_anchor]).astype(np.float64, copy=False)
    anchor_yx = _reject_midline_anchor_outliers(anchor_yx)
    anchor_xy = np.column_stack([anchor_yx[:, 1], anchor_yx[:, 0]]).astype(np.float64, copy=False)

    # Scatter sample of mapped cells (visual only).
    if t_lookup.size > int(max_points_scatter):
        idx2 = rng.choice(np.arange(t_lookup.size), size=int(max_points_scatter), replace=False)
        t_sc = t_lookup[idx2]
        r_sc = r_signed[idx2]
    else:
        t_sc = t_lookup
        r_sc = r_signed

    # Recompute floor for the scatter sample (close enough for visualization).
    t_sc_clip, _t_bins2, _rfb2, _rcb2, (rf_at_t2, _rc_at_t2) = dbg._rolling_floor_ceil(
        t_lookup=t_sc,
        r_signed=r_sc,
        n_bins=int(r_bins),
        roll_half_window=int(r_roll_half_window),
    )
    r_um_sc = (r_sc - rf_at_t2) * float(r_source_um_per_px)
    r_um_sc = np.clip(np.asarray(r_um_sc, dtype=np.float64), 0.0, np.inf)

    t_cell = np.clip(np.asarray(t_sc_clip, dtype=np.float64), t_support_lo, t_support_hi)
    if mapping_mode == "lut":
        if t_sup is None or lo_sup is None or hi_sup is None:
            return "internal error: lut support missing"
        lo_cell = dbg._interp_curve_by_t(t_support=t_sup, y_support=lo_sup, t_query=t_cell, mode=str(interp))
        hi_cell = dbg._interp_curve_by_t(t_support=t_sup, y_support=hi_sup, t_query=t_cell, mode=str(interp))
        if axis == "coronal":
            lo_cell_xy = np.column_stack([lo_cell[:, 2], lo_cell[:, 1]]).astype(np.float64, copy=False)
            hi_cell_xy = np.column_stack([hi_cell[:, 2], hi_cell[:, 1]]).astype(np.float64, copy=False)
        else:
            lo_cell_xy = np.column_stack([lo_cell[:, 1], lo_cell[:, 0]]).astype(np.float64, copy=False)
            hi_cell_xy = np.column_stack([hi_cell[:, 1], hi_cell[:, 0]]).astype(np.float64, copy=False)
        v_cell = hi_cell_xy - lo_cell_xy
        n_cell = np.linalg.norm(v_cell, axis=1)
        u_cell = v_cell / np.maximum(n_cell[:, None], 1.0e-12)
        base_cell_xy = lo_cell_xy
    else:
        mid_cell_ijk, normal_cell_ijk = evaluate_midline_normal_bundle(columns=cols_mid, axis=axis, t_query=t_cell)
        if axis == "coronal":
            base_cell_xy = np.column_stack([mid_cell_ijk[:, 2], mid_cell_ijk[:, 1]]).astype(np.float64, copy=False)
            u_cell = np.column_stack([normal_cell_ijk[:, 2], normal_cell_ijk[:, 1]]).astype(np.float64, copy=False)
        else:
            base_cell_xy = np.column_stack([mid_cell_ijk[:, 1], mid_cell_ijk[:, 0]]).astype(np.float64, copy=False)
            u_cell = np.column_stack([normal_cell_ijk[:, 1], normal_cell_ijk[:, 0]]).astype(np.float64, copy=False)
        u_cell /= np.maximum(np.linalg.norm(u_cell, axis=1, keepdims=True), 1.0e-12)
    r_px_target = r_um_sc / float(r_target_um_per_px)
    pts_xy = base_cell_xy + r_px_target[:, None] * u_cell

    # Iso-curves (constant r_um) under the mapping.
    for r0 in r_iso_um.tolist():
        r0_px = float(r0) / float(r_target_um_per_px)
        xy0 = base_curve_xy + r0_px * radial_unit
        xy0 = dbg._gauss_smooth_xy(xy0, sigma=float(smooth_iso))
        ax.plot(xy0[:, 0], xy0[:, 1], linewidth=0.8, alpha=0.65, color="#888888")

    if mapping_mode == "lut":
        if lo_xy is None or hi_xy is None:
            return "internal error: missing low/high curves in lut mode"
        ax.plot(lo_xy[:, 0], lo_xy[:, 1], color="black", linewidth=0.9)
        ax.plot(hi_xy[:, 0], hi_xy[:, 1], color="black", linewidth=0.9, linestyle="--")
    ax.plot(anchor_curve_xy[:, 0], anchor_curve_xy[:, 1], color="gray", linewidth=1.0, alpha=0.9)
    ax.scatter(anchor_xy[:, 0], anchor_xy[:, 1], s=6.0, alpha=0.85, color="magenta", linewidths=0.0)
    ax.scatter(pts_xy[:, 0], pts_xy[:, 1], s=0.6, alpha=0.08, color="tab:green", linewidths=0.0)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(xlabel, fontsize=7)
    ax.set_ylabel(ylabel, fontsize=7)
    return None


def _discover_h5ads(*, working_root: Path, sample_glob: str) -> list[Path]:
    out: list[Path] = []
    for sample_dir in sorted(working_root.glob(sample_glob)):
        ccf_dir = sample_dir / "analysis" / "output" / "ccf-transforms"
        if not ccf_dir.exists():
            continue
        out.extend(sorted(ccf_dir.glob("*/*.princurve.h5ad")))
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="Batch render the center panel of debug_tr_to_ijk_plot.py into a grid figure.")
    p.add_argument("--working-root", type=Path, default=Path("/working"))
    p.add_argument("--sample-glob", type=str, default="2025*Jax*")
    p.add_argument("--lut-outdir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--ncols", type=int, default=10)
    p.add_argument("--max-panels", type=int, default=None)
    p.add_argument("--interp", type=str, default="linear", choices=("linear", "cubic"))
    p.add_argument("--mapping-mode", type=str, default="midline-normal", choices=("lut", "midline-normal"))
    p.add_argument("--smooth-mid", type=float, default=3.0)
    p.add_argument("--smooth-iso", type=float, default=2.0)
    p.add_argument("--n-t", type=int, default=512)
    p.add_argument("--r-bins", type=int, default=512)
    p.add_argument("--r-roll-half-window", type=int, default=12)
    p.add_argument("--r-source-um-per-px", type=float, default=0.216)
    p.add_argument("--r-target-um-per-px", type=float, default=20.0)
    p.add_argument("--r-iso-fracs", type=float, nargs="+", default=(0.0, 0.1, 0.25, 0.5, 0.8, 1.0))
    p.add_argument("--max-points-floor", type=int, default=200_000, help="Downsample for floor/ceil + r_hi estimation.")
    p.add_argument("--max-points-scatter", type=int, default=10_000, help="Downsample for green mapped-cell scatter.")
    args = p.parse_args()

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt  # noqa: E402

    h5ads = _discover_h5ads(working_root=args.working_root, sample_glob=str(args.sample_glob))
    if args.max_panels is not None:
        h5ads = h5ads[: int(args.max_panels)]
    if not h5ads:
        raise SystemExit("No .princurve.h5ad files found.")

    lut_coronal = _load_lut_cache(outdir=args.lut_outdir, axis="coronal")
    lut_sagittal = _load_lut_cache(outdir=args.lut_outdir, axis="sagittal")
    midline_coronal = load_coronal_midline_columns(args.lut_outdir / "coronal_midline_columns.csv")
    midline_sagittal = load_sagittal_midline_columns(args.lut_outdir / "sagittal_midline_columns.csv")

    ncols = int(max(1, args.ncols))
    nrows = int(np.ceil(len(h5ads) / float(ncols)))
    # Keep panels readable without generating absurdly large images.
    panel_in = 2.25
    fig = plt.figure(figsize=(panel_in * ncols, panel_in * nrows), dpi=160)

    for idx, h5ad in enumerate(h5ads):
        ax = fig.add_subplot(nrows, ncols, idx + 1)
        sample = h5ad.parents[4].name if len(h5ad.parents) >= 5 else h5ad.parent.parent.name
        roi = h5ad.parent.name
        err = _render_center_panel(
            ax=ax,
            h5ad_path=h5ad,
            lut_coronal=lut_coronal,
            lut_sagittal=lut_sagittal,
            midline_coronal=midline_coronal,
            midline_sagittal=midline_sagittal,
            mapping_mode=str(args.mapping_mode),
            interp=str(args.interp),
            smooth_mid=float(args.smooth_mid),
            smooth_iso=float(args.smooth_iso),
            n_t=int(args.n_t),
            r_bins=int(args.r_bins),
            r_roll_half_window=int(args.r_roll_half_window),
            r_source_um_per_px=float(args.r_source_um_per_px),
            r_target_um_per_px=float(args.r_target_um_per_px),
            r_iso_fracs=np.asarray(args.r_iso_fracs, dtype=np.float64),
            max_points_floor=int(args.max_points_floor),
            max_points_scatter=int(args.max_points_scatter),
        )
        if err is None:
            ax.set_title(f"{sample}/{roi}\n{h5ad.name}", fontsize=7)
        else:
            ax.set_title(f"{sample}/{roi}\n{h5ad.name}\nERR: {err}", fontsize=7, color="crimson")

    # Hide unused axes.
    for j in range(len(h5ads), nrows * ncols):
        ax = fig.add_subplot(nrows, ncols, j + 1)
        ax.axis("off")

    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out} (panels={len(h5ads)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
