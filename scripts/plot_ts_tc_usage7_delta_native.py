#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import pathlib
import sys
from typing import Any

import matplotlib
import numpy as np
from scipy.spatial import Delaunay


matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from fishtools.ccf.transforms import DEFAULT_NATIVE_CAMERA_AZIM_DEG
from fishtools.ccf.transforms import DEFAULT_NATIVE_CAMERA_ELEV_DEG
from fishtools.ccf.transforms import DEFAULT_NATIVE_CAMERA_ROLL_DEG
from fishtools.ccf.transforms import DEFAULT_NATIVE_PROJ_TYPE
from fishtools.ccf.transforms import build_apml_native_surface_projection_context


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
MULTINOMIAL_PATH = SCRIPT_DIR / "fit_apml_multinomial_animal_meta.py"
NATIVE_PATH = SCRIPT_DIR / "plot_ts_tc_native_with_sagittal_line.py"
DEFAULT_H5AD = pathlib.Path("~/nvme/all_excit.h5ad")
DEFAULT_OBSM_H5AD = pathlib.Path("~/nvme/obsm.h5ad")
DEFAULT_REFEXTRACT_OUTDIR = pathlib.Path("ccf/out/refextract/midsurface_neocortex_mesocortex_allocortex_3d")
DEFAULT_OUTDIR = pathlib.Path("scripts/_out/apml_pooled_usage7_delta_native")
DEFAULT_REFEXTRACT_SLICE_I_MIN = 161
DEFAULT_REFEXTRACT_SLICE_I_MAX = 305
DEFAULT_REFEXTRACT_N_T = 257
DEFAULT_REFEXTRACT_REF_T = 0.5
DEFAULT_REFEXTRACT_BAND_FRAC = 0.15
DEFAULT_REFEXTRACT_RES_IJK_UM = (20.0, 20.0, 20.0)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render native Ts/Tc difference maps for layer 1 Usage_7-filtered vs unfiltered.")
    parser.add_argument("--h5ad", type=pathlib.Path, default=DEFAULT_H5AD)
    parser.add_argument("--obsm-h5ad", type=pathlib.Path, default=DEFAULT_OBSM_H5AD)
    parser.add_argument("--clusters", type=str, default="0,1,2,3,4,5,6,7,8")
    parser.add_argument("--exclude-animals", type=str, default="JaxA2")
    parser.add_argument("--delta-t-hours", type=float, default=1.5)
    parser.add_argument("--n-draws", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--usage7-lte", type=float, default=None)
    parser.add_argument("--usage7-frac-lte", type=float, default=None)
    parser.add_argument("--refextract-outdir", type=pathlib.Path, default=DEFAULT_REFEXTRACT_OUTDIR)
    parser.add_argument("--outdir", type=pathlib.Path, default=DEFAULT_OUTDIR)
    return parser.parse_args()


def _load_module(name: str, path: pathlib.Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_multinomial_module() -> Any:
    return _load_module("fit_apml_multinomial_animal_meta", MULTINOMIAL_PATH)


def _load_native_module() -> Any:
    return _load_module("plot_ts_tc_native_with_sagittal_line", NATIVE_PATH)


def _output_stem(*, usage7_lte: float | None, usage7_frac_lte: float | None) -> str:
    if usage7_frac_lte is not None:
        frac_tag = str(usage7_frac_lte).replace(".", "p")
        return f"ts_tc_native_diff_manual_layer_1__Usage7_over_13467_lte_{frac_tag}_minus_unfiltered"
    if usage7_lte is None:
        raise ValueError("Expected at least one Usage_7 threshold.")
    usage_tag = str(usage7_lte).replace(".", "p")
    return f"ts_tc_native_diff_manual_layer_1__Usage7_lte_{usage_tag}_minus_unfiltered"


def _evaluate_surface(
    *,
    meta_mod: Any,
    df: Any,
    subset: str,
    delta_t_hours: float,
    n_draws: int,
    seed: int,
    context: dict[str, object],
) -> tuple[np.ndarray, np.ndarray]:
    surface_eval = meta_mod._evaluate_pooled_multinomial_at_queries(
        df,
        subset=subset,
        delta_t_hours=delta_t_hours,
        n_draws=n_draws,
        seed=seed,
        ap_mm=np.asarray(context["ap_um_flat"], dtype=np.float64) / 1000.0,
        ml_mm=np.asarray(context["ml_um_flat"], dtype=np.float64) / 1000.0,
    )
    ts_values = surface_eval["Ts_hours_mean"].to_numpy(dtype=float)
    tc_values = surface_eval["Tc_hours_mean"].to_numpy(dtype=float)
    support_mask = np.asarray(context["support_flat"], dtype=bool)
    ts_values[~support_mask] = np.nan
    tc_values[~support_mask] = np.nan
    return ts_values, tc_values


def _symmetric_vlim(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return -1.0, 1.0
    vmax = float(np.nanmax(np.abs(finite)))
    if vmax == 0.0:
        vmax = 1.0
    return -vmax, vmax


def _build_difference_figure(
    *,
    native_mod: Any,
    helper_mod: Any,
    context: dict[str, object],
    support_hull: Delaunay,
    tc_diff_values: np.ndarray,
    ts_diff_values: np.ndarray,
    title_suffix: str,
) -> plt.Figure:
    fig = plt.figure(figsize=(12.6, 6.2), dpi=180)
    axes = [
        fig.add_subplot(1, 2, 1, projection="3d" if DEFAULT_NATIVE_PROJ_TYPE == "persp" else None),
        fig.add_subplot(1, 2, 2, projection="3d" if DEFAULT_NATIVE_PROJ_TYPE == "persp" else None),
    ]
    panels = (
        ("Tc difference", np.asarray(tc_diff_values, dtype=np.float64), "Tc diff (hours)"),
        ("Ts difference", np.asarray(ts_diff_values, dtype=np.float64), "Ts diff (hours)"),
    )
    for ax, (panel_title, values, cbar_label) in zip(axes, panels, strict=True):
        vmin, vmax = _symmetric_vlim(values)
        native_mod._draw_native_scalar_surface_panel(
            fig=fig,
            ax=ax,
            helper_mod=helper_mod,
            context=context,
            values=values,
            support_hull=support_hull,
            panel_title=panel_title,
            cbar_label=cbar_label,
            native_elev_deg=DEFAULT_NATIVE_CAMERA_ELEV_DEG,
            native_azim_deg=DEFAULT_NATIVE_CAMERA_AZIM_DEG,
            native_roll_deg=DEFAULT_NATIVE_CAMERA_ROLL_DEG,
            cmap_name="coolwarm",
            vmin=vmin,
            vmax=vmax,
        )
    fig.suptitle(f"Layer 1 pooled native difference: {title_suffix} minus unfiltered", fontsize=13)
    return fig


def _save_difference_figure(
    *,
    native_mod: Any,
    helper_mod: Any,
    context: dict[str, object],
    support_hull: Delaunay,
    tc_diff_values: np.ndarray,
    ts_diff_values: np.ndarray,
    title_suffix: str,
    out_png: pathlib.Path,
) -> None:
    fig = _build_difference_figure(
        native_mod=native_mod,
        helper_mod=helper_mod,
        context=context,
        support_hull=support_hull,
        tc_diff_values=tc_diff_values,
        ts_diff_values=ts_diff_values,
        title_suffix=title_suffix,
    )
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    if args.usage7_lte is not None and args.usage7_frac_lte is not None:
        raise ValueError("Specify at most one of --usage7-lte or --usage7-frac-lte.")
    usage7_lte = 0.2 if args.usage7_lte is None and args.usage7_frac_lte is None else args.usage7_lte
    meta_mod = _load_multinomial_module()
    native_mod = _load_native_module()
    helper_mod = meta_mod._load_plot_apml_module()
    clusters = {c.strip() for c in str(args.clusters).split(",") if c.strip()}
    exclude_animals = {a.strip() for a in str(args.exclude_animals).split(",") if a.strip()}

    context = build_apml_native_surface_projection_context(
        outdir=args.refextract_outdir.expanduser(),
        slice_i_min=DEFAULT_REFEXTRACT_SLICE_I_MIN,
        slice_i_max=DEFAULT_REFEXTRACT_SLICE_I_MAX,
        n_t=DEFAULT_REFEXTRACT_N_T,
        ref_t=DEFAULT_REFEXTRACT_REF_T,
        band_frac=DEFAULT_REFEXTRACT_BAND_FRAC,
        res_ijk_um=DEFAULT_REFEXTRACT_RES_IJK_UM,
        elev_deg=DEFAULT_NATIVE_CAMERA_ELEV_DEG,
        azim_deg=DEFAULT_NATIVE_CAMERA_AZIM_DEG,
        roll_deg=DEFAULT_NATIVE_CAMERA_ROLL_DEG,
    )
    adata = helper_mod._load_adata_with_external_obsm(
        h5ad_path=args.h5ad.expanduser(),
        obsm_h5ad_path=args.obsm_h5ad.expanduser(),
    )

    unfiltered_df = meta_mod._build_model_df(
        helper_mod=helper_mod,
        adata=adata,
        clusters=clusters,
        manual_layer="1",
        eomes_gt=None,
        eomes_lt=None,
        usage7_lte=None,
        usage7_fraction_lte=None,
        label="manual_layer_1",
        exclude_animals=exclude_animals,
    )
    subset_label = (
        f"manual_layer_1__Usage7over13467_lte_{args.usage7_frac_lte:g}"
        if args.usage7_frac_lte is not None
        else f"manual_layer_1__Usage7_lte_{usage7_lte:g}"
    )
    title_suffix = (
        f"Usage_7/(Usage_1+3+4+6+7)<={args.usage7_frac_lte:g}"
        if args.usage7_frac_lte is not None
        else f"Usage_7<={usage7_lte:g}"
    )
    filtered_df = meta_mod._build_model_df(
        helper_mod=helper_mod,
        adata=adata,
        clusters=clusters,
        manual_layer="1",
        eomes_gt=None,
        eomes_lt=None,
        usage7_lte=None if usage7_lte is None else float(usage7_lte),
        usage7_fraction_lte=args.usage7_frac_lte,
        label=subset_label,
        exclude_animals=exclude_animals,
    )

    _ap_bounds_um, _ml_bounds_um, support_hull = native_mod._build_subset_support_hull(
        ap_um=filtered_df["ap_mm"].to_numpy(dtype=float) * 1000.0,
        ml_um=filtered_df["ml_mm"].to_numpy(dtype=float) * 1000.0,
    )
    unfiltered_ts, unfiltered_tc = _evaluate_surface(
        meta_mod=meta_mod,
        df=unfiltered_df,
        subset="manual_layer_1",
        delta_t_hours=float(args.delta_t_hours),
        n_draws=int(args.n_draws),
        seed=int(args.seed),
        context=context,
    )
    filtered_ts, filtered_tc = _evaluate_surface(
        meta_mod=meta_mod,
        df=filtered_df,
        subset=subset_label,
        delta_t_hours=float(args.delta_t_hours),
        n_draws=int(args.n_draws),
        seed=int(args.seed),
        context=context,
    )
    ts_diff = filtered_ts - unfiltered_ts
    tc_diff = filtered_tc - unfiltered_tc

    outdir = args.outdir.expanduser()
    outdir.mkdir(parents=True, exist_ok=True)
    out_png = outdir / f"{_output_stem(usage7_lte=usage7_lte, usage7_frac_lte=args.usage7_frac_lte)}.png"
    _save_difference_figure(
        native_mod=native_mod,
        helper_mod=helper_mod,
        context=context,
        support_hull=support_hull,
        tc_diff_values=tc_diff,
        ts_diff_values=ts_diff,
        title_suffix=title_suffix,
        out_png=out_png,
    )
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
