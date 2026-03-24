from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
from scipy.spatial import Delaunay


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "plot_ts_tc_usage7_delta_native.py"
NATIVE_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "plot_ts_tc_native_with_sagittal_line.py"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_delta_module():
    return _load_module(SCRIPT_PATH, "plot_ts_tc_usage7_delta_native")


def _load_native_module():
    return _load_module(NATIVE_SCRIPT_PATH, "plot_ts_tc_native_with_sagittal_line")


def _make_fake_render_context() -> dict[str, object]:
    x3 = np.asarray([0.0, 1.0, 2.0, 0.0, 1.0, 2.0], dtype=float)
    y3 = np.asarray([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=float)
    z3 = np.asarray([0.0, 0.2, 0.4, 0.0, 0.2, 0.4], dtype=float)
    faces = np.asarray([[0, 1, 4], [0, 4, 3], [1, 2, 5], [1, 5, 4]], dtype=np.int32)
    support = np.asarray([True, True, True, True, True, True], dtype=bool)
    return {
        "x2d": x3.copy(),
        "y2d": y3.copy(),
        "z2d": z3.copy(),
        "x3": x3,
        "y3": y3,
        "z3": z3,
        "faces": faces,
        "tri_support": np.asarray([True, True, True, True], dtype=bool),
        "tri_neomeso": np.asarray([True, True, True, True], dtype=bool),
        "support_flat": support,
        "neomeso_flat": support.copy(),
        "ap_um_flat": np.asarray([0.0, 50.0, 100.0, 0.0, 50.0, 100.0], dtype=float),
        "ml_um_flat": np.asarray([-20.0, -20.0, -20.0, 20.0, 20.0, 20.0], dtype=float),
        "n_rows": 2,
        "n_cols": 3,
        "ordered_geom_support": None,
        "neomeso_start_fit": None,
        "neomeso_end_fit": None,
    }


def test_save_difference_figure_smoke(tmp_path: Path) -> None:
    mod = _load_delta_module()
    native_mod = _load_native_module()
    helper_mod = native_mod._load_plot_apml_module()
    context = _make_fake_render_context()
    support_hull = Delaunay(np.asarray([[0.0, -20.0], [100.0, -20.0], [0.0, 20.0], [100.0, 20.0]], dtype=float))
    out_png = tmp_path / "usage7_delta_native.png"

    mod._save_difference_figure(
        native_mod=native_mod,
        helper_mod=helper_mod,
        context=context,
        support_hull=support_hull,
        tc_diff_values=np.linspace(-2.0, 2.0, np.asarray(context["ap_um_flat"]).size, dtype=float),
        ts_diff_values=np.linspace(-0.5, 0.5, np.asarray(context["ap_um_flat"]).size, dtype=float),
        title_suffix="Usage_7<=0.2",
        out_png=out_png,
    )

    assert out_png.exists()
    assert out_png.stat().st_size > 0


def test_output_stem_supports_usage7_fraction_filter() -> None:
    mod = _load_delta_module()

    stem = mod._output_stem(usage7_lte=None, usage7_frac_lte=0.1)

    assert stem == "ts_tc_native_diff_manual_layer_1__Usage7_over_13467_lte_0p1_minus_unfiltered"
