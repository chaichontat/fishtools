from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pandas as pd


def _load_labeler_module():  # type: ignore[return-type]
    script_path = Path(__file__).parents[1] / "scripts/segmentation/brdu_edu_labeler_app.py"
    spec = importlib.util.spec_from_file_location("brdu_edu_labeler_app", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load labeler app from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_audit_top_n_slider_args_skips_slider_for_small_cell_counts() -> None:
    module = _load_labeler_module()
    slider_args = getattr(module, "_audit_top_n_slider_args", None)
    if slider_args is None:
        raise AttributeError("brdu_edu_labeler_app missing _audit_top_n_slider_args")

    assert slider_args(3) is None
    assert slider_args(10) is None
    assert slider_args(11) == (10, 11, 11, 10)
    assert slider_args(57) == (10, 57, 50, 10)


def test_label_tab_cells_excludes_already_labeled_cells() -> None:
    module = _load_labeler_module()
    label_tab_cells = getattr(module, "_label_tab_cells", None)
    if label_tab_cells is None:
        raise AttributeError("brdu_edu_labeler_app missing _label_tab_cells")

    meta = pd.DataFrame(
        {
            "cell": [
                "20251228_JaxA4_Sag4:1|1",
                "20251228_JaxA4_Sag4:1|2",
                "20251224_JaxA4_Sag1:6|3",
            ]
        }
    )
    labels = pd.DataFrame(
        {
            "cell": [
                "20251228_JaxA4_Sag4:1|2",
                "20251224_JaxA4_Sag1:6|3",
            ],
            "brdu": [1, 0],
            "edu": [0, 1],
        }
    )

    assert label_tab_cells(meta=meta, labels=labels) == ["20251228_JaxA4_Sag4:1|1"]


def test_label_picker_cells_keeps_history_selected_labeled_cell_visible() -> None:
    module = _load_labeler_module()
    label_picker_cells = getattr(module, "_label_picker_cells", None)
    if label_picker_cells is None:
        raise AttributeError("brdu_edu_labeler_app missing _label_picker_cells")

    assert label_picker_cells(
        available_cells=["20251228_JaxA4_Sag4:1|2", "20251224_JaxA4_Sag1:6|3"],
        current_cell="20251228_JaxA4_Sag4:1|1",
    ) == [
        "20251228_JaxA4_Sag4:1|1",
        "20251228_JaxA4_Sag4:1|2",
        "20251224_JaxA4_Sag1:6|3",
    ]
    assert label_picker_cells(
        available_cells=["20251228_JaxA4_Sag4:1|1", "20251228_JaxA4_Sag4:1|2"],
        current_cell="20251228_JaxA4_Sag4:1|1",
    ) == [
        "20251228_JaxA4_Sag4:1|1",
        "20251228_JaxA4_Sag4:1|2",
    ]


def test_label_action_specs_exposes_four_one_click_choices() -> None:
    module = _load_labeler_module()
    label_action_specs = getattr(module, "_label_action_specs", None)
    if label_action_specs is None:
        raise AttributeError("brdu_edu_labeler_app missing _label_action_specs")

    assert label_action_specs() == [
        ("BrdU only", 1, 0),
        ("EdU only", 0, 1),
        ("Dual pos", 1, 1),
        ("None", 0, 0),
    ]
