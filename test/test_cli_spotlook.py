from __future__ import annotations

import types
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from matplotlib.legend import Legend
from matplotlib.figure import Figure
from scipy.interpolate import RegularGridInterpolator

from fishtools.preprocess.cli_spotlook import (
    ROIThresholdContext,
    ThresholdCurve,
    _compute_threshold_curve,
    _compute_contour_levels,
    _prompt_threshold_levels,
    _save_combined_threshold_plot,
)
from fishtools.utils.plot import format_si
from fishtools.preprocess.config import SpotThresholdParams


def _make_interpolator() -> RegularGridInterpolator:
    grid = np.linspace(0.0, 1.0, 3)
    values = np.array(
        [
            [0.0, 0.2, 0.4],
            [0.2, 0.4, 0.6],
            [0.4, 0.6, 0.8],
        ]
    )
    return RegularGridInterpolator((grid, grid), values)


def test_compute_threshold_curve_generates_expected_counts() -> None:
    spots = pl.DataFrame(
        {
            "x_": [0.1, 0.2, 0.4, 0.6],
            "y_": [0.1, 0.4, 0.6, 0.8],
            "is_blank": [False, True, False, True],
        }
    )
    contours = types.SimpleNamespace(levels=np.linspace(0.0, 1.0, 8))
    interpolator = _make_interpolator()

    curve = _compute_threshold_curve(spots, contours, interpolator)  # type: ignore[arg-type]

    expected_levels = [1, 3, 5, 7]
    expected_counts = []
    expected_blanks = []
    point_densities = interpolator(spots.select(["y_", "x_"]).to_numpy())
    for idx in expected_levels:
        threshold_value = contours.levels[idx]
        mask = point_densities < threshold_value
        expected_counts.append(int(mask.sum()))
        blank_mask = mask & spots["is_blank"].to_numpy()
        expected_blanks.append(blank_mask.sum() / max(1, mask.sum()))

    assert curve.levels == expected_levels
    assert curve.spot_counts == expected_counts
    assert curve.blank_proportions == pytest.approx(expected_blanks)
    assert curve.max_level == len(contours.levels) - 1


def test_prompt_threshold_levels_enforces_order_and_ranges(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def fake_text(message: str, validate):
        captured["message"] = message
        captured["validate"] = validate

        class _Prompt:
            def ask(self_inner):
                return "2,0"

        return _Prompt()

    monkeypatch.setattr("fishtools.preprocess.cli_spotlook.questionary.text", fake_text)

    interpolator = _make_interpolator()
    contours = types.SimpleNamespace(levels=np.linspace(0.0, 1.0, 6))
    curve_a = ThresholdCurve(levels=[1, 3], spot_counts=[10, 6], blank_proportions=[0.1, 0.2], max_level=5)
    curve_b = ThresholdCurve(levels=[1, 3], spot_counts=[12, 8], blank_proportions=[0.05, 0.1], max_level=2)

    context_a = ROIThresholdContext(
        spots=pl.DataFrame({"x_": [0.1], "y_": [0.1], "is_blank": [False]}),
        contours=contours,  # type: ignore[arg-type]
        interpolator=interpolator,
        curve=curve_a,
        artifact_paths={
            "contours": tmp_path / "contours-a.png",
            "threshold": tmp_path / "threshold-a.png",
            "spots_contours": tmp_path / "spots-a.png",
        },
    )
    context_b = ROIThresholdContext(
        spots=pl.DataFrame({"x_": [0.2], "y_": [0.2], "is_blank": [True]}),
        contours=contours,  # type: ignore[arg-type]
        interpolator=interpolator,
        curve=curve_b,
        artifact_paths={
            "contours": tmp_path / "contours-b.png",
            "threshold": tmp_path / "threshold-b.png",
            "spots_contours": None,
        },
    )

    contexts = {"roi_a": context_a, "roi_b": context_b}
    result = _prompt_threshold_levels(["roi_a", "roi_b"], contexts, tmp_path, "cb1")

    assert result == {"roi_a": 2, "roi_b": 0}

    message = str(captured["message"])
    assert "roi_a: 0-5" in message
    assert "roi_b: 0-2" in message
    assert message.index("Generated artifacts:") < message.index("Combined plot:")
    assert message.index("Combined plot:") < message.index("Enter threshold levels")
    prompt_line = "Please enter the threshold levels now (comma-separated integers):"
    assert prompt_line in message
    assert message.index("Enter threshold levels") < message.index(prompt_line)

    validator = captured["validate"]
    assert callable(validator)
    assert "Expected" in validator("2")  # type: ignore[index]
    assert "ROI roi_a" in validator("6,0")  # type: ignore[index]
    assert validator("2,0") is True  # type: ignore[index]


def test_save_combined_threshold_plot_sets_line_styles(monkeypatch, tmp_path: Path) -> None:
    params = SpotThresholdParams()
    curve_a = ThresholdCurve(levels=[1, 3], spot_counts=[100, 80], blank_proportions=[0.2, 0.1], max_level=4)
    curve_b = ThresholdCurve(levels=[1, 3], spot_counts=[90, 70], blank_proportions=[0.3, 0.15], max_level=4)
    curves = {"roi_b": curve_b, "roi_a": curve_a}

    captured: dict[str, object] = {}

    def fake_savefig(self, path, *args, **kwargs):  # noqa: ANN001
        captured.update({"fig": self, "path": Path(path)})

    monkeypatch.setattr(Figure, "savefig", fake_savefig, raising=False)

    path = _save_combined_threshold_plot(curves, tmp_path, "cb1", params)

    assert path == (tmp_path / "threshold_selection_all+cb1.png").resolve()
    assert captured["path"] == path

    fig = captured["fig"]
    ax_counts, ax_blank = fig.axes
    assert all(line.get_linestyle() == "-" for line in ax_counts.lines)
    assert all(line.get_linestyle() == "--" for line in ax_blank.lines)

    roi_legends = [artist for artist in ax_counts.artists if isinstance(artist, Legend)]
    assert roi_legends, "Expected ROI legend to be attached to the axis"
    legend_labels = [text.get_text() for text in roi_legends[0].get_texts()]
    assert legend_labels == ["roi_a", "roi_b"]


def test_si_formatter_human_friendly() -> None:
    # No prefix under 1k and no trailing .0
    assert format_si(950) == "950"

    # Exact multiples show no .0 and no space before prefix
    assert format_si(1_000) == "1k"
    assert format_si(10_000) == "10k"
    assert format_si(1_000_000) == "1M"

    # Non-integers keep one decimal
    assert format_si(1_500) == "1.5k"
    assert format_si(2_300_000) == "2.3M"


def test_contour_level_modes() -> None:
    # Construct small arrays with controlled ranges
    z_lin = np.array([[0.0, 50.0], [100.0, 75.0]])
    z_pos = np.array([[1.0, 10.0], [100.0, 1000.0]])

    # Linear: equally spaced 5 levels from min to max
    lin_levels = _compute_contour_levels(z_lin, "linear", 5)
    assert np.allclose(lin_levels, np.linspace(0.0, 100.0, 5))

    # Sqrt: equally spaced in sqrt space, then squared back
    sqrt_levels = _compute_contour_levels(z_lin, "sqrt", 5)
    expected_sqrt = np.linspace(np.sqrt(0.0), np.sqrt(100.0), 5) ** 2
    assert np.allclose(sqrt_levels, expected_sqrt)

    # Log: log-spaced between min positive and max
    log_levels = _compute_contour_levels(z_pos, "log", 4)
    assert np.allclose(log_levels, 10 ** np.linspace(0.0, 3.0, 4))
