from __future__ import annotations

import json
import types
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import click
from click.testing import CliRunner
from matplotlib.figure import Figure
from matplotlib.legend import Legend
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from scipy.interpolate import RegularGridInterpolator

from fishtools.preprocess.cli_spotlook import (
    ROIThresholdContext,
    ThresholdCurve,
    _save_threshold_plot,
    _compute_contour_levels,
    _compute_main_mass_mask,
    _compute_threshold_curve,
    _load_spots_data,
    _parse_threshold_levels_response,
    _prompt_threshold_levels,
    _select_threshold_levels_by_blank_proportion,
    _save_combined_spots_plot,
    _save_combined_threshold_plot,
    _write_spotlook_summary_json,
    threshold,
)
from fishtools.preprocess.config import SpotThresholdParams
from fishtools.utils.plot import format_si


def _make_interpolator() -> RegularGridInterpolator:
    grid = np.linspace(0.0, 1.0, 3)
    values = np.array([
        [0.0, 0.2, 0.4],
        [0.2, 0.4, 0.6],
        [0.4, 0.6, 0.8],
    ])
    return RegularGridInterpolator((grid, grid), values)


def test_compute_main_mass_mask_returns_all_true_when_no_spots() -> None:
    count_grid = np.zeros((32, 24), dtype=float)

    mask = _compute_main_mass_mask(count_grid)

    assert mask.dtype == bool
    assert mask.shape == count_grid.shape
    assert mask.all()


def test_compute_main_mass_mask_selects_component_containing_maximum() -> None:
    count_grid = np.zeros((100, 100), dtype=float)
    count_grid[20, 20] = 100.0
    count_grid[80, 80] = 90.0

    mask = _compute_main_mass_mask(count_grid, hdr_percentile=50.0)

    assert mask.dtype == bool
    assert mask.shape == count_grid.shape
    assert mask[20, 20]
    assert not mask[80, 80]


def test_compute_main_mass_mask_higher_hdr_percentile_is_subset() -> None:
    count_grid = np.zeros((64, 64), dtype=float)
    count_grid[32, 32] = 100.0
    count_grid[31, 32] = 50.0
    count_grid[33, 32] = 25.0
    count_grid[32, 31] = 10.0
    count_grid[32, 33] = 5.0

    mask_lo = _compute_main_mass_mask(count_grid, hdr_percentile=10.0)
    mask_hi = _compute_main_mass_mask(count_grid, hdr_percentile=90.0)

    assert mask_lo[32, 32]
    assert mask_hi[32, 32]
    assert mask_hi.sum() <= mask_lo.sum()
    assert np.all(mask_hi <= mask_lo)


def test_compute_threshold_curve_generates_expected_counts() -> None:
    spots = pl.DataFrame({
        "x_": [0.1, 0.2, 0.4, 0.6],
        "y_": [0.1, 0.4, 0.6, 0.8],
        "is_blank": [False, True, False, True],
    })
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

    def fake_echo(message: str = "", **_kwargs: object) -> None:
        captured["message"] = message

    def fake_prompt(_text: str, **_kwargs: object) -> str:
        return "2,0"

    def fake_confirm(_text: str, **_kwargs: object) -> bool:
        return True

    monkeypatch.setattr("fishtools.preprocess.cli_spotlook.click.echo", fake_echo)
    monkeypatch.setattr("fishtools.preprocess.cli_spotlook.click.prompt", fake_prompt)
    monkeypatch.setattr("fishtools.preprocess.cli_spotlook.click.confirm", fake_confirm)

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
    combined_path = (tmp_path / "spots_all--cb1.png").resolve()
    result = _prompt_threshold_levels(
        ["roi_a", "roi_b"],
        contexts,
        tmp_path,
        "cb1",
        combined_spots_path=combined_path,
    )

    assert result == {"roi_a": 2, "roi_b": 0}

    message = str(captured["message"])
    assert "Enter threshold levels for each ROI (comma-separated) between 0-5" in message
    assert "roi_a: 0-" not in message
    assert "roi_b: 0-" not in message
    assert message.index("Generated artifacts:") < message.index("Combined plot:")
    combined_line = f"Combined spots: {combined_path}"
    assert combined_line in message
    assert message.index("Combined plot:") < message.index(combined_line)
    assert message.index(combined_line) < message.index("Enter threshold levels")
    prompt_line = "Please enter the threshold levels now (comma-separated integers):"
    assert prompt_line in message
    assert message.index("Enter threshold levels") < message.index(prompt_line)

    with pytest.raises(ValueError, match="Expected 2 comma-separated values"):
        _parse_threshold_levels_response("2", ["roi_a", "roi_b"], contexts)
    with pytest.raises(ValueError, match="required"):
        _parse_threshold_levels_response("2, ", ["roi_a", "roi_b"], contexts)
    with pytest.raises(ValueError, match="ROI roi_a"):
        _parse_threshold_levels_response("6,0", ["roi_a", "roi_b"], contexts)
    assert _parse_threshold_levels_response("2,0", ["roi_a", "roi_b"], contexts) == {
        "roi_a": 2,
        "roi_b": 0,
    }
    assert _parse_threshold_levels_response("blank=0.08", ["roi_a", "roi_b"], contexts) == {
        "roi_a": 1,
        "roi_b": 2,
    }


def test_parse_threshold_levels_response_blank_proportion_requires_confirmation(monkeypatch, tmp_path: Path) -> None:
    interpolator = _make_interpolator()
    contours = types.SimpleNamespace(levels=np.linspace(0.0, 1.0, 6))

    contexts = {
        "roi_a": ROIThresholdContext(
            spots=pl.DataFrame({"x_": [0.1], "y_": [0.1], "is_blank": [False]}),
            contours=contours,  # type: ignore[arg-type]
            interpolator=interpolator,
            curve=ThresholdCurve(levels=[1, 3], spot_counts=[10, 6], blank_proportions=[0.1, 0.2], max_level=5),
            artifact_paths={"contours": tmp_path / "contours-a.png", "threshold": tmp_path / "threshold-a.png"},
        )
    }

    monkeypatch.setattr("fishtools.preprocess.cli_spotlook.click.confirm", lambda *_a, **_k: False)
    with pytest.raises(ValueError, match="Cancelled; please re-enter"):
        _parse_threshold_levels_response("blank=0.0075", ["roi_a"], contexts)

    monkeypatch.setattr("fishtools.preprocess.cli_spotlook.click.confirm", lambda *_a, **_k: True)
    assert _parse_threshold_levels_response("blank=0.0075", ["roi_a"], contexts) == {"roi_a": 1}


def test_parse_threshold_levels_response_bare_decimal_treated_as_blank_proportion(monkeypatch, tmp_path: Path) -> None:
    interpolator = _make_interpolator()
    contours = types.SimpleNamespace(levels=np.linspace(0.0, 1.0, 6))

    contexts = {
        "roi_a": ROIThresholdContext(
            spots=pl.DataFrame({"x_": [0.1], "y_": [0.1], "is_blank": [False]}),
            contours=contours,  # type: ignore[arg-type]
            interpolator=interpolator,
            curve=ThresholdCurve(levels=[1, 3], spot_counts=[10, 6], blank_proportions=[0.1, 0.2], max_level=5),
            artifact_paths={"contours": tmp_path / "contours-a.png", "threshold": tmp_path / "threshold-a.png"},
        )
    }

    monkeypatch.setattr("fishtools.preprocess.cli_spotlook.click.confirm", lambda *_a, **_k: True)
    assert _parse_threshold_levels_response("0.00075", ["roi_a"], contexts) == {"roi_a": 1}


def test_parse_threshold_levels_response_rejects_percent(monkeypatch, tmp_path: Path) -> None:
    interpolator = _make_interpolator()
    contours = types.SimpleNamespace(levels=np.linspace(0.0, 1.0, 6))

    contexts = {
        "roi_a": ROIThresholdContext(
            spots=pl.DataFrame({"x_": [0.1], "y_": [0.1], "is_blank": [False]}),
            contours=contours,  # type: ignore[arg-type]
            interpolator=interpolator,
            curve=ThresholdCurve(levels=[1, 3], spot_counts=[10, 6], blank_proportions=[0.1, 0.2], max_level=5),
            artifact_paths={"contours": tmp_path / "contours-a.png", "threshold": tmp_path / "threshold-a.png"},
        )
    }

    monkeypatch.setattr("fishtools.preprocess.cli_spotlook.click.confirm", lambda *_a, **_k: True)
    with pytest.raises(ValueError, match="Percent inputs are not supported"):
        _parse_threshold_levels_response("blank=0.1%", ["roi_a"], contexts)
    with pytest.raises(ValueError, match="Percent inputs are not supported"):
        _parse_threshold_levels_response("0.1%", ["roi_a"], contexts)


def test_prompt_threshold_levels_ctrl_c_raises_keyboard_interrupt(monkeypatch, tmp_path: Path) -> None:
    def fake_echo(_message: str = "", **_kwargs: object) -> None:
        return None

    def fake_prompt(_text: str, **_kwargs: object) -> str:
        raise click.Abort()

    monkeypatch.setattr("fishtools.preprocess.cli_spotlook.click.echo", fake_echo)
    monkeypatch.setattr("fishtools.preprocess.cli_spotlook.click.prompt", fake_prompt)

    interpolator = _make_interpolator()
    contours = types.SimpleNamespace(levels=np.linspace(0.0, 1.0, 3))
    curve = ThresholdCurve(levels=[1], spot_counts=[10], blank_proportions=[0.1], max_level=2)

    contexts = {
        "roi_a": ROIThresholdContext(
            spots=pl.DataFrame({"x_": [0.1], "y_": [0.1], "is_blank": [False]}),
            contours=contours,  # type: ignore[arg-type]
            interpolator=interpolator,
            curve=curve,
            artifact_paths={"contours": tmp_path / "contours-a.png", "threshold": tmp_path / "threshold-a.png"},
        )
    }

    with pytest.raises(KeyboardInterrupt):
        _prompt_threshold_levels(["roi_a"], contexts, tmp_path, "cb1")


def test_select_threshold_levels_by_blank_proportion_picks_closest_level(tmp_path: Path) -> None:
    interpolator = _make_interpolator()
    contours = types.SimpleNamespace(levels=np.linspace(0.0, 1.0, 10))
    contexts = {
        "roi_a": ROIThresholdContext(
            spots=pl.DataFrame({"x_": [0.1], "y_": [0.1], "is_blank": [False]}),
            contours=contours,  # type: ignore[arg-type]
            interpolator=interpolator,
            curve=ThresholdCurve(levels=[1, 3, 5], spot_counts=[10, 8, 6], blank_proportions=[0.05, 0.1, 0.2], max_level=9),
            artifact_paths={"contours": tmp_path / "contours-a.png", "threshold": tmp_path / "threshold-a.png"},
        ),
        "roi_b": ROIThresholdContext(
            spots=pl.DataFrame({"x_": [0.2], "y_": [0.2], "is_blank": [True]}),
            contours=contours,  # type: ignore[arg-type]
            interpolator=interpolator,
            curve=ThresholdCurve(levels=[1, 3, 5], spot_counts=[12, 9, 7], blank_proportions=[0.02, 0.07, 0.15], max_level=9),
            artifact_paths={"contours": tmp_path / "contours-b.png", "threshold": tmp_path / "threshold-b.png"},
        ),
    }

    selected = _select_threshold_levels_by_blank_proportion(["roi_a", "roi_b"], contexts, 0.08)
    assert selected == {"roi_a": 2, "roi_b": 3}


def test_save_combined_spots_plot_creates_grid(monkeypatch, tmp_path: Path) -> None:
    params = SpotThresholdParams(subsample=5, figsize_spots=(4.0, 4.0), dpi=100)
    interpolator = _make_interpolator()
    contours = types.SimpleNamespace(levels=np.linspace(0.0, 1.0, 6))
    curve = ThresholdCurve(levels=[1], spot_counts=[10], blank_proportions=[0.1], max_level=5)

    contexts: dict[str, ROIThresholdContext] = {}
    for idx in range(3):
        spots = pl.DataFrame({"x": np.arange(20) + idx, "y": np.arange(20)})
        contexts[f"roi_{idx}"] = ROIThresholdContext(
            spots=spots,
            contours=contours,  # type: ignore[arg-type]
            interpolator=interpolator,
            curve=curve,
            artifact_paths={},
        )

    captured: dict[str, object] = {}

    def fake_savefig(self, path, *args, **kwargs):  # noqa: ANN001
        captured.update({"fig": self, "path": Path(path)})

    monkeypatch.setattr(Figure, "savefig", fake_savefig, raising=False)

    path = _save_combined_spots_plot(contexts, tmp_path, "cb1", params)

    expected_path = (tmp_path / "spots_final" / "spots_all--cb1.png").resolve()
    assert path == expected_path
    assert captured["path"] == expected_path

    fig: Figure = captured["fig"]  # type: ignore[assignment]
    assert len(fig.axes) == 3

    for idx, ax in enumerate(fig.axes):
        scatter = ax.collections[0]
        assert len(scatter.get_offsets()) == 5
        scale_bars = [artist for artist in ax.artists if isinstance(artist, AnchoredSizeBar)]
        if idx == 0:
            assert scale_bars, f"Expected scale bar on axis {idx}"
        else:
            assert not scale_bars, f"Unexpected scale bar on axis {idx}"

    assert fig._suptitle is not None  # type: ignore[attr-defined]


def test_threshold_cli_fixed_blank_proportion_skips_prompt(monkeypatch, tmp_path: Path) -> None:
    workspace_root = tmp_path / "ws"
    output_root = workspace_root / "analysis" / "output"
    output_root.mkdir(parents=True)

    codebook_path = tmp_path / "cb.json"
    codebook_path.write_text("{}")

    class DummyWorkspace:
        def __init__(self, root: Path):
            self.path = Path(root)
            self.output = types.SimpleNamespace(root=output_root)
            self.rois = ["alpha", "beta"]

        def resolve_rois(self, rois: list[str]) -> list[str]:
            return rois

        def decoded_spots_parquet(self, roi: str, codebook: str) -> Path:
            return self.path / "analysis" / "deconv" / f"{roi}+{codebook}" / "missing.parquet"

        def threshold_parquet(self, roi: str, codebook: str, *, raw: bool = False, output_dir: Path | None = None) -> Path:
            base = (output_dir / "parquets") if output_dir is not None else (self.output.root / "parquets")
            suffix = ".raw.parquet" if raw else ".parquet"
            return base / f"{roi}+{codebook}{suffix}"

    class DummyCodebook:
        def __init__(self, path: Path):
            self.path = Path(path)
            self.name = Path(path).stem

        def blank_stats(self) -> tuple[int, int]:
            return 100, 10

        def to_dataframe(self) -> pl.DataFrame:
            return pl.DataFrame(schema={"target": pl.Utf8, "bit0": pl.UInt8, "bit1": pl.UInt8, "bit2": pl.UInt8})

    def fake_load_spots_data(_path: Path, roi: str, _codebook: DummyCodebook, *, output_dir: Path | None = None) -> pl.DataFrame:
        return pl.DataFrame({"roi": [roi], "x_": [0.1], "y_": [0.1], "is_blank": [roi == "beta"]})

    def fake_apply_initial_filters(spots: pl.DataFrame, _rng: np.random.Generator, _params: SpotThresholdParams) -> pl.DataFrame:
        return spots

    def fake_calculate_density_map(spots: pl.DataFrame, _params: SpotThresholdParams):  # noqa: ANN001
        interpolator = _make_interpolator()
        contours = types.SimpleNamespace(levels=np.linspace(0.0, 1.0, 10))
        surface = types.SimpleNamespace()
        return object(), contours, interpolator, surface

    def fake_compute_threshold_curve(
        _spots: pl.DataFrame,
        _contours,  # noqa: ANN001
        _interp,  # noqa: ANN001
        *,
        point_densities: np.ndarray | None = None,  # noqa: ARG001
    ) -> ThresholdCurve:
        if bool(_spots["roi"][0] == "alpha"):
            return ThresholdCurve(levels=[1, 3, 5], spot_counts=[10, 8, 6], blank_proportions=[0.05, 0.1, 0.2], max_level=9)
        return ThresholdCurve(levels=[1, 3, 5], spot_counts=[12, 9, 7], blank_proportions=[0.02, 0.07, 0.15], max_level=9)

    def fail_prompt(*_args: object, **_kwargs: object) -> dict[str, int]:
        raise AssertionError("_prompt_threshold_levels should not be called when --fixed-blank-proportion is set")

    captured: dict[str, object] = {}

    def capture_summary_json(  # noqa: ANN001
        _output_dir: Path,
        *,
        contexts: dict[str, ROIThresholdContext],
        selected_levels: dict[str, int],
        codebook_name: str,
        n_total_codes: int,
        n_blank_codes: int,
        params: SpotThresholdParams,
    ) -> Path:
        captured["selected_levels"] = selected_levels
        return tmp_path / "summary.json"

    monkeypatch.setattr("fishtools.preprocess.cli_spotlook.Workspace", DummyWorkspace)
    monkeypatch.setattr("fishtools.preprocess.cli_spotlook.Codebook", DummyCodebook)
    monkeypatch.setattr("fishtools.preprocess.cli_spotlook._load_spots_data", fake_load_spots_data)
    monkeypatch.setattr("fishtools.preprocess.cli_spotlook._apply_initial_filters", fake_apply_initial_filters)
    monkeypatch.setattr("fishtools.preprocess.cli_spotlook._calculate_density_map", fake_calculate_density_map)
    monkeypatch.setattr("fishtools.preprocess.cli_spotlook._compute_threshold_curve", fake_compute_threshold_curve)
    monkeypatch.setattr("fishtools.preprocess.cli_spotlook._create_spots_contours_figure", lambda *_a, **_k: None)
    monkeypatch.setattr("fishtools.preprocess.cli_spotlook.save_figure", lambda *_a, **_k: None)
    monkeypatch.setattr("fishtools.preprocess.cli_spotlook._save_threshold_plot", lambda *_a, **_k: tmp_path / "threshold.png")
    monkeypatch.setattr("fishtools.preprocess.cli_spotlook._save_fdr_diagnostic_plots", lambda *_a, **_k: None)
    monkeypatch.setattr("fishtools.preprocess.cli_spotlook._save_combined_threshold_plot", lambda *_a, **_k: tmp_path / "combined.png")
    monkeypatch.setattr("fishtools.preprocess.cli_spotlook._apply_final_filter", lambda spots, *_a, **_k: spots)
    monkeypatch.setattr("fishtools.preprocess.cli_spotlook._generate_final_outputs", lambda *_a, **_k: None)
    monkeypatch.setattr(
        "fishtools.preprocess.cli_spotlook._save_combined_spots_plot",
        lambda *_a, **_k: tmp_path / "spots_all.png",
    )
    monkeypatch.setattr("fishtools.preprocess.cli_spotlook._write_spotlook_summary_json", capture_summary_json)
    monkeypatch.setattr("fishtools.preprocess.cli_spotlook._prompt_threshold_levels", fail_prompt)
    monkeypatch.setattr("fishtools.preprocess.cli_spotlook.setup_cli_logging", lambda *_a, **_k: None)

    runner = CliRunner()
    result = runner.invoke(
        threshold,
        [
            str(workspace_root),
            "-c",
            str(codebook_path),
            "*",
            "--fixed-blank-proportion",
            "0.08",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    assert captured["selected_levels"] == {"alpha": 2, "beta": 3}


def test_save_combined_spots_plot_clamps_dpi(monkeypatch, tmp_path: Path) -> None:
    # Extremely high requested DPI would exceed the Agg backend pixel limit
    params = SpotThresholdParams(subsample=10, dpi=20000)
    interpolator = _make_interpolator()
    contours = types.SimpleNamespace(levels=np.linspace(0.0, 1.0, 6))
    curve = ThresholdCurve(levels=[1], spot_counts=[10], blank_proportions=[0.1], max_level=5)

    spots = pl.DataFrame({"x": np.arange(10), "y": np.arange(10)})
    contexts: dict[str, ROIThresholdContext] = {
        "roi_0": ROIThresholdContext(
            spots=spots,
            contours=contours,  # type: ignore[arg-type]
            interpolator=interpolator,
            curve=curve,
            artifact_paths={},
        )
    }

    captured: dict[str, object] = {}

    def fake_savefig(self, path, *args, **kwargs):  # noqa: ANN001
        captured.update({"fig": self, "path": Path(path), "dpi": kwargs.get("dpi")})

    monkeypatch.setattr(Figure, "savefig", fake_savefig, raising=False)

    path = _save_combined_spots_plot(contexts, tmp_path, "cb1", params)

    expected_path = (tmp_path / "spots_final" / "spots_all--cb1.png").resolve()
    assert path == expected_path
    assert captured["path"] == expected_path

    # With 1 panel of 4x4 inches, the maximum safe dpi is floor(65535/4) == 16383
    assert isinstance(captured["dpi"], int)
    assert captured["dpi"] <= 16383


def test_save_combined_threshold_plot_sets_line_styles(monkeypatch, tmp_path: Path) -> None:
    params = SpotThresholdParams()
    curve_a = ThresholdCurve(levels=[1, 3], spot_counts=[100, 80], blank_proportions=[0.2, 0.1], max_level=4)
    curve_b = ThresholdCurve(levels=[1, 3], spot_counts=[90, 70], blank_proportions=[0.3, 0.15], max_level=4)
    curves = {"roi_b": curve_b, "roi_a": curve_a}
    fdr_blank_prop = 0.02
    selected_levels = {"roi_a": 3, "roi_b": 3}
    selected_spot_counts = {"roi_a": 80, "roi_b": 70}

    captured: dict[str, object] = {}

    def fake_savefig(self, path, *args, **kwargs):  # noqa: ANN001
        captured.update({"fig": self, "path": Path(path)})

    monkeypatch.setattr(Figure, "savefig", fake_savefig, raising=False)

    path = _save_combined_threshold_plot(
        curves,
        tmp_path,
        "cb1",
        params,
        fdr_blank_proportion=fdr_blank_prop,
        selected_levels=selected_levels,
        selected_spot_counts=selected_spot_counts,
    )

    assert path == (tmp_path / "threshold_selection" / "threshold_selection_all+cb1.png").resolve()
    assert captured["path"] == path

    fig = captured["fig"]
    ax_counts = next(ax for ax in fig.axes if ax.get_ylabel() == "Remaining Spots (normalized)")
    ax_blank = next(ax for ax in fig.axes if ax.get_ylabel() == "Blank Proportion")
    ax_diff = next(ax for ax in fig.axes if ax.get_ylabel() == "Δ Spots per Step (normalized)")
    ax_diff_blank = next(ax for ax in fig.axes if ax.get_ylabel() == "Δ Blank Proportion")

    count_lines = [line for line in ax_counts.lines if line.get_marker() in (None, "None")]
    assert all(line.get_linestyle() == "-" for line in count_lines)
    assert all(line.get_linestyle() == "--" for line in ax_blank.lines)
    assert all(line.get_linestyle() == "-" for line in ax_diff.lines)
    assert all(line.get_linestyle() == "--" for line in ax_diff_blank.lines)
    assert ax_counts.get_ylim()[0] == 0
    assert ax_counts.get_ylim()[1] <= 1.05

    expected_a = np.array([1.0, 0.8])
    expected_b = np.array([1.0, 70 / 90])
    actual_counts = [np.array(line.get_ydata()) for line in count_lines]
    assert any(np.allclose(y, expected_a) for y in actual_counts)
    assert any(np.allclose(y, expected_b) for y in actual_counts)

    stars = [line for line in ax_counts.lines if line.get_marker() == "*" and line.get_linestyle() == "None"]
    assert len(stars) == 2
    star_points = sorted((int(line.get_xdata()[0]), float(line.get_ydata()[0])) for line in stars)
    assert star_points == pytest.approx([(3, 70 / 90), (3, 0.8)])

    assert ax_diff_blank.get_ylim()[0] == 0.0
    assert ax_diff_blank.get_ylim()[1] == pytest.approx(0.15)
    blank_deltas = sorted(float(line.get_ydata()[0]) for line in ax_diff_blank.lines)
    assert blank_deltas == pytest.approx([0.1, 0.15])

    fdr_lines = [line for line in ax_blank.lines if line.get_label() == "1% FDR"]
    assert len(fdr_lines) == 1
    fdr_line = fdr_lines[0]
    assert np.allclose(fdr_line.get_ydata(), np.array([fdr_blank_prop, fdr_blank_prop]))

    roi_legends = [artist for artist in ax_counts.artists if isinstance(artist, Legend)]
    assert roi_legends, "Expected ROI legend to be attached to the axis"
    legend_labels = [text.get_text() for text in roi_legends[0].get_texts()]
    assert legend_labels == ["roi_a", "roi_b"]


def test_save_threshold_plot_draws_fdr_reference_line(monkeypatch, tmp_path: Path) -> None:
    params = SpotThresholdParams()
    curve = ThresholdCurve(levels=[1, 3], spot_counts=[100, 80], blank_proportions=[0.2, 0.1], max_level=4)
    fdr_blank_prop = 0.02

    captured: dict[str, object] = {}

    def fake_save_figure(fig, output_dir, name, roi, codebook, *, dpi=300, log_level="DEBUG"):  # noqa: ANN001
        captured.update({"fig": fig, "output_dir": output_dir, "name": name, "roi": roi, "codebook": codebook, "dpi": dpi})
        return (Path(output_dir) / f"{name}--{roi}+{codebook}.png").resolve()

    monkeypatch.setattr("fishtools.preprocess.cli_spotlook.save_figure", fake_save_figure)

    path = _save_threshold_plot(
        curve,
        tmp_path,
        "roi_a",
        "cb1",
        params,
        fdr_blank_proportion=fdr_blank_prop,
        selected_level=3,
        selected_spot_count=80,
    )

    assert path == (tmp_path / "threshold_selection" / "threshold_selection--roi_a+cb1.png").resolve()
    fig: Figure = captured["fig"]  # type: ignore[assignment]
    ax_blank = next(ax for ax in fig.axes if ax.get_ylabel() == "Blank Proportion")
    fdr_lines = [line for line in ax_blank.lines if line.get_label() == "1% FDR"]
    assert len(fdr_lines) == 1
    fdr_line = fdr_lines[0]
    assert np.allclose(fdr_line.get_ydata(), np.array([fdr_blank_prop, fdr_blank_prop]))

    ax_counts = next(ax for ax in fig.axes if ax.get_ylabel() == "Number of Spots")
    stars = [line for line in ax_counts.lines if line.get_marker() == "*" and line.get_linestyle() == "None"]
    assert len(stars) == 1
    star = stars[0]
    assert int(star.get_xdata()[0]) == 3
    assert float(star.get_ydata()[0]) == pytest.approx(80)

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


def test_write_spotlook_summary_json_emits_expected_fields(tmp_path: Path) -> None:
    params = SpotThresholdParams()
    contours = types.SimpleNamespace(levels=np.array([0.1, 0.2, 0.3]))

    spots = pl.DataFrame({
        "x_": [1.0, 2.0, 3.0, 4.0],
        "y_": [0.1, 0.2, 0.3, 0.4],
        "is_blank": [False, True, False, True],
    })
    filtered = pl.DataFrame({
        "x_": [1.0, 3.0, 4.0],
        "y_": [0.1, 0.3, 0.4],
        "is_blank": [False, False, True],
    })

    ctx = ROIThresholdContext(
        spots=spots,
        contours=contours,  # type: ignore[arg-type]
        interpolator=_make_interpolator(),
        curve=ThresholdCurve(levels=[1], spot_counts=[3], blank_proportions=[1 / 3], max_level=2),
        artifact_paths={},
        spots_final=filtered,
    )

    out_path = _write_spotlook_summary_json(
        tmp_path,
        contexts={"roi1": ctx},
        selected_levels={"roi1": 1},
        codebook_name="cb1",
        n_total_codes=100,
        n_blank_codes=20,
        params=params,
    )

    assert out_path == (tmp_path / "summary.json")
    row = json.loads(out_path.read_text())[0]
    assert row["roi"] == "roi1"
    assert row["codebook"] == "cb1"
    assert row["total_spots"] == 4
    assert row["total_blank"] == 2
    assert row["total_non_blank"] == 2
    assert row["threshold_level"] == 1
    assert row["threshold_value"] == pytest.approx(0.2)
    assert row["filtered_spots"] == 3
    assert row["filtered_blank"] == 1
    assert row["filtered_blank_proportion"] == pytest.approx(1 / 3)
    assert row["filtered_non_blank"] == 2
    assert row["fdr_estimated"] == pytest.approx(min((1 / 3) * (100 / 20), 1.0))
    assert "threshold_params" in row
    assert row["threshold_params"]["area_min"] == pytest.approx(params.area_min)
    assert "dpi" not in row["threshold_params"]


def test_write_spotlook_summary_json_merges_existing(tmp_path: Path) -> None:
    params = SpotThresholdParams()
    contours = types.SimpleNamespace(levels=np.array([0.1, 0.2, 0.3]))

    existing_path = tmp_path / "summary.json"
    existing_path.write_text(
        json.dumps(
            [
                {"roi": "roi_keep", "total_spots": 1},
                {"roi": "roi1", "total_spots": 999},
            ],
            indent=2,
        )
        + "\n"
    )

    spots = pl.DataFrame({"x_": [1.0], "y_": [0.1], "is_blank": [True]})
    ctx = ROIThresholdContext(
        spots=spots,
        contours=contours,  # type: ignore[arg-type]
        interpolator=_make_interpolator(),
        curve=ThresholdCurve(levels=[1], spot_counts=[1], blank_proportions=[1.0], max_level=2),
        artifact_paths={},
        spots_final=spots,
    )

    out_path = _write_spotlook_summary_json(
        tmp_path,
        contexts={"roi1": ctx},
        selected_levels={"roi1": 1},
        codebook_name="cb1",
        n_total_codes=10,
        n_blank_codes=2,
        params=params,
    )

    assert out_path == existing_path
    merged = json.loads(out_path.read_text())
    assert [item["roi"] for item in merged] == ["roi_keep", "roi1"]
    assert merged[0]["total_spots"] == 1
    assert merged[1]["total_spots"] == 1
    assert merged[1]["threshold_value"] == pytest.approx(0.2)


def test_threshold_cli_accepts_roi_argument(monkeypatch, tmp_path: Path) -> None:
    workspace_root = tmp_path / "ws"
    workspace_root.mkdir()
    codebook_path = tmp_path / "cb.json"
    codebook_path.write_text("{}")

    class DummyWorkspace:
        last_resolved: list[str] | None = None

        def __init__(self, root: Path):
            self.path = Path(root)
            self.rois = ["alpha", "beta"]
            self.output = types.SimpleNamespace(root=self.path / "analysis" / "output")

        def decoded_spots_parquet(self, roi: str, codebook: str) -> Path:
            return self.path / f"{roi}+{codebook}.parquet"

        def resolve_rois(self, rois: list[str]) -> list[str]:
            DummyWorkspace.last_resolved = list(rois)
            return [roi for roi in rois if roi in self.rois]

    class DummyCodebook:
        def __init__(self, path: Path):
            self.path = Path(path)
            self.name = Path(path).stem

        def blank_stats(self) -> tuple[int, int]:
            return 1, 0

        def to_dataframe(self) -> pl.DataFrame:
            return pl.DataFrame()

    def fake_load_spots_data(
        path: Path, roi: str, codebook: DummyCodebook, *, output_dir: Path | None = None
    ) -> pl.DataFrame | None:
        return None

    monkeypatch.setattr("fishtools.preprocess.cli_spotlook.Workspace", DummyWorkspace)
    monkeypatch.setattr("fishtools.preprocess.cli_spotlook.Codebook", DummyCodebook)
    monkeypatch.setattr("fishtools.preprocess.cli_spotlook._load_spots_data", fake_load_spots_data)

    runner = CliRunner()
    result = runner.invoke(
        threshold,
        [
            str(workspace_root),
            "-c",
            str(codebook_path),
            "alpha",
            "beta",
        ],
        catch_exceptions=True,
    )

    assert isinstance(result.exception, SystemExit)
    assert result.exit_code == 1
    assert DummyWorkspace.last_resolved == ["alpha", "beta"]


def test_load_spots_data_falls_back_to_output_copy(monkeypatch, tmp_path: Path) -> None:
    workspace_root = tmp_path / "ws"
    output_root = workspace_root / "analysis" / "output"
    (output_root / "parquets").mkdir(parents=True)

    roi = "alpha"
    codebook_name = "cb"
    raw_copy_path = output_root / "parquets" / f"{roi}+{codebook_name}.raw.parquet"
    pl.DataFrame({"target": ["GeneA", "Blank1"]}).write_parquet(raw_copy_path)

    class DummyWorkspace:
        def __init__(self, root: Path):
            self.path = Path(root)
            self.output = types.SimpleNamespace(root=output_root)

        def decoded_spots_parquet(self, roi: str, codebook: str) -> Path:
            return self.path / "analysis" / "deconv" / f"{roi}+{codebook}" / "missing.parquet"

        def threshold_parquet(self, roi: str, codebook: str, *, raw: bool = False, output_dir: Path | None = None) -> Path:
            base = (output_dir / "parquets") if output_dir is not None else (self.output.root / "parquets")
            suffix = ".raw.parquet" if raw else ".parquet"
            return base / f"{roi}+{codebook}{suffix}"

    class DummyCodebook:
        def __init__(self, _path: Path):
            self.name = codebook_name

        def to_dataframe(self) -> pl.DataFrame:
            return pl.DataFrame(schema={"target": pl.Utf8, "bit0": pl.UInt8, "bit1": pl.UInt8, "bit2": pl.UInt8})

    monkeypatch.setattr("fishtools.preprocess.cli_spotlook.Workspace", DummyWorkspace)

    spots = _load_spots_data(workspace_root, roi, DummyCodebook(tmp_path / "cb.json"))

    assert spots is not None
    assert set(spots.columns) >= {"roi", "is_blank", "target"}
    assert spots.sort("target")["is_blank"].to_list() == [True, False]


def test_load_spots_data_does_not_fall_back_to_thresholded_parquet(monkeypatch, tmp_path: Path) -> None:
    workspace_root = tmp_path / "ws"
    output_root = workspace_root / "analysis" / "output"
    (output_root / "parquets").mkdir(parents=True)

    roi = "alpha"
    codebook_name = "cb"
    thresholded_path = output_root / "parquets" / f"{roi}+{codebook_name}.parquet"
    pl.DataFrame({"target": ["GeneA"]}).write_parquet(thresholded_path)

    class DummyWorkspace:
        def __init__(self, root: Path):
            self.path = Path(root)
            self.output = types.SimpleNamespace(root=output_root)

        def decoded_spots_parquet(self, roi: str, codebook: str) -> Path:
            return self.path / "analysis" / "deconv" / f"{roi}+{codebook}" / "missing.parquet"

        def threshold_parquet(self, roi: str, codebook: str, *, raw: bool = False, output_dir: Path | None = None) -> Path:
            base = (output_dir / "parquets") if output_dir is not None else (self.output.root / "parquets")
            suffix = ".raw.parquet" if raw else ".parquet"
            return base / f"{roi}+{codebook}{suffix}"

    class DummyCodebook:
        def __init__(self, _path: Path):
            self.name = codebook_name

        def to_dataframe(self) -> pl.DataFrame:
            return pl.DataFrame(schema={"target": pl.Utf8, "bit0": pl.UInt8, "bit1": pl.UInt8, "bit2": pl.UInt8})

    monkeypatch.setattr("fishtools.preprocess.cli_spotlook.Workspace", DummyWorkspace)

    spots = _load_spots_data(workspace_root, roi, DummyCodebook(tmp_path / "cb.json"))

    assert spots is None


def test_load_spots_data_falls_back_to_legacy_output_root_copy(monkeypatch, tmp_path: Path) -> None:
    workspace_root = tmp_path / "ws"
    output_root = workspace_root / "analysis" / "output"
    output_root.mkdir(parents=True)

    roi = "alpha"
    codebook_name = "cb"
    raw_copy_path = output_root / f"{roi}+{codebook_name}.raw.parquet"
    pl.DataFrame({"target": ["GeneA", "Blank1"]}).write_parquet(raw_copy_path)

    class DummyWorkspace:
        def __init__(self, root: Path):
            self.path = Path(root)
            self.output = types.SimpleNamespace(root=output_root)

        def decoded_spots_parquet(self, roi: str, codebook: str) -> Path:
            return self.path / "analysis" / "deconv" / f"{roi}+{codebook}" / "missing.parquet"

        def threshold_parquet(self, roi: str, codebook: str, *, raw: bool = False, output_dir: Path | None = None) -> Path:
            base = (output_dir / "parquets") if output_dir is not None else (self.output.root / "parquets")
            suffix = ".raw.parquet" if raw else ".parquet"
            return base / f"{roi}+{codebook}{suffix}"

    class DummyCodebook:
        def __init__(self, _path: Path):
            self.name = codebook_name

        def to_dataframe(self) -> pl.DataFrame:
            return pl.DataFrame(schema={"target": pl.Utf8, "bit0": pl.UInt8, "bit1": pl.UInt8, "bit2": pl.UInt8})

    monkeypatch.setattr("fishtools.preprocess.cli_spotlook.Workspace", DummyWorkspace)

    spots = _load_spots_data(workspace_root, roi, DummyCodebook(tmp_path / "cb.json"))

    assert spots is not None
    assert set(spots.columns) >= {"roi", "is_blank", "target"}
    assert spots.sort("target")["is_blank"].to_list() == [True, False]
