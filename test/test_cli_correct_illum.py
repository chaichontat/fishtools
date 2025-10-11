from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from fishtools.preprocess.cli_correct_illum import _mask_union_of_tiles, correct_illum
from fishtools.preprocess.spots.illumination import RangeFieldPointsModel


def _run_cli(args: list[str]) -> tuple[int, str]:
    runner = CliRunner(mix_stderr=False)
    result = runner.invoke(correct_illum, args, catch_exceptions=False)
    return result.exit_code, result.output


def test_range_field_points_model_roundtrip(tmp_path: Path) -> None:
    model = RangeFieldPointsModel(
        xy=np.array([[0.0, 0.0], [0.0, 10.0], [10.0, 0.0], [10.0, 10.0]], dtype=np.float32),
        vlow=np.full(4, 1.0, dtype=np.float32),
        vhigh=np.full(4, 2.0, dtype=np.float32),
        meta={
            "roi": "roi",
            "codebook": "cb",
            "channel": "c0",
            "kernel": "thin_plate_spline",
            "smoothing": 2.0,
            "neighbors": 32,
            "tile_w": 1024,
            "tile_h": 1024,
            "p_low_key": "0.1",
            "p_high_key": "99.9",
            "grid_step_suggest": 128.0,
        },
    )

    out_path = tmp_path / "field.npz"
    saved = model.to_npz(out_path)
    roundtrip = RangeFieldPointsModel.from_npz(saved)

    assert np.allclose(roundtrip.xy, model.xy)
    assert np.allclose(roundtrip.vlow, model.vlow)
    assert np.allclose(roundtrip.vhigh, model.vhigh)
    assert roundtrip.meta["p_low_key"] == "0.1"
    assert np.isclose(roundtrip.meta.get("range_mean", 0.0), 1.0)
    # New clamping defaults
    assert np.isclose(roundtrip.meta.get("range_min", 0.0), 1.0 / 3.0)
    assert np.isclose(roundtrip.meta.get("range_max", 0.0), 3.0)
    # Legacy key mirrors the minimum for backward compatibility
    assert np.isclose(roundtrip.meta.get("range_floor", 0.0), 1.0 / 3.0)

    xs, ys, low_field, high_field = roundtrip.evaluate(0.0, 0.0, 20.0, 20.0, grid_step=10.0, neighbors=4)

    assert xs.shape == (3,)
    assert ys.shape == (3,)
    assert low_field.shape == (3, 3)
    assert high_field.shape == (3, 3)
    assert np.allclose(low_field, 1.0)
    assert np.allclose(high_field - low_field, 1.0)


def test_range_field_points_model_from_subtiles(monkeypatch, tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    reg_dir = workspace / "analysis" / "deconv" / "registered--roi+cb"
    reg_dir.mkdir(parents=True)

    monkeypatch.setattr(
        "fishtools.preprocess.spots.illumination._load_tile_origins",
        lambda ws, roi: (np.array([0]), np.array([0.0]), np.array([0.0])),
    )

    def fake_gather(reg_dir, indices, origin_map, *, channel, p_low_key, p_high_key, suffix):
        xs = np.array([5.0], dtype=np.float32)
        ys = np.array([7.0], dtype=np.float32)
        return xs, ys, [1.0], [2.0], channel or "c0"

    def fake_infer(reg_dir, indices, *, subtile_suffix):
        return ["c0"], "0.1", "99.9", 1024, 1024, 128.0

    monkeypatch.setattr(RangeFieldPointsModel, "_gather_subtile_values", staticmethod(fake_gather))
    monkeypatch.setattr(RangeFieldPointsModel, "_infer_subtile_metadata", staticmethod(fake_infer))

    model = RangeFieldPointsModel.from_subtiles(workspace, "roi", "cb")

    assert model.meta["channel"] == "c0"
    assert model.meta["workspace"] == str(workspace)
    assert model.xy.shape == (1, 2)
    assert np.allclose(model.vlow, [1.0])
    assert np.allclose(model.vhigh, [2.0])


@pytest.fixture()
def workspace_with_registered(tmp_path: Path) -> Path:
    root = tmp_path / "workspace"
    reg_dir = root / "analysis" / "deconv" / "registered--roi+cb"
    reg_dir.mkdir(parents=True)
    (reg_dir / "reg-0000.tif").touch()
    return root


def test_calculate_percentiles_cpu(monkeypatch, workspace_with_registered: Path) -> None:
    calls: dict[str, int] = {"load": 0, "compute": 0}

    def fake_load(tile: Path, use_gpu: bool):
        calls["load"] += 1
        arr = np.zeros((2, 4, 8, 8), dtype=np.float32)
        channels = ["c0", "c1"]
        return arr, channels

    def fake_compute(arr: np.ndarray, percentiles: tuple[float, float], grid: int):
        calls["compute"] += 1
        lo = np.full((grid, grid, arr.shape[0]), 1.0, dtype=np.float32)
        hi = np.full((grid, grid, arr.shape[0]), 2.0, dtype=np.float32)
        return lo, hi

    monkeypatch.setenv("GPU", "0")
    from fishtools.preprocess import cli_correct_illum as module

    monkeypatch.setattr(module, "_load_czyx", fake_load)
    monkeypatch.setattr(module, "_compute_subtile_percentiles_cpu", fake_compute)

    exit_code, output = _run_cli(
        [
            "calculate-percentiles",
            str(workspace_with_registered),
            "roi",
            "--grid",
            "2",
            "--out-suffix",
            ".json",
            "--codebook",
            "cb",
        ]
    )

    assert exit_code == 0
    assert "Percentile JSONs" in output
    assert calls["load"] == 1
    assert calls["compute"] == 1

    reg_dir = workspace_with_registered / "analysis" / "deconv" / "registered--roi+cb"
    payload = json.loads((reg_dir / "reg-0000.json").read_text())
    assert set(payload.keys()) == {"c0", "c1"}
    entry = payload["c0"][0]
    assert entry["percentiles"]["0.1"] == 1.0
    assert entry["percentiles"]["99.9"] == 2.0


def test_field_generate_writes_npz(monkeypatch, workspace_with_registered: Path, tmp_path: Path) -> None:
    class DummyModel(RangeFieldPointsModel):
        saved = False

        def to_npz(self, out_path: Path) -> Path:  # type: ignore[override]
            DummyModel.saved = True
            out_path.write_bytes(b"NPZ")
            return out_path

    dummy = DummyModel(
        xy=np.zeros((1, 2), dtype=np.float32),
        vlow=np.zeros(1, dtype=np.float32),
        vhigh=np.ones(1, dtype=np.float32),
        meta={
            "roi": "roi",
            "codebook": "cb",
            "channel": "c0",
            "p_low_key": "0.1",
            "p_high_key": "99.9",
            "grid_step_suggest": 128.0,
        },
    )

    def fake_from_subtiles(*args, **kwargs):
        return dummy

    monkeypatch.setattr(RangeFieldPointsModel, "from_subtiles", staticmethod(fake_from_subtiles))

    out_npz = tmp_path / "field.npz"
    exit_code, output = _run_cli(
        [
            "field-generate",
            str(workspace_with_registered),
            "roi",
            "cb",
            "--output",
            str(out_npz),
        ]
    )

    assert exit_code == 0
    assert DummyModel.saved
    assert out_npz.read_bytes() == b"NPZ"
    assert "illumination field model" in output


def test_plot_field(monkeypatch, tmp_path: Path) -> None:
    model_path = tmp_path / "model.npz"
    model_path.write_bytes(b"NPZ")

    class DummyModel(RangeFieldPointsModel):
        @classmethod
        def from_npz(cls, path: Path):  # type: ignore[override]
            return dummy

        def evaluate(self, *_, **__):  # type: ignore[override]
            xs = np.array([0.0, 5.0], dtype=np.float32)
            ys = np.array([0.0, 5.0], dtype=np.float32)
            low = np.ones((2, 2), dtype=np.float32)
            high = np.full((2, 2), 2.0, dtype=np.float32)
            return xs, ys, low, high

    dummy = DummyModel(
        xy=np.array([[0.0, 0.0]], dtype=np.float32),
        vlow=np.array([1.0], dtype=np.float32),
        vhigh=np.array([2.0], dtype=np.float32),
        meta={
            "roi": "roi",
            "codebook": "cb",
            "channel": "c0",
            "tile_w": 10,
            "tile_h": 10,
            "p_low_key": "0.1",
            "p_high_key": "99.9",
        },
    )

    monkeypatch.setattr(RangeFieldPointsModel, "from_npz", classmethod(lambda cls, path: dummy))

    from fishtools.preprocess import cli_correct_illum as module

    monkeypatch.setattr(
        module,
        "_resolve_tile_origins",
        lambda meta, ws, roi: (
            np.array([0, 1], dtype=np.int32),
            np.array([0.0, 10.0], dtype=np.float64),
            np.array([0.0, 0.0], dtype=np.float64),
        ),
    )

    out_png = tmp_path / "field.png"
    exit_code, output = _run_cli(
        [
            "plot-field",
            str(model_path),
            "--workspace",
            str(tmp_path),
            "--roi",
            "roi",
            "--codebook",
            "cb",
            "--output",
            str(out_png),
        ]
    )

    assert exit_code == 0
    assert out_png.exists()
    assert "Saved illumination plot" in output


def test_mask_union_of_tiles() -> None:
    xs = np.array([1.0, 2.5, 4.0, 11.0])
    ys = np.array([1.0, 2.5, 6.0])
    xs0 = np.array([0.0, 10.0])
    ys0 = np.array([0.0, 0.0])

    mask = _mask_union_of_tiles(xs, ys, xs0, ys0, tile_w=5.0, tile_h=5.0)

    grid_x, grid_y = np.meshgrid(xs, ys, indexing="xy")
    expected = np.zeros_like(grid_x, dtype=bool)
    for xo, yo in zip(xs0, ys0, strict=False):
        expected |= (
            (grid_x >= xo)
            & (grid_x <= xo + 5.0)
            & (grid_y >= yo)
            & (grid_y <= yo + 5.0)
        )

    assert mask.shape == expected.shape
    assert np.array_equal(mask, expected)


def test_mask_union_of_tiles_matches_rect_union() -> None:
    xs = np.linspace(0.0, 4096.0, 33)
    ys = np.linspace(0.0, 4096.0, 33)
    xs0 = np.array([0.0, 2048.0])
    ys0 = np.array([0.0, 0.0])

    mask = _mask_union_of_tiles(xs, ys, xs0, ys0, tile_w=2048.0, tile_h=2048.0)

    grid_x, grid_y = np.meshgrid(xs, ys, indexing="xy")
    expected = np.zeros_like(grid_x, dtype=bool)
    for x_origin, y_origin in zip(xs0, ys0, strict=False):
        expected |= (
            (grid_x >= x_origin)
            & (grid_x <= x_origin + 2048.0)
            & (grid_y >= y_origin)
            & (grid_y <= y_origin + 2048.0)
        )

    assert mask.shape == expected.shape
    assert np.array_equal(mask, expected)


def test_range_field_points_model_field_patch(monkeypatch) -> None:
    class DummyModel(RangeFieldPointsModel):
        def evaluate(self, x0, y0, x1, y1, *, grid_step=None, neighbors=None, kernel=None, smoothing=None):  # type: ignore[override]
            xs = np.linspace(x0, x1, 5, dtype=np.float32)
            ys = np.linspace(y0, y1, 7, dtype=np.float32)
            X, Y = np.meshgrid(xs, ys, indexing="xy")
            low = X + Y
            high = low + 1.0
            return xs, ys, low.astype(np.float32), high.astype(np.float32)

    model = DummyModel(
        xy=np.zeros((1, 2), dtype=np.float32),
        vlow=np.zeros(1, dtype=np.float32),
        vhigh=np.ones(1, dtype=np.float32),
        meta={
            "tile_w": 4.0,
            "tile_h": 4.0,
            "tile_origins": [[0.0, 0.0], [4.0, 0.0]],
            "grid_step_suggest": 1.0,
        },
    )

    patch = model.field_patch(
        x0=0.0,
        y0=0.0,
        width=4,
        height=4,
        mode="range",
        tile_origins=[(0.0, 0.0), (4.0, 0.0)],
        tile_w=4.0,
        tile_h=4.0,
    )

    assert patch.shape == (4, 4)
    # Since high-low == 1 everywhere in DummyModel, expect all ones
    assert np.allclose(patch, 1.0)


def test_range_field_points_model_field_patch_normalizes_range(monkeypatch) -> None:
    class DummyModel(RangeFieldPointsModel):
        def evaluate(self, x0, y0, x1, y1, *, grid_step=None, neighbors=None, kernel=None, smoothing=None):  # type: ignore[override]
            xs = np.linspace(x0, x1, 5, dtype=np.float32)
            ys = np.linspace(y0, y1, 5, dtype=np.float32)
            X, Y = np.meshgrid(xs, ys, indexing="xy")
            low = X * 0.0
            high = low + (X + 2 * Y)
            return xs, ys, low.astype(np.float32), high.astype(np.float32)

    model = DummyModel(
        xy=np.zeros((1, 2), dtype=np.float32),
        vlow=np.zeros(1, dtype=np.float32),
        vhigh=np.full(1, 6.0, dtype=np.float32),
        meta={
            "tile_w": 4.0,
            "tile_h": 4.0,
            "grid_step_suggest": 1.0,
        },
    )

    assert pytest.approx(6.0) == model.meta.get("range_mean")
    # Defaults now clamp to [1/3, 3]
    assert pytest.approx(1.0 / 3.0) == model.meta.get("range_floor")

    low_patch = model.field_patch(
        x0=0.0,
        y0=0.0,
        width=4,
        height=4,
        mode="low",
        tile_origins=[(0.0, 0.0)],
        tile_w=4.0,
        tile_h=4.0,
    )

    high_patch = model.field_patch(
        x0=0.0,
        y0=0.0,
        width=4,
        height=4,
        mode="high",
        tile_origins=[(0.0, 0.0)],
        tile_w=4.0,
        tile_h=4.0,
    )

    patch = model.field_patch(
        x0=0.0,
        y0=0.0,
        width=4,
        height=4,
        mode="range",
        tile_origins=[(0.0, 0.0)],
        tile_w=4.0,
        tile_h=4.0,
    )

    assert patch.shape == (4, 4)
    norm = (high_patch - low_patch) / float(model.meta["range_mean"])
    norm = np.clip(norm, 1.0 / 3.0, 3.0)
    expected = 1.0 / norm
    expected /= expected.mean()
    expected = np.clip(expected, 1.0 / 3.0, 3.0)
    assert np.allclose(patch, expected)
    assert np.isclose(patch.mean(), 1.0, atol=1e-6)


def test_render_field_patch_cli(monkeypatch, tmp_path: Path) -> None:
    model_path = tmp_path / "field-model.npz"
    model_path.write_bytes(b"NPZ")

    call_args: dict[str, float | int | str] = {}

    class DummyModel(RangeFieldPointsModel):
        @classmethod
        def from_npz(cls, path: Path):  # type: ignore[override]
            return dummy

        def field_patch(self, *, x0: float, y0: float, width: int, height: int, mode: str = "range", **kwargs):  # type: ignore[override]
            call_args.update({
                "x0": x0,
                "y0": y0,
                "width": width,
                "height": height,
                "mode": mode,
            })
            X, Y = np.meshgrid(np.linspace(0, 1, width), np.linspace(0, 1, height), indexing="xy")
            return (X + Y).astype(np.float32)

    dummy = DummyModel(
        xy=np.zeros((1, 2), dtype=np.float32),
        vlow=np.zeros(1, dtype=np.float32),
        vhigh=np.ones(1, dtype=np.float32),
        meta={
            "roi": "roi",
            "codebook": "cb",
            "channel": "c0",
            "tile_w": 4.0,
            "tile_h": 4.0,
        },
    )

    monkeypatch.setattr(RangeFieldPointsModel, "from_npz", classmethod(lambda cls, path: dummy))

    from fishtools.preprocess import cli_correct_illum as module

    monkeypatch.setattr(
        module,
        "_resolve_tile_origins",
        lambda meta, ws, roi: (
            np.array([7, 42], dtype=np.int32),
            np.array([12.0, 100.0], dtype=np.float64),
            np.array([34.0, 200.0], dtype=np.float64),
        ),
    )

    out_png = tmp_path / "field.png"
    exit_code, output = _run_cli(
        [
            "plot-field-tile",
            str(model_path),
                "--tile",
                "42",
            "--width",
            "32",
            "--height",
            "32",
            "--output",
            str(out_png),
        ]
    )

    assert exit_code == 0
    assert out_png.exists()
    assert "Saved tile field plot" in output
    assert call_args["x0"] == 100.0
    assert call_args["y0"] == 200.0
    assert call_args["width"] == 32
    assert call_args["height"] == 32


def test_plot_field_tile_requires_location(monkeypatch, tmp_path: Path) -> None:
    model_path = tmp_path / "field-model.npz"
    model_path.write_bytes(b"NPZ")

    dummy = RangeFieldPointsModel(
        xy=np.zeros((1, 2), dtype=np.float32),
        vlow=np.zeros(1, dtype=np.float32),
        vhigh=np.ones(1, dtype=np.float32),
        meta={
            "roi": "roi",
            "codebook": "cb",
            "tile_w": 4.0,
            "tile_h": 4.0,
        },
    )

    monkeypatch.setattr(RangeFieldPointsModel, "from_npz", classmethod(lambda cls, path: dummy))

    runner = CliRunner(mix_stderr=False)
    result = runner.invoke(correct_illum, ["plot-field-tile", str(model_path)], catch_exceptions=False)

    assert result.exit_code != 0
    assert "Missing option" in (result.stderr or "")
