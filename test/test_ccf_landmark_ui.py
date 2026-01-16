from __future__ import annotations

import types

import numpy as np
import pytest


def test_pick_atlas_slice_idx_returns_latest_slider_value(monkeypatch: pytest.MonkeyPatch) -> None:
    import fishtools.ccf.landmark_ui as ui

    captured: dict[str, object] = {}
    original_slider = ui.Slider

    class CapturingSlider(original_slider):  # type: ignore[misc]
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(*args, **kwargs)
            captured["slider"] = self

    def _show() -> None:
        slider = captured["slider"]
        assert isinstance(slider, original_slider)
        slider.set_val(7)
        fig = ui.plt.gcf()
        ui.plt.close(fig)

    monkeypatch.setattr(ui, "Slider", CapturingSlider)
    monkeypatch.setattr(ui.plt, "show", _show)

    atlas_reference = np.zeros((10, 8, 8), dtype=np.float32)
    picker = ui.pick_atlas_slice_idx(atlas_reference_zyx=atlas_reference, initial_idx=2)
    assert picker.idx == 7


def test_pick_rotation_deg_returns_latest_slider_value(monkeypatch: pytest.MonkeyPatch) -> None:
    import fishtools.ccf.landmark_ui as ui

    captured: dict[str, object] = {}
    original_slider = ui.Slider

    class CapturingSlider(original_slider):  # type: ignore[misc]
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(*args, **kwargs)
            captured["slider"] = self

    def _show() -> None:
        slider = captured["slider"]
        assert isinstance(slider, original_slider)
        slider.set_val(10)
        fig = ui.plt.gcf()
        ui.plt.close(fig)

    monkeypatch.setattr(ui, "Slider", CapturingSlider)
    monkeypatch.setattr(ui.plt, "show", _show)

    moving = np.zeros((20, 20), dtype=np.float32)
    picker = ui.pick_rotation_deg(moving_image_yx=moving, initial_deg=0, step_deg=2)
    assert picker.deg == 10


def test_pick_rotation_deg_flip_x_toggle_updates_picker_state(monkeypatch: pytest.MonkeyPatch) -> None:
    import fishtools.ccf.landmark_ui as ui

    captured: dict[str, object] = {}
    original_slider = ui.Slider
    original_button = ui.Button

    class CapturingSlider(original_slider):  # type: ignore[misc]
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(*args, **kwargs)
            captured["slider"] = self

    class CapturingButton(original_button):  # type: ignore[misc]
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(*args, **kwargs)
            captured["button"] = self

        def on_clicked(self, func):  # type: ignore[no-untyped-def]
            captured["button_callback"] = func
            return super().on_clicked(func)

    def _show() -> None:
        cb = captured.get("button_callback")
        assert callable(cb)
        cb(None)
        fig = ui.plt.gcf()
        ui.plt.close(fig)

    monkeypatch.setattr(ui, "Slider", CapturingSlider)
    monkeypatch.setattr(ui, "Button", CapturingButton)
    monkeypatch.setattr(ui.plt, "show", _show)

    moving = np.zeros((20, 20), dtype=np.float32)
    picker = ui.pick_rotation_deg(moving_image_yx=moving, initial_deg=0, initial_flip_x=False)
    assert picker.flip_x is True


def test_pick_paired_landmarks_undo_button_click_removes_last_pair(monkeypatch: pytest.MonkeyPatch) -> None:
    import matplotlib.backend_bases

    import fishtools.ccf.landmark_ui as ui

    captured: dict[str, object] = {}
    original_mpl_connect = matplotlib.backend_bases.FigureCanvasBase.mpl_connect

    def _capturing_mpl_connect(self, event: str, callback):  # type: ignore[no-untyped-def]
        if event == "button_press_event":
            captured["button_press_event"] = callback
        return original_mpl_connect(self, event, callback)

    def _show() -> None:
        fig = ui.plt.gcf()
        cb = captured["button_press_event"]
        assert callable(cb)

        # Axes order: fixed, moving, then the button axes.
        ax_fixed, ax_moving, ax_undo, _ax_clear, _ax_save = fig.axes

        def click(inaxes, xdata: float | None = None, ydata: float | None = None) -> None:
            event = types.SimpleNamespace(button=1, inaxes=inaxes, xdata=xdata, ydata=ydata)
            cb(event)

        click(ax_fixed, xdata=10.0, ydata=12.0)
        click(ax_moving, xdata=4.0, ydata=5.0)
        click(ax_undo)

    monkeypatch.setattr(matplotlib.backend_bases.FigureCanvasBase, "mpl_connect", _capturing_mpl_connect)
    monkeypatch.setattr(ui.plt, "show", _show)

    picker = ui.pick_paired_landmarks(
        fixed_image_yx=np.zeros((10, 10), dtype=np.float32),
        moving_image_yx_preview=np.zeros((10, 10), dtype=np.float32),
        moving_downsample=2,
        min_pairs=0,
    )
    fixed, moving = picker.get_points(min_pairs=0)
    assert fixed == []
    assert moving == []


def test_pick_paired_landmarks_save_button_click_sets_status(monkeypatch: pytest.MonkeyPatch) -> None:
    import matplotlib.backend_bases

    import fishtools.ccf.landmark_ui as ui

    captured: dict[str, object] = {}
    original_mpl_connect = matplotlib.backend_bases.FigureCanvasBase.mpl_connect

    def _capturing_mpl_connect(self, event: str, callback):  # type: ignore[no-untyped-def]
        if event == "button_press_event":
            captured["button_press_event"] = callback
        return original_mpl_connect(self, event, callback)

    def _show() -> None:
        fig = ui.plt.gcf()
        cb = captured["button_press_event"]
        assert callable(cb)

        ax_fixed, ax_moving, _ax_undo, _ax_clear, ax_save = fig.axes

        def click(inaxes, xdata: float | None = None, ydata: float | None = None) -> None:
            event = types.SimpleNamespace(button=1, inaxes=inaxes, xdata=xdata, ydata=ydata)
            cb(event)

        click(ax_fixed, xdata=10.0, ydata=12.0)
        click(ax_moving, xdata=4.0, ydata=5.0)
        click(ax_save)

        assert fig._suptitle is not None
        assert "Saved" in fig._suptitle.get_text()
        ui.plt.close(fig)

    monkeypatch.setattr(matplotlib.backend_bases.FigureCanvasBase, "mpl_connect", _capturing_mpl_connect)
    monkeypatch.setattr(ui.plt, "show", _show)

    def _on_change(fixed, moving):  # type: ignore[no-untyped-def]
        return None

    ui.pick_paired_landmarks(
        fixed_image_yx=np.zeros((10, 10), dtype=np.float32),
        moving_image_yx_preview=np.zeros((10, 10), dtype=np.float32),
        moving_downsample=2,
        min_pairs=0,
        on_change=_on_change,
    )
