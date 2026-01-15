from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.widgets import Button, Slider
from scipy.ndimage import rotate as ndimage_rotate

from fishtools.ccf.landmark import PointXY


def _robust_clim(
    image_yx: np.ndarray,
    *,
    low_percentile: float = 1.0,
    high_percentile: float = 99.8,
    max_samples: int = 200_000,
) -> tuple[float, float]:
    arr = np.asarray(image_yx, dtype=np.float32)
    flat = arr.ravel()
    if flat.size == 0:
        return 0.0, 1.0

    step = max(1, int(flat.size // max_samples))
    sample = flat[::step]
    sample = sample[np.isfinite(sample)]
    if sample.size == 0:
        return 0.0, 1.0

    vmin, vmax = np.percentile(sample, [low_percentile, high_percentile])
    vmin_f = float(vmin)
    vmax_f = float(vmax)
    if not np.isfinite(vmin_f) or not np.isfinite(vmax_f):
        return 0.0, 1.0
    if vmin_f == vmax_f:
        # Avoid singular clim (imshow warns and the image looks empty).
        return vmin_f, vmin_f + 1.0
    return vmin_f, vmax_f


@dataclass(frozen=True, slots=True)
class AtlasSliceIdxPicker:
    fig: Figure
    slider: Slider
    z_min: int
    z_max: int

    @property
    def idx(self) -> int:
        return int(np.clip(int(round(float(self.slider.val))), self.z_min, self.z_max))

    def close(self) -> None:
        plt.close(self.fig)


def pick_atlas_slice_idx(
    *,
    atlas_reference_zyx: np.ndarray,
    moving_image_yx: np.ndarray | None = None,
    initial_idx: int = 0,
    z_min_idx: int = 0,
    z_max_idx: int | None = None,
) -> AtlasSliceIdxPicker:
    """Create an interactive atlas Z-slice picker.

    Intended for notebook/VSCode interactive use (non-blocking): returns a picker object whose
    `.idx` reflects the current slider value. If `moving_image_yx` is provided, it is shown
    on the right as a fixed reference to help select the matching atlas slice.
    """

    z_max_available = int(atlas_reference_zyx.shape[0]) - 1
    if z_max_available < 0:
        raise ValueError(f"Expected a non-empty atlas volume, got shape={atlas_reference_zyx.shape}.")

    z_min = int(np.clip(int(z_min_idx), 0, z_max_available))
    if z_max_idx is None:
        z_max = z_max_available
    else:
        z_max = int(np.clip(int(z_max_idx), z_min, z_max_available))

    idx0 = int(np.clip(int(initial_idx), z_min, z_max))

    if moving_image_yx is None:
        fig, ax_ref = plt.subplots(1, 1, figsize=(7, 7))
        axes = (ax_ref,)
    else:
        fig, (ax_ref, ax_moving) = plt.subplots(1, 2, figsize=(10, 6))
        axes = (ax_ref, ax_moving)

    plt.subplots_adjust(bottom=0.18)

    atlas_slice0 = atlas_reference_zyx[idx0]
    vmin0, vmax0 = _robust_clim(atlas_slice0)
    im_ref = axes[0].imshow(atlas_slice0, cmap="gray", vmin=vmin0, vmax=vmax0)
    axes[0].set_title(f"Atlas reference (z={idx0})")
    axes[0].axis("off")

    if moving_image_yx is not None:
        axes[1].imshow(moving_image_yx, cmap="gray")
        axes[1].set_title("Sample (fixed)")
        axes[1].axis("off")

    ax_slider = plt.axes([0.15, 0.08, 0.6, 0.04])
    slider = Slider(
        ax=ax_slider,
        label="Atlas z",
        valmin=z_min,
        valmax=z_max,
        valinit=idx0,
        valstep=1,
    )

    def _update(*args: object) -> None:
        z = int(round(float(slider.val)))
        atlas_slice = atlas_reference_zyx[z]
        vmin, vmax = _robust_clim(atlas_slice)
        im_ref.set_data(atlas_slice)
        im_ref.set_clim(vmin=vmin, vmax=vmax)
        axes[0].set_title(f"Atlas reference (z={z})")
        fig.canvas.draw_idle()

    slider.on_changed(_update)

    _update()

    plt.show()
    return AtlasSliceIdxPicker(fig=fig, slider=slider, z_min=z_min, z_max=z_max)


@dataclass(frozen=True, slots=True)
class RotationDegPicker:
    fig: Figure
    slider: Slider
    flip_button: Button
    flip_state: dict[str, bool]

    @property
    def deg(self) -> int:
        return int(round(float(self.slider.val)))

    @property
    def flip_x(self) -> bool:
        return bool(self.flip_state.get("flip_x", False))

    def close(self) -> None:
        plt.close(self.fig)


def pick_rotation_deg(
    *,
    moving_image_yx: np.ndarray,
    initial_deg: int = 0,
    initial_flip_x: bool = False,
    step_deg: int = 2,
    vmin_deg: int = -180,
    vmax_deg: int = 180,
) -> RotationDegPicker:
    """Create an interactive coarse in-plane rotation picker for the moving image.

    Intended for notebook/VSCode interactive use (non-blocking): returns a picker object whose
    `.deg` reflects the current slider value.
    """

    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    plt.subplots_adjust(bottom=0.18)

    im_moving = ax.imshow(moving_image_yx, cmap="gray")
    ax.set_title("Sample (MOVING)")
    ax.axis("off")

    ax_slider = plt.axes([0.15, 0.08, 0.5, 0.04])
    slider = Slider(
        ax=ax_slider,
        label="Rotation (°)",
        valmin=float(vmin_deg),
        valmax=float(vmax_deg),
        valinit=float(initial_deg),
        valstep=float(step_deg),
    )

    flip_state: dict[str, bool] = {"flip_x": bool(initial_flip_x)}

    ax_flip = plt.axes([0.68, 0.08, 0.27, 0.04])
    flip_button = Button(ax_flip, "Flip X: OFF" if not flip_state["flip_x"] else "Flip X: ON")

    def _update(*args: object) -> None:
        deg = int(round(float(slider.val)))
        moving = moving_image_yx[:, ::-1] if flip_state["flip_x"] else moving_image_yx
        rotated = ndimage_rotate(moving, deg, reshape=False, order=1)
        im_moving.set_data(rotated)
        flip_tag = " +flipX" if flip_state["flip_x"] else ""
        ax.set_title(f"Sample (MOVING) — rot {deg}°{flip_tag}")
        fig.canvas.draw_idle()

    slider.on_changed(_update)

    def _toggle_flip(*args: object) -> None:
        flip_state["flip_x"] = not flip_state["flip_x"]
        flip_button.label.set_text("Flip X: ON" if flip_state["flip_x"] else "Flip X: OFF")
        _update()

    flip_button.on_clicked(_toggle_flip)

    _update()

    plt.show()
    return RotationDegPicker(fig=fig, slider=slider, flip_button=flip_button, flip_state=flip_state)


@dataclass(frozen=True, slots=True)
class PairedLandmarksPicker:
    fig: Figure
    fixed_points: list[PointXY]
    moving_points_fullres_xy_in_rotated_crop: list[PointXY]
    min_pairs: int

    @property
    def fixed_points_cropped_xy(self) -> list[PointXY]:
        return list(self.fixed_points)

    def get_points(self, *, min_pairs: int | None = None) -> tuple[list[PointXY], list[PointXY]]:
        min_pairs_ = self.min_pairs if min_pairs is None else int(min_pairs)
        if len(self.fixed_points) != len(self.moving_points_fullres_xy_in_rotated_crop):
            raise ValueError("Finish the pair: click the MOVING point before proceeding.")
        if len(self.fixed_points) < min_pairs_:
            raise ValueError(f"Need at least {min_pairs_} landmark pairs, got {len(self.fixed_points)}.")
        return list(self.fixed_points), list(self.moving_points_fullres_xy_in_rotated_crop)

    def close(self) -> None:
        plt.close(self.fig)


def pick_paired_landmarks(
    *,
    fixed_image_yx: np.ndarray,
    moving_image_yx_preview: np.ndarray,
    moving_downsample: int,
    initial_fixed_points_cropped_xy: list[PointXY] | None = None,
    initial_moving_points_fullres_xy_in_rotated_crop: list[PointXY] | None = None,
    min_pairs: int = 3,
    on_change: Callable[[list[PointXY], list[PointXY]], None] | None = None,
    fixed_title: str = "FIXED (atlas)",
    moving_title: str = "MOVING (sample)",
) -> PairedLandmarksPicker:
    """Paired landmark picker (operator-friendly; no keyboard shortcuts).

    Enforces click order: FIXED point N then MOVING point N. Provides 3 buttons:
    - Undo last pair
    - Clear all
    - Save landmarks (calls `on_change` with the current points)

    Power-user shortcut:
    - Ctrl+click near an existing marker to delete that pair (removes both FIXED+MOVING points).
    """

    moving_downsample_ = int(moving_downsample)
    if moving_downsample_ <= 0:
        raise ValueError(f"moving_downsample must be >0, got {moving_downsample_}")

    fixed_points: list[PointXY] = list(initial_fixed_points_cropped_xy or [])
    moving_points_fullres: list[PointXY] = list(initial_moving_points_fullres_xy_in_rotated_crop or [])

    if len(fixed_points) != len(moving_points_fullres):
        fixed_points = []
        moving_points_fullres = []

    next_side: str = "fixed"
    ctrl_state: dict[str, bool] = {"down": False}

    fig, (ax_fixed, ax_moving) = plt.subplots(1, 2, figsize=(16, 8))
    plt.subplots_adjust(bottom=0.16)

    ax_fixed.imshow(fixed_image_yx, cmap="gray")
    ax_fixed.axis("off")

    ax_moving.imshow(moving_image_yx_preview, cmap="gray")
    ax_moving.axis("off")

    scatter_fixed = ax_fixed.scatter([], [], c="red", s=80, marker="x")
    scatter_moving = ax_moving.scatter([], [], c="cyan", s=80, marker="x")
    fixed_texts: list[plt.Text] = []
    moving_texts: list[plt.Text] = []

    status = fig.suptitle("", fontsize=11, fontweight="bold")

    ax_undo = plt.axes([0.15, 0.06, 0.2, 0.05])
    Button(ax_undo, "Undo last pair")

    ax_clear = plt.axes([0.38, 0.06, 0.2, 0.05])
    Button(ax_clear, "Clear all")

    ax_save = plt.axes([0.61, 0.06, 0.2, 0.05])
    Button(ax_save, "Save landmarks")

    def _moving_points_preview() -> list[PointXY]:
        return [(x / moving_downsample_, y / moving_downsample_) for x, y in moving_points_fullres]

    def _set_status(msg: str) -> None:
        status.set_text(msg)

    def _update_artists(*, msg: str | None = None) -> None:
        nonlocal fixed_texts, moving_texts

        if fixed_points:
            xs, ys = zip(*fixed_points)
            scatter_fixed.set_offsets(np.c_[xs, ys])
        else:
            scatter_fixed.set_offsets(np.empty((0, 2)))

        moving_preview = _moving_points_preview()
        if moving_preview:
            xs, ys = zip(*moving_preview)
            scatter_moving.set_offsets(np.c_[xs, ys])
        else:
            scatter_moving.set_offsets(np.empty((0, 2)))

        for t in fixed_texts:
            t.remove()
        for t in moving_texts:
            t.remove()
        fixed_texts = []
        moving_texts = []

        for i, (x, y) in enumerate(fixed_points, start=1):
            fixed_texts.append(
                ax_fixed.annotate(
                    str(i),
                    (x, y),
                    xytext=(4, 4),
                    textcoords="offset points",
                    color="red",
                    fontsize=9,
                    weight="bold",
                )
            )

        for i, (x, y) in enumerate(moving_preview, start=1):
            moving_texts.append(
                ax_moving.annotate(
                    str(i),
                    (x, y),
                    xytext=(4, 4),
                    textcoords="offset points",
                    color="cyan",
                    fontsize=9,
                    weight="bold",
                )
            )

        ax_fixed.set_title(f"{fixed_title}: {len(fixed_points)}")
        ax_moving.set_title(f"{moving_title}: {len(moving_points_fullres)}")

        if on_change is not None:
            on_change(list(fixed_points), list(moving_points_fullres))

        if msg is not None:
            _set_status(msg)
        else:
            needed = max(0, min_pairs - len(fixed_points))
            if next_side == "fixed":
                _set_status(f"Next: click {fixed_title} for point #{len(fixed_points) + 1}  (need {needed} more after this)")
            else:
                _set_status(f"Next: click {moving_title} for point #{len(moving_points_fullres) + 1}")

        fig.canvas.draw_idle()

    def _ctrl_pressed(event) -> bool:
        key = getattr(event, "key", None)
        # Some interactive backends (notably ipympl/widget) don't reliably populate `event.key` on mouse events,
        # so we also track a key-down latch via key_press/key_release events.
        if key is None:
            return bool(ctrl_state["down"])

        # Matplotlib backends use "control" (common) or "ctrl" (some toolkits).
        parts = {p.strip() for p in str(key).lower().split("+")}
        return bool(parts & {"control", "ctrl"}) or bool(ctrl_state["down"])

    def _on_key_press(event) -> None:
        key = getattr(event, "key", None)
        if key is None:
            return
        parts = {p.strip() for p in str(key).lower().split("+")}
        if parts & {"control", "ctrl"}:
            ctrl_state["down"] = True

    def _on_key_release(event) -> None:
        key = getattr(event, "key", None)
        if key is None:
            return
        parts = {p.strip() for p in str(key).lower().split("+")}
        if parts & {"control", "ctrl"}:
            ctrl_state["down"] = False

    def _closest_index(points: list[PointXY], *, x: float, y: float) -> tuple[int | None, float]:
        if not points:
            return None, float("inf")
        xs = np.array([p[0] for p in points], dtype=np.float32)
        ys = np.array([p[1] for p in points], dtype=np.float32)
        d2 = (xs - float(x)) ** 2 + (ys - float(y)) ** 2
        idx = int(d2.argmin())
        return idx, float(np.sqrt(float(d2[idx])))

    def _delete_pair_at(index: int) -> None:
        nonlocal next_side

        # Full pair exists: remove both sides.
        if 0 <= index < len(moving_points_fullres) and 0 <= index < len(fixed_points):
            fixed_points.pop(index)
            moving_points_fullres.pop(index)
        # Incomplete pair (FIXED clicked, waiting for MOVING): allow deleting that last FIXED point.
        elif next_side == "moving" and index == len(moving_points_fullres) and index < len(fixed_points):
            fixed_points.pop(index)
        else:
            _update_artists(msg="No pair at that index to delete.")
            return

        if len(fixed_points) == len(moving_points_fullres):
            next_side = "fixed"
        elif len(fixed_points) == len(moving_points_fullres) + 1:
            next_side = "moving"
        else:
            # Should not happen, but keep UI usable.
            next_side = "fixed"

        _update_artists(msg=f"Deleted pair #{index + 1}.")

    def _maybe_ctrl_click_delete(event) -> bool:
        if not _ctrl_pressed(event):
            return False
        if event.inaxes not in {ax_fixed, ax_moving}:
            return False
        if event.xdata is None or event.ydata is None:
            return True

        # Threshold in image pixel coordinates; avoids accidental deletions.
        tol_px = 12.0

        if event.inaxes == ax_fixed:
            idx, dist = _closest_index(fixed_points, x=float(event.xdata), y=float(event.ydata))
        else:
            idx, dist = _closest_index(_moving_points_preview(), x=float(event.xdata), y=float(event.ydata))

        if idx is None or dist > tol_px:
            _update_artists(msg="Ctrl+click closer to an existing marker to delete a pair.")
            return True

        _delete_pair_at(idx)
        return True

    def _on_click(event) -> None:
        nonlocal next_side

        if event.button != 1:
            return
        if event.inaxes == ax_undo:
            _undo()
            return
        if event.inaxes == ax_clear:
            _clear()
            return
        if event.inaxes == ax_save:
            _save()
            return
        if event.xdata is None or event.ydata is None:
            return
        if _maybe_ctrl_click_delete(event):
            return

        if next_side == "fixed":
            if event.inaxes != ax_fixed:
                _update_artists(msg=f"Click {fixed_title} first for point #{len(fixed_points) + 1}.")
                return
            fixed_points.append((float(event.xdata), float(event.ydata)))
            next_side = "moving"
            _update_artists()
            return

        if event.inaxes != ax_moving:
            _update_artists(msg=f"Click {moving_title} next for point #{len(moving_points_fullres) + 1}.")
            return
        moving_points_fullres.append(
            (float(event.xdata) * moving_downsample_, float(event.ydata) * moving_downsample_)
        )
        next_side = "fixed"
        _update_artists()

    def _undo(*args: object) -> None:
        nonlocal next_side

        if next_side == "moving":
            if fixed_points:
                fixed_points.pop()
            next_side = "fixed"
            _update_artists()
            return

        if fixed_points and moving_points_fullres:
            fixed_points.pop()
            moving_points_fullres.pop()
            _update_artists()
            return

        _update_artists(msg="Nothing to undo.")

    def _clear(*args: object) -> None:
        nonlocal next_side

        fixed_points.clear()
        moving_points_fullres.clear()
        next_side = "fixed"
        _update_artists()

    def _save(*args: object) -> None:
        if on_change is None:
            _set_status("No save callback configured.")
            fig.canvas.draw_idle()
            return
        if len(fixed_points) != len(moving_points_fullres):
            _set_status("Finish the pair (click MOVING) before saving.")
            fig.canvas.draw_idle()
            return
        on_change(list(fixed_points), list(moving_points_fullres))
        _set_status(f"Saved {len(fixed_points)} landmark pair(s).")
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_press_event", _on_click)
    fig.canvas.mpl_connect("key_press_event", _on_key_press)
    fig.canvas.mpl_connect("key_release_event", _on_key_release)

    _update_artists()
    plt.show()
    return PairedLandmarksPicker(
        fig=fig,
        fixed_points=fixed_points,
        moving_points_fullres_xy_in_rotated_crop=moving_points_fullres,
        min_pairs=min_pairs,
    )


@dataclass(frozen=True, slots=True)
class OverlayPairedLandmarksPicker:
    fig: Figure
    fixed_points: list[PointXY]
    moving_points_in_fixed_cropped_xy: list[PointXY]
    min_pairs: int

    def get_points(self, *, min_pairs: int | None = None) -> tuple[list[PointXY], list[PointXY]]:
        min_pairs_ = self.min_pairs if min_pairs is None else int(min_pairs)
        if len(self.fixed_points) != len(self.moving_points_in_fixed_cropped_xy):
            raise ValueError("Finish the pair: click the sample point before proceeding.")
        if len(self.fixed_points) < min_pairs_:
            raise ValueError(f"Need at least {min_pairs_} landmark pairs, got {len(self.fixed_points)}.")
        return list(self.fixed_points), list(self.moving_points_in_fixed_cropped_xy)

    def close(self) -> None:
        plt.close(self.fig)


def pick_paired_landmarks_overlay(
    *,
    overlay_image_yx_rgb: np.ndarray,
    initial_fixed_points_cropped_xy: list[PointXY] | None = None,
    initial_moving_points_in_fixed_cropped_xy: list[PointXY] | None = None,
    min_pairs: int = 3,
    on_save: Callable[[list[PointXY], list[PointXY]], None] | None = None,
    title: str = "Overlay + landmarks",
    fixed_label: str = "Atlas point",
    moving_label: str = "Sample point",
) -> OverlayPairedLandmarksPicker:
    """Paired landmark editor on a single overlay axis.

    Enforces click order: first click adds the FIXED (atlas) point, second click adds the MOVING (sample) point.

    Power-user shortcut:
    - Ctrl+click near an existing marker (either side) to delete that pair.
    """

    fixed_points: list[PointXY] = list(initial_fixed_points_cropped_xy or [])
    moving_points: list[PointXY] = list(initial_moving_points_in_fixed_cropped_xy or [])
    if len(fixed_points) != len(moving_points):
        fixed_points = []
        moving_points = []

    next_side: str = "fixed"
    ctrl_state: dict[str, bool] = {"down": False}

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    plt.subplots_adjust(bottom=0.16)

    overlay_rgb = np.asarray(overlay_image_yx_rgb, dtype=np.float32)
    if overlay_rgb.ndim != 3 or overlay_rgb.shape[2] != 3:
        raise ValueError(f"Expected overlay_image_yx_rgb to have shape (y,x,3), got {overlay_rgb.shape}.")

    overlay_rgb_no_moving = overlay_rgb.copy()
    overlay_rgb_no_moving[..., 1] = 0.0
    moving_visible: dict[str, bool] = {"visible": True}

    im = ax.imshow(overlay_rgb)
    ax.axis("off")
    status = fig.suptitle("", fontsize=11, fontweight="bold")

    scatter_fixed = ax.scatter([], [], s=35, c="deepskyblue", marker="x", linewidths=2, alpha=0.6)
    scatter_moving = ax.scatter([], [], s=35, facecolors="none", edgecolors="yellow", marker="o", linewidths=2, alpha=0.6)
    fixed_texts: list[plt.Text] = []
    moving_texts: list[plt.Text] = []
    connector_lines: list[plt.Line2D] = []

    ax_undo = plt.axes([0.12, 0.06, 0.22, 0.05])
    Button(ax_undo, "Undo last pair")

    ax_clear = plt.axes([0.36, 0.06, 0.22, 0.05])
    Button(ax_clear, "Clear all")

    ax_save = plt.axes([0.60, 0.06, 0.28, 0.05])
    Button(ax_save, "Save landmarks")

    ax_toggle_moving = plt.axes([0.90, 0.06, 0.09, 0.05])
    toggle_moving_button = Button(ax_toggle_moving, "Hide sample")

    def _set_status(msg: str) -> None:
        status.set_text(msg)

    def _ctrl_pressed(event) -> bool:
        key = getattr(event, "key", None)
        if key is None:
            return bool(ctrl_state["down"])
        parts = {p.strip() for p in str(key).lower().split("+")}
        return bool(parts & {"control", "ctrl"}) or bool(ctrl_state["down"])

    def _on_key_press(event) -> None:
        key = getattr(event, "key", None)
        if key is None:
            return
        parts = {p.strip() for p in str(key).lower().split("+")}
        if parts & {"control", "ctrl"}:
            ctrl_state["down"] = True

    def _on_key_release(event) -> None:
        key = getattr(event, "key", None)
        if key is None:
            return
        parts = {p.strip() for p in str(key).lower().split("+")}
        if parts & {"control", "ctrl"}:
            ctrl_state["down"] = False

    def _closest_index(points: list[PointXY], *, x: float, y: float) -> tuple[int | None, float]:
        if not points:
            return None, float("inf")
        xs = np.array([p[0] for p in points], dtype=np.float32)
        ys = np.array([p[1] for p in points], dtype=np.float32)
        d2 = (xs - float(x)) ** 2 + (ys - float(y)) ** 2
        idx = int(d2.argmin())
        return idx, float(np.sqrt(float(d2[idx])))

    def _update_artists(*, msg: str | None = None) -> None:
        nonlocal fixed_texts, moving_texts, connector_lines

        im.set_data(overlay_rgb if moving_visible["visible"] else overlay_rgb_no_moving)

        if fixed_points:
            xs, ys = zip(*fixed_points)
            scatter_fixed.set_offsets(np.c_[xs, ys])
        else:
            scatter_fixed.set_offsets(np.empty((0, 2)))

        if moving_points:
            xs, ys = zip(*moving_points)
            scatter_moving.set_offsets(np.c_[xs, ys])
        else:
            scatter_moving.set_offsets(np.empty((0, 2)))

        for t in fixed_texts:
            t.remove()
        for t in moving_texts:
            t.remove()
        for ln in connector_lines:
            ln.remove()
        fixed_texts = []
        moving_texts = []
        connector_lines = []

        for i, (x, y) in enumerate(fixed_points, start=1):
            fixed_texts.append(
                ax.annotate(
                    f"F{i}",
                    (x, y),
                    xytext=(4, 4),
                    textcoords="offset points",
                    color="deepskyblue",
                    fontsize=9,
                    weight="bold",
                )
            )

        for i, (x, y) in enumerate(moving_points, start=1):
            moving_texts.append(
                ax.annotate(
                    f"M{i}",
                    (x, y),
                    xytext=(4, 4),
                    textcoords="offset points",
                    color="yellow",
                    fontsize=9,
                    weight="bold",
                )
            )

        for (x_fixed, y_fixed), (x_moving, y_moving) in zip(fixed_points, moving_points):
            connector_lines.append(ax.plot([x_fixed, x_moving], [y_fixed, y_moving], color="white", linewidth=1, alpha=0.5)[0])

        ax.set_title(f"{title}: {len(fixed_points)} pair(s)", pad=2)

        if msg is not None:
            _set_status(msg)
        else:
            needed = max(0, min_pairs - len(fixed_points))
            if next_side == "fixed":
                _set_status(f"Next: click {fixed_label} for pair #{len(fixed_points) + 1} (need {needed} more after this).")
            else:
                _set_status(f"Next: click {moving_label} for pair #{len(moving_points) + 1}.")

        fig.canvas.draw_idle()

    def _toggle_moving(*args: object) -> None:
        moving_visible["visible"] = not moving_visible["visible"]
        toggle_moving_button.label.set_text("Hide sample" if moving_visible["visible"] else "Show sample")
        _update_artists(msg="Sample visibility toggled.")

    def _delete_pair_at(index: int) -> None:
        nonlocal next_side

        if 0 <= index < len(moving_points) and 0 <= index < len(fixed_points):
            fixed_points.pop(index)
            moving_points.pop(index)
        elif next_side == "moving" and index == len(moving_points) and index < len(fixed_points):
            fixed_points.pop(index)
        else:
            _update_artists(msg="No pair at that index to delete.")
            return

        if len(fixed_points) == len(moving_points):
            next_side = "fixed"
        elif len(fixed_points) == len(moving_points) + 1:
            next_side = "moving"
        else:
            next_side = "fixed"

        _update_artists(msg=f"Deleted pair #{index + 1}.")

    def _maybe_ctrl_click_delete(event) -> bool:
        if not _ctrl_pressed(event):
            return False
        if event.inaxes != ax:
            return False
        if event.xdata is None or event.ydata is None:
            return True

        tol_px = 12.0
        idx_fixed, dist_fixed = _closest_index(fixed_points, x=float(event.xdata), y=float(event.ydata))
        idx_moving, dist_moving = _closest_index(moving_points, x=float(event.xdata), y=float(event.ydata))

        best_idx: int | None
        best_dist: float
        if dist_fixed <= dist_moving:
            best_idx, best_dist = idx_fixed, dist_fixed
        else:
            best_idx, best_dist = idx_moving, dist_moving

        if best_idx is None or best_dist > tol_px:
            _update_artists(msg="Ctrl+click closer to an existing marker to delete a pair.")
            return True

        _delete_pair_at(best_idx)
        return True

    def _undo(*args: object) -> None:
        nonlocal next_side

        if next_side == "moving":
            if fixed_points:
                fixed_points.pop()
            next_side = "fixed"
            _update_artists()
            return

        if fixed_points and moving_points:
            fixed_points.pop()
            moving_points.pop()
            _update_artists()
            return

        _update_artists(msg="Nothing to undo.")

    def _clear(*args: object) -> None:
        nonlocal next_side

        fixed_points.clear()
        moving_points.clear()
        next_side = "fixed"
        _update_artists()

    def _save(*args: object) -> None:
        if on_save is None:
            _set_status("No save callback configured.")
            fig.canvas.draw_idle()
            return
        if len(fixed_points) != len(moving_points):
            _set_status("Finish the pair (click sample) before saving.")
            fig.canvas.draw_idle()
            return
        on_save(list(fixed_points), list(moving_points))
        _set_status(f"Saved {len(fixed_points)} landmark pair(s).")
        fig.canvas.draw_idle()

    def _on_click(event) -> None:
        nonlocal next_side

        if event.button != 1:
            return
        if event.inaxes == ax_undo:
            _undo()
            return
        if event.inaxes == ax_clear:
            _clear()
            return
        if event.inaxes == ax_save:
            _save()
            return
        if event.inaxes == ax_toggle_moving:
            _toggle_moving()
            return
        if event.inaxes != ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        if _maybe_ctrl_click_delete(event):
            return

        if next_side == "fixed":
            fixed_points.append((float(event.xdata), float(event.ydata)))
            next_side = "moving"
            _update_artists()
            return

        moving_points.append((float(event.xdata), float(event.ydata)))
        next_side = "fixed"
        _update_artists()

    fig.canvas.mpl_connect("button_press_event", _on_click)
    fig.canvas.mpl_connect("key_press_event", _on_key_press)
    fig.canvas.mpl_connect("key_release_event", _on_key_release)

    _update_artists()
    plt.show()
    return OverlayPairedLandmarksPicker(
        fig=fig,
        fixed_points=fixed_points,
        moving_points_in_fixed_cropped_xy=moving_points,
        min_pairs=min_pairs,
    )
