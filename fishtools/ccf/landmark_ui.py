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

    @property
    def deg(self) -> int:
        return int(round(float(self.slider.val)))

    def close(self) -> None:
        plt.close(self.fig)


def pick_rotation_deg(
    *,
    moving_image_yx: np.ndarray,
    initial_deg: int = 0,
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

    ax_slider = plt.axes([0.15, 0.08, 0.6, 0.04])
    slider = Slider(
        ax=ax_slider,
        label="Rotation (°)",
        valmin=float(vmin_deg),
        valmax=float(vmax_deg),
        valinit=float(initial_deg),
        valstep=float(step_deg),
    )

    def _update(*args: object) -> None:
        deg = int(round(float(slider.val)))
        rotated = ndimage_rotate(moving_image_yx, deg, reshape=False, order=1)
        im_moving.set_data(rotated)
        ax.set_title(f"Sample (MOVING) — rot {deg}°")
        fig.canvas.draw_idle()

    slider.on_changed(_update)

    _update()

    plt.show()
    return RotationDegPicker(fig=fig, slider=slider)


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

    Enforces click order: FIXED point N then MOVING point N. Provides 2 buttons:
    - Undo last pair
    - Clear all
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

    fig, (ax_fixed, ax_moving) = plt.subplots(1, 2, figsize=(12, 6))
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
        if event.xdata is None or event.ydata is None:
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

    fig.canvas.mpl_connect("button_press_event", _on_click)

    _update_artists()
    plt.show()
    return PairedLandmarksPicker(
        fig=fig,
        fixed_points=fixed_points,
        moving_points_fullres_xy_in_rotated_crop=moving_points_fullres,
        min_pairs=min_pairs,
    )
