from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Tuple

import numpy as np
from scipy.interpolate import RBFInterpolator
from skimage.transform import resize

from fishtools.io.workspace import Workspace


def create_opt_path(
    *,
    codebook_path: Path,
    mode: str,  # "folder"
    roi: str | None = None,
    path_folder: Path | None = None,
) -> Path:
    """
    Return the optimization folder path following your convention:
      <path_folder>/opt_<codebook>{+<roi>}
    """
    if roi is None:
        roi = "*"
    if path_folder is None:
        raise ValueError("create_opt_path: path_folder is required")
    base = path_folder / f"opt_{codebook_path.stem}{f'+{roi}' if roi != '*' else ''}"
    if mode == "folder":
        base.mkdir(parents=True, exist_ok=True)
        return base
    raise ValueError(f"Unknown mode {mode}")


def load_codebook(
    path: Path,
    bit_mapping: dict[str, int],
    exclude: set[str] | None = None,
    simple: bool = False,
) -> Tuple[Any, list[int], np.ndarray, np.ndarray]:
    """
    Minimal codebook loader that mirrors your existing behavior sufficiently for illum collection:
      - returns used_bits, names, and zeroed-blanks mask (arr_zeroblank). The Codebook object is not used here,
        but we keep the signature for compatibility.
    """
    cb_json: dict[str, list[int]] = json.loads(path.read_text())
    for k in exclude or set():
        cb_json.pop(k, None)

    if simple:
        cb_json = {k: [v[0]] for k, v in cb_json.items()}

    # Filter to imaged bits only
    available_bits = sorted(bit_mapping)
    cb_json = {k: v for k, v in cb_json.items() if all(str(bit) in available_bits for bit in v)}
    if not cb_json:
        raise ValueError("No genes in codebook are imaged. Check codebook and bit mapping.")

    used_bits = sorted(set(b for bits in cb_json.values() for b in bits))
    names = np.array(list(cb_json.keys()))
    is_blank = np.array([n.startswith("Blank") for n in names])[:, None]
    arr = np.zeros((len(cb_json), len(used_bits)), dtype=bool)
    bit_index = {b: i for i, b in enumerate(used_bits)}
    for i, bits in enumerate(cb_json.values()):
        for b in bits:
            arr[i, bit_index[b]] = True
    arr_zeroblank = arr * ~is_blank
    return None, used_bits, names, arr_zeroblank


# --------------------------------------------------------------------------------------
# IO helpers for coordinates / fields
# --------------------------------------------------------------------------------------


@dataclass
class IllumFieldMeta:
    roi: str
    codebook: str
    kernel: str
    smoothing: float
    epsilon: float | None
    grid_step: float
    x0: float
    y0: float
    x1: float
    y1: float
    mode: str  # "divide" or "range"
    range_epsilon: float | None  # denominator floor when mode="range"


@dataclass
class RangeFieldPointsModel:
    xy: np.ndarray
    vlow: np.ndarray
    vhigh: np.ndarray
    meta: dict[str, Any]
    range_mean: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.xy = np.asarray(self.xy, dtype=np.float32)
        self.vlow = np.asarray(self.vlow, dtype=np.float32)
        self.vhigh = np.asarray(self.vhigh, dtype=np.float32)
        if not isinstance(self.meta, dict):
            self.meta = dict(self.meta)

        meta_value = self.meta.get("range_mean")
        candidate: float
        if meta_value is not None:
            try:
                candidate = float(meta_value)
            except (TypeError, ValueError):
                candidate = math.nan
        else:
            diff = self.vhigh.astype(np.float64, copy=False) - self.vlow.astype(np.float64, copy=False)
            candidate = float(diff.mean()) if diff.size else 1.0

        if not math.isfinite(candidate) or candidate <= 0.0:
            diff = self.vhigh.astype(np.float64, copy=False) - self.vlow.astype(np.float64, copy=False)
            candidate = float(diff.mean()) if diff.size else 1.0
        if not math.isfinite(candidate) or candidate <= 0.0:
            candidate = 1.0

        self.range_mean = candidate
        self.meta["range_mean"] = candidate
        # Clamp defaults: enforce multiplicative changes in [1/3, 3]
        # Keep backward compatibility with older artifacts that used only a floor.
        self.meta.setdefault("range_min", 1.0 / 3.0)
        self.meta.setdefault("range_max", 3.0)
        # Preserve legacy key but align default to the new minimum for readers that still consult it.
        self.meta.setdefault("range_floor", self.meta["range_min"])  # legacy compat

    @classmethod
    def from_subtiles(
        cls,
        workspace: Path | Workspace,
        roi: str,
        codebook: str,
        *,
        channel: str | None = None,
        kernel: str = "thin_plate_spline",
        smoothing: float = 2.0,
        subtile_suffix: str = ".subtiles.json",
    ) -> "RangeFieldPointsModel":
        ws = workspace if isinstance(workspace, Workspace) else Workspace(workspace)
        reg_dir = ws.registered(roi, Path(codebook).stem if Path(codebook).suffix else codebook)
        idx_array, xs0, ys0 = _load_tile_origins(ws, roi)
        indices = [int(i) for i in idx_array.tolist()]
        channels, p_low_key, p_high_key, tile_w, tile_h, grid_step = cls._infer_subtile_metadata(
            reg_dir,
            indices,
            subtile_suffix=subtile_suffix,
        )
        origin_map = {indices[i]: (float(xs0[i]), float(ys0[i])) for i in range(len(indices))}
        xs, ys, lows, highs, channel_used = cls._gather_subtile_values(
            reg_dir,
            indices,
            origin_map,
            channel=channel,
            p_low_key=p_low_key,
            p_high_key=p_high_key,
            suffix=subtile_suffix,
        )
        if xs.size == 0:
            raise RuntimeError("No subtile percentile JSON data found; run calc_percentile_gpu first.")

        meta: dict[str, Any] = {
            "roi": roi,
            "codebook": Path(codebook).stem if Path(codebook).suffix else codebook,
            "channel": channel_used,
            "kernel": kernel,
            "smoothing": float(smoothing),
            "epsilon": None,
            "neighbors": 64,
            "tile_w": int(tile_w),
            "tile_h": int(tile_h),
            "p_low_key": p_low_key,
            "p_high_key": p_high_key,
            "grid_step_suggest": float(grid_step),
            "subtile_suffix": subtile_suffix,
            "channels": channels,
            "workspace": str(ws.path),
        }

        return cls(
            xy=np.c_[xs, ys].astype(np.float32, copy=False),
            vlow=np.asarray(lows, dtype=np.float32),
            vhigh=np.asarray(highs, dtype=np.float32),
            meta=meta,
        )

    def to_npz(self, out_path: Path) -> Path:
        out_path = out_path.with_suffix(out_path.suffix or ".npz")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_path,
            xy=self.xy,
            vlow=self.vlow,
            vhigh=self.vhigh,
            meta=json.dumps(self.meta, indent=2).encode("utf-8"),
        )
        return out_path

    @classmethod
    def from_npz(cls, path: Path) -> "RangeFieldPointsModel":
        data = np.load(path)
        meta = json.loads(bytes(data["meta"]).decode("utf-8"))
        return cls(
            xy=data["xy"].astype(np.float32, copy=False),
            vlow=data["vlow"].astype(np.float32, copy=False),
            vhigh=data["vhigh"].astype(np.float32, copy=False),
            meta=meta,
        )

    def evaluate(
        self,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        *,
        grid_step: float | None = None,
        neighbors: int | None = None,
        kernel: str | None = None,
        smoothing: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        step = float(grid_step) if grid_step is not None else float(self.meta.get("grid_step_suggest", 192.0))
        nx = max(2, int(np.ceil((x1 - x0) / step)) + 1)
        ny = max(2, int(np.ceil((y1 - y0) / step)) + 1)
        xs = np.linspace(float(x0), float(x1), nx, dtype=np.float32)
        ys = np.linspace(float(y0), float(y1), ny, dtype=np.float32)

        xy = self.xy.astype(np.float64, copy=False)
        vlow = self.vlow.astype(np.float64, copy=False)
        vhigh = self.vhigh.astype(np.float64, copy=False)
        x_mean, x_std = float(xy[:, 0].mean()), float(xy[:, 0].std() or 1.0)
        y_mean, y_std = float(xy[:, 1].mean()), float(xy[:, 1].std() or 1.0)
        xy_n = np.c_[(xy[:, 0] - x_mean) / x_std, (xy[:, 1] - y_mean) / y_std]
        Xg, Yg = np.meshgrid(xs, ys, indexing="xy")
        grid = np.c_[(Xg.ravel() - x_mean) / x_std, (Yg.ravel() - y_mean) / y_std]

        ker = kernel or self.meta.get("kernel", "thin_plate_spline")
        smooth = float(smoothing if smoothing is not None else self.meta.get("smoothing", 1.0))
        neigh = int(neighbors) if neighbors is not None else int(self.meta.get("neighbors", 64))

        kwargs = {"neighbors": neigh} if neigh > 0 else {}
        rbfi_l = RBFInterpolator(xy_n, vlow, kernel=ker, smoothing=smooth, epsilon=None, **kwargs)
        rbfi_h = RBFInterpolator(xy_n, vhigh, kernel=ker, smoothing=smooth, epsilon=None, **kwargs)
        low_pred = rbfi_l(grid).reshape(Yg.shape)
        high_pred = rbfi_h(grid).reshape(Yg.shape)

        eps = 1e-3
        high_pred = np.maximum(high_pred, low_pred + eps)
        return xs, ys, low_pred.astype(np.float32, copy=False), high_pred.astype(np.float32, copy=False)

    def field_patch(
        self,
        *,
        x0: float,
        y0: float,
        width: int,
        height: int,
        mode: str = "range",
        tile_origins: Iterable[tuple[float, float]] | None = None,
        tile_w: float | None = None,
        tile_h: float | None = None,
        grid_step: float | None = None,
        neighbors: int | None = None,
        kernel: str | None = None,
        smoothing: float | None = None,
    ) -> np.ndarray:
        """Return a field patch sampled over ``width×height`` pixels starting at ``(x0, y0)``.

        ``mode`` controls which field is returned:
            - ``"low"``: low percentile surface
            - ``"high"``: high percentile surface
            - ``"range"`` (default): high-low difference
        """

        if width <= 0 or height <= 0:
            raise ValueError("width and height must be positive")

        if tile_origins is None:
            raise ValueError("tile_origins must be provided to extract field patches")

        origins_arr = np.asarray(list(tile_origins), dtype=np.float64)
        if origins_arr.size == 0:
            raise ValueError("tile_origins must be non-empty")

        tw = float(tile_w if tile_w is not None else self.meta.get("tile_w", 0.0))
        th = float(tile_h if tile_h is not None else self.meta.get("tile_h", 0.0))
        if tw <= 0.0 or th <= 0.0:
            raise ValueError("tile dimensions must be positive")

        xs0 = origins_arr[:, 0]
        ys0 = origins_arr[:, 1]
        x_min = float(xs0.min())
        y_min = float(ys0.min())
        x_max = float(xs0.max() + tw)
        y_max = float(ys0.max() + th)

        xs, ys, low_field, high_field = self.evaluate(
            x_min,
            y_min,
            x_max,
            y_max,
            grid_step=grid_step,
            neighbors=neighbors,
            kernel=kernel,
            smoothing=smoothing,
        )

        x1 = float(x0) + float(width)
        y1 = float(y0) + float(height)

        # Normalize globally over the union-of-tiles mask to avoid per-tile seams.
        # Build mask at the evaluated grid resolution so every tile shares the same denominator.
        def _mask_union_of_tiles_like(
            xs_arr: np.ndarray,
            ys_arr: np.ndarray,
            xs0_arr: np.ndarray,
            ys0_arr: np.ndarray,
            tw: float,
            th: float,
        ) -> np.ndarray:
            xs_arr = np.asarray(xs_arr, dtype=np.float64)
            ys_arr = np.asarray(ys_arr, dtype=np.float64)
            xs0_arr = np.asarray(xs0_arr, dtype=np.float64)
            ys0_arr = np.asarray(ys0_arr, dtype=np.float64)
            if xs_arr.size == 0 or ys_arr.size == 0 or xs0_arr.size == 0:
                return np.ones((ys_arr.size, xs_arr.size), dtype=bool)
            x1_arr = xs0_arr + float(tw)
            y1_arr = ys0_arr + float(th)
            dx = float(np.min(np.diff(xs_arr))) if xs_arr.size > 1 else 0.0
            dy = float(np.min(np.diff(ys_arr))) if ys_arr.size > 1 else 0.0
            tol = max(dx, dy, 1.0) * 1e-6
            X = (xs_arr[None, :] >= (xs0_arr[:, None] - tol)) & (xs_arr[None, :] <= (x1_arr[:, None] + tol))
            Y = (ys_arr[None, :] >= (ys0_arr[:, None] - tol)) & (ys_arr[None, :] <= (y1_arr[:, None] + tol))
            counts = Y.astype(np.uint16).T @ X.astype(np.uint16)
            return counts > 0

        mask = _mask_union_of_tiles_like(xs, ys, xs0, ys0, tw, th)

        mode_l = mode.lower()
        if mode_l == "low":
            field_global = low_field
        elif mode_l == "high":
            field_global = high_field
        elif mode_l == "range":
            field_global = self.range_correction(low_field, high_field, mask=mask)
        else:
            raise ValueError(f"Unsupported mode '{mode}'; expected 'low', 'high', or 'range'")

        patch = _slice_field_to_tile(
            field_global,
            xs,
            ys,
            float(x0),
            float(y0),
            x1,
            y1,
            (int(height), int(width)),
        )
        return patch
        raise ValueError(f"Unsupported mode '{mode}'; expected 'low', 'high', or 'range'")

    def range_correction(
        self,
        low_field: np.ndarray,
        high_field: np.ndarray,
        *,
        mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute the normalized inverse range correction field with clamping.

        Steps:
        - Normalize range by ``range_mean``
        - Clip normalized range to [range_min, range_max]
        - Invert and normalize to unit mean over ``mask`` (or globally if ``mask`` is None)
        - Clip the final multiplier again to [range_min, range_max]
        """
        scale = float(self.meta.get("range_mean", self.range_mean))
        if not math.isfinite(scale) or scale <= 0.0:
            scale = float(self.range_mean) if math.isfinite(self.range_mean) and self.range_mean > 0 else 1.0

        rmin = float(self.meta.get("range_min", self.meta.get("range_floor", 1.0 / 3.0)))
        rmax = float(self.meta.get("range_max", 3.0))
        if not math.isfinite(rmin) or rmin <= 0.0:
            rmin = 1.0 / 3.0
        if not math.isfinite(rmax) or rmax <= rmin:
            rmax = 3.0

        norm = (high_field.astype(np.float64, copy=False) - low_field.astype(np.float64, copy=False)) / float(
            scale
        )
        # Ensure strictly positive range and clip into bounds
        norm = np.clip(norm, rmin, rmax)

        inv = 1.0 / norm
        if mask is not None:
            m = np.asarray(mask, dtype=bool)
            denom = float(np.mean(inv[m])) if m.any() else float(np.mean(inv)) if inv.size else 1.0
        else:
            denom = float(np.mean(inv)) if inv.size else 1.0
        if not math.isfinite(denom) or denom <= 0.0:
            denom = 1.0
        inv /= denom
        # Final safety clamp to the same bounds
        inv = np.clip(inv, rmin, rmax)
        return inv.astype(np.float32, copy=False)

    @staticmethod
    def _gather_subtile_values(
        reg_dir: Path,
        indices: Iterable[int],
        origin_map: dict[int, tuple[float, float]],
        *,
        channel: Optional[str] = None,
        p_low_key: str = "0.1",
        p_high_key: str = "99.9",
        suffix: str = ".subtiles.json",
    ) -> tuple[np.ndarray, np.ndarray, list[float], list[float], str]:
        xs: list[float] = []
        ys: list[float] = []
        lows: list[float] = []
        highs: list[float] = []
        chosen_channel: Optional[str] = channel

        for idx in indices:
            jpath = reg_dir / f"reg-{int(idx):04d}{suffix}"
            if not jpath.exists():
                continue
            try:
                data = json.loads(jpath.read_text())
            except Exception:
                continue

            if chosen_channel is None:
                if not data:
                    continue
                # Preserve original channel order as written (matches TIFF order)
                for k in data.keys():
                    if isinstance(k, str):
                        chosen_channel = k
                        break
                if chosen_channel is None:
                    continue

            ch_entries = data.get(chosen_channel)
            if not isinstance(ch_entries, list):
                continue

            origin = origin_map.get(int(idx))
            if origin is None:
                continue

            x0, y0 = origin
            for entry in ch_entries:
                try:
                    ex = float(entry["x"])
                    ey = float(entry["y"])
                    exs = float(entry["x_size"])
                    eys = float(entry["y_size"])
                    pmap = entry.get("percentiles", {})
                    if p_low_key not in pmap or p_high_key not in pmap:
                        continue
                    lo = float(pmap[p_low_key])
                    hi = float(pmap[p_high_key])
                except Exception:
                    continue

                xs.append(x0 + ex + exs * 0.5)
                ys.append(y0 + ey + eys * 0.5)
                lows.append(lo)
                highs.append(hi)

        if chosen_channel is None:
            chosen_channel = channel or "unknown"

        return (
            np.asarray(xs, dtype=np.float32),
            np.asarray(ys, dtype=np.float32),
            lows,
            highs,
            chosen_channel,
        )

    @staticmethod
    def _infer_subtile_metadata(
        reg_dir: Path,
        indices: Iterable[int],
        *,
        subtile_suffix: str = ".subtiles.json",
    ) -> tuple[list[str], str, str, int, int, float]:
        for idx in indices:
            jpath = reg_dir / f"reg-{int(idx):04d}{subtile_suffix}"
            if not jpath.exists():
                continue
            data = json.loads(jpath.read_text())
            if not isinstance(data, dict) or not data:
                continue
            # Preserve channel order as written (matches TIFF order)
            channels = [k for k in data.keys() if isinstance(k, str)]
            if not channels:
                continue
            first_entries = data.get(channels[0])
            if not isinstance(first_entries, list) or not first_entries:
                continue
            try:
                tile_w = int(max((float(e["x"]) + float(e["x_size"])) for e in first_entries))
                tile_h = int(max((float(e["y"]) + float(e["y_size"])) for e in first_entries))
            except Exception:
                tile_w = tile_h = 1968
            try:
                xsizes = [float(e.get("x_size", 0.0)) for e in first_entries]
                ysizes = [float(e.get("y_size", 0.0)) for e in first_entries]
                med_x = (
                    float(np.median([v for v in xsizes if v > 0])) if any(v > 0 for v in xsizes) else 192.0
                )
                med_y = (
                    float(np.median([v for v in ysizes if v > 0])) if any(v > 0 for v in ysizes) else 192.0
                )
                base = float(min(med_x, med_y))
                grid_step = max(64.0, base / 2.0)
            except Exception:
                grid_step = 192.0
            pmap = first_entries[0].get("percentiles", {}) if first_entries else {}
            if isinstance(pmap, dict) and len(pmap) >= 2:
                try:
                    keys_sorted = sorted((str(k) for k in pmap.keys()), key=lambda s: float(s))
                except Exception:
                    keys_sorted = list(pmap.keys())
                p_low_key = keys_sorted[0]
                p_high_key = keys_sorted[-1]
            else:
                p_low_key, p_high_key = "0.1", "99.9"
            return channels, p_low_key, p_high_key, tile_w, tile_h, grid_step
        return [], "0.1", "99.9", 1968, 1968, 192.0


def _load_tile_origins(ws: Workspace, roi: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return arrays (indices, x0s, y0s) for tile origins from the Workspace tile configuration."""

    tc = ws.tileconfig(roi)
    df = tc.df
    idx = df.select("index").to_numpy().reshape(-1).astype(np.int32)
    xs0 = df.select("x").to_numpy().reshape(-1).astype(np.float64)
    ys0 = df.select("y").to_numpy().reshape(-1).astype(np.float64)
    return idx, xs0, ys0


# --------------------------------------------------------------------------------------
# Core math: fit RBF/TPS on scattered points and evaluate on grid
# --------------------------------------------------------------------------------------


def _slice_field_to_tile(
    field: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    out_shape: tuple[int, int],
) -> np.ndarray:
    """
    Crop the global field by (x0,y0,x1,y1) in global coords and resample to out_shape (Y,X).
    Assumes xs, ys form a regular monotonic grid.
    """
    dx = float(xs[1] - xs[0]) if len(xs) > 1 else 1.0
    dy = float(ys[1] - ys[0]) if len(ys) > 1 else 1.0
    i0 = max(0, int(np.floor((x0 - xs[0]) / dx)))
    i1 = min(len(xs) - 1, int(np.ceil((x1 - xs[0]) / dx)))
    j0 = max(0, int(np.floor((y0 - ys[0]) / dy)))
    j1 = min(len(ys) - 1, int(np.ceil((y1 - ys[0]) / dy)))

    patch = field[j0 : j1 + 1, i0 : i1 + 1]
    if patch.size == 0:
        patch = np.ones((2, 2), dtype=np.float32)

    out = resize(patch, out_shape, preserve_range=True, anti_aliasing=True).astype(np.float32)
    out = np.clip(out, 1e-6, np.inf)
    return out


# --------------------------------------------------------------------------------------
# CLI: illum-collect (P1 and P99.99 per tile)
# --------------------------------------------------------------------------------------
