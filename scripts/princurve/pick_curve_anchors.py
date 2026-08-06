# %% [markdown]
# # Pick princurve start/end anchors (click UI)
#
# Click two points on the spatial scatter plot to choose a **start** and **end**
# anchor (saved as `cell_id`s). This is meant for **path** datasets (no loops).
#
# Run cells sequentially. Outputs are written to `OUTDIR`.

# %%
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import anndata as ad
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from IPython import get_ipython
from matplotlib.widgets import Button

from ccf.princurve import fit_anchor_curve, project_to_polyline_arclength
from fishtools.io.workspace import Workspace

matplotlib.rcdefaults()
# Use widget backend for VS Code interactive mode
ip = get_ipython()
if ip is not None:
    ip.run_line_magic("matplotlib", "widget")

folders = sorted(Path("/working").expanduser().glob("2025*Jax*"))
FOLDER = [f for f in folders if f.name.startswith("20251225")][0]
print(FOLDER)
j = 0
# %% [markdown]
# ## Config (EDIT THESE)

print("Using folder:", FOLDER)
ws = Workspace(f"{FOLDER}")

ccf_dir = Path(FOLDER) / "analysis" / "output" / "ccf-transforms"
roi_to_annotated: dict[str, list[Path]] = {}
for p in sorted(ccf_dir.glob("*/*.annotated.h5ad")):
    roi_to_annotated.setdefault(p.parent.name, []).append(p)

missing_subrois_by_roi: dict[str, set[str]] = {}
for roi_name, files in roi_to_annotated.items():
    for h5ad in files:
        subrois: list[str] = []
        adata_scan = ad.read_h5ad(h5ad, backed="r")
        try:
            if "ccf_adjusted" in adata_scan.obs.columns:
                roi_series = adata_scan.obs["ccf_adjusted"].astype(str)
                subrois = [
                    v
                    for v in roi_series.unique().tolist()
                    if v != "" and v.lower() not in {"nan", "none"}
                ]
        finally:
            if getattr(adata_scan, "isbacked", False) and getattr(adata_scan, "file", None) is not None:
                adata_scan.file.close()

        if not subrois:
            continue

        stem = h5ad.stem
        for subroi in subrois:
            if not (h5ad.parent / f"{stem}.{subroi}.anchors.json").exists():
                missing_subrois_by_roi.setdefault(roi_name, set()).add(str(subroi))

print(
    "ROIs with .annotated.h5ad but missing curve anchors (by subROI):",
    ", ".join(sorted(missing_subrois_by_roi)) if missing_subrois_by_roi else "(none)",
)
for roi_name in sorted(missing_subrois_by_roi):
    missing_subrois = ", ".join(sorted(missing_subrois_by_roi[roi_name]))
    print(f"  - {roi_name}: {missing_subrois}")

ROI =ws.rois[j]
j += 1

INPUT_H5AD = Path(
    f"{FOLDER}/analysis/output/ccf-transforms/{ROI}/{ROI}.syn.annotated.h5ad"
)


# Optional extra filter (leave as None/None if INPUT_H5AD is already filtered)
SUBSET_OBS_KEY: str | None = None  # e.g. "ccf_adjusted"
SUBSET_OBS_VALUE: str | None = None  # e.g. "cortex"

OUTDIR = INPUT_H5AD.parent
OUTDIR.mkdir(parents=True, exist_ok=True)

def anchors_json_path(*, subroi: str) -> Path:
    subroi_clean = str(subroi).strip()
    if subroi_clean == "":
        raise ValueError("subroi must be non-empty for anchors JSON path.")
    return OUTDIR / f"{INPUT_H5AD.stem}.{subroi_clean}.anchors.json"

# If the input contains multiple ROI values in `ROI_OBS_KEY`, run the click UI once per value.
ROI_OBS_KEY = "ccf_adjusted"
# If set, restrict ROI iteration to only these values.
# If None, iterate over all values present in `ROI_OBS_KEY` (excluding NaNs).
ROI_VALUES: tuple[str, ...] | None = None

PLOT_MAX_POINTS = 100_000  # downsample for rendering only
POINT_SIZE = 2
POINT_ALPHA = 0.35
SHOW_STATUS_BOX = False

# Background context points (cells not in current `roi_value`) to show context.
PLOT_MAX_CONTEXT_POINTS = 100_000
CONTEXT_COLOR = "lightgray"
CONTEXT_ALPHA = 0.2

# Scatter color by gene expression (from `.X` or `.raw.X`), if available.
COLOR_GENE: str | None = "Celf2"
COLOR_GENE_LOG1P = True
COLOR_GENE_CMAP = "CMRmap"
COLOR_GENE_CLIP_Q = (0.01, 0.99)

# Job index (incremented in the plotting cell below)
i = 0

# Signed-r review settings (used by the review cell near the bottom).
REVIEW_CURVE_N_DENSE = 5_000
REVIEW_ANCHOR_SMOOTHING = 0.5
REVIEW_PLOT_MAX_POINTS = 100_000
REVIEW_R_SIGN_ENDPOINT_EXTRAPOLATION = 0.25
SHOW_LUT_CONTINUITY_WARNINGS = True
LUT_QC_REPORT_PATH = Path(
    "/home/chaichontat/fishtools2/ccf/out/refextract/midsurface_neocortex_mesocortex_allocortex_3d/lut_discontinuity_qc.json"
)
LUT_WARN_JUMP_THR = 0.2
LUT_WARN_N_JUMPS = 2

if not INPUT_H5AD.exists():
    raise FileNotFoundError(INPUT_H5AD)

adata = ad.read_h5ad(INPUT_H5AD)

if (SUBSET_OBS_KEY is None) != (SUBSET_OBS_VALUE is None):
    raise ValueError("Use SUBSET_OBS_KEY and SUBSET_OBS_VALUE together, or leave both None.")
if SUBSET_OBS_KEY is not None:
    if SUBSET_OBS_KEY not in adata.obs.columns:
        raise KeyError(f"obs column not found: {SUBSET_OBS_KEY}")
    mask = adata.obs[SUBSET_OBS_KEY].astype(str) == str(SUBSET_OBS_VALUE)
    if int(mask.sum()) == 0:
        raise ValueError(f"No rows match obs[{SUBSET_OBS_KEY!r}] == {SUBSET_OBS_VALUE!r}")
    adata = adata[mask].copy()
    print(f"Subset: kept {adata.n_obs} cells where obs[{SUBSET_OBS_KEY}] == {SUBSET_OBS_VALUE}")

if "spatial" not in adata.obsm:
    raise KeyError("Missing adata.obsm['spatial']")


# ## Phase 1: Click UI (pick anchors in order)
#
# Controls:
# - Each click adds an anchor (nearest cell).
# - Shift+click inserts an anchor between existing anchors (nearest segment in the current anchor polyline).
# - Ctrl+click removes the nearest existing anchor.
# - **Undo** removes the last anchor.
# - **Reset** clears all anchors.
# - **Save** writes `OUT_JSON` (requires ≥2 anchors).

@dataclass(frozen=True)
class Anchor:
    cell_id: str
    x: float
    y: float
    index: int


def pick_anchors(
    *,
    adata_in: ad.AnnData,
    adata_context: ad.AnnData | None = None,
    out_json: Path,
    title: str,
    roi_value: str | None,
    initial_anchors: list[Anchor] | None = None,
) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)

    xy_full = np.asarray(adata_in.obsm["spatial"])
    if xy_full.ndim != 2 or xy_full.shape[1] < 2:
        raise ValueError(f"Unexpected spatial shape: {xy_full.shape}")
    xy_full = xy_full[:, :2].astype(float, copy=False)

    cell_ids = adata_in.obs_names.astype(str).to_numpy()
    assert xy_full.shape[0] == cell_ids.shape[0]
    cell_id_to_index = {cid: i for i, cid in enumerate(cell_ids)}

    modifiers = {"shift": False, "ctrl": False}
    mode = {"insert": False}

    def _has_modifier(event, name: str) -> bool:
        key = getattr(event, "key", None)
        if not isinstance(key, str):
            return False
        key = key.lower()
        if name == "ctrl":
            return ("ctrl" in key) or ("control" in key)
        return name in key

    def nearest_anchor(x: float, y: float) -> Anchor:
        dx = xy_full[:, 0] - x
        dy = xy_full[:, 1] - y
        i = int(np.argmin(dx * dx + dy * dy))
        return Anchor(cell_id=str(cell_ids[i]), x=float(xy_full[i, 0]), y=float(xy_full[i, 1]), index=i)

    def insertion_index(x: float, y: float) -> int:
        if len(anchors) < 2:
            return len(anchors)
        p = np.array([float(x), float(y)], dtype=float)
        axy = np.array([[a.x, a.y] for a in anchors], dtype=float)
        seg_a = axy[:-1]
        seg_b = axy[1:]
        v = seg_b - seg_a
        vv = np.sum(v * v, axis=1)
        vv = np.where(vv > 0, vv, 1.0)
        w = p[None, :] - seg_a
        tau = np.clip(np.sum(w * v, axis=1) / vv, 0.0, 1.0)
        proj = seg_a + tau[:, None] * v
        d2 = np.sum((p[None, :] - proj) ** 2, axis=1)
        seg_i = int(np.argmin(d2))
        return seg_i + 1

    anchors: list[Anchor] = []
    if initial_anchors:
        for a in initial_anchors:
            idx = cell_id_to_index.get(a.cell_id)
            if idx is None:
                print(f"Skipping existing anchor (cell_id not in this view): {a.cell_id}")
                continue
            anchors.append(
                Anchor(
                    cell_id=str(a.cell_id),
                    x=float(xy_full[idx, 0]),
                    y=float(xy_full[idx, 1]),
                    index=int(idx),
                )
            )

    rng = np.random.default_rng(0)
    plot_idx = np.arange(xy_full.shape[0])
    if plot_idx.size > PLOT_MAX_POINTS:
        plot_idx = rng.choice(plot_idx, size=PLOT_MAX_POINTS, replace=False)
    plot_xy = xy_full[plot_idx]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_title(title)

    def _get_gene_expr(gene: str) -> np.ndarray | None:
        if gene in adata_in.var_names:
            x = adata_in[:, gene].X
        elif adata_in.raw is not None and gene in adata_in.raw.var_names:
            x = adata_in.raw[:, gene].X
        else:
            return None

        if sp.issparse(x):
            x = x.toarray()
        arr = np.asarray(x, dtype=float)
        if arr.ndim == 2:
            if arr.shape[1] != 1:
                raise ValueError(f"Unexpected expression slice shape for {gene!r}: {arr.shape}")
            arr = arr[:, 0]
        return arr

    c = "black"
    cmap = "CMRmap_r"
    vmin = None
    vmax = None
    if COLOR_GENE is not None:
        expr = _get_gene_expr(str(COLOR_GENE))
        if expr is None:
            print(f"Note: gene not found for coloring: {COLOR_GENE!r} (using black)")
        else:
            expr = expr.astype(float, copy=False)
            if COLOR_GENE_LOG1P:
                expr = np.log1p(np.clip(expr, 0.0, np.inf))
            expr_plot = expr[plot_idx]
            finite = expr_plot[np.isfinite(expr_plot)]
            if finite.size:
                qlo, qhi = (float(COLOR_GENE_CLIP_Q[0]), float(COLOR_GENE_CLIP_Q[1]))
                vmin, vmax = np.quantile(finite, [qlo, qhi]).tolist()
                if np.isfinite(vmin) and np.isfinite(vmax) and vmax > vmin:
                    c = expr_plot
                    cmap = str(COLOR_GENE_CMAP)

    sc = ax.scatter(
        plot_xy[:, 0],
        plot_xy[:, 1],
        s=POINT_SIZE,
        alpha=POINT_ALPHA,
        c=c,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        linewidths=0,
        zorder=3,
    )

    xlim_roi = ax.get_xlim()
    ylim_roi = ax.get_ylim()
    xspan = float(xlim_roi[1] - xlim_roi[0])
    yspan = float(ylim_roi[1] - ylim_roi[0])
    if np.isfinite(xspan) and np.isfinite(yspan) and xspan > 0 and yspan > 0:
        cx = 0.5 * float(xlim_roi[0] + xlim_roi[1])
        cy = 0.5 * float(ylim_roi[0] + ylim_roi[1])
        half_x = 0.5 * xspan
        half_y = 0.5 * yspan
        if xspan <= yspan:
            half_x *= 1.5
        if yspan <= xspan:
            half_y *= 1.5
        xlim_roi = (cx - half_x, cx + half_x)
        ylim_roi = (cy - half_y, cy + half_y)
        ax.set_xlim(xlim_roi)
        ax.set_ylim(ylim_roi)

    if (
        roi_value is not None
        and adata_context is not None
        and "spatial" in adata_context.obsm
        and ROI_OBS_KEY in adata_context.obs.columns
        and int(PLOT_MAX_CONTEXT_POINTS) > 0
    ):
        roi_series_all = adata_context.obs[ROI_OBS_KEY].astype(str).to_numpy()
        idx_all = np.flatnonzero(roi_series_all != str(roi_value))
        if idx_all.size:
            if idx_all.size > int(PLOT_MAX_CONTEXT_POINTS):
                idx_all = rng.choice(idx_all, size=int(PLOT_MAX_CONTEXT_POINTS), replace=False)
            xy_all = np.asarray(adata_context.obsm["spatial"], dtype=float)[:, :2]
            ax.scatter(
                xy_all[idx_all, 0],
                xy_all[idx_all, 1],
                s=POINT_SIZE,
                alpha=float(CONTEXT_ALPHA),
                c=str(CONTEXT_COLOR),
                linewidths=0,
                zorder=1,
            )
            ax.set_xlim(xlim_roi)
            ax.set_ylim(ylim_roi)
    if cmap is not None and COLOR_GENE is not None:
        label = f"{COLOR_GENE} expression"
        if COLOR_GENE_LOG1P:
            label = f"log1p({label})"
        fig.colorbar(sc, ax=ax, label=label)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")

    anchor_artist = ax.scatter([], [], s=70, c="lime", edgecolors="black", linewidths=0.5, zorder=6)
    status_text = None
    if SHOW_STATUS_BOX:
        status_text = ax.text(
            0.01,
            0.99,
            "Click to add anchors",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=10,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )
    anchor_labels: list = []

    def refresh() -> None:
        if not anchors:
            anchor_artist.set_offsets(np.empty((0, 2)))
        else:
            anchor_artist.set_offsets(np.array([[a.x, a.y] for a in anchors]))

        for t in anchor_labels:
            try:
                t.remove()
            except Exception:
                pass
        anchor_labels.clear()

        for i, a in enumerate(anchors, start=1):
            txt = ax.text(
                a.x,
                a.y,
                str(i),
                fontsize=9,
                color="black",
                ha="center",
                va="center",
                bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 1},
                zorder=7,
            )
            anchor_labels.append(txt)

        if not anchors:
            status = "Click to add anchors"
        else:
            status = "Anchors:\n" + "\n".join([f"{i+1}. {a.cell_id}" for i, a in enumerate(anchors)])

        if status_text is not None:
            status_text.set_text(status)
        fig.canvas.draw_idle()

    def on_key_press(event) -> None:
        key = getattr(event, "key", None)
        if not isinstance(key, str):
            return
        key = key.lower()
        if "shift" in key:
            modifiers["shift"] = True
        if ("ctrl" in key) or ("control" in key):
            modifiers["ctrl"] = True

    def on_key_release(event) -> None:
        key = getattr(event, "key", None)
        if not isinstance(key, str):
            return
        key = key.lower()
        if "shift" in key:
            modifiers["shift"] = False
        if ("ctrl" in key) or ("control" in key):
            modifiers["ctrl"] = False

    def on_click(event) -> None:
        if event.inaxes is not ax:
            return
        if event.xdata is None or event.ydata is None:
            return

        x = float(event.xdata)
        y = float(event.ydata)

        ctrl = modifiers["ctrl"] or _has_modifier(event, "ctrl")
        shift = modifiers["shift"] or _has_modifier(event, "shift")
        insert = mode["insert"] or shift

        if ctrl:
            if not anchors:
                return
            axy = np.array([[a.x, a.y] for a in anchors], dtype=float)
            p = np.array([x, y], dtype=float)
            d2 = np.sum((axy - p[None, :]) ** 2, axis=1)
            j = int(np.argmin(d2))
            removed = anchors.pop(j)
            print("Removed:", removed)
            refresh()
            return

        a = nearest_anchor(x, y)

        if insert:
            j = insertion_index(x, y)
            ids = [aa.cell_id for aa in anchors]
            if a.cell_id in ids:
                prev = ids.index(a.cell_id)
                removed = anchors.pop(prev)
                if prev < j:
                    j -= 1
                anchors.insert(j, removed)
                print(f"Moved: {removed} -> position {j + 1}")
            else:
                anchors.insert(j, a)
                print(f"Inserted {j + 1}:", a)
            refresh()
            return

        if a.cell_id in [aa.cell_id for aa in anchors]:
            return
        anchors.append(a)
        print(f"ANCHOR {len(anchors)}:", a)
        refresh()

    def on_reset(_event) -> None:
        anchors.clear()
        print("Reset.")
        refresh()

    def on_undo(_event) -> None:
        if not anchors:
            return
        removed = anchors.pop()
        print("Undo:", removed)
        refresh()

    def on_reverse(_event) -> None:
        if not anchors:
            print("No anchors to reverse.")
            return
        anchors.reverse()
        print("Reversed anchors.")
        refresh()

    def on_save(_event) -> None:
        if len(anchors) < 2:
            print("Pick at least 2 anchors before saving.")
            return

        payload = {
            "input_h5ad": str(INPUT_H5AD),
            "subset_obs_key": SUBSET_OBS_KEY,
            "subset_obs_value": SUBSET_OBS_VALUE,
            "roi_obs_key": ROI_OBS_KEY if roi_value is not None else None,
            "roi_value": roi_value,
            "n_obs": int(adata_in.n_obs),
            "reverse_r_sign": False,
            "anchors": [asdict(a) for a in anchors],
            # Backward-compat fields for older scripts:
            "start": asdict(anchors[0]),
            "end": asdict(anchors[-1]),
        }
        out_json.write_text(json.dumps(payload, indent=2) + "\n")
        print("Wrote:", out_json)

    def on_toggle_mode(_event) -> None:
        mode["insert"] = not mode["insert"]
        btn_mode.label.set_text("Mode: insert" if mode["insert"] else "Mode: append")
        refresh()

    fig.canvas.mpl_connect("key_press_event", on_key_press)
    fig.canvas.mpl_connect("key_release_event", on_key_release)
    fig.canvas.mpl_connect("button_press_event", on_click)

    ax_mode = fig.add_axes([0.52, 0.01, 0.09, 0.06])
    ax_undo = fig.add_axes([0.616, 0.01, 0.09, 0.06])
    ax_reverse = fig.add_axes([0.712, 0.01, 0.09, 0.06])
    ax_reset = fig.add_axes([0.808, 0.01, 0.09, 0.06])
    ax_save = fig.add_axes([0.904, 0.01, 0.09, 0.06])
    btn_mode = Button(ax_mode, "Mode: append")
    btn_undo = Button(ax_undo, "Undo")
    btn_reverse = Button(ax_reverse, "Reverse")
    btn_reset = Button(ax_reset, "Reset")
    btn_save = Button(ax_save, "Save")
    btn_mode.on_clicked(on_toggle_mode)
    btn_undo.on_clicked(on_undo)
    btn_reverse.on_clicked(on_reverse)
    btn_reset.on_clicked(on_reset)
    btn_save.on_clicked(on_save)

    # In `%matplotlib widget` (VS Code/Jupyter), `plt.show()` is non-blocking.
    # Keep widget objects alive so the buttons keep working after this function returns.
    fig._pick_curve_anchors_handles = (  # type: ignore[attr-defined]
        btn_mode,
        btn_undo,
        btn_reverse,
        btn_reset,
        btn_save,
    )

    refresh()
    plt.show()


jobs: list[tuple[str | None, ad.AnnData, Path]] = []
if SUBSET_OBS_KEY is not None:
    assert SUBSET_OBS_VALUE is not None
    subroi = str(SUBSET_OBS_VALUE).strip()
    jobs = [(subroi, adata, anchors_json_path(subroi=subroi))]
elif ROI_OBS_KEY in adata.obs.columns:
    roi_series = adata.obs[ROI_OBS_KEY].astype(str)
    roi_unique = [
        v for v in roi_series.unique().tolist() if v != "" and v.lower() not in {"nan", "none"}
    ]
    roi_present = sorted(roi_unique) if ROI_VALUES is None else [roi for roi in ROI_VALUES if roi in set(roi_unique)]
    if roi_present:
        for roi in roi_present:
            out_json = anchors_json_path(subroi=roi)
            jobs.append((roi, adata[roi_series == roi].copy(), out_json))

if not jobs:
    jobs = [("all", adata, anchors_json_path(subroi="all"))]

print("Loaded:")
print("  n_obs:", adata.n_obs)
if ROI_OBS_KEY in adata.obs.columns:
    print(f"  {ROI_OBS_KEY}:", sorted(set(adata.obs[ROI_OBS_KEY].astype(str))))
print("  jobs:", [(roi, int(a.n_obs), str(p)) for roi, a, p in jobs])

def load_existing_anchors(path: Path, *, cell_ids_in_view: set[str]) -> list[Anchor]:
    if not path.exists():
        return []

    data = json.loads(path.read_text())
    raw = data.get("anchors")
    if isinstance(raw, list) and raw:
        out: list[Anchor] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            cid = item.get("cell_id")
            if not isinstance(cid, str) or cid == "":
                continue
            if cid not in cell_ids_in_view:
                continue
            out.append(Anchor(cell_id=cid, x=0.0, y=0.0, index=-1))
        return out

    start = data.get("start")
    end = data.get("end")
    out: list[Anchor] = []
    for item in (start, end):
        if not isinstance(item, dict):
            continue
        cid = item.get("cell_id")
        if not isinstance(cid, str) or cid == "":
            continue
        if cid not in cell_ids_in_view:
            continue
        out.append(Anchor(cell_id=cid, x=0.0, y=0.0, index=-1))
    return out


def _extract_anchor_ids_from_payload(payload: dict[str, object]) -> list[str]:
    raw = payload.get("anchors")
    if isinstance(raw, list) and raw:
        out: list[str] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            cid = item.get("cell_id")
            if isinstance(cid, str) and cid != "":
                out.append(cid)
        if len(out) >= 2:
            return out

    start = payload.get("start")
    end = payload.get("end")
    out2: list[str] = []
    for item in (start, end):
        if not isinstance(item, dict):
            continue
        cid = item.get("cell_id")
        if isinstance(cid, str) and cid != "":
            out2.append(cid)
    return out2


def _extract_lut_row_qc_for_input(payload: dict[str, object]) -> dict[str, object] | None:
    if not SHOW_LUT_CONTINUITY_WARNINGS:
        return None
    report_path = Path(LUT_QC_REPORT_PATH)
    if not report_path.exists():
        return None
    input_h5ad = payload.get("input_h5ad")
    if not isinstance(input_h5ad, str) or input_h5ad.strip() == "":
        return None
    p1_path = Path(input_h5ad).with_name("p1_landmarks.json")
    if not p1_path.exists():
        return None
    try:
        p1 = json.loads(p1_path.read_text())
    except json.JSONDecodeError:
        return None
    axis = str(p1.get("atlas_plane", "")).strip().lower()
    if axis not in {"coronal", "sagittal"}:
        return None
    try:
        atlas_slice_idx = int(p1["atlas_slice_idx"])
    except (KeyError, TypeError, ValueError):
        return None

    try:
        qc = json.loads(report_path.read_text())
    except json.JSONDecodeError:
        return None
    maps = qc.get("maps")
    if not isinstance(maps, dict):
        return None
    map_key = "coronal_to_sagittal" if axis == "coronal" else "sagittal_to_coronal"
    rows_obj = maps.get(map_key, {})
    if not isinstance(rows_obj, dict):
        return None
    rows = rows_obj.get("rows")
    if not isinstance(rows, list):
        return None
    for row in rows:
        if not isinstance(row, dict):
            continue
        if int(row.get("source_slice", -1)) == atlas_slice_idx:
            return row
    return None


def review_signed_r_plot(
    *,
    adata_in: ad.AnnData,
    out_json: Path,
    n_dense: int,
    smoothing: float,
    max_points: int,
) -> None:
    if not out_json.exists():
        print(f"Missing anchors JSON: {out_json}")
        return

    payload = json.loads(out_json.read_text())
    reverse_r_sign = payload.get("reverse_r_sign")
    if reverse_r_sign is None:
        reverse_r_sign = False
        payload["reverse_r_sign"] = False
        out_json.write_text(json.dumps(payload, indent=2) + "\n")
    elif not isinstance(reverse_r_sign, bool):
        raise ValueError(f"Invalid reverse_r_sign in anchors JSON (expected bool): {out_json}")
    anchor_ids = _extract_anchor_ids_from_payload(payload)
    if len(anchor_ids) < 2:
        raise ValueError(f"Need at least 2 anchors in JSON to review signed r: {out_json}")

    xy = np.asarray(adata_in.obsm["spatial"], dtype=float)
    if xy.ndim != 2 or xy.shape[1] < 2:
        raise ValueError(f"Unexpected spatial shape: {xy.shape}")
    xy = xy[:, :2]
    cell_ids = adata_in.obs_names.astype(str).to_numpy()
    cell_to_index = {cid: i for i, cid in enumerate(cell_ids)}
    missing = [cid for cid in anchor_ids if cid not in cell_to_index]
    if missing:
        raise ValueError(f"Anchors are missing in current view: {missing[:5]}{'...' if len(missing) > 5 else ''}")

    anchor_idx = np.asarray([cell_to_index[cid] for cid in anchor_ids], dtype=int)
    anchor_xy = xy[anchor_idx, :]
    curve = fit_anchor_curve(anchor_xy=anchor_xy, n_dense=int(n_dense), smoothing=float(smoothing))
    _t, r_signed, _proj = project_to_polyline_arclength(
        xy=xy,
        line=curve,
        k=50,
        endpoint_extrapolation=float(REVIEW_R_SIGN_ENDPOINT_EXTRAPOLATION),
    )
    if reverse_r_sign:
        r_signed = -np.asarray(r_signed, dtype=float)

    rng = np.random.default_rng(0)
    plot_idx = np.arange(xy.shape[0])
    if plot_idx.size > int(max_points):
        plot_idx = rng.choice(plot_idx, size=int(max_points), replace=False)

    finite_abs = np.abs(r_signed[np.isfinite(r_signed)])
    lim = float(np.quantile(finite_abs, 0.99)) if finite_abs.size else 1.0
    if not np.isfinite(lim) or lim <= 0:
        lim = 1.0

    fig, ax = plt.subplots(figsize=(10, 8))
    sc = ax.scatter(
        xy[plot_idx, 0],
        xy[plot_idx, 1],
        c=r_signed[plot_idx],
        s=POINT_SIZE,
        alpha=POINT_ALPHA,
        cmap="coolwarm",
        vmin=-lim,
        vmax=lim,
        linewidths=0,
        zorder=2,
    )
    ax.plot(curve[:, 0], curve[:, 1], color="black", linewidth=1.5, zorder=3)
    ax.scatter(anchor_xy[:, 0], anchor_xy[:, 1], c="yellow", s=30, edgecolors="black", linewidths=0.5, zorder=4)
    ax.scatter([anchor_xy[0, 0]], [anchor_xy[0, 1]], c="lime", s=70, edgecolors="black", linewidths=0.5, zorder=5)
    ax.scatter([anchor_xy[-1, 0]], [anchor_xy[-1, 1]], c="red", s=70, edgecolors="black", linewidths=0.5, zorder=5)
    ax.set_title("Signed r review (coolwarm): green=start, red=end")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")

    lut_row = _extract_lut_row_qc_for_input(payload)
    if isinstance(lut_row, dict):
        after = lut_row.get("after")
        if isinstance(after, dict):
            max_abs_dt = float(after.get("max_abs_dt", float("nan")))
            n_jump_ge_0p2 = int(after.get("n_jump_ge_0p2", 0))
            status = "WARN" if (
                np.isfinite(max_abs_dt)
                and max_abs_dt >= float(LUT_WARN_JUMP_THR)
                and n_jump_ge_0p2 >= int(LUT_WARN_N_JUMPS)
            ) else "OK"
            txt = (
                f"LUT continuity: {status}\n"
                f"max|dt|={max_abs_dt:.3f}\n"
                f"n(|dt|>=0.2)={n_jump_ge_0p2}"
            )
            ax.text(
                0.01,
                0.99,
                txt,
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=9,
                bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "black", "linewidth": 0.5},
            )

    fig.colorbar(sc, ax=ax, label="signed r")
    print(f"Rendered signed-r review plot (reverse_r_sign={reverse_r_sign}).")
    plt.show()


def apply_signed_r_review_decision(*, out_json: Path, answer: str | None = None) -> None:
    if not out_json.exists():
        print(f"Missing anchors JSON: {out_json}")
        return
    payload = json.loads(out_json.read_text())

    if answer is None:
        cur = payload.get("reverse_r_sign", False)
        if not isinstance(cur, bool):
            raise ValueError(f"Invalid reverse_r_sign in anchors JSON (expected bool): {out_json}")
        answer = input(f"reverse_r_sign is {cur}. Set to True/False? [y/n/Enter=keep]: ")
    ans = str(answer).strip().lower()
    if ans == "":
        reverse = payload.get("reverse_r_sign", False)
    elif ans in {"y", "yes"}:
        reverse = True
    elif ans in {"n", "no"}:
        reverse = False
    else:
        print(f"Unrecognized answer {answer!r}; keeping existing reverse_r_sign.")
        reverse = payload.get("reverse_r_sign", False)
    if not isinstance(reverse, bool):
        reverse = False
    payload["reverse_r_sign"] = reverse
    print(f"Recorded reverse_r_sign={reverse} (anchor order unchanged).")

    out_json.write_text(json.dumps(payload, indent=2) + "\n")
    print("Updated:", out_json)

# %% [markdown]
# ## Run next job (re-run this cell)
#
# VS Code's matplotlib integration can make it awkward to close figures. Instead of looping
# over ROIs in one go, run one job per cell execution.

if i >= len(jobs):
    print(f"Done: i={i} >= n_jobs={len(jobs)}")
else:
    roi_value, adata_roi, out_json = jobs[i]
    title = "Click anchors in order (start → ... → end)" if roi_value is None else f"Click anchors: {roi_value}"
    print(f"Job {i + 1}/{len(jobs)}: roi={roi_value} n_obs={int(adata_roi.n_obs)} out={out_json}")
    initial = load_existing_anchors(out_json, cell_ids_in_view=set(adata_roi.obs_names.astype(str)))
    if initial:
        print(f"Loaded existing anchors: n={len(initial)} from {out_json}")
    pick_anchors(

        adata_in=adata_roi,
        adata_context=adata,
        out_json=out_json,
        title=title,
        roi_value=roi_value,
        initial_anchors=initial,
    )
    i += 1



# %%
REVIEW_JOB_INDEX = max(0, i - 1)

if not jobs:
    print("No jobs available for review.")
elif REVIEW_JOB_INDEX >= len(jobs):
    print(f"Review job index out of range: {REVIEW_JOB_INDEX} (n_jobs={len(jobs)})")
else:
    roi_value, adata_roi, out_json = jobs[REVIEW_JOB_INDEX]
    print(f"Reviewing: roi={roi_value} n_obs={int(adata_roi.n_obs)} json={out_json}")
    review_signed_r_plot(
        adata_in=adata_roi,
        out_json=out_json,
        n_dense=REVIEW_CURVE_N_DENSE,
        smoothing=REVIEW_ANCHOR_SMOOTHING,
        max_points=REVIEW_PLOT_MAX_POINTS,
    )
    print("Next: run the decision cell to save reverse_r_sign=true/false.")
#%%

if not jobs:
    print("No jobs available for decision.")
elif REVIEW_JOB_INDEX >= len(jobs):
    print(f"Review job index out of range: {REVIEW_JOB_INDEX} (n_jobs={len(jobs)})")
else:
    roi_value, adata_roi, out_json = jobs[REVIEW_JOB_INDEX]
    print(f"Saving decision: roi={roi_value} n_obs={int(adata_roi.n_obs)} json={out_json}")
    apply_signed_r_review_decision(out_json=out_json)

# %%
