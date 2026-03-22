from __future__ import annotations

import argparse
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import anndata as ad
import matplotlib.pyplot as plt
import pandas as pd
import zarr
from loguru import logger

from fishtools.io.workspace import Workspace as IOWorkspace
from fishtools.brdu.barrage import load_barrage_adata
from fishtools.segment.cell_thumbnail import cell_thumbnail_from_fused


ALL_DATASETS = "all"
DEFAULT_H5AD_PATH = Path("~/nvme/vzsvz.h5ad")
WORKSPACE_ROOT_BASES = (Path("/working"), Path.home() / "nvme")
FUSED_ZARR_NAME = "fused.zarr"
THUMB_DISPLAY_VMIN = 0.0
THUMB_DISPLAY_VMAX = 32_768.0


@dataclass(frozen=True)
class BarrageConfig:
    dataset: str
    h5ad_path: Path
    out_dir: Path
    overwrite: bool
    n_cells: int
    seed: int
    n_bins: int
    codebook: str
    seg_codebook: str
    segmentation_name: str
    size: int
    n_channels: int
    cmap: str
    outline_color: str
    render_all_outlines: bool


def _sanitize_filename_fragment(value: str) -> str:
    value = value.replace("|", "__")
    value = re.sub(r"[^a-zA-Z0-9._-]+", "_", value)
    return value.strip("._-") or "cell"


def _atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", newline="", delete=False, dir=path.parent, suffix=path.suffix) as tmp:
        df.to_csv(tmp.name, index=False)
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)


def _select_cells_stratified(
    *,
    obs: pd.DataFrame,
    feature_cols: list[str],
    n_cells: int,
    n_bins: int,
    seed: int,
) -> list[str]:
    if n_cells <= 0:
        raise ValueError("n_cells must be positive")
    if n_bins <= 1:
        raise ValueError("n_bins must be >= 2")
    missing = sorted(set(feature_cols) - set(obs.columns))
    if missing:
        raise ValueError(f"Missing required obs columns: {missing}")

    feats = obs[feature_cols].astype(float).to_numpy(dtype=np.float64)
    feats_log = np.log10(feats + 1.0)

    finite = np.isfinite(feats_log).all(axis=1)
    if not finite.any():
        raise ValueError("No finite rows in features; cannot select cells.")

    feats_log_f = feats_log[finite]
    lo = np.min(feats_log_f, axis=0)
    hi = np.max(feats_log_f, axis=0)
    span = hi - lo
    span[span <= 0] = 1.0

    scaled = (feats_log - lo) / span
    scaled = np.clip(scaled, 0.0, 1.0)
    bins = np.floor(scaled * float(n_bins)).astype(np.int16)
    bins[bins == n_bins] = n_bins - 1
    bins[~finite] = -1

    rng = np.random.default_rng(seed)
    groups: dict[tuple[int, ...], list[str]] = {}
    for cell, b in zip(obs.index.astype(str).tolist(), bins.tolist(), strict=True):
        key = tuple(int(x) for x in b)
        groups.setdefault(key, []).append(cell)
    for cells in groups.values():
        rng.shuffle(cells)

    keys = list(groups.keys())
    rng.shuffle(keys)

    selected: list[str] = []
    while len(selected) < n_cells:
        progressed = False
        for k in keys:
            cells = groups[k]
            if not cells:
                continue
            selected.append(str(cells.pop()))
            progressed = True
            if len(selected) >= n_cells:
                return selected
        if not progressed:
            break

    return selected


def _select_additional_barrage_cells(
    *,
    obs: pd.DataFrame,
    feature_cols: list[str],
    existing_cells: list[str],
    preselected: list[str],
    n_cells: int,
    n_bins: int,
    seed: int,
) -> tuple[list[str], list[str]]:
    if n_cells <= 0:
        return existing_cells, []

    preserved = [str(cell) for cell in existing_cells]
    preserved_set = set(preserved)
    prioritized = [str(cell) for cell in preselected if str(cell) not in preserved_set]
    if len(prioritized) >= n_cells:
        return preserved, prioritized[:n_cells]

    blocked = preserved_set | set(prioritized)
    obs_remaining = obs.drop(index=list(blocked), errors="ignore")
    additional = _select_cells_stratified(
        obs=obs_remaining,
        feature_cols=feature_cols,
        n_cells=n_cells - len(prioritized),
        n_bins=n_bins,
        seed=seed,
    )
    return preserved, prioritized + additional


def _select_barrage_adata(*, adata: ad.AnnData, dataset: str) -> ad.AnnData:
    if dataset == ALL_DATASETS:
        if "dataset" not in adata.obs.columns:
            raise ValueError("Sampling across all datasets requires adata.obs['dataset'].")
        return adata

    selected = adata[adata.obs["dataset"].astype(str) == dataset]
    if selected.n_obs == 0:
        raise ValueError(f"No cells found for dataset {dataset!r}.")
    return selected


def _workspace_for_dataset(
    *,
    dataset: str,
    cache: dict[str, IOWorkspace],
) -> IOWorkspace:
    ws = cache.get(dataset)
    if ws is not None:
        return ws

    workspace_root = None
    for base in WORKSPACE_ROOT_BASES:
        candidate = base / dataset
        if candidate.exists():
            workspace_root = candidate
            break
    if workspace_root is None:
        checked = ", ".join(str(base / dataset) for base in WORKSPACE_ROOT_BASES)
        raise FileNotFoundError(f"workspace_root not found for dataset {dataset!r}; checked: {checked}")
    ws = IOWorkspace(workspace_root)
    cache[dataset] = ws
    return ws


def _cell_id_roi_and_label(cell_id: str) -> tuple[str, int]:
    if "|" not in cell_id:
        raise ValueError(f"Expected cell id '<roi>|<label>' or '<dataset>:<roi>|<label>', got {cell_id!r}")
    roi_token, label_str = cell_id.split("|", 1)
    roi = roi_token.rsplit(":", 1)[-1]
    return roi, int(label_str)


def plot_cell_thumbnail_with_outline(
    *,
    adata: ad.AnnData,
    dataset: str,
    ws: IOWorkspace,
    cell: str,
    codebook: str,
    seg_codebook: str,
    segmentation_name: str,
    fused_cache: dict[tuple[str, str, str], zarr.Array],
    seg_cache: dict[tuple[str, str, str, str], zarr.Array],
    size: int = 100,
    n_channels: int = 3,
    cmap: str = "magma",
    outline_color: str = "cyan",
    render_all_outlines: bool = False,
) -> tuple[plt.Figure, list[plt.Axes], np.ndarray]:
    row = adata.obs.loc[cell]
    roi = str(row["roi"])
    x_center = int(np.rint(float(row["x"])))
    y_center = int(np.rint(float(row["y"])))
    z_index = int(np.rint(float(row["z"])))

    fused_key = (dataset, roi, codebook)
    fused = fused_cache.get(fused_key)
    if fused is None:
        fused = zarr.open_array(ws.stitch(roi, codebook) / FUSED_ZARR_NAME, mode="r")
        fused_cache[fused_key] = fused

    thumb = cell_thumbnail_from_fused(
        fused,
        z_index=z_index,
        x_center=x_center,
        y_center=y_center,
        size=size,
    )

    n_channels_eff = min(n_channels, int(thumb.shape[-1]))
    if n_channels_eff < 1:
        raise ValueError(f"Expected thumb with channels, got shape={thumb.shape}")

    channel_keys_raw = fused.attrs.get("key")
    channel_keys = channel_keys_raw if isinstance(channel_keys_raw, list) else None

    fig, axs = plt.subplots(nrows=1, ncols=n_channels_eff, figsize=(4 * n_channels_eff, 4), dpi=200)
    axes = [axs] if n_channels_eff == 1 else list(axs)

    for i, ax in enumerate(axes):
        ch_label: str | int = i
        if channel_keys is not None and i < len(channel_keys) and isinstance(channel_keys[i], str):
            ch_label = channel_keys[i]
        ax.set_title(f"cell={cell}\nch={ch_label}")
        ax.imshow(thumb[..., i], cmap=cmap, vmin=THUMB_DISPLAY_VMIN, vmax=THUMB_DISPLAY_VMAX)
        ax.axis("off")

    cell_id = str(cell)
    roi_from_id, label = _cell_id_roi_and_label(cell_id)
    if roi_from_id != roi:
        raise ValueError(f"Cell id roi {roi_from_id!r} != obs roi {roi!r} (cell={cell_id!r})")

    seg_key = (dataset, roi, seg_codebook, segmentation_name)
    seg = seg_cache.get(seg_key)
    if seg is None:
        seg = zarr.open_array(ws.stitch(roi, seg_codebook) / segmentation_name, mode="r")
        seg_cache[seg_key] = seg

    half = size // 2
    y0 = y_center - half
    x0 = x_center - half

    ys0 = max(0, y0)
    ys1 = min(int(seg.shape[1]), y0 + size)
    xs0 = max(0, x0)
    xs1 = min(int(seg.shape[2]), x0 + size)

    crop: np.ndarray
    if ys0 < ys1 and xs0 < xs1:
        if seg.ndim == 3:
            crop = np.asarray(seg[z_index, ys0:ys1, xs0:xs1])
        elif seg.ndim == 4:
            crop = np.asarray(seg[z_index, ys0:ys1, xs0:xs1, 0])
        else:
            raise ValueError(f"Unexpected segmentation array ndim={seg.ndim} shape={seg.shape}")
    else:
        crop = np.zeros((0, 0), dtype=np.int32)

    labels_full = np.zeros((size, size), dtype=crop.dtype if crop.size else np.int32)
    if crop.size:
        labels_full[(ys0 - y0) : (ys1 - y0), (xs0 - x0) : (xs1 - x0)] = crop
    mask_full = labels_full == label

    if mask_full.any():
        for i, ax in enumerate(axes):
            mean_intensity = float(np.mean(thumb[..., i][mask_full]))
            ch_label: str | int = i
            if channel_keys is not None and i < len(channel_keys) and isinstance(channel_keys[i], str):
                ch_label = channel_keys[i]
            ax.set_title(
                f"cell={cell}\n"
                f"ch={ch_label} mean={mean_intensity:.1f}"
            )

    if render_all_outlines:
        for other_label in np.unique(labels_full):
            if int(other_label) == 0 or int(other_label) == label:
                continue
            other_mask = labels_full == other_label
            if not other_mask.any():
                continue
            for ax in axes:
                ax.contour(
                    other_mask.astype(np.float32),
                    levels=[0.5],
                    colors="white",
                    linewidths=0.6,
                    linestyles="--",
                    alpha=0.35,
                )

    if mask_full.any():
        for ax in axes:
            ax.contour(
                mask_full.astype(np.float32),
                levels=[0.5],
                colors=outline_color,
                linewidths=0.8,
                linestyles="--",
                alpha=0.7,
            )

    return fig, axes, thumb


def _parse_args() -> BarrageConfig:
    p = argparse.ArgumentParser(description="Generate a barrage of BrdU/EdU cell thumbnails for labeling.")
    p.add_argument("--dataset", default=ALL_DATASETS, help="Dataset to sample, or 'all' for all datasets in the h5ad.")
    p.add_argument(
        "--h5ad",
        type=Path,
        default=DEFAULT_H5AD_PATH,
        help=(
            "Input h5ad. Accepts either a single .h5ad file (concatenated) or a directory "
            "containing per-ROI .h5ad files."
        ),
    )
    p.add_argument("--out-dir", type=Path, default=Path("output/brdu_edu_barrage"))
    p.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Delete existing images/index before rendering (keeps labels.csv).",
    )
    p.add_argument("--n-cells", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-bins", type=int, default=6)
    p.add_argument("--codebook", default="edu")
    p.add_argument("--seg-codebook", default="pi")
    p.add_argument("--segmentation-name", default="output_segmentation-sam_postproc_s1-2-2_v500.zarr")
    p.add_argument("--size", type=int, default=100)
    p.add_argument("--n-channels", type=int, default=3)
    p.add_argument("--cmap", default="magma")
    p.add_argument("--outline-color", default="cyan")
    p.add_argument(
        "--render-all-outlines",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Render outlines for nearby labels (in addition to the target cell).",
    )
    args = p.parse_args()

    return BarrageConfig(
        dataset=str(args.dataset),
        h5ad_path=Path(args.h5ad).expanduser(),
        out_dir=Path(args.out_dir),
        overwrite=bool(args.overwrite),
        n_cells=int(args.n_cells),
        seed=int(args.seed),
        n_bins=int(args.n_bins),
        codebook=str(args.codebook),
        seg_codebook=str(args.seg_codebook),
        segmentation_name=str(args.segmentation_name),
        size=int(args.size),
        n_channels=int(args.n_channels),
        cmap=str(args.cmap),
        outline_color=str(args.outline_color),
        render_all_outlines=bool(args.render_all_outlines),
    )


def main() -> None:
    cfg = _parse_args()
    h5ad_path = cfg.h5ad_path
    if not h5ad_path.exists():
        raise FileNotFoundError(f"h5ad path not found: {h5ad_path}")

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = cfg.out_dir / "images"
    index_path = cfg.out_dir / "index.csv"
    lock_path = cfg.out_dir / ".barrage.lock"

    try:
        with lock_path.open("x") as f:
            f.write("brdu_edu_generate_barrage\n")
    except FileExistsError as e:
        raise RuntimeError(
            f"Barrage lock exists (another run in progress?): {lock_path}\n"
            "If this is stale, delete it and rerun."
        ) from e

    try:
        if cfg.overwrite:
            if images_dir.exists():
                shutil.rmtree(images_dir)
            if index_path.exists():
                index_path.unlink()
            logger.info(f"Overwrite enabled: cleared {images_dir} and {index_path} (keeping labels.csv).")

        images_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Loading adata from: {h5ad_path}")
        adata = load_barrage_adata(h5ad_path=h5ad_path, dataset=cfg.dataset)
        adata = _select_barrage_adata(adata=adata, dataset=cfg.dataset)

        feature_cols = ["brdu_mean", "brdu_std", "edu_mean", "edu_std"]
        preselected: list[str] = []
        labels_path = cfg.out_dir / "labels.csv"
        if labels_path.exists():
            try:
                labels_df = pd.read_csv(labels_path)
            except Exception as e:
                raise RuntimeError(f"Failed to read existing labels csv: {labels_path}") from e
            if "cell" in labels_df.columns:
                preselected = labels_df["cell"].astype(str).drop_duplicates().tolist()
                obs_name_set = set(adata.obs_names.astype(str).tolist())
                preselected = [c for c in preselected if c in obs_name_set]
                if preselected:
                    logger.info(f"Including {len(preselected)} already-labeled cells from {labels_path}")

        existing_rows = pd.read_csv(index_path) if (not cfg.overwrite and index_path.exists()) else pd.DataFrame()
        existing_cells = existing_rows["cell"].astype(str).tolist() if not existing_rows.empty else []

        if not cfg.overwrite:
            preserved_cells, new_cells = _select_additional_barrage_cells(
                obs=adata.obs,
                feature_cols=feature_cols,
                existing_cells=existing_cells,
                preselected=preselected,
                n_cells=cfg.n_cells,
                n_bins=cfg.n_bins,
                seed=cfg.seed,
            )
            selected_cells = preserved_cells + new_cells
            logger.info(
                f"Preserving {len(preserved_cells)} existing barrage cells and adding {len(new_cells)} new cells."
            )
        elif len(preselected) >= cfg.n_cells:
            selected_cells = preselected[: cfg.n_cells]
            new_cells = selected_cells
        else:
            obs_remaining = adata.obs.drop(index=preselected, errors="ignore") if preselected else adata.obs
            new_cells = preselected + _select_cells_stratified(
                obs=obs_remaining,
                feature_cols=feature_cols,
                n_cells=cfg.n_cells - len(preselected),
                n_bins=cfg.n_bins,
                seed=cfg.seed,
            )
            selected_cells = new_cells
        logger.info(f"Selected {len(selected_cells)} cells for barrage (requested n_cells={cfg.n_cells}).")

        workspace_cache: dict[str, IOWorkspace] = {}
        fused_cache: dict[tuple[str, str, str], zarr.Array] = {}
        seg_cache: dict[tuple[str, str, str, str], zarr.Array] = {}

        rows: list[dict[str, object]] = (
            existing_rows.to_dict(orient="records") if not existing_rows.empty else []
        )
        rank_offset = len(rows)
        for new_idx, cell in enumerate(new_cells):
            rank = rank_offset + new_idx
            img_name = f"{rank:04d}_{_sanitize_filename_fragment(cell)}.png"
            img_rel = Path("images") / img_name
            img_path = cfg.out_dir / img_rel

            obs_row = adata.obs.loc[str(cell)]
            dataset = str(obs_row["dataset"])
            ws = _workspace_for_dataset(dataset=dataset, cache=workspace_cache)
            fig, _axes, thumb = plot_cell_thumbnail_with_outline(
                adata=adata,
                dataset=dataset,
                ws=ws,
                cell=str(cell),
                codebook=cfg.codebook,
                seg_codebook=cfg.seg_codebook,
                segmentation_name=cfg.segmentation_name,
                fused_cache=fused_cache,
                seg_cache=seg_cache,
                size=cfg.size,
                n_channels=cfg.n_channels,
                cmap=cfg.cmap,
                outline_color=cfg.outline_color,
                render_all_outlines=cfg.render_all_outlines,
            )
            fig.tight_layout()
            fig.savefig(img_path, dpi=200)
            plt.close(fig)

            rows.append(
                {
                    "rank": rank,
                    "cell": str(cell),
                    "dataset": dataset,
                    "roi": str(obs_row["roi"]),
                    "x": float(obs_row["x"]),
                    "y": float(obs_row["y"]),
                    "z": float(obs_row["z"]),
                    "brdu_mean": float(obs_row["brdu_mean"]),
                    "brdu_min": float(obs_row["brdu_min"]),
                    "brdu_max": float(obs_row["brdu_max"]),
                    "brdu_median": float(obs_row["brdu_median"]),
                    "brdu_std": float(obs_row["brdu_std"]),
                    "edu_mean": float(obs_row["edu_mean"]),
                    "edu_min": float(obs_row["edu_min"]),
                    "edu_max": float(obs_row["edu_max"]),
                    "edu_median": float(obs_row["edu_median"]),
                    "edu_std": float(obs_row["edu_std"]),
                    "thumb_shape": str(tuple(int(x) for x in thumb.shape)),
                    "image_path": str(img_rel),
                }
            )

            if (rank + 1) % 25 == 0:
                logger.info(f"Rendered {rank + 1}/{len(selected_cells)} images...")

        _atomic_write_csv(pd.DataFrame(rows), index_path)
        logger.info(f"Wrote index: {index_path}")
    finally:
        if lock_path.exists():
            lock_path.unlink()


if __name__ == "__main__":
    main()
