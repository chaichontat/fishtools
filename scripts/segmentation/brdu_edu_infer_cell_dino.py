from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import zarr
from loguru import logger

from fishtools.brdu.barrage import load_obs_from_h5ad
from fishtools.io.workspace import Workspace


WORKSPACE_ROOT_BASES = (Path("/working"), Path.home() / "nvme")
FUSED_ZARR_NAME = "fused.zarr"
UINT16_TO_UINT8_SCALE = 256.0
REQUIRED_BUNDLE_KEYS = {
    "feature_extractor",
    "weights",
    "dinov2_repo",
    "feature_scaler",
    "brdu_model",
    "edu_model",
    "codebook",
    "seg_codebook",
    "segmentation_name",
    "crop_size",
    "resize_size",
    "crop_eval_size",
}


@dataclass(frozen=True)
class InferConfig:
    barrage_dir: Path
    h5ad_path: Path
    model_path: Path
    output_path: Path
    dinov2_repo: Path | None
    weights: Path | None
    datasets: tuple[str, ...]
    rois: tuple[str, ...]
    batch_size: int


def _parse_args() -> InferConfig:
    parser = argparse.ArgumentParser(
        description="Run Cell-DINO BrdU/EdU inference over h5ad cells grouped by dataset x roi."
    )
    parser.add_argument("--barrage-dir", type=Path, default=Path("output/brdu_edu_barrage"))
    parser.add_argument(
        "--h5ad",
        type=Path,
        required=True,
        help="Input h5ad path, or directory of per-ROI h5ads.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Model bundle path. Default: <barrage-dir>/brdu_edu_cell_dino_cp_vits8_logreg.joblib",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Output parquet path. Default: <barrage-dir>/cell_dino_predictions.parquet",
    )
    parser.add_argument("--dinov2-repo", type=Path, default=None, help="Override dinov2 repo path from the bundle.")
    parser.add_argument("--weights", type=Path, default=None, help="Override Cell-DINO checkpoint path from the bundle.")
    parser.add_argument(
        "--dataset",
        dest="datasets",
        action="append",
        default=[],
        help="Optional dataset filter. Repeat to keep multiple datasets.",
    )
    parser.add_argument(
        "--roi",
        dest="rois",
        action="append",
        default=[],
        help="Optional roi filter. Repeat to keep multiple rois.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    barrage_dir = Path(args.barrage_dir)
    model_path = (
        Path(args.model_path)
        if args.model_path is not None
        else (barrage_dir / "brdu_edu_cell_dino_cp_vits8_logreg.joblib")
    )
    output_path = (
        Path(args.output_path) if args.output_path is not None else (barrage_dir / "cell_dino_predictions.parquet")
    )
    return InferConfig(
        barrage_dir=barrage_dir,
        h5ad_path=Path(args.h5ad).expanduser(),
        model_path=model_path,
        output_path=output_path,
        dinov2_repo=Path(args.dinov2_repo).expanduser() if args.dinov2_repo is not None else None,
        weights=Path(args.weights).expanduser() if args.weights is not None else None,
        datasets=tuple(str(x) for x in args.datasets),
        rois=tuple(str(x) for x in args.rois),
        batch_size=int(args.batch_size),
    )


def _resolve_workspace_root(dataset: str) -> Path:
    for base in WORKSPACE_ROOT_BASES:
        candidate = base / dataset
        if candidate.exists():
            return candidate
    checked = ", ".join(str(base / dataset) for base in WORKSPACE_ROOT_BASES)
    raise FileNotFoundError(f"workspace_root not found for dataset {dataset!r}; checked: {checked}")


def _crop_centered_yxc(
    arr_yxc: np.ndarray,
    *,
    x_center: int,
    y_center: int,
    size: int,
) -> np.ndarray:
    if arr_yxc.ndim != 3:
        raise ValueError(f"Expected YXC array, got shape={arr_yxc.shape}")

    y_dim, x_dim, c_dim = arr_yxc.shape
    half = size // 2
    y0 = y_center - half
    y1 = y0 + size
    x0 = x_center - half
    x1 = x0 + size

    out = np.zeros((size, size, c_dim), dtype=arr_yxc.dtype)
    src_y0 = max(0, y0)
    src_y1 = min(y_dim, y1)
    src_x0 = max(0, x0)
    src_x1 = min(x_dim, x1)
    if src_y0 >= src_y1 or src_x0 >= src_x1:
        return out

    dst_y0 = src_y0 - y0
    dst_y1 = dst_y0 + (src_y1 - src_y0)
    dst_x0 = src_x0 - x0
    dst_x1 = dst_x0 + (src_x1 - src_x0)
    out[dst_y0:dst_y1, dst_x0:dst_x1, :] = arr_yxc[src_y0:src_y1, src_x0:src_x1, :]
    return out


def _crop_centered_yx(
    arr_yx: np.ndarray,
    *,
    x_center: int,
    y_center: int,
    size: int,
) -> np.ndarray:
    if arr_yx.ndim != 2:
        raise ValueError(f"Expected YX array, got shape={arr_yx.shape}")

    y_dim, x_dim = arr_yx.shape
    half = size // 2
    y0 = y_center - half
    y1 = y0 + size
    x0 = x_center - half
    x1 = x0 + size

    out = np.zeros((size, size), dtype=arr_yx.dtype)
    src_y0 = max(0, y0)
    src_y1 = min(y_dim, y1)
    src_x0 = max(0, x0)
    src_x1 = min(x_dim, x1)
    if src_y0 >= src_y1 or src_x0 >= src_x1:
        return out

    dst_y0 = src_y0 - y0
    dst_y1 = dst_y0 + (src_y1 - src_y0)
    dst_x0 = src_x0 - x0
    dst_x1 = dst_x0 + (src_x1 - src_x0)
    out[dst_y0:dst_y1, dst_x0:dst_x1] = arr_yx[src_y0:src_y1, src_x0:src_x1]
    return out


def _resolve_segmentation_mask_from_plane(
    seg_plane: np.ndarray,
    *,
    x_center: int,
    y_center: int,
    size: int,
    label: int,
) -> np.ndarray:
    return _crop_centered_yx(seg_plane, x_center=x_center, y_center=y_center, size=size) == label


def _slab_bounds_for_group(
    group: pd.DataFrame,
    *,
    crop_size: int,
    y_dim: int,
    x_dim: int,
) -> tuple[int, int, int, int]:
    half = crop_size // 2
    x_centers = np.rint(group["x"].to_numpy(dtype=np.float64)).astype(np.int64)
    y_centers = np.rint(group["y"].to_numpy(dtype=np.float64)).astype(np.int64)

    y0 = max(0, int(y_centers.min()) - half)
    x0 = max(0, int(x_centers.min()) - half)
    y1 = min(y_dim, int(y_centers.max()) - half + crop_size)
    x1 = min(x_dim, int(x_centers.max()) - half + crop_size)
    return y0, y1, x0, x1


def _cell_id_roi_and_label(cell: str) -> tuple[str, int]:
    if "|" not in cell:
        raise ValueError(f"Expected cell id '<roi>|<label>' or '<dataset>:<roi>|<label>', got {cell!r}")
    roi_token, label_str = cell.split("|", 1)
    roi = roi_token.rsplit(":", 1)[-1]
    return roi, int(label_str)


def _compose_cell_dino_channels(masked_thumb: np.ndarray, *, mask: np.ndarray) -> np.ndarray:
    """Mirror training preprocessing: masked brdu/edu, zero-padded to 5 channels."""

    if masked_thumb.ndim != 3:
        raise ValueError(f"Expected masked_thumb with shape (H, W, C), got {masked_thumb.shape}")
    if masked_thumb.shape[-1] != 2:
        raise ValueError(f"Expected exactly 2 channels (brdu, edu), got {masked_thumb.shape[-1]}")

    output = np.zeros((*masked_thumb.shape[:2], 5), dtype=np.float32)
    for channel_index in range(masked_thumb.shape[-1]):
        channel = masked_thumb[..., channel_index].astype(np.float32)
        scaled = channel / UINT16_TO_UINT8_SCALE
        scaled[~mask] = 0.0
        output[..., channel_index] = scaled
    return output.astype(np.uint8)


def _compose_cell_dino_channels_chw(thumb: np.ndarray, *, mask: np.ndarray) -> np.ndarray:
    """Return Cell-DINO input as CHW uint8 without creating an intermediate HWC image."""

    if thumb.ndim != 3:
        raise ValueError(f"Expected thumb with shape (H, W, C), got {thumb.shape}")
    if thumb.shape[-1] != 2:
        raise ValueError(f"Expected exactly 2 channels (brdu, edu), got {thumb.shape[-1]}")

    height, width, _ = thumb.shape
    output = np.zeros((5, height, width), dtype=np.uint8)
    for channel_index in range(thumb.shape[-1]):
        channel = thumb[..., channel_index].astype(np.float32)
        channel[~mask] = 0.0
        output[channel_index] = (channel / UINT16_TO_UINT8_SCALE).astype(np.uint8)
    return output


def _load_model_bundle(path: Path) -> dict[str, Any]:
    bundle = joblib.load(path)
    if not isinstance(bundle, dict):
        raise TypeError(f"Expected dict model bundle at {path}, got {type(bundle)}")
    missing = sorted(REQUIRED_BUNDLE_KEYS - set(bundle))
    if missing:
        raise ValueError(f"Model bundle missing required keys: {missing}")
    if bundle["feature_extractor"] != "cell_dino_cp_vits8":
        raise ValueError(f"Unexpected feature extractor {bundle['feature_extractor']!r} in {path}")
    return bundle


def _load_obs_table(cfg: InferConfig) -> pd.DataFrame:
    if not cfg.h5ad_path.exists():
        raise FileNotFoundError(f"Input h5ad path not found: {cfg.h5ad_path}")

    if cfg.h5ad_path.is_dir():
        obs = pd.concat([load_obs_from_h5ad(path) for path in sorted(cfg.h5ad_path.glob("*.h5ad"))], axis=0)
        if obs.empty:
            raise FileNotFoundError(f"No .h5ad files found under directory: {cfg.h5ad_path}")
    else:
        obs = load_obs_from_h5ad(cfg.h5ad_path)
    table = obs.copy()
    table.index = table.index.astype(str)
    if table.index.has_duplicates:
        dup = table.index[table.index.duplicated()].unique().tolist()[:10]
        raise ValueError(f"Duplicate cell IDs in h5ad input (showing up to 10): {dup}")
    table["cell"] = table.index.astype(str)

    if "dataset" not in table.columns:
        if len(cfg.datasets) == 1:
            table["dataset"] = cfg.datasets[0]
        else:
            raise ValueError("Input obs is missing 'dataset'. Pass exactly one --dataset to define it.")

    required = {"cell", "dataset", "roi", "x", "y", "z"}
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"Input obs missing required columns: {missing}")

    table["cell"] = table["cell"].astype(str)
    table["dataset"] = table["dataset"].astype(str)
    table["roi"] = table["roi"].astype(str)

    if cfg.datasets:
        table = table[table["dataset"].isin(cfg.datasets)].copy()
    if cfg.rois:
        table = table[table["roi"].isin(cfg.rois)].copy()
    if table.empty:
        raise ValueError("No h5ad rows left after dataset/roi filtering.")

    table["_row_order"] = np.arange(len(table), dtype=np.int64)
    return table


def _build_transform(*, repo: Path, resize_size: int, crop_size: int) -> Any:
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from dinov2.data.cell_dino.transforms import make_classification_eval_cell_transform

    return make_classification_eval_cell_transform(resize_size=resize_size, crop_size=crop_size)


def _load_cell_dino_model(*, repo: Path, weights: Path, device: torch.device) -> Any:
    model = torch.hub.load(str(repo), "cell_dino_cp_vits8", source="local", pretrained_path=str(weights))
    return model.eval().to(device)


def _load_seg_slab(
    seg: zarr.Array,
    *,
    z_index: int,
    y0: int,
    y1: int,
    x0: int,
    x1: int,
) -> np.ndarray:
    if seg.ndim == 3:
        return np.asarray(seg[z_index, y0:y1, x0:x1])
    if seg.ndim == 4:
        return np.asarray(seg[z_index, y0:y1, x0:x1, 0])
    raise ValueError(f"Unexpected segmentation array ndim={seg.ndim} shape={seg.shape}")


def _batch_indices(n_rows: int, batch_size: int) -> list[tuple[int, int]]:
    return [(start, min(start + batch_size, n_rows)) for start in range(0, n_rows, batch_size)]


def _assemble_cell_dino_batch(
    *,
    batch_rows: pd.DataFrame,
    fused_slab: np.ndarray,
    seg_slab: np.ndarray,
    crop_size: int,
    slab_x0: int,
    slab_y0: int,
) -> np.ndarray:
    images = np.zeros((len(batch_rows), 5, crop_size, crop_size), dtype=np.uint8)

    for batch_index, row in enumerate(batch_rows.to_dict("records")):
        cell = str(row["cell"])
        row_roi, label = _cell_id_roi_and_label(cell)
        if row_roi != str(row["roi"]):
            raise ValueError(f"Cell id roi {row_roi!r} != row roi {row['roi']!r}")

        x_center = int(np.rint(float(row["x"]))) - slab_x0
        y_center = int(np.rint(float(row["y"]))) - slab_y0
        thumb = _crop_centered_yxc(fused_slab, x_center=x_center, y_center=y_center, size=crop_size)
        mask = _resolve_segmentation_mask_from_plane(
            seg_slab,
            x_center=x_center,
            y_center=y_center,
            size=crop_size,
            label=label,
        )
        if not mask.any():
            raise ValueError(f"Target cell mask is empty for {cell}")

        images[batch_index] = _compose_cell_dino_channels_chw(thumb, mask=mask)

    return images


def _predict_batch(
    *,
    batch_rows: pd.DataFrame,
    raw_batch: np.ndarray,
    model: Any,
    transform: Any,
    scaler: Any,
    brdu_model: Any,
    edu_model: Any,
    device: torch.device,
) -> pd.DataFrame:
    batch_cpu = torch.from_numpy(raw_batch).to(dtype=torch.float32)
    batch_cpu = transform(batch_cpu)
    if device.type == "cuda":
        batch_cpu = batch_cpu.pin_memory()

    batch = batch_cpu.to(device, non_blocking=device.type == "cuda")
    with torch.inference_mode():
        features = torch.nn.functional.normalize(model(batch), dim=1, p=2).cpu().numpy()

    scaled = scaler.transform(features)
    p_brdu = brdu_model.predict_proba(scaled)[:, 1].astype(np.float32)
    p_edu = edu_model.predict_proba(scaled)[:, 1].astype(np.float32)

    batch_out = batch_rows.loc[:, ["cell", "dataset", "roi", "_row_order"]].copy()
    batch_out["brdu_prob"] = p_brdu
    batch_out["edu_prob"] = p_edu
    return batch_out


def _predict_group(
    *,
    group: pd.DataFrame,
    model: Any,
    transform: Any,
    bundle: dict[str, Any],
    device: torch.device,
    batch_size: int,
) -> pd.DataFrame:
    dataset = str(group.iloc[0]["dataset"])
    roi = str(group.iloc[0]["roi"])
    ws = Workspace(_resolve_workspace_root(dataset))

    fused = zarr.open_array(ws.stitch(roi, bundle["codebook"]) / FUSED_ZARR_NAME, mode="r")
    seg = zarr.open_array(ws.stitch(roi, bundle["seg_codebook"]) / bundle["segmentation_name"], mode="r")

    channel_names_raw = fused.attrs.get("key")
    channel_names = list(channel_names_raw) if isinstance(channel_names_raw, list) else None
    if channel_names is None:
        raise ValueError(f"Missing fused channel names in attrs for dataset={dataset} roi={roi}")
    channel_indices = []
    for channel_name in ("brdu", "edu"):
        if channel_name not in channel_names:
            raise ValueError(f"Required channel {channel_name!r} not found in fused attrs keys={channel_names}")
        channel_indices.append(channel_names.index(channel_name))

    scaler = bundle["feature_scaler"]
    brdu_model = bundle["brdu_model"]
    edu_model = bundle["edu_model"]
    outputs: list[pd.DataFrame] = []
    t0 = time.perf_counter()
    crop_size = int(bundle["crop_size"])
    group = group.copy()
    group["_z_index"] = np.rint(group["z"].to_numpy(dtype=np.float64)).astype(np.int64)
    for z_index, z_group in group.groupby("_z_index", sort=False):
        y0, y1, x0, x1 = _slab_bounds_for_group(z_group, crop_size=crop_size, y_dim=int(fused.shape[1]), x_dim=int(fused.shape[2]))
        fused_slab = np.asarray(fused[int(z_index), y0:y1, x0:x1, channel_indices])
        seg_slab = _load_seg_slab(seg, z_index=int(z_index), y0=y0, y1=y1, x0=x0, x1=x1)

        for start, stop in _batch_indices(len(z_group), batch_size):
            batch_rows = z_group.iloc[start:stop].copy()
            raw_batch = _assemble_cell_dino_batch(
                batch_rows=batch_rows,
                fused_slab=fused_slab,
                seg_slab=seg_slab,
                crop_size=crop_size,
                slab_x0=x0,
                slab_y0=y0,
            )
            outputs.append(
                _predict_batch(
                    batch_rows=batch_rows,
                    raw_batch=raw_batch,
                    model=model,
                    transform=transform,
                    scaler=scaler,
                    brdu_model=brdu_model,
                    edu_model=edu_model,
                    device=device,
                )
            )

    elapsed = time.perf_counter() - t0
    logger.info(
        f"Predicted dataset={dataset} roi={roi} rows={len(group)} in {elapsed:.2f}s "
        f"({len(group) / max(elapsed, 1e-6):.1f} rows/s)."
    )
    return pd.concat(outputs, axis=0, ignore_index=True)


def main() -> None:
    cfg = _parse_args()
    bundle = _load_model_bundle(cfg.model_path)
    table = _load_obs_table(cfg)

    repo = cfg.dinov2_repo if cfg.dinov2_repo is not None else Path(bundle["dinov2_repo"]).expanduser()
    weights = cfg.weights if cfg.weights is not None else Path(bundle["weights"]).expanduser()
    transform = _build_transform(
        repo=repo,
        resize_size=int(bundle["resize_size"]),
        crop_size=int(bundle["crop_eval_size"]),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_cell_dino_model(repo=repo, weights=weights, device=device)

    group_outputs: list[pd.DataFrame] = []
    for (dataset, roi), group in table.groupby(["dataset", "roi"], sort=False):
        group = group.reset_index(drop=True)
        logger.info(f"Starting dataset={dataset} roi={roi} rows={len(group)}")
        group_outputs.append(
            _predict_group(
                group=group,
                model=model,
                transform=transform,
                bundle=bundle,
                device=device,
                batch_size=cfg.batch_size,
            )
        )

    out = pd.concat(group_outputs, axis=0, ignore_index=True)
    out = out.sort_values("_row_order").drop(columns="_row_order").set_index("cell")
    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(cfg.output_path)

    print(f"device={device}")
    print(f"h5ad_path={cfg.h5ad_path}")
    print(f"model_path={cfg.model_path}")
    print(f"output_path={cfg.output_path}")
    print(f"rows={len(out)}")
    print(f"groups={out[['dataset', 'roi']].drop_duplicates().shape[0]}")


if __name__ == "__main__":
    main()
