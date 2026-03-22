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
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from fishtools.brdu.barrage import load_obs_from_h5ad
from fishtools.io.workspace import Workspace


WORKSPACE_ROOT_BASES = (Path("/working"), Path.home() / "nvme")
FUSED_ZARR_NAME = "fused.zarr"
MODEL_INPUT_CHANNELS = {
    "cell_dino_cp_vits8": 5,
    "cell_dino_hpa_vitl16": 4,
}
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
    output_path: Path | None
    dinov2_repo: Path | None
    weights: Path | None
    datasets: tuple[str, ...]
    rois: tuple[str, ...]
    batch_size: int
    num_workers: int


@dataclass(frozen=True)
class _SlabBatchSpec:
    row_indices: tuple[int, ...]
    z_index: int
    y0: int
    y1: int
    x0: int
    x1: int


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
        help="Model bundle path. Default: <barrage-dir>/brdu_edu_cell_dino_hpa_vitl16_logreg.joblib",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Output parquet path. Default: ~/nvme/dinoinfer/{dataset}-dino-vit16.parquet",
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
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    barrage_dir = Path(args.barrage_dir)
    model_path = (
        Path(args.model_path)
        if args.model_path is not None
        else (barrage_dir / "brdu_edu_cell_dino_hpa_vitl16_logreg.joblib")
    )
    return InferConfig(
        barrage_dir=barrage_dir,
        h5ad_path=Path(args.h5ad).expanduser(),
        model_path=model_path,
        output_path=Path(args.output_path).expanduser() if args.output_path is not None else None,
        dinov2_repo=Path(args.dinov2_repo).expanduser() if args.dinov2_repo is not None else None,
        weights=Path(args.weights).expanduser() if args.weights is not None else None,
        datasets=tuple(str(x) for x in args.datasets),
        rois=tuple(str(x) for x in args.rois),
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
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
    if "_x_center" in group.columns:
        x_centers = group["_x_center"].to_numpy(dtype=np.int64, copy=False)
    else:
        x_centers = np.rint(group["x"].to_numpy(dtype=np.float64)).astype(np.int64)
    if "_y_center" in group.columns:
        y_centers = group["_y_center"].to_numpy(dtype=np.int64, copy=False)
    else:
        y_centers = np.rint(group["y"].to_numpy(dtype=np.float64)).astype(np.int64)

    y0 = max(0, int(y_centers.min()) - half)
    x0 = max(0, int(x_centers.min()) - half)
    y1 = min(y_dim, int(y_centers.max()) - half + crop_size)
    x1 = min(x_dim, int(x_centers.max()) - half + crop_size)
    return y0, y1, x0, x1


def _make_slab_batch_specs(
    group: pd.DataFrame,
    *,
    crop_size: int,
    y_dim: int,
    x_dim: int,
    batch_size: int,
) -> tuple[_SlabBatchSpec, ...]:
    specs: list[_SlabBatchSpec] = []
    for z_index, z_group in group.groupby("_z_index", sort=False):
        y0, y1, x0, x1 = _slab_bounds_for_group(z_group, crop_size=crop_size, y_dim=y_dim, x_dim=x_dim)
        row_indices = z_group.index.to_numpy(dtype=np.int64, copy=False)
        for start, stop in _batch_indices(len(row_indices), batch_size):
            specs.append(
                _SlabBatchSpec(
                    row_indices=tuple(int(x) for x in row_indices[start:stop]),
                    z_index=int(z_index),
                    y0=y0,
                    y1=y1,
                    x0=x0,
                    x1=x1,
                )
            )
    return tuple(specs)


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

    output = np.zeros((*masked_thumb.shape[:2], 5), dtype=np.uint8)
    for channel_index in range(masked_thumb.shape[-1]):
        scaled = np.right_shift(masked_thumb[..., channel_index], 8).astype(np.uint8, copy=False)
        np.copyto(output[..., channel_index], scaled, where=mask)
    return output


def _load_model_bundle(path: Path) -> dict[str, Any]:
    bundle = joblib.load(path)
    if not isinstance(bundle, dict):
        raise TypeError(f"Expected dict model bundle at {path}, got {type(bundle)}")
    missing = sorted(REQUIRED_BUNDLE_KEYS - set(bundle))
    if missing:
        raise ValueError(f"Model bundle missing required keys: {missing}")
    feature_extractor = str(bundle["feature_extractor"])
    if feature_extractor not in MODEL_INPUT_CHANNELS:
        raise ValueError(f"Unexpected feature extractor {feature_extractor!r} in {path}")
    input_channels = int(bundle.get("input_channels", MODEL_INPUT_CHANNELS[feature_extractor]))
    if input_channels != MODEL_INPUT_CHANNELS[feature_extractor]:
        raise ValueError(
            f"Bundle input_channels={input_channels} does not match feature_extractor={feature_extractor!r}"
        )
    return bundle


def _default_output_path_for_dataset(dataset: str) -> Path:
    return (Path("~/nvme/dinoinfer").expanduser() / f"{dataset}-dino-vit16.parquet").expanduser()


def _resolve_output_path(cfg: InferConfig, table: pd.DataFrame) -> Path:
    if cfg.output_path is not None:
        return cfg.output_path
    datasets = table["dataset"].astype(str).drop_duplicates().tolist()
    if len(datasets) != 1:
        raise ValueError("Default output path requires exactly one dataset after filtering. Pass --output-path.")
    return _default_output_path_for_dataset(datasets[0])


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

    x_centers = np.rint(table["x"].to_numpy(dtype=np.float64)).astype(np.int64)
    y_centers = np.rint(table["y"].to_numpy(dtype=np.float64)).astype(np.int64)
    z_indices = np.rint(table["z"].to_numpy(dtype=np.float64)).astype(np.int64)
    labels = np.empty(len(table), dtype=np.int64)
    for index, (cell, roi) in enumerate(
        zip(table["cell"].astype(str).to_numpy(), table["roi"].astype(str).to_numpy(), strict=False)
    ):
        row_roi, label = _cell_id_roi_and_label(cell)
        if row_roi != roi:
            raise ValueError(f"Cell id roi {row_roi!r} != row roi {roi!r}")
        labels[index] = label

    table["_x_center"] = x_centers
    table["_y_center"] = y_centers
    table["_z_index"] = z_indices
    table["_label"] = labels
    table["_row_order"] = np.arange(len(table), dtype=np.int64)
    return table


def _build_transform(*, repo: Path, resize_size: int, crop_size: int) -> Any:
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from dinov2.data.cell_dino.transforms import make_classification_eval_cell_transform

    return make_classification_eval_cell_transform(resize_size=resize_size, crop_size=crop_size)


def _load_cell_dino_model(*, repo: Path, hub_model: str, weights: Path, device: torch.device) -> Any:
    model = torch.hub.load(str(repo), hub_model, source="local", pretrained_path=str(weights))
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


def _identity_collate(sample: Any) -> Any:
    return sample


def _worker_batch_spec_bounds(n_items: int, *, worker_id: int, num_workers: int) -> tuple[int, int]:
    start = (n_items * worker_id) // num_workers
    stop = (n_items * (worker_id + 1)) // num_workers
    return start, stop


class _SlabBatchIterableDataset(IterableDataset[tuple[torch.Tensor, dict[str, Any]]]):
    """Yield transformed Cell-DINO batches while preserving ROI/z-slab locality."""

    def __init__(
        self,
        group: pd.DataFrame,
        *,
        fused_path: Path,
        seg_path: Path,
        channel_indices: tuple[int, ...],
        crop_size: int,
        input_channels: int,
        resize_size: int,
        crop_eval_size: int,
        batch_size: int,
        repo: Path,
        y_dim: int,
        x_dim: int,
    ) -> None:
        self.cells = group["cell"].astype(str).to_numpy()
        self.datasets = group["dataset"].astype(str).to_numpy()
        self.rois = group["roi"].astype(str).to_numpy()
        self.row_orders = group["_row_order"].to_numpy(dtype=np.int64, copy=False)
        self.x_centers = group["_x_center"].to_numpy(dtype=np.int64, copy=False)
        self.y_centers = group["_y_center"].to_numpy(dtype=np.int64, copy=False)
        self.labels = group["_label"].to_numpy(dtype=np.int64, copy=False)
        self.fused_path = fused_path
        self.seg_path = seg_path
        self.channel_indices = channel_indices
        self.crop_size = crop_size
        self.input_channels = input_channels
        self.resize_size = resize_size
        self.crop_eval_size = crop_eval_size
        self.repo = repo
        self.batch_specs = _make_slab_batch_specs(
            group,
            crop_size=crop_size,
            y_dim=y_dim,
            x_dim=x_dim,
            batch_size=batch_size,
        )

    def __iter__(self) -> Any:
        worker_info = get_worker_info()
        if worker_info is None:
            start, stop = 0, len(self.batch_specs)
        else:
            start, stop = _worker_batch_spec_bounds(
                len(self.batch_specs),
                worker_id=worker_info.id,
                num_workers=worker_info.num_workers,
            )
        if start >= stop:
            return

        transform = _build_transform(repo=self.repo, resize_size=self.resize_size, crop_size=self.crop_eval_size)
        fused = zarr.open_array(self.fused_path, mode="r")
        seg = zarr.open_array(self.seg_path, mode="r")
        current_slab_key: tuple[int, int, int, int, int] | None = None
        fused_slab: np.ndarray | None = None
        seg_slab: np.ndarray | None = None

        for spec in self.batch_specs[start:stop]:
            slab_key = (spec.z_index, spec.y0, spec.y1, spec.x0, spec.x1)
            if slab_key != current_slab_key:
                fused_slab = np.asarray(
                    fused[spec.z_index, spec.y0 : spec.y1, spec.x0 : spec.x1, list(self.channel_indices)]
                )
                seg_slab = _load_seg_slab(
                    seg,
                    z_index=spec.z_index,
                    y0=spec.y0,
                    y1=spec.y1,
                    x0=spec.x0,
                    x1=spec.x1,
                )
                current_slab_key = slab_key

            if fused_slab is None or seg_slab is None:
                raise RuntimeError("Expected slab cache to be initialized before batch assembly.")

            row_indices = np.asarray(spec.row_indices, dtype=np.int64)
            images = np.zeros((len(row_indices), self.input_channels, self.crop_size, self.crop_size), dtype=np.uint8)
            for batch_index, row_index in enumerate(row_indices):
                has_mask = _copy_cell_into_batch_image(
                    images[batch_index],
                    fused_slab=fused_slab,
                    seg_slab=seg_slab,
                    x_center=int(self.x_centers[row_index]) - spec.x0,
                    y_center=int(self.y_centers[row_index]) - spec.y0,
                    crop_size=self.crop_size,
                    label=int(self.labels[row_index]),
                )
                if not has_mask:
                    raise ValueError(f"Target cell mask is empty for {self.cells[row_index]}")

            batch_tensor = transform(torch.from_numpy(images).to(dtype=torch.float32))
            yield batch_tensor, {
                "cell": self.cells[row_indices].tolist(),
                "dataset": self.datasets[row_indices].tolist(),
                "roi": self.rois[row_indices].tolist(),
                "_row_order": self.row_orders[row_indices].copy(),
            }


def _copy_cell_into_batch_image(
    batch_image: np.ndarray,
    *,
    fused_slab: np.ndarray,
    seg_slab: np.ndarray,
    x_center: int,
    y_center: int,
    crop_size: int,
    label: int,
) -> bool:
    """Write one masked Cell-DINO crop directly into a preallocated CHW batch slot."""

    if batch_image.ndim != 3:
        raise ValueError(f"Expected batch_image with shape (C, H, W), got {batch_image.shape}")

    half = crop_size // 2
    y0 = y_center - half
    y1 = y0 + crop_size
    x0 = x_center - half
    x1 = x0 + crop_size

    src_y0 = max(0, y0)
    src_y1 = min(seg_slab.shape[0], y1)
    src_x0 = max(0, x0)
    src_x1 = min(seg_slab.shape[1], x1)
    if src_y0 >= src_y1 or src_x0 >= src_x1:
        return False

    dst_y0 = src_y0 - y0
    dst_y1 = dst_y0 + (src_y1 - src_y0)
    dst_x0 = src_x0 - x0
    dst_x1 = dst_x0 + (src_x1 - src_x0)
    mask = seg_slab[src_y0:src_y1, src_x0:src_x1] == label
    if not mask.any():
        return False

    for channel_index in range(min(2, batch_image.shape[0])):
        src = fused_slab[src_y0:src_y1, src_x0:src_x1, channel_index]
        dst = batch_image[channel_index, dst_y0:dst_y1, dst_x0:dst_x1]
        scaled = np.right_shift(src, 8).astype(np.uint8, copy=False)
        np.copyto(dst, scaled, where=mask)
    return True


def _assemble_cell_dino_batch(
    *,
    batch_rows: pd.DataFrame,
    fused_slab: np.ndarray,
    seg_slab: np.ndarray,
    crop_size: int,
    slab_x0: int,
    slab_y0: int,
    input_channels: int,
) -> np.ndarray:
    images = np.zeros((len(batch_rows), input_channels, crop_size, crop_size), dtype=np.uint8)

    cells = batch_rows["cell"].astype(str).to_numpy()
    x_centers = batch_rows["_x_center"].to_numpy(dtype=np.int64, copy=False) - slab_x0
    y_centers = batch_rows["_y_center"].to_numpy(dtype=np.int64, copy=False) - slab_y0
    labels = batch_rows["_label"].to_numpy(dtype=np.int64, copy=False)

    for batch_index, (cell, x_center, y_center, label) in enumerate(
        zip(cells, x_centers, y_centers, labels, strict=False)
    ):
        has_mask = _copy_cell_into_batch_image(
            images[batch_index],
            fused_slab=fused_slab,
            seg_slab=seg_slab,
            x_center=int(x_center),
            y_center=int(y_center),
            crop_size=crop_size,
            label=int(label),
        )
        if not has_mask:
            raise ValueError(f"Target cell mask is empty for {cell}")

    return images


def _build_batch_loader(
    dataset: IterableDataset[tuple[torch.Tensor, dict[str, Any]]],
    *,
    num_workers: int,
    device: torch.device,
) -> DataLoader[Any]:
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": None,
        "shuffle": False,
        "num_workers": num_workers,
        "collate_fn": _identity_collate,
        "pin_memory": device.type == "cuda",
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(**kwargs)


def _predict_transformed_batch(
    *,
    batch_meta: dict[str, Any],
    batch_images: torch.Tensor,
    model: Any,
    scaler: Any,
    brdu_model: Any,
    edu_model: Any,
    device: torch.device,
) -> pd.DataFrame:
    batch = batch_images.to(device, non_blocking=device.type == "cuda")
    with torch.inference_mode():
        features = torch.nn.functional.normalize(model(batch), dim=1, p=2).cpu().numpy()

    scaled = scaler.transform(features)
    batch_out = pd.DataFrame(
        {
            "cell": list(batch_meta["cell"]),
            "dataset": list(batch_meta["dataset"]),
            "roi": list(batch_meta["roi"]),
            "_row_order": np.asarray(batch_meta["_row_order"], dtype=np.int64),
        }
    )
    batch_out["brdu_prob"] = brdu_model.predict_proba(scaled)[:, 1].astype(np.float32)
    batch_out["edu_prob"] = edu_model.predict_proba(scaled)[:, 1].astype(np.float32)
    return batch_out


def _predict_group(
    *,
    group: pd.DataFrame,
    model: Any,
    bundle: dict[str, Any],
    repo: Path,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> pd.DataFrame:
    dataset = str(group.iloc[0]["dataset"])
    roi = str(group.iloc[0]["roi"])
    ws = Workspace(_resolve_workspace_root(dataset))
    fused_path = ws.stitch(roi, bundle["codebook"]) / FUSED_ZARR_NAME
    seg_path = ws.stitch(roi, bundle["seg_codebook"]) / bundle["segmentation_name"]
    fused = zarr.open_array(fused_path, mode="r")

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
    input_channels = int(bundle.get("input_channels", MODEL_INPUT_CHANNELS[str(bundle["feature_extractor"])]))
    outputs: list[pd.DataFrame] = []
    t0 = time.perf_counter()
    crop_size = int(bundle["crop_size"])
    batch_dataset = _SlabBatchIterableDataset(
        group,
        fused_path=fused_path,
        seg_path=seg_path,
        channel_indices=tuple(channel_indices),
        crop_size=crop_size,
        input_channels=input_channels,
        resize_size=int(bundle["resize_size"]),
        crop_eval_size=int(bundle["crop_eval_size"]),
        batch_size=batch_size,
        repo=repo,
        y_dim=int(fused.shape[1]),
        x_dim=int(fused.shape[2]),
    )
    loader = _build_batch_loader(batch_dataset, num_workers=num_workers, device=device)
    for batch_images, batch_meta in loader:
        outputs.append(
            _predict_transformed_batch(
                batch_meta=batch_meta,
                batch_images=batch_images,
                model=model,
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
    return pd.concat(outputs, axis=0, ignore_index=True).sort_values("_row_order", kind="stable").reset_index(drop=True)


def main() -> None:
    cfg = _parse_args()
    bundle = _load_model_bundle(cfg.model_path)
    table = _load_obs_table(cfg)
    output_path = _resolve_output_path(cfg, table)

    repo = cfg.dinov2_repo if cfg.dinov2_repo is not None else Path(bundle["dinov2_repo"]).expanduser()
    weights = cfg.weights if cfg.weights is not None else Path(bundle["weights"]).expanduser()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_cell_dino_model(
        repo=repo,
        hub_model=str(bundle["feature_extractor"]),
        weights=weights,
        device=device,
    )

    group_outputs: list[pd.DataFrame] = []
    for (dataset, roi), group in table.groupby(["dataset", "roi"], sort=False):
        group = group.reset_index(drop=True)
        logger.info(f"Starting dataset={dataset} roi={roi} rows={len(group)}")
        group_outputs.append(
            _predict_group(
                group=group,
                model=model,
                bundle=bundle,
                repo=repo,
                device=device,
                batch_size=cfg.batch_size,
                num_workers=cfg.num_workers,
            )
        )

    out = pd.concat(group_outputs, axis=0, ignore_index=True)
    out = out.sort_values("_row_order").drop(columns="_row_order").set_index("cell")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path)

    print(f"device={device}")
    print(f"h5ad_path={cfg.h5ad_path}")
    print(f"model_path={cfg.model_path}")
    print(f"output_path={output_path}")
    print(f"rows={len(out)}")
    print(f"groups={out[['dataset', 'roi']].drop_duplicates().shape[0]}")


if __name__ == "__main__":
    main()
