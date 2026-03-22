from __future__ import annotations

import argparse
import hashlib
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import zarr
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from fishtools.brdu.model import load_barrage_training_table
from fishtools.io.workspace import Workspace
from fishtools.segment.cell_thumbnail import cell_thumbnail_from_fused


WORKSPACE_ROOT_BASES = (Path("/working"), Path.home() / "nvme")
FUSED_ZARR_NAME = "fused.zarr"
C_POWER_RANGE = np.linspace(-6.0, 5.0, 45)
UINT16_TO_UINT8_SCALE = 256.0
FEATURE_CACHE_VERSION = "v2_cell_dino_model_select_masked_brdu_edu_uint8"

MODEL_INPUT_CHANNELS = {
    "cell_dino_cp_vits8": 5,
    "cell_dino_hpa_vitl16": 4,
}


@dataclass(frozen=True)
class TrainConfig:
    barrage_dir: Path
    model_out: Path
    feature_cache: Path
    dinov2_repo: Path
    hub_model: str
    input_channels: int
    weights: Path
    codebook: str
    seg_codebook: str
    segmentation_name: str
    crop_size: int
    resize_size: int
    crop_eval_size: int
    batch_size: int
    num_workers: int
    cv_folds: int
    test_fraction: float
    val_fraction: float
    limit: int
    seed: int


def _parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(
        description="Train frozen Cell-DINO logistic regressions on labeled BrdU/EdU barrage cells."
    )
    parser.add_argument("--barrage-dir", type=Path, default=Path("output/brdu_edu_barrage"))
    parser.add_argument(
        "--model-out",
        type=Path,
        default=None,
        help="Output joblib path. Default: <barrage-dir>/brdu_edu_cell_dino_hpa_vitl16_logreg.joblib",
    )
    parser.add_argument(
        "--feature-cache",
        type=Path,
        default=None,
        help="Output joblib path for cached Cell-DINO features. Default: <barrage-dir>/cell_dino_hpa_vitl16_feature_cache.joblib",
    )
    parser.add_argument("--dinov2-repo", type=Path, default=Path("~/dinov2").expanduser())
    parser.add_argument(
        "--hub-model",
        choices=sorted(MODEL_INPUT_CHANNELS),
        default="cell_dino_hpa_vitl16",
        help="Local torch.hub Cell-DINO model entrypoint.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("~/cell_dino_vitl16_pretrain_hpa_sc-d3ab8938.pth").expanduser(),
    )
    parser.add_argument("--codebook", default="edu")
    parser.add_argument("--seg-codebook", default="pi")
    parser.add_argument("--segmentation-name", default="output_segmentation-sam_postproc_s1-2-2_v500.zarr")
    parser.add_argument("--crop-size", type=int, default=100, help="Centered crop size used in barrage generation.")
    parser.add_argument("--resize-size", type=int, default=224)
    parser.add_argument("--crop-eval-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--limit", type=int, default=0, help="Optional cap on labeled rows for quick experiments.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    barrage_dir = Path(args.barrage_dir)
    hub_model = str(args.hub_model)
    model_out = (
        Path(args.model_out)
        if args.model_out is not None
        else (barrage_dir / f"brdu_edu_{hub_model}_logreg.joblib")
    )
    feature_cache = (
        Path(args.feature_cache)
        if args.feature_cache is not None
        else (barrage_dir / f"{hub_model}_feature_cache.joblib")
    )
    return TrainConfig(
        barrage_dir=barrage_dir,
        model_out=model_out,
        feature_cache=feature_cache,
        dinov2_repo=Path(args.dinov2_repo).expanduser(),
        hub_model=hub_model,
        input_channels=MODEL_INPUT_CHANNELS[hub_model],
        weights=Path(args.weights).expanduser(),
        codebook=str(args.codebook),
        seg_codebook=str(args.seg_codebook),
        segmentation_name=str(args.segmentation_name),
        crop_size=int(args.crop_size),
        resize_size=int(args.resize_size),
        crop_eval_size=int(args.crop_eval_size),
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        cv_folds=int(args.cv_folds),
        test_fraction=float(args.test_fraction),
        val_fraction=float(args.val_fraction),
        limit=int(args.limit),
        seed=int(args.seed),
    )


def _resolve_segmentation_mask(
    seg: zarr.Array,
    *,
    z_index: int,
    x_center: int,
    y_center: int,
    size: int,
    label: int,
) -> np.ndarray:
    half = size // 2
    y0 = y_center - half
    x0 = x_center - half

    ys0 = max(0, y0)
    ys1 = min(int(seg.shape[1]), y0 + size)
    xs0 = max(0, x0)
    xs1 = min(int(seg.shape[2]), x0 + size)

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
    return labels_full == label


def _compose_cell_dino_channels(masked_thumb: np.ndarray, *, mask: np.ndarray) -> np.ndarray:
    """Map masked BrdU/EdU into uint8 and zero-pad to the selected Cell-DINO input width."""

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


def _feature_cache_metadata(cfg: TrainConfig) -> dict[str, Any]:
    return {
        "feature_cache_version": FEATURE_CACHE_VERSION,
        "feature_extractor": cfg.hub_model,
        "input_channels": cfg.input_channels,
        "weights": str(cfg.weights),
        "codebook": cfg.codebook,
        "seg_codebook": cfg.seg_codebook,
        "segmentation_name": cfg.segmentation_name,
        "crop_size": cfg.crop_size,
        "resize_size": cfg.resize_size,
        "crop_eval_size": cfg.crop_eval_size,
        "uint16_to_uint8_scale": UINT16_TO_UINT8_SCALE,
    }


def _row_feature_cache_key(row: pd.Series) -> str:
    payload = "|".join(
        [
            str(row["dataset"]),
            str(row["roi"]),
            str(row["cell"]),
            str(int(np.rint(float(row["x"])))),
            str(int(np.rint(float(row["y"])))),
            str(int(np.rint(float(row["z"])))),
        ]
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _load_feature_cache(cfg: TrainConfig) -> dict[str, Any]:
    if not cfg.feature_cache.exists():
        return {"metadata": _feature_cache_metadata(cfg), "entries": {}}

    bundle = joblib.load(cfg.feature_cache)
    expected_metadata = _feature_cache_metadata(cfg)
    if bundle.get("metadata") != expected_metadata:
        logger.info(f"Ignoring incompatible feature cache at {cfg.feature_cache}")
        return {"metadata": expected_metadata, "entries": {}}

    entries = bundle.get("entries")
    if not isinstance(entries, dict):
        raise ValueError(f"Expected dict feature cache entries at {cfg.feature_cache}")
    return {"metadata": expected_metadata, "entries": entries}


def _save_feature_cache(cfg: TrainConfig, cache_bundle: dict[str, Any]) -> None:
    cfg.feature_cache.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(cache_bundle, cfg.feature_cache)


def _resolve_workspace_root(dataset: str) -> Path:
    for base in WORKSPACE_ROOT_BASES:
        candidate = base / dataset
        if candidate.exists():
            return candidate
    checked = ", ".join(str(base / dataset) for base in WORKSPACE_ROOT_BASES)
    raise FileNotFoundError(f"workspace_root not found for dataset {dataset!r}; checked: {checked}")


def _cell_id_roi_and_label(cell: str) -> tuple[str, int]:
    if "|" not in cell:
        raise ValueError(f"Expected cell id '<roi>|<label>' or '<dataset>:<roi>|<label>', got {cell!r}")
    roi_token, label_str = cell.split("|", 1)
    roi = roi_token.rsplit(":", 1)[-1]
    return roi, int(label_str)


class BarrageCellDataset(Dataset[tuple[np.ndarray, np.ndarray]]):
    """Load barrage-selected cell crops and format them for Cell-DINO ViT-S/8."""

    def __init__(self, rows: pd.DataFrame, *, cfg: TrainConfig) -> None:
        self.rows = rows.reset_index(drop=True).copy()
        self.cfg = cfg
        self.workspace_cache: dict[str, Workspace] = {}
        self.fused_cache: dict[tuple[str, str, str], zarr.Array] = {}
        self.seg_cache: dict[tuple[str, str, str, str], zarr.Array] = {}

        if str(cfg.dinov2_repo) not in sys.path:
            sys.path.insert(0, str(cfg.dinov2_repo))
        from dinov2.data.cell_dino.transforms import make_classification_eval_cell_transform

        self.transform = make_classification_eval_cell_transform(
            resize_size=cfg.resize_size,
            crop_size=cfg.crop_eval_size,
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        row = self.rows.iloc[index]
        image = self._load_masked_cell_image(row)
        target = np.asarray([int(row["brdu"]), int(row["edu"])], dtype=np.int64)
        tensor = self.transform(torch.from_numpy(np.moveaxis(image, -1, 0)).float())
        return tensor, target

    def _workspace_for_dataset(self, dataset: str) -> Workspace:
        ws = self.workspace_cache.get(dataset)
        if ws is None:
            workspace_root = _resolve_workspace_root(dataset)
            ws = Workspace(workspace_root)
            self.workspace_cache[dataset] = ws
        return ws

    def _load_masked_cell_image(self, row: pd.Series) -> np.ndarray:
        dataset = str(row["dataset"])
        roi = str(row["roi"])
        cell = str(row["cell"])
        x_center = int(np.rint(float(row["x"])))
        y_center = int(np.rint(float(row["y"])))
        z_index = int(np.rint(float(row["z"])))

        roi_from_id, label = _cell_id_roi_and_label(cell)
        if roi_from_id != roi:
            raise ValueError(f"Cell id roi {roi_from_id!r} != row roi {roi!r}")

        ws = self._workspace_for_dataset(dataset)
        fused_key = (dataset, roi, self.cfg.codebook)
        fused = self.fused_cache.get(fused_key)
        if fused is None:
            fused = zarr.open_array(ws.stitch(roi, self.cfg.codebook) / FUSED_ZARR_NAME, mode="r")
            self.fused_cache[fused_key] = fused

        channel_names_raw = fused.attrs.get("key")
        channel_names = list(channel_names_raw) if isinstance(channel_names_raw, list) else None
        if channel_names is None:
            raise ValueError(f"Missing fused channel names in attrs for dataset={dataset} roi={roi}")
        required_channels = ["brdu", "edu"]
        channel_indices = []
        for channel_name in required_channels:
            if channel_name not in channel_names:
                raise ValueError(
                    f"Required channel {channel_name!r} not found in fused attrs keys={channel_names}"
                )
            channel_indices.append(channel_names.index(channel_name))

        thumb = cell_thumbnail_from_fused(
            fused,
            z_index=z_index,
            x_center=x_center,
            y_center=y_center,
            size=self.cfg.crop_size,
            channels=channel_indices,
        )

        seg_key = (dataset, roi, self.cfg.seg_codebook, self.cfg.segmentation_name)
        seg = self.seg_cache.get(seg_key)
        if seg is None:
            seg = zarr.open_array(ws.stitch(roi, self.cfg.seg_codebook) / self.cfg.segmentation_name, mode="r")
            self.seg_cache[seg_key] = seg

        mask = _resolve_segmentation_mask(
            seg,
            z_index=z_index,
            x_center=x_center,
            y_center=y_center,
            size=self.cfg.crop_size,
            label=label,
        )
        if not mask.any():
            raise ValueError(f"Target cell mask is empty for {cell}")

        masked_thumb = thumb.copy()
        masked_thumb[~mask] = 0
        image = _compose_cell_dino_channels(masked_thumb, mask=mask)
        return image[..., : self.cfg.input_channels]


def _extract_features(
    *,
    model: Any,
    dataset: Dataset[tuple[np.ndarray, np.ndarray]],
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []

    for images, batch_targets in loader:
        images = images.to(device, non_blocking=True)
        with torch.inference_mode():
            batch_features = torch.nn.functional.normalize(model(images), dim=1, p=2)
        features.append(batch_features.cpu().numpy())
        targets.append(batch_targets.cpu().numpy())

    return np.concatenate(features, axis=0), np.concatenate(targets, axis=0)


def _extract_features_with_cache(
    *,
    model: Any,
    rows: pd.DataFrame,
    cfg: TrainConfig,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    cache_bundle: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, bool]:
    entries = cache_bundle["entries"]
    cache_keys = [_row_feature_cache_key(row) for _, row in rows.iterrows()]
    missing_mask = [cache_key not in entries for cache_key in cache_keys]
    missing_rows = rows.iloc[np.flatnonzero(missing_mask)].reset_index(drop=True)

    cache_updated = False
    if len(missing_rows):
        missing_dataset = BarrageCellDataset(missing_rows, cfg=cfg)
        missing_features, _missing_targets = _extract_features(
            model=model,
            dataset=missing_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
        )
        for (_, row), feature in zip(missing_rows.iterrows(), missing_features, strict=True):
            entries[_row_feature_cache_key(row)] = {"feature": feature.astype(np.float32, copy=False)}
        cache_updated = True

    logger.info(f"Feature cache at {cfg.feature_cache}: hits={len(rows) - len(missing_rows)}, misses={len(missing_rows)}")
    features = np.stack([entries[cache_key]["feature"] for cache_key in cache_keys], axis=0)
    targets = rows.loc[:, ["brdu", "edu"]].to_numpy(dtype=np.int64, copy=True)
    return features, targets, cache_updated


def _best_logreg_for_target_cv(
    *,
    features: np.ndarray,
    labels: np.ndarray,
    seed: int,
    cv_folds: int,
) -> tuple[float, float, int]:
    class_counts = np.bincount(labels.astype(np.int64))
    positive_counts = class_counts[class_counts > 0]
    if len(positive_counts) < 2:
        raise ValueError("Need both classes present to run stratified cross-validation.")

    n_splits = min(cv_folds, int(positive_counts.min()))
    if n_splits < 2:
        raise ValueError(f"Need at least 2 rows in each class for stratified CV, got counts={class_counts.tolist()}")

    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    best_c = 1.0
    best_accuracy = -1.0

    for exponent in C_POWER_RANGE:
        c_value = float(10.0**exponent)
        fold_accuracies: list[float] = []
        for fold_train_idx, fold_val_idx in splitter.split(features, labels):
            scaler = StandardScaler()
            fold_train = scaler.fit_transform(features[fold_train_idx])
            fold_val = scaler.transform(features[fold_val_idx])
            model = LogisticRegression(solver="lbfgs", C=c_value, max_iter=1_000, tol=1e-12)
            model.fit(fold_train, labels[fold_train_idx])
            fold_accuracy = accuracy_score(labels[fold_val_idx], model.predict(fold_val))
            fold_accuracies.append(float(fold_accuracy))
        mean_accuracy = float(np.mean(fold_accuracies))
        if mean_accuracy > best_accuracy:
            best_c = c_value
            best_accuracy = mean_accuracy

    return best_c, best_accuracy, n_splits


def _make_split_tables(train: pd.DataFrame, *, cfg: TrainConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    def _choose_stratify_labels(frame: pd.DataFrame) -> np.ndarray | None:
        candidates = [
            frame["brdu"].astype(str) + "_" + frame["edu"].astype(str),
            frame["brdu"].astype(str),
            frame["edu"].astype(str),
        ]
        for candidate in candidates:
            counts = candidate.value_counts()
            if not counts.empty and int(counts.min()) >= 2:
                return candidate.to_numpy()
        return None

    if cfg.limit > 0 and cfg.limit < len(train):
        limited_idx, _ = train_test_split(
            np.arange(len(train)),
            train_size=cfg.limit,
            random_state=cfg.seed,
            stratify=_choose_stratify_labels(train),
        )
        train = train.iloc[np.sort(limited_idx)].reset_index(drop=True)

    train_val_idx, test_idx = train_test_split(
        np.arange(len(train)),
        test_size=cfg.test_fraction,
        random_state=cfg.seed,
        stratify=_choose_stratify_labels(train),
    )
    train_val = train.iloc[np.sort(train_val_idx)].reset_index(drop=True)
    test = train.iloc[np.sort(test_idx)].reset_index(drop=True)
    return train_val, test


def _load_cell_dino_model(*, repo: Path, hub_model: str, weights: Path, device: torch.device) -> Any:
    model = torch.hub.load(str(repo), hub_model, source="local", pretrained_path=str(weights))
    return model.eval().to(device)


def main() -> None:
    cfg = _parse_args()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    train_table = load_barrage_training_table(barrage_dir=cfg.barrage_dir)
    train_table = train_table.drop_duplicates(subset=["cell"], keep="last").reset_index(drop=True)
    if len(train_table) < 8:
        raise ValueError(f"Need more labeled rows to fit/train split, got {len(train_table)}")

    train_rows, test_rows = _make_split_tables(train_table, cfg=cfg)
    logger.info(
        f"Using labeled barrage rows: train={len(train_rows)} (CV), test={len(test_rows)}, cv_folds={cfg.cv_folds}"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_bundle = _load_feature_cache(cfg)
    all_rows = pd.concat([train_rows, test_rows], axis=0, ignore_index=True)
    all_cache_keys = {_row_feature_cache_key(row) for _, row in all_rows.iterrows()}
    entries = cache_bundle["entries"]
    needs_extraction = any(cache_key not in entries for cache_key in all_cache_keys)
    model = (
        _load_cell_dino_model(repo=cfg.dinov2_repo, hub_model=cfg.hub_model, weights=cfg.weights, device=device)
        if needs_extraction
        else None
    )

    train_features, train_targets, train_cache_updated = _extract_features_with_cache(
        model=model,
        rows=train_rows,
        cfg=cfg,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        device=device,
        cache_bundle=cache_bundle,
    )
    test_features, test_targets, test_cache_updated = _extract_features_with_cache(
        model=model,
        rows=test_rows,
        cfg=cfg,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        device=device,
        cache_bundle=cache_bundle,
    )
    if train_cache_updated or test_cache_updated:
        _save_feature_cache(cfg, cache_bundle)

    brdu_c, brdu_cv_accuracy, brdu_cv_folds = _best_logreg_for_target_cv(
        features=train_features,
        labels=train_targets[:, 0],
        seed=cfg.seed,
        cv_folds=cfg.cv_folds,
    )
    edu_c, edu_cv_accuracy, edu_cv_folds = _best_logreg_for_target_cv(
        features=train_features,
        labels=train_targets[:, 1],
        seed=cfg.seed,
        cv_folds=cfg.cv_folds,
    )

    final_scaler = StandardScaler()
    final_train_scaled = final_scaler.fit_transform(train_features)
    final_test_scaled = final_scaler.transform(test_features)

    brdu_final = LogisticRegression(solver="lbfgs", C=brdu_c, max_iter=1_000, tol=1e-12)
    brdu_final.fit(final_train_scaled, train_targets[:, 0])
    edu_final = LogisticRegression(solver="lbfgs", C=edu_c, max_iter=1_000, tol=1e-12)
    edu_final.fit(final_train_scaled, train_targets[:, 1])

    brdu_test_accuracy = accuracy_score(test_targets[:, 0], brdu_final.predict(final_test_scaled))
    edu_test_accuracy = accuracy_score(test_targets[:, 1], edu_final.predict(final_test_scaled))

    bundle = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "feature_extractor": cfg.hub_model,
        "weights": str(cfg.weights),
        "dinov2_repo": str(cfg.dinov2_repo),
        "input_channels": int(cfg.input_channels),
        "feature_cache": str(cfg.feature_cache),
        "codebook": cfg.codebook,
        "seg_codebook": cfg.seg_codebook,
        "segmentation_name": cfg.segmentation_name,
        "crop_size": cfg.crop_size,
        "resize_size": cfg.resize_size,
        "crop_eval_size": cfg.crop_eval_size,
        "feature_dim": int(final_train_scaled.shape[1]),
        "feature_scaler": final_scaler,
        "cv_folds": int(cfg.cv_folds),
        "brdu_cv_folds": int(brdu_cv_folds),
        "edu_cv_folds": int(edu_cv_folds),
        "train_rows": int(len(train_rows)),
        "val_rows": 0,
        "test_rows": int(len(test_rows)),
        "brdu_best_c": float(brdu_c),
        "edu_best_c": float(edu_c),
        "brdu_model": brdu_final,
        "edu_model": edu_final,
    }
    cfg.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, cfg.model_out)

    print(f"device={device}")
    print(f"model_out={cfg.model_out}")
    print(f"feature_cache={cfg.feature_cache}")
    print(f"train_rows={len(train_rows)}")
    print(f"cv_folds={cfg.cv_folds}")
    print("val_rows=0")
    print(f"test_rows={len(test_rows)}")
    print(f"feature_dim={final_train_scaled.shape[1]}")
    print(f"brdu_best_c={brdu_c:.10g}")
    print(f"edu_best_c={edu_c:.10g}")
    print(f"brdu_cv_accuracy={brdu_cv_accuracy:.6f}")
    print(f"edu_cv_accuracy={edu_cv_accuracy:.6f}")
    print(f"brdu_test_accuracy={brdu_test_accuracy:.6f}")
    print(f"edu_test_accuracy={edu_test_accuracy:.6f}")


if __name__ == "__main__":
    main()
