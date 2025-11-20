from __future__ import annotations

import random
import re
import shutil
from pathlib import Path
from typing import Iterable, Pattern, Sequence
import warnings

import numpy as np
import tifffile
import torch
from cellpose.contrib.packed_infer import PackedCellposeModelTRT
from loguru import logger
from pydantic import BaseModel, Field, model_validator

from fishtools.segment.data_discovery import _IMAGE_EXTENSIONS, _compile_patterns, _discover_training_dirs
from fishtools.utils.logging import setup_workspace_logging
from fishtools.utils.pretty_print import progress_bar
from fishtools.utils.utils import noglobal

_PLAN_SANITIZE_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")


def _sanitize_device_name(name: str) -> str:
    sanitized = _PLAN_SANITIZE_PATTERN.sub("_", name).strip("_")
    return sanitized or "cuda"


def _plan_path_for_device(model_path: Path, device_name: str) -> Path:
    safe_device_name = _sanitize_device_name(device_name)
    base_name = model_path.name
    return model_path.with_name(f"{base_name}-{safe_device_name}.plan")


class DistillConfig(BaseModel):
    outdir: str
    model_name: str
    training_paths: list[str]
    include: list[str] = Field(default_factory=list)
    exclude: list[str] = Field(default_factory=list)
    samples_per_directory: int = Field(gt=0)
    seed: int = 0
    channels: tuple[int, int] = (0, 0)
    normalization_percentiles: tuple[float, float] = (1.0, 99.5)


class _TeacherConfig(BaseModel):
    training_paths: list[str]
    include: list[str] = Field(default_factory=list)
    exclude: list[str] = Field(default_factory=list)


def _has_teacher_mask(image_path: Path) -> bool:
    stem = image_path.stem
    seg_path = image_path.parent / f"{stem}_seg.npy"
    if seg_path.is_file():
        return True
    mask_path = image_path.parent / f"{stem}_masks.tif"
    return mask_path.is_file()

    @model_validator(mode="after")
    def _validate(self) -> "DistillConfig":
        if len(self.channels) != 2:
            raise ValueError("channels must define exactly two entries (cytoplasm, nucleus).")
        low, high = self.normalization_percentiles
        if not 0.0 <= low < high <= 100.0:
            raise ValueError("Normalization percentiles must satisfy 0 <= low < high <= 100.")
        return self


def _strip_line_comments(text: str) -> str:
    lines = text.splitlines()
    kept = [line for line in lines if not line.lstrip().startswith("//")]
    return "\n".join(kept) + ("\n" if text.endswith("\n") else "")


def _load_distill_config(models_dir: Path, outdir: str) -> tuple[DistillConfig, Path]:
    config_path = models_dir / f"{outdir}.json"
    raw = config_path.read_text()
    config = DistillConfig.model_validate_json(_strip_line_comments(raw))
    if config.outdir != outdir:
        raise ValueError(f"Configured outdir {config.outdir!r} does not match requested outdir {outdir!r}.")
    return config, config_path


def _collect_teacher_samples(workspace: Path, *, model_name: str) -> set[Path]:
    models_dir = workspace / "models"
    teacher_config_path = models_dir / f"{model_name}.json"
    if not teacher_config_path.is_file():
        return set()

    try:
        raw = teacher_config_path.read_text()
        config = _TeacherConfig.model_validate_json(_strip_line_comments(raw))
    except Exception as exc:  # pragma: no cover
        logger.warning(
            "Failed to parse teacher config %s: %s. Distill will not filter training images.",
            teacher_config_path,
            exc,
        )
        return set()

    try:
        teacher_dirs = _discover_training_dirs(workspace, config.training_paths)
    except Exception as exc:  # pragma: no cover
        logger.warning(
            "Unable to resolve teacher training directories from %s: %s. Skipping filtering.",
            teacher_config_path,
            exc,
        )
        return set()

    include_regexes = _compile_patterns(config.include) if config.include else []
    exclude_regexes = _compile_patterns(config.exclude) if config.exclude else []

    teacher_images: set[Path] = set()
    for relative in teacher_dirs:
        absolute = workspace / relative
        images = _gather_images(absolute)
        filtered = _filter_image_paths(
            images,
            include=include_regexes,
            exclude=exclude_regexes,
            workspace=workspace,
        )
        for image_path in filtered:
            if not _has_teacher_mask(image_path):
                continue
            try:
                relative_image = image_path.relative_to(workspace)
            except ValueError:
                relative_image = image_path
            teacher_images.add(relative_image)

    return teacher_images


def _gather_images(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    files = [
        path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in _IMAGE_EXTENSIONS
    ]
    return sorted(files)


def _pattern_matches(patterns: Sequence[Pattern[str]], candidates: Sequence[str]) -> bool:
    return any(regex.search(candidate) for regex in patterns for candidate in candidates)


def _filter_image_paths(
    paths: Iterable[Path],
    *,
    include: list,
    exclude: list,
    workspace: Path,
) -> list[Path]:
    filtered: list[Path] = []
    for path in paths:
        try:
            relative = path.relative_to(workspace).as_posix()
        except ValueError:
            relative = path.as_posix()
        resolved = path.resolve(strict=False).as_posix()
        candidates = (relative, path.as_posix(), resolved)

        if include and not _pattern_matches(include, candidates):
            continue
        if exclude and _pattern_matches(exclude, candidates):
            continue
        filtered.append(path)
    return filtered


def _find_trt_plan(model_path: Path) -> tuple[Path, str] | None:
    if PackedCellposeModelTRT is None:
        return None
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        return None

    device_index = 0
    device_name = torch.cuda.get_device_name(device_index)
    plan_candidate = _plan_path_for_device(model_path, device_name)
    if plan_candidate.is_file():
        return plan_candidate, device_name
    return None


def _load_model(model_path: Path, *, logger) -> object:
    if PackedCellposeModelTRT is None:  # pragma: no cover - depends on env
        raise RuntimeError(
            "PackedCellposeModelTRT is unavailable; install cellpose with TensorRT support before distilling."
        )
    if not model_path.exists():
        raise FileNotFoundError(f"Model file {model_path} does not exist.")

    plan_selection = _find_trt_plan(model_path)
    if plan_selection is None:
        raise RuntimeError(
            "No TensorRT plan found for distillation. Build one via `segment run --trt-build` for the target GPU."
        )

    plan_path, device_name = plan_selection
    logger.info(
        f"Using TensorRT plan {plan_path.name} for CUDA device '{device_name}' while distilling {model_path.name}."
    )
    return PackedCellposeModelTRT(pretrained_model=str(plan_path), gpu=True)


def _write_mask(mask: np.ndarray, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    mask_to_write = np.asarray(mask, dtype=np.uint16)
    tifffile.imwrite(destination, mask_to_write)


def _run_inference(image_path: Path, model, *, channels: tuple[int, int], percentiles: tuple[float, float]):
    image = tifffile.imread(image_path)
    with warnings.catch_warnings():
        masks, *_ = model.eval(
            image,
            channels=channels,
            normalize={"percentile": percentiles, "normalize": True},
            batch_size=4,
        )
    return masks


def _process_image(
    *,
    image_path: Path,
    workspace: Path,
    output_root: Path,
    model,
    channels: tuple[int, int],
    percentiles: tuple[float, float],
) -> None:
    relative_image = image_path.relative_to(workspace)
    destination_image = output_root / relative_image
    destination_image.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image_path, destination_image)

    masks = _run_inference(image_path, model, channels=channels, percentiles=percentiles)
    mask_path = destination_image.with_name(f"{destination_image.stem}_masks.tif")
    _write_mask(masks, mask_path)


def _get_logger():
    return logger


@noglobal
def run_distill(workspace: Path, outdir: str) -> list[str]:
    models_dir = workspace / "models"
    config, config_path = _load_distill_config(models_dir, outdir)
    model_file = config_path.parent / config.model_name
    _logger = _get_logger()
    log_destination = setup_workspace_logging(
        workspace=workspace,
        component="segment.distill",
        file=f"models/{outdir}.json",
        extra={"model": config.model_name, "outdir": config.outdir},
    )
    _logger.debug(f"Segment distill logs routed to {log_destination}")

    include_regexes = _compile_patterns(config.include) if config.include else []
    exclude_regexes = _compile_patterns(config.exclude) if config.exclude else []

    discovered_dirs = _discover_training_dirs(workspace, config.training_paths)
    if not discovered_dirs:
        raise ValueError("No input directories discovered for distillation.")

    rng = random.Random(config.seed)
    model = _load_model(model_file, logger=_logger)
    teacher_samples = _collect_teacher_samples(workspace, model_name=config.model_name)
    output_root = workspace / config.outdir
    warnings: list[str] = []

    total_written = 0
    with progress_bar(len(discovered_dirs)) as advance:
        for relative_dir in discovered_dirs:
            absolute_dir = workspace / relative_dir
            all_images = _gather_images(absolute_dir)
            filtered_images = _filter_image_paths(
                all_images,
                include=include_regexes,
                exclude=exclude_regexes,
                workspace=workspace,
            )

            remaining_images = []
            skipped_for_teacher = 0
            for image_path in filtered_images:
                try:
                    relative_image = image_path.relative_to(workspace)
                except ValueError:
                    relative_image = image_path
                if relative_image in teacher_samples:
                    skipped_for_teacher += 1
                    continue
                remaining_images.append(image_path)

            if not remaining_images:
                _logger.info(f"No images matched filters in {relative_dir.as_posix()}; skipping.")
                advance(description=relative_dir.as_posix())
                continue

            if skipped_for_teacher:
                _logger.info(
                    f"Skipped {skipped_for_teacher} images in {relative_dir.as_posix()} because they exist in the teacher dataset."
                )

            desired = config.samples_per_directory
            if len(remaining_images) < desired:
                message = (
                    f"Requested {desired} images from {relative_dir.as_posix()} but only {len(remaining_images)} available "
                    "after filtering."
                )
                _logger.warning(message)
                warnings.append(message)

            take = min(desired, len(remaining_images))
            selected = rng.sample(remaining_images, take) if take else []
            for image_path in selected:
                _process_image(
                    image_path=image_path,
                    workspace=workspace,
                    output_root=output_root,
                    model=model,
                    channels=config.channels,
                    percentiles=config.normalization_percentiles,
                )
            total_written += take
            advance(description=relative_dir.as_posix())

    _logger.info(f"Distillation samples written: {total_written}")
    return warnings
