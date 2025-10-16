import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Pattern, Sequence, cast

from loguru import logger
from pydantic import BaseModel, Field

from fishtools.utils.logging import setup_workspace_logging
from fishtools.utils.utils import noglobal


class TrainConfig(BaseModel):
    name: str
    base_model: str = "cyto3"
    channels: tuple[int, int]
    training_paths: list[str]
    include: list[str] = Field(default_factory=list)
    exclude: list[str] = Field(default_factory=list)
    n_epochs: int = 200

    learning_rate: float = 0.008
    batch_size: int = 16
    bsize: int = 224
    weight_decay: float = 1e-5
    SGD: bool = False

    normalization_percs: tuple[float, float] = (1, 99.5)
    train_losses: list[float] = []


_IMAGE_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


# Regex utilities are kept local to avoid cross-module dependencies while still
# providing clear error messages for invalid user-supplied patterns.
def _compile_patterns(patterns: Sequence[str]) -> list[Pattern[str]]:
    compiled: list[Pattern[str]] = []
    for pattern in patterns:
        try:
            compiled.append(re.compile(pattern))
        except re.error as exc:
            raise ValueError(f"Invalid regex pattern {pattern!r}: {exc}") from exc
    return compiled


def _matches_any(patterns: Sequence[Pattern[str]], value: str) -> bool:
    return any(regex.search(value) for regex in patterns)


def _iter_image_dirs(start: Path) -> Iterable[Path]:
    if start.is_file():
        yield start.parent
        return

    if not start.exists():
        return

    discovered_dirs: set[Path] = set()
    for file_path in start.rglob("*"):
        if not file_path.is_file():
            continue
        suffix = file_path.suffix.lower()
        if suffix in _IMAGE_EXTENSIONS or file_path.name.endswith("_seg.npy"):
            discovered_dirs.add(file_path.parent)

    if not discovered_dirs:
        if start.is_dir():
            yield start
        return

    for directory in sorted(discovered_dirs):
        yield directory


def _discover_training_dirs(root: Path, training_paths: Sequence[str]) -> list[Path]:
    discovered: list[Path] = []
    seen: set[Path] = set()

    for entry in training_paths:
        entry_path = Path(entry)
        if ".." in entry_path.parts:
            raise ValueError(f"Training path {entry} cannot contain '..'.")

        candidate = root / entry_path
        if not candidate.exists():
            raise FileNotFoundError(f"Training path {candidate} does not exist.")

        for directory in _iter_image_dirs(candidate):
            try:
                relative = directory.relative_to(root)
            except ValueError as exc:
                raise ValueError(
                    f"Discovered training directory {directory} is outside the provided training root {root}."
                ) from exc

            if ".." in relative.parts:
                raise ValueError(
                    f"Discovered training directory {directory} escapes the training root {root}. "
                    "Remove '..' segments from training_paths."
                )

            if relative in seen:
                continue

            seen.add(relative)
            discovered.append(relative)

    if not discovered:
        raise ValueError(
            "No training directories discovered under the configured training paths. "
            f"Checked entries={list(training_paths)}"
        )

    return discovered


def _filter_images_by_patterns(
    *,
    images: list,
    labels: list,
    image_names: Sequence[str | Path] | None,
    include_regexes: list[Pattern[str]] | None,
    exclude_regexes: list[Pattern[str]] | None,
    training_root: Path,
    resolved_sample_map: dict[Path, Path],
) -> tuple[list, list, list | None, int, int, int]:
    """Filter image/label pairs by include/exclude regexes.

    Preserves existing behavior and logs while isolating complexity.

    Returns
    -------
    filtered_images, filtered_labels, filtered_image_names, total_images, kept_images, excluded_images
    """
    if not (include_regexes or exclude_regexes):
        total = len(images)
        return images, labels, list(image_names) if image_names is not None else None, total, total, 0

    if image_names is None:
        raise ValueError("Cellpose did not provide image names; include/exclude filters require them.")

    # Normalize raw names and compute alias/resolved candidates for robust matching
    original_strings = [
        (name.as_posix() if isinstance(name, Path) else str(name)).replace("\\", "/") for name in image_names
    ]

    def _normalize_image_name(raw: Path | str) -> tuple[str, str]:
        raw_string = raw.as_posix() if isinstance(raw, Path) else str(raw)
        normalized_string = raw_string.replace("\\", "/")
        normalized_path = Path(normalized_string)

        if normalized_path.is_absolute():
            resolved_path = normalized_path.resolve(strict=False)
        else:
            resolved_path = (training_root / normalized_path).resolve(strict=False)

        alias_candidate: str | None = None
        for resolved_root, alias in resolved_sample_map.items():
            if resolved_path.is_relative_to(resolved_root):
                rebased = training_root / alias / resolved_path.relative_to(resolved_root)
                alias_candidate = rebased.as_posix()
                break

        return alias_candidate or normalized_string, resolved_path.as_posix()

    normalized_infos = [_normalize_image_name(name) for name in image_names]

    # records: (img, lbl, original_name, original_string, alias_string, resolved_string)
    records = [
        (img, lbl, original_name, original_string, alias_res[0], alias_res[1])
        for img, lbl, original_name, original_string, alias_res in zip(
            images, labels, image_names, original_strings, normalized_infos
        )
    ]

    def _record_candidates(record: tuple) -> tuple[str, str, str]:
        return record[3], record[4], record[5]

    def _patterns_match(patterns: list[Pattern[str]] | None, record: tuple) -> bool:
        if not patterns:
            return False
        candidates = _record_candidates(record)
        return any(regex.search(candidate) for regex in patterns for candidate in candidates)

    dropped_by_include = [
        record[4] for record in records if include_regexes and not _patterns_match(include_regexes, record)
    ]
    candidate_records = [
        record for record in records if not include_regexes or _patterns_match(include_regexes, record)
    ]
    dropped_by_exclude = [
        record[4] for record in candidate_records if _patterns_match(exclude_regexes, record)
    ]

    filtered_records = [
        record
        for record in candidate_records
        if not exclude_regexes or not _patterns_match(exclude_regexes, record)
    ]

    if dropped_by_include:
        logger.info(
            f"Skipped {len(dropped_by_include)} image paths due to include filters: {dropped_by_include}"
        )

    if dropped_by_exclude:
        logger.info(
            f"Skipped {len(dropped_by_exclude)} image paths due to exclude filters: {dropped_by_exclude}"
        )

    if not filtered_records:
        raise ValueError(
            "No training images remain after applying include/exclude filters."
            f" include={[r.pattern for r in include_regexes] if include_regexes else []},"
            f" exclude={[r.pattern for r in exclude_regexes] if exclude_regexes else []}"
        )

    filtered_images = [record[0] for record in filtered_records]
    filtered_labels = [record[1] for record in filtered_records]
    filtered_image_names = [record[2] for record in filtered_records]

    total_images = len(images)
    kept_images = len(filtered_images)
    excluded_images = total_images - kept_images

    return filtered_images, filtered_labels, filtered_image_names, total_images, kept_images, excluded_images


# models = sorted(
#     (file for file in (path / "models").glob("*") if "train_losses" not in file.name),
#     key=lambda x: x.stat().st_mtime,
#     reverse=True,
# )


def concat_output(
    path: Path,
    samples: list[str],
    mask_filter: list[str] | str = "_seg.npy",
    one_level_down: bool = False,
) -> tuple[list, ...]:
    from .cp_io import load_train_test_data

    if isinstance(mask_filter, list) and len(samples) != len(mask_filter):
        raise ValueError("Number of samples must match number of mask filters.")

    first: tuple = load_train_test_data(
        (path / samples[0]).as_posix(),
        mask_filter=mask_filter if isinstance(mask_filter, str) else mask_filter[0],
        look_one_level_down=one_level_down,
    )
    for i, sample in enumerate(samples[1:], 1):
        _out = load_train_test_data(
            (path / sample).as_posix(),
            # test_dir=(path.parent / "pi-wgatest").as_posix(),
            mask_filter=mask_filter if isinstance(mask_filter, str) else mask_filter[i],
            look_one_level_down=one_level_down,
        )

        for lis, new in zip(first, _out):
            if lis is not None and new is not None:
                lis.extend(new)
    if first[0] is not None:
        assert len(first[0]) == len(first[2])

    return first


@noglobal
def _train(out: tuple[Any, ...], path: Path, name: str, train_config: TrainConfig):
    from cellpose.models import CellposeModel
    from cellpose.train import train_seg

    # if name is None:
    # name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    images, labels, image_names, test_images, test_labels, image_names_test = out

    model = CellposeModel(gpu=True, pretrained_model=cast(bool, train_config.base_model))
    model_path, train_losses, test_losses = train_seg(
        model.net,
        train_data=images,
        train_labels=labels,
        save_path=path,  # /models automatically appended
        channels=train_config.channels,
        batch_size=train_config.batch_size,
        channel_axis=0,
        test_data=test_images,
        test_labels=test_labels,
        weight_decay=train_config.weight_decay,
        SGD=train_config.SGD,
        learning_rate=train_config.learning_rate,
        n_epochs=train_config.n_epochs,
        model_name=name,
        bsize=train_config.bsize,
        normalize=cast(bool, {"percentile": train_config.normalization_percs}),
        min_train_masks=1,
    )
    return model_path, train_losses, test_losses


# @noglobal
def run_train(name: str, path: Path, train_config: TrainConfig):
    log_destination = setup_workspace_logging(
        workspace=path,
        component="segment.train",
        file=f"models/{name}.json",
        extra={"model": name, "training_root": path.as_posix()},
    )
    logger.debug(f"Segment training logs routed to {log_destination}")

    logger.info(f"Started training {name} with paths: {train_config.training_paths}")

    include_regexes = _compile_patterns(train_config.include) if train_config.include else []
    exclude_regexes = _compile_patterns(train_config.exclude) if train_config.exclude else []

    discovered_samples = _discover_training_dirs(path, train_config.training_paths)
    if exclude_regexes:
        kept_samples: list[Path] = []
        dropped_samples: list[Path] = []

        for sample in discovered_samples:
            alias_path = path / sample
            resolved_path = alias_path.resolve(strict=False)
            sample_values = (sample.as_posix(), alias_path.as_posix(), resolved_path.as_posix())
            if any(_matches_any(exclude_regexes, value) for value in sample_values):
                dropped_samples.append(sample)
            else:
                kept_samples.append(sample)

        if dropped_samples:
            logger.info(
                "Skipped %d training directories due to exclude filters: %s",
                len(dropped_samples),
                [sample.as_posix() for sample in dropped_samples],
            )

        discovered_samples = kept_samples

    if not discovered_samples:
        raise ValueError(
            "No training images remain after applying include/exclude filters."
            f" include={train_config.include}, exclude={train_config.exclude}"
        )

    resolved_sample_map: dict[Path, Path] = {}
    for sample in discovered_samples:
        alias_path = path / sample
        try:
            resolved_target = alias_path.resolve(strict=True)
        except FileNotFoundError:
            resolved_target = alias_path.resolve(strict=False)
        resolved_sample_map.setdefault(resolved_target, sample)

    logger.info(
        f"Loading training data from {len(discovered_samples)} discovered directories: "
        f"{[sample.as_posix() for sample in discovered_samples]}"
    )

    out2 = concat_output(
        path,
        [sample.as_posix() for sample in discovered_samples],
        mask_filter="_seg.npy",
        one_level_down=True,
    )

    logger.info(f"Pre-filter training discovered {len(out2[0])} images")
    images, labels, image_names, test_images, test_labels, image_names_test = out2

    (
        filtered_images,
        filtered_labels,
        filtered_image_names,
        total_images,
        kept_images,
        excluded_images,
    ) = _filter_images_by_patterns(
        images=images,
        labels=labels,
        image_names=image_names,
        include_regexes=include_regexes,
        exclude_regexes=exclude_regexes,
        training_root=path,
        resolved_sample_map=resolved_sample_map,
    )

    logger.info(f"Filter summary: kept {kept_images}/{total_images} images; excluded {excluded_images}.")

    filtered: tuple[list, ...] = (
        filtered_images,
        filtered_labels,
        filtered_image_names if filtered_image_names is not None else None,
        test_images,
        test_labels,
        image_names_test,
    )

    if not filtered[0]:
        raise ValueError(
            "No training images remain after applying include/exclude filters."
            f" include={train_config.include}, exclude={train_config.exclude}"
        )

    _model_path, train_losses, _test_losses = _train(filtered, path, train_config=train_config, name=name)

    normalized_train_losses = train_losses.tolist() if hasattr(train_losses, "tolist") else list(train_losses)

    return train_config.model_copy(update=dict(train_losses=normalized_train_losses))
