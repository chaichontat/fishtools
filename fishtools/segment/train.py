import re
from pathlib import Path
from typing import Any, Pattern, Sequence, cast

from loguru import logger
from pydantic import BaseModel, Field

from fishtools.utils.utils import noglobal


class TrainConfig(BaseModel):
    name: str
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
    from cellpose.io import load_train_test_data

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

    return first


@noglobal
def _train(out: tuple[Any, ...], path: Path, name: str, train_config: TrainConfig):
    from cellpose.models import CellposeModel
    from cellpose.train import train_seg

    # if name is None:
    # name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    images, labels, image_names, test_images, test_labels, image_names_test = out

    model = CellposeModel(gpu=True, pretrained_model=cast(bool, "cyto3"))
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
    logger.info(f"Started training {name} with paths: {train_config.training_paths}")

    logger.info("Loading training data")
    out2 = concat_output(
        path,
        train_config.training_paths,
        mask_filter="_seg.npy",
        one_level_down=True,
    )

    logger.info(f"Training data loaded: {len(out2[0])} images")
    images, labels, image_names, test_images, test_labels, image_names_test = out2

    # Build resolved paths from image_names; if names are relative, use as-is.
    if train_config.include or train_config.exclude:
        if image_names is None:
            raise ValueError("Cellpose did not provide image names; include/exclude filters require them.")

        include_regexes = _compile_patterns(train_config.include)
        exclude_regexes = _compile_patterns(train_config.exclude)

        normalized_paths = [
            (name.as_posix() if isinstance(name, Path) else str(name)).replace("\\", "/")
            for name in image_names
        ]

        records = [
            (img, lbl, original_name, normalized_path)
            for img, lbl, original_name, normalized_path in zip(
                images, labels, image_names, normalized_paths
            )
        ]

        dropped_by_include = [
            normalized_path
            for _, _, _, normalized_path in records
            if include_regexes and not _matches_any(include_regexes, normalized_path)
        ]

        candidate_records = [
            record
            for record in records
            if not include_regexes or _matches_any(include_regexes, record[3])
        ]

        dropped_by_exclude = [
            normalized_path
            for _, _, _, normalized_path in candidate_records
            if exclude_regexes and _matches_any(exclude_regexes, normalized_path)
        ]

        filtered_records = [
            record
            for record in candidate_records
            if not exclude_regexes or not _matches_any(exclude_regexes, record[3])
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
                f" include={train_config.include}, exclude={train_config.exclude}"
            )

        filtered_images = [record[0] for record in filtered_records]
        filtered_labels = [record[1] for record in filtered_records]
        filtered_image_names = [record[2] for record in filtered_records]
    else:
        filtered_images = images
        filtered_labels = labels
        filtered_image_names = image_names

    total_images = len(images)
    kept_images = len(filtered_images)
    excluded_images = total_images - kept_images
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
