from pathlib import Path
from typing import Any, cast

from loguru import logger
from pydantic import BaseModel

from fishtools.utils.utils import noglobal


class TrainConfig(BaseModel):
    name: str
    channels: tuple[int, int]
    training_paths: list[str]
    n_epochs: int = 200

    learning_rate: float = 0.008
    batch_size: int = 16
    bsize: int = 224
    weight_decay: float = 1e-5
    SGD: bool = False

    normalization_percs: tuple[float, float] = (1, 99.5)
    train_losses: list[float] = []


# models = sorted(
#     (file for file in (path / "models").glob("*") if "train_losses" not in file.name),
#     key=lambda x: x.stat().st_mtime,
#     reverse=True,
# )


def concat_output(
    path: Path, samples: list[str], mask_filter: list[str] | str = "_seg.npy", one_level_down: bool = False
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

    logger.info(f"Loading training data")
    out2 = concat_output(
        path,
        train_config.training_paths,
        mask_filter="_seg.npy",
        one_level_down=True,
    )

    logger.info(f"Training data loaded: {len(out2[0])} images")
    model_path, train_losses, test_losses = _train(out2, path, train_config=train_config, name=name)
    return train_config.model_copy(update=dict(train_losses=train_losses))
