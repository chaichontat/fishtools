from pathlib import Path
from typing import Annotated

import typer

from fishtools.segment.train import TrainConfig, run_train

app = typer.Typer()


@app.command("train", help="Train a model")
def train(
    path: Annotated[
        Path,
        typer.Argument(
            help="Path to the main training directory",
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
    ],
    name: Annotated[str, typer.Argument(help="Name of the model")],
):
    models_path = path / "models"
    if not models_path.exists():
        raise FileNotFoundError(
            f"Models path {models_path} does not exist. If this is the correct directory, create one."
        )

    config_path = models_path / f"{name}.json"

    try:
        train_config = TrainConfig.model_validate_json(config_path.read_text())
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Config file {name}.json not found in {models_path}. Please create it first."
        )

    config_path.write_text(run_train(name, path, train_config).model_dump_json(indent=2))


@app.command("run", help="Run a model")
def run(): ...


def main():
    app()


if __name__ == "__main__":
    app()
