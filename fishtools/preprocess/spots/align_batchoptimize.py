# %%
import json
import subprocess
from pathlib import Path

import rich_click as click
from loguru import logger

# %%


@click.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=Path))
@click.argument("codebook_path", type=click.Path(exists=True, dir_okay=False, file_okay=True, path_type=Path))
@click.option("--rounds", type=int, default=10)
@click.option("--threads", type=int, default=10)
@click.option("--batch-size", type=int, default=50)
@click.option("--max-proj", is_flag=True)
def optimize(
    path: Path,
    codebook_path: Path,
    rounds: int = 10,
    threads: int = 10,
    max_proj: bool = False,
    batch_size: int = 50,
):
    if not len(list(path.glob("registered--*"))):
        raise ValueError(
            "No registered images found. Verify that you're in the base working directory, not the registered folder."
        )

    try:
        existing = len(Path(path / f"opt_{codebook_path.stem}" / "global_scale.txt").read_text().splitlines())
        logger.info(f"Found {existing} existing rounds.")
    except FileNotFoundError:
        logger.info("No existing global scale found. Starting from scratch.")
        existing = 0

    if (path / f"opt_{codebook_path.stem}" / "percentiles.json").exists():
        existing_perc = json.loads(
            Path(path / f"opt_{codebook_path.stem}" / "percentiles.json").read_text()
        ).__len__()
        logger.info(f"Found {existing_perc} existing percentiles.")
    else:
        existing_perc = 0

    if existing_perc == existing == rounds:
        logger.info("All rounds already exist. Exiting.")
        return

    if min(existing, existing_perc) < rounds:
        logger.info(f"Starting from round {min(existing, existing_perc)}")

    if existing < existing_perc:
        raise Exception("Percentiles count < existing rounds. Should not happen.")

    for i in range(min(existing, existing_perc), rounds):
        logger.critical(f"Starting round {i}")
        # Increases global scale count
        subprocess.run(
            [
                "preprocess",
                "spots",
                "step-optimize",
                str(path),
                "--codebook",
                codebook_path,
                f"--batch-size={batch_size}",
                "--subsample-z=1",
                f"--round={i}",
                f"--threads={threads}",
                "--overwrite",
                "--split=0",
                *(["--max-proj=1"] if max_proj else []),
            ],
            check=True,
            capture_output=False,
        )

        subprocess.run(
            [
                "preprocess",
                "spots",
                "combine",
                str(path),
                "--codebook",
                codebook_path,
                "--batch-size=50",
                f"--round={i}",
            ],
            check=True,
            capture_output=False,
        )

        # Increases percentiles count
        subprocess.run(
            [
                "preprocess",
                "spots",
                "find-threshold",
                str(path),
                "--codebook",
                codebook_path,
                f"--round={i}",
            ],
            check=True,
            capture_output=False,
        )


if __name__ == "__main__":
    run()
