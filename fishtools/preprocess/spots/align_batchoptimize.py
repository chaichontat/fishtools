# %%
import json
import subprocess
from pathlib import Path

import rich_click as click
from loguru import logger

# %%


@click.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=Path))
@click.argument("roi", type=str, default="*")
@click.option(
    "--codebook",
    type=click.Path(exists=True, dir_okay=False, file_okay=True, path_type=Path),
)
@click.option("--rounds", type=int, default=10)
@click.option("--threads", type=int, default=10)
@click.option("--batch-size", type=int, default=50)
@click.option("--max-proj", is_flag=True)
@click.option("--blank", type=str, default=None, help="Blank image to subtract")
@click.option("--threshold", type=float, default=0.008, help="CV before stopping")
def optimize(
    path: Path,
    roi: str,
    codebook: Path,
    rounds: int = 10,
    threads: int = 10,
    max_proj: bool = False,
    batch_size: int = 50,
    blank: str | None = None,
    threshold: float = 0.008,
):
    if not len(list(path.glob(f"registered--{roi}{'*' if roi != '*' else ''}"))):
        raise ValueError(
            f"No registered images found under registered--{roi}. Verify that you're in the base working directory, not the registered folder."
        )

    wd = path / f"opt_{codebook.stem}{f'+{roi}' if roi != '*' else ''}"
    try:
        existing = len((wd / "global_scale.txt").read_text().splitlines())
        logger.info(f"Found {existing} existing rounds.")
    except FileNotFoundError:
        logger.info("No existing global scale found. Starting from scratch.")
        existing = 0

    if (wd / "percentiles.json").exists():
        existing_perc = json.loads((wd / "percentiles.json").read_text()).__len__()
        logger.info(f"Found {existing_perc} existing percentiles.")
    else:
        existing_perc = 0

    if existing_perc == existing == rounds:
        logger.info("All rounds already exist. Exiting.")
        return

    if min(existing, existing_perc) < rounds:
        logger.info(f"Starting from round {min(existing, existing_perc)}")

    if existing < existing_perc:
        raise Exception("Existing rounds < Percentiles rounds. Should not happen.")

    for i in range(min(existing, existing_perc), rounds):
        try:
            if i < 2:
                raise IndexError  # Skip CV check for first two rounds
            curr_cv = (wd / "mse.txt").read_text().splitlines()[i - 2]
        except (FileNotFoundError, IndexError):
            logger.warning(f"CV for round {i} not found. Assuming it is above threshold.")
        else:
            if float(curr_cv.split("\t")[1]) < threshold:
                logger.info(f"CV at round {i} already below threshold {threshold}. Converged. Stopping.")
                return

        logger.critical(f"Starting round {i}")
        # Increases global scale count
        subprocess.run(
            [
                "preprocess",
                "spots",
                "step-optimize",
                str(path),
                roi,
                "--codebook",
                codebook,
                f"--batch-size={batch_size}",
                "--subsample-z=1",
                f"--round={i}",
                f"--threads={threads}",
                "--overwrite",
                "--split=0",
                *(["--max-proj=1"] if max_proj else []),
                *(["--blank", blank] if blank else []),
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
                roi,
                "--codebook",
                codebook,
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
                roi,
                "--codebook",
                codebook,
                f"--round={i}",
                *(["--blank", blank] if blank else []),
            ],
            check=True,
            capture_output=False,
        )


# if __name__ == "__main__":
#     run()
