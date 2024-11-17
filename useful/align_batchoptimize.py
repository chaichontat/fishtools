# %%
import subprocess
from pathlib import Path

import rich_click as click
from loguru import logger


@click.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=Path))
@click.argument("codebook_path", type=click.Path(exists=True, dir_okay=False, file_okay=True, path_type=Path))
@click.option("--rounds", type=int, default=10)
@click.option("--threads", type=int, default=6)
def run(path: Path, codebook_path: Path, rounds: int = 10, threads: int = 6):
    if not len(list(path.glob("registered--*"))):
        raise ValueError(
            "No registered images found. Verify that you're in the base working directory, not the registered folder."
        )

    try:
        existing = len(Path(path / f"opt_{codebook_path.stem}" / "global_scale.txt").read_text().splitlines())
        logger.info(f"Found {existing} existing rounds.")
        if existing == rounds:
            logger.info("All rounds already exist. Exiting.")
            return
    except FileNotFoundError:
        logger.info("No existing global scale found. Starting from scratch.")
        existing = 0

    for i in range(existing, rounds):
        logger.critical(f"Starting round {i}")
        subprocess.run(
            [
                "python",
                Path(__file__).parent / "align_prod.py",
                "optimize",
                str(path),
                "--codebook",
                codebook_path,
                "--batch-size=50",
                "--subsample-z=1",
                f"--round={i}",
                f"--threads={threads}",
                "--overwrite",
                "--split=0",
            ],
            check=True,
            capture_output=False,
        )
        subprocess.run(
            [
                "python",
                Path(__file__).parent / "align_prod.py",
                "combine",
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
