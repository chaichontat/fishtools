# %%
import json
import subprocess
import time
from pathlib import Path

import rich_click as click
from loguru import logger

from fishtools.io.workspace import Workspace


def _field_store_path(ws: Workspace, roi: str, codebook_label: str) -> Path:
    slug = Workspace.sanitize_codebook_name(codebook_label)
    return ws.path / "analysis" / "deconv" / f"fields+{slug}" / f"field--{roi}+{slug}.zarr"


def _ensure_field_stores(ws: Workspace, rois: list[str], codebook_label: str) -> None:
    missing: list[str] = []
    for roi in rois:
        if not _field_store_path(ws, roi, codebook_label).is_dir():
            missing.append(roi)
    if missing:
        joined = ", ".join(sorted(set(missing)))
        raise click.ClickException(
            f"Missing illumination field store(s) for ROI(s): {joined}. Run 'preprocess correct-illum export-field' first."
        )


def _format_command(args: list[str | Path]) -> str:
    return " ".join(str(a) for a in args)


def _run_with_logging(
    args: list[str | Path],
    *,
    round_index: int,
    label: str,
) -> None:
    formatted = _format_command(args)
    logger.info("[round {}] starting {}: {}", round_index, label, formatted)
    start = time.perf_counter()
    try:
        subprocess.run(args, check=True, capture_output=False)
    except subprocess.CalledProcessError as exc:
        duration = time.perf_counter() - start
        logger.error(
            "[round {}] {} failed after {:.2f}s (returncode={}, cmd={})",
            round_index,
            label,
            duration,
            exc.returncode,
            formatted,
        )
        raise
    except Exception:
        duration = time.perf_counter() - start
        logger.exception(
            "[round {}] {} crashed after {:.2f}s (cmd={})", round_index, label, duration, formatted
        )
        raise
    else:
        duration = time.perf_counter() - start
        logger.info("[round {}] completed {} in {:.2f}s", round_index, label, duration)


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
@click.option("--blank", type=str, default=None, help="Blank image to subtract (config preferred)")
@click.option("--threshold", type=float, default=0.008, help="CV before stopping")
@click.option(
    "--config",
    "json_config",
    type=click.Path(exists=True, dir_okay=False, file_okay=True, path_type=Path),
    default=None,
    help="Optional project config (JSON) forwarded to subcommands.",
)
@click.option(
    "--field-correct/--no-field-correct",
    default=False,
    help="Apply illumination field correction using discovered TCYX field stores for all subcommands.",
)
def optimize(
    path: Path,
    roi: str,
    codebook: Path,
    rounds: int = 10,
    threads: int = 10,
    batch_size: int = 50,
    blank: str | None = None,
    threshold: float = 0.008,
    json_config: Path | None = None,
    field_correct: bool = False,
):
    if not len(list(path.glob(f"registered--{roi}{'*' if roi != '*' else ''}"))):
        raise ValueError(
            f"No registered images found under registered--{roi}. Verify that you're in the base working directory, not the registered folder."
        )

    logger.info(
        "Batch optimize start: workspace='{}' roi='{}' codebook='{}' rounds={} threads={} batch_size={} field_correct={}",
        path,
        roi,
        codebook,
        rounds,
        threads,
        batch_size,
        field_correct,
    )
    if field_correct:
        workspace = Workspace(path)
        rois_needed = workspace.rois if roi == "*" else [roi]
        logger.info("Checking illumination field stores for ROI(s): {}", ", ".join(sorted(rois_needed)))
        _ensure_field_stores(workspace, rois_needed, codebook.stem)
        logger.info("All required illumination field stores present.")

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
        step_cmd: list[str | Path] = [
            "preprocess",
            "spots",
            "step-optimize",
            str(path),
            roi,
            "--codebook",
            codebook,
            f"--batch-size={batch_size}",
            f"--round={i}",
            f"--threads={threads}",
            "--overwrite",
            "--split=0",
            *(["--blank", blank] if blank else []),
            *(["--config", json_config] if json_config else []),
            *(["--field-correct"] if field_correct else []),
        ]
        _run_with_logging(step_cmd, round_index=i, label="step-optimize")

        combine_cmd: list[str | Path] = [
            "preprocess",
            "spots",
            "combine",
            str(path),
            roi,
            "--codebook",
            codebook,
            f"--round={i}",
        ]
        _run_with_logging(combine_cmd, round_index=i, label="combine")

        find_cmd: list[str | Path] = [
            "preprocess",
            "spots",
            "find-threshold",
            str(path),
            roi,
            "--codebook",
            codebook,
            f"--round={i}",
            *(["--blank", blank] if blank else []),
            *(["--config", json_config] if json_config else []),
            *(["--field-correct"] if field_correct else []),
        ]
        _run_with_logging(find_cmd, round_index=i, label="find-threshold")


# if __name__ == "__main__":
#     run()
