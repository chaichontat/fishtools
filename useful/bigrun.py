# %%
import os
import pty
import select
import subprocess
import termios
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from shlex import split
from typing import Any

import rich_click as click
from prefect import flow, task
from prefect.logging import get_run_logger

from fishtools.utils.io import Workspace


@contextmanager
def setwd(path: Path | str):
    old = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


@task(name="cli", retries=1, tags=["script"], task_run_name="{args}")
def execute_script(args: str, *, cwd: Path | None = None, overwrite: bool = False, **kwargs: Any) -> str:
    """Execute a shell command with real-time colored output using a pseudo-terminal.

    Runs a command and displays its output in realtime, preserving
    ANSI color codes and terminal formatting from both stdout and stderr.

    Args:
        args: Command string to execute (will be split into arguments)
        cwd: Optional working directory for the command
        **kwargs: Additional keyword arguments passed to subprocess.Popen

    Returns:
        str: Combined output from the command

    Raises:
        subprocess.CalledProcessError: If the command returns non-zero exit code

    Example:
        >>> execute_script("ls --color -la")
        >>> execute_script("make test", cwd=Path("./project"))
    """
    logger = get_run_logger()
    logger.info(f"Running {args}")

    output = []

    master, slave = pty.openpty()
    modes = termios.tcgetattr(slave)
    modes[3] &= ~(termios.ECHO | termios.ICANON)  # lflags
    termios.tcsetattr(slave, termios.TCSANOW, modes)

    process = subprocess.Popen(
        split(args + (" --overwrite" if overwrite else "")),
        stdout=slave,
        stderr=slave,
        stdin=slave,
        cwd=cwd,
        close_fds=True,
        **kwargs,
    )

    os.close(slave)

    while True:
        try:
            r, _, _ = select.select([master], [], [])
            if r:
                data = os.read(master, 1024).decode("utf-8")
                if data:
                    print(data, end="", flush=True)
                    output.append(data)
        except OSError:
            break

    return_code = process.wait()
    os.close(master)

    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, args)

    return "".join(output)


@task
def check(conditions: list[Callable[..., bool]]) -> bool:
    return all(condition() for condition in conditions)


@flow(name="Deconv")
def deconv():
    execute_script("preprocess deconv batch . --basic-name=all")
    with setwd("analysis/deconv"):
        execute_script("preprocess deconv compute-range . --overwrite")


@flow(name="Register")
def register(ws: Workspace, codebook: Path, threads: int):
    execute_script("preprocess deconv compute-range . --overwrite")
    execute_script(f"preprocess register batch . --codebook={codebook} --threads={threads}")


@flow
def optimize(ws: Workspace, codebook: Path, threads: int, *, rounds: int = 6):
    execute_script(f"preprocess spots optimize . --codebook={codebook} --rounds={rounds} --threads={threads}")


@flow
def call_spots(ws: Workspace, codebook: Path, threads: int):
    execute_script(f"preprocess spots batch . --codebook={codebook} --threads={threads} --split")
    execute_script(f"preprocess spots stitch . --codebook={codebook} --threads={threads}")


@flow(name="call-spots")
def main_workflow(path: Path, codebook: Path, threads: int):
    ws = Workspace(path)
    # deconv(ws)
    with setwd(ws.deconved):
        register(ws, codebook, threads)
        optimize(ws, codebook, threads)
        call_spots(ws, codebook, threads)


@click.group()
def cli(): ...


@cli.command()
@click.argument("path", type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path))
@click.option(
    "--codebook",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to the codebook file",
)
@click.option("--threads", type=int, default=15, help="Number of threads to use")
def spots(path: Path, codebook: Path, threads: int):
    main_workflow(path, codebook, threads)


@flow
def stitch_register(ws: Workspace, rois: list[str], threads: int, overwrite: bool = False):
    tileconfigs = [ws.stitch(roi) / "TileConfiguration.registered.txt" for roi in rois]
    if not all(Path(p).exists() for p in tileconfigs):
        execute_script('preprocess stitch register . "*" --idx=0 --max-proj', overwrite=overwrite)


@flow
def stitch_fuse(ws: Workspace, rois: list[str], codebook: str, threads: int, overwrite: bool = False):
    logger = get_run_logger()
    for roi in rois:
        path = ws.stitch(roi, codebook)
        done = (
            path.exists()
            and (path / "fused.zarr").exists()
            and not any(p.name.isdigit() for p in path.iterdir())  # Intermediate fuses
        )

        if done:
            logger.info(f"{roi} already fused.")

        if overwrite or not done:
            execute_script(
                f"preprocess stitch fuse . {roi} --codebook={codebook}",
                overwrite=overwrite,
            )
            execute_script(f"preprocess stitch combine . {roi} --codebook={codebook}", overwrite=overwrite)

        if not (path / "fused.zarr").exists():
            raise ValueError(f"No fused image found at {path / 'fused.zarr'}")


@cli.command()
@click.argument("path", type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path))
@click.option("--codebook", type=str, help="Codebook name")
@click.option("--threads", type=int, default=15, help="Number of threads to use")
@click.option("--overwrite", is_flag=True, help="Overwrite existing files")
def stitch(path: Path, codebook: str, threads: int, overwrite: bool):
    ws = Workspace(path)
    with setwd(ws.deconved):
        stitch_register(ws, ws.rois, threads, overwrite=overwrite)
        stitch_fuse(ws, ws.rois, codebook, threads, overwrite=overwrite)


if __name__ == "__main__":
    cli()
