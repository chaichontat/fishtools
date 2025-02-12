# %%
import fcntl
import os
import pty
import select
import subprocess
import termios
from collections.abc import Callable
from pathlib import Path
from shlex import split
from typing import Any

import rich_click as click
from prefect import flow, task
from prefect.logging import get_run_logger


@task(retries=0, tags=["script"])
def execute_script(args: str, *, cwd: Path | None = None, **kwargs: Any) -> str:
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
        split(args), stdout=slave, stderr=slave, stdin=slave, cwd=cwd, close_fds=True, **kwargs
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


@flow(name="Master Workflow")
def master_workflow():
    execute_script(
        "python '/home/chaichontat/fishtools/useful/stitch_spot_prod.py' run . --codebook /home/chaichontat/fishtools/starwork3/ordered/tricycleplus.json"
    )


@click.command()
@click.argument("path", type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path))
def main(path: Path):
    master_workflow()


if __name__ == "__main__":
    main()
