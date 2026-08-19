from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from functools import wraps
from inspect import getcallargs
from pathlib import Path
from subprocess import PIPE, Popen
from typing import Any, ParamSpec, TypeVar

import loguru

P = ParamSpec("P")
R = TypeVar("R")


def check_if_posix(function: Callable[P, R]) -> Callable[P, R]:
    @wraps(function)
    def inner(*args: P.args, **kwargs: P.kwargs) -> R:
        if not Path("/").exists():
            raise OSError("Not a POSIX system")
        return function(*args, **kwargs)

    return inner


def run_process(command: Sequence[str], input: bytes) -> bytes:
    process = Popen(command, stdin=PIPE, stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate(input=input)
    if process.returncode != 0:
        logging.critical(stderr.decode())
        raise RuntimeError(f"{command} failed")
    return stdout


def check_if_exists(
    logger: loguru.Logger,
    name_detector: Callable[[dict[str, Any]], Path | str] = lambda kwargs: str(
        next(iter(kwargs.values()))
    ),
):
    """Skip a completed file-producing operation unless overwrite is requested."""

    def decorator(function: Callable[P, R]) -> Callable[P, R | None]:
        @wraps(function)
        def inner(*args: P.args, **kwargs: P.kwargs) -> R | None:
            path = Path(name_detector(getcallargs(function, *args, **kwargs)))
            if not path.exists() or kwargs.get("overwrite", False):
                return function(*args, **kwargs)
            logger.warning(f"{path} already exists. Skipping.")
            return None

        return inner

    return decorator
