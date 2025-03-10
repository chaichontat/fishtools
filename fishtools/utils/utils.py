import logging
import subprocess
import sys
from functools import cache, wraps
from inspect import getcallargs
from pathlib import Path
from subprocess import PIPE, Popen
from typing import Any, Callable, Concatenate, ParamSpec, Sequence, TypeVar, cast

import loguru
from loguru import logger

P = ParamSpec("P")
R = TypeVar("R", covariant=True)
TType = TypeVar("TType", bound=type)
TAny = TypeVar("TAny")


def copy_signature(
    kwargs_call: Callable[P, Any],
) -> Callable[[Callable[..., R]], Callable[P, R]]:
    """Decorator does nothing but returning the casted original function"""

    def return_func(func: Callable[..., R]) -> Callable[P, R]:
        return cast(Callable[P, R], func)

    return return_func


def copy_signature_method(
    kwargs_call: Callable[P, Any], cls: TType
) -> Callable[[Callable[..., R]], Callable[Concatenate[TType, P], R]]:
    """Decorator does nothing but returning the casted original function"""

    def return_func(func: Callable[..., R]) -> Callable[Concatenate[TType, P], R]:
        return cast(Callable[Concatenate[TType, P], R], func)

    return return_func


def setup_logging():
    logger.remove()
    logger.add(
        sys.stderr,
        colorize=True,
        format="<green>{time:HH:mm:ss}</green> | "
        "<level>{level: >8}</level> | "
        "<cyan>{name}</cyan> | <level>{message}</level>",
    )

    class InterceptHandler(logging.Handler):
        def emit(self, record: logging.LogRecord):
            # Get corresponding Loguru level if it exists.
            try:
                level = logger.level(record.levelname).name
            except ValueError:
                level = record.levelno

            # Find caller from where originated the logged message.
            frame, depth = sys._getframe(6), 6
            while frame and frame.f_code.co_filename == logging.__file__:
                frame = frame.f_back
                depth += 1

            logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())

    logging.basicConfig(handlers=[InterceptHandler()], level=logging.DEBUG, force=True)
    logging.getLogger("biothings").setLevel(logging.CRITICAL)
    return logger


def check_if_posix(f: Callable[P, R]) -> Callable[P, R]:
    @wraps(f)
    def inner(*args: P.args, **kwargs: P.kwargs) -> R:
        if not Path("/").exists():
            raise OSError("Not a POSIX system")
        return f(*args, **kwargs)

    return inner


def run_process(cmd: Sequence[str], input: bytes) -> bytes:
    process = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate(input=input)

    if process.returncode != 0:
        logging.critical(stderr.decode())
        raise RuntimeError(f"{cmd} failed")

    return stdout


@cache
def slide(x: str, n: int = 20) -> list[str]:
    return [x[i : i + n] for i in range(len(x) - n + 1)]


def check_if_exists(
    logger: "loguru.Logger",
    name_detector: Callable[[dict[str, Any]], Path | str] = lambda kwargs: str(next(iter(kwargs.values()))),
):
    """
    A decorator that checks if a file exists before executing a function.
    Can override with the `overwrite` keyword argument.

    Parameters:
    -----------
    logger : loguru.Logger
        A logger object that is used to log warning messages if the file already exists.

    name_detector : Callable[[list[Any], dict[str, Any]], str], optional
        A callable that takes two arguments: a list of positional arguments and a dictionary of keyword arguments.
        The kwargs include all the arguments passed to the decorated function (including args).
        Should return the name of the file to check.

    Returns:
    --------
    Callable[P, R | None]]
        A decorator that may return None if the file already exists.
    """

    def decorator(f: Callable[P, R]) -> Callable[P, R | None]:
        @wraps(f)
        def inner(*args: P.args, **kwargs: P.kwargs) -> R | None:
            if not (path := Path(name_detector(getcallargs(f, *args, **kwargs)))).exists() or kwargs.get(
                "overwrite", False
            ):
                return f(*args, **kwargs)
            logger.warning(f"{path} already exists. Skipping.")

        return inner

    return decorator


def batchable():
    def decorator(f: Callable[P, R]) -> Callable[P, R]:
        @wraps(f)
        def inner(*args: P.args, **kwargs: P.kwargs) -> R:
            return f(*args, **kwargs)

        return inner

    return decorator


def git_hash() -> str:
    return (
        subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).parent.parent)
        .decode("ascii")
        .strip()
    )


_T = TypeVar("_T")


def batch_roi(look_for: str = "registered--*"):
    def decorator(func: Callable[P, _T]) -> Callable[P, _T | None]:
        @wraps(func)
        def inner(*args: P.args, **kwargs: P.kwargs) -> _T | None:
            if kwargs["roi"] == "*":
                for p in Path(kwargs["path"]).glob(look_for):  # type: ignore
                    print(kwargs | dict(roi=p.name.split("--")[1]))
                    kwargs = kwargs | dict(roi=p.name.split("--")[1])  # type: ignore
                    func(*args, **kwargs)
                return
            return func(*args, **kwargs)

        return inner

    return decorator
