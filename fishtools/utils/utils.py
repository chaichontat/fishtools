import logging
import sys
from functools import cache, wraps
from pathlib import Path
from subprocess import PIPE, Popen
from typing import Any, Callable, Concatenate, ParamSpec, Sequence, TypeVar, cast

from loguru import logger

P = ParamSpec("P")
R = TypeVar("R", covariant=True)
TType = TypeVar("TType", bound=type)
TAny = TypeVar("TAny")


def copy_signature(kwargs_call: Callable[P, Any]) -> Callable[[Callable[..., R]], Callable[P, R]]:
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

    logging.basicConfig(handlers=[InterceptHandler()], level=logging.INFO, force=True)
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
