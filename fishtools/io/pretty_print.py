import json
import logging
import sys
from typing import Any, Callable, Concatenate, ParamSpec, Protocol, TypeVar

from loguru import logger
from rich.console import Console
from rich.syntax import Syntax

console = Console()

P = ParamSpec("P")
R, T = TypeVar("R", covariant=True), TypeVar("T")


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


# Massive hack to get rid of the first two arguments from the type signature.
class _JPrint(Protocol[P]):
    def __call__(self, code: str, lexer: str, *args: P.args, **kwargs: P.kwargs) -> Any:
        ...


def _jprint(f: _JPrint[P]) -> Callable[Concatenate[Any, P], None]:
    def inner(d: Any, *args: P.args, **kwargs: P.kwargs):
        return log.info(f(json.dumps(d, indent=2), "json", *args, **kwargs))

    return inner


jprint = _jprint(Syntax)
