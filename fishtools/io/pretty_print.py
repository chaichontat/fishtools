import json
from typing import Any, Callable, Concatenate, ParamSpec, Protocol, TypeVar

from loguru import logger
from rich.console import Console
from rich.syntax import Syntax

console = Console()

P = ParamSpec("P")
R, T = TypeVar("R", covariant=True), TypeVar("T")


# Massive hack to get rid of the first two arguments from the type signature.
class _JPrint(Protocol[P]):
    def __call__(self, code: str, lexer: str, *args: P.args, **kwargs: P.kwargs) -> Any:
        ...


def _jprint(f: _JPrint[P]) -> Callable[Concatenate[Any, P], None]:
    def inner(d: Any, *args: P.args, **kwargs: P.kwargs):
        return logger.info(f(json.dumps(d, indent=2), "json", *args, **kwargs))

    return inner


jprint = _jprint(Syntax)
