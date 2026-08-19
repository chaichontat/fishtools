from collections.abc import Callable
from typing import Any, Concatenate, ParamSpec, TypeVar, cast

P = ParamSpec("P")
R = TypeVar("R", covariant=True)
TType = TypeVar("TType", bound=type)


def copy_signature(
    kwargs_call: Callable[P, Any],
) -> Callable[[Callable[..., R]], Callable[P, R]]:
    """Give a wrapper the static signature of the callable it forwards to."""

    def return_func(function: Callable[..., R]) -> Callable[P, R]:
        return cast(Callable[P, R], function)

    return return_func


def copy_signature_method(
    kwargs_call: Callable[P, Any], cls: TType
) -> Callable[[Callable[..., R]], Callable[Concatenate[TType, P], R]]:
    """Give a method wrapper the static signature of the callable it forwards to."""

    def return_func(function: Callable[..., R]) -> Callable[Concatenate[TType, P], R]:
        return cast(Callable[Concatenate[TType, P], R], function)

    return return_func

