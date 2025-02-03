import json
import signal
import threading
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from typing import Any, Callable, Concatenate, Generator, ParamSpec, Protocol, TypeVar

import colorama
from loguru import logger
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.syntax import Syntax


@contextmanager
def progress_bar(n: int) -> Generator[Callable[..., int], None, None]:
    """Progress bar.

    Args:
        n: Number of iterations.

    Yields:
        Callback function to be called to update the progress bar.
    """
    logger.debug("Starting...")
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.1f}%"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
    ) as p:
        lock = threading.RLock()
        track = iter(p.track(range(n)))

        def callback(*args: Any, **kwargs: Any):
            with lock:
                return next(track)

        yield callback

        try:
            from collections import deque

            deque(track, maxlen=0)  # Exhaust the iterator
        except StopIteration:
            ...


P = ParamSpec("P")
R, T = TypeVar("R", covariant=True), TypeVar("T")


@contextmanager
def progress_bar_threadpool(n: int, *, threads: int):
    futs: list[Future] = []
    with progress_bar(n) as callback, ThreadPoolExecutor(threads) as exc:

        def signal_handler(signum, frame):
            logger.info("\nShutting down...")
            exc.shutdown(wait=False, cancel_futures=True)
            exit(0)

        signal.signal(signal.SIGINT, signal_handler)

        def submit(f: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> Future[R]:
            futs.append(exc.submit(f, *args, **kwargs))
            return futs[-1]

        yield submit

        for f in as_completed(futs):
            f.result()
            callback()


console = Console()


# Massive hack to get rid of the first two arguments from the type signature.
class _JPrint(Protocol[P]):
    def __call__(self, code: str, lexer: str, *args: P.args, **kwargs: P.kwargs) -> Any: ...


def _jprint(f: _JPrint[P]) -> Callable[Concatenate[Any, P], None]:
    def inner(d: Any, *args: P.args, **kwargs: P.kwargs):
        return console.print(f(json.dumps(d, indent=2), "json", *args, **kwargs))

    return inner


jprint = _jprint(Syntax)


def printc(seq: str):
    for c in seq:
        if c == "A" or c == "a":
            print(colorama.Fore.GREEN + c, end="")
        elif c == "T" or c == "t":
            print(colorama.Fore.RED + c, end="")
        elif c == "C" or c == "c":
            print(colorama.Fore.BLUE + c, end="")
        elif c == "G" or c == "g":
            print(colorama.Fore.YELLOW + c, end="")
        else:
            print(colorama.Fore.WHITE + c, end="")
    print(colorama.Fore.RESET)
