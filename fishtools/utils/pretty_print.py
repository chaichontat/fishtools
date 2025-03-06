import json
import signal
import threading
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from types import FrameType
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
_R = TypeVar("_R")
T = TypeVar("T")


class TaskCancelledException(Exception): ...


@contextmanager
def progress_bar_threadpool(n: int, *, threads: int):
    """Creates a thread pool with a progress bar that handles graceful shutdown.

    Args:
        n: Total number of tasks to track in the progress bar
        threads: Number of worker threads in the pool

    Returns:
        submit: Function to submit tasks to the thread pool. Has signature:
            submit(func: Callable, *args, **kwargs) -> Future
            - func: The function to execute
            - args/kwargs: Arguments to pass to the function
            - returns a Future object representing the pending task

    Example:
        # Basic usage
        with progress_bar_threadpool(len(items), threads=4) as submit:
            for item in items:
                submit(process_item, item)

    Notes:
        - Progress bar advances automatically as tasks complete
        - If any task raises an exception, all pending tasks are canceled
        - SIGINT (Ctrl+C) cancels all pending tasks gracefully
        - TaskCancelledException is raised for canceled tasks
    """
    futs: list[Future] = []
    should_terminate = threading.Event()

    with progress_bar(n) as callback, ThreadPoolExecutor(threads) as exc:

        def signal_handler(signum: int, frame: FrameType | None) -> None:
            logger.info("\nShutting down...")
            should_terminate.set()
            for fut in futs:
                fut.cancel()
            exc.shutdown(wait=False, cancel_futures=True)
            raise KeyboardInterrupt

        prior_handler = signal.signal(signal.SIGINT, signal_handler)

        def submit(f: Callable[P, _R], *args: P.args, **kwargs: P.kwargs) -> Future[_R]:
            def wrapped_f(*args: P.args, **kwargs: P.kwargs) -> _R:
                if should_terminate.is_set():
                    raise TaskCancelledException
                return f(*args, **kwargs)

            futs.append(exc.submit(wrapped_f, *args, **kwargs))
            return futs[-1]

        try:
            yield submit

            for f in as_completed(futs):
                if should_terminate.is_set():
                    break
                callback()
                try:
                    f.result()
                except Exception as e:
                    should_terminate.set()
                    for fut in futs:
                        fut.cancel()
                    if isinstance(e, TaskCancelledException):
                        continue
                    raise e
        finally:
            signal.signal(signal.SIGINT, prior_handler)


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
