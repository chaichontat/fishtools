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
                try:
                    return next(track)
                except StopIteration:
                    return n

        callback()
        yield callback

        try:
            from collections import deque

            deque(track, maxlen=0)  # Exhaust the iterator
        except (StopIteration, ValueError):
            ...


P = ParamSpec("P")
_R = TypeVar("_R")
T = TypeVar("T")


class TaskCancelledException(Exception): ...


@contextmanager
def progress_bar_threadpool(n: int, *, threads: int, stop_on_exception: bool = False):
    """Creates a thread pool with a progress bar that handles graceful shutdown.

    Args:
        n: Total number of tasks to track in the progress bar
        threads: Number of worker threads in the pool
        stop_on_exception: If True (default), stops all processing immediately
            when any task raises an exception. If False, continues processing
            other tasks and logs exceptions.

    Returns:
        submit: Function to submit tasks to the thread pool. Has signature:
            submit(func: Callable, *args, **kwargs) -> Future
            - func: The function to execute
            - args/kwargs: Arguments to pass to the function
            - returns a Future object representing the pending task

    Example:
        # Basic usage (stops on first exception)
        with progress_bar_threadpool(len(items), threads=4) as submit:
            for item in items:
                submit(process_item, item)

        # Continue processing even if some tasks fail
        with progress_bar_threadpool(len(items), threads=4, stop_on_exception=False) as submit:
            for item in items:
                submit(process_item, item)

    Notes:
        - Progress bar advances automatically as tasks complete
        - If `stop_on_exception` is True and a task raises an exception, all pending tasks are canceled
        - SIGINT (Ctrl+C) cancels all pending tasks gracefully regardless of `stop_on_exception`
        - TaskCancelledException is raised for canceled tasks
    """
    futs: list[Future] = []
    should_terminate = threading.Event()
    exceptions_encountered: list[Exception] = []

    with progress_bar(n) as callback, ThreadPoolExecutor(threads) as exc:

        def signal_handler(signum: int, frame: FrameType | None) -> None:
            logger.info("\nShutting down...")
            should_terminate.set()
            for fut in futs:
                fut.cancel()
            exc.shutdown(wait=False, cancel_futures=True)

        prior_handler = signal.signal(signal.SIGINT, signal_handler)

        def submit(
            f: Callable[P, _R], *args: P.args, **kwargs: P.kwargs
        ) -> Future[_R] | Future[TaskCancelledException]:
            def wrapped_f(*args: P.args, **kwargs: P.kwargs) -> _R:
                if should_terminate.is_set():
                    raise TaskCancelledException("Task cancelled due to shutdown signal or prior exception.")
                return f(*args, **kwargs)

            # Don't submit if already terminating
            if should_terminate.is_set():
                # Create a cancelled future to represent the skipped task
                fut = Future()
                fut.set_exception(TaskCancelledException("Shutdown initiated before task submission."))
                futs.append(fut)
                return fut
            else:
                fut = exc.submit(wrapped_f, *args, **kwargs)

            fut._args = args  # type: ignore
            fut._kwargs = kwargs  # type: ignore
            futs.append(fut)
            return fut

        try:
            yield submit

            for f in as_completed(futs):
                if should_terminate.is_set():
                    # If termination was signaled, try to cancel unprocessed futures
                    # Note: as_completed yields futures as they complete,
                    # so 'f' is already done. We cancel others in futs.
                    # The signal handler already does this, but adding redundancy here
                    # might catch edge cases depending on timing.
                    for fut_to_cancel in futs:
                        if not fut_to_cancel.done():
                            fut_to_cancel.cancel()
                    break
                try:
                    f.result()
                    # Only advance progress bar on successful completion or
                    # if we are not stopping on exceptions
                    if not f.exception() or not stop_on_exception:
                        callback()
                except Exception as e:
                    if isinstance(e, TaskCancelledException):
                        # Ignore TaskCancelledException, it's expected during shutdown
                        # or if stop_on_exception is True
                        continue  # Don't advance progress bar for cancelled tasks

                    # Log the exception regardless
                    task_args = getattr(f, "_args", "N/A")
                    task_kwargs = getattr(f, "_kwargs", "N/A")
                    logger.error(f"Task raised an exception: {e} (args={task_args}, kwargs={task_kwargs})")
                    exceptions_encountered.append(e)

                    if stop_on_exception:
                        should_terminate.set()  # Signal termination
                        # Cancel all other *pending* futures
                        for fut_to_cancel in futs:
                            # Check if it's not the current future and if it's not done/running
                            if fut_to_cancel is not f and not fut_to_cancel.done():
                                fut_to_cancel.cancel()
                        # Don't raise immediately, let the loop break naturally
                        # on the next iteration due to should_terminate.is_set()
                        # This allows the finally block to execute cleanly.
                        # Alternatively, could raise here, but ensure finally runs.
                        # For simplicity, let the loop check handle the break.
                    else:
                        # If not stopping, just advance the progress bar as if it completed
                        # (since we've handled the exception by logging it)
                        callback()

            # After loop finishes, if stop_on_exception was True and we had errors,
            # raise the first one encountered to signal failure.
            if stop_on_exception and exceptions_encountered:
                raise exceptions_encountered[0]
            elif not stop_on_exception and exceptions_encountered:
                logger.warning(f"Finished processing with {len(exceptions_encountered)} errors.")

        finally:
            # Ensure original signal handler is restored
            signal.signal(signal.SIGINT, prior_handler)
            # Ensure executor is shut down cleanly, cancelling any remaining futures
            # if termination was signaled. The context manager (`with ThreadPoolExecutor...`)
            # also calls shutdown, but calling it explicitly ensures cancellation
            # logic is triggered if `should_terminate` was set.
            if should_terminate.is_set():
                exc.shutdown(wait=True, cancel_futures=True)  # Wait for cancellations
            else:
                exc.shutdown(wait=True)  # Wait for normal completion


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
