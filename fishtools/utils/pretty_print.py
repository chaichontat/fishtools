import io
import itertools
import json
import os
import signal
import subprocess
import threading
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from contextlib import contextmanager, nullcontext
from types import FrameType
from typing import IO, Any, Callable, Concatenate, Generator, ParamSpec, Protocol, TypeVar

import colorama
from loguru import logger
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    ProgressColumn,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.syntax import Syntax

from fishtools.utils.logging import CONSOLE_SKIP_EXTRA

# A single console shared by progress bars and any ad-hoc prints while a Live display is active.
# Using the same Console prevents duplicate/redrawn bars when printing logs.
_SHARED_CONSOLE = Console()


def get_shared_console() -> Console:
    return _SHARED_CONSOLE


class _StepStats:
    """Tracks per-step durations and rolling averages.

    Design notes:
    - We record the wall time between successive progress updates (i.e., calls
      to the bar's callback). This treats each update as one "step" regardless of
      work size, which matches typical usage of our `progress_bar` helper.
    - Rolling windows are computed over a bounded deque so memory stays constant.
    - The first update only primes the clock; no duration is recorded until the
      second update to avoid an artificial initial zero-length step.
    """

    def __init__(self, window: int = 10) -> None:
        self.window = max(1, int(window))
        self._durations: list[float] = []  # seconds, bounded to `window`
        self._last_t: float | None = None

    def prime(self) -> None:
        self._last_t = time.perf_counter()

    def step(self) -> None:
        now = time.perf_counter()
        if self._last_t is not None:
            dt = max(0.0, now - self._last_t)
            self._durations.append(dt)
            if len(self._durations) > self.window:
                # Keep only the most recent `window` durations
                del self._durations[: len(self._durations) - self.window]
        self._last_t = now

    @property
    def last(self) -> float | None:
        return self._durations[-1] if self._durations else None

    def avg(self, k: int) -> float | None:
        if not self._durations:
            return None
        if k <= 0:
            return None
        tail = self._durations[-k:]
        if not tail:
            return None
        return sum(tail) / len(tail)


def _fmt_seconds(seconds: float | None) -> str:
    """Format seconds with a compact, human-friendly unit.

    Examples: 0.0042 -> 4.2 ms, 0.12 -> 120 ms, 1.23 -> 1.23 s, 95 -> 1m35s
    """
    if seconds is None:
        return "—"
    s = max(0.0, float(seconds))
    if s < 0.001:
        return f"{s * 1_000_000:.1f}µs"
    if s < 1.0:
        return f"{s * 1_000:.0f} ms"
    if s < 60.0:
        return f"{s:.2f} s"
    m, sec = divmod(int(s), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h{m}m{sec}s"
    return f"{m}m{sec}s"


class StepStatsColumn(ProgressColumn):
    """Rich Progress column showing last and rolling-average step durations.

    Displays: "last X • avgN1 Y • avgN2 Z ..." with compact units, where N1/N2
    are provided windows (e.g., (5, 10)).

    Implementation detail: the column reads from a shared `_StepStats` instance
    that is updated by the progress callback. This keeps logic out of Render
    (which may be called frequently) and avoids relying on `task.fields`.
    """

    def __init__(self, stats: _StepStats, *, windows: tuple[int, ...] = (5, 10)) -> None:
        super().__init__()
        self._stats = stats
        # Sanitize windows: positive integers, unique order preserved
        seen: set[int] = set()
        cleaned: list[int] = []
        for w in windows:
            try:
                iw = int(w)
            except Exception:
                continue
            if iw <= 0 or iw in seen:
                continue
            seen.add(iw)
            cleaned.append(iw)
        self._windows: tuple[int, ...] = tuple(cleaned) if cleaned else (5, 10)

    def render(self, task):
        from rich.text import Text

        parts = [f"last {_fmt_seconds(self._stats.last)}"]
        for w in self._windows:
            parts.append(f"avg{w} {_fmt_seconds(self._stats.avg(w))}")
        return Text(" • ".join(parts), style="dim")


@contextmanager
def progress_bar(
    n: int,
    *,
    step_stats: tuple[int, ...] | None = (5, 10),
) -> Generator[Callable[..., int], None, None]:
    """Progress bar.

    Args:
        n: Number of iterations.
        step_stats: Tuple | None of window sizes for rolling averages (default: (5, 10)).

    Yields:
        Callback function to be called to update the progress bar.
    """
    logger.debug("Starting...")
    # Optional step stats tracking shared with the custom column. Keep enough
    # history to satisfy the largest requested window.
    max_window = max(step_stats) if step_stats else 10
    stats = _StepStats(window=max_window)

    columns: list[ProgressColumn] = [
        SpinnerColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.1f}%"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
    ]
    if step_stats:
        columns += [TextColumn("•"), StepStatsColumn(stats, windows=step_stats)]

    with Progress(*columns, console=_SHARED_CONSOLE) as p:
        lock = threading.RLock()
        track = iter(p.track(range(n)))
        primed = False

        def callback(*args: Any, **kwargs: Any):
            with lock:
                try:
                    nonlocal primed
                    if step_stats:
                        if not primed:
                            stats.prime()
                            primed = True
                        else:
                            stats.step()
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


class ProgressReporter:
    """Helper that coordinates progress advancement and console-safe output."""

    __slots__ = ("_advance", "_console")

    def __init__(self, advance: Callable[[], int], *, console: Console | None = None) -> None:
        self._advance = advance
        self._console = console or get_shared_console()

    def advance(self) -> int:
        """Advance the underlying progress task."""

        return self._advance()

    def print(self, message: str, *, style: str | None = None, end: str = "\n") -> None:
        """Render ``message`` above the live progress bar without forcing a redraw."""

        if style:
            self._console.print(f"[{style}]{message}[/{style}]", end=end)
        else:
            self._console.print(message, end=end)

    def log(self, *args: Any, **kwargs: Any) -> None:
        """Delegate to the shared console's ``log`` helper."""

        self._console.log(*args, **kwargs)

    @property
    def console(self) -> Console:
        return self._console


@contextmanager
def progress_reporter(
    n: int,
    *,
    step_stats: tuple[int, ...] | None = (5, 10),
) -> Generator[ProgressReporter, None, None]:
    """Context manager yielding a :class:`ProgressReporter` bound to ``n`` steps."""

    with progress_bar(n, step_stats=step_stats) as advance:
        yield ProgressReporter(advance)


def wrap_progress(advance: Callable[[], int]) -> ProgressReporter:
    """Wrap an existing progress callback into a :class:`ProgressReporter`."""

    return ProgressReporter(advance)


# Maintain stable worker identifiers for streaming subprocess output.
_STREAM_THREAD_IDS: dict[int, int] = {}
_STREAM_THREAD_COUNTER = itertools.count(1)
_STREAM_ID_LOCK = threading.RLock()


def _resolve_stream_thread_index(explicit: int | None) -> int:
    if explicit is not None:
        return explicit
    ident = threading.get_ident()
    with _STREAM_ID_LOCK:
        existing = _STREAM_THREAD_IDS.get(ident)
        if existing is not None:
            return existing
        next_id = next(_STREAM_THREAD_COUNTER)
        _STREAM_THREAD_IDS[ident] = next_id
        return next_id


Command = Sequence[str] | str


def run_subprocess_streaming(
    args: Command,
    *,
    thread_index: int | None = None,
    reporter: ProgressReporter | None = None,
    cwd: str | os.PathLike[str] | None = None,
    env: Mapping[str, str] | None = None,
    check: bool = True,
    shell: bool = False,
    encoding: str = "utf-8",
    errors: str = "replace",
    timeout: float | None = None,
    start_new_session: bool | None = None,
    creationflags: int = 0,
    preexec_fn: Callable[[], None] | None = None,
    stdin: IO[str] | int | None = None,
    emit_to_console: bool = True,
    line_handler: Callable[[str, str], None] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a subprocess, streaming stdout/stderr through the shared console.

    Each output line is prefixed with ``[T{thread_index}]`` and printed via the
    provided :class:`ProgressReporter` (or the shared console when omitted).

    Args:
        args: Command to execute (identical to :func:`subprocess.run`).
        thread_index: Identifier used in the output prefix (auto-assigned when omitted).
        reporter: Optional :class:`ProgressReporter` for console-safe printing.
        cwd: Working directory for the subprocess.
        env: Environment overrides for the subprocess.
        check: Raise :class:`CalledProcessError` when the return code is non-zero.
        shell: Whether to execute through the shell.
        encoding: Text decoding applied to process output.
        errors: Error handling strategy used while decoding.
        timeout: Maximum time in seconds to wait for completion.

        :class:`subprocess.CompletedProcess` containing collected stdout/stderr.
    """

    console = reporter.console if reporter is not None else get_shared_console()
    resolved_index = _resolve_stream_thread_index(thread_index)
    lock = threading.RLock()

    popen_env = dict(env) if env is not None else None

    process = subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding=encoding,
        errors=errors,
        bufsize=1,
        universal_newlines=True,
        cwd=cwd,
        env=popen_env,
        shell=shell,
        start_new_session=start_new_session,
        creationflags=creationflags,
        preexec_fn=preexec_fn,
        stdin=stdin,
    )
    register_subprocess(process)

    assert process.stdout is not None
    assert process.stderr is not None

    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []

    def _emit(line: str, kind: str) -> None:
        # Preserve ANSI color codes by disabling Rich markup parsing.
        stripped = line.rstrip("\r\n")
        if line_handler is not None:
            try:
                line_handler(line, kind)
            except Exception:
                ...
        if not emit_to_console:
            return
        prefix = f"[T{resolved_index}] "
        with lock:
            console.print(f"{prefix}{stripped}", markup=False)

    def _reader(stream: "io.TextIOBase", sink: list[str], kind: str) -> None:
        try:
            for line in iter(stream.readline, ""):
                sink.append(line)
                _emit(line, kind)
        finally:
            try:
                stream.close()
            except Exception:
                ...

    threads = [
        threading.Thread(target=_reader, args=(process.stdout, stdout_chunks, "stdout"), daemon=True),
        threading.Thread(target=_reader, args=(process.stderr, stderr_chunks, "stderr"), daemon=True),
    ]
    for t in threads:
        t.start()

    try:
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            raise
    finally:
        unregister_subprocess(process)
        for t in threads:
            t.join()

    stdout_text = "".join(stdout_chunks)
    stderr_text = "".join(stderr_chunks)

    result = subprocess.CompletedProcess(
        args=args,
        returncode=process.returncode,
        stdout=stdout_text,
        stderr=stderr_text,
    )

    cancel_event = get_cancel_event()
    cancelled = cancel_event.is_set()

    if check and result.returncode != 0:
        if cancelled:
            raise TaskCancelledException("Subprocess cancelled by shutdown signal.")
        raise subprocess.CalledProcessError(
            returncode=result.returncode,
            cmd=args,
            output=stdout_text,
            stderr=stderr_text,
        )

    return result


P = ParamSpec("P")
_R = TypeVar("_R")
T = TypeVar("T")


class TaskCancelledException(Exception): ...


# Global cancellation event exposed so long-running tasks can poll cooperatively
_GLOBAL_CANCEL = threading.Event()


def get_cancel_event() -> threading.Event:
    """Return a process-wide cancellation event set on SIGINT.

    Long-running worker functions should periodically check this event and
    exit early when it becomes set to ensure prompt shutdown.
    """
    return _GLOBAL_CANCEL


# Track child subprocesses so we can terminate their process groups on SIGINT
_SUBPROCESSES_LOCK = threading.RLock()
_SUBPROCESSES: set["subprocess.Popen[Any]"] = set()


def register_subprocess(p: "subprocess.Popen[Any]") -> None:
    with _SUBPROCESSES_LOCK:
        _SUBPROCESSES.add(p)


def unregister_subprocess(p: "subprocess.Popen[Any]") -> None:
    with _SUBPROCESSES_LOCK:
        _SUBPROCESSES.discard(p)


def kill_registered_subprocesses(sig: int = signal.SIGTERM, grace_seconds: float = 2.0) -> None:
    """Send signal to all tracked subprocesses (entire groups when possible).

    Attempts SIGTERM first and escalates to SIGKILL after a short grace period
    for any processes still alive.
    """
    procs: list["subprocess.Popen[Any]"]
    with _SUBPROCESSES_LOCK:
        procs = list(_SUBPROCESSES)

    # First pass: try graceful termination
    for p in procs:
        try:
            if os.name == "posix":
                os.killpg(os.getpgid(p.pid), sig)
            else:
                # Prefer CTRL_BREAK_EVENT to terminate the whole process group
                try:
                    p.send_signal(signal.CTRL_BREAK_EVENT)
                except Exception:
                    p.terminate()
        except Exception:
            ...

    # Give processes a brief window to exit
    deadline = time.time() + grace_seconds
    for p in procs:
        while time.time() < deadline:
            if p.poll() is not None:
                break
            time.sleep(0.1)

    # Second pass: force kill anything still running
    for p in procs:
        if p.poll() is None:
            try:
                if os.name == "posix":
                    os.killpg(os.getpgid(p.pid), signal.SIGKILL)
                else:
                    p.kill()
            except Exception:
                ...


@contextmanager
def progress_bar_threadpool(
    n: int,
    *,
    threads: int,
    stop_on_exception: bool = False,
    executor: ThreadPoolExecutor | None = None,
    debug: bool = False,
    step_stats: tuple[int, ...] | None = (5, 10),
):
    """Creates a thread pool with a progress bar that handles graceful shutdown.

    Args:
        n: Total number of tasks to track in the progress bar
        threads: Number of worker threads in the pool
        stop_on_exception: If True (default), stops all processing immediately
            when any task raises an exception. If False, continues processing
            other tasks and logs exceptions.
        executor: Optional pre-configured executor. When provided, the caller
            remains responsible for its lifecycle. This allows sharing a thread
            pool across multiple progress contexts.
        step_stats: Tuple of window sizes for rolling averages (default: (5, 10)).
            Set to None to disable the step statistics column.

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

    manage_executor = executor is None
    executor_cm = ThreadPoolExecutor(threads) if manage_executor else nullcontext(executor)

    with progress_bar(n, step_stats=step_stats) as callback, executor_cm as exc:
        console = get_shared_console()
        console_logger = logger.bind(**{CONSOLE_SKIP_EXTRA: True})

        def log_to_console(
            level: str,
            message: str,
            *,
            color: str | None = None,
            exception: Exception | None = None,
        ) -> None:
            """Emit a message above the progress bar without forcing a redraw."""

            styled = f"[{color}]{message}[/]" if color else message
            console.log(styled)

            target = console_logger.opt(exception=exception) if exception else console_logger
            getattr(target, level)(message)

        def signal_handler(signum: int, frame: FrameType | None) -> None:
            # Avoid using the logger inside signal handlers (Loguru is not re-entrant)
            try:
                os.write(2, b"\nShutting down...\n")
            except Exception:
                ...
            should_terminate.set()
            _GLOBAL_CANCEL.set()
            try:
                kill_registered_subprocesses()
            except Exception:
                ...
            for fut in futs:
                fut.cancel()
            # Defer full executor shutdown to the normal control flow
            try:
                exc.shutdown(wait=False, cancel_futures=True)
            except Exception:
                ...

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
                    message = f"Task raised an exception: {e} (args={task_args}, kwargs={task_kwargs})"
                    log_to_console("error", message, color="bold red", exception=e if debug else None)
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
            if should_terminate.is_set():
                # Propagate cancellation to callers so higher levels can stop
                raise TaskCancelledException("Cancelled by SIGINT")
            if stop_on_exception and exceptions_encountered:
                raise exceptions_encountered[0]
            elif not stop_on_exception and exceptions_encountered:
                log_to_console(
                    "warning",
                    f"Finished processing with {len(exceptions_encountered)} errors.",
                    color="yellow",
                )

        finally:
            # Ensure original signal handler is restored
            signal.signal(signal.SIGINT, prior_handler)
            _GLOBAL_CANCEL.clear()
            # Ensure executor is shut down cleanly, cancelling any remaining futures
            # if termination was signaled. The context manager (`with ThreadPoolExecutor...`)
            # also calls shutdown, but calling it explicitly ensures cancellation
            # logic is triggered if `should_terminate` was set.
            if manage_executor:
                if should_terminate.is_set():
                    exc.shutdown(wait=True, cancel_futures=True)
                else:
                    exc.shutdown(wait=True)


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
