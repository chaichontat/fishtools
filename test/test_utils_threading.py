"""Tests for threading utilities."""

from __future__ import annotations

from collections.abc import Callable

import fishtools.utils.threading as threading_utils
import pytest

from fishtools.utils.pretty_print import progress_bar_threadpool


def _noop() -> None:
    return None


def test_staggered_executor_staggers_worker_start(monkeypatch: pytest.MonkeyPatch) -> None:
    launches: list[float] = []

    def record_sleep(duration: float) -> None:
        launches.append(duration)

    monkeypatch.setattr(threading_utils.time, "sleep", record_sleep)

    with threading_utils.StaggeredThreadPoolExecutor(max_workers=3, launch_delay=0.1) as executor:
        futures = [executor.submit(_noop) for _ in range(3)]
        for future in futures:
            future.result()

    assert launches, "executor should sleep before creating worker threads"
    assert len(launches) == 3
    assert all(duration == pytest.approx(0.1) for duration in launches)


def test_progress_bar_threadpool_reuses_external_executor(monkeypatch: pytest.MonkeyPatch) -> None:
    with threading_utils.shared_thread_pool(max_workers=2, launch_delay=0.0) as executor:
        shutdown_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
        original_shutdown: Callable[..., None] = executor.shutdown

        def tracked_shutdown(*args: object, **kwargs: object) -> None:
            shutdown_calls.append((args, kwargs))
            original_shutdown(*args, **kwargs)

        monkeypatch.setattr(executor, "shutdown", tracked_shutdown)

        with progress_bar_threadpool(2, threads=2, executor=executor) as submit:
            futures = [submit(_noop) for _ in range(2)]

        assert not shutdown_calls, "progress_bar_threadpool must not manage external shutdown"
        for future in futures:
            future.result()

    assert len(shutdown_calls) == 1
