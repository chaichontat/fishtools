"""Tests for threading utilities."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from loguru import logger

import fishtools.utils.threading as threading_utils
from fishtools.utils.logging import CONSOLE_SKIP_EXTRA
from fishtools.utils.pretty_print import (
    TaskCancelledException,
    get_shared_console,
    progress_bar_threadpool,
)


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


def test_progress_bar_threadpool_logs_use_shared_console(monkeypatch: pytest.MonkeyPatch) -> None:
    console_messages: list[tuple[tuple[object, ...], dict[str, object]]] = []
    console = get_shared_console()

    def capture_log(*args: object, **kwargs: object) -> None:
        console_messages.append((args, kwargs))

    monkeypatch.setattr(console, "log", capture_log)

    log_records: list[dict[str, object]] = []

    def sink(message: Any) -> None:
        log_records.append(message.record)

    sink_id = logger.add(sink, level="ERROR", format="{message}")

    def boom() -> None:
        raise RuntimeError("boom")

    with pytest.raises(TaskCancelledException):
        with progress_bar_threadpool(1, threads=1, stop_on_exception=True) as submit:
            submit(boom)

    logger.remove(sink_id)

    assert console_messages, "Exceptions must be routed through shared console"
    console_payload = console_messages[0][0][0]
    assert isinstance(console_payload, str)
    assert "boom" in console_payload

    matching = [record for record in log_records if "boom" in record["message"]]
    assert matching, "Expected error record capturing the exception"
    assert all(record["extra"].get(CONSOLE_SKIP_EXTRA) is True for record in matching), (
        "Console sink should be skipped for captured logs"
    )
