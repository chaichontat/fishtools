"""Threading utilities for fishtools.

Provides helpers for working with :class:`ThreadPoolExecutor` instances that
need gentler start-up characteristics than the default implementation.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import Iterator


class StaggeredThreadPoolExecutor(ThreadPoolExecutor):
    """Thread pool that delays worker start-up to stagger resource usage."""

    def __init__(self, max_workers: int, *, launch_delay: float = 0.0, **kwargs):
        if launch_delay < 0:
            raise ValueError("launch_delay must be non-negative")
        self._launch_delay = launch_delay
        super().__init__(max_workers=max_workers, **kwargs)

    def _adjust_thread_count(self) -> None:  # noqa: D401 - match superclass signature
        if self._launch_delay > 0 and len(self._threads) < self._max_workers:
            time.sleep(self._launch_delay)
        super()._adjust_thread_count()


@contextmanager
def shared_thread_pool(
    *,
    max_workers: int,
    launch_delay: float = 0.0,
    thread_name_prefix: str | None = None,
) -> Iterator[StaggeredThreadPoolExecutor]:
    """Yield a ``StaggeredThreadPoolExecutor`` that shuts down on exit."""

    executor = StaggeredThreadPoolExecutor(
        max_workers=max_workers,
        launch_delay=launch_delay,
        thread_name_prefix=thread_name_prefix,
    )
    try:
        yield executor
    finally:
        executor.shutdown(wait=True)
