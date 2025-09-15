"""Lightweight IPC queue server using multiprocessing.Manager.

Exposes a single shared queue with a small, explicit API suitable for
local inter‑process communication.

Methods exposed to clients:
  - put(item): enqueue at tail (FIFO)
  - get(): dequeue from head (FIFO); raises IndexError if empty
  - pop(): remove from tail; raises IndexError if empty
  - peek(): view head item or None if empty
  - list(limit: int | None = None): snapshot of up to ``limit`` items
  - extend(items): enqueue many; returns count added
  - size(): current length
  - empty(): True if no items
  - clear(): remove all; returns count removed
  - remove(item): remove first matching item; returns True if removed
  - position(item): 0‑based index or -1 if not present
  - get_if_head(expected): atomically pop only if ``expected`` is at head

Server configuration via env vars (defaults in parentheses):
  - SCHED_HOST (127.0.0.1)
  - SCHED_PORT (50000)
  - SCHED_AUTHKEY (change-me)
"""

from __future__ import annotations

import logging
import os
import signal
import sys
import threading
import time
from collections import deque
from multiprocessing.managers import BaseManager
from threading import Lock
from typing import Any, Iterable


# Helpers (must be defined before main() because serve_forever() blocks)
def _preview(obj: Any, maxlen: int = 120) -> str | None:
    if obj is None:
        return None
    try:
        s = repr(obj)
    except Exception:
        s = f"<non-repr {type(obj).__name__}>"
    return s if len(s) <= maxlen else s[: maxlen - 1] + "…"


# Configure module logger (standard library)
logger = logging.getLogger("fishtools.queue")


class IPCQueue:
    def __init__(self) -> None:
        self._dq: deque[Any] = deque()
        self._lock = Lock()
        logger.debug("IPCQueue initialized")
        # Heartbeat tracking
        self._last_seen: dict[Any, float] = {}
        self._timeout_s: float = float(os.getenv("SCHED_HEARTBEAT_TIMEOUT", "5"))
        self._sweep_interval_s: float = float(os.getenv("SCHED_SWEEP_INTERVAL", "1"))
        self._sweeper = threading.Thread(target=self._sweep_loop, name="ipcqueue-sweeper", daemon=True)
        self._sweeper.start()

    # Queue-like API
    def put(self, item: Any) -> None:
        with self._lock:
            before = len(self._dq)
            self._dq.append(item)
            after = len(self._dq)
            self._last_seen[item] = time.time()
            logger.info(
                f"queue.put: size {before} -> {after} head={_preview(self._dq[0]) if self._dq else None}"
            )

    def get(self) -> Any:
        with self._lock:
            if not self._dq:
                logger.info("queue.get: empty")
                raise IndexError("get from empty queue")
            item = self._dq.popleft()
            self._last_seen.pop(item, None)
            logger.info(f"queue.get: popped head={_preview(item)}, size now {len(self._dq)}")
            return item

    def pop(self) -> Any:
        with self._lock:
            if not self._dq:
                logger.info("queue.pop: empty")
                raise IndexError("pop from empty queue")
            item = self._dq.pop()
            self._last_seen.pop(item, None)
            logger.info(f"queue.pop: removed tail={_preview(item)}, size now {len(self._dq)}")
            return item

    def peek(self) -> Any | None:
        with self._lock:
            head = self._dq[0] if self._dq else None
            logger.debug(f"queue.peek: head={_preview(head)}")
            return head

    def list(self, limit: int | None = None) -> list[Any]:
        with self._lock:
            if limit is None or limit >= len(self._dq):
                out = list(self._dq)
            else:
                # Slice without copying the whole deque
                out = [self._dq[i] for i in range(limit)]
            logger.debug(f"queue.list: size={len(self._dq)} returned={len(out)}")
            return out

    def extend(self, items: Iterable[Any]) -> int:
        with self._lock:
            before = len(self._dq)
            n = 0
            for it in items:
                self._dq.append(it)
                n += 1
            after = len(self._dq)
            logger.info(f"queue.extend: +{n} size {before} -> {after}")
            return n

    def size(self) -> int:
        with self._lock:
            s = len(self._dq)
            logger.debug(f"queue.size: {s}")
            return s

    def empty(self) -> bool:
        e = self.size() == 0
        logger.debug(f"queue.empty: {e}")
        return e

    def clear(self) -> int:
        with self._lock:
            n = len(self._dq)
            self._dq.clear()
            self._last_seen.clear()
            logger.info(f"queue.clear: removed {n} items")
            return n

    def remove(self, item: Any) -> bool:
        with self._lock:
            try:
                self._dq.remove(item)
                self._last_seen.pop(item, None)
                logger.info(f"queue.remove: removed {_preview(item)} size now {len(self._dq)}")
                return True
            except ValueError:
                logger.debug(f"queue.remove: not found {_preview(item)}")
                return False

    def position(self, item: Any) -> int:
        with self._lock:
            try:
                # deque.index is O(n) which is fine for small control queues
                pos = self._dq.index(item)  # type: ignore[attr-defined]
                logger.debug(f"queue.position: {_preview(item)} -> {pos}")
                return pos
            except ValueError:
                logger.debug(f"queue.position: {_preview(item)} not present")
                return -1

    def get_if_head(self, expected: Any) -> bool:
        with self._lock:
            if self._dq and self._dq[0] == expected:
                self._dq.popleft()
                self._last_seen.pop(expected, None)
                logger.info(f"queue.get_if_head: popped {_preview(expected)} size now {len(self._dq)}")
                return True
            logger.debug(
                f"queue.get_if_head: head={_preview(self._dq[0]) if self._dq else None} did not match expected={_preview(expected)}"
            )
            return False

    # Heartbeat API
    def heartbeat(self, item: Any) -> None:
        with self._lock:
            # Only track known entries
            if item in self._dq:
                self._last_seen[item] = time.time()
                logger.debug(f"queue.heartbeat: {_preview(item)} @ {self._last_seen[item]}")

    # Background sweeper to prune stale entries
    def _sweep_loop(self) -> None:
        while True:
            time.sleep(self._sweep_interval_s)
            now = time.time()
            removed: list[Any] = []
            with self._lock:
                if not self._dq:
                    continue
                fresh = deque()
                for it in list(self._dq):
                    last = self._last_seen.get(it, now)
                    if now - last > self._timeout_s:
                        removed.append(it)
                        self._last_seen.pop(it, None)
                    else:
                        fresh.append(it)
                if removed:
                    self._dq = fresh
            for it in removed:
                logger.warning(
                    f"queue.sweeper: removed stale entry {_preview(it)} after {self._timeout_s:.2f}s without heartbeat"
                )


class QueueManager(BaseManager):
    pass


_QUEUE = IPCQueue()  # Singleton shared across all clients


QueueManager.register(
    "get_queue",
    callable=lambda: _QUEUE,
    exposed=[
        "put",
        "get",
        "pop",
        "peek",
        "list",
        "extend",
        "size",
        "empty",
        "clear",
        "remove",
        "position",
        "get_if_head",
        "heartbeat",
    ],
)


def main() -> None:
    host = os.getenv("SCHED_HOST", "127.0.0.1")
    port = int(os.getenv("SCHED_PORT", "50001"))
    authkey = os.getenv("SCHED_AUTHKEY", "change-me").encode()
    # Configure standard logging
    log_level_name = os.getenv("SCHED_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, log_level_name, logging.INFO)
    logger.setLevel(level)
    # Reset handlers to avoid duplicates
    logger.handlers.clear()
    stream = logging.StreamHandler(sys.stderr)
    stream.setLevel(level)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    stream.setFormatter(fmt)
    logger.addHandler(stream)

    mgr = QueueManager(address=(host, port), authkey=authkey)
    server = mgr.get_server()

    # Graceful shutdown on SIGINT/SIGTERM
    def _shutdown(_sig, _frame) -> None:
        logger.info(f"server: received signal {_sig} — shutting down")
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)
    logger.info(
        f"server: starting at {host}:{port} pid={os.getpid()} authkey_len={len(authkey)} log_level={log_level_name}"
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
