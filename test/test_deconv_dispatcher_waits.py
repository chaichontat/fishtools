import threading
import time
from pathlib import Path
from typing import Any

import pytest


class _QueueStub:
    """Thread-based queue stub matching minimal multiprocessing.Queue API used."""

    def __init__(self, maxsize: int = 0) -> None:
        import queue as _q

        self._q: _q.Queue[Any] = _q.Queue(maxsize=maxsize)

    # put / get API (with timeout and nowait variants)
    def put(self, item: Any) -> None:  # noqa: D401
        self._q.put(item)

    def put_nowait(self, item: Any) -> None:
        self._q.put_nowait(item)

    def get(self, timeout: float | None = None):
        if timeout is None:
            return self._q.get()
        return self._q.get(timeout=timeout)

    def get_nowait(self):
        return self._q.get_nowait()

    # compatibility no-ops
    def close(self) -> None:  # pragma: no cover - compatibility only
        pass

    def cancel_join_thread(self) -> None:  # pragma: no cover - compatibility only
        pass


class _ProcessStub:
    """Minimal process stub: appears alive; no real OS process is spawned."""

    def __init__(self, *_, **__):
        self._alive = True

    def start(self) -> None:  # pragma: no cover - noop
        self._alive = True

    def is_alive(self) -> bool:
        return self._alive

    def join(self, timeout: float | None = None) -> None:  # pragma: no cover - noop
        # Do not block; emulate an already-joined lightweight worker.
        self._alive = False

    def terminate(self) -> None:  # pragma: no cover - noop
        self._alive = False


class _CtxStub:
    """Stub for fishtools.preprocess.deconv.worker.MP_CONTEXT providing Queue/Process."""

    def __init__(self, queues: list[_QueueStub]):
        # Queues are provided in the order they will be requested by run_multi_gpu
        # 1) request_queue
        # 2) result_queue
        # 3..) one task_queue per worker
        self._queues = queues
        self.procs: list[_ProcessStub] = []

    def Queue(self, maxsize: int = 0) -> _QueueStub:  # noqa: N802 - match attribute name
        # Ignore maxsize; queues are preconstructed with any desired capacity
        return self._queues.pop(0)

    def Process(self, *_, **__):  # noqa: N802 - match attribute name
        p = _ProcessStub()
        self.procs.append(p)
        return p


def _wm(worker_id: int, path: Path | None, status: str) -> "WorkerMessage":
    from fishtools.preprocess.deconv.worker import WorkerMessage

    return WorkerMessage(worker_id=worker_id, path=path, status=status)  # type: ignore[arg-type]


@pytest.mark.timeout(5)
def test_dispatcher_waits_for_stopped_before_exit(monkeypatch):
    # Inject a lightweight stub for the heavy GPU backend module before importing worker
    import sys
    import types
    backend_stub = types.ModuleType("fishtools.preprocess.deconv.backend")
    # Minimal symbols required by worker.py type imports
    class _OutputArtifacts:  # pragma: no cover - structure only
        pass
    class _TileProcessor:  # pragma: no cover - structure only
        pass
    backend_stub.OutputArtifacts = _OutputArtifacts
    backend_stub.ProcessorFactory = object  # sentinel placeholder
    backend_stub.TileProcessor = _TileProcessor
    sys.modules["fishtools.preprocess.deconv.backend"] = backend_stub

    # Import after stubbing to avoid importing the real GPU-bound backend
    from fishtools.preprocess.deconv import worker as w

    # Prepare shared stub queues
    request_q = _QueueStub()
    result_q = _QueueStub()
    task_q = _QueueStub()

    # Install context stub so run_multi_gpu uses our queues and process stub
    ctx = _CtxStub([request_q, result_q, task_q])
    monkeypatch.setattr(w, "MP_CONTEXT", ctx)

    files = [Path("/tmp/tile-0001.tif")]
    devices = [0]

    # Driver thread simulating a worker's behavior using our queues
    stopped_posted = threading.Event()
    ok_posted = threading.Event()

    def driver():
        # Initial readiness: worker advertises capacity equal to queue depth (2)
        request_q.put(0)
        request_q.put(0)
        # Receive first assignment
        first = task_q.get(timeout=1.0)
        assert isinstance(first, Path)
        # Simulate processing and emit OK
        time.sleep(0.05)
        result_q.put(_wm(0, first, "ok"))
        ok_posted.set()
        # Request again (dispatcher will be out of files now)
        request_q.put(0)
        # Receive sentinel and only then emit stopped
        sentinel = task_q.get(timeout=1.0)
        assert sentinel is None
        # Delay a bit to ensure dispatcher must truly wait
        time.sleep(0.05)
        result_q.put(_wm(0, None, "stopped"))
        # Simulate process exit shortly after signaling stopped so the
        # dispatcher can drop it on timeout cycles.
        if ctx.procs:
            ctx.procs[0]._alive = False
        stopped_posted.set()

    t = threading.Thread(target=driver, daemon=True)
    t.start()

    events: list[tuple[str, Any]] = []

    def on_progress(msg):
        events.append((msg.status, msg.path))

    failures = w.run_multi_gpu(
        files,
        devices=devices,
        processor_factory=lambda *_: None,  # type: ignore[assignment]
        queue_depth=2,
        stop_on_error=True,
        progress_callback=on_progress,
        debug=False,
    )

    # The dispatcher should have waited until the driver posted 'stopped'.
    assert stopped_posted.is_set(), "run_multi_gpu returned before worker signaled 'stopped'"
    # Verify progress callback saw both the ok completion and the stopped lifecycle event
    statuses = [s for s, _ in events]
    assert "ok" in statuses
    assert "stopped" in statuses
    # No failures should be reported in this happy-path drain
    assert failures == []
