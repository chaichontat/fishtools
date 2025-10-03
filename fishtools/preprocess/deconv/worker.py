"""Shared multi-GPU worker orchestration utilities for deconvolution."""

from __future__ import annotations

import multiprocessing as mp
import os
import queue as pyqueue
import signal
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Sequence

import cupy as cp
import numpy as np
from loguru import logger

from fishtools.preprocess.deconv.backend import OutputArtifacts, ProcessorFactory, TileProcessor
from fishtools.preprocess.deconv.logging_utils import configure_logging

DEFAULT_QUEUE_DEPTH = 5

try:
    MP_CONTEXT = mp.get_context("spawn")
except ValueError:  # pragma: no cover - spawn should exist on supported platforms
    MP_CONTEXT = mp.get_context("forkserver")  # pyright: ignore[reportConstantRedefinition]


WorkerStatus = Literal["ok", "error", "stopped"]


@dataclass(slots=True)
class WorkerMessage:
    worker_id: int
    path: Path | None
    status: WorkerStatus
    duration: float | None = None
    error: str | None = None


def configure_worker_logging(debug: bool, *, process_label: str) -> None:
    """Configure logging for a worker or coordinating process."""

    configure_logging(debug, process_label=process_label)


def parse_device_spec(spec: str | Sequence[int] | None) -> list[int]:
    """Resolve a device specification into explicit CUDA device indices."""

    visible_env = os.environ.get("CUDA_VISIBLE_DEVICES")
    visible_mask: list[int] | None = None
    if visible_env:
        visible_mask = [int(item.strip()) for item in visible_env.split(",") if item.strip()]

    try:
        total = cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError as exc:  # pragma: no cover - guarded in caller
        raise RuntimeError("No CUDA devices available for multi-GPU execution") from exc

    if total <= 0:
        raise RuntimeError("No CUDA devices detected")

    if spec is None or spec == "auto":
        return list(range(total))

    if isinstance(spec, str):
        items = [s.strip() for s in spec.split(",") if s.strip()]
        result = [int(it) for it in items]
    else:
        result = [int(it) for it in spec]

    if not result:
        raise ValueError("Device specification resolved to an empty list")

    if visible_mask is not None:
        max_idx = len(visible_mask) - 1
        for dev in result:
            if dev < 0 or dev > max_idx:
                raise ValueError(f"Device index {dev} is outside the visible range [0, {max_idx}]")
    else:
        for dev in result:
            if dev < 0 or dev >= total:
                raise ValueError(f"Device index {dev} is outside the range [0, {total - 1}]")

    return result


def _worker_loop(
    *,
    worker_id: int,
    device_id: int,
    queue_depth: int,
    task_queue: mp.Queue,
    request_queue: mp.Queue,
    result_queue: mp.Queue,
    processor_factory: ProcessorFactory,
    stop_on_error: bool,
    debug: bool,
) -> None:
    """Worker entrypoint executed in a separate process."""

    configure_worker_logging(debug, process_label=str(worker_id + 1))
    logger.debug(f"Worker {worker_id} starting on device {device_id} with queue depth {queue_depth}.")
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    if hasattr(cp.cuda, "Device"):
        cp.cuda.Device(device_id).use()

    processor = processor_factory(device_id)
    processor.setup()

    simple_mode = not all(hasattr(processor, attr) for attr in ("read_tile", "compute_tile", "write_tile"))

    STOP = object()
    if simple_mode and hasattr(processor, "process"):
        try:
            for _ in range(queue_depth):
                request_queue.put(worker_id)
            while True:
                try:
                    item = task_queue.get()
                except (EOFError, OSError):
                    break
                if item is None:
                    break
                request_queue.put(worker_id)
                try:
                    duration = processor.process(item)
                except Exception as exc:  # noqa: BLE001
                    import traceback

                    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
                    result_queue.put(
                        WorkerMessage(worker_id=worker_id, path=item, status="error", duration=None, error=tb)
                    )
                    if stop_on_error:
                        break
                    continue
                result_queue.put(
                    WorkerMessage(worker_id=worker_id, path=item, status="ok", duration=duration)
                )
        finally:
            processor.teardown()
            result_queue.put(WorkerMessage(worker_id=worker_id, path=None, status="stopped"))
        return

    q_in: "pyqueue.Queue[tuple[Path, np.ndarray, np.ndarray, dict[str, Any], tuple[int, int]] | object]" = (
        pyqueue.Queue(maxsize=2)
    )
    q_out: "pyqueue.Queue[tuple[Path, np.ndarray, dict[str, Any], OutputArtifacts, dict[str, float]] | object]" = (
        pyqueue.Queue(maxsize=2)
    )
    stop_evt = threading.Event()

    def reader() -> None:
        for _ in range(queue_depth):
            request_queue.put(worker_id)

        try:
            while not stop_evt.is_set():
                try:
                    item = task_queue.get()
                except (EOFError, OSError):
                    break
                if item is None:
                    break
                request_queue.put(worker_id)
                try:
                    path, nofid, fid, metadata, hw, *_ = processor.read_tile(item)
                except Exception as exc:  # noqa: BLE001
                    import traceback

                    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
                    result_queue.put(
                        WorkerMessage(worker_id=worker_id, path=item, status="error", duration=None, error=tb)
                    )
                    if stop_on_error:
                        stop_evt.set()
                        break
                    continue
                q_in.put((path, nofid, fid, metadata, hw))
        finally:
            q_in.put(STOP)

    def compute() -> None:
        if hasattr(cp.cuda, "Device"):
            cp.cuda.Device(device_id).use()
        while not stop_evt.is_set():
            job = q_in.get()
            if job is STOP:
                q_out.put(STOP)
                break
            path, nofid, fid_np, metadata, hw = job
            t0 = time.perf_counter()
            try:
                artifacts, metadata_out, t_dict = processor.compute_tile(nofid, path, hw, metadata)
                elapsed = time.perf_counter() - t0
                t_gpu = (
                    t_dict.get("basic", 0.0)
                    + t_dict.get("deconv", 0.0)
                    + t_dict.get("quant", 0.0)
                    + t_dict.get("post", 0.0)
                )
                logger.info(
                    f"[GPU{device_id}] {path.name}: gpu={t_gpu:.2f}s (basic={t_dict.get('basic', 0.0):.2f}+"
                    f"dec={t_dict.get('deconv', 0.0):.2f}+quant={t_dict.get('quant', 0.0):.2f}+post={t_dict.get('post', 0.0):.2f})"
                    f" stage_total={elapsed:.2f}s"
                )
                q_out.put((path, fid_np, metadata_out, artifacts, t_dict))
            except Exception as exc:  # noqa: BLE001
                import traceback

                tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
                result_queue.put(
                    WorkerMessage(worker_id=worker_id, path=path, status="error", duration=None, error=tb)
                )
                if stop_on_error:
                    stop_evt.set()
                    q_out.put(STOP)
                    break
            finally:
                del nofid

    def writer() -> None:
        while True:
            job = q_out.get()
            if job is STOP:
                break
            path, fid_np, metadata_out, artifacts, t_dict = job
            try:
                t_write = processor.write_tile(path, fid_np, metadata_out, artifacts)
                t_gpu = (
                    t_dict.get("basic", 0.0)
                    + t_dict.get("deconv", 0.0)
                    + t_dict.get("quant", 0.0)
                    + t_dict.get("post", 0.0)
                )
                logger.debug(f"[{path.name}] Write stage took: {t_write:.4f}s")
                result_queue.put(
                    WorkerMessage(worker_id=worker_id, path=path, status="ok", duration=t_gpu + t_write)
                )
            except Exception as exc:  # noqa: BLE001
                import traceback

                tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
                result_queue.put(
                    WorkerMessage(worker_id=worker_id, path=path, status="error", duration=None, error=tb)
                )
                if stop_on_error:
                    stop_evt.set()
                    break

    t_reader = threading.Thread(target=reader, name="reader", daemon=True)
    t_compute = threading.Thread(target=compute, name="compute", daemon=True)
    t_writer = threading.Thread(target=writer, name="writer", daemon=True)

    try:
        t_reader.start()
        t_compute.start()
        t_writer.start()
        t_reader.join()
        t_compute.join()
        t_writer.join()
    finally:
        processor.teardown()
        result_queue.put(WorkerMessage(worker_id=worker_id, path=None, status="stopped"))


def run_multi_gpu(
    files: Sequence[Path],
    *,
    devices: Sequence[int],
    processor_factory: ProcessorFactory,
    queue_depth: int = DEFAULT_QUEUE_DEPTH,
    stop_on_error: bool = True,
    progress_callback: Callable[[WorkerMessage], None] | None = None,
    debug: bool = False,
) -> list[WorkerMessage]:
    """Execute deconvolution across multiple GPUs with dynamic scheduling."""

    if queue_depth <= 0:
        raise ValueError("queue_depth must be positive")
    if not files:
        return []

    logger.info(
        f"Dispatching {len(files)} tile(s) across {len(devices)} worker(s) with queue depth {queue_depth}."
    )

    worker_processes: list[mp.Process] = []
    task_queues: list[mp.Queue] = []

    def _run_sequential_fallback() -> list[WorkerMessage]:
        logger.warning("Falling back to sequential deconvolution due to multiprocessing limitations.")
        failures_local: list[WorkerMessage] = []
        device_id = devices[0] if devices else 0
        processor = processor_factory(device_id)
        processor.setup()
        try:
            for path in files:
                try:
                    duration = processor.process(path)
                except Exception as exc:  # noqa: BLE001
                    import traceback

                    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
                    msg = WorkerMessage(
                        worker_id=0,
                        path=path,
                        status="error",
                        duration=None,
                        error=tb,
                    )
                    failures_local.append(msg)
                    if progress_callback is not None:
                        progress_callback(msg)
                    if stop_on_error:
                        break
                    continue
                msg = WorkerMessage(worker_id=0, path=path, status="ok", duration=duration)
                if progress_callback is not None:
                    progress_callback(msg)
        finally:
            processor.teardown()
        return failures_local

    try:
        request_queue: mp.Queue = MP_CONTEXT.Queue(maxsize=max(1, queue_depth * max(len(devices), 1)))
        result_queue: mp.Queue = MP_CONTEXT.Queue(maxsize=max(1, len(files) + len(devices)))
    except PermissionError:
        return _run_sequential_fallback()

    assigned: dict[int, list[Path]] = {}

    for wid, device_id in enumerate(devices):
        task_queue: mp.Queue = MP_CONTEXT.Queue(maxsize=queue_depth)
        proc = MP_CONTEXT.Process(
            target=_worker_loop,
            kwargs={
                "worker_id": wid,
                "device_id": device_id,
                "queue_depth": queue_depth,
                "task_queue": task_queue,
                "request_queue": request_queue,
                "result_queue": result_queue,
                "processor_factory": processor_factory,
                "stop_on_error": stop_on_error,
                "debug": debug,
            },
            name=f"P{wid + 1}",
            daemon=True,
        )
        proc.start()
        worker_processes.append(proc)
        task_queues.append(task_queue)
        assigned[wid] = []

    pending_files = deque(Path(path) for path in files)
    active_workers = set(range(len(devices)))
    pending_stop: set[int] = set()
    failures: list[WorkerMessage] = []
    cancelled_paths: set[Path] = set()

    try:
        while active_workers:
            try:
                wid = request_queue.get(timeout=1.0)
            except pyqueue.Empty:
                for proc in worker_processes:
                    if not proc.is_alive():
                        active_workers.discard(worker_processes.index(proc))
                continue

            if wid not in active_workers:
                continue

            task_queue = task_queues[wid]

            if wid in pending_stop:
                task_queue.put(None)
                active_workers.discard(wid)
                continue

            if not pending_files:
                task_queue.put(None)
                active_workers.discard(wid)
                continue

            next_path = pending_files.popleft()
            task_queue.put(next_path)
            assigned[wid].append(next_path)

            while True:
                try:
                    message = result_queue.get_nowait()
                except pyqueue.Empty:
                    break

                if message.path is not None:
                    assigned_list = assigned.get(message.worker_id, [])
                    if message.path in assigned_list:
                        assigned_list.remove(message.path)

                if progress_callback is not None:
                    progress_callback(message)

                if message.status == "error":
                    failures.append(message)
                    if stop_on_error:
                        for wid_requeue, items in assigned.items():
                            for pending_path in items:
                                cancelled_paths.add(pending_path)
                            assigned[wid_requeue] = []
                        cancelled_paths.update(pending_files)
                        pending_files.clear()
                        pending_stop.update(active_workers)
                elif message.status == "stopped":
                    active_workers.discard(message.worker_id)
    finally:
        for proc, task_queue in zip(worker_processes, task_queues):
            if proc.is_alive():
                try:
                    task_queue.put_nowait(None)
                except pyqueue.Full:
                    try:
                        task_queue.close()
                        task_queue.cancel_join_thread()
                    except Exception:  # pragma: no cover
                        pass
            else:
                try:
                    task_queue.close()
                except Exception:  # pragma: no cover
                    pass

        for proc in worker_processes:
            if proc.is_alive():
                proc.join(timeout=10.0)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=5.0)

        while True:
            try:
                message = result_queue.get_nowait()
            except pyqueue.Empty:
                break
            if progress_callback is not None:
                progress_callback(message)
            if message.status == "error":
                failures.append(message)

        for cancelled in cancelled_paths:
            failures.append(
                WorkerMessage(
                    worker_id=-1,
                    path=cancelled,
                    status="error",
                    duration=None,
                    error="Cancelled after earlier failure; tile not processed.",
                )
            )

    return failures


__all__ = [
    "DEFAULT_QUEUE_DEPTH",
    "MP_CONTEXT",
    "ProcessorFactory",
    "TileProcessor",
    "WorkerMessage",
    "configure_worker_logging",
    "parse_device_spec",
    "run_multi_gpu",
]
