"""
GPU-aware Dask cluster utilities used by distributed_segmentation.

Provides a simple way to:
 - Start a SpecCluster with multiple workers per GPU (via CUDA_VISIBLE_DEVICES pinning)
 - Optionally start a LocalCUDACluster (one worker per GPU) when dask-cuda is available

Expose a context-managed cluster wrapper `myGPUCluster` with `.client` and
`.dashboard_link` to mirror the interface expected by distributed_segmentation.
"""

from __future__ import annotations

import os
import subprocess
from typing import Any

from dask.distributed import Client

DEFAULT_WORKER_MEMORY = "16GB"


def _auto_device_tokens() -> list[str]:
    """Use nvidia-smi to build a sequential list of GPU indices."""
    try:
        out = subprocess.check_output(["nvidia-smi", "-L"], text=True, stderr=subprocess.DEVNULL)
        tokens: list[str] = []
        for idx, line in enumerate(out.splitlines()):
            if line.strip():
                tokens.append(str(idx))
        return tokens
    except Exception:
        return []


def _visible_devices() -> list[str]:
    """Respect CUDA_VISIBLE_DEVICES if provided, otherwise enumerate GPUs."""
    env_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env_devices:
        tokens = [token.strip() for token in env_devices.split(",") if token.strip()]
        if tokens:
            return tokens
    return _auto_device_tokens()


def _gpu_count() -> int:
    return len(_visible_devices())


def _build_worker_spec(*, devices: list[str], workers_per_gpu: int, threads_per_worker: int) -> dict[str, Any]:
    """Construct a SpecCluster worker spec with env pinning per GPU."""
    from distributed.nanny import Nanny

    worker_spec: dict[str, Any] = {}
    for dev, token in enumerate(devices):
        for k in range(workers_per_gpu):
            name = f"gpu-{dev}-w{k}"
            worker_spec[name] = {
                "cls": Nanny,
                "options": {
                    "nthreads": threads_per_worker,
                    "memory_limit": DEFAULT_WORKER_MEMORY,
                    "env": {"CUDA_VISIBLE_DEVICES": token},
                },
            }
    return worker_spec


def _start_speccluster(*, workers_per_gpu: int, threads_per_worker: int) -> tuple[Client, str | None]:
    from distributed import Scheduler
    from distributed.deploy.spec import SpecCluster

    devices = _visible_devices()
    if not devices:
        raise RuntimeError("No GPUs detected; cannot start GPU SpecCluster")

    spec = _build_worker_spec(
        devices=devices,
        workers_per_gpu=int(workers_per_gpu),
        threads_per_worker=int(threads_per_worker),
    )
    cluster = SpecCluster(workers=spec, scheduler={"cls": Scheduler, "options": {}})
    client = Client(cluster)
    return client, getattr(cluster, "dashboard_link", None)


def _start_localcuda(*, n_workers: int | None, threads_per_worker: int) -> tuple[Client, str | None]:
    try:
        from dask_cuda import LocalCUDACluster  # type: ignore
    except Exception as e:  # pragma: no cover - env dependent
        raise RuntimeError("dask-cuda is required for LocalCUDACluster but is not installed") from e

    cluster = LocalCUDACluster(
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        memory_limit=DEFAULT_WORKER_MEMORY,
    )
    client = Client(cluster)
    return client, getattr(cluster, "dashboard_link", None)


class myGPUCluster:
    """Context-managed GPU cluster exposing `.client` and `.dashboard_link`.

    Parameters
    ----------
    workers_per_gpu : int
        If >1, uses SpecCluster with CUDA_VISIBLE_DEVICES pinning and this many
        workers per GPU. If <=1 and `use_localcuda=True`, starts LocalCUDACluster.
    threads_per_worker : int
        Dask threads per worker (default 1 for GPU-bound work).
    use_localcuda : bool
        If True and workers_per_gpu <= 1, use LocalCUDACluster (requires dask-cuda).
    n_workers : int | None
        When using LocalCUDACluster, number of workers (typically equals number of GPUs).
    """

    def __init__(
        self,
        *,
        workers_per_gpu: int = 2,
        threads_per_worker: int = 1,
        use_localcuda: bool = False,
        n_workers: int | None = None,
        **_: Any,
    ) -> None:
        self._client: Client | None = None
        self._dashboard: str | None = None
        self._close: list[Any] = []

        if workers_per_gpu and workers_per_gpu > 1:
            self._client, self._dashboard = _start_speccluster(
                workers_per_gpu=workers_per_gpu, threads_per_worker=threads_per_worker
            )
        else:
            if use_localcuda:
                self._client, self._dashboard = _start_localcuda(
                    n_workers=n_workers, threads_per_worker=threads_per_worker
                )
            else:
                # Fall back to SpecCluster with a single worker per GPU to avoid hard dependency on dask-cuda
                self._client, self._dashboard = _start_speccluster(
                    workers_per_gpu=1, threads_per_worker=threads_per_worker
                )

        # Mirror attributes expected by distributed_segmentation
        self.client = self._client  # type: ignore[assignment]
        self.dashboard_link = self._dashboard

    def __enter__(self) -> "myGPUCluster":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            if self._client is not None:
                self._client.close()
        finally:
            self._client = None
