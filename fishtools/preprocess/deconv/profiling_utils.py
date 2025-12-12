from __future__ import annotations

from contextlib import contextmanager
from typing import Dict

import cupy as cp


class GpuTimer:
    """
    Event-based GPU timer for CuPy. Timings are per current stream.
    Use as: with timer.section("name"):  # enqueue kernels ...
    """

    def __init__(self, stream: cp.cuda.Stream | None = None):
        self._stream = stream
        self._stream_initialized = stream is not None
        self.times: Dict[str, float] = {}

    @property
    def stream(self) -> cp.cuda.Stream:
        """Lazy initialization of CUDA stream - only accessed when timer is actually used."""
        if not self._stream_initialized:
            self._stream = cp.cuda.get_current_stream()
            self._stream_initialized = True
        return self._stream

    @contextmanager
    def section(self, name: str):
        start = cp.cuda.Event()
        end = cp.cuda.Event()
        start.record(self.stream)
        try:
            yield
        finally:
            end.record(self.stream)
            end.synchronize()
            ms = cp.cuda.get_elapsed_time(start, end)
            self.times[name] = self.times.get(name, 0.0) + ms * 1e-3  # seconds

    def reset(self):
        self.times = {}

    def to_dict(self) -> Dict[str, float]:
        return dict(self.times)


@contextmanager
def nvtx_range(msg: str):
    from cupy.cuda import nvtx

    nvtx.RangePush(msg)
    try:
        yield
    finally:
        nvtx.RangePop()
