"""GPU memory management helpers.

This module centralizes best‑effort cleanup of GPU allocators used in the
pipeline. It is intentionally defensive and safe to call even when no GPU
work has been performed.

Current behavior:
- CuPy: synchronize device and free default device and pinned memory pools.

Design notes:
- Keep dependencies optional and lightweight; errors during cleanup should not
  affect pipeline success.
"""

from __future__ import annotations

from loguru import logger

try:  # pragma: no cover - tests provide a NumPy-backed stub
    import cupy as cp  # type: ignore
except Exception:  # pragma: no cover - runtime fallback when CuPy unavailable
    cp = None  # type: ignore[assignment]


def release_all() -> None:
    """Best‑effort GPU cleanup.

    Safe to call multiple times. Swallows errors but logs at DEBUG level.
    """

    # CuPy: synchronize and flush memory pools
    try:
        if cp is not None:  # pragma: no branch - simple guard
            # Synchronize current device; then free cached blocks
            try:
                runtime = getattr(cp, "cuda", None)
                if runtime is not None:
                    rt = getattr(runtime, "runtime", None)
                    if rt is not None:
                        try:
                            rt.deviceSynchronize()  # type: ignore[attr-defined]
                        except Exception:
                            # Not fatal; proceed to free pools
                            ...
            except Exception:
                ...

            try:
                pool_getter = getattr(cp, "get_default_memory_pool", None)
                if callable(pool_getter):
                    pool_getter().free_all_blocks()
            except Exception:
                logger.opt(exception=True).debug("Unable to release CuPy default memory pool.")

            try:
                pinned_getter = getattr(cp, "get_default_pinned_memory_pool", None)
                if callable(pinned_getter):
                    pinned_getter().free_all_blocks()
            except Exception:
                logger.opt(exception=True).debug("Unable to release CuPy pinned memory pool.")
    except Exception:
        # Absolutely avoid bubbling cleanup issues into the pipeline
        logger.opt(exception=True).debug("GPU memory cleanup encountered an error; continuing.")


__all__ = ["release_all"]

