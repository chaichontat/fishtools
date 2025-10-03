from __future__ import annotations

import sys
from loguru import logger


def configure_logging(debug: bool, *, process_label: str | None = None) -> None:
    """Configure loguru consistently for CLI commands.

    - INFO by default with simple message format
    - DEBUG adds timestamped rich format and a profiling file sink
    - Optional process label prefix for multi-process displays
    """
    logger.remove()
    if process_label is not None:
        logger.configure(extra={"process_label": process_label})
        prefix = "[P{extra[process_label]}] "
    else:
        prefix = ""

    if debug:
        log_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{process.name: <13}</cyan> | "
            f"{prefix}<level>{{message}}</level>"
        )
        logger.add(sys.stderr, level="DEBUG", format=log_format, enqueue=True)
        logger.add("profiling_{time}.log", level="DEBUG", format=log_format, enqueue=True)
        logger.debug("Debug logging configured (with profiling sink).")
    else:
        logger.add(sys.stderr, level="INFO", format=f"{prefix}{{message}}", enqueue=True)

