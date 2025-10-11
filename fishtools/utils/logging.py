from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

from loguru import logger

from fishtools.io.workspace import Workspace

CONSOLE_SKIP_EXTRA = "_skip_console_sink"

# Unified log line format for console and file sinks
_DEFAULT_LOGGER_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{line}</cyan> [<magenta>{extra[component]}</magenta>] {extra[idx]} {extra[file]} | "
    "- <level>{message}</level>"
)


def setup_logging() -> None:
    """Minimal console logging suitable for library/CLI startup.

    Configures a colorized stderr sink with a compact format. Use
    ``configure_cli_logging`` or ``setup_workspace_logging`` for file sinks.
    """
    logger.remove()
    logger.add(
        sys.stderr,
        colorize=True,
        format="<green>{time:HH:mm:ss}</green> | "
        "<level>{level: >8}</level> | "
        "<cyan>{name}</cyan> | <level>{message}</level>",
    )


def configure_cli_logging(
    workspace: Path | None,
    component: str,
    *,
    console_level: str = "INFO",
    file_level: str = "INFO",
    rotation: str | int | None = "20 MB",
    retention: str | int | None = "30 days",
    extra: dict[str, Any] | None = None,
    enqueue: bool = False,
    use_shared_console: bool = True,
) -> Path | None:
    """Configure loguru sinks for CLI commands.

    Ensures warning-or-higher messages are persisted to
    ``<workspace>/analysis/logs/<component>.log`` while maintaining the
    existing rich console formatting.
    """

    logger.remove()
    combined_extra: dict[str, str | bool]
    combined_extra = {"idx": "", "file": "", "component": component, CONSOLE_SKIP_EXTRA: False}
    if extra:
        combined_extra.update(extra)
    logger.configure(extra=combined_extra)

    if use_shared_console:
        # Route console logs through the shared Rich Console so they play nicely
        # with progress bars (no duplicate bar redraws on each log line).
        from fishtools.utils.pretty_print import get_shared_console  # local import to avoid cycle

        def _sink(message: str) -> None:
            # Print above any active Progress Live display
            get_shared_console().print(message, end="")

        console_format = (
            "{time:HH:mm:ss} | {level: >8} | [{extra[component]}] {extra[idx]} {extra[file]} - {message}"
        )

        def _console_filter(record: dict[str, Any]) -> bool:
            return not record["extra"].get(CONSOLE_SKIP_EXTRA, False)

        logger.add(
            _sink,
            level=console_level,
            format=console_format,
            filter=_console_filter,
            enqueue=False,
            backtrace=False,
            diagnose=False,
        )
    else:
        logger.add(
            sys.stderr,
            level=console_level,
            format=_DEFAULT_LOGGER_FORMAT,
            colorize=True,
            enqueue=enqueue,
            backtrace=False,
            diagnose=False,
        )

    if workspace is None:
        return None

    log_root = workspace / "analysis" / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    log_file = log_root / f"{component}.log"

    logger.add(
        log_file,
        level=file_level,
        format=_DEFAULT_LOGGER_FORMAT,
        rotation=rotation,
        retention=retention,
        enqueue=enqueue,
        backtrace=False,
        diagnose=False,
    )

    os.environ["FISHTOOLS_ACTIVE_LOG"] = str(log_file)
    return log_file


def initialize_logger(idx: int | None = None, debug: bool = False, file: str = "") -> None:
    """Initialize logger with CLI-specific format and context.

    Sets up console sink and stdlib logging interception so third-party
    modules are captured. Does not create a file sink; combine with
    ``configure_cli_logging`` if you want a file sink without workspace
    context.
    """

    configure_cli_logging(
        workspace=None,
        component="cli",
        console_level="DEBUG" if debug else "WARNING",
        file_level="CRITICAL",  # no file sink when no workspace
        extra={"idx": f"{idx:04d}" if idx is not None else "", "file": file},
    )

    class InterceptHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - simple bridge
            try:
                level: Any = logger.level(record.levelname).name
            except ValueError:
                level = record.levelno

            frame, depth = sys._getframe(6), 6
            while frame and frame.f_code.co_filename == logging.__file__:
                frame = frame.f_back
                depth += 1

            logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())

    logging.basicConfig(handlers=[InterceptHandler()], level=logging.DEBUG, force=True)
    logging.getLogger("biothings").setLevel(logging.CRITICAL)


def setup_workspace_logging(
    workspace: Path,
    *,
    component: str,
    idx: int | None = None,
    file: str = "",
    debug: bool = False,
    console_level: str = "INFO",
    file_level: str = "INFO",
    rotation: str | int | None = "20 MB",
    retention: str | int | None = "30 days",
    extra: dict[str, Any] | None = None,
    enqueue: bool = False,
    use_shared_console: bool = True,
) -> Path:
    """Enable workspace-scoped logging for CLI entrypoints.

    - Console remains concise and progress bars render to the console only.
    - File sink writes to ``{workspace}/analysis/logs/{component}.log``.
    - Stdlib logging is intercepted into loguru.
    """

    initialize_logger(idx=idx, debug=debug, file=file)
    log_file = configure_cli_logging(
        workspace=workspace,
        component=component,
        console_level="DEBUG" if debug else console_level,
        file_level="DEBUG" if debug else file_level,
        rotation=rotation,
        retention=retention,
        extra={"idx": f"{idx:04d}" if idx is not None else "", "file": file, **(extra or {})},
        enqueue=enqueue,
        use_shared_console=use_shared_console,
    )
    assert log_file is not None
    return log_file


def resolve_workspace_root(path: Path | str) -> tuple[Path, bool]:
    """Resolve a path to the nearest workspace root.

    Args:
        path: Candidate path pointing to a workspace, analysis subdir, or file inside.

    Returns:
        A tuple ``(root, has_analysis)`` where ``root`` is the resolved directory that best
        represents the workspace root, and ``has_analysis`` indicates whether an ``analysis``
        subdirectory was found during resolution.
    """

    resolved = Path(path).expanduser().resolve()
    base = resolved if resolved.is_dir() else resolved.parent

    try:
        workspace = Workspace(base)
    except (NotADirectoryError, ValueError):
        for candidate in (base, *base.parents):
            analysis_dir = candidate / "analysis"
            if analysis_dir.is_dir():
                return candidate, True
        return base, False

    workspace_root = workspace.path
    has_analysis = workspace.analysis.is_dir()
    return workspace_root, has_analysis


def setup_cli_logging(
    path: Path | None,
    *,
    component: str,
    file: str,
    idx: int | None = None,
    debug: bool = False,
    extra: dict[str, Any] | None = None,
    console_level: str | None = None,
    file_level: str | None = None,
    rotation: str | int | None = "20 MB",
    retention: str | int | None = "30 days",
    enqueue: bool = False,
    use_shared_console: bool = True,
) -> Path | None:
    """Configure CLI logging with workspace-aware file sinks when available.

    When ``path`` resolves to a workspace (determined by locating an ``analysis`` directory),
    this function delegates to :func:`setup_workspace_logging` so logs are written under
    ``analysis/logs``. Otherwise it falls back to console-only logging via
    :func:`initialize_logger` and :func:`configure_cli_logging`.
    """

    extra_payload = dict(extra or {})
    if idx is not None:
        extra_payload.setdefault("idx", f"{idx:04d}")

    desired_console = console_level or ("DEBUG" if debug else "INFO")
    desired_file = file_level or ("DEBUG" if debug else "INFO")

    if path is not None:
        workspace_root, found = resolve_workspace_root(path)
        if found:
            return setup_workspace_logging(
                workspace_root,
                component=component,
                idx=idx,
                file=file,
                debug=debug,
                console_level=desired_console,
                file_level=desired_file,
                rotation=rotation,
                retention=retention,
                extra=extra_payload,
                enqueue=enqueue,
                use_shared_console=use_shared_console,
            )

    initialize_logger(idx=idx, debug=debug)
    configure_cli_logging(
        workspace=None,
        component=component,
        console_level=desired_console,
        file_level="CRITICAL",
        extra=extra_payload,
        enqueue=enqueue,
        use_shared_console=use_shared_console,
    )
    return None
