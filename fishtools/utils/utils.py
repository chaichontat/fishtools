import functools
import importlib
import logging
import subprocess
import types
from collections.abc import Callable, Sequence
from functools import cache, wraps
from inspect import getcallargs
from pathlib import Path
from subprocess import PIPE, Popen
from typing import Any, Concatenate, ParamSpec, TypeVar, cast

import loguru
import numpy as np
from loguru import logger

P = ParamSpec("P")
R = TypeVar("R", covariant=True)
TType = TypeVar("TType", bound=type)
TAny = TypeVar("TAny")


def copy_signature(
    kwargs_call: Callable[P, Any],
) -> Callable[[Callable[..., R]], Callable[P, R]]:
    """Decorator does nothing but returning the casted original function"""

    def return_func(func: Callable[..., R]) -> Callable[P, R]:
        return cast(Callable[P, R], func)

    return return_func


def copy_signature_method(
    kwargs_call: Callable[P, Any], cls: TType
) -> Callable[[Callable[..., R]], Callable[Concatenate[TType, P], R]]:
    """Decorator does nothing but returning the casted original function"""

    def return_func(func: Callable[..., R]) -> Callable[Concatenate[TType, P], R]:
        return cast(Callable[Concatenate[TType, P], R], func)

    return return_func


def add_file_context(exc: BaseException, *paths: Path | str | None) -> None:
    """Annotate exceptions with helpful path context.

    Appends file path information to exception args and (when available) to
    the PEP 678 ``Exception.add_note`` for richer error messages upstream.

    This is a tiny utility used across CLIs to ensure failures include the
    relevant filenames without coupling to specific call sites.

    Example:
        try:
            ...
        except Exception as e:
            add_file_context(e, in_path, out_path)
            raise
    """

    path_strings: list[str] = []
    for path in paths:
        if path is None:
            continue
        try:
            path_strings.append(str(Path(path)))
        except TypeError:
            path_strings.append(str(path))

    if not path_strings:
        return

    prefix = ", ".join(path_strings)
    if exc.args:
        first, *rest = exc.args
        if isinstance(first, str):
            if prefix not in first:
                exc.args = (f"{prefix}: {first}", *rest)
        else:
            exc.args = (f"{prefix}: {first!r}", *rest)
    else:
        exc.args = (f"Error while processing {prefix}",)

    try:
        exc.add_note(f"While processing file(s): {prefix}")
    except AttributeError:
        pass


def check_if_posix(f: Callable[P, R]) -> Callable[P, R]:
    @wraps(f)
    def inner(*args: P.args, **kwargs: P.kwargs) -> R:
        if not Path("/").exists():
            raise OSError("Not a POSIX system")
        return f(*args, **kwargs)

    return inner


def run_process(cmd: Sequence[str], input: bytes) -> bytes:
    process = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate(input=input)

    if process.returncode != 0:
        logging.critical(stderr.decode())
        raise RuntimeError(f"{cmd} failed")

    return stdout


@cache
def slide(x: str, n: int = 20) -> list[str]:
    return [x[i : i + n] for i in range(len(x) - n + 1)]


def check_if_exists(
    logger: "loguru.Logger",
    name_detector: Callable[[dict[str, Any]], Path | str] = lambda kwargs: str(next(iter(kwargs.values()))),
):
    """
    A decorator that checks if a file exists before executing a function.
    Can override with the `overwrite` keyword argument.

    Parameters:
    -----------
    logger : loguru.Logger
        A logger object that is used to log warning messages if the file already exists.

    name_detector : Callable[[list[Any], dict[str, Any]], str], optional
        A callable that takes two arguments: a list of positional arguments and a dictionary of keyword arguments.
        The kwargs include all the arguments passed to the decorated function (including args).
        Should return the name of the file to check.

    Returns:
    --------
    Callable[P, R | None]]
        A decorator that may return None if the file already exists.
    """

    def decorator(f: Callable[P, R]) -> Callable[P, R | None]:
        @wraps(f)
        def inner(*args: P.args, **kwargs: P.kwargs) -> R | None:
            if not (path := Path(name_detector(getcallargs(f, *args, **kwargs)))).exists() or kwargs.get(
                "overwrite", False
            ):
                return f(*args, **kwargs)
            logger.warning(f"{path} already exists. Skipping.")

        return inner

    return decorator


def batchable():
    def decorator(f: Callable[P, R]) -> Callable[P, R]:
        @wraps(f)
        def inner(*args: P.args, **kwargs: P.kwargs) -> R:
            return f(*args, **kwargs)

        return inner

    return decorator


def git_hash() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).parent)
            .decode("ascii")
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        ...
    return "unknown"


_T = TypeVar("_T")


def batch_roi(
    look_for: str = "registered--*", include_codebook: bool = False, split_codebook: bool = True
) -> Callable[[Callable[P, _T]], Callable[P, _T | None]]:
    """Decorator enabling batch ROI processing when roi='*' is specified.

    Transforms functions that process individual ROIs into batch processors that can
    handle multiple ROIs sequentially. When roi='*' is passed, the decorator discovers
    all matching ROI directories and calls the decorated function once for each ROI.

    Args:
        look_for: Glob pattern for discovering ROI directories. Defaults to "registered--*"
                 for standard preprocessing workflows.
        include_codebook: Whether to include codebook information in directory matching.
                         Requires 'codebook' parameter in function kwargs when enabled.
        split_codebook: Whether to split codebook suffix from ROI names. If False,
                       ROI names include "+codebook" suffix.

    Returns:
        Decorator function that transforms ROI processors into batch processors.

    Raises:
        ValueError: If required parameters ('roi', 'path') are missing or if
                   codebook parameter is missing when include_codebook=True.
        RuntimeError: If ROI processing fails, with specific ROI context included.

    Example:
        @batch_roi(look_for="stitch--*", include_codebook=True)
        def process_roi(path: Path, roi: str, codebook: str) -> None:
            # Process individual ROI
            pass

        # Single ROI processing
        process_roi(path=workspace, roi="cortex", codebook="book1")

        # Batch processing (processes all matching ROIs)
        process_roi(path=workspace, roi="*", codebook="book1")

    Note:
        - Batch processing always returns None regardless of individual function return types
        - ROIs are processed sequentially in sorted order for reproducible results
        - Uses fishtools.utils.io.Workspace for ROI discovery
        - Failed ROI processing stops the entire batch with clear error context
    """

    def decorator(func: Callable[P, _T]) -> Callable[P, _T | None]:
        @wraps(func)
        def inner(*args: P.args, **kwargs: P.kwargs) -> _T | None:
            # Validate required parameters
            if "roi" not in kwargs:
                raise ValueError("batch_roi requires 'roi' keyword argument")
            if "path" not in kwargs:
                raise ValueError("batch_roi requires 'path' keyword argument")

            if kwargs["roi"] != "*":
                return func(*args, **kwargs)

            # Create local copy to avoid state mutation bug
            current_look_for = look_for

            if include_codebook:
                if "codebook" not in kwargs:
                    raise ValueError(
                        "batch_roi with include_codebook=True requires codebook keyword argument"
                    )
                if isinstance(kwargs["codebook"], str):
                    current_look_for = f"{look_for}+{kwargs['codebook']}"
                elif isinstance(kwargs["codebook"], Path):
                    current_look_for = f"{look_for}+{kwargs['codebook'].stem}"
                else:
                    raise ValueError("codebook must be a string or Path")

            # Use Workspace to get ROIs instead of manual directory parsing
            try:
                from fishtools.io.workspace import Workspace

                # Convert path to Path object for type safety
                workspace_path = Path(str(kwargs["path"]))
                workspace = Workspace(workspace_path)

                # Filter ROIs based on pattern and codebook requirements
                if include_codebook:
                    # Find directories matching the full pattern including codebook
                    matching_dirs = list(workspace_path.glob(current_look_for))
                    if split_codebook:
                        rois = {
                            p.name.split("--")[1].split("+")[0]
                            for p in matching_dirs
                            if "--" in p.name and "+" in p.name.split("--")[1]
                        }
                    else:
                        rois = {p.name.split("--")[1] for p in matching_dirs if "--" in p.name}
                else:
                    # Use Workspace.rois for non-codebook patterns
                    if look_for == "registered--*":
                        # Standard case - use all workspace ROIs
                        rois = set(workspace.rois)
                    else:
                        # Custom pattern - fall back to glob
                        matching_dirs = list(workspace_path.glob(current_look_for))
                        rois = {
                            p.name.split("--")[1].split("+")[0] if split_codebook else p.name.split("--")[1]
                            for p in matching_dirs
                            if "--" in p.name
                        }

                if not rois:
                    logger.warning(f"No ROIs found matching pattern '{current_look_for}' in {kwargs['path']}")
                    return None

                # Process each ROI with proper kwargs isolation
                for roi in sorted(rois):  # Sort for consistent ordering
                    # Skip empty ROI names which indicate malformed directory names
                    if not roi.strip():
                        logger.warning("Skipping empty/whitespace ROI name from malformed directory")
                        continue

                    roi_kwargs = {**kwargs, "roi": roi}
                    logger.info(f"Batching {roi}")
                    try:
                        func(*args, **roi_kwargs)  # type: ignore[misc]
                    except Exception as e:
                        # Emit full traceback to logs to aid root-cause analysis
                        logger.opt(exception=True).error("Exception while processing ROI '{roi}'", roi=roi)
                        # Re-raise with exception type for clearer CLI surface
                        raise RuntimeError(f"Failed processing ROI '{roi}': {type(e).__name__}: {e}") from e

                return None

            except Exception as e:
                logger.error(f"Error during batch processing: {e}")
                raise

        return inner

    return decorator


def noglobal(fn: Callable[P, R]) -> Callable[P, R]:
    """
    Creates a copy of the input function `fn` that runs in a restricted environment.

    This new function will not have access to non-callable global variables
    from the module where `fn` was defined. It will, however, have access to:
    1.  Modules imported into `fn`'s original module.
    2.  Other functions and classes defined in or imported into `fn`'s original module.

    This is to ensure that a function doesn't unintentionally rely on or
    change global state, making its behavior more predictable and self-contained.

    Example:
    --------
    >>> my_global_var = 10
    >>> def my_func():
    ...     print(f"Inside my_func: {my_global_var}")
    ...     return my_global_var * 2
    >>>
    >>> my_func()  # Regular call, can access my_global_var
    Inside my_func: 10
    20
    >>>
    >>> isolated_func = noglobal(my_func)
    >>> try:
    ...     isolated_func()  # Call through noglobal
    ... except NameError as e:
    ...     print(e)
    name 'my_global_var' is not defined
    """

    restricted_globals = {
        name: val
        for name, val in fn.__globals__.items()
        # Keep modules and any callable (functions, classes, etc.)
        if isinstance(val, types.ModuleType) or callable(val)
    }
    fn_restricted = types.FunctionType(
        fn.__code__,
        restricted_globals,
        name=fn.__name__,
        argdefs=fn.__defaults__,
        closure=fn.__closure__,  # Preserves the closure (for nested functions accessing nonlocals)
    )

    @functools.wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        return fn_restricted(*args, **kwargs)

    return wrapper


def create_rotation_matrix(angle_degrees: float) -> np.ndarray:
    angle_radians = np.deg2rad(angle_degrees)
    cos_angle = np.cos(angle_radians)
    sin_angle = np.sin(angle_radians)
    return np.array([[cos_angle, -sin_angle], [sin_angle, cos_angle]])


def make_lazy_getattr(
    module_globals: dict[str, Any],
    mapping: dict[str, tuple[str, str]],
    extras: Sequence[str] | None = None,
) -> tuple[Callable[[str], Any], Callable[[], list[str]], tuple[str, ...]]:
    """Create PEP 562-style lazy attribute loader for a module.

    This factory returns ``__getattr__``, ``__dir__``, and ``__all__`` suitable
    for assignment in a module to expose a set of names that are resolved only
    on first access and then cached in the module namespace.

    Args:
        module_globals: The target module's ``globals()`` dict. Used for caching
            resolved attributes and computing ``__dir__``.
        mapping: Name → (module, attribute) pairs to resolve lazily.
        extras: Additional exported names (e.g., constants defined eagerly) to
            include in ``__all__`` and completion results.

    Returns:
        (__getattr__, __dir__, __all__)

    Example:
        >>> __getattr__, __dir__, __all__ = make_lazy_getattr(
        ...     globals(),
        ...     {"Thing": ("pkg.sub", "Thing")},
        ...     extras=("CONSTANT",),
        ... )
    """

    lazy_names = dict(mapping)  # shallow copy for safety
    extra_set = set(extras or ())
    module_name = module_globals.get("__name__", "<module>")

    def __getattr__(name: str) -> Any:
        if name in lazy_names:
            mod_name, attr = lazy_names[name]
            try:
                module = importlib.import_module(mod_name)
            except Exception as e:  # provide context; do not swallow
                raise AttributeError(
                    f"{module_name}: failed to lazily import '{name}' from '{mod_name}': {e}"
                ) from e
            try:
                value = getattr(module, attr)
            except AttributeError as e:
                raise AttributeError(
                    f"{module_name}: module '{mod_name}' has no attribute '{attr}' for '{name}'"
                ) from e
            module_globals[name] = value  # cache
            return value
        raise AttributeError(f"module '{module_name}' has no attribute '{name}'")

    def __dir__() -> list[str]:
        return sorted(set(module_globals.keys()) | set(lazy_names.keys()) | extra_set)

    __all__ = tuple(sorted(set(lazy_names.keys()) | extra_set))

    return __getattr__, __dir__, __all__
