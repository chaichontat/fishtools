"""Compatibility re-exports for legacy imports.

Prefer importing from:
- fishtools.io.workspace (Workspace, OptimizePath, CorruptedTiffError, safe_imwrite)
- fishtools.utils.tiff (get_metadata, get_channels)
- fishtools.utils.fs (download, get_file_name, set_cwd)
"""

from __future__ import annotations

from fishtools.io.workspace import (
    Workspace,
    OptimizePath,
    CorruptedTiffError,
    safe_imwrite,
)
from fishtools.utils.tiff import get_metadata, get_channels
from fishtools.utils.fs import download, get_file_name, set_cwd

__all__ = [
    "Workspace",
    "OptimizePath",
    "CorruptedTiffError",
    "safe_imwrite",
    "get_metadata",
    "get_channels",
    "download",
    "get_file_name",
    "set_cwd",
]
