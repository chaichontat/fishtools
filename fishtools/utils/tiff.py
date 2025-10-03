from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from tifffile import TiffFile


def read_metadata_from_tif(tif: TiffFile) -> dict[str, Any]:
    """Read shaped or ImageJ metadata from an open TiffFile.

    Returns a plain dict; missing metadata yields an empty dict.
    """
    try:
        shaped = getattr(tif, "shaped_metadata", None)
        if shaped:
            return dict(shaped[0])
    except (IndexError, KeyError, TypeError):
        ...
    return dict(tif.imagej_metadata or {})


def normalize_channel_names(count: int, metadata: dict[str, Any]) -> list[str]:
    """Normalize channel names from metadata to a list of strings.

    Falls back to [f"channel_{i}"] when names are absent or incomplete.
    """
    if count <= 0:
        return []
    raw = metadata.get("key") or metadata.get("channel_names") or metadata.get("channels")
    names: Sequence[Any] | None
    if raw is None:
        names = None
    elif isinstance(raw, str):
        names = [raw]
    elif isinstance(raw, (list, tuple)):
        names = raw
    elif isinstance(raw, np.ndarray):
        names = raw.tolist()
    else:
        names = None
    if names is None:
        return [f"channel_{i}" for i in range(count)]
    return [str(names[i]) if i < len(names) else f"channel_{i}" for i in range(count)]


def compose_metadata(
    axes: str,
    channel_names: Sequence[str] | None,
    *,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compose a minimal TIFF metadata dict with axes and key.

    Adds optional extra fields (e.g., processing params) when provided.
    """
    metadata: dict[str, Any] = {"axes": axes}
    if channel_names:
        channel_names_list = [str(name) for name in channel_names]
        metadata["key"] = channel_names_list
    if extra:
        metadata.update(extra)
    return metadata

