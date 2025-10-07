from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
from tifffile import TiffFile


def _safe_dict_from_mapping(mapping: dict) -> dict:
    """Return a plain dict from a mapping, resilient to concurrent mutation.

    Some TIFF readers expose dict-like metadata structures that may be
    populated lazily and mutate during iteration when accessed from
    multiple threads. This helper attempts a safe copy.
    """
    try:
        return dict(mapping)
    except RuntimeError as e:
        if "dictionary changed size during iteration" in str(e):
            try:
                keys = list(mapping.keys())
                return {k: mapping.get(k) for k in keys}
            except Exception:
                return {}
        raise


def read_metadata_from_tif(tif: TiffFile) -> dict[str, Any]:
    """Read shaped or ImageJ metadata from an open TiffFile.

    Returns a plain dict; missing metadata yields an empty dict.
    """
    try:
        shaped = getattr(tif, "shaped_metadata", None)
        if shaped:
            return _safe_dict_from_mapping(shaped[0])
    except (IndexError, KeyError, TypeError, RuntimeError):
        ...
    try:
        return _safe_dict_from_mapping(tif.imagej_metadata or {})
    except Exception:
        return {}


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


def get_metadata(file: "Path") -> dict[str, Any]:
    """Return shaped or ImageJ metadata from a TIFF path.

    Raises AttributeError when no metadata is found.
    """
    with TiffFile(file) as tif:
        try:
            meta = tif.shaped_metadata[0]  # type: ignore[index]
        except KeyError as exc:  # pragma: no cover - exercised by callers
            raise AttributeError("No metadata found.") from exc
    return meta


def get_channels(file: "Path") -> list[str]:
    meta = get_metadata(file)
    waveform = meta.get("waveform")
    if isinstance(waveform, str):
        import json

        waveform = json.loads(waveform)
    powers = (waveform or {}).get("params", {}).get("powers", {})
    return [name[-3:] for name in powers]
