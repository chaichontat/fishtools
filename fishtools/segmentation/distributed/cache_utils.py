from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fishtools.preprocess.config import NumpyEncoder


def read_nonempty_cache(path: Path, blocksize: tuple[int, ...]) -> list[int] | None:
    if not path.exists():
        return None
    try:
        parsed = json.loads(path.read_text())
        if tuple(parsed.get("block_size", ())) != tuple(blocksize):
            return None
        return [int(i) for i in parsed["idxs"]]
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


def write_nonempty_cache(path: Path, blocksize: tuple[int, ...], idxs: list[int]) -> None:
    payload = {"idxs": idxs, "block_size": list(blocksize)}
    path.write_text(json.dumps(payload, indent=2))


def read_normalization_cache(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def write_normalization_cache(path: Path, normalization: dict[str, Any]) -> None:
    path.write_text(json.dumps(normalization, indent=2, cls=NumpyEncoder))
