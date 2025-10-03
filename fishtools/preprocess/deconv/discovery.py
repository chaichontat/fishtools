from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

from loguru import logger

from fishtools.io.workspace import get_metadata


def infer_psf_step(tile: Path, *, default: int = 6) -> Tuple[int, bool]:
    """Infer PSF step from TIFF metadata waveform; fall back to `default`.

    Returns `(step, inferred)` where `inferred` indicates metadata success.
    """
    try:
        metadata = get_metadata(tile)
        waveform = json.loads(metadata["waveform"])  # type: ignore[index]
        step = int(waveform["params"]["step"] * 10)
        return step, True
    except Exception as exc:  # noqa: BLE001
        logger.debug(f"infer_psf_step: fallback to default due to {exc!r}")
        return default, False
