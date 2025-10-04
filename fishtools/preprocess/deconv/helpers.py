from __future__ import annotations

import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import numpy.typing as npt
from loguru import logger
import line_profiler as line_profiler

__all__ = ["scale_deconv", "safe_delete_origin_dirs"]


@line_profiler.profile
def scale_deconv(
    img: npt.NDArray[np.float32],
    idx: int,
    *,
    global_deconv_scaling: npt.NDArray[np.float32],
    metadata: dict[str, Any],
    name: str | None = None,
    debug: bool = False,
) -> npt.NDArray[np.float32]:
    """Scale a deconvolved image back to uint16 dynamic range.

    Parameters
    ----------
    img
        Deconvolved tile data shaped (Z, Y, X) or (Z, C, Y, X) flattened along Z.
    idx
        Channel index corresponding to the tile.
    global_deconv_scaling
        Global min/scale array shaped (2, C) with rows (min, scale).
    metadata
        Original per-channel metadata containing `deconv_scale` and `deconv_min`.
    name
        Optional name for logging context.
    debug
        Emit additional debug logs when ``True``.
    """

    scaling = global_deconv_scaling.reshape((2, -1))
    m_val = scaling[0, idx]
    s_val = scaling[1, idx]

    scale_factor = np.float32(s_val / metadata["deconv_scale"][idx])
    offset = np.float32(s_val * (metadata["deconv_min"][idx] - m_val))
    scaled = scale_factor * img.astype(np.float32) + offset

    if debug:
        logger.debug("Deconvolution scaling=%s offset=%s", scale_factor, offset)

    if name and scaled.max() > 65535:
        logger.debug("Scaled image %s has max > 65535.", name)

    if np.all(scaled < 0):
        logger.warning("Scaled image %s has all negative values.", name or "<unnamed>")

    clipped = np.clip(scaled, 0, 65534).astype(np.float32, copy=False)
    return clipped


def safe_delete_origin_dirs(files: Iterable[Path], out: Path) -> None:
    """Delete source round folders only when outputs exist for every input file."""

    # Materialise once so callers can hand in generators without surprises.
    file_list = list(files)
    if not file_list:
        return

    grouped: dict[Path, list[Path]] = defaultdict(list)
    for file_path in file_list:
        grouped[file_path.parent].append(file_path)

    for src_dir, src_files in grouped.items():
        dst_dir = out / src_dir.name
        if not dst_dir.exists():
            logger.warning(
                "Skip delete for %s: destination %s does not exist.", src_dir, dst_dir
            )
            continue

        missing = [src for src in src_files if not (dst_dir / src.name).exists()]
        if missing:
            sample = missing[0].name
            logger.warning(
                "Skip delete for %s: %d inputs missing outputs (e.g., %s).",
                src_dir,
                len(missing),
                sample,
            )
            continue

        if "analysis" in src_dir.parts:
            logger.error("Refusing to delete source in analysis tree: %s", src_dir)
            continue

        logger.info("Deleting origin folder %s", src_dir)
        shutil.rmtree(src_dir)
