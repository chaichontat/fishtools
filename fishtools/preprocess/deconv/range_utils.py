from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

import numpy as np
from loguru import logger

from fishtools.utils.io import get_metadata
from fishtools.utils.pretty_print import progress_bar

PROTEIN_PERC_MIN = 50.0
PROTEIN_PERC_SCALE = 50.0

DEFAULT_PERCENTILE_OVERRIDES: dict[str, tuple[float, float]] = {
    "edu": (5.0, 5.0),
    "brdu": (5.0, 5.0),
    "wga": (50.0, 50.0),
    "pi": (1.0, 1.0),
}


def get_percentiles_for_round(
    round_name: str,
    default_perc_min: float,
    default_perc_scale: float,
    max_rna_bit: int,
    override: dict[str, tuple[float, float]] | None = None,
) -> tuple[list[float], list[float]]:
    """Compute per-bit percentile overrides for a given round name."""

    percentile_mins: list[float] = []
    percentile_scales: list[float] = []
    override = override or {}

    bits = round_name.split("_")
    for bit in bits:
        if bit in override:
            min_val, scale_val = override[bit]
            logger.info(
                "Using override for bit '%s' -> (%s, %s) in round '%s'.",
                bit,
                min_val,
                scale_val,
                round_name,
            )
            percentile_mins.append(min_val)
            percentile_scales.append(scale_val)
            continue

        is_rna_bit = (bit.isdigit() and int(bit) <= max_rna_bit) or re.match(r"^b\d{3}$", bit)
        if not is_rna_bit:
            logger.info(
                "Bit '%s' in round '%s' treated as protein stain; using protein percentiles (%s, %s).",
                bit,
                round_name,
                PROTEIN_PERC_MIN,
                PROTEIN_PERC_SCALE,
            )
        percentile_mins.append(PROTEIN_PERC_MIN if not is_rna_bit else default_perc_min)
        percentile_scales.append(PROTEIN_PERC_SCALE if not is_rna_bit else default_perc_scale)

    return percentile_mins, percentile_scales


def compute_range_for_round(
    path: Path,
    round_name: str,
    *,
    perc_min: float | Iterable[float] = 0.1,
    perc_scale: float | Iterable[float] = 0.1,
) -> None:
    """Aggregate deconvolution metadata and persist global min/scale arrays."""

    files = sorted(path.glob(f"{round_name}*/*.tif"))
    n_c = len(round_name.split("_"))
    n = len(files)

    deconv_min = np.zeros((n, n_c))
    deconv_scale = np.zeros((n, n_c))
    logger.info("[%s] Found %s files", round_name, n)
    if not files:
        raise FileNotFoundError(f"No files found in {path}")

    with progress_bar(len(files)) as pbar:
        for idx, file in enumerate(files):
            try:
                meta = json.loads(Path(file).with_suffix(".deconv.json").read_text())
            except FileNotFoundError:
                meta = get_metadata(file)

            try:
                deconv_min[idx, :] = meta["deconv_min"]
                deconv_scale[idx, :] = meta["deconv_scale"]
            except KeyError as exc:  # pragma: no cover - validated upstream
                raise AttributeError("No deconv metadata found.") from exc
            pbar()

    logger.info(
        "[%s] Calculating percentiles for min=%s scale=%s",
        round_name,
        perc_min,
        perc_scale,
    )

    if isinstance(perc_min, float) and isinstance(perc_scale, float):
        m_glob = np.percentile(deconv_min, perc_min, axis=0)
        s_glob = np.percentile(deconv_scale, perc_scale, axis=0)
    else:
        perc_min = list(perc_min)  # type: ignore[arg-type]
        perc_scale = list(perc_scale)  # type: ignore[arg-type]
        if len(perc_min) != n_c or len(perc_scale) != n_c:
            raise ValueError("perc_min and perc_scale must match number of channels")
        m_glob = np.zeros(n_c)
        s_glob = np.zeros(n_c)
        for c in range(n_c):
            m_glob[c] = np.percentile(deconv_min[:, c], perc_min[c])
            s_glob[c] = np.percentile(deconv_scale[:, c], perc_scale[c])

    if np.any(m_glob == 0) or np.any(s_glob == 0):
        raise ValueError("Found a channel with min=0; invalid scaling.")

    scaling_dir = path / "deconv_scaling"
    scaling_dir.mkdir(exist_ok=True)
    np.savetxt(scaling_dir / f"{round_name}.txt", np.vstack([m_glob, s_glob]))
    logger.info("[%s] Saved scaling to %s", round_name, scaling_dir / f"{round_name}.txt")

