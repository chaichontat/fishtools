from __future__ import annotations

import pickle
from pathlib import Path
from typing import Sequence

import cupy as cp
import numpy as np
from loguru import logger


def load_basic_profiles(paths: Sequence[Path]) -> tuple[cp.ndarray, cp.ndarray]:
    """Load BaSiC darkfield/flatfield profiles from pickle payloads.

    Returns GPU arrays shaped (1, C, H, W) for darkfield and flatfield.
    """

    darks: list[np.ndarray] = []
    flats: list[np.ndarray] = []
    reference_shape: tuple[int, int] | None = None

    for pkl_path in paths:
        loaded = pickle.loads(pkl_path.read_bytes())  # type: ignore[name-defined]
        basic = loaded.get("basic") if isinstance(loaded, dict) else loaded
        assert basic
        if not hasattr(basic, "darkfield") or not hasattr(basic, "flatfield"):
            raise TypeError("BaSiC payload must expose 'darkfield' and 'flatfield'.")

        dark = np.asarray(basic.darkfield, dtype=np.float32)
        flat = np.asarray(basic.flatfield, dtype=np.float32)

        if dark.shape != flat.shape:
            raise ValueError(f"Dark/flat shape mismatch for {pkl_path}: {dark.shape} vs {flat.shape}.")
        if reference_shape is None:
            reference_shape = dark.shape
        elif dark.shape != reference_shape:
            raise ValueError("Inconsistent BaSiC profile geometry; all profiles must match.")
        if not np.all(np.isfinite(flat)) or np.any(flat <= 0):
            raise ValueError(f"Invalid flatfield values encountered in {pkl_path}.")
        if not np.all(np.isfinite(dark)):
            raise ValueError(f"Invalid darkfield values encountered in {pkl_path}.")
        darks.append(dark)
        flats.append(flat)

    if not darks:
        raise ValueError("No BaSiC profiles provided.")

    darkfield_np = np.stack(darks, axis=0)[np.newaxis, ...].astype(np.float32, copy=False)
    flatfield_np = np.stack(flats, axis=0)[np.newaxis, ...].astype(np.float32, copy=False)
    return cp.asarray(darkfield_np), cp.asarray(flatfield_np)


def resolve_basic_paths(
    workspace: Path,
    *,
    round_name: str,
    channels: Sequence[str],
    basic_name: str | None,
) -> list[Path]:
    paths: list[Path] = []
    for channel in channels:
        primary: Path | None = None
        # If the user provided a name explicitly, honor it (only that name).
        if basic_name:
            candidate = workspace / "basic" / f"{basic_name}-{channel}.pkl"
            if candidate.exists():
                primary = candidate
            else:
                raise FileNotFoundError(f"BaSiC profile not found: {candidate} (explicit --basic-name).")

        else:
            # Auto mode: prefer round-specific prefix, then fall back to 'all'.
            preferred = workspace / "basic" / f"{round_name}-{channel}.pkl"
            fallback_all = workspace / "basic" / f"all-{channel}.pkl"
            if preferred.exists():
                primary = preferred
            elif fallback_all.exists():
                primary = fallback_all
            else:
                raise FileNotFoundError(
                    "Missing BaSiC profile for channel "
                    f"{channel}: tried {preferred.name} then {fallback_all.name}."
                )

        paths.append(primary)
        logger.debug(f"Loaded BaSiC profile for {channel} from {primary}")
    return paths
