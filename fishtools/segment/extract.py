from __future__ import annotations

from pathlib import Path

import click
from loguru import logger

from fishtools.io.workspace import Workspace
from fishtools.segment.extract_core import (
    _is_zarr_path,
    normalize_numeric_options,
    run_single_file_extract,
    run_workspace_extract,
)
from fishtools.utils.logging import configure_cli_logging, setup_cli_logging

# ---------- Commands ----------


def cmd_extract(
    mode: str,
    path: Path,
    *,
    roi: str | None = None,
    codebook: str,
    out: Path | None = None,
    dz: int = 1,
    n: int = 50,
    anisotropy: int = 4,
    channels: str | None = None,
    crop: int = 0,
    threads: int = 8,
    upscale: float | None = None,
    seed: int | None = None,
    every: int = 1,
    max_from: str | None = None,
    use_zarr: bool = False,
    masks: Path | None = None,
    enrich_boundaries: Path | None = None,
    enable_enrich_boundaries: bool = True,
    roi_points: Path | None = None,
) -> None:
    mode = mode.lower().strip()
    if mode not in {"z", "ortho"}:
        raise click.BadParameter("Mode must be 'z' or 'ortho'.")

    setup_cli_logging(
        path,
        component="segment.extract",
        file=f"segment-extract-{mode}",
        extra={
            "mode": mode,
            "roi": roi or "all",
            "codebook": codebook,
            "threads": threads,
        },
    )

    upscale_val = normalize_numeric_options(
        mode=mode,
        dz=dz,
        anisotropy=anisotropy,
        upscale=upscale,
        use_zarr=use_zarr,
        has_max_from=max_from is not None,
        ortho_anisotropy_default=4,
    )

    ws = Workspace(path)

    if roi is not None:
        rois = ws.resolve_rois([roi])
    else:
        if not ws.rois:
            raise FileNotFoundError("Workspace contains no ROIs.")
        rois = ws.resolve_rois(ws.rois)

    run_workspace_extract(
        ws=ws,
        mode=mode,
        codebook=codebook,
        rois=rois,
        out=out,
        dz=dz,
        n=n,
        anisotropy=anisotropy,
        channels=channels,
        crop=crop,
        threads=threads,
        upscale=upscale_val,
        seed=seed,
        every=every,
        max_from=max_from,
        use_zarr=use_zarr,
        masks=masks,
        enrich_boundaries=enrich_boundaries,
        enable_enrich_boundaries=enable_enrich_boundaries,
        roi_points=roi_points,
    )


def cmd_extract_single(
    mode: str,
    registered: Path,
    *,
    out: Path | None = None,
    dz: int = 1,
    n: int = 50,
    anisotropy: int = 6,
    channels: str | None = None,
    crop: int = 0,
    threads: int = 8,
    upscale: float | None = None,
    seed: int | None = None,
    max_from: Path | None = None,
    label: str | None = None,
    masks: Path | None = None,
    enrich_boundaries: Path | None = None,
) -> None:
    mode = mode.lower().strip()
    if mode not in {"z", "ortho"}:
        raise click.BadParameter("Mode must be 'z' or 'ortho'.")

    upscale_val = normalize_numeric_options(
        mode=mode,
        dz=dz,
        anisotropy=anisotropy,
        upscale=upscale,
        use_zarr=False,
        has_max_from=max_from is not None,
        ortho_anisotropy_default=6,
    )

    configure_cli_logging(
        workspace=None,
        component="segment.extract-single",
        extra={"mode": mode},
    )

    registered = registered.resolve()

    if _is_zarr_path(registered):
        raise click.BadParameter(
            "Zarr input is not supported for extract-single. "
            "Use TIFF files or the 'segment extract' command for Zarr inputs."
        )

    if registered.is_dir():
        raise click.BadParameter("Registered input must be a TIFF file.")

    label_value = label or (registered.stem if registered.suffix else registered.name)

    out_dir = out if out is not None else registered.parent / "segment_extract"
    if out_dir.is_file():
        raise click.BadParameter("--out must point to a directory, not a file.")
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"[{label_value}] Input: {registered}")
    logger.info(f"[{label_value}] Output: {out_dir}")
    logger.info(f"[{label_value}] Upscale factor: {upscale_val}")

    # In single-file mode, Z extraction should emit a single crop per Z-plane.
    # Override any user-provided --n to 1 for 'z' mode to avoid multiple crops.
    n_effective = 1 if mode == "z" else n

    max_from_path: Path | None = None
    if max_from is not None:
        max_from_path = max_from.resolve()

    run_single_file_extract(
        mode=mode,
        registered=registered,
        out=out_dir,
        dz=dz,
        n=n_effective,
        anisotropy=anisotropy,
        channels=channels,
        crop=crop,
        threads=threads,
        upscale=upscale_val,
        seed=seed,
        max_from_path=max_from_path,
        label=label_value,
        masks=masks,
        enrich_boundaries=enrich_boundaries,
    )
