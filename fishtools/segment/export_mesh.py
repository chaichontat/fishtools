from __future__ import annotations

from pathlib import Path
import threading
import time

import numpy as np
import rich_click as click
import zarr
from loguru import logger

from fishtools.io.workspace import Workspace
from fishtools.utils.pretty_print import progress_bar
from fishtools.segmentation.mesh import (
    labels_zyx_to_polydata_pyvista,
    polydata_to_mesh,
    write_ply_binary_little_endian,
)


def _load_zarr_labels_zyx(
    arr: zarr.Array,
    *,
    downsample: int,
    show_progress_bar: bool,
) -> np.ndarray:
    shape = arr.shape
    if len(shape) == 2:
        vol = np.asarray(arr)[None, ::downsample, ::downsample]
        return vol.astype(np.int32, copy=False) if not np.issubdtype(vol.dtype, np.integer) else vol

    if len(shape) != 3:
        raise click.ClickException(f"Expected 2D/3D segmentation, got shape {shape}")

    z, y, x = shape
    chunk_z = max(1, int(arr.chunks[0]))
    chunk_z = min(chunk_z, 16)

    out_dtype = arr.dtype if np.issubdtype(arr.dtype, np.integer) else np.int32
    y_ds = (y + downsample - 1) // downsample
    x_ds = (x + downsample - 1) // downsample
    vol = np.empty((z, y_ds, x_ds), dtype=out_dtype)

    n_chunks = (z + chunk_z - 1) // chunk_z
    logger.info(f"[export-mesh] Loading Zarr volume in {n_chunks} slabs (chunk_z={chunk_z}).")

    if not show_progress_bar:
        for z0 in range(0, z, chunk_z):
            z1 = min(z0 + chunk_z, z)
            vol[z0:z1] = np.asarray(arr[z0:z1, ::downsample, ::downsample])
    else:
        with progress_bar(n_chunks) as advance:
            for z0 in range(0, z, chunk_z):
                z1 = min(z0 + chunk_z, z)
                vol[z0:z1] = np.asarray(arr[z0:z1, ::downsample, ::downsample])
                advance()

    return vol


def export_mesh_cmd(
    path: Path,
    roi: str | None,
    *,
    seg_codebook: str,
    segmentation_name: str,
    output: Path | None,
    labels: list[int] | None,
    spacing_zyx: tuple[float, float, float],
    origin_zyx: tuple[float, float, float],
    downsample: int,
    show_progress_bar: bool,
) -> None:
    ws = Workspace(path)

    if roi is None:
        candidate_rois = ws.rois
    else:
        try:
            candidate_rois = ws.resolve_rois([roi])
        except ValueError as exc:
            raise click.ClickException(str(exc)) from exc

    rois: list[str] = []
    for r in candidate_rois:
        seg_path = ws.stitch(r, seg_codebook) / segmentation_name
        if seg_path.exists():
            rois.append(r)

    if not rois:
        roi_hint = roi if roi is not None else "<auto>"
        raise click.ClickException(
            f"No segmentation zarr found for roi={roi_hint} seg_codebook={seg_codebook} name={segmentation_name}."
        )

    if output is not None and len(rois) != 1:
        raise click.ClickException("--output requires a single ROI (or specify ROI explicitly).")

    for r in rois:
        seg_zarr_path = ws.stitch(r, seg_codebook) / segmentation_name
        arr = zarr.open_array(str(seg_zarr_path), mode="r")
        shape = arr.shape
        est_bytes = int(np.prod(shape)) * int(arr.dtype.itemsize)
        logger.info(
            f"[export-mesh] ROI {r}: segmentation={seg_zarr_path} shape={shape} dtype={arr.dtype} "
            f"chunks={arr.chunks} ~{est_bytes / (1024**3):.2f} GiB"
        )

        t_load0 = time.perf_counter()
        vol = _load_zarr_labels_zyx(arr, downsample=downsample, show_progress_bar=show_progress_bar)
        t_load1 = time.perf_counter()
        logger.info(
            f"[export-mesh] ROI {r}: loaded volume shape={vol.shape} dtype={vol.dtype} "
            f"in {t_load1 - t_load0:.1f}s"
        )

        if labels is not None:
            logger.info(f"[export-mesh] ROI {r}: applying label filter (n={len(labels)}).")
            keep = np.isin(vol, np.asarray(labels, dtype=vol.dtype))
            vol = vol.copy()
            vol[~keep] = 0

        nnz = int(np.count_nonzero(vol))
        effective_spacing_zyx = (spacing_zyx[0], spacing_zyx[1] * downsample, spacing_zyx[2] * downsample)
        logger.info(
            f"[export-mesh] ROI {r}: meshing start (nonzero_voxels={nnz}, downsample={downsample}, "
            f"spacing_zyx={effective_spacing_zyx}, origin_zyx={origin_zyx})."
        )

        stop_evt = threading.Event()
        t_mesh0 = time.perf_counter()
        heartbeat_s = 30.0

        def _heartbeat() -> None:
            while not stop_evt.wait(heartbeat_s):
                logger.info(f"[export-mesh] ROI {r}: still meshing... elapsed={time.perf_counter() - t_mesh0:.0f}s")

        hb = threading.Thread(target=_heartbeat, daemon=True)
        hb.start()
        try:
            poly = labels_zyx_to_polydata_pyvista(vol, spacing_zyx=effective_spacing_zyx, origin_zyx=origin_zyx)
        finally:
            stop_evt.set()
            hb.join()
        logger.info(f"[export-mesh] ROI {r}: meshing done in {time.perf_counter() - t_mesh0:.1f}s.")

        mesh = polydata_to_mesh(poly)
        if mesh.vertices_xyz.size == 0 or mesh.faces.size == 0:
            logger.warning(f"[export-mesh] ROI {r}: mesh was empty; nothing to export.")
            continue

        if output is None:
            out_path = seg_zarr_path / "mesh.ply"
        else:
            out_path = output
        out_vtp = out_path.with_suffix(".vtp")

        write_ply_binary_little_endian(out_path, mesh)
        poly.save(out_vtp, binary=True)
        logger.info(
            f"[export-mesh] ROI {r}: wrote 1 mesh ({mesh.vertices_xyz.shape[0]} verts, {mesh.faces.shape[0]} faces) "
            f"to {out_path} and {out_vtp}"
        )
