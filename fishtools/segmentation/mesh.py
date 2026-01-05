from __future__ import annotations

import os
import tempfile
from collections.abc import Iterator
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, TYPE_CHECKING

import numpy as np
from loguru import logger
from numpy.typing import NDArray

from fishtools.utils.pretty_print import progress_bar

if TYPE_CHECKING:
    import pyvista as pv


@dataclass(frozen=True, slots=True)
class Mesh:
    vertices_xyz: NDArray[np.float32]
    faces: NDArray[np.int32]


@dataclass(frozen=True, slots=True)
class MeshResult:
    label: int
    mesh: Mesh | None
    error: str | None = None


def labels_zyx_to_polydata_pyvista(
    labels_zyx: NDArray[np.integer],
    *,
    spacing_zyx: tuple[float, float, float] = (2.0, 1.0, 1.0),
    origin_zyx: tuple[float, float, float] = (0.0, 0.0, 0.0),
    background_value: int = 0,
    boundary_style: str = "external",
    output_mesh_type: str = "triangles",
    smoothing: bool = False,
    smoothing_iterations: int = 16,
    smoothing_relaxation: float = 0.5,
) -> "pv.PolyData":
    """Create a single mesh (PolyData) from a Z,Y,X label volume (cell-aligned)."""
    import pyvista as pv

    labels_np = np.asarray(labels_zyx)
    if labels_np.ndim != 3:
        raise ValueError(f"Expected (Z,Y,X) labels, got shape {labels_np.shape}")
    if not np.issubdtype(labels_np.dtype, np.integer):
        labels_np = labels_np.astype(np.int32, copy=False)

    z, y, x = labels_np.shape
    spacing_xyz = (spacing_zyx[2], spacing_zyx[1], spacing_zyx[0])
    origin_xyz = (origin_zyx[2], origin_zyx[1], origin_zyx[0])

    grid = pv.ImageData(dimensions=(x + 1, y + 1, z + 1))
    grid.spacing = spacing_xyz
    grid.origin = origin_xyz
    grid.cell_data["labels"] = labels_np.ravel(order="C")

    return grid.contour_labels(
        scalars="labels",
        boundary_style=boundary_style,
        background_value=background_value,
        pad_background=True,
        output_mesh_type=output_mesh_type,
        orient_faces=True,
        smoothing=smoothing,
        smoothing_iterations=smoothing_iterations,
        smoothing_relaxation=smoothing_relaxation,
    )


def polydata_to_mesh(poly: "pv.PolyData") -> Mesh:
    points = np.asarray(poly.points, dtype=np.float32)
    if points.size == 0 or poly.faces.size == 0:
        return Mesh(vertices_xyz=np.empty((0, 3), dtype=np.float32), faces=np.empty((0, 3), dtype=np.int32))

    faces_raw = np.asarray(poly.faces)
    faces = faces_raw.reshape(-1, 4)
    if not np.all(faces[:, 0] == 3):
        raise ValueError("Expected triangular faces from PyVista output.")
    faces = faces[:, 1:].astype(np.int32, copy=False)

    return Mesh(vertices_xyz=points, faces=faces)


def _default_max_workers() -> int:
    cpu_count = os.cpu_count() or 1
    return max(1, cpu_count - 4)


def _generate_mesh_worker(
    task: tuple[int, tuple[slice, slice, slice], int],
    relabeled: NDArray[np.integer],
    spacing_zyx: tuple[float, float, float],
    origin_zyx: tuple[float, float, float],
) -> MeshResult:
    old_label, obj, new_label = task

    mask = (relabeled[obj] == new_label).astype(np.uint8, copy=False)
    padded = np.pad(mask, 1, mode="constant", constant_values=0)

    from skimage.measure import marching_cubes

    try:
        verts_zyx, faces, _normals, _values = marching_cubes(padded, level=0.5, spacing=spacing_zyx)
    except (ValueError, RuntimeError) as exc:
        return MeshResult(label=old_label, mesh=None, error=f"{type(exc).__name__}: {exc}")

    # Offset: crop start minus 1 voxel padding, then convert to physical coordinates.
    z0, y0, x0 = (int(s.start) for s in obj)
    offset_zyx = np.array(
        [
            origin_zyx[0] + (z0 - 1) * spacing_zyx[0],
            origin_zyx[1] + (y0 - 1) * spacing_zyx[1],
            origin_zyx[2] + (x0 - 1) * spacing_zyx[2],
        ],
        dtype=np.float32,
    )

    verts_zyx = np.asarray(verts_zyx, dtype=np.float32)
    verts_zyx += offset_zyx
    verts_xyz = verts_zyx[:, [2, 1, 0]]
    return MeshResult(label=old_label, mesh=Mesh(vertices_xyz=verts_xyz, faces=np.asarray(faces, dtype=np.int32)))


def label_volume_to_meshes_parallel(
    labels_zyx: NDArray[np.integer],
    *,
    spacing_xyz: tuple[float, float, float] = (1.0, 1.0, 1.0),
    origin_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
    label_ids: list[int] | None = None,
    max_workers: int | None = None,
) -> Iterator[MeshResult]:
    """Convert a 3D integer label volume (Z,Y,X) into per-label surface meshes in parallel.

    Yields results as they complete.
    """
    if labels_zyx.ndim != 3:
        raise ValueError(f"Expected labels volume with shape (Z, Y, X), got {labels_zyx.shape}")

    labels_np = np.asarray(labels_zyx)
    if not np.issubdtype(labels_np.dtype, np.integer):
        raise ValueError(f"Expected integer labels, got dtype {labels_np.dtype}")

    sx, sy, sz = spacing_xyz
    ox, oy, oz = origin_xyz
    if sx <= 0 or sy <= 0 or sz <= 0:
        raise ValueError("spacing_xyz values must be > 0.")

    spacing_zyx = (sz, sy, sx)
    origin_zyx = (oz, oy, ox)

    if label_ids is None:
        logger.info("[mesh] Collecting unique labels (can be slow for large volumes).")
        requested_set: set[int] = set()
        z_count = int(labels_np.shape[0])
        chunk_size = 5
        n_chunks = (z_count + chunk_size - 1) // chunk_size
        with progress_bar(n_chunks) as advance:
            for z0 in range(0, z_count, chunk_size):
                z1 = min(z0 + chunk_size, z_count)
                requested_set.update(map(int, np.unique(labels_np[z0:z1])))
                advance()
        requested_set.discard(0)
        requested = sorted(requested_set)
        logger.info(f"[mesh] Found {len(requested)} unique labels (excluding 0).")
    else:
        requested = [x for x in label_ids if x != 0]

    if not requested:
        return iter(())

    from skimage.segmentation import relabel_sequential
    from scipy import ndimage

    logger.info("[mesh] Relabeling to sequential IDs.")
    relabeled, _forward, inverse = relabel_sequential(labels_np)
    inverse = np.asarray(inverse)
    logger.info("[mesh] Finding object bounds for relabeled volume.")
    objects = ndimage.find_objects(relabeled)

    requested_set = set(requested)
    tasks: list[tuple[int, tuple[slice, slice, slice], int]] = []
    for new_label, obj in enumerate(objects, start=1):
        if obj is None:
            continue
        old_label = int(inverse[new_label])
        if old_label == 0 or old_label not in requested_set:
            continue
        tasks.append((old_label, obj, new_label))

    if not tasks:
        return iter(())

    if max_workers is None:
        max_workers = _default_max_workers()
    if max_workers <= 0:
        raise ValueError("max_workers must be > 0.")

    logger.info(
        f"[mesh] Prepared {len(tasks)} tasks from {len(requested)} labels "
        f"(workers={max_workers}, requested={'all' if label_ids is None else len(requested)})"
    )

    def _iter_results() -> Iterator[MeshResult]:
        tasks_iter = iter(tasks)
        in_flight: set[Future[MeshResult]] = set()
        with progress_bar(len(tasks)) as advance, ThreadPoolExecutor(max_workers=max_workers) as executor:
            limit = max_workers * 4
            for _ in range(min(limit, len(tasks))):
                try:
                    task = next(tasks_iter)
                except StopIteration:
                    break
                in_flight.add(
                    executor.submit(
                        _generate_mesh_worker,
                        task,
                        relabeled,
                        spacing_zyx,
                        origin_zyx,
                    )
                )

            while in_flight:
                done, in_flight = wait(in_flight, return_when=FIRST_COMPLETED)
                for fut in done:
                    advance()
                    yield fut.result()
                    try:
                        task = next(tasks_iter)
                    except StopIteration:
                        continue
                    in_flight.add(
                        executor.submit(
                            _generate_mesh_worker,
                            task,
                            relabeled,
                            spacing_zyx,
                            origin_zyx,
                        )
                    )

    return _iter_results()


def _write_face_records(
    f: BinaryIO,
    faces: NDArray[np.int32],
    *,
    vertex_offset: int,
    chunk_faces: int = 250_000,
) -> None:
    face_dtype = np.dtype([("n", "u1"), ("v1", "<i4"), ("v2", "<i4"), ("v3", "<i4")])
    n_faces = int(faces.shape[0])
    for start in range(0, n_faces, chunk_faces):
        end = min(start + chunk_faces, n_faces)
        face_chunk = faces[start:end].astype("<i4", copy=False)
        if vertex_offset:
            face_chunk = face_chunk + vertex_offset
        face_data = np.empty((face_chunk.shape[0],), dtype=face_dtype)
        face_data["n"] = 3
        face_data["v1"] = face_chunk[:, 0]
        face_data["v2"] = face_chunk[:, 1]
        face_data["v3"] = face_chunk[:, 2]
        face_data.tofile(f)


def combine_and_write_parallel(
    labels_zyx: NDArray[np.integer],
    output_path: Path,
    *,
    spacing_xyz: tuple[float, float, float] = (1.0, 1.0, 1.0),
    origin_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
    label_ids: list[int] | None = None,
    max_workers: int | None = None,
    batch_size: int = 100_000,
) -> tuple[int, int, int, int]:
    """Generate meshes in parallel and write one combined PLY.

    Returns (succeeded, failed, vertex_count, face_count).
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    effective_workers = max_workers if max_workers is not None else _default_max_workers()
    logger.info(
        f"[mesh] Parallel meshing start: volume_shape={labels_zyx.shape} "
        f"labels={'all' if label_ids is None else len(label_ids)} workers={effective_workers} "
        f"batch_size={batch_size} output={output_path}"
    )

    succeeded = 0
    failed = 0

    with tempfile.TemporaryDirectory(dir=output_path.parent, prefix=f".{output_path.name}.batches.") as tmpdir:
        tmp_dir = Path(tmpdir)
        batch_meshes: dict[int, Mesh] = {}
        batch_files: list[tuple[Path, Path, int, int]] = []
        batch_idx = 0

        def _flush_batch() -> None:
            nonlocal batch_idx
            if not batch_meshes:
                return
            batch_meshes_sorted = dict(sorted(batch_meshes.items()))
            combined = combine_meshes(batch_meshes_sorted)
            v_path = tmp_dir / f"batch_{batch_idx:06d}_vertices.npy"
            f_path = tmp_dir / f"batch_{batch_idx:06d}_faces.npy"
            np.save(v_path, combined.vertices_xyz.astype("<f4", copy=False), allow_pickle=False)
            np.save(f_path, combined.faces.astype("<i4", copy=False), allow_pickle=False)
            batch_files.append((v_path, f_path, int(combined.vertices_xyz.shape[0]), int(combined.faces.shape[0])))
            batch_meshes.clear()
            logger.info(
                f"[mesh] Wrote batch {batch_idx} ({len(batch_meshes_sorted)} labels, "
                f"{combined.vertices_xyz.shape[0]} verts, {combined.faces.shape[0]} faces)"
            )
            batch_idx += 1

        for res in label_volume_to_meshes_parallel(
            labels_zyx,
            spacing_xyz=spacing_xyz,
            origin_xyz=origin_xyz,
            label_ids=label_ids,
            max_workers=max_workers,
        ):
            if res.mesh is None:
                failed += 1
                continue
            succeeded += 1
            batch_meshes[res.label] = res.mesh
            if len(batch_meshes) >= batch_size:
                _flush_batch()

        _flush_batch()

        vertex_count = sum(vn for _vp, _fp, vn, _fn in batch_files)
        face_count = sum(fn for _vp, _fp, _vn, fn in batch_files)
        logger.info(f"[mesh] Writing PLY header: vertices={vertex_count} faces={face_count}")

        header = (
            "\n".join(
                [
                    "ply",
                    "format binary_little_endian 1.0",
                    f"element vertex {vertex_count}",
                    "property float x",
                    "property float y",
                    "property float z",
                    f"element face {face_count}",
                    "property list uchar int vertex_indices",
                    "end_header",
                ]
            )
            + "\n"
        ).encode("ascii")

        with output_path.open("wb") as f:
            f.write(header)

            face_pass: list[tuple[Path, int]] = []
            v_offset = 0
            for v_path, f_path, v_n, _f_n in batch_files:
                verts = np.load(v_path, mmap_mode="r")
                verts.astype("<f4", copy=False).tofile(f)
                face_pass.append((f_path, v_offset))
                v_offset += v_n

            for f_path, v_off in face_pass:
                faces = np.load(f_path, mmap_mode="r")
                _write_face_records(f, faces, vertex_offset=v_off)

    logger.info(
        f"[mesh] Parallel meshing complete: succeeded={succeeded} failed={failed} "
        f"vertices={vertex_count} faces={face_count}"
    )
    return (succeeded, failed, vertex_count, face_count)


def write_ply_binary_little_endian(path: Path, mesh: Mesh) -> None:
    """Write a triangle mesh to PLY (binary_little_endian 1.0).

    The output stores vertices as float32 XYZ and faces as list(uchar,int32[3]).
    """
    vertices = mesh.vertices_xyz.astype("<f4", copy=False)
    faces = mesh.faces.astype("<i4", copy=False)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"Expected vertices array with shape (N, 3), got {vertices.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"Expected faces array with shape (M, 3), got {faces.shape}")
    if faces.size and (faces.min() < 0 or faces.max() >= vertices.shape[0]):
        raise ValueError("Faces contain vertex indices out of range.")

    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "\n".join(
            [
                "ply",
                "format binary_little_endian 1.0",
                f"element vertex {vertices.shape[0]}",
                "property float x",
                "property float y",
                "property float z",
                f"element face {faces.shape[0]}",
                "property list uchar int vertex_indices",
                "end_header",
            ]
        )
        + "\n"
    ).encode("ascii")

    face_dtype = np.dtype([("n", "u1"), ("v1", "<i4"), ("v2", "<i4"), ("v3", "<i4")])
    face_data = np.empty((faces.shape[0],), dtype=face_dtype)
    face_data["n"] = 3
    face_data["v1"] = faces[:, 0]
    face_data["v2"] = faces[:, 1]
    face_data["v3"] = faces[:, 2]

    with path.open("wb") as f:
        f.write(header)
        vertices.tofile(f)
        face_data.tofile(f)


def combine_meshes(meshes_by_label: dict[int, Mesh]) -> Mesh:
    """Concatenate multiple triangle meshes into a single mesh."""
    if not meshes_by_label:
        return Mesh(vertices_xyz=np.empty((0, 3), dtype=np.float32), faces=np.empty((0, 3), dtype=np.int32))

    vertices_list: list[NDArray[np.float32]] = []
    faces_list: list[NDArray[np.int32]] = []
    v_offset = 0
    for mesh in meshes_by_label.values():
        vertices = mesh.vertices_xyz
        faces = mesh.faces
        if vertices.size == 0 or faces.size == 0:
            continue
        vertices_list.append(vertices)
        faces_list.append(faces + v_offset)
        v_offset += vertices.shape[0]

    if not vertices_list:
        return Mesh(vertices_xyz=np.empty((0, 3), dtype=np.float32), faces=np.empty((0, 3), dtype=np.int32))

    return Mesh(vertices_xyz=np.vstack(vertices_list), faces=np.vstack(faces_list))


def label_volume_to_meshes(
    labels_zyx: NDArray[np.integer],
    *,
    spacing_xyz: tuple[float, float, float] = (1.0, 1.0, 1.0),
    origin_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
    label_ids: list[int] | None = None,
) -> dict[int, Mesh]:
    """Convert a 3D integer label volume (Z,Y,X) into per-label surface meshes.

    Notes
    -----
    - Uses marching cubes at level=0.5 on a padded binary mask per label.
    - Outputs vertices in XYZ coordinate order (Blender-friendly).
    - ``spacing_xyz`` and ``origin_xyz`` are interpreted in XYZ order; they are
      converted internally to the input array's Z,Y,X axis order.
    """
    if labels_zyx.ndim != 3:
        raise ValueError(f"Expected labels volume with shape (Z, Y, X), got {labels_zyx.shape}")

    labels_np = np.asarray(labels_zyx)
    if not np.issubdtype(labels_np.dtype, np.integer):
        raise ValueError(f"Expected integer labels, got dtype {labels_np.dtype}")

    sx, sy, sz = spacing_xyz
    ox, oy, oz = origin_xyz
    if sx <= 0 or sy <= 0 or sz <= 0:
        raise ValueError("spacing_xyz values must be > 0.")

    spacing_zyx = (sz, sy, sx)
    origin_zyx = (oz, oy, ox)

    if label_ids is None:
        requested_set: set[int] = set()
        z_count = int(labels_np.shape[0])
        chunk_size = 5
        n_chunks = (z_count + chunk_size - 1) // chunk_size
        with progress_bar(n_chunks) as advance:
            for z0 in range(0, z_count, chunk_size):
                z1 = min(z0 + chunk_size, z_count)
                requested_set.update(map(int, np.unique(labels_np[z0:z1])))
                advance()
        requested_set.discard(0)
        requested = sorted(requested_set)
    else:
        requested = [x for x in label_ids if x != 0]

    if not requested:
        return {}

    # Re-label to sequential integers so scipy.ndimage.find_objects is safe even
    # when the original labels are sparse/high-valued.
    from skimage.segmentation import relabel_sequential
    from scipy import ndimage

    relabeled, _forward, inverse = relabel_sequential(labels_np)
    objects = ndimage.find_objects(relabeled)

    inverse = np.asarray(inverse)

    requested_set = set(requested)
    meshes: dict[int, Mesh] = {}

    for new_label, obj in enumerate(objects, start=1):
        if obj is None:
            continue
        old_label = int(inverse[new_label])
        if old_label == 0 or old_label not in requested_set:
            continue

        mask = (relabeled[obj] == new_label).astype(np.uint8, copy=False)
        padded = np.pad(mask, 1, mode="constant", constant_values=0)

        from skimage.measure import marching_cubes

        verts_zyx, faces, _normals, _values = marching_cubes(padded, level=0.5, spacing=spacing_zyx)

        # Offset: crop start minus 1 voxel padding, then convert to physical coordinates.
        z0, y0, x0 = (int(s.start) for s in obj)
        offset_zyx = np.array(
            [
                origin_zyx[0] + (z0 - 1) * spacing_zyx[0],
                origin_zyx[1] + (y0 - 1) * spacing_zyx[1],
                origin_zyx[2] + (x0 - 1) * spacing_zyx[2],
            ],
            dtype=np.float32,
        )
        verts_zyx = np.asarray(verts_zyx, dtype=np.float32)
        verts_zyx += offset_zyx
        verts_xyz = verts_zyx[:, [2, 1, 0]]

        meshes[old_label] = Mesh(vertices_xyz=verts_xyz, faces=np.asarray(faces, dtype=np.int32))

    return meshes
