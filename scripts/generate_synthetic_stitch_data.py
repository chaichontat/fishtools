"""
Generate synthetic segmentation data for profiling the stitch/merge phase.

Creates realistic block-based segmentation with overlapping labels at boundaries
to exercise the merge pipeline.
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
import zarr
from scipy import ndimage


def generate_block_segmentation(
    shape: tuple[int, int, int],
    n_cells: int,
    seed: int,
) -> np.ndarray:
    """Generate random cell-like segmentation for a single block (fast)."""
    rng = np.random.default_rng(seed)
    
    # Fast approach: use random labels with morphological operations
    # Create sparse random seeds
    labels = np.zeros(shape, dtype=np.uint32)
    n_seeds = min(n_cells, np.prod(shape) // 100)
    
    # Random positions for cell centers
    positions = rng.integers(0, [shape[0], shape[1], shape[2]], size=(n_seeds, 3))
    for i, (z, y, x) in enumerate(positions, start=1):
        labels[z, y, x] = i
    
    # Expand seeds using distance transform (fast watershed-like)
    from scipy.ndimage import distance_transform_edt, label
    
    # Dilate seeds to fill space
    dist, indices = distance_transform_edt(labels == 0, return_indices=True)
    labels = labels[tuple(indices)]
    
    return labels


def block_faces(segmentation: np.ndarray) -> list[np.ndarray]:
    """Extract start/end faces along each axis for a segmented block.
    
    Matches the format expected by merge_utils.adjacent_faces:
    - Returns 6 faces for 3D: [z_start, z_end, y_start, y_end, x_start, x_end]
    - Each face is a single-pixel-thick slice
    """
    faces = []
    for ax in range(segmentation.ndim):
        slices = [slice(None)] * segmentation.ndim
        slices[ax] = slice(0, 1)
        faces.append(segmentation[tuple(slices)].copy())
        slices[ax] = slice(-1, None)
        faces.append(segmentation[tuple(slices)].copy())
    return faces


def generate_synthetic_data(
    output_dir: Path,
    grid_shape: tuple[int, int, int] = (4, 4, 2),
    block_shape: tuple[int, int, int] = (128, 128, 32),
    overlap: int = 16,
    cells_per_block: int = 50,
    seed: int = 42,
) -> None:
    """
    Generate synthetic segmentation data mimicking distributed_eval output.

    Parameters
    ----------
    output_dir : Path
        Directory to write segmentation_unstitched.zarr and intermediate_state.npz
    grid_shape : tuple
        Number of blocks in (z, y, x)
    block_shape : tuple
        Shape of each block (z, y, x)
    overlap : int
        Overlap between blocks
    cells_per_block : int
        Approximate number of cells per block
    seed : int
        Random seed for reproducibility
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    nz, ny, nx = grid_shape
    bz, by, bx = block_shape

    # Total volume shape (with overlaps)
    total_shape = (
        nz * (bz - overlap) + overlap,
        ny * (by - overlap) + overlap,
        nx * (bx - overlap) + overlap,
    )

    print(f"Grid: {grid_shape}, Block: {block_shape}, Overlap: {overlap}")
    print(f"Total volume: {total_shape}")
    print(f"Total blocks: {nz * ny * nx}")

    # Create zarr store
    zarr_path = output_dir / "segmentation_unstitched.zarr"
    if zarr_path.exists():
        shutil.rmtree(zarr_path)

    store = zarr.open(
        zarr_path,
        mode="w",
        shape=total_shape,
        chunks=block_shape,
        dtype=np.uint32,
    )

    rng = np.random.default_rng(seed)

    faces_list = []
    boxes_list = []
    box_ids_list = []
    non_empty_indices = []

    global_id_counter = 1

    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                block_idx = (iz, iy, ix)

                # Compute block position in global coordinates
                z_start = iz * (bz - overlap)
                y_start = iy * (by - overlap)
                x_start = ix * (bx - overlap)

                z_end = min(z_start + bz, total_shape[0])
                y_end = min(y_start + by, total_shape[1])
                x_end = min(x_start + bx, total_shape[2])

                actual_shape = (z_end - z_start, y_end - y_start, x_end - x_start)

                # Generate block segmentation
                block_labels = generate_block_segmentation(
                    actual_shape,
                    cells_per_block,
                    seed=rng.integers(0, 2**31),
                )

                # Relabel to global IDs
                unique_local = np.unique(block_labels)
                unique_local = unique_local[unique_local > 0]

                if len(unique_local) == 0:
                    continue

                # Create global ID mapping
                local_to_global = np.zeros(int(unique_local.max()) + 1, dtype=np.uint32)
                for local_id in unique_local:
                    local_to_global[local_id] = global_id_counter
                    global_id_counter += 1

                block_labels = local_to_global[block_labels]

                # Write to zarr
                store[z_start:z_end, y_start:y_end, x_start:x_end] = block_labels

                # Extract faces using the same format as merge_utils.block_faces
                faces_list.append(block_faces(block_labels))

                # Extract bounding boxes for each cell
                block_boxes = []
                block_box_ids = []

                for label_id in np.unique(block_labels):
                    if label_id == 0:
                        continue

                    slices = ndimage.find_objects(block_labels == label_id)
                    if slices and slices[0] is not None:
                        # Convert to global coordinates
                        sz, sy, sx = slices[0]
                        global_slices = (
                            slice(sz.start + z_start, sz.stop + z_start),
                            slice(sy.start + y_start, sy.stop + y_start),
                            slice(sx.start + x_start, sx.stop + x_start),
                        )
                        block_boxes.append(global_slices)
                        block_box_ids.append(label_id)

                boxes_list.append(block_boxes)
                box_ids_list.append(np.array(block_box_ids, dtype=np.uint32))
                non_empty_indices.append(block_idx)

    # Save intermediate state
    np.savez(
        output_dir / "intermediate_state.npz",
        faces=np.array(faces_list, dtype=object),
        boxes=np.array(boxes_list, dtype=object),
        box_ids=np.array(box_ids_list, dtype=object),
        non_empty_indices=np.array(non_empty_indices),
    )

    print(f"\nGenerated data:")
    print(f"  Total cells: {global_id_counter - 1}")
    print(f"  Non-empty blocks: {len(non_empty_indices)}")
    print(f"  Total faces: {sum(len(f) for f in faces_list)}")
    print(f"  Zarr path: {zarr_path}")
    print(f"  State file: {output_dir / 'intermediate_state.npz'}")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic stitch data for profiling")
    parser.add_argument("output_dir", type=Path, help="Output directory")
    parser.add_argument("--grid", type=str, default="4,4,2", help="Grid shape z,y,x (default: 4,4,2)")
    parser.add_argument("--block", type=str, default="128,128,32", help="Block shape z,y,x (default: 128,128,32)")
    parser.add_argument("--overlap", type=int, default=16, help="Overlap between blocks (default: 16)")
    parser.add_argument("--cells", type=int, default=50, help="Cells per block (default: 50)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    args = parser.parse_args()

    grid_shape = tuple(int(x) for x in args.grid.split(","))
    block_shape = tuple(int(x) for x in args.block.split(","))

    generate_synthetic_data(
        args.output_dir,
        grid_shape=grid_shape,
        block_shape=block_shape,
        overlap=args.overlap,
        cells_per_block=args.cells,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
