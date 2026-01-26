from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import zarr

from fishtools.utils.zarr_utils import (
    DEFAULT_TARGET_SHARD_SIZE_BYTES,
    choose_shard_shape,
    default_zarr_codecs,
    label_zarr_codecs,
)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="rewrite_zarr_sharded",
        description=(
            "Rewrite an existing Zarr store to use large shards (fewer objects) and our default compression."
        ),
    )
    p.add_argument("input", type=Path, help="Input .zarr path (array or group)")
    p.add_argument("output", type=Path, help="Output .zarr path")
    p.add_argument("--overwrite", action="store_true", help="Overwrite output if it exists")
    p.add_argument(
        "--target-shard-mb",
        type=int,
        default=DEFAULT_TARGET_SHARD_SIZE_BYTES // (1024 * 1024),
        help="Target shard size in MB (default: 512)",
    )
    p.add_argument(
        "--chunk-yx",
        type=int,
        default=2048,
        help="Chunk size for Y/X axes (default: 2048)",
    )
    p.add_argument(
        "--verify",
        type=int,
        default=0,
        help="Verify N random chunk positions per array (0 disables)",
    )
    return p.parse_args(argv)


def _infer_axes(obj: zarr.Array | zarr.Group) -> str | None:
    axes = obj.attrs.get("axes")
    if isinstance(axes, str):
        return axes
    return None


def _forced_chunks(*, shape: tuple[int, ...], axes: str | None, chunk_yx: int) -> tuple[int, ...]:
    if axes is None or len(axes) != len(shape):
        # Conservative fallback: chunk the two largest dimensions (assumed Y/X),
        # keep all other axes chunked as 1.
        if len(shape) < 2:
            return tuple(1 for _ in shape)
        yx = sorted(range(len(shape)), key=lambda i: int(shape[i]), reverse=True)[:2]
        out = [1] * len(shape)
        for i in yx:
            out[i] = min(int(chunk_yx), int(shape[i]))
        return tuple(out)

    out = []
    for dim, ax in zip(shape, axes, strict=True):
        dim_i = int(dim)
        if ax in {"Y", "X"}:
            out.append(min(int(chunk_yx), dim_i))
        else:
            out.append(1)
    return tuple(out)


def _copy_attrs(dst: zarr.Array | zarr.Group, src: zarr.Array | zarr.Group) -> None:
    dst.attrs.clear()
    dst.attrs.update(dict(src.attrs))


def _ceildiv(a: int, b: int) -> int:
    return -(-a // b)


def _copy_by_shards(*, src: zarr.Array, dst: zarr.Array, verify: int) -> None:
    shard_shape_raw = getattr(dst, "shards", None)
    if shard_shape_raw is None:
        raise RuntimeError("Destination array does not expose shard shape; cannot do shard-safe copy.")
    shard_shape = tuple(int(s) for s in shard_shape_raw)

    if len(shard_shape) != len(src.shape):
        raise RuntimeError(f"Shard ndim={len(shard_shape)} does not match array ndim={len(src.shape)}")

    grid = tuple(_ceildiv(int(dim), int(ss)) for dim, ss in zip(src.shape, shard_shape, strict=True))
    total = int(np.prod(grid))

    for n, idx in enumerate(np.ndindex(*grid), start=1):
        slices: list[slice] = []
        for i, (dim, ss) in enumerate(zip(src.shape, shard_shape, strict=True)):
            start = int(idx[i]) * int(ss)
            stop = min(int(dim), start + int(ss))
            slices.append(slice(start, stop))

        block = src[tuple(slices)]
        dst[tuple(slices)] = block
        if (n % max(1, min(total, 25))) == 0 or n == total:
            print(f"[{dst.path or '/'}] shards {n}/{total}")

    _verify_samples(src=src, dst=dst, chunks=tuple(int(c) for c in dst.chunks), n=verify)


def _verify_samples(
    *,
    src: zarr.Array,
    dst: zarr.Array,
    chunks: tuple[int, ...],
    n: int,
) -> None:
    if n <= 0:
        return

    rng = np.random.default_rng(0)
    for _ in range(n):
        slices: list[slice] = []
        for dim, c in zip(src.shape, chunks, strict=True):
            if dim <= c:
                start = 0
            else:
                start = int(rng.integers(0, int(dim - c) + 1))
            slices.append(slice(start, start + int(c)))
        src_block = src[tuple(slices)]
        dst_block = dst[tuple(slices)]
        if src_block.dtype.kind == "f":
            ok = np.allclose(src_block, dst_block, equal_nan=True)
        else:
            ok = np.array_equal(src_block, dst_block)
        if not ok:
            raise RuntimeError(f"Verification failed for slices={slices}")


def _rewrite_array(
    *,
    src: zarr.Array,
    dst_group: zarr.Group,
    name: str,
    target_shard_size_bytes: int,
    chunk_yx: int,
    verify: int,
) -> None:
    axes = _infer_axes(src)
    shape = tuple(int(s) for s in src.shape)
    chunks = _forced_chunks(shape=shape, axes=axes, chunk_yx=chunk_yx)

    dtype = np.dtype(src.dtype)
    codecs = label_zarr_codecs(dtype) if dtype == np.dtype(np.uint32) else default_zarr_codecs(dtype)
    shard_shape = choose_shard_shape(
        shape=shape,
        chunks=chunks,
        dtype=dtype,
        target_shard_size_bytes=target_shard_size_bytes,
    )
    dst = dst_group.create_array(
        name,
        shape=shape,
        chunks=chunks,
        shards=shard_shape,
        dtype=src.dtype,
        serializer=codecs[0],
        compressors=tuple(codecs[1:]),
        overwrite=True,
    )
    _copy_attrs(dst, src)
    _copy_by_shards(src=src, dst=dst, verify=verify)


def _rewrite_group(
    *,
    src: zarr.Group,
    dst: zarr.Group,
    target_shard_size_bytes: int,
    chunk_yx: int,
    verify: int,
) -> None:
    _copy_attrs(dst, src)
    for name, member in src.items():
        if isinstance(member, zarr.Group):
            sub = dst.create_group(name, overwrite=True)
            _rewrite_group(
                src=member,
                dst=sub,
                target_shard_size_bytes=target_shard_size_bytes,
                chunk_yx=chunk_yx,
                verify=verify,
            )
        else:
            assert isinstance(member, zarr.Array)
            _rewrite_array(
                src=member,
                dst_group=dst,
                name=name,
                target_shard_size_bytes=target_shard_size_bytes,
                chunk_yx=chunk_yx,
                verify=verify,
            )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    in_path = args.input
    out_path = args.output

    target_shard_size_bytes = args.target_shard_mb * 1024 * 1024

    if not in_path.exists():
        raise FileNotFoundError(in_path)

    partial_out = out_path.with_name(f"{out_path.name}.partial")
    if partial_out.exists():
        shutil.rmtree(partial_out, ignore_errors=True)

    if out_path.exists():
        if not args.overwrite:
            raise FileExistsError(out_path)
        shutil.rmtree(out_path, ignore_errors=True)

    root = zarr.open(str(in_path), mode="r")
    if isinstance(root, zarr.Group):
        dst_root = zarr.group(str(partial_out), overwrite=True)
        _rewrite_group(
            src=root,
            dst=dst_root,
            target_shard_size_bytes=target_shard_size_bytes,
            chunk_yx=args.chunk_yx,
            verify=args.verify,
        )
    else:
        assert isinstance(root, zarr.Array)
        axes = _infer_axes(root)
        shape = tuple(int(s) for s in root.shape)
        chunks = _forced_chunks(shape=shape, axes=axes, chunk_yx=args.chunk_yx)
        dtype = np.dtype(root.dtype)
        codecs = label_zarr_codecs(dtype) if dtype == np.dtype(np.uint32) else default_zarr_codecs(dtype)
        shards = choose_shard_shape(
            shape=shape,
            chunks=chunks,
            dtype=dtype,
            target_shard_size_bytes=target_shard_size_bytes,
        )
        dst = zarr.create_array(
            str(partial_out),
            shape=shape,
            chunks=chunks,
            shards=shards,
            dtype=dtype,
            serializer=codecs[0],
            compressors=tuple(codecs[1:]),
            overwrite=True,
        )
        _copy_attrs(dst, root)
        _copy_by_shards(src=root, dst=dst, verify=args.verify)

    partial_out.rename(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
