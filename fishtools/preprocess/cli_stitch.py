# %%
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
import rich_click as click
from loguru import logger
from tifffile import TiffFile, TiffFileError, imread, imwrite

from fishtools.preprocess.tileconfig import TileConfiguration
from fishtools.utils.pretty_print import progress_bar, progress_bar_threadpool
from fishtools.utils.utils import batch_roi


def create_tile_config(
    path: Path,
    df: pd.DataFrame,
    *,
    name: str = "TileConfiguration.txt",
    pixel: int = 1024,
):
    scale = 2048 / pixel
    actual = pixel * (0.108 * scale)
    scaling = 200 / actual
    adjusted = pd.DataFrame(
        dict(y=(df[0] - df[0].min()), x=(df[1] - df[1].min())),
        dtype=int,
    )
    adjusted["x"] -= adjusted["x"].min()
    adjusted["x"] *= -(1 / 200) * pixel * scaling
    adjusted["y"] -= adjusted["y"].min()
    adjusted["y"] *= -(1 / 200) * pixel * scaling

    ats = adjusted.copy()
    ats["x"] -= ats["x"].min()
    ats["y"] -= ats["y"].min()

    with open(path / name, "w") as f:
        f.write("dim=2\n")
        for idx, row in ats.iterrows():
            f.write(f"{idx:04d}.tif; ; ({row['x']}, {row['y']})\n")


def run_imagej(
    path: Path,
    *,
    compute_overlap: bool = False,
    fuse: bool = True,
    threshold: float = 0.4,
    name: str = "TileConfiguration.txt",
    capture_output: bool = False,
):
    options = "subpixel_accuracy"  # compute_overlap
    if compute_overlap:
        options += " compute_overlap"
    fusion = (
        "Linear Blending"
        if fuse
        else "Do not fuse images (only write TileConfiguration)"
    )

    macro = f"""
    run("Memory & Threads...", "maximum=102400 parallel=32");

    // run("Memory & Threads...", "parallel=16");

    run("Grid/Collection stitching", "type=[Positions from file] \
    order=[Defined by TileConfiguration] directory={path.resolve()} \
    layout_file={name}{"" if compute_overlap else ".registered"}.txt \
    fusion_method=[{fusion}] regression_threshold={threshold} \
    max/avg_displacement_threshold=1.5 absolute_displacement_threshold=2.5 {options} \
    computation_parameters=[Save computation time (but use more RAM)] \
    image_output=[Write to disk] \
    output_directory={path.resolve()}");
    """

    with NamedTemporaryFile("wt") as f:
        f.write(macro)
        f.flush()
        subprocess.run(
            f"/home/chaichontat/Fiji.app/ImageJ-linux64 --headless --console -macro {f.name}",
            capture_output=capture_output,
            check=True,
            shell=True,
        )


def copy_registered(reference_path: Path, actual_path: Path):
    for file in reference_path.glob("*.registered.txt"):
        shutil.copy(file, actual_path)


def extract_channel(
    path: Path,
    out: Path,
    *,
    idx: int | None = None,
    trim: int = 0,
    max_proj: bool = False,
    downsample: int = 1,
    reduce_bit_depth: int = 0,
):
    try:
        with TiffFile(path) as tif:
            if trim < 0:
                raise ValueError("Trim must be positive")

            if max_proj:
                img = tif.asarray().max(axis=(0, 1))
            elif len(tif.pages) == 1 and tif.pages[0].asarray().ndim == 3:
                img = tif.pages[0].asarray()[idx]
            else:
                img = tif.pages[idx].asarray()

    except TiffFileError as e:
        logger.critical(f"Error reading {path}: {e}")
        raise e

    img = (
        img[trim:-trim:downsample, trim:-trim:downsample]
        if trim
        else img[::downsample, ::downsample]
    )
    if reduce_bit_depth:
        if img.dtype != np.uint16:
            raise ValueError("Cannot reduce bit depth if image is not uint16")
        img >>= reduce_bit_depth

    imwrite(
        out,
        img,
        compression=22610,
        metadata={"axes": "YX"},
        compressionargs={"level": 0.7},
    )


@click.group()
def stitch(): ...


@stitch.command()
@click.argument(
    "path",
    type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=Path),
)
@click.option(
    "--tileconfig",
    type=click.Path(exists=True, dir_okay=False, file_okay=True, path_type=Path),
)
@click.option("--fuse", is_flag=True)
@click.option("--downsample", type=int, default=2)
def register_simple(path: Path, tileconfig: Path, fuse: bool, downsample: int):
    if downsample > 1:
        logger.info(
            f"Downsampling tile config by {downsample}x to {path / 'TileConfiguration.txt'}"
        )
    TileConfiguration.from_file(tileconfig).downsample(downsample).write(
        path / "TileConfiguration.txt"
    )

    run_imagej(
        path,
        compute_overlap=True,
        fuse=fuse,
        name="TileConfiguration",
    )


@stitch.command()
# "Path to registered images folder. Expects CYX images. Will reshape to CYX if not. Assumes ${name}-${idx}.tif",
@click.argument(
    "path",
    type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=Path),
)
@click.argument("roi", type=str)
@click.option(
    "--position_file",
    type=click.Path(exists=True, dir_okay=False, file_okay=True, path_type=Path),
)
@click.option("--idx", "-i", type=int, help="Channel (index) to use for registration")
@click.option("--fid", is_flag=True)
@click.option("--threshold", type=float, default=0.4)
@click.option("--overwrite", is_flag=True)
@click.option("--max-proj", is_flag=True)
@batch_roi()
def register(
    path: Path,
    roi: str,
    *,
    position_file: Path | None = None,
    idx: int | None = None,
    fid: bool = False,
    max_proj: bool = False,
    overwrite: bool = False,
    threshold: float = 0.4,
):
    base_path = path
    if roi == "*":
        rois = sorted(path.glob("*"))
    path = next(path.glob(f"registered--{roi}*"))
    if not path.exists():
        raise ValueError(f"No registered images at {path.resolve()} found.")

    imgs = sorted((f for f in path.glob("*.tif") if not f.name.endswith(".hp.tif")))
    (out_path := path.parent / f"stitch--{roi.split('+')[0]}").mkdir(exist_ok=True)

    if overwrite:
        [p.unlink() for p in out_path.glob("*.tif")]

    def get_idx(path: Path):
        return int(path.stem.split("-")[1])

    if fid:
        logger.info(f"Found {len(imgs)} files. Extracting channel {idx} to {out_path}.")
        with progress_bar(len(imgs)) as callback, ThreadPoolExecutor(6) as exc:
            futs = []
            for path in imgs:
                out_name = f"{get_idx(path):04d}.tif"
                if (out_path / out_name).exists() and not overwrite:
                    continue
                _idx = path.stem.split("-")[1]
                futs.append(
                    exc.submit(
                        extract_channel,
                        path.parent.parent / f"fids--{roi}" / f"fids-{_idx}.tif",
                        out_path / out_name,
                        idx=0,
                        downsample=1,
                        max_proj=False,  # already max proj
                    )
                )

            for f in as_completed(futs):
                f.result()
                callback()

    elif max_proj or idx is not None:
        logger.info(f"Found {len(imgs)} files. Extracting channel {idx} to {out_path}.")
        with progress_bar(len(imgs)) as callback, ThreadPoolExecutor(6) as exc:
            futs = []
            for path in imgs:
                out_name = f"{get_idx(path):04d}.tif"
                if (out_path / out_name).exists() and not overwrite:
                    continue
                futs.append(
                    exc.submit(
                        extract_channel,
                        path,
                        out_path / out_name,
                        idx=idx,
                        max_proj=max_proj,
                    )
                )

            for f in as_completed(futs):
                f.result()
                callback()

    del path
    if overwrite or not (out_path / "TileConfiguration.registered.txt").exists():
        files = sorted(
            (f for f in out_path.glob("*.tif") if not f.name.endswith(".hp.tif"))
        )
        files_idx = [
            int(file.stem.split("-")[-1])
            for file in files
            if file.stem.split("-")[-1].isdigit()
        ]
        logger.debug(f"Using {files_idx}")
        tileconfig = TileConfiguration.from_pos(
            pd.read_csv(
                position_file
                or (
                    base_path / f"{roi}.csv"
                    if (base_path / f"{roi}.csv").exists()
                    else (base_path.resolve().parent.parent / f"{roi}.csv")
                ),
                header=None,
            ).iloc[sorted(files_idx)]
        )
        tileconfig.write(out_path / "TileConfiguration.txt")
        logger.info(f"Created TileConfiguration at {out_path}.")
        logger.info("Running first.")
        run_imagej(
            out_path,
            compute_overlap=True,
            fuse=False,
            threshold=threshold,
            name="TileConfiguration",
        )


def extract(
    path: Path,
    out_path: Path,
    *,
    trim: int = 0,
    downsample: int = 2,
    reduce_bit_depth: int = 0,
    subsample_z: int = 1,
    max_proj: bool = False,
    is_2d: bool = False,
    channels: list[int] | None = None,
):
    try:
        img = imread(path)

        if is_2d:
            if len(img.shape) == 4:  # ZCYX
                if not max_proj:
                    raise ValueError(
                        "Please set is_3d to True if you want to segment 3D images or max_proj to True for max projection."
                    )
                img = img.max(axis=0)

            img = np.atleast_3d(img)
            img = (
                img[
                    channels if channels else slice(None),
                    trim:-trim:downsample,
                    trim:-trim:downsample,
                ]
                if trim
                else img[
                    channels if channels else slice(None), ::downsample, ::downsample
                ]
            )
            if reduce_bit_depth:
                img >>= reduce_bit_depth
            for i in range(img.shape[0]):
                (out_path / f"{i:02d}").mkdir(exist_ok=True)
                imwrite(
                    out_path / f"{i:02d}" / (path.stem.split("-")[1] + ".tif"),
                    img[i],
                    compression=22610,
                    metadata={"axes": "YX"},
                    compressionargs={"level": 0.7},
                )
            del img
            return

        if len(img.shape) < 3:
            raise ValueError("Image must be at least 3D")

        if len(img.shape) == 3:  # ZYX
            img = img[::subsample_z, np.newaxis, ...]
        elif len(img.shape) > 4:
            raise ValueError("Image must be 3D or 4D")

        img = (
            img[
                ::subsample_z,
                channels if channels else slice(None),
                trim:-trim:downsample,
                trim:-trim:downsample,
            ]
            if trim
            else img[
                ::subsample_z,
                channels if channels else slice(None),
                ::downsample,
                ::downsample,
            ]
        )

        if max_proj:
            img = img.max(axis=0, keepdims=True)

        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                p = out_path / f"{i:02d}" / f"{j:02d}"
                p.mkdir(exist_ok=True, parents=True)
                imwrite(
                    p / (path.stem.split("-")[1] + ".tif"),
                    img[i, j],
                    compression=22610,
                    metadata={"axes": "YX"},
                    compressionargs={"level": 0.75},
                )

        del img
        return
    except Exception as e:
        raise Exception(f"Error reading {path}") from e


@stitch.command()
@click.argument(
    "path", type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=Path)
)
@click.argument("roi", type=str)
@click.option(
    "--tile_config",
    type=click.Path(exists=True, dir_okay=False, file_okay=True, path_type=Path),
)
@click.option(
    "--split",
    type=int,
    default=1,
    help="Split tiles into this many parts. Mainly to avoid overflows in very large images.",
)
@click.option("--overwrite", is_flag=True)
@click.option("--downsample", "-d", type=int, default=2)
@click.option("--subsample-z", type=int, default=1)
@click.option("--is-2d", is_flag=True)
@click.option("--threads", "-t", type=int, default=8)
@click.option("--channels", type=str, default="all")
@click.option("--max-proj", is_flag=True)
@click.option("--debug", is_flag=True)
@click.option("--skip-extract", is_flag=True)
@batch_roi("stitch--*")
def fuse(
    path: Path,
    roi: str,
    *,
    tile_config: Path | None = None,
    split: int = 1,
    overwrite: bool = False,
    downsample: int = 2,
    is_2d: bool = False,
    threads: int = 8,
    channels: str = "all",
    subsample_z: int = 1,
    max_proj: bool = False,
    debug: bool = False,
    skip_extract: bool = False,
):
    if "--" in path.as_posix():
        raise ValueError("Please be in the workspace folder.")

    path_img = path / f"registered--{roi}"
    path = path / f"stitch--{roi}"
    files = sorted(path_img.glob("*.tif"))
    if not len(files):
        raise ValueError(f"No images found at {path_img.resolve()}")
    logger.info(f"Found {len(files)} images at {path_img.resolve()}")

    if overwrite:
        for p in path.iterdir():
            if p.is_dir():
                shutil.rmtree(p)

    if tile_config is None:
        tile_config = (
            path.parent / path.name.split("+")[0] / "TileConfiguration.registered.txt"
        )

    logger.info(f"Getting tile configuration from {tile_config.resolve()}")
    tileconfig = TileConfiguration.from_file(tile_config).downsample(downsample)
    n = len(tileconfig) // split

    if channels == "all":
        channels = ",".join(map(str, range(imread(files[0]).shape[1])))

    imgs = {int(p.stem.split("-")[1]) for p in path_img.glob("*.tif")}
    needed = set(tileconfig.df["index"])

    if len(needed & imgs) != len(needed):
        raise ValueError(
            f"Not all images are present in {path_img}. Missing: {needed - imgs}"
        )

    if not skip_extract:
        logger.info(
            f"Found {len(files)} files. Extracting channel {channels} to {path}"
        )
        with progress_bar_threadpool(
            len(files), threads=threads, stop_on_exception=True
        ) as submit:
            for file in files:
                submit(
                    extract,
                    file,
                    path,
                    downsample=downsample,
                    subsample_z=subsample_z,
                    is_2d=is_2d,
                    channels=list(map(int, channels.split(","))),
                    max_proj=max_proj,
                )

    def run_folder(folder: Path, capture_output: bool = False):
        for i in range(split):
            tileconfig[i * n : (i + 1) * n].write(
                folder / f"TileConfiguration{i + 1}.registered.txt"
            )
            try:
                run_imagej(
                    folder,
                    name=f"TileConfiguration{i + 1}",
                    capture_output=capture_output,
                )
            except Exception as e:
                logger.critical(f"Error running ImageJ for {folder}: {e}")
                raise e
            Path(folder / "img_t1_z1_c1").rename(
                folder / f"fused_{folder.name}-{i + 1}.tif"
            )

    # Get all folders without subfolders
    folders = [
        folder
        for folder, subfolders, _ in (path.parent / f"stitch--{roi}").walk()
        if not subfolders
        and not folder.name.endswith(".zarr")
        and not ".zarr" in folder.resolve().as_posix()
    ]

    logger.info(f"Calling ImageJ on {len(folders)} folders.")
    with progress_bar_threadpool(
        len(folders), threads=threads, stop_on_exception=True
    ) as submit:
        for folder in folders:
            if not folder.is_dir():
                raise Exception("Invalid folder")
            if not folder.name.isdigit():
                raise ValueError(
                    f"Invalid folder name {folder.name}. No external folders allowed."
                )

            existings = list(folder.glob("fused*"))
            if existings and not overwrite:
                logger.warning(f"{existings} already exists. Skipping this folder.")
                continue
            submit(run_folder, folder, capture_output=not debug)

    if split > 1:
        for folder in folders:
            final_stitch(folder, split)


def numpy_array_to_zarr[*Ts](
    write_path: Path | str, array: np.ndarray, chunks: tuple[*Ts]
):
    import zarr
    # from numcodecs import Blosc

    # codec = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)

    zarr_array = zarr.create_array(
        write_path,
        shape=array.shape,
        chunks=chunks,
        dtype=array.dtype,
        # codecs=[codec],
    )
    zarr_array[...] = array
    return zarr_array


@stitch.command()
@click.argument(
    "path", type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=Path)
)
@click.argument("roi", type=str)
@click.option("--chunk-size", type=int, default=2048)
@batch_roi("stitch--*")
def combine(path: Path, roi: str, chunk_size: int = 2048):
    import zarr

    path = Path(path / f"stitch--{roi}")
    # Group folders by Z index (parent directory name)
    folders_by_z = {}
    for folder, subfolders, _ in path.walk():
        if (
            not subfolders
            and folder.name.isdigit()
            and folder.parent.name.isdigit()
            and not ".zarr" in folder.resolve().as_posix()
        ):
            z_idx = int(folder.parent.name)
            if z_idx not in folders_by_z:
                folders_by_z[z_idx] = []
            folders_by_z[z_idx].append(folder)

    if not folders_by_z:
        raise ValueError(f"No valid Z/C folders found in {path}")

    # Sort folders within each Z index by C index (folder name)
    for z_idx in folders_by_z:
        folders_by_z[z_idx].sort(key=lambda f: int(f.name))

    zs = max(folders_by_z.keys()) + 1
    cs = (
        max(int(f.name) for f in folders_by_z[0]) + 1
    )  # Assume C count is same for all Z

    # Check for fused images in the first Z plane to get dimensions
    first_z_folders = folders_by_z[0]
    for folder in first_z_folders:
        if not (folder / f"fused_{folder.name}-1.tif").exists():
            raise ValueError(f"No fused image found for {folder.name} in Z=0")

    # Get shape from the first image of the first Z plane
    first_folder = first_z_folders[0]
    first_img = imread(first_folder / f"fused_{first_folder.name}-1.tif")
    img_shape = first_img.shape
    dtype = first_img.dtype
    final_shape = (zs, img_shape[0], img_shape[1], cs)
    logger.info(f"Final Zarr shape: {final_shape}, dtype: {dtype}")

    # Initialize the Zarr array
    zarr_path = path / "fused.zarr"
    logger.info(f"Writing to {zarr_path.resolve()}")
    # Define chunks: chunk along Z=1, use user chunk_size for Y/X, full chunk for C
    zarr_chunks = (1, chunk_size, chunk_size, cs)
    z_array = zarr.create_array(
        zarr_path,
        shape=final_shape,
        chunks=zarr_chunks,
        dtype=dtype,
        overwrite=True,
    )

    z_plane_data = np.zeros((img_shape[0], img_shape[1], cs), dtype=dtype)

    with progress_bar(len(folders_by_z)) as progress:
        for i in sorted(folders_by_z.keys()):
            logger.info(f"Processing Z-plane {i + 1}/{zs}")
            z_plane_folders = folders_by_z[i]

            for folder in z_plane_folders:
                j = int(folder.name)
                try:
                    img = imread(folder / f"fused_{folder.name}-1.tif")
                    # Write into the C dimension of the current Z-plane array
                    z_plane_data[:, :, j] = img[:, :]
                    del img
                except FileNotFoundError:
                    raise FileNotFoundError(
                        f"File not found: {folder / f'fused_{folder.name}-1.tif'}"
                    )
                except Exception as e:
                    raise Exception(
                        f"Error reading {folder / f'fused_{folder.name}-1.tif'}"
                    ) from e

            logger.info(f"Writing Z-plane {i} to Zarr array")
            z_array[i, :, :, :] = z_plane_data
            progress()

    # Add metadata (channel names)
    try:
        first_reg_file = next((path.parent / f"registered--{roi}").glob("*.tif"))
        with TiffFile(first_reg_file) as tif:
            # Attempt to get channel names, handle potential errors
            names = tif.shaped_metadata[0].get("key") if tif.shaped_metadata else None
            if names:
                z_array.attrs["key"] = names
                logger.info(f"Added channel names: {names}")
            else:
                logger.warning("Could not find channel names ('key') in TIF metadata.")
    except StopIteration:
        logger.warning(
            f"No registered TIF file found in {path.parent / f'registered--{roi}'} to read channel names."
        )
    except Exception as e:
        logger.warning(f"Error reading metadata from TIF file: {e}")

    logger.info("Deleting source folders.")
    all_folders = [f for z_folders in folders_by_z.values() for f in z_folders]

    for folder in all_folders:
        shutil.rmtree(folder.parent)  # Remove the Z-level parent folder

    # Remove empty Z directories if necessary (walk might not list them if empty)
    for z_idx in sorted(folders_by_z.keys()):
        z_dir = path / str(z_idx)
        if z_dir.exists() and z_dir.is_dir():
            try:
                z_dir.rmdir()  # Remove empty directory
            except OSError:
                logger.warning(
                    f"Could not remove directory {z_dir}, might not be empty."
                )

    logger.info("Done.")


@stitch.command()
@click.argument(
    "path", type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=Path)
)
@click.argument("roi", type=str, default="*")
@click.option(
    "--tile_config",
    type=click.Path(exists=True, dir_okay=False, file_okay=True, path_type=Path),
)
@click.option("--codebook", type=str)
@click.option("--overwrite", is_flag=True)
@click.option("--downsample", "-d", type=int, default=2)
@click.option("--subsample-z", type=int, default=1)
@click.option("--threads", "-t", type=int, default=8)
@click.option("--channels", type=str, default="-3,-2,-1")
@batch_roi()
def run(
    path: Path,
    roi: str,
    *,
    codebook: str,
    tile_config: Path | None = None,
    overwrite: bool = False,
    downsample: int = 2,
    threads: int = 8,
    channels: str = "-3,-2,-1",
    subsample_z: int = 1,
):
    register.callback(path, roi=roi, position_file=None, fid=True)
    fuse.callback(
        path,
        roi=f"{roi}+{codebook}",
        tile_config=tile_config,
        split=True,
        overwrite=overwrite,
        downsample=downsample,
        threads=threads,
        channels=channels,
        subsample_z=subsample_z,
    )
    combine.callback(path, roi=f"{roi}+{codebook}", threads=threads)


def final_stitch(path: Path, n: int):
    logger.info(f"Combining splits of {n} for {path}.")
    import polars as pl

    tcs = [
        TileConfiguration.from_file(f"{path}/TileConfiguration{i + 1}.registered.txt")
        for i in range(n)
    ]

    bmin = pl.concat([tc.df.min() for tc in tcs])
    bmax = pl.concat([tc.df.max() for tc in tcs])

    mins = (bmin.min()[0, "y"], bmin.min()[0, "x"])
    maxs = (bmax.max()[0, "y"] + 1024, bmax.max()[0, "x"] + 1024)

    out = np.zeros(
        (n, int(maxs[0] - mins[0] + 1), int(maxs[1] - mins[1] + 1)), dtype=np.uint16
    )

    for i in range(n):
        img = imread(f"{path}/fused_{i + 1}.tif")
        offsets = (int(bmin[i, "y"] - mins[i]), int(bmin[i, "x"] - mins[i]))
        out[
            i,
            offsets[0] : img.shape[0] + offsets[0],
            offsets[1] : img.shape[1] + offsets[1],
        ] = img

    out = out.max(axis=0)
    imwrite(
        path / "fused.tif",
        out,
        compression=22610,
        compressionargs={"level": 0.8},
        metadata={"axes": "YX"},
    )
    del out


if __name__ == "__main__":
    stitch()

# %%
