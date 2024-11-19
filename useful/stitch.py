# %%
import re
import shutil
import subprocess
import threading
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

    # mat = ats[["y", "x"]].to_numpy() @ np.loadtxt("/home/chaichontat/fishtools/data/stage_rotation.txt")
    # ats["x"] = mat[:, 0]
    # ats["y"] = mat[:, 1]

    # print(ats)

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
    fusion = "Linear Blending" if fuse else "Do not fuse images (only write TileConfiguration)"

    align = f"""
    run("Calculate pairwise shifts ...", "select={path.resolve()}/dataset.xml \
    process_angle=[All angles] process_channel=[All channels] process_illumination=[All illuminations] process_tile=[All tiles] \
    process_timepoint=[All Timepoints] method=[Phase Correlation] \
    downsample_in_x=1 downsample_in_y=1");

    run("Filter pairwise shifts ...", "select={path.resolve()}/dataset.xml \
    filter_by_link_quality min_r=0.6 max_r=1 \
    max_shift_in_x=0 max_shift_in_y=0 max_shift_in_z=0 \
    max_displacement=0");

    run("Optimize globally and apply shifts ...", "select={path.resolve()}/dataset.xml \
    process_angle=[All angles] process_channel=[All channels] \
    process_illumination=[All illuminations] process_tile=[All tiles] \
    process_timepoint=[All Timepoints] \
    relative=1.500 absolute=2.500 \
    global_optimization_strategy=[Two-Round using Metadata to align unconnected Tiles and iterative dropping of bad links] \
    fix_group_0-0");
"""

    # run("Define dataset ...", "define_dataset=[Automatic Loader (Bioformats based)] \
    # project_filename=dataset.xml path={path.resolve()}/*.tif \
    # exclude=10 pattern_0=Tiles move_tiles_to_grid_(per_angle)?=[Do not move Tiles to Grid (use Metadata if available)] \
    # how_to_load_images=[Load raw data directly] \
    # dataset_save_path={path.resolve()}");

    # run("Load TileConfiguration from File...", "select={path.resolve()}/dataset.xml \
    # tileconfiguration={path.resolve()}/TileConfiguration.registered.txt \
    # use_pixel_units keep_metadata_rotation");

    # run("Fuse dataset ...", "select={path.resolve()}/dataset.xml \
    # process_angle=[All angles] process_channel=[All channels] process_illumination=[All illuminations] \
    # process_tile=[All tiles] process_timepoint=[All Timepoints] bounding_box=[Currently Selected Views] \
    # downsampling=1 interpolation=[Linear Interpolation] pixel_type=[16-bit unsigned integer] \
    # interest_points_for_non_rigid=[-= Disable Non-Rigid =-] \
    # blend produce=[Each timepoint & channel] \
    # fused_image=[Save as (compressed) TIFF stacks] \
    # define_input=[Auto-load from input data (values shown below)] \
    # output_file_directory={path.resolve()}");

    macro = f"""
    run("Memory & Threads...", "parallel=16");

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
    # tr = actual_path / "TileConfiguration.registered.txt"
    # tr.write_text(
    #     "\n".join(
    #         map(
    #             lambda x: x.replace(
    #                 f"_{int(reference_path.stem):03d}.tif", f"_{int(actual_path.stem):03d}.tif"
    #             ),
    #             tr.read_text().splitlines(),
    #         )
    #     )
    # )


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
                img = tif.asarray()[:, idx].max(axis=0)
            elif len(tif.pages) == 1 and tif.pages[0].asarray().ndim == 3:
                img = tif.pages[0].asarray()[idx]
            else:
                img = tif.pages[idx].asarray()

    except TiffFileError as e:
        logger.critical(f"Error reading {path}: {e}")
        raise e

    img = img[trim:-trim:downsample, trim:-trim:downsample] if trim else img[::downsample, ::downsample]
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
def cli(): ...


@cli.command()
# "Path to registered images folder. Expects CYX images. Will reshape to CYX if not. Assumes ${name}-${idx}.tif",
@click.argument(
    "path",
    type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=Path),
)
@click.argument("roi", type=str)
@click.argument("position_file", type=click.Path(exists=True, dir_okay=False, file_okay=True, path_type=Path))
@click.option("--idx", "-i", type=int, help="Channel (index) to use for registration")
@click.option("--fid", is_flag=True)
@click.option("--threshold", type=float, default=0.4)
@click.option("--overwrite", is_flag=True)
@click.option("--max-proj", is_flag=True)
def register(
    path: Path,
    roi: str,
    position_file: Path,
    *,
    idx: int | None = None,
    fid: bool = False,
    max_proj: bool = False,
    overwrite: bool = False,
    threshold: float = 0.4,
):
    path = path / f"registered--{roi}"
    if not path.exists():
        raise ValueError(f"No registered images at {path.resolve()} found.")

    imgs = sorted((f for f in path.glob("*.tif") if not f.name.endswith(".hp.tif")))
    (out_path := path.parent / f"stitch--{roi}").mkdir(exist_ok=True)

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
                    exc.submit(extract_channel, path, out_path / out_name, idx=idx, max_proj=max_proj)
                )

            for f in as_completed(futs):
                f.result()
                callback()

    del path
    if overwrite or not (out_path / "TileConfiguration.registered.txt").exists():
        files = sorted((f for f in out_path.glob("*.tif") if not f.name.endswith(".hp.tif")))
        files_idx = [int(file.stem.split("-")[-1]) for file in files if file.stem.split("-")[-1].isdigit()]
        logger.debug(f"Using {files_idx}")
        tileconfig = TileConfiguration.from_pos(
            pd.read_csv(position_file, header=None).iloc[sorted(files_idx)]
        )
        tileconfig.write(out_path / "TileConfiguration.txt")
        logger.info(f"Created TileConfiguration at {out_path}.")
        logger.info("Running first.")
        run_imagej(out_path, compute_overlap=True, fuse=False, threshold=threshold, name="TileConfiguration")


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
            img[channels if channels else slice(None), trim:-trim:downsample, trim:-trim:downsample]
            if trim
            else img[channels if channels else slice(None), ::downsample, ::downsample]
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

    if max_proj:
        raise ValueError("Cannot use max_proj with is_3d")

    if len(img.shape) < 3:
        raise ValueError("Image must be at least 3D")

    if len(img.shape) == 3:  # ZYX
        img = img[::subsample_z, np.newaxis, ...]
    elif len(img.shape) > 4:
        raise ValueError("Image must be 3D or 4D")

    img = (
        img[
            ::subsample_z, channels if channels else slice(None), trim:-trim:downsample, trim:-trim:downsample
        ]
        if trim
        else img[::subsample_z, channels if channels else slice(None), ::downsample, ::downsample]
    )

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


@cli.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=Path))
@click.argument("roi", type=str)
@click.option("--tile_config", type=click.Path(exists=True, dir_okay=False, file_okay=True, path_type=Path))
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
@click.option("--channels", type=str, default="-3,-2,-1")
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
    channels: str = "-3,-2,-1",
    subsample_z: int = 1,
):
    if "--" in path.as_posix():
        raise ValueError("Please be in the workspace folder.")

    path_img = path / f"registered--{roi}"
    path = path / f"stitch--{roi}"
    files = sorted(path_img.glob("*.tif"))
    if not len(files):
        raise ValueError(f"No images found at {path_img.resolve()}.")
    logger.info(f"Found {len(files)} images at {path_img.resolve()}.")

    if tile_config is None:
        tile_config = path / "TileConfiguration.registered.txt"

    print(tile_config.resolve())

    tileconfig = TileConfiguration.from_file(tile_config).downsample(downsample)
    n = len(tileconfig) // split

    with progress_bar_threadpool(len(files), threads=threads) as submit:
        for file in files:
            submit(
                extract,
                file,
                path,
                downsample=downsample,
                subsample_z=subsample_z,
                is_2d=is_2d,
                channels=list(map(int, channels.split(","))),
            )

    def run_folder(folder: Path, capture_output: bool = False):
        for i in range(split):
            tileconfig[i * n : (i + 1) * n].write(folder / f"TileConfiguration{i + 1}.registered.txt")
            run_imagej(folder, name=f"TileConfiguration{i + 1}", capture_output=capture_output)
            Path(folder / "img_t1_z1_c1").rename(folder / f"fused_{folder.name}-{i + 1}.tif")

    # Get all folders without subfolders
    folders = [folder for folder, subfolders, _ in (path.parent / f"stitch--{roi}").walk() if not subfolders]

    with progress_bar_threadpool(len(folders), threads=threads) as submit:
        for folder in folders:
            if not folder.is_dir():
                raise Exception("Invalid folder")
            if not folder.name.isdigit():
                raise ValueError(f"Invalid folder name {folder.name}. No external folders allowed.")

            existings = list(folder.glob("fused*"))
            if existings and not overwrite:
                logger.warning(f"{existings} already exists. Skipping this folder.")
                continue
            submit(run_folder, folder, capture_output=True)

    if split > 1:
        for folder in folders:
            final_stitch(folder, split)


@cli.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=Path))
@click.argument("roi", type=str)
@click.option("--threads", "-t", type=int, default=8)
def combine(path: Path, roi: str, threads: int = 8):
    path = Path(path / f"stitch--{roi}")
    folders = sorted(folder for folder, subfolders, _ in (path).walk() if not subfolders)
    toplevels = [x for x in path.iterdir() if x.is_dir()]
    zs = max(int(x.name) for x in toplevels) + 1
    cs = max(int(x.name) for x in toplevels[0].iterdir() if x.is_dir()) + 1

    for folder in folders:
        if not (folder / f"fused_{folder.name}-1.tif").exists():
            raise ValueError(f"No fused image found for {folder.name}")

    folder = folders[0]
    img = imread(folder / f"fused_{folder.name}-1.tif")
    out = np.zeros((zs, cs, img.shape[0], img.shape[1]), dtype=np.uint16)
    logger.info(f"Out shape: {out.shape}")
    del img

    lock = threading.Lock()

    def load(folder: Path):
        i = int(folder.parent.name)
        j = int(folder.name)
        print(f"Processing z={i} c={j}")
        img = imread(folder / f"fused_{folder.name}-1.tif")
        with lock:
            out[i, j, :, :] = img[:, :]
            # np.copyto(out[i, j], img)
        del img

    with progress_bar_threadpool(len(folders), threads=threads) as submit:
        for folder in folders:
            submit(load, folder)

    logger.info(f"Writing to {path.resolve() / 'fused.tif'}")
    imwrite(
        path / "fused.tif",
        out,
        compression="zlib",
        metadata={"axes": "ZCYX"},
        # compressionargs={"level": 0.75},
        bigtiff=True,
    )
    logger.info("Done.")

    # out = np.zeros((zs, cs), dtype=np.uint16)

    # f"fused_{folder.name}-{i + 1}.tif")

    # logger.info(f"Combining splits of {n} for {path.resolve()}.")


def final_stitch(path: Path, n: int):
    logger.info(f"Combining splits of {n} for {path}.")
    import polars as pl

    tcs = [TileConfiguration.from_file(f"{path}/TileConfiguration{i + 1}.registered.txt") for i in range(n)]

    bmin = pl.concat([tc.df.min() for tc in tcs])
    bmax = pl.concat([tc.df.max() for tc in tcs])

    mins = (bmin.min()[0, "y"], bmin.min()[0, "x"])
    maxs = (bmax.max()[0, "y"] + 1024, bmax.max()[0, "x"] + 1024)

    out = np.zeros((n, int(maxs[0] - mins[0] + 1), int(maxs[1] - mins[1] + 1)), dtype=np.uint16)

    for i in range(n):
        img = imread(f"{path}/fused_{i + 1}.tif")
        offsets = (int(bmin[i, "y"] - mins[i]), int(bmin[i, "x"] - mins[i]))
        out[i, offsets[0] : img.shape[0] + offsets[0], offsets[1] : img.shape[1] + offsets[1]] = img

    out = out.max(axis=0)
    imwrite(
        path / "fused.tif", out, compression=22610, compressionargs={"level": 0.8}, metadata={"axes": "YX"}
    )
    del out


if __name__ == "__main__":
    cli()

# %%
