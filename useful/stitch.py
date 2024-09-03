# %%
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
import rich_click as click
from loguru import logger
from tifffile import TiffFile, imread, imwrite

from fishtools.analysis.tileconfig import TileConfiguration


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

    mat = ats[["y", "x"]].to_numpy() @ np.loadtxt("/home/chaichontat/fishtools/data/stage_rotation.txt")
    ats["x"] = mat[:, 0]
    ats["y"] = mat[:, 1]

    print(ats)

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

    print(macro)

    with NamedTemporaryFile("wt") as f:
        f.write(macro)
        f.flush()
        subprocess.run(
            f"/home/chaichontat/Fiji.app/ImageJ-linux64 --headless --console -macro {f.name}",
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


def extract_fiducial(path: Path, out: Path, trim: int = 25):
    with TiffFile(path) as tif:
        img = tif.pages[-1][trim:-trim, trim:-trim]
        imwrite(
            out / path.name,
            img,
            compression=22610,
            metadata={"axes": "YX"},
            compressionargs={"level": 0.7},
        )


@click.group()
def cli(): ...


@cli.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=Path))
@click.argument("position_file", type=click.Path(exists=True, dir_okay=False, file_okay=True, path_type=Path))
@click.option("--recreate", is_flag=True)
@click.option("--threshold", type=float, default=0.4)
def register(
    path: Path,
    position_file: Path,
    *,
    recreate: bool = False,
    threshold: float = 0.4,
):
    def get_idxs(folder: Path):
        return {int(x.stem) for x in sorted(folder.glob("*.tif"))}

    print((path / "TileConfiguration.registered.txt").exists())
    # functools.reduce(lambda x, y: x & y, [get_idxs(f) for f in folders]
    if recreate or not (path / "TileConfiguration.registered.txt").exists():
        files_idx = get_idxs(path)
        # pixel = 2**bits
        tileconfig = TileConfiguration.from_pos(
            pd.read_csv(position_file, header=None).iloc[sorted(files_idx)], downsample=3
        )
        tileconfig.write(path / "TileConfiguration.txt")
        logger.info(f"Created TileConfiguration at {path}.")
        logger.info("Running first.")
        run_imagej(path, compute_overlap=True, fuse=False, threshold=threshold, name="TileConfiguration")


@cli.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=Path))
@click.argument(
    "tile_config_file", type=click.Path(exists=True, dir_okay=False, file_okay=True, path_type=Path)
)
@click.option("--split", type=int, default=1)
@click.option("--reference", type=str)
@click.option("--overwrite", is_flag=True)
@click.option("--downsample", type=int, default=1)
def fuse(
    path: Path,
    tile_config_file: Path,
    *,
    split: int = 1,
    overwrite: bool = False,
    downsample: int = 1,
):
    folders = sorted([p for p in path.iterdir() if p.is_dir()], key=lambda x: int(x.stem))
    logger.info(f"Found {[f.name for f in folders]} at {path}.")

    tileconfig = TileConfiguration.from_file(tile_config_file)
    print(tileconfig.df)

    tileconfig = tileconfig.downsample(downsample)
    n = len(tileconfig) // split

    def run_folder(folder: Path):
        for i in range(split):
            tileconfig[i * n : (i + 1) * n].write(folder / f"TileConfiguration{i + 1}.registered.txt")
            run_imagej(folder, name=f"TileConfiguration{i + 1}")
            Path(folder / f"img_t1_z1_c1").rename(folder / f"fused_{i + 1}.tif")

    with ThreadPoolExecutor(4) as exc:
        for folder in folders:
            if list(folder.glob("fused*")) and not overwrite:
                logger.warning("Image already exists. Skipping.")
                continue
            exc.submit(run_folder, folder)

    if split > 1:
        for folder in folders:
            final_stitch(folder, split)


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
