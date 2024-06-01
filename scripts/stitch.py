# %%
import functools
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from itertools import cycle, islice
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Literal

import pandas as pd
import rich_click as click
from loguru import logger


def create_tile_config(
    path: Path, df: pd.DataFrame, sub: str, *, pixel: int = 1024, flavor: Literal["big", "stitch"] = "stitch"
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

    with open(path / "TileConfiguration.txt", "w") as f:
        f.write("dim=2\n")
        if flavor == "big":
            for idx, row in islice(cycle(ats.iterrows()), 27 * len(ats)):
                f.write(f"{idx}; ; ({row['x']}, {row['y']})\n")
        else:
            for idx, row in ats.iterrows():
                f.write(f"{idx:03d}_{sub}.tif; ; ({row['x']}, {row['y']})\n")


def run_imagej(path: Path, *, compute_overlap: bool = False, threshold: float = 0.4):
    options = "subpixel_accuracy"  # compute_overlap
    if compute_overlap:
        options += " compute_overlap"
    fusion = "Linear Blending"

    macro = f"""
    run("Memory & Threads...", "parallel=8");
    run("Grid/Collection stitching", "type=[Positions from file] \
order=[Defined by TileConfiguration] directory={path.resolve()} \
layout_file=TileConfiguration{"" if compute_overlap else ".registered"}.txt \
fusion_method=[{fusion}] regression_threshold={threshold} \
max/avg_displacement_threshold=1.5 absolute_displacement_threshold=2.5 {options} \
computation_parameters=[Save computation time (but use more RAM)] \
image_output=[Write to disk] \
output_directory={path.resolve()}");
    """

    #     macro = f"""
    #     run("Memory & Threads...", "parallel=8");
    #     run("Grid/Collection stitching", "type=[Unknown position] \
    # order=[All files in directory] directory={path.resolve()} \
    # output_textfile_name=TileConfiguration.txt \
    # fusion_method=[{fusion}] regression_threshold={threshold} \
    # max/avg_displacement_threshold=1.50 absolute_displacement_threshold=2.50 subpixel_accuracy \
    # computation_parameters=[Save computation time (but use more RAM)] \
    # image_output=[Write to disk] \
    # output_directory={path.resolve()}");
    #     """

    with NamedTemporaryFile("wt") as f:
        f.write(macro)
        f.flush()
        subprocess.run(
            f"/home/chaichontat/Fiji.app/ImageJ-linux64 --headless --console -macro {f.name}",
            shell=True,
        )


import re


def copy_registered(reference_path: Path, actual_path: Path):
    try:
        shutil.copy(reference_path / "TileConfiguration.registered.txt", actual_path)
    except FileNotFoundError:
        ...
    tr = actual_path / "TileConfiguration.registered.txt"
    tr.write_text(
        "\n".join(
            map(
                lambda x: x.replace(
                    f"_{int(reference_path.stem):03d}.tif", f"_{int(actual_path.stem):03d}.tif"
                ),
                tr.read_text().splitlines(),
            )
        )
    )


# path = Path("/fast2/thicc/thicc/")
# reference_path = path / "down2" / "0"
# actual_path = path / "down2" / "16"
# prefix = "3_11_19"
# glob = prefix + "-*.tif"


# @cli.command()
# @click.argument("path", type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=Path))
# # @click.option("--downsample", type=int, default=2)


@click.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=Path))
@click.argument("position_file", type=click.Path(exists=True, dir_okay=False, file_okay=True, path_type=Path))
@click.option("--recreate", is_flag=True)
@click.option("--reference", type=str)
@click.option("--threshold", type=float, default=0.4)
def main(
    path: Path,
    position_file: Path,
    *,
    reference: str | None = None,
    recreate: bool = False,
    threshold: float = 0.4,
):
    folders = sorted([p for p in path.iterdir() if p.is_dir()], key=lambda x: int(x.stem))
    ref = folders[-1] if reference is None else [f for f in folders if f.stem == reference][0]
    logger.info(f"Found {[f.name for f in folders]} at {path}.")

    def get_idxs(folder: Path):
        return {int(x.stem.split("_")[0]) for x in folder.glob("*.tif")}

    if recreate or not (ref / "TileConfiguration.txt").exists():
        files_idx = functools.reduce(lambda x, y: x & y, [get_idxs(f) for f in folders], get_idxs(ref))

        df = pd.read_csv(position_file, header=None).iloc[sorted(files_idx)]
        print(df)

        create_tile_config(ref, df, f"{int(ref.stem):03d}", pixel=1024)
        logger.info(f"Created TileConfiguration with {len(df)} files at {path}.")

    if recreate or not (ref / "TileConfiguration.registered.txt").exists():
        logger.info("Running first.")
        run_imagej(ref, compute_overlap=True, threshold=threshold)

    assert (ref / "TileConfiguration.registered.txt").exists(), "No registered file found."
    for f in folders:
        if f is not ref:
            copy_registered(ref, f)

    with ThreadPoolExecutor(4) as exc:
        for f in folders:
            if f is not ref:
                exc.submit(run_imagej, f)

    # Run first


# df = pl.read_csv(Path("/raid/data/star/sagittal/positions.csv"), has_header=False).to_numpy()
# df += -df.min(axis=0)
# df * 2048 / 200 * 0.9
# os.environ["JAVA_TOOL_OPTIONS"] = "-Djava.net.useSystemProxies=true"

# scyjava.config.add_option("-Xmx100g")
# ij = imagej.init("/home/chaichontat/Fiji.app", add_legacy=True)
# assert ij

# %%

# %%
if __name__ == "__main__":
    main()

# %%

# %%


# ImageJ-linux64 --update add-update-site BigStitcher https://sites.imagej.net/BigStitcher/
# ImageJ-linux64 --update update

# macro = f"""
# run("Memory & Threads...", "parallel=24");
# run("Define dataset ...", "define_dataset=[Automatic Loader (Bioformats based)] project_filename=dataset.xml path={path}/{glob} exclude=10 bioformats_channels_are?=Channels pattern_0=Tiles modify_voxel_size? voxel_size_x=0.108 voxel_size_y=0.108 voxel_size_z=1.0000 voxel_size_unit=µm move_tiles_to_grid_(per_angle)?=[Do not move Tiles to Grid (use Metadata if available)] how_to_load_images=[Re-save as multiresolution HDF5] dataset_save_path={path} manual_mipmap_setup subsampling_factors=[{{ {{1,1,1}}, {{2,2,1}} }}] hdf5_chunk_sizes=[{{ {{128,128,1}}, {{128,128,1}} }}] timepoints_per_partition=1 setups_per_partition=0 use_deflate_compression");

# run("Load TileConfiguration from File...", "select={path}/dataset.xml tileconfiguration={path}/TileConfiguration.txt use_pixel_units keep_metadata_rotation");

# run("Calculate pairwise shifts ...", "select={path}/dataset.xml process_angle=[All angles] process_channel=[All channels] process_illumination=[All illuminations] process_tile=[All tiles] process_timepoint=[All Timepoints] method=[Phase Correlation] show_expert_algorithm_parameters downsample_in_x=1 downsample_in_y=1 subpixel_accuracy");//

# run("Optimize globally and apply shifts ...", "select={path}/dataset.xml process_angle=[All angles] process_channel=[All channels] process_illumination=[All illuminations] process_tile=[All tiles] process_timepoint=[All Timepoints] relative=2.500 absolute=3.500 global_optimization_strategy=[Two-Round using Metadata to align unconnected Tiles and iterative dropping of bad links] fix_group_0-0,");

# // run("Fuse dataset ...", "select={path}/dataset.xml process_angle=[All angles] process_channel=[All channels] process_illumination=[All illuminations] process_tile=[All tiles] process_timepoint=[All Timepoints] bounding_box=[Currently Selected Views] downsampling=1 interpolation=[Linear Interpolation] pixel_type=[16-bit unsigned integer] interest_points_for_non_rigid=[-= Disable Non-Rigid =-] produce=[Each timepoint & channel] fused_image=[Save as (compressed) TIFF stacks] define_input=[Auto-load from input data (values shown below)] output_file_directory={path}/ filename_addition=[]");

# // run("Fuse dataset ...", "select={path}/dataset.xml process_angle=[All angles] process_channel=[All channels] process_illumination=[All illuminations] process_tile=[All tiles] process_timepoint=[All Timepoints] bounding_box=[Currently Selected Views] downsampling=2 interpolation=[Linear Interpolation] pixel_type=[16-bit unsigned integer] interest_points_for_non_rigid=[-= Disable Non-Rigid =-] produce=[Each timepoint & channel] fused_image=[ZARR/N5/HDF5 export using N5-API] define_input=[Auto-load from input data (values shown below)] export=HDF5 hdf5_file={path}/fused.h5 hdf5_base_dataset=/ hdf5_dataset_extension=/s0");

# """

# /path/to/fiji/ImageJ-linux64 –headless –console -macro /path/to/macro/bigStitcherBatch.ijm “/path/to/data 2 3”
