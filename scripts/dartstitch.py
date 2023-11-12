# %%
import os
from pathlib import Path

import imagej
import pandas as pd
import polars as pl

# scyjava.config.add_option("-Xmx220g")
# ij = imagej.init("/home/chaichontat/Fiji.app")
# assert ij


path = "/raid/data/equiphi"
globs = "4_29_30-*_max.tif"
macro = f"""
// run("Define dataset ...", "define_dataset=[Automatic Loader (Bioformats based)] project_filename=dataset.xml path={path}/{globs} exclude=10 bioformats_channels_are?=Channels pattern_0=Tiles modify_voxel_size? voxel_size_x=0.108 voxel_size_y=0.108 voxel_size_z=1.0000 voxel_size_unit=µm move_tiles_to_grid_(per_angle)?=[Do not move Tiles to Grid (use Metadata if available)] how_to_load_images=[Re-save as multiresolution HDF5] dataset_save_path={path} manual_mipmap_setup subsampling_factors=[{{ {{1,1,1}}, {{2,2,1}} }}] hdf5_chunk_sizes=[{{ {{128,128,1}}, {{128,128,1}} }}] timepoints_per_partition=1 setups_per_partition=0 use_deflate_compression");

// run("Load TileConfiguration from File...", "select={path}/dataset.xml tileconfiguration={path}/TileConfiguration.txt use_pixel_units keep_metadata_rotation");

// run("Calculate pairwise shifts ...", "select={path}/dataset.xml process_angle=[All angles] process_channel=[All channels] process_illumination=[All illuminations] process_tile=[All tiles] process_timepoint=[All Timepoints] method=[Phase Correlation] show_expert_algorithm_parameters downsample_in_x=1 downsample_in_y=1 subpixel_accuracy");//

// run("Optimize globally and apply shifts ...", "select={path}/dataset.xml process_angle=[All angles] process_channel=[All channels] process_illumination=[All illuminations] process_tile=[All tiles] process_timepoint=[All Timepoints] relative=2.500 absolute=3.500 global_optimization_strategy=[Two-Round using Metadata to align unconnected Tiles and iterative dropping of bad links] fix_group_0-0,");

// run("Fuse dataset ...", "select={path}/dataset.xml process_angle=[All angles] process_channel=[All channels] process_illumination=[All illuminations] process_tile=[All tiles] process_timepoint=[All Timepoints] bounding_box=[Currently Selected Views] downsampling=1 interpolation=[Linear Interpolation] pixel_type=[16-bit unsigned integer] interest_points_for_non_rigid=[-= Disable Non-Rigid =-] produce=[Each timepoint & channel] fused_image=[Save as (compressed) TIFF stacks] define_input=[Auto-load from input data (values shown below)] output_file_directory={path}/ filename_addition=[]");

run("Fuse dataset ...", "select={path}/dataset.xml process_angle=[All angles] process_channel=[All channels] process_illumination=[All illuminations] process_tile=[All tiles] process_timepoint=[All Timepoints] bounding_box=[Currently Selected Views] downsampling=1 interpolation=[Linear Interpolation] pixel_type=[16-bit unsigned integer] interest_points_for_non_rigid=[-= Disable Non-Rigid =-] produce=[Each timepoint & channel] fused_image=[ZARR/N5/HDF5 export using N5-API] define_input=[Auto-load from input data (values shown below)] export=HDF5 hdf5_file={path}/fused.h5 hdf5_base_dataset=/ hdf5_dataset_extension=/s0");

"""
Path("scripts/stitch.ijm").write_text(macro)

# %%

saver = f"""

"""
Path("scripts/saver.ijm").write_text(macro)

# %%
import tifffile

imgs = [tifffile.imread(Path(path) / f"fused_tp_0_ch_{i}.tif") for i in range(3)]
# %%

imgs[0] = (np.clip(imgs[0], 0, 16383) // 64).astype(np.uint8)
imgs[1] = (np.clip(imgs[1], 0, 32767) // 128).astype(np.uint8)
imgs[2] = (np.clip(imgs[1], 0, 65535) // 256).astype(np.uint8)

# %%
import numpy as np

st = np.stack(imgs)
# %%
tifffile.imwrite("fused.tif", st, compression="lzw")

# %%
