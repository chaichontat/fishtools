import datetime
import functools
import getpass
import glob
import os
import pathlib
import tempfile
from pathlib import Path

import cellpose.io
import cellpose.models
import dask
import dask_image.ndmeasure
import dask_jobqueue
import distributed
import imagecodecs
import numpy as np
import scipy
import tifffile
import yaml
import zarr

from fishtools.utils.pretty_print import progress_bar


######################## File format functions ################################
def numpy_array_to_zarr(write_path: Path | str, array, chunks):
    """
    Store an in memory numpy array to disk as a chunked Zarr array

    Parameters
    ----------
    write_path : string
        Filepath where Zarr array will be created

    array : numpy.ndarray
        The already loaded in-memory numpy array to store as zarr

    chunks : tuple, must be array.ndim length
        How the array will be chunked in the Zarr array

    Returns
    -------
    zarr.core.Array
        A read+write reference to the zarr array on disk
    """

    zarr_array = zarr.open(
        write_path,
        mode="w",
        shape=array.shape,
        chunks=chunks,
        dtype=array.dtype,
    )
    zarr_array[...] = array
    return zarr_array


def wrap_folder_of_tiffs(
    filename_pattern,
    block_index_pattern=r"_(Z)(\d+)(Y)(\d+)(X)(\d+)",
):
    """
    Wrap a folder of tiff files with a zarr array without duplicating data.
    Tiff files must all contain images with the same shape and data type.
    Tiff file names must contain a pattern indicating where individual files
    lie in the block grid.

    Distributed computing requires parallel access to small regions of your
    image from different processes. This is best accomplished with chunked
    file formats like Zarr and N5. This function can accommodate a folder of
    tiff files, but it is not equivalent to reformating your data as Zarr or N5.
    If your individual tiff files/tiles are huge, distributed performance will
    be poor or not work at all.

    It does not make sense to use this function if you have only one tiff file.
    That tiff file will become the only chunk in the zarr array, which means all
    workers will have to load the entire image to fetch their crop of data anyway.
    If you have a single tiff image, you should just reformat it with the
    numpy_array_to_zarr function. Single tiff files too large to fit into system
    memory are not be supported.

    Parameters
    ----------
    filename_pattern : string
        A glob pattern that will match all needed tif files

    block_index_pattern : regular expression string (default: r'_(Z)(\d+)(Y)(\d+)(X)(\d+)')
        A regular expression pattern that indicates how to parse tiff filenames
        to determine where each tiff file lies in the overall block grid
        The default pattern assumes filenames like the following:
            {any_prefix}_Z000Y000X000{any_suffix}
            {any_prefix}_Z000Y000X001{any_suffix}
            ... and so on

    Returns
    -------
    zarr.core.Array
    """

    # define function to read individual files
    def imread(fname):
        with open(fname, "rb") as fh:
            return imagecodecs.tiff_decode(fh.read(), index=None)

    # create zarr store, open it as zarr array and return
    store = tifffile.imread(
        filename_pattern,
        aszarr=True,
        imread=imread,
        pattern=block_index_pattern,
        axestiled={x: x for x in range(3)},
    )
    return zarr.open(store=store)


######################## Cluster related functions ############################

# ----------------------- config stuff ----------------------------------------#
DEFAULT_CONFIG_FILENAME = "distributed_cellpose_dask_config.yaml"


def _config_path(config_name):
    """Add config directory path to config filename"""
    return str(pathlib.Path.home()) + "/.config/dask/" + config_name


def _modify_dask_config(
    config,
    config_name=DEFAULT_CONFIG_FILENAME,
):
    """
    Modifies dask config dictionary, but also dumps modified
    config to disk as a yaml file in ~/.config/dask/. This
    ensures that workers inherit config options.
    """
    dask.config.set(config)
    with open(_config_path(config_name), "w") as f:
        yaml.dump(dask.config.config, f, default_flow_style=False)


def _remove_config_file(
    config_name=DEFAULT_CONFIG_FILENAME,
):
    """Removes a config file from disk"""
    config_path = _config_path(config_name)
    if os.path.exists(config_path):
        os.remove(config_path)


# ----------------------- clusters --------------------------------------------#
class myLocalCluster(distributed.LocalCluster):
    """
    This is a thin wrapper extending dask.distributed.LocalCluster to set
    configs before the cluster or workers are initialized.

    For a list of full arguments (how to specify your worker resources) see:
    https://distributed.dask.org/en/latest/api.html#distributed.LocalCluster
    You need to know how many cpu cores and how much RAM your machine has.

    Most users will only need to specify:
    n_workers
    ncpus (number of physical cpu cores per worker)
    memory_limit (which is the limit per worker, should be a string like '16GB')
    threads_per_worker (for most workflows this should be 1)

    You can also modify any dask configuration option through the
    config argument.

    If your workstation has a GPU, one of the workers will have exclusive
    access to it by default. That worker will be much faster than the others.
    You may want to consider creating only one worker (which will have access
    to the GPU) and letting that worker process all blocks serially.
    """

    def __init__(
        self,
        ncpus,
        config={},
        config_name=DEFAULT_CONFIG_FILENAME,
        persist_config=False,
        **kwargs,
    ):
        # config
        self.config_name = config_name
        self.persist_config = persist_config
        scratch_dir = f"{os.getcwd()}/"
        scratch_dir += f".{getpass.getuser()}_distributed_cellpose/"
        config_defaults = {"temporary-directory": scratch_dir}
        config = {**config_defaults, **config}
        _modify_dask_config(config, config_name)

        # construct
        if "host" not in kwargs:
            kwargs["host"] = ""
        super().__init__(**kwargs)
        self.client = distributed.Client(self)

        # set environment variables for workers (threading)
        environment_vars = {
            "MKL_NUM_THREADS": str(2 * ncpus),
            "NUM_MKL_THREADS": str(2 * ncpus),
            "OPENBLAS_NUM_THREADS": str(2 * ncpus),
            "OPENMP_NUM_THREADS": str(2 * ncpus),
            "OMP_NUM_THREADS": str(2 * ncpus),
        }

        def set_environment_vars():
            for k, v in environment_vars.items():
                os.environ[k] = v

        self.client.run(set_environment_vars)

        print("Cluster dashboard link: ", self.dashboard_link)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if not self.persist_config:
            _remove_config_file(self.config_name)
        self.client.close()
        super().__exit__(exc_type, exc_value, traceback)


# ----------------------- decorator -------------------------------------------#
def cluster(func):
    """
    This decorator ensures a function will run inside a cluster
    as a context manager. The decorated function, "func", must
    accept "cluster" and "cluster_kwargs" as parameters. If
    "cluster" is not None then the user has provided an existing
    cluster and we just run func. If "cluster" is None then
    "cluster_kwargs" are used to construct a new cluster, and
    the function is run inside that cluster context.
    """

    @functools.wraps(func)
    def create_or_pass_cluster(*args, **kwargs):
        # TODO: this only checks if args are explicitly present in function call
        #       it does not check if they are set correctly in any way
        assert "cluster" in kwargs or "cluster_kwargs" in kwargs, (
            "Either cluster or cluster_kwargs must be defined"
        )
        if not "cluster" in kwargs:
            cluster_constructor = myLocalCluster
            F = lambda x: x in kwargs["cluster_kwargs"]
            if F("ncpus") and F("min_workers") and F("max_workers"):
                cluster_constructor = janeliaLSFCluster
            with cluster_constructor(**kwargs["cluster_kwargs"]) as cluster:
                kwargs["cluster"] = cluster
                return func(*args, **kwargs)
        return func(*args, **kwargs)

    return create_or_pass_cluster


######################## the function to run on each block ####################


# ----------------------- The main function -----------------------------------#
def process_block(
    block_index,
    crop,
    input_zarr,
    model_kwargs,
    eval_kwargs,
    blocksize,
    overlap,
    output_zarr,
    preprocessing_steps=[],
    worker_logs_directory=None,
    test_mode=False,
):
    """
    Preprocess and segment one block, of many, with eventual merger
    of all blocks in mind. The block is processed as follows:

    (1) Read block from disk, preprocess, and segment.
    (2) Remove overlaps.
    (3) Get bounding boxes for every segment.
    (4) Remap segment IDs to globally unique values.
    (5) Write segments to disk.
    (6) Get segmented block faces.

    A user may want to test this function on one block before running
    the distributed function. When test_mode=True, steps (5) and (6)
    are omitted and replaced with:

    (5) return remapped segments as a numpy array, boxes, and box_ids

    Parameters
    ----------
    block_index : tuple
        The (i, j, k, ...) index of the block in the overall block grid

    crop : tuple of slice objects
        The bounding box of the data to read from the input_zarr array

    input_zarr : zarr.core.Array
        The image data we want to segment

    preprocessing_steps : list of tuples (default: the empty list)
        Optionally apply an arbitrary pipeline of preprocessing steps
        to the image block before running cellpose.

        Must be in the following format:
        [(f, {'arg1':val1, ...}), ...]
        That is, each tuple must contain only two elements, a function
        and a dictionary. The function must have the following signature:
        def F(image, ..., crop=None)
        That is, the first argument must be a numpy array, which will later
        be populated by the image data. The function must also take a keyword
        argument called crop, even if it is not used in the function itself.
        All other arguments to the function are passed using the dictionary.
        Here is an example:

        def F(image, sigma, crop=None):
            return gaussian_filter(image, sigma)
        def G(image, radius, crop=None):
            return median_filter(image, radius)
        preprocessing_steps = [(F, {'sigma':2.0}), (G, {'radius':4})]

    model_kwargs : dict
        Arguments passed to cellpose.models.Cellpose
        This is how you select and parameterize a model.

    eval_kwargs : dict
        Arguments passed to the eval function of the Cellpose model
        This is how you parameterize model evaluation.

    blocksize : iterable (list, tuple, np.ndarray)
        The number of voxels (the shape) of blocks without overlaps

    overlap : int
        The number of voxels added to the blocksize to provide context
        at the edges

    output_zarr : zarr.core.Array
        A location where segments can be stored temporarily before
        merger is complete

    worker_logs_directory : string (default: None)
        A directory path where log files for each worker can be created
        The directory must exist

    test_mode : bool (default: False)
        The primary use case of this function is to be called by
        distributed_eval (defined later in this same module). However
        you may want to call this function manually to test what
        happens to an individual block; this is a good idea before
        ramping up to process big data and also useful for debugging.

        When test_mode is False (default) this function stores
        the segments and returns objects needed for merging between
        blocks.

        When test_mode is True this function does not store the
        segments, and instead returns them to the caller as a numpy
        array. The boxes and box IDs are also returned. When test_mode
        is True, you can supply dummy values for many of the inputs,
        such as:

        block_index = (0, 0, 0)
        output_zarr=None

    Returns
    -------
    If test_mode == False (the default), three things are returned:
        faces : a list of numpy arrays - the faces of the block segments
        boxes : a list of crops (tuples of slices), bounding boxes of segments
        box_ids : 1D numpy array, parallel to boxes, the segment IDs of the
                  boxes

    If test_mode == True, three things are returned:
        segments : np.ndarray containing the segments with globally unique IDs
        boxes : a list of crops (tuples of slices), bounding boxes of segments
        box_ids : 1D numpy array, parallel to boxes, the segment IDs of the
                  boxes
    """
    print("RUNNING BLOCK: ", block_index, "\tREGION: ", crop, flush=True)
    segmentation_3d = read_preprocess_and_segment(
        input_zarr,
        crop,
        preprocessing_steps,
        model_kwargs,
        eval_kwargs,
        worker_logs_directory,
    )
    print(f"Block {block_index}: Segmentation result shape (3D): {segmentation_3d.shape}")

    # --- Define 3D spatial parameters for overlap removal and writing ---
    spatial_crop_slices = crop[:-1]  # Get ZYX slices from the 4D crop
    spatial_blocksize = blocksize[:-1]  # Get ZYX blocksize
    print(f"Block {block_index}: Spatial crop (3D): {spatial_crop_slices}")
    print(f"Block {block_index}: Spatial blocksize (3D): {spatial_blocksize}")

    # --- Remove overlaps on the 3D segmentation ---
    # Pass 3D segmentation, 3D crop slices, 3D blocksize to remove_overlaps
    segmentation_trimmed_3d, crop_trimmed_3d = remove_overlaps(
        segmentation_3d,  # Operate on 3D data
        spatial_crop_slices,  # Use 3D crop info
        overlap,
        spatial_blocksize,  # Use 3D blocksize info
    )
    crop_trimmed_3d = tuple(crop_trimmed_3d)  # Convert to tuple
    print(f"Block {block_index}: Trimmed segmentation shape (3D): {segmentation_trimmed_3d.shape}")
    print(f"Block {block_index}: Trimmed crop (3D): {crop_trimmed_3d}")

    # --- Calculate bounding boxes (3D) ---
    # Pass 3D trimmed segmentation and 3D trimmed crop
    boxes = bounding_boxes_in_global_coordinates(segmentation_trimmed_3d, crop_trimmed_3d)

    # --- Calculate global IDs (operates on labels, dimensions not critical here) ---
    # Need nblocks in 3D for correct mapping if using ravel_multi_index on 3D indices
    nblocks_3d = get_nblocks(input_zarr.shape[:-1], spatial_blocksize)  # Use 3D shapes
    # Assuming block_index is ZYX index (needs verification how it's generated/used)
    # If block_index is ZYXC, need to adapt. Let's assume it corresponds to ZYX blocks for now.
    # If block_indices were generated using 4D nblocks, adjust here:
    nblocks_4d = get_nblocks(input_zarr.shape, blocksize)  # Original 4D nblocks
    block_index_3d = block_index[:-1]  # Assume we only need ZYX index part

    segmentation_global_3d, remap = global_segment_ids(segmentation_trimmed_3d, block_index_3d, nblocks_3d)

    # Extract non-zero IDs for this block from the *final* global 3D segmentation
    final_unique_ids_3d = np.unique(segmentation_global_3d)
    box_ids_for_this_block = final_unique_ids_3d[final_unique_ids_3d != 0]

    if test_mode:
        return segmentation_global_3d, boxes, box_ids_for_this_block  # Return 3D results

    # --- Write 3D results to 3D output Zarr ---
    print(f"Block {block_index}: Writing to output Zarr at {crop_trimmed_3d}")
    output_zarr[crop_trimmed_3d] = segmentation_global_3d  # 3D[3D_slice] = 3D array (This should work)

    # --- Calculate faces (3D) ---
    faces = block_faces(segmentation_global_3d)  # Faces are 3D

    # Return 3D faces, 3D boxes, and corresponding IDs
    return faces, boxes, box_ids_for_this_block


# ----------------------- component functions ---------------------------------#
def read_preprocess_and_segment(
    input_zarr,
    crop,
    preprocessing_steps,
    model_kwargs,
    eval_kwargs,
    worker_logs_directory,
):
    """Read block from zarr array, run all preprocessing steps, run cellpose"""
    image = input_zarr[crop]
    for pp_step in preprocessing_steps:
        pp_step[1]["crop"] = crop
        image = pp_step[0](image, **pp_step[1])
    log_file = None
    if worker_logs_directory is not None:
        log_file = f"dask_worker_{distributed.get_worker().name}.log"
        log_file = pathlib.Path(worker_logs_directory).joinpath(log_file)
    cellpose.io.logger_setup(stdout_file_replacement=log_file)
    model = cellpose.models.CellposeModel(**model_kwargs)
    return model.eval(image, **eval_kwargs)[0].astype(np.uint32)


def remove_overlaps(array, crop, overlap, blocksize):
    """overlaps only there to provide context for boundary voxels
    and can be removed after segmentation is complete
    reslice array to remove the overlaps"""
    crop_trimmed = list(crop)
    for axis in range(array.ndim):
        if crop[axis].start != 0:
            slc = [
                slice(None),
            ] * array.ndim
            slc[axis] = slice(overlap, None)
            array = array[tuple(slc)]
            a, b = crop[axis].start, crop[axis].stop
            crop_trimmed[axis] = slice(a + overlap, b)
        if array.shape[axis] > blocksize[axis]:
            slc = [
                slice(None),
            ] * array.ndim
            slc[axis] = slice(None, blocksize[axis])
            array = array[tuple(slc)]
            a = crop_trimmed[axis].start
            crop_trimmed[axis] = slice(a, a + blocksize[axis])
    return array, crop_trimmed


def bounding_boxes_in_global_coordinates(segmentation, crop):
    """bounding boxes (tuples of slices) are super useful later
    best to compute them now while things are distributed"""
    boxes = scipy.ndimage.find_objects(segmentation)
    boxes = [b for b in boxes if b is not None]
    translate = lambda a, b: slice(a.start + b.start, a.start + b.stop)
    for iii, box in enumerate(boxes):
        boxes[iii] = tuple(translate(a, b) for a, b in zip(crop, box))
    return boxes


def get_nblocks(shape, blocksize):
    """Given a shape and blocksize determine the number of blocks per axis"""
    return np.ceil(np.array(shape) / blocksize).astype(int)


def global_segment_ids(segmentation, block_index, nblocks):
    """pack the block index into the segment IDs so they are
    globally unique. Everything gets remapped to [1..N] later.
    A uint32 is split into 5 digits on left and 5 digits on right.
    This creates limits: 42950 maximum number of blocks and
    99999 maximum number of segments per block"""
    unique, unique_inverse = np.unique(segmentation, return_inverse=True)
    p = str(np.ravel_multi_index(block_index, nblocks))
    remap = [np.uint32(p + str(x).zfill(5)) for x in unique]
    if unique[0] == 0:
        remap[0] = np.uint32(0)  # 0 should just always be 0
    segmentation = np.array(remap)[unique_inverse.reshape(segmentation.shape)]
    return segmentation, remap


def block_faces(segmentation):
    """slice faces along every axis"""
    faces = []
    for iii in range(segmentation.ndim):
        a = [
            slice(None),
        ] * segmentation.ndim
        a[iii] = slice(0, 1)
        faces.append(segmentation[tuple(a)])
        a = [
            slice(None),
        ] * segmentation.ndim
        a[iii] = slice(-1, None)
        faces.append(segmentation[tuple(a)])
    return faces


######################## Distributed Cellpose #################################


# ----------------------- The main function -----------------------------------#
@cluster
def distributed_eval(
    input_zarr,
    blocksize,
    write_path,
    mask=None,
    preprocessing_steps=[],
    model_kwargs={},
    eval_kwargs={},
    cluster: myLocalCluster = None,
    cluster_kwargs={},
    temporary_directory: Path = None,
):
    """
    Evaluate a cellpose model on overlapping blocks of a big image.
    Distributed over workstation or cluster resources with Dask.
    Optionally run preprocessing steps on the blocks before running cellpose.
    Optionally use a mask to ignore background regions in image.
    Either cluster or cluster_kwargs parameter must be set to a
    non-default value; please read these parameter descriptions below.
    If using cluster_kwargs, the workstation and Janelia LSF cluster cases
    are distinguished by the arguments present in the dictionary.

    PC/Mac/Linux workstations and the Janelia LSF cluster are supported;
    running on a different institute cluster will require implementing your
    own dask cluster class. Look at the JaneliaLSFCluster class in this
    module as an example, also look at the dask_jobqueue library. A PR with
    a solid start is the right way to get help running this on your own
    institute cluster.

    If running on a workstation, please read the docstring for the
    LocalCluster class defined in this module. That will tell you what to
    put in the cluster_kwargs dictionary. If using the Janelia cluster,
    please read the docstring for the JaneliaLSFCluster class.

    Parameters
    ----------
    input_zarr : zarr.core.Array
        A zarr.core.Array instance containing the image data you want to
        segment.

    blocksize : iterable
        The size of blocks in voxels. E.g. [128, 256, 256]

    write_path : string
        The location of a zarr file on disk where you'd like to write your results

    mask : numpy.ndarray (default: None)
        A foreground mask for the image data; may be at a different resolution
        (e.g. lower) than the image data. If given, only blocks that contain
        foreground will be processed. This can save considerable time and
        expense. It is assumed that the domain of the input_zarr image data
        and the mask is the same in physical units, but they may be on
        different sampling/voxel grids.

    preprocessing_steps : list of tuples (default: the empty list)
        Optionally apply an arbitrary pipeline of preprocessing steps
        to the image blocks before running cellpose.

        Must be in the following format:
        [(f, {'arg1':val1, ...}), ...]
        That is, each tuple must contain only two elements, a function
        and a dictionary. The function must have the following signature:
        def F(image, ..., crop=None)
        That is, the first argument must be a numpy array, which will later
        be populated by the image data. The function must also take a keyword
        argument called crop, even if it is not used in the function itself.
        All other arguments to the function are passed using the dictionary.
        Here is an example:

        def F(image, sigma, crop=None):
            return gaussian_filter(image, sigma)
        def G(image, radius, crop=None):
            return median_filter(image, radius)
        preprocessing_steps = [(F, {'sigma':2.0}), (G, {'radius':4})]

    model_kwargs : dict (default: {})
        Arguments passed to cellpose.models.Cellpose

    eval_kwargs : dict (default: {})
        Arguments passed to cellpose.models.Cellpose.eval

    cluster : A dask cluster object (default: None)
        Only set if you have constructed your own static cluster. The default
        behavior is to construct a dask cluster for the duration of this function,
        then close it when the function is finished.

    cluster_kwargs : dict (default: {})
        Arguments used to parameterize your cluster.
        If you are running locally, see the docstring for the myLocalCluster
        class in this module. If you are running on the Janelia LSF cluster, see
        the docstring for the janeliaLSFCluster class in this module. If you are
        running on a different institute cluster, you may need to implement
        a dask cluster object that conforms to the requirements of your cluster.

    temporary_directory : string (default: None)
        Temporary files are created during segmentation. The temporary files
        will be in their own folder within the temporary_directory. The default
        is the current directory. Temporary files are removed if the function
        completes successfully.

    Returns
    -------
    Two values are returned:
    (1) A reference to the zarr array on disk containing the stitched cellpose
        segments for your entire image
    (2) Bounding boxes for every segment. This is a list of tuples of slices:
        [(slice(z1, z2), slice(y1, y2), slice(x1, x2)), ...]
        The list is sorted according to segment ID. That is the smallest segment
        ID is the first tuple in the list, the largest segment ID is the last
        tuple in the list.
    """

    timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    worker_logs_dirname = f"dask_worker_logs_{timestamp}"
    worker_logs_dir = pathlib.Path().absolute().joinpath(worker_logs_dirname)
    worker_logs_dir.mkdir()

    if "diameter" not in eval_kwargs.keys():
        eval_kwargs["diameter"] = 30
    overlap = eval_kwargs["diameter"] * 2
    block_indices, block_crops = get_block_crops(
        input_zarr.shape,
        blocksize,
        overlap,
        mask,
    )

    def check_block_has_data(crop, zarr_array, threshold=0):
        """
        Checks if a given crop in a Zarr array contains any data above a threshold.
        Returns True if data exists, False otherwise.
        """
        try:
            # Access the data slice as a Dask array and perform the check
            data_slice = zarr_array[crop]
            if threshold == 0:
                # Efficient check for any non-zero element
                return data_slice.any()  # Compute the single boolean result
            else:
                # Check against the threshold
                return (data_slice > threshold).any()  # Compute the single boolean result
        except Exception as e:
            print(f"Error checking block crop {crop}: {e}")
            # Decide how to handle errors - conservatively assume it *might* have data?
            # Or return False to skip? Let's return False to avoid processing potentially bad blocks.
            return False

    # Use client.map to run the check function on workers
    # Pass necessary arguments: input_zarr and zero_input_threshold
    # Note: Passing large arrays like input_zarr repeatedly can add overhead,
    # but Dask should handle passing the reference efficiently.
    # Alternatively, scatter input_zarr once if performance is an issue.
    offset = 0
    n = None
    check_futures = cluster.client.map(
        check_block_has_data,
        block_crops[offset : None if n is None else offset + n],  # Iterate over crops
        zarr_array=input_zarr,  # Pass input_zarr reference
        threshold=0,  # Pass threshold
    )

    # Gather the boolean results from the futures
    print("Gathering input data check results...")
    try:
        non_zero_results = cluster.client.gather(check_futures, errors="raise")
        print("Input data check results gathered.")
    except Exception as e:
        print(f"Error gathering input check results: {e}")
        print("Skipping zero-input block filtering due to error.")
        raise e

    # Filter the indices and crops based on the results
    final_block_indices, final_block_crops = [], []
    for i, is_non_zero in enumerate(non_zero_results, offset):
        if is_non_zero:
            final_block_indices.append(block_indices[i])
            final_block_crops.append(block_crops[i])
    del block_indices, block_crops

    print(f"Selected {len(final_block_indices)} blocks with non-zero input data.")

    output_shape = input_zarr.shape[:-1]  # Get ZYX dimensions
    output_blocksize = blocksize[:-1]

    Path(temporary_directory).mkdir(parents=True, exist_ok=True)
    assert temporary_directory.exists()
    temp_zarr_path = Path(temporary_directory) / "segmentation_unstitched.zarr"

    temp_zarr = zarr.open(
        temp_zarr_path,
        "w",
        shape=output_shape,  # Use 3D shape
        chunks=output_blocksize,  # Use 3D chunks
        dtype=np.uint32,
    )
    futures = cluster.client.map(
        process_block,
        final_block_indices,
        final_block_crops,
        input_zarr=input_zarr,
        preprocessing_steps=preprocessing_steps,
        model_kwargs=model_kwargs,
        eval_kwargs=eval_kwargs,
        blocksize=blocksize,
        overlap=overlap,
        output_zarr=temp_zarr,
        worker_logs_directory=str(worker_logs_dir),
    )

    with progress_bar(len(final_block_indices)) as submit:
        [fut.add_done_callback(submit) for fut in futures]
        results = cluster.client.gather(futures)

    if isinstance(cluster, dask_jobqueue.core.JobQueueCluster):
        cluster.scale(0)

    faces, boxes_, box_ids_ = list(zip(*results))
    boxes = [box for sublist in boxes_ for box in sublist]
    box_ids = np.concatenate(box_ids_).astype(int)  # unsure how but without cast these are float64

    print(f"Box IDs: {len(box_ids)}")
    new_labeling = determine_merge_relabeling(
        [(bi[0], bi[1], bi[2]) for bi in final_block_indices], faces, box_ids
    )
    new_labeling_path = Path(temporary_directory) / "new_labeling.npy"
    np.save(new_labeling_path, new_labeling)

    # stitching step is cheap, we should release gpus and use small workers
    if isinstance(cluster, dask_jobqueue.core.JobQueueCluster):
        cluster.change_worker_attributes(
            min_workers=cluster.locals_store["min_workers"],
            max_workers=12,
            ncpus=1,
            memory="15GB",
            mem=int(15e9),
            queue=None,
            job_extra_directives=[],
        )
        cluster.scale(32)

    print("Relabeling blocks...")
    segmentation_da = dask.array.from_zarr(temp_zarr)
    relabeled = dask.array.map_blocks(
        lambda block: np.load(new_labeling_path)[block],
        segmentation_da,
        dtype=np.uint32,
        chunks=segmentation_da.chunks,
    )
    dask.array.to_zarr(relabeled, write_path, overwrite=True)

    print("Merging boxes...")
    merged_boxes = merge_all_boxes(boxes, new_labeling[box_ids])
    return zarr.open(write_path, mode="r"), merged_boxes


# ----------------------- component functions ---------------------------------#
def get_block_crops(shape, blocksize, overlap, mask):
    """Given a voxel grid shape, blocksize, and overlap size, construct
    tuples of slices for every block; optionally only include blocks
    that contain foreground in the mask. Returns parallel lists,
    the block indices and the slice tuples."""
    blocksize = np.array(blocksize)
    if mask is not None:
        ratio = np.array(mask.shape) / shape
        mask_blocksize = np.round(ratio * blocksize).astype(int)

    indices, crops = [], []
    nblocks = get_nblocks(shape, blocksize)
    for index in np.ndindex(*nblocks):
        start = blocksize * index - overlap
        stop = start + blocksize + 2 * overlap
        start = np.maximum(0, start)
        stop = np.minimum(shape, stop)
        crop = tuple(slice(x, y) for x, y in zip(start, stop))

        foreground = True
        if mask is not None:
            start = mask_blocksize * index
            stop = start + mask_blocksize
            stop = np.minimum(mask.shape, stop)
            mask_crop = tuple(slice(x, y) for x, y in zip(start, stop))
            if not np.any(mask[mask_crop]):
                foreground = False
        if foreground:
            indices.append(index)
            crops.append(crop)
    return indices, crops


def determine_merge_relabeling(block_indices, faces, used_labels):
    """Determine boundary segment mergers, remap all label IDs to merge
    and put all label IDs in range [1..N] for N global segments found"""
    faces = adjacent_faces(block_indices, faces)
    # FIX float parameters
    # print("Used labels:", used_labels, "Type:", type(used_labels))
    used_labels = used_labels.astype(int)
    # print("Used labels:", used_labels, "Type:", type(used_labels))
    label_range = int(np.max(used_labels))

    label_groups = block_face_adjacency_graph(faces, label_range)
    new_labeling = scipy.sparse.csgraph.connected_components(label_groups, directed=False)[1]
    # XXX: new_labeling is returned as int32. Loses half range. Potentially a problem.
    unused_labels = np.ones(label_range + 1, dtype=bool)
    unused_labels[used_labels] = 0
    new_labeling[unused_labels] = 0
    unique, unique_inverse = np.unique(new_labeling, return_inverse=True)
    new_labeling = np.arange(len(unique), dtype=np.uint32)[unique_inverse]
    return new_labeling


def adjacent_faces(block_indices, faces):
    """Find faces which touch and pair them together in new data structure"""
    face_pairs = []
    faces_index_lookup = {a: b for a, b in zip(block_indices, faces)}
    for block_index in block_indices:
        for ax in range(len(block_index)):
            neighbor_index = np.array(block_index)
            neighbor_index[ax] += 1
            neighbor_index = tuple(neighbor_index)
            try:
                a = faces_index_lookup[block_index][2 * ax + 1]
                b = faces_index_lookup[neighbor_index][2 * ax]
                face_pairs.append(np.concatenate((a, b), axis=ax))
            except KeyError:
                continue
    return face_pairs


def block_face_adjacency_graph(faces, nlabels):
    """Shrink labels in face plane, then find which labels touch across the
    face boundary"""
    # FIX float parameters
    # print("Initial nlabels:", nlabels, "Type:", type(nlabels))
    nlabels = int(nlabels)
    # print("Final nlabels:", nlabels, "Type:", type(nlabels))

    all_mappings = []
    structure = scipy.ndimage.generate_binary_structure(3, 1)
    for face in faces:
        sl0 = tuple(slice(0, 1) if d == 2 else slice(None) for d in face.shape)
        sl1 = tuple(slice(1, 2) if d == 2 else slice(None) for d in face.shape)
        a = shrink_labels(face[sl0], 1.0)
        b = shrink_labels(face[sl1], 1.0)
        face = np.concatenate((a, b), axis=np.argmin(a.shape))
        mapped = dask_image.ndmeasure._utils._label._across_block_label_grouping(face, structure)
        all_mappings.append(mapped)
    i, j = np.concatenate(all_mappings, axis=1)
    v = np.ones_like(i)
    return scipy.sparse.coo_matrix((v, (i, j)), shape=(nlabels + 1, nlabels + 1)).tocsr()


def shrink_labels(plane, threshold):
    """Shrink labels in plane by some distance from their boundary"""
    gradmag = np.linalg.norm(np.gradient(plane.squeeze()), axis=0)
    shrunk_labels = np.copy(plane.squeeze())
    shrunk_labels[gradmag > 0] = 0
    distances = scipy.ndimage.distance_transform_edt(shrunk_labels)
    shrunk_labels[distances <= threshold] = 0
    return shrunk_labels.reshape(plane.shape)


def merge_all_boxes(boxes, box_ids):
    """Merge all boxes that map to the same box_ids"""
    merged_boxes = []
    boxes_array = np.array(boxes, dtype=object)
    # FIX float parameters
    # print("Box IDs:", box_ids, "Type:", type(box_ids))
    box_ids = box_ids.astype(int)
    # print("Box IDs:", box_ids, "Type:", type(box_ids))

    for iii in np.unique(box_ids):
        merge_indices = np.argwhere(box_ids == iii).squeeze()
        if merge_indices.shape:
            merged_box = merge_boxes(boxes_array[merge_indices])
        else:
            merged_box = boxes_array[merge_indices]
        merged_boxes.append(merged_box)
    return merged_boxes


def merge_boxes(boxes):
    """Take union of two or more parallelpipeds"""
    box_union = boxes[0]
    for iii in range(1, len(boxes)):
        local_union = []
        for s1, s2 in zip(box_union, boxes[iii]):
            start = min(s1.start, s2.start)
            stop = max(s1.stop, s2.stop)
            local_union.append(slice(start, stop))
        box_union = tuple(local_union)
    return box_union


# =================== Example Usage ===================
if __name__ == "__main__":
    # ---- 1. Configuration ----
    # path = Path("/working/20250327_benchmark_coronal2/analysis/deconv/stitch--brain+polyA")
    path = Path("/working/20250407_cs3_2/analysis/deconv/stitch--br+atp")
    tiff_path = path / "fused.tif"  # <-- SET THIS: Path to your input TIFF file
    zarr_input_path = path / "input_image.zarr"  # Path to store intermediate Zarr
    zarr_output_path = path / "output_segmentation.zarr"  # <-- SET THIS: Path for final segmentation Zarr

    # Define block size (e.g., for a 3D image Z, Y, X) - ADJUST TO YOUR DATA AND MEMORY
    # Should be large enough for context but small enough for memory per worker
    processing_blocksize = (20, 512, 512, 2)  # Example for 3D

    # Cellpose Model Configuration
    # cellpose_model_kwargs = {
    #     "pretrained_model": "/working/20250327_benchmark_coronal2/analysis/deconv/segment--brain+polyA/models/2025-04-16_21-32-16",  # Or 'nuclei', 'cyto', or path to custom model
    #     "pretrained_model_ortho": "/working/20250327_benchmark_coronal2/analysis/deconv/segment--brain+polyA/ortho/models/2025-04-14_18-38-00",
    #     "gpu": True,
    # }
    import json

    config = json.loads((path.parent / "config.json").read_text())
    cellpose_model_kwargs = {
        "pretrained_model": config["pretrained_model"],  # Or 'nuclei', 'cyto', or path to custom model
        "pretrained_model_ortho": config["pretrained_model_ortho"],
        "gpu": True,
    }

    # Cellpose Evaluation Configuration
    cellpose_eval_kwargs = {
        "diameter": 31,  # MUST BE INT <-- SET THIS: Estimated average object diameter in pixels
        "channels": [1, 2],  # Use [0,0] for grayscale, [1,2] for R=cyto G=nucleus etc.
        "batch_size": 16,  # Adjust based on GPU memory (if using GPU)
        # Use Cellpose's internal normalization (or False if pre-normalized)
        # "normalize": {
        # "lowhigh": config["normalize"]["lowhigh"],
        # },
        "normalize": {
            "lowhigh": [
                [2094.82608835, 6389.98397762],
                [1519.15741648, 2441.97173317],
            ],  # [[1439.001281128774, 6329.462243341022], [1134.1972039273114, 2175.110389829798]]
        },
        "flow_threshold": 0,  # Useless in 3D
        "cellprob_threshold": -2,  # Default is 0.0, adjust if needed
        "anisotropy": 3.0,
        "resample": False,
        "flow3D_smooth": 3,
        "do_3D": True,
    }

    # Dask Cluster Configuration (for local machine)
    # Adjust n_workers and memory_limit based on your system (CPU cores, RAM, GPU)
    # If gpu=True in model_kwargs, often n_workers=1 is best.
    local_cluster_kwargs = {
        "n_workers": 4,  # Example: 1 worker for GPU, 4 for CPU
        "ncpus": 4,  # Physical cores per worker (adjust based on your CPU)
        "memory_limit": "64GB",  # RAM per worker - CRITICAL: Must fit block+model+overhead
        "threads_per_worker": 1,  # Usually 1 for Cellpose
        # 'config': {'distributed.worker.resources.GPU': 1} # Optional: Explicitly assign GPU resource if needed
    }

    # Optional Preprocessing (Example: Gaussian Blur)
    # from scipy.ndimage import gaussian_filter
    # def apply_gaussian(image, sigma=1, crop=None): # Must accept crop kwarg
    #     print(f"Applying Gaussian blur (sigma={sigma}) to block at {crop}")
    #     return gaussian_filter(image, sigma=sigma)
    #
    # preprocessing_pipeline = [
    #      (apply_gaussian, {'sigma': 1.5}),
    #      # Add other steps here if needed
    # ]
    from skimage.filters import unsharp_mask

    def unsharp_all(img: np.ndarray, crop: None = None):
        return unsharp_mask(img, preserve_range=True, radius=3, channel_axis=3)

    preprocessing_pipeline = [(unsharp_all, {})]

    # Optional Mask (set path to a mask file if you have one)
    mask_path = None  # Example: "path/to/foreground_mask.tif"
    foreground_mask = None
    if mask_path and os.path.exists(mask_path):
        print(f"Loading mask from {mask_path}")
        foreground_mask = tifffile.imread(mask_path) > 0  # Assuming mask is binary or thresholded

    # ---- 2. Convert TIFF to Zarr ----
    print(f"Loading TIFF: {tiff_path}")
    if not os.path.exists(zarr_input_path):
        try:
            image_data = tifffile.imread(tiff_path).transpose(0, 2, 3, 1)
            print(f"Image loaded. Shape: {image_data.shape}, dtype: {image_data.dtype}")

            # Ensure blocksize matches image dimensions
            if len(processing_blocksize) != image_data.ndim:
                raise ValueError(
                    f"Blocksize dimension {len(processing_blocksize)} != Image dimension {image_data.ndim}"
                )

            print(f"Converting to Zarr: {zarr_input_path} with chunks={processing_blocksize}")
            # Use the processing blocksize as the chunk size for the input Zarr
            input_zarr_array = numpy_array_to_zarr(zarr_input_path, image_data, chunks=processing_blocksize)
            print("Zarr conversion complete.")

        except FileNotFoundError:
            print(f"Error: Input TIFF file not found at {tiff_path}")
            exit()
        except Exception as e:
            print(f"Error during TIFF loading or Zarr conversion: {e}")
            exit()
    else:
        input_zarr_array = zarr.open_array(zarr_input_path, mode="r")

    # ---- 3. Run Distributed Evaluation ----
    print("Starting distributed Cellpose evaluation...")
    try:
        # The @cluster decorator handles cluster creation/management
        final_segmentation_zarr, final_bounding_boxes = distributed_eval(
            input_zarr=input_zarr_array,
            blocksize=processing_blocksize,
            write_path=zarr_output_path,
            mask=foreground_mask,
            preprocessing_steps=preprocessing_pipeline,
            model_kwargs=cellpose_model_kwargs,
            eval_kwargs=cellpose_eval_kwargs,
            cluster_kwargs=local_cluster_kwargs,  # Pass kwargs for cluster creation
            temporary_directory=path / "cellpose_temp",  # Optional: specify temp dir location
        )

        print("\n--- Run Finished ---")
        print(f"Final segmentation saved to: {zarr_output_path}")
        print(f"Output Zarr shape: {final_segmentation_zarr.shape}, dtype: {final_segmentation_zarr.dtype}")
        print(f"Number of segmented objects found: {len(final_bounding_boxes)}")
        # You can now inspect the zarr_output_path (e.g., with napari-zarr)
        # final_bounding_boxes contains the list of bounding boxes for labels [1, 2, ...]

        # Example: Print the first 5 bounding boxes
        # for i, bbox in enumerate(final_bounding_boxes[:5]):
        #     print(f"  Label {i+1}: {bbox}")

    except Exception as e:
        print(f"\n--- Error during distributed evaluation ---")
        import traceback

        traceback.print_exc()

    finally:
        # Clean up intermediate Zarr input file if desired
        # import shutil
        # if os.path.exists(zarr_input_path):
        #     print(f"Removing intermediate Zarr: {zarr_input_path}")
        #     shutil.rmtree(zarr_input_path)
        print("Script finished.")
