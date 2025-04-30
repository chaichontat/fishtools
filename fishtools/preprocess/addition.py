from copy import deepcopy
from typing import Optional

import numpy as np
import xarray as xr
from starfish.core.image.Filter._base import FilterAlgorithm
from starfish.core.imagestack.imagestack import ImageStack
from starfish.core.types import Levels
from starfish.core.util.levels import levels


class ElementWiseAddition(FilterAlgorithm):
    """
    Perform element-wise addition on the image tensor.

    Parameters
    ----------
    add_array : xr.DataArray
        the image is element-wise added with this array
    level_method : :py:class:`~starfish.types.Levels`
        Controls the way that data are scaled to retain skimage dtype requirements that float data
        fall in [0, 1].  In all modes, data below 0 are set to 0.

        - Levels.CLIP (default): data above 1 are set to 1.
        - Levels.SCALE_SATURATED_BY_IMAGE: when any data in the entire ImageStack is greater
          than 1, the entire ImageStack is scaled by the maximum value in the ImageStack.
        - Levels.SCALE_BY_IMAGE: scale the entire ImageStack by the maximum value in the
          ImageStack.
    """

    def __init__(
        self,
        add_array: xr.DataArray,
        level_method: Levels = Levels.CLIP,
    ) -> None:
        self.add_array = add_array
        self.level_method = level_method
        if self.level_method in (Levels.SCALE_BY_CHUNK, Levels.SCALE_SATURATED_BY_CHUNK):
            raise ValueError("`scale_by_chunk` is not a valid clip_method for ElementWiseAddition")

    _DEFAULT_TESTING_PARAMETERS = {
        "add_array": xr.DataArray(np.array([[[[[0.1]]], [[[0.2]]]]]), dims=("r", "c", "z", "y", "x"))
    }

    def run(
        self,
        stack: ImageStack,
        in_place: bool = False,
        verbose: bool = False,
        n_processes: Optional[int] = None,
        *args,
    ) -> Optional[ImageStack]:
        """Perform filtering of an image stack

        Parameters
        ----------
        stack : ImageStack
            Stack to be filtered.
        in_place : bool
            if True, process ImageStack in-place, otherwise return a new stack
        verbose : bool
            if True, report on filtering progress (default = False)
        n_processes : Optional[int]
            Number of parallel processes to devote to applying the filter. If None, defaults to
            the result of os.cpu_count(). (default None)

        Returns
        -------
        ImageStack :
            If in-place is False, return the results of filter as a new stack.  Otherwise return the
            original stack.

        """
        # Align the axes of the addends with ImageStack
        add_array_aligned: np.ndarray = self.add_array.transpose(*stack.xarray.dims).values
        if not in_place:
            stack = deepcopy(stack)
            # Run in-place on the copy
            self.run(stack, in_place=True, verbose=verbose, n_processes=n_processes)
            return stack

        # Perform addition
        stack.xarray.values += add_array_aligned

        # Apply level method
        if self.level_method == Levels.CLIP:
            stack.xarray.values = levels(stack.xarray.values)
        elif self.level_method == Levels.SCALE_BY_IMAGE:
            stack.xarray.values = levels(stack.xarray.values, rescale=True)
        elif self.level_method == Levels.SCALE_SATURATED_BY_IMAGE:
            stack.xarray.values = levels(stack.xarray.values, rescale_saturated=True)
        else:
            raise ValueError(
                f"Unknown level method {self.level_method}. See starfish.types.Levels for valid options"
            )
        return None
