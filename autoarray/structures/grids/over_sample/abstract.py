import numpy as np
from typing import List, Union


from autoconf import conf

from autoarray.mask.mask_2d import Mask2D
from autoarray.structures.arrays.uniform_2d import Array2D
from autoarray.structures.grids.uniform_2d import Grid2D

from autoarray.structures.grids import grid_2d_util
from autoarray.mask.mask_2d import mask_2d_util
from autoarray.numpy_wrapper import numpy as npw


class AbstractOverSample:
    def sub_length_from(self, mask, sub_size: int) -> int:
        """
        The total number of sub-pixels in a give pixel,

        For example, a sub-size of 3x3 means every pixel has 9 sub-pixels.
        """
        return int(sub_size**mask.dimensions)

    def sub_fraction_from(self, mask, sub_size: int) -> float:
        """
        The fraction of the area of a pixel every sub-pixel contains.

        For example, a sub-size of 3x3 mean every pixel contains 1/9 the area.
        """
        sub_length = self.sub_length_from(sub_size=sub_size, mask=mask)

        return 1.0 / sub_length

    def sub_pixels_in_mask_from(self, mask, sub_size: int) -> int:
        """
        The total number of unmasked sub-pixels (values are `False`) in the mask.
        """
        return sub_size**mask.dimensions * mask.pixels_in_mask

    def oversampled_grid_2d_via_mask_from(self, mask: Mask2D, sub_size: int) -> Grid2D:
        sub_grid_1d = grid_2d_util.grid_2d_slim_via_mask_from(
            mask_2d=np.array(mask),
            pixel_scales=mask.pixel_scales,
            sub_size=sub_size,
            origin=mask.origin,
        )

        over_sample_mask = mask_2d_util.oversample_mask_2d_from(
            mask=np.array(mask), sub_size=sub_size
        )

        pixel_scales = (
            mask.pixel_scales[0] / sub_size,
            mask.pixel_scales[1] / sub_size,
        )

        mask = Mask2D(
            mask=over_sample_mask, pixel_scales=pixel_scales, origin=mask.origin
        )

        return Grid2D(values=sub_grid_1d, mask=mask, over_sample=self)

    def binned_array_2d_from(self, array: Array2D, sub_size: int) -> "Array2D":
        """
        Convenience method to access the binned-up array in its 1D representation, which is a Grid2D stored as an
        ``ndarray`` of shape [total_unmasked_pixels, 2].

        The binning up process converts a array from (y,x) values where each value is a coordinate on the sub-array to
        (y,x) values where each coordinate is at the centre of its mask (e.g. a array with a sub_size of 1). This is
        performed by taking the mean of all (y,x) values in each sub pixel.

        If the array is stored in 1D it is return as is. If it is stored in 2D, it must first be mapped from 2D to 1D.

        In **PyAutoCTI** all `Array2D` objects are used in their `native` representation without sub-gridding.
        Significant memory can be saved by only store this format, thus the `native_binned_only` config override
        can force this behaviour. It is recommended users do not use this option to avoid unexpected behaviour.
        """
        if conf.instance["general"]["structures"]["native_binned_only"]:
            return self

        sub_length = self.sub_length_from(mask=array.mask, sub_size=sub_size)
        sub_fraction = self.sub_fraction_from(mask=array.mask, sub_size=sub_size)

        array_2d_slim = array.slim

        binned_array_1d = npw.multiply(
            sub_fraction,
            array_2d_slim.reshape(-1, sub_length).sum(axis=1),
        )

        return Array2D(
            values=binned_array_1d,
            mask=array.mask[::sub_size, ::sub_size],
            header=array.header,
        )

    def evaluated_func_obj_from(self, func, cls, mask, sub_size):
        grid = self.oversampled_grid_2d_via_mask_from(mask=mask, sub_size=sub_size)
        values = func(cls, np.asarray(grid))

        if not isinstance(values, list):
            values = grid.over_sample.structure_2d_from(result=values, mask=grid.mask)
            return self.binned_array_2d_from(array=values, sub_size=sub_size)
        else:
            values_list = []
            for value in values:
                value = grid.over_sample.structure_2d_from(result=value, mask=grid.mask)
                values_list.append(
                    self.binned_array_2d_from(array=value, sub_size=sub_size)
                )
            return values_list

    def evaluated_func_from(self, func, mask, sub_size):
        grid = self.oversampled_grid_2d_via_mask_from(mask=mask, sub_size=sub_size)
        values = func(grid=grid, profile=None)
        values = grid.over_sample.structure_2d_from(result=values, mask=grid.mask)
        return self.binned_array_2d_from(array=values, sub_size=sub_size)

    def structure_2d_from(
        self, result: np.ndarray, mask: Mask2D
    ) -> Union[Array2D, "Grid2D"]:
        """
        Convert a result from an ndarray to an aa.Array2D or aa.Grid2D structure, where the conversion depends on
        type(result) as follows:

        - 1D np.ndarray   -> aa.Array2D
        - 2D np.ndarray   -> aa.Grid2D

        This function is used by the grid_2d_to_structure decorator to convert the output result of a function
        to an autoarray structure when a `Grid2D` instance is passed to the decorated function.

        Parameters
        ----------
        result or [np.ndarray]
            The input result (e.g. of a decorated function) that is converted to a PyAutoArray structure.
        """
        from autoarray.structures.grids.transformed_2d import Grid2DTransformed
        from autoarray.structures.grids.transformed_2d import Grid2DTransformedNumpy

        if len(result.shape) == 1:
            return Array2D(values=result, mask=mask)
        else:
            if isinstance(result, Grid2DTransformedNumpy):
                return Grid2DTransformed(values=result, mask=mask, over_sample=self)
            return Grid2D(values=result, mask=mask, over_sample=self)

    def structure_2d_list_from(
        self, result_list: List, mask: Mask2D
    ) -> List[Union[Array2D, "Grid2D"]]:
        """
        Convert a result from a list of ndarrays to a list of aa.Array2D or aa.Grid2D structure, where the conversion
        depends on type(result) as follows:

        - [1D np.ndarray] -> [aa.Array2D]
        - [2D np.ndarray] -> [aa.Grid2D]

        This function is used by the grid_like_list_to_structure-list decorator to convert the output result of a
        function to a list of autoarray structure when a `Grid2D` instance is passed to the decorated function.

        Parameters
        ----------
        result_list or [np.ndarray]
            The input result (e.g. of a decorated function) that is converted to a PyAutoArray structure.
        """
        return [
            self.structure_2d_from(result=result, mask=mask) for result in result_list
        ]
