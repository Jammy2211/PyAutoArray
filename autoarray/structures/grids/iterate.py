import numpy as np

from autoarray import decorator_util
from autoarray.structures import abstract_structure, arrays, grids
from autoarray.structures.grids import abstract_grid
from autoarray.mask import mask as msk
from autoarray.util import array_util, grid_util
from autoarray import exc


def sub_steps_from_none(sub_steps):

    if sub_steps is None:
        return [2, 4, 8, 16]
    return sub_steps


class GridIterate(abstract_grid.AbstractGrid):
    def __new__(
        cls,
        grid,
        mask,
        fractional_accuracy=0.9999,
        sub_steps=None,
        store_in_1d=True,
        *args,
        **kwargs,
    ):
        """Represents a grid of coordinates as described for the *Grid* class, but using an iterative sub-grid that
        adapts its resolution when it is input into a function that performs an analytic calculation.

        A *Grid* represents (y,x) coordinates using a sub-grid, where functions are evaluated once at every coordinate
        on the sub-grid and averaged to give a more precise evaluation an analytic function. A *GridIterate* does not
        have a specified sub-grid size, but instead iteratively recomputes the analytic function at increasing sub-grid
        sizes until an input fractional accuracy is reached.

        Iteration is performed on a per (y,x) coordinate basis, such that the sub-grid size will adopt low values
        wherever doing so can meet the fractional accuracy and high values only where it is required to meet the
        fractional accuracy. For functions where a wide range of sub-grid sizes allow fractional accuracy to be met
        this ensures the function can be evaluated accurate in a computaionally efficient manner.

        This overcomes a limitation of the *Grid* class whereby if a small subset of pixels require high levels of
        sub-gridding to be evaluated accuracy, the entire grid would require this sub-grid size thus leading to
        unecessary expensive function evaluations.

        Parameters
        ----------
        grid : np.ndarray
            The (y,x) coordinates of the grid.
        mask : msk.Mask
            The 2D mask associated with the grid, defining the pixels each grid coordinate is paired with and
            originates from.
        fractional_accuracy : float
            The fractional accuracy the function evaluated must meet to be accepted, where this accuracy is the ratio
            of the value at a higher sub_size to othe value computed using the previous sub_size.
        sub_steps : [int] or None
            The sub-size values used to iteratively evaluated the function at high levels of sub-gridding. If None,
            they are setup as the default values [2, 4, 8, 16].
        store_in_1d : bool
            If True, the grid is stored in 1D as an ndarray of shape [total_unmasked_pixels, 2]. If False, it is
            stored in 2D as an ndarray of shape [total_y_pixels, total_x_pixels, 2].
        """

        sub_steps = sub_steps_from_none(sub_steps=sub_steps)
        if store_in_1d and len(grid.shape) != 2:
            raise exc.GridException(
                "An grid input into the grids.Grid.__new__ method has store_in_1d = True but"
                "the input shape of the array is not 1."
            )

        obj = grid.view(cls)
        obj.mask = mask
        obj.store_in_1d = store_in_1d
        obj.grid = grids.Grid.manual_mask(
            grid=np.asarray(obj), mask=mask, store_in_1d=store_in_1d
        )
        obj.fractional_accuracy = fractional_accuracy
        obj.sub_steps = sub_steps
        return obj

    @classmethod
    def manual_1d(
        cls,
        grid,
        shape_2d,
        pixel_scales,
        origin=(0.0, 0.0),
        fractional_accuracy=0.9999,
        sub_steps=None,
        store_in_1d=True,
    ):
        """Create a GridIterate (see *GridIterate.__new__*) by inputting the grid coordinates in 1D, for example:

            grid=np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])

            grid=[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]]

        From 1D input the method cannot determine the 2D shape of the grid and its mask, thus the shape_2d must be
        input into this method. The mask is setup as a unmasked *Mask* of shape_2d.

        Parameters
        ----------
        grid : np.ndarray or list
            The (y,x) coordinates of the grid input as an ndarray of shape [total_unmasked_pixells*(sub_size**2), 2]
            or a list of lists.
        shape_2d : (float, float)
            The 2D shape of the mask the grid is paired with.
        pixel_scales : (float, float) or float
            The pixel conversion scale of a pixel in the y and x directions. If input as a float, the pixel_scales
            are converted to the format (float, float).
        fractional_accuracy : float
            The fractional accuracy the function evaluated must meet to be accepted, where this accuracy is the ratio
            of the value at a higher sub_size to othe value computed using the previous sub_size.
        sub_steps : [int] or None
            The sub-size values used to iteratively evaluated the function at high levels of sub-gridding. If None,
            they are setup as the default values [2, 4, 8, 16].
        origin : (float, float)
            The origin of the grid's mask.
        store_in_1d : bool
            If True, the grid is stored in 1D as an ndarray of shape [total_unmasked_pixels, 2]. If False, it is
            stored in 2D as an ndarray of shape [total_y_pixels, total_x_pixels, 2].
        """
        grid = abstract_grid.convert_grid(grid=grid)
        pixel_scales = abstract_structure.convert_pixel_scales(
            pixel_scales=pixel_scales
        )

        mask = msk.Mask.unmasked(
            shape_2d=shape_2d, pixel_scales=pixel_scales, sub_size=1, origin=origin
        )

        if store_in_1d:
            return GridIterate(
                grid=grid,
                mask=mask,
                fractional_accuracy=fractional_accuracy,
                sub_steps=sub_steps,
                store_in_1d=store_in_1d,
            )

        grid_2d = grid_util.sub_grid_2d_from(sub_grid_1d=grid, mask=mask, sub_size=1)

        return GridIterate(
            grid=grid_2d,
            mask=mask,
            fractional_accuracy=fractional_accuracy,
            sub_steps=sub_steps,
            store_in_1d=store_in_1d,
        )

    @classmethod
    def uniform(
        cls,
        shape_2d,
        pixel_scales,
        origin=(0.0, 0.0),
        fractional_accuracy=0.9999,
        sub_steps=None,
        store_in_1d=True,
    ):
        """Create a GridIterate (see *GridIterate.__new__*) as a uniform grid of (y,x) values given an input
        shape_2d and pixel scale of the grid:

        Parameters
        ----------
        shape_2d : (float, float)
            The 2D shape of the uniform grid and the mask that it is paired with.
        pixel_scales : (float, float) or float
            The pixel conversion scale of a pixel in the y and x directions. If input as a float, the pixel_scales
            are converted to the format (float, float).
        fractional_accuracy : float
            The fractional accuracy the function evaluated must meet to be accepted, where this accuracy is the ratio
            of the value at a higher sub_size to othe value computed using the previous sub_size.
        sub_steps : [int] or None
            The sub-size values used to iteratively evaluated the function at high levels of sub-gridding. If None,
            they are setup as the default values [2, 4, 8, 16].
        origin : (float, float)
            The origin of the grid's mask.
        store_in_1d : bool
            If True, the grid is stored in 1D as an ndarray of shape [total_unmasked_pixels, 2]. If False, it is
            stored in 2D as an ndarray of shape [total_y_pixels, total_x_pixels, 2].
        """

        pixel_scales = abstract_structure.convert_pixel_scales(
            pixel_scales=pixel_scales
        )

        grid_1d = grid_util.grid_1d_via_shape_2d_from(
            shape_2d=shape_2d, pixel_scales=pixel_scales, sub_size=1, origin=origin
        )

        return GridIterate.manual_1d(
            grid=grid_1d,
            shape_2d=shape_2d,
            pixel_scales=pixel_scales,
            fractional_accuracy=fractional_accuracy,
            sub_steps=sub_steps,
            origin=origin,
            store_in_1d=store_in_1d,
        )

    @classmethod
    def from_mask(
        cls, mask, fractional_accuracy=0.9999, sub_steps=None, store_in_1d=True
    ):
        """Create a GridIterate (see *GridIterate.__new__*) from a mask, where only unmasked pixels are included in
        the grid (if the grid is represented in 2D masked values are (0.0, 0.0)).

        The mask's pixel_scales and origin properties are used to compute the grid (y,x) coordinates.

        Parameters
        ----------
        mask : Mask
            The mask whose masked pixels are used to setup the sub-pixel grid.
        fractional_accuracy : float
            The fractional accuracy the function evaluated must meet to be accepted, where this accuracy is the ratio
            of the value at a higher sub_size to othe value computed using the previous sub_size.
        sub_steps : [int] or None
            The sub-size values used to iteratively evaluated the function at high levels of sub-gridding. If None,
            they are setup as the default values [2, 4, 8, 16].
        store_in_1d : bool
            If True, the grid is stored in 1D as an ndarray of shape [total_unmasked_pixels, 2]. If False, it is
            stored in 2D as an ndarray of shape [total_y_pixels, total_x_pixels, 2].
        """

        grid_1d = grid_util.grid_1d_via_mask_from(
            mask=mask, pixel_scales=mask.pixel_scales, sub_size=1, origin=mask.origin
        )

        if store_in_1d:
            return grids.GridIterate(
                grid=grid_1d,
                mask=mask.mask_sub_1,
                fractional_accuracy=fractional_accuracy,
                sub_steps=sub_steps,
                store_in_1d=store_in_1d,
            )

        grid_2d = grid_util.sub_grid_2d_from(
            sub_grid_1d=grid_1d, mask=mask.mask_sub_1, sub_size=1
        )

        return grids.GridIterate(
            grid=grid_2d,
            mask=mask.mask_sub_1,
            fractional_accuracy=fractional_accuracy,
            sub_steps=sub_steps,
            store_in_1d=store_in_1d,
        )

    @classmethod
    def blurring_grid_from_mask_and_kernel_shape(
        cls,
        mask,
        kernel_shape_2d,
        fractional_accuracy=0.9999,
        sub_steps=None,
        store_in_1d=True,
    ):
        """Setup a blurring-grid from a mask, where a blurring grid consists of all pixels that are masked (and
        therefore have their values set to (0.0, 0.0)), but are close enough to the unmasked pixels that their values
        will be convolved into the unmasked those pixels. This occurs in *PyAutoGalaxy* when computing images from
        light profile objects.

        See *grids.Grid.blurring_grid_from_mask_and_kernel_shape* for a full description of a blurring grid. This
        method creates the blurring grid as a GridIterate.

        Parameters
        ----------
        mask : Mask
            The mask whose masked pixels are used to setup the blurring grid.
        kernel_shape_2d : (float, float)
            The 2D shape of the kernel which convolves signal from masked pixels to unmasked pixels.
        fractional_accuracy : float
            The fractional accuracy the function evaluated must meet to be accepted, where this accuracy is the ratio
            of the value at a higher sub_size to othe value computed using the previous sub_size.
        sub_steps : [int] or None
            The sub-size values used to iteratively evaluated the function at high levels of sub-gridding. If None,
            they are setup as the default values [2, 4, 8, 16].
        store_in_1d : bool
            If True, the grid is stored in 1D as an ndarray of shape [total_unmasked_pixels, 2]. If False, it is
            stored in 2D as an ndarray of shape [total_y_pixels, total_x_pixels, 2].
        """

        blurring_mask = mask.regions.blurring_mask_from_kernel_shape(
            kernel_shape_2d=kernel_shape_2d
        )

        return cls.from_mask(
            mask=blurring_mask,
            fractional_accuracy=fractional_accuracy,
            sub_steps=sub_steps,
            store_in_1d=store_in_1d,
        )

    def grid_from_deflection_grid(self, deflection_grid):
        """Compute a new GridIterate from this grid, where the (y,x) coordinates of this grid have a grid of (y,x) values,
         termed the deflection grid, subtracted from them to determine the new grid of (y,x) values.

        This is used by PyAutoLens to perform grid ray-tracing.

        Parameters
        ----------
        deflection_grid : ndarray
            The grid of (y,x) coordinates which is subtracted from this grid.
        """
        return GridIterate(
            grid=self - deflection_grid,
            mask=self.mask,
            fractional_accuracy=self.fractional_accuracy,
            sub_steps=self.sub_steps,
            store_in_1d=self.store_in_1d,
        )

    def blurring_grid_from_kernel_shape(self, kernel_shape_2d):
        """Compute the blurring grid from a grid and create it as a GridIterate, via an input 2D kernel shape.

        For a full description of blurring grids, checkout *blurring_grid_from_mask_and_kernel_shape*.

        Parameters
        ----------
        kernel_shape_2d : (float, float)
            The 2D shape of the kernel which convolves signal from masked pixels to unmasked pixels.
        """

        return GridIterate.blurring_grid_from_mask_and_kernel_shape(
            mask=self.mask,
            kernel_shape_2d=kernel_shape_2d,
            fractional_accuracy=self.fractional_accuracy,
            sub_steps=self.sub_steps,
            store_in_1d=self.store_in_1d,
        )

    def padded_grid_from_kernel_shape(self, kernel_shape_2d):
        """When the edge pixels of a mask are unmasked and a convolution is to occur, the signal of edge pixels will be
        'missing' if the grid is used to evaluate the signal via an analytic function.

        To ensure this signal is included the padded grid is used, which is 'buffed' such that it includes all pixels
        whose signal will be convolved into the unmasked pixels given the 2D kernel shape.

        Parameters
        ----------
        kernel_shape_2d : (float, float)
            The 2D shape of the kernel which convolves signal from masked pixels to unmasked pixels.
        """
        shape = self.mask.shape

        padded_shape = (
            shape[0] + kernel_shape_2d[0] - 1,
            shape[1] + kernel_shape_2d[1] - 1,
        )

        padded_mask = msk.Mask.unmasked(
            shape_2d=padded_shape,
            pixel_scales=self.mask.pixel_scales,
            sub_size=self.mask.sub_size,
        )

        return grids.GridIterate.from_mask(
            mask=padded_mask,
            fractional_accuracy=self.fractional_accuracy,
            sub_steps=self.sub_steps,
        )

    def __array_finalize__(self, obj):

        super(GridIterate, self).__array_finalize__(obj)

        if hasattr(obj, "grid"):
            self.grid = obj.grid

        if hasattr(obj, "fractional_accuracy"):
            self.fractional_accuracy = obj.fractional_accuracy

        if hasattr(obj, "sub_steps"):
            self.sub_steps = obj.sub_steps

    def _new_grid(self, grid, mask, store_in_1d):
        """Conveninence method for creating a new instance of the GridIterate class from this grid.

        This method is used in the 'in_1d', 'in_2d', etc. convenience methods. By overwritin this method such that a
        GridIterate is created the in_1d and in_2d methods will return instances of the GridIterate.

        Parameters
        ----------
        grid : np.ndarray or list
            The (y,x) coordinates of the grid input as an ndarray of shape [total_sub_coordinates, 2] or list of lists.
        mask : msk.Mask
            The 2D mask associated with the grid, defining the pixels each grid coordinate is paired with and
            originates from.
        store_in_1d : bool
            If True, the grid is stored in 1D as an ndarray of shape [total_unmasked_pixels, 2]. If False, it is
            stored in 2D as an ndarray of shape [total_y_pixels, total_x_pixels, 2].
            """
        return GridIterate(
            grid=grid,
            mask=mask,
            fractional_accuracy=self.fractional_accuracy,
            sub_steps=self.sub_steps,
            store_in_1d=store_in_1d,
        )

    @staticmethod
    def array_at_sub_size_from_func_and_mask(func, cls, mask, sub_size):

        mask_higher_sub = mask.mask_new_sub_size_from_mask(mask=mask, sub_size=sub_size)

        grid_compute = grids.Grid.from_mask(mask=mask_higher_sub)
        array_higher_sub = func(cls, grid_compute)
        return grid_compute.structure_from_result(result=array_higher_sub).in_2d_binned

    @staticmethod
    def grid_at_sub_size_from_func_and_mask(func, cls, mask, sub_size):

        mask_higher_sub = mask.mask_new_sub_size_from_mask(mask=mask, sub_size=sub_size)

        grid_compute = grids.Grid.from_mask(mask=mask_higher_sub)
        grid_higher_sub = func(cls, grid_compute)
        return grid_compute.structure_from_result(result=grid_higher_sub).in_2d_binned

    def fractional_mask_from_arrays(
        self, array_lower_sub_2d, array_higher_sub_2d
    ) -> msk.Mask:
        """ Compute a fractional mask from a result array, where the fractional mask describes whether the evaluated
        value in the result array is within the *GridIterate*'s specified fractional accuracy. The fractional mask thus
        determines whether a pixel on the grid needs to be reevaluated at a higher level of sub-gridding to meet the
        specified fractional accuracy. If it must be re-evaluated, the fractional masks's entry is *False*.

        The fractional mask is computed by comparing the results evaluated at one level of sub-gridding to another
        at a higher level of sub-griding. Thus, the sub-grid size in chosen on a per-pixel basis until the function
        is evaluated at the specified fractional accuracy.

        Parameters
        ----------
        array_lower_sub_2d : arrays.Array
            The results computed by a function using a lower sub-grid size
        array_higher_sub_2d : arrays.Array
            The results computed by a function using a higher sub-grid size.
        """

        fractional_mask = msk.Mask.unmasked(
            shape_2d=array_lower_sub_2d.shape_2d, invert=True
        )

        fractional_mask = self.fractional_mask_jit_from_array(
            fractional_accuracy_threshold=self.fractional_accuracy,
            fractional_mask=fractional_mask,
            array_higher_sub_2d=array_higher_sub_2d,
            array_lower_sub_2d=array_lower_sub_2d,
            array_higher_mask=array_higher_sub_2d.mask,
        )

        return msk.Mask(
            mask=fractional_mask,
            pixel_scales=array_higher_sub_2d.pixel_scales,
            origin=array_higher_sub_2d.origin,
        )

    @staticmethod
    @decorator_util.jit()
    def fractional_mask_jit_from_array(
        fractional_accuracy_threshold,
        fractional_mask,
        array_higher_sub_2d,
        array_lower_sub_2d,
        array_higher_mask,
    ):
        """Jitted functioon to determine the fractional mask, which is a mask where:

        - *True* entries signify the function has been evaluated in that pixel to desired fractional accuracy and
           therefore does not need to be iteratively computed at higher levels of sub-gridding.

        - *False* entries signify the function has not been evaluated in that pixel to desired fractional accuracy and
           therefore must be iterative computed at higher levels of sub-gridding to meet this accuracy."""

        for y in range(fractional_mask.shape[0]):
            for x in range(fractional_mask.shape[1]):
                if not array_higher_mask[y, x]:

                    if array_lower_sub_2d[y, x] > 0:

                        fractional_accuracy = (
                            array_lower_sub_2d[y, x] / array_higher_sub_2d[y, x]
                        )

                        if fractional_accuracy > 1.0:
                            fractional_accuracy = 1.0 / fractional_accuracy

                    else:

                        fractional_accuracy = 0.0

                    if fractional_accuracy < fractional_accuracy_threshold:
                        fractional_mask[y, x] = False

        return fractional_mask

    def iterated_array_from_func(self, func, cls, array_lower_sub_2d) -> arrays.Array:
        """Iterate over a function that returns an array of values until the it meets a specified fractional accuracy.
        The function returns a result on a pixel-grid where evaluating it on more points on a higher resolution
        sub-grid followed by binning lead to a more precise evaluation of the function. The function is assumed to 
        belong to a class, which is input into tthe method.

        The function is first called for a sub-grid size of 1 and a higher resolution grid. The ratio of values give
        the fractional accuracy of each function evaluation. Pixels which do not meet the fractional accuracy are
        iteratively revaluated on higher resolution sub-grids. This is repeated until all pixels meet the fractional
        accuracy or the highest sub-size specified in the *sub_steps* attribute is computed.

        If the function return all zeros, the iteration is terminated early given that all levels of sub-gridding will
        return zeros. This occurs when a function is missing optional objects that contribute to the calculation.

        An example use case of this function is when a "image_from_grid" methods in **PyAutoGalaxy**'s
        _LightProfile_ module is comomputed, which by evaluating the function on a higher resolution sub-grids sample
        the analytic light profile at more points and thus more precisely.

        Parameters
        ----------
        func : func
            The function which is iterated over to compute a more precise evaluation.
        cls : cls
            The class the function belongs to.
        grid_lower_sub_2d : arrays.Array
            The results computed by the function using a lower sub-grid size
        """

        if not np.any(array_lower_sub_2d):
            return array_lower_sub_2d.in_1d

        iterated_array = np.zeros(shape=self.shape_2d)

        fractional_mask_lower_sub = self.mask

        for sub_size in self.sub_steps[:-1]:

            array_higher_sub = self.array_at_sub_size_from_func_and_mask(
                func=func, cls=cls, mask=fractional_mask_lower_sub, sub_size=sub_size
            )

            fractional_mask_higher_sub = self.fractional_mask_from_arrays(
                array_lower_sub_2d=array_lower_sub_2d,
                array_higher_sub_2d=array_higher_sub,
            )

            iterated_array = self.iterated_array_jit_from(
                iterated_array=iterated_array,
                fractional_mask_higher_sub=fractional_mask_higher_sub,
                fractional_mask_lower_sub=fractional_mask_lower_sub,
                array_higher_sub_2d=array_higher_sub,
            )

            if fractional_mask_higher_sub.is_all_true:

                iterated_array_1d = array_util.sub_array_1d_from(
                    mask=self.mask, sub_array_2d=iterated_array, sub_size=1
                )

                return arrays.Array(
                    array=iterated_array_1d, mask=self.mask.mask_sub_1, store_in_1d=True
                )

            array_lower_sub_2d = array_higher_sub
            fractional_mask_lower_sub = fractional_mask_higher_sub

        array_higher_sub = self.array_at_sub_size_from_func_and_mask(
            func=func,
            cls=cls,
            mask=fractional_mask_lower_sub,
            sub_size=self.sub_steps[-1],
        )

        iterated_array_2d = iterated_array + array_higher_sub.in_2d_binned

        iterated_array_1d = array_util.sub_array_1d_from(
            mask=self.mask, sub_array_2d=iterated_array_2d, sub_size=1
        )

        return arrays.Array(
            array=iterated_array_1d, mask=self.mask.mask_sub_1, store_in_1d=True
        )

    @staticmethod
    @decorator_util.jit()
    def iterated_array_jit_from(
        iterated_array,
        fractional_mask_higher_sub,
        fractional_mask_lower_sub,
        array_higher_sub_2d,
    ):
        """Create the iterated array from a result array that is computed at a higher sub size leel than the previous grid.

        The iterated array is only updated for pixels where the fractional accuracy is met."""

        for y in range(iterated_array.shape[0]):
            for x in range(iterated_array.shape[1]):
                if (
                    fractional_mask_higher_sub[y, x]
                    and not fractional_mask_lower_sub[y, x]
                ):
                    iterated_array[y, x] = array_higher_sub_2d[y, x]

        return iterated_array

    def fractional_mask_from_grids(
        self, grid_lower_sub_2d, grid_higher_sub_2d
    ) -> msk.Mask:
        """ Compute a fractional mask from a result array, where the fractional mask describes whether the evaluated
        value in the result array is within the *GridIterate*'s specified fractional accuracy. The fractional mask thus
        determines whether a pixel on the grid needs to be reevaluated at a higher level of sub-gridding to meet the
        specified fractional accuracy. If it must be re-evaluated, the fractional masks's entry is *False*.

        The fractional mask is computed by comparing the results evaluated at one level of sub-gridding to another
        at a higher level of sub-griding. Thus, the sub-grid size in chosen on a per-pixel basis until the function
        is evaluated at the specified fractional accuracy.

        Parameters
        ----------
        grid_lower_sub_2d : arrays.Array
            The results computed by a function using a lower sub-grid size
        grid_higher_sub_2d : grids.Array
            The results computed by a function using a higher sub-grid size.
        """

        fractional_mask = msk.Mask.unmasked(
            shape_2d=grid_lower_sub_2d.shape_2d, invert=True
        )

        fractional_mask = self.fractional_mask_jit_from_grid(
            fractional_accuracy_threshold=self.fractional_accuracy,
            fractional_mask=fractional_mask,
            grid_higher_sub_2d=grid_higher_sub_2d,
            grid_lower_sub_2d=grid_lower_sub_2d,
            grid_higher_mask=grid_higher_sub_2d.mask,
        )

        return msk.Mask(
            mask=fractional_mask,
            pixel_scales=grid_higher_sub_2d.pixel_scales,
            origin=grid_higher_sub_2d.origin,
        )

    @staticmethod
    @decorator_util.jit()
    def fractional_mask_jit_from_grid(
        fractional_accuracy_threshold,
        fractional_mask,
        grid_higher_sub_2d,
        grid_lower_sub_2d,
        grid_higher_mask,
    ):
        """Jitted function to determine the fractional mask, which is a mask where:

        - *True* entries signify the function has been evaluated in that pixel to desired fractional accuracy and
           therefore does not need to be iteratively computed at higher levels of sub-gridding.

        - *False* entries signify the function has not been evaluated in that pixel to desired fractional accuracy and
           therefore must be iterative computed at higher levels of sub-gridding to meet this accuracy."""

        for y in range(fractional_mask.shape[0]):
            for x in range(fractional_mask.shape[1]):
                if not grid_higher_mask[y, x]:

                    if abs(grid_higher_sub_2d[y, x, 0]) > 0:

                        fractional_accuracy_y = (
                            grid_lower_sub_2d[y, x, 0] / grid_higher_sub_2d[y, x, 0]
                        )

                    else:

                        fractional_accuracy_y = 1.0

                    if abs(grid_higher_sub_2d[y, x, 1]) > 0:

                        fractional_accuracy_x = (
                            grid_lower_sub_2d[y, x, 1] / grid_higher_sub_2d[y, x, 1]
                        )

                    else:
                        fractional_accuracy_x = 1.0

                    if fractional_accuracy_y > 1.0:
                        fractional_accuracy_y = 1.0 / fractional_accuracy_y

                    if fractional_accuracy_x > 1.0:
                        fractional_accuracy_x = 1.0 / fractional_accuracy_x

                    fractional_accuracy = min(
                        fractional_accuracy_y, fractional_accuracy_x
                    )

                    if fractional_accuracy < fractional_accuracy_threshold:
                        fractional_mask[y, x] = False

        return fractional_mask

    def iterated_grid_from_func(self, func, cls, grid_lower_sub_2d):
        """Iterate over a function that returns a grid of values until the it meets a specified fractional accuracy.
        The function returns a result on a pixel-grid where evaluating it on more points on a higher resolution
        sub-grid followed by binning lead to a more precise evaluation of the function. For the fractional accuracy of
        the grid to be met, both the y and x values must meet it.

        The function is first called for a sub-grid size of 1 and a higher resolution grid. The ratio of values give
        the fractional accuracy of each function evaluation. Pixels which do not meet the fractional accuracy are
        iteratively revaulated on higher resolution sub-grids. This is repeated until all pixels meet the fractional
        accuracy or the highest sub-size specified in the *sub_steps* attribute is computed.

        If the function return all zeros, the iteration is terminated early given that all levels of sub-gridding will
        return zeros. This occurs when a function is missing optional objects that contribute to the calculation.

        An example use case of this function is when a "deflections_from_grid" methods in **PyAutoLens**'s _MassProfile_
        module is computed, which by evaluating the function on a higher resolution sub-grid samples the analytic
        mass profile at more points and thus more precisely.

        Parameters
        ----------
        func : func
            The function which is iterated over to compute a more precise evaluation.
        cls : object
            The class the function belongs to.
        grid_lower_sub_2d : arrays.Array
            The results computed by the function using a lower sub-grid size
        """

        if not np.any(grid_lower_sub_2d):
            return grid_lower_sub_2d.in_1d

        iterated_grid = np.zeros(shape=(self.shape_2d[0], self.shape_2d[1], 2))

        fractional_mask_lower_sub = self.mask

        for sub_size in self.sub_steps[:-1]:

            grid_higher_sub = self.grid_at_sub_size_from_func_and_mask(
                func=func, cls=cls, mask=fractional_mask_lower_sub, sub_size=sub_size
            )

            fractional_mask_higher_sub = self.fractional_mask_from_grids(
                grid_lower_sub_2d=grid_lower_sub_2d, grid_higher_sub_2d=grid_higher_sub
            )

            iterated_grid = self.iterated_grid_jit_from(
                iterated_grid=iterated_grid,
                fractional_mask_higher_sub=fractional_mask_higher_sub,
                fractional_mask_lower_sub=fractional_mask_lower_sub,
                grid_higher_sub_2d=grid_higher_sub,
            )

            if fractional_mask_higher_sub.is_all_true:

                iterated_grid_1d = grid_util.sub_grid_1d_from(
                    mask=self.mask, sub_grid_2d=iterated_grid, sub_size=1
                )

                return grids.Grid(
                    grid=iterated_grid_1d, mask=self.mask.mask_sub_1, store_in_1d=True
                )

            grid_lower_sub_2d = grid_higher_sub
            fractional_mask_lower_sub = fractional_mask_higher_sub

        grid_higher_sub = self.grid_at_sub_size_from_func_and_mask(
            func=func,
            cls=cls,
            mask=fractional_mask_lower_sub,
            sub_size=self.sub_steps[-1],
        )

        iterated_grid_2d = iterated_grid + grid_higher_sub.in_2d_binned

        iterated_grid_1d = grid_util.sub_grid_1d_from(
            mask=self.mask, sub_grid_2d=iterated_grid_2d, sub_size=1
        )

        return grids.Grid(
            grid=iterated_grid_1d, mask=self.mask.mask_sub_1, store_in_1d=True
        )

    @staticmethod
    @decorator_util.jit()
    def iterated_grid_jit_from(
        iterated_grid,
        fractional_mask_higher_sub,
        fractional_mask_lower_sub,
        grid_higher_sub_2d,
    ):
        """Create the iterated grid from a result grid that is computed at a higher sub size level than the previous
        grid.

        The iterated grid is only updated for pixels where the fractional accuracy is met in both the (y,x) coodinates."""

        for y in range(iterated_grid.shape[0]):
            for x in range(iterated_grid.shape[1]):
                if (
                    fractional_mask_higher_sub[y, x]
                    and not fractional_mask_lower_sub[y, x]
                ):
                    iterated_grid[y, x, :] = grid_higher_sub_2d[y, x, :]

        return iterated_grid

    def iterated_result_from_func(self, func, cls):
        """Iterate over a function that returns an array or grid of values until the it meets a specified fractional
        accuracy. The function returns a result on a pixel-grid where evaluating it on more points on a higher
        resolution sub-grid followed by binning lead to a more precise evaluation of the function.

        A full description of the iteration method can be found in the functions *iterated_array_from_func* and
        *iterated_grid_from_func*. This function computes the result on a grid with a sub-size of 1, and uses its
        shape to call the correct function.

        Parameters
        ----------
        func : func
            The function which is iterated over to compute a more precise evaluation.
        cls : object
            The class the function belongs to.
            """
        result_sub_1_1d = func(cls, self.grid)
        result_sub_1_2d = self.grid.structure_from_result(
            result=result_sub_1_1d
        ).in_2d_binned

        if len(result_sub_1_2d.shape) == 2:
            return self.iterated_array_from_func(
                func=func, cls=cls, array_lower_sub_2d=result_sub_1_2d
            )
        elif len(result_sub_1_2d.shape) == 3:
            return self.iterated_grid_from_func(
                func=func, cls=cls, grid_lower_sub_2d=result_sub_1_2d
            )
