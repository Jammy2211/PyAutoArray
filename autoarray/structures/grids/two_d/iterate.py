import numpy as np

from autoarray import decorator_util
from autoarray.structures import arrays, grids
from autoarray.structures.grids import abstract_grid
from autoarray.mask import mask_2d as msk
from autoarray.structures.grids.two_d import grid_2d_util
from autoarray.geometry import geometry_util
from autoarray.structures.arrays.two_d import array_2d_util
from autoarray import exc


def sub_steps_from_none(sub_steps):

    if sub_steps is None:
        return [2, 4, 8, 16]
    return sub_steps


class Grid2DIterate(abstract_grid.AbstractGrid2D):
    def __new__(
        cls,
        grid,
        mask,
        fractional_accuracy=0.9999,
        sub_steps=None,
        store_slim=True,
        *args,
        **kwargs,
    ):
        """
        Represents a grid of coordinates as described for the `Grid2D` class, but using an iterative sub-grid that
        adapts its resolution when it is input into a function that performs an analytic calculation.

        A `Grid2D` represents (y,x) coordinates using a sub-grid, where functions are evaluated once at every coordinate
        on the sub-grid and averaged to give a more precise evaluation an analytic function. A `Grid2DIterate` does not
        have a specified sub-grid size, but instead iteratively recomputes the analytic function at increasing sub-grid
        sizes until an input fractional accuracy is reached.

        Iteration is performed on a per (y,x) coordinate basis, such that the sub-grid size will adopt low values
        wherever doing so can meet the fractional accuracy and high values only where it is required to meet the
        fractional accuracy. For functions where a wide range of sub-grid sizes allow fractional accuracy to be met
        this ensures the function can be evaluated accurate in a computaionally efficient manner.

        This overcomes a limitation of the `Grid2D` class whereby if a small subset of pixels require high levels of
        sub-gridding to be evaluated accuracy, the entire grid would require this sub-grid size thus leading to
        unecessary expensive function evaluations.

        Parameters
        ----------
        grid : np.ndarray
            The (y,x) coordinates of the grid.
        mask : msk.Mask2D
            The 2D mask associated with the grid, defining the pixels each grid coordinate is paired with and
            originates from.
        fractional_accuracy : float
            The fractional accuracy the function evaluated must meet to be accepted, where this accuracy is the ratio
            of the value at a higher sub_size to othe value computed using the previous sub_size.
        sub_steps : [int] or None
            The sub-size values used to iteratively evaluated the function at high levels of sub-gridding. If None,
            they are setup as the default values [2, 4, 8, 16].
        store_slim : bool
            If True, the grid is stored in 1D as an ndarray of shape [total_unmasked_pixels, 2]. If False, it is
            stored in 2D as an ndarray of shape [total_y_pixels, total_x_pixels, 2].
        """

        sub_steps = sub_steps_from_none(sub_steps=sub_steps)
        if store_slim and len(grid.shape) != 2:
            raise exc.GridException(
                "An grid input into the grids.Grid2D.__new__ method has store_slim = `True` but"
                "the input shape of the array is not 1."
            )

        obj = grid.view(cls)
        obj.mask = mask
        obj.store_slim = store_slim
        obj.grid = grids.Grid2D.manual_mask(
            grid=np.asarray(obj), mask=mask, store_slim=store_slim
        )
        obj.fractional_accuracy = fractional_accuracy
        obj.sub_steps = sub_steps
        return obj

    def __array_finalize__(self, obj):

        super(Grid2DIterate, self).__array_finalize__(obj)

        if hasattr(obj, "grid"):
            self.grid = obj.grid

        if hasattr(obj, "fractional_accuracy"):
            self.fractional_accuracy = obj.fractional_accuracy

        if hasattr(obj, "sub_steps"):
            self.sub_steps = obj.sub_steps

    def _new_structure(self, grid, mask, store_slim):
        """
        Conveninence method for creating a new instance of the Grid2DIterate class from this grid.

        This method is used in the 'slim', 'native', etc. convenience methods. By overwritin this method such that a
        Grid2DIterate is created the slim and native methods will return instances of the Grid2DIterate.

        Parameters
        ----------
        grid : np.ndarray or list
            The (y,x) coordinates of the grid input as an ndarray of shape [total_sub_coordinates, 2] or list of lists.
        mask : msk.Mask2D
            The 2D mask associated with the grid, defining the pixels each grid coordinate is paired with and
            originates from.
        store_slim : bool
            If True, the grid is stored in 1D as an ndarray of shape [total_unmasked_pixels, 2]. If False, it is
            stored in 2D as an ndarray of shape [total_y_pixels, total_x_pixels, 2].
        """
        return Grid2DIterate(
            grid=grid,
            mask=mask,
            fractional_accuracy=self.fractional_accuracy,
            sub_steps=self.sub_steps,
            store_slim=store_slim,
        )

    @classmethod
    def manual_slim(
        cls,
        grid,
        shape_native,
        pixel_scales,
        origin=(0.0, 0.0),
        fractional_accuracy=0.9999,
        sub_steps=None,
        store_slim=True,
    ):
        """
        Create a Grid2DIterate (see *Grid2DIterate.__new__*) by inputting the grid coordinates in 1D, for example:

        grid=np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])

        grid=[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]]

        From 1D input the method cannot determine the 2D shape of the grid and its mask, thus the shape_native must be
        input into this method. The mask is setup as a unmasked `Mask2D` of shape_native.

        Parameters
        ----------
        grid : np.ndarray or list
            The (y,x) coordinates of the grid input as an ndarray of shape [total_unmasked_pixells*(sub_size**2), 2]
            or a list of lists.
        shape_native : (float, float)
            The 2D shape of the mask the grid is paired with.
        pixel_scales: (float, float) or float
            The (y,x) scaled units to pixel units conversion factors of every pixel. If this is input as a ``float``,
            it is converted to a (float, float) structure.
        fractional_accuracy : float
            The fractional accuracy the function evaluated must meet to be accepted, where this accuracy is the ratio
            of the value at a higher sub_size to othe value computed using the previous sub_size.
        sub_steps : [int] or None
            The sub-size values used to iteratively evaluated the function at high levels of sub-gridding. If None,
            they are setup as the default values [2, 4, 8, 16].
        origin : (float, float)
            The origin of the grid's mask.
        store_slim : bool
            If True, the grid is stored in 1D as an ndarray of shape [total_unmasked_pixels, 2]. If False, it is
            stored in 2D as an ndarray of shape [total_y_pixels, total_x_pixels, 2].
        """
        grid = abstract_grid.convert_grid(grid=grid)
        pixel_scales = geometry_util.convert_pixel_scales_2d(pixel_scales=pixel_scales)

        mask = msk.Mask2D.unmasked(
            shape_native=shape_native,
            pixel_scales=pixel_scales,
            sub_size=1,
            origin=origin,
        )

        if store_slim:
            return Grid2DIterate(
                grid=grid,
                mask=mask,
                fractional_accuracy=fractional_accuracy,
                sub_steps=sub_steps,
                store_slim=store_slim,
            )

        grid_2d = grid_2d_util.grid_2d_native_from(
            grid_2d_slim=grid, mask_2d=mask, sub_size=1
        )

        return Grid2DIterate(
            grid=grid_2d,
            mask=mask,
            fractional_accuracy=fractional_accuracy,
            sub_steps=sub_steps,
            store_slim=store_slim,
        )

    @classmethod
    def uniform(
        cls,
        shape_native,
        pixel_scales,
        origin=(0.0, 0.0),
        fractional_accuracy=0.9999,
        sub_steps=None,
        store_slim=True,
    ):
        """
        Create a Grid2DIterate (see *Grid2DIterate.__new__*) as a uniform grid of (y,x) values given an input
        shape_native and pixel scale of the grid:

        Parameters
        ----------
        shape_native : (float, float)
            The 2D shape of the uniform grid and the mask that it is paired with.
        pixel_scales: (float, float) or float
            The (y,x) scaled units to pixel units conversion factors of every pixel. If this is input as a ``float``,
            it is converted to a (float, float) structure.
        fractional_accuracy : float
            The fractional accuracy the function evaluated must meet to be accepted, where this accuracy is the ratio
            of the value at a higher sub_size to othe value computed using the previous sub_size.
        sub_steps : [int] or None
            The sub-size values used to iteratively evaluated the function at high levels of sub-gridding. If None,
            they are setup as the default values [2, 4, 8, 16].
        origin : (float, float)
            The origin of the grid's mask.
        store_slim : bool
            If True, the grid is stored in 1D as an ndarray of shape [total_unmasked_pixels, 2]. If False, it is
            stored in 2D as an ndarray of shape [total_y_pixels, total_x_pixels, 2].
        """

        pixel_scales = geometry_util.convert_pixel_scales_2d(pixel_scales=pixel_scales)

        grid_slim = grid_2d_util.grid_2d_slim_via_shape_native_from(
            shape_native=shape_native,
            pixel_scales=pixel_scales,
            sub_size=1,
            origin=origin,
        )

        return Grid2DIterate.manual_slim(
            grid=grid_slim,
            shape_native=shape_native,
            pixel_scales=pixel_scales,
            fractional_accuracy=fractional_accuracy,
            sub_steps=sub_steps,
            origin=origin,
            store_slim=store_slim,
        )

    @classmethod
    def from_mask(
        cls, mask, fractional_accuracy=0.9999, sub_steps=None, store_slim=True
    ):
        """
        Create a Grid2DIterate (see *Grid2DIterate.__new__*) from a mask, where only unmasked pixels are included in
        the grid (if the grid is represented in 2D masked values are (0.0, 0.0)).

        The mask's pixel_scales and origin properties are used to compute the grid (y,x) coordinates.

        Parameters
        ----------
        mask : Mask2D
            The mask whose masked pixels are used to setup the sub-pixel grid.
        fractional_accuracy : float
            The fractional accuracy the function evaluated must meet to be accepted, where this accuracy is the ratio
            of the value at a higher sub_size to othe value computed using the previous sub_size.
        sub_steps : [int] or None
            The sub-size values used to iteratively evaluated the function at high levels of sub-gridding. If None,
            they are setup as the default values [2, 4, 8, 16].
        store_slim : bool
            If True, the grid is stored in 1D as an ndarray of shape [total_unmasked_pixels, 2]. If False, it is
            stored in 2D as an ndarray of shape [total_y_pixels, total_x_pixels, 2].
        """

        grid_slim = grid_2d_util.grid_2d_slim_via_mask_from(
            mask_2d=mask, pixel_scales=mask.pixel_scales, sub_size=1, origin=mask.origin
        )

        if store_slim:
            return grids.Grid2DIterate(
                grid=grid_slim,
                mask=mask.mask_sub_1,
                fractional_accuracy=fractional_accuracy,
                sub_steps=sub_steps,
                store_slim=store_slim,
            )

        grid_2d = grid_2d_util.grid_2d_native_from(
            grid_2d_slim=grid_slim, mask_2d=mask.mask_sub_1, sub_size=1
        )

        return grids.Grid2DIterate(
            grid=grid_2d,
            mask=mask.mask_sub_1,
            fractional_accuracy=fractional_accuracy,
            sub_steps=sub_steps,
            store_slim=store_slim,
        )

    @classmethod
    def blurring_grid_from_mask_and_kernel_shape(
        cls,
        mask,
        kernel_shape_native,
        fractional_accuracy=0.9999,
        sub_steps=None,
        store_slim=True,
    ):
        """
        Setup a blurring-grid from a mask, where a blurring grid consists of all pixels that are masked (and
        therefore have their values set to (0.0, 0.0)), but are close enough to the unmasked pixels that their values
        will be convolved into the unmasked those pixels. This occurs in *PyAutoGalaxy* when computing images from
        light profile objects.

        See *grids.Grid2D.blurring_grid_from_mask_and_kernel_shape* for a full description of a blurring grid. This
        method creates the blurring grid as a Grid2DIterate.

        Parameters
        ----------
        mask : Mask2D
            The mask whose masked pixels are used to setup the blurring grid.
        kernel_shape_native : (float, float)
            The 2D shape of the kernel which convolves signal from masked pixels to unmasked pixels.
        fractional_accuracy : float
            The fractional accuracy the function evaluated must meet to be accepted, where this accuracy is the ratio
            of the value at a higher sub_size to othe value computed using the previous sub_size.
        sub_steps : [int] or None
            The sub-size values used to iteratively evaluated the function at high levels of sub-gridding. If None,
            they are setup as the default values [2, 4, 8, 16].
        store_slim : bool
            If True, the grid is stored in 1D as an ndarray of shape [total_unmasked_pixels, 2]. If False, it is
            stored in 2D as an ndarray of shape [total_y_pixels, total_x_pixels, 2].
        """

        blurring_mask = mask.blurring_mask_from_kernel_shape(
            kernel_shape_native=kernel_shape_native
        )

        return cls.from_mask(
            mask=blurring_mask,
            fractional_accuracy=fractional_accuracy,
            sub_steps=sub_steps,
            store_slim=store_slim,
        )

    def grid_from_deflection_grid(self, deflection_grid):
        """
        Returns a new Grid2DIterate from this grid, where the (y,x) coordinates of this grid have a grid of (y,x) values,
        termed the deflection grid, subtracted from them to determine the new grid of (y,x) values.

        This is used by PyAutoLens to perform grid ray-tracing.

        Parameters
        ----------
        deflection_grid : np.ndarray
            The grid of (y,x) coordinates which is subtracted from this grid.
        """
        return Grid2DIterate(
            grid=self - deflection_grid,
            mask=self.mask,
            fractional_accuracy=self.fractional_accuracy,
            sub_steps=self.sub_steps,
            store_slim=self.store_slim,
        )

    def blurring_grid_from_kernel_shape(self, kernel_shape_native):
        """
        Returns the blurring grid from a grid and create it as a Grid2DIterate, via an input 2D kernel shape.

            For a full description of blurring grids, checkout *blurring_grid_from_mask_and_kernel_shape*.

            Parameters
            ----------
            kernel_shape_native : (float, float)
                The 2D shape of the kernel which convolves signal from masked pixels to unmasked pixels.
        """

        return Grid2DIterate.blurring_grid_from_mask_and_kernel_shape(
            mask=self.mask,
            kernel_shape_native=kernel_shape_native,
            fractional_accuracy=self.fractional_accuracy,
            sub_steps=self.sub_steps,
            store_slim=self.store_slim,
        )

    def padded_grid_from_kernel_shape(self, kernel_shape_native):
        """
        When the edge pixels of a mask are unmasked and a convolution is to occur, the signal of edge pixels will be
        'missing' if the grid is used to evaluate the signal via an analytic function.

        To ensure this signal is included the padded grid is used, which is 'buffed' such that it includes all pixels
        whose signal will be convolved into the unmasked pixels given the 2D kernel shape.

        Parameters
        ----------
        kernel_shape_native : (float, float)
            The 2D shape of the kernel which convolves signal from masked pixels to unmasked pixels.
        """
        shape = self.mask.shape

        padded_shape = (
            shape[0] + kernel_shape_native[0] - 1,
            shape[1] + kernel_shape_native[1] - 1,
        )

        padded_mask = msk.Mask2D.unmasked(
            shape_native=padded_shape,
            pixel_scales=self.mask.pixel_scales,
            sub_size=self.mask.sub_size,
        )

        return grids.Grid2DIterate.from_mask(
            mask=padded_mask,
            fractional_accuracy=self.fractional_accuracy,
            sub_steps=self.sub_steps,
        )

    @staticmethod
    def array_at_sub_size_from_func_and_mask(func, cls, mask, sub_size):

        mask_higher_sub = mask.mask_new_sub_size_from_mask(mask=mask, sub_size=sub_size)

        grid_compute = grids.Grid2D.from_mask(mask=mask_higher_sub)
        array_higher_sub = func(cls, grid_compute)
        return grid_compute.structure_from_result(result=array_higher_sub).native_binned

    @staticmethod
    def grid_at_sub_size_from_func_and_mask(func, cls, mask, sub_size):

        mask_higher_sub = mask.mask_new_sub_size_from_mask(mask=mask, sub_size=sub_size)

        grid_compute = grids.Grid2D.from_mask(mask=mask_higher_sub)
        grid_higher_sub = func(cls, grid_compute)
        return grid_compute.structure_from_result(result=grid_higher_sub).native_binned

    def fractional_mask_from_arrays(
        self, array_lower_sub_2d, array_higher_sub_2d
    ) -> msk.Mask2D:
        """
        Returns a fractional mask from a result array, where the fractional mask describes whether the evaluated
        value in the result array is within the `Grid2DIterate`'s specified fractional accuracy. The fractional mask thus
        determines whether a pixel on the grid needs to be reevaluated at a higher level of sub-gridding to meet the
        specified fractional accuracy. If it must be re-evaluated, the fractional masks's entry is `False`.

        The fractional mask is computed by comparing the results evaluated at one level of sub-gridding to another
        at a higher level of sub-griding. Thus, the sub-grid size in chosen on a per-pixel basis until the function
        is evaluated at the specified fractional accuracy.

        Parameters
        ----------
        array_lower_sub_2d : arrays.Array2D
            The results computed by a function using a lower sub-grid size
        array_higher_sub_2d : arrays.Array2D
            The results computed by a function using a higher sub-grid size.
        """

        fractional_mask = msk.Mask2D.unmasked(
            shape_native=array_lower_sub_2d.shape_native,
            pixel_scales=array_lower_sub_2d.pixel_scales,
            invert=True,
        )

        fractional_mask = self.fractional_mask_jit_from_array(
            fractional_accuracy_threshold=self.fractional_accuracy,
            fractional_mask=fractional_mask,
            array_higher_sub_2d=array_higher_sub_2d,
            array_lower_sub_2d=array_lower_sub_2d,
            array_higher_mask=array_higher_sub_2d.mask,
        )

        return msk.Mask2D(
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
        """
        Jitted functioon to determine the fractional mask, which is a mask where:

        - `True` entries signify the function has been evaluated in that pixel to desired fractional accuracy and
           therefore does not need to be iteratively computed at higher levels of sub-gridding.

        - `False` entries signify the function has not been evaluated in that pixel to desired fractional accuracy and
           therefore must be iterative computed at higher levels of sub-gridding to meet this accuracy.
        """

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

    def iterated_array_from_func(self, func, cls, array_lower_sub_2d):
        """
        Iterate over a function that returns an array of values until the it meets a specified fractional accuracy.
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
        `LightProfile` module is comomputed, which by evaluating the function on a higher resolution sub-grids sample
        the analytic light profile at more points and thus more precisely.

        Parameters
        ----------
        func : func
            The function which is iterated over to compute a more precise evaluation.
        cls : cls
            The class the function belongs to.
        grid_lower_sub_2d : arrays.Array2D
            The results computed by the function using a lower sub-grid size
        """

        if not np.any(array_lower_sub_2d):
            return array_lower_sub_2d.slim

        iterated_array = np.zeros(shape=self.shape_native)

        fractional_mask_lower_sub = self.mask

        for sub_size in self.sub_steps[:-1]:

            array_higher_sub = self.array_at_sub_size_from_func_and_mask(
                func=func, cls=cls, mask=fractional_mask_lower_sub, sub_size=sub_size
            )

            try:

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

            except ZeroDivisionError:

                return self.return_iterated_array_result(iterated_array=iterated_array)

            if fractional_mask_higher_sub.is_all_true:

                return self.return_iterated_array_result(iterated_array=iterated_array)

            array_lower_sub_2d = array_higher_sub
            fractional_mask_lower_sub = fractional_mask_higher_sub

        array_higher_sub = self.array_at_sub_size_from_func_and_mask(
            func=func,
            cls=cls,
            mask=fractional_mask_lower_sub,
            sub_size=self.sub_steps[-1],
        )

        iterated_array_2d = iterated_array + array_higher_sub.native_binned

        return self.return_iterated_array_result(iterated_array=iterated_array_2d)

    def return_iterated_array_result(
        self, iterated_array: np.ndarray
    ) -> arrays.Array2D:
        """
        Returns the resulting iterated array, by mapping it to 1D and then passing it back as an `Array2D` structure.

        Parameters
        ----------
        iterated_array : np.ndarray

        Returns
        -------
        iterated_array
            The resulting array computed via iteration.
        """

        iterated_array_1d = array_2d_util.array_2d_slim_from(
            mask_2d=self.mask, array_2d_native=iterated_array, sub_size=1
        )

        return arrays.Array2D(
            array=iterated_array_1d, mask=self.mask.mask_sub_1, store_slim=True
        )

    @staticmethod
    @decorator_util.jit()
    def iterated_array_jit_from(
        iterated_array,
        fractional_mask_higher_sub,
        fractional_mask_lower_sub,
        array_higher_sub_2d,
    ):
        """
        Create the iterated array from a result array that is computed at a higher sub size leel than the previous grid.

        The iterated array is only updated for pixels where the fractional accuracy is met.
        """

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
    ) -> msk.Mask2D:
        """
        Returns a fractional mask from a result array, where the fractional mask describes whether the evaluated
        value in the result array is within the `Grid2DIterate`'s specified fractional accuracy. The fractional mask thus
        determines whether a pixel on the grid needs to be reevaluated at a higher level of sub-gridding to meet the
        specified fractional accuracy. If it must be re-evaluated, the fractional masks's entry is `False`.

        The fractional mask is computed by comparing the results evaluated at one level of sub-gridding to another
        at a higher level of sub-griding. Thus, the sub-grid size in chosen on a per-pixel basis until the function
        is evaluated at the specified fractional accuracy.

        Parameters
        ----------
        grid_lower_sub_2d : arrays.Array2D
            The results computed by a function using a lower sub-grid size
        grid_higher_sub_2d : grids.Array2D
            The results computed by a function using a higher sub-grid size.
        """

        fractional_mask = msk.Mask2D.unmasked(
            shape_native=grid_lower_sub_2d.shape_native,
            pixel_scales=grid_lower_sub_2d.pixel_scales,
            invert=True,
        )

        fractional_mask = self.fractional_mask_jit_from_grid(
            fractional_accuracy_threshold=self.fractional_accuracy,
            fractional_mask=fractional_mask,
            grid_higher_sub_2d=grid_higher_sub_2d,
            grid_lower_sub_2d=grid_lower_sub_2d,
            grid_higher_mask=grid_higher_sub_2d.mask,
        )

        return msk.Mask2D(
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
        """
        Jitted function to determine the fractional mask, which is a mask where:

        - `True` entries signify the function has been evaluated in that pixel to desired fractional accuracy and
           therefore does not need to be iteratively computed at higher levels of sub-gridding.

        - `False` entries signify the function has not been evaluated in that pixel to desired fractional accuracy and
           therefore must be iterative computed at higher levels of sub-gridding to meet this accuracy.
        """

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
        """
        Iterate over a function that returns a grid of values until the it meets a specified fractional accuracy.
        The function returns a result on a pixel-grid where evaluating it on more points on a higher resolution
        sub-grid followed by binning lead to a more precise evaluation of the function. For the fractional accuracy of
        the grid to be met, both the y and x values must meet it.

        The function is first called for a sub-grid size of 1 and a higher resolution grid. The ratio of values give
        the fractional accuracy of each function evaluation. Pixels which do not meet the fractional accuracy are
        iteratively revaulated on higher resolution sub-grids. This is repeated until all pixels meet the fractional
        accuracy or the highest sub-size specified in the *sub_steps* attribute is computed.

        If the function return all zeros, the iteration is terminated early given that all levels of sub-gridding will
        return zeros. This occurs when a function is missing optional objects that contribute to the calculation.

        An example use case of this function is when a "deflections_from_grid" methods in **PyAutoLens**'s `MassProfile`
        module is computed, which by evaluating the function on a higher resolution sub-grid samples the analytic
        mass profile at more points and thus more precisely.

        Parameters
        ----------
        func : func
            The function which is iterated over to compute a more precise evaluation.
        cls : object
            The class the function belongs to.
        grid_lower_sub_2d : arrays.Array2D
            The results computed by the function using a lower sub-grid size
        """

        if not np.any(grid_lower_sub_2d):
            return grid_lower_sub_2d.slim

        iterated_grid = np.zeros(shape=(self.shape_native[0], self.shape_native[1], 2))

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

                iterated_grid_1d = grid_2d_util.grid_2d_slim_from(
                    mask=self.mask, grid_2d_native=iterated_grid, sub_size=1
                )

                return grids.Grid2D(
                    grid=iterated_grid_1d, mask=self.mask.mask_sub_1, store_slim=True
                )

            grid_lower_sub_2d = grid_higher_sub
            fractional_mask_lower_sub = fractional_mask_higher_sub

        grid_higher_sub = self.grid_at_sub_size_from_func_and_mask(
            func=func,
            cls=cls,
            mask=fractional_mask_lower_sub,
            sub_size=self.sub_steps[-1],
        )

        iterated_grid_2d = iterated_grid + grid_higher_sub.native_binned

        iterated_grid_1d = grid_2d_util.grid_2d_slim_from(
            mask=self.mask, grid_2d_native=iterated_grid_2d, sub_size=1
        )

        return grids.Grid2D(
            grid=iterated_grid_1d, mask=self.mask.mask_sub_1, store_slim=True
        )

    @staticmethod
    @decorator_util.jit()
    def iterated_grid_jit_from(
        iterated_grid,
        fractional_mask_higher_sub,
        fractional_mask_lower_sub,
        grid_higher_sub_2d,
    ):
        """
        Create the iterated grid from a result grid that is computed at a higher sub size level than the previous grid.

        The iterated grid is only updated for pixels where the fractional accuracy is met in both the (y,x) coodinates.
        """

        for y in range(iterated_grid.shape[0]):
            for x in range(iterated_grid.shape[1]):
                if (
                    fractional_mask_higher_sub[y, x]
                    and not fractional_mask_lower_sub[y, x]
                ):
                    iterated_grid[y, x, :] = grid_higher_sub_2d[y, x, :]

        return iterated_grid

    def iterated_result_from_func(self, func, cls):
        """
        Iterate over a function that returns an array or grid of values until the it meets a specified fractional
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
        ).native_binned

        if len(result_sub_1_2d.shape) == 2:
            return self.iterated_array_from_func(
                func=func, cls=cls, array_lower_sub_2d=result_sub_1_2d
            )
        elif len(result_sub_1_2d.shape) == 3:
            return self.iterated_grid_from_func(
                func=func, cls=cls, grid_lower_sub_2d=result_sub_1_2d
            )
