import numpy as np
from typing import Callable, Union, List, Tuple, Optional

from autoarray.structures.abstract_structure import Structure2D

from autoarray.mask.mask_2d import Mask2D
from autoarray.structures.two_d.grids.grid_2d import Grid2D

from autoarray.structures.two_d.array_2d import Array2D

from autoarray.structures.two_d import array_2d_util
from autoarray import numba_util
from autoarray.geometry import geometry_util
from autoarray.structures.two_d.grids import grid_2d_util

from autoarray import type as ty


def sub_steps_from(sub_steps):

    if sub_steps is None:
        return [2, 4, 8, 16]
    return sub_steps


class Grid2DIterate(Grid2D):
    def __new__(
        cls,
        grid: np.ndarray,
        mask: Mask2D,
        fractional_accuracy: float = 0.9999,
        relative_accuracy: Optional[float] = None,
        sub_steps: List[int] = None,
        *args,
        **kwargs
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
        grid
            The (y,x) coordinates of the grid.
        mask :Mask2D
            The 2D mask associated with the grid, defining the pixels each grid coordinate is paired with and
            originates from.
        fractional_accuracy
            The fractional accuracy the function evaluated must meet to be accepted, where this accuracy is the ratio
            of the value at a higher sub size to the value computed using the previous sub_size. The fractional
            accuracy does not depend on the units or magnitude of the function being evaluated.
        relative_accuracy
            The relative accuracy the function evaluted must meet to be accepted, where this value is the absolute
            difference of the values computed using the higher sub size and lower sub size grids. The relative
            accuracy depends on the units / magnitude of the function being evaluated.
        sub_steps : [int] or None
            The sub-size values used to iteratively evaluated the function at high levels of sub-gridding. If None,
            they are setup as the default values [2, 4, 8, 16].
        """

        sub_steps = sub_steps_from(sub_steps=sub_steps)

        obj = grid.view(cls)
        obj.mask = mask
        obj.grid = Grid2D.manual_mask(grid=np.asarray(obj), mask=mask)
        obj.fractional_accuracy = fractional_accuracy
        obj.relative_accuracy = relative_accuracy
        obj.sub_steps = sub_steps

        return obj

    def __array_finalize__(self, obj):

        super().__array_finalize__(obj)

        if hasattr(obj, "grid"):
            self.grid = obj.grid

        if hasattr(obj, "fractional_accuracy"):
            self.fractional_accuracy = obj.fractional_accuracy

        if hasattr(obj, "relative_accuracy"):
            self.relative_accuracy = obj.relative_accuracy

        if hasattr(obj, "sub_steps"):
            self.sub_steps = obj.sub_steps

    @classmethod
    def manual_slim(
        cls,
        grid: Union[np.ndarray, List],
        shape_native: Tuple[int, int],
        pixel_scales: ty.PixelScales,
        origin: Tuple[float, float] = (0.0, 0.0),
        fractional_accuracy: float = 0.9999,
        relative_accuracy: Optional[float] = None,
        sub_steps: Optional[List[int]] = None,
    ) -> "Grid2DIterate":
        """
        Create a Grid2DIterate (see *Grid2DIterate.__new__*) by inputting the grid coordinates in 1D, for example:

        grid=np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])

        grid=[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]]

        From 1D input the method cannot determine the 2D shape of the grid and its mask, thus the shape_native must be
        input into this method. The mask is setup as a unmasked `Mask2D` of shape_native.

        Parameters
        ----------
        grid or list
            The (y,x) coordinates of the grid input as an ndarray of shape [total_unmasked_pixells*(sub_size**2), 2]
            or a list of lists.
        shape_native
            The 2D shape of the mask the grid is paired with.
        pixel_scales
            The (y,x) scaled units to pixel units conversion factors of every pixel. If this is input as a ``float``,
            it is converted to a (float, float) structure.
        fractional_accuracy
            The fractional accuracy the function evaluated must meet to be accepted, where this accuracy is the ratio
            of the value at a higher sub size to the value computed using the previous sub_size. The fractional
            accuracy does not depend on the units or magnitude of the function being evaluated.
        relative_accuracy
            The relative accuracy the function evaluted must meet to be accepted, where this value is the absolute
            difference of the values computed using the higher sub size and lower sub size grids. The relative
            accuracy depends on the units / magnitude of the function being evaluated.
        sub_steps : [int] or None
            The sub-size values used to iteratively evaluated the function at high levels of sub-gridding. If None,
            they are setup as the default values [2, 4, 8, 16].
        origin
            The origin of the grid's mask.
        """
        grid = grid_2d_util.convert_grid(grid=grid)
        pixel_scales = geometry_util.convert_pixel_scales_2d(pixel_scales=pixel_scales)

        mask = Mask2D.unmasked(
            shape_native=shape_native,
            pixel_scales=pixel_scales,
            sub_size=1,
            origin=origin,
        )

        return Grid2DIterate(
            grid=grid,
            mask=mask,
            fractional_accuracy=fractional_accuracy,
            relative_accuracy=relative_accuracy,
            sub_steps=sub_steps,
        )

    @classmethod
    def uniform(
        cls,
        shape_native: Tuple[int, int],
        pixel_scales: ty.PixelScales,
        origin: Tuple[float, float] = (0.0, 0.0),
        fractional_accuracy: float = 0.9999,
        relative_accuracy: Optional[float] = None,
        sub_steps: Optional[List[int]] = None,
    ) -> "Grid2DIterate":
        """
        Create a Grid2DIterate (see *Grid2DIterate.__new__*) as a uniform grid of (y,x) values given an input
        shape_native and pixel scale of the grid:

        Parameters
        ----------
        shape_native
            The 2D shape of the uniform grid and the mask that it is paired with.
        pixel_scales
            The (y,x) scaled units to pixel units conversion factors of every pixel. If this is input as a ``float``,
            it is converted to a (float, float) structure.
        fractional_accuracy
            The fractional accuracy the function evaluated must meet to be accepted, where this accuracy is the ratio
            of the value at a higher sub size to the value computed using the previous sub_size. The fractional
            accuracy does not depend on the units or magnitude of the function being evaluated.
        relative_accuracy
            The relative accuracy the function evaluted must meet to be accepted, where this value is the absolute
            difference of the values computed using the higher sub size and lower sub size grids. The relative
            accuracy depends on the units / magnitude of the function being evaluated.
        sub_steps : [int] or None
            The sub-size values used to iteratively evaluated the function at high levels of sub-gridding. If None,
            they are setup as the default values [2, 4, 8, 16].
        origin
            The origin of the grid's mask.
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
            relative_accuracy=relative_accuracy,
            sub_steps=sub_steps,
            origin=origin,
        )

    @classmethod
    def from_mask(
        cls,
        mask: Mask2D,
        fractional_accuracy: float = 0.9999,
        relative_accuracy: Optional[float] = None,
        sub_steps: Optional[List[int]] = None,
    ) -> "Grid2DIterate":
        """
        Create a Grid2DIterate (see *Grid2DIterate.__new__*) from a mask, where only unmasked pixels are included in
        the grid (if the grid is represented in 2D masked values are (0.0, 0.0)).

        The mask's pixel_scales and origin properties are used to compute the grid (y,x) coordinates.

        Parameters
        ----------
        mask : Mask2D
            The mask whose masked pixels are used to setup the sub-pixel grid.
        fractional_accuracy
            The fractional accuracy the function evaluated must meet to be accepted, where this accuracy is the ratio
            of the value at a higher sub size to the value computed using the previous sub_size. The fractional
            accuracy does not depend on the units or magnitude of the function being evaluated.
        relative_accuracy
            The relative accuracy the function evaluted must meet to be accepted, where this value is the absolute
            difference of the values computed using the higher sub size and lower sub size grids. The relative
            accuracy depends on the units / magnitude of the function being evaluated.
        sub_steps : [int] or None
            The sub-size values used to iteratively evaluated the function at high levels of sub-gridding. If None,
            they are setup as the default values [2, 4, 8, 16].
        """

        grid_slim = grid_2d_util.grid_2d_slim_via_mask_from(
            mask_2d=mask, pixel_scales=mask.pixel_scales, sub_size=1, origin=mask.origin
        )

        return Grid2DIterate(
            grid=grid_slim,
            mask=mask.mask_sub_1,
            fractional_accuracy=fractional_accuracy,
            relative_accuracy=relative_accuracy,
            sub_steps=sub_steps,
        )

    @classmethod
    def blurring_grid_from(
        cls,
        mask: Mask2D,
        kernel_shape_native: Tuple[int, int],
        fractional_accuracy: float = 0.9999,
        relative_accuracy: Optional[float] = None,
        sub_steps: Optional[List[int]] = None,
    ) -> "Grid2DIterate":
        """
        Setup a blurring-grid from a mask, where a blurring grid consists of all pixels that are masked (and
        therefore have their values set to (0.0, 0.0)), but are close enough to the unmasked pixels that their values
        will be convolved into the unmasked those pixels. This when computing images from
        light profile objects.

        See *Grid2D.blurring_grid_from* for a full description of a blurring grid. This
        method creates the blurring grid as a Grid2DIterate.

        Parameters
        ----------
        mask : Mask2D
            The mask whose masked pixels are used to setup the blurring grid.
        kernel_shape_native
            The 2D shape of the kernel which convolves signal from masked pixels to unmasked pixels.
        fractional_accuracy
            The fractional accuracy the function evaluated must meet to be accepted, where this accuracy is the ratio
            of the value at a higher sub size to the value computed using the previous sub_size. The fractional
            accuracy does not depend on the units or magnitude of the function being evaluated.
        relative_accuracy
            The relative accuracy the function evaluted must meet to be accepted, where this value is the absolute
            difference of the values computed using the higher sub size and lower sub size grids. The relative
            accuracy depends on the units / magnitude of the function being evaluated.
        sub_steps : [int] or None
            The sub-size values used to iteratively evaluated the function at high levels of sub-gridding. If None,
            they are setup as the default values [2, 4, 8, 16].
        """

        blurring_mask = mask.blurring_mask_from(kernel_shape_native=kernel_shape_native)

        return cls.from_mask(
            mask=blurring_mask,
            fractional_accuracy=fractional_accuracy,
            relative_accuracy=relative_accuracy,
            sub_steps=sub_steps,
        )

    @property
    def slim(self) -> "Grid2DIterate":
        """
        Return a `Grid2D` where the data is stored its `slim` representation, which is an ndarray of shape
        [total_unmasked_pixels * sub_size**2, 2].

        If it is already stored in its `slim` representation  it is returned as it is. If not, it is  mapped from
        `native` to `slim` and returned as a new `Grid2D`.
        """
        return Grid2DIterate(
            grid=super().slim,
            mask=self.mask,
            fractional_accuracy=self.fractional_accuracy,
            sub_steps=self.sub_steps,
        )

    @property
    def native(self) -> "Grid2DIterate":
        """
        Return a `Grid2D` where the data is stored in its `native` representation, which has shape
        [sub_size*total_y_pixels, sub_size*total_x_pixels, 2].

        If it is already stored in its `native` representation it is return as it is. If not, it is mapped from
        `slim` to `native` and returned as a new `Grid2D`.

        This method is used in the child `Grid2D` classes to create their `native` properties.
        """
        return Grid2DIterate(
            grid=super().native,
            mask=self.mask,
            fractional_accuracy=self.fractional_accuracy,
            sub_steps=self.sub_steps,
        )

    @property
    def binned(self) -> "Grid2DIterate":
        """
        Return a `Grid2D` of the binned-up grid in its 1D representation, which is stored with
        shape [total_unmasked_pixels, 2].

        The binning up process converts a grid from (y,x) values where each value is a coordinate on the sub-grid to
        (y,x) values where each coordinate is at the centre of its mask (e.g. a grid with a sub_size of 1). This is
        performed by taking the mean of all (y,x) values in each sub pixel.

        If the grid is stored in 1D it is return as is. If it is stored in 2D, it must first be mapped from 2D to 1D.
        """
        return Grid2DIterate(
            grid=super().binned,
            mask=self.mask.mask_sub_1,
            fractional_accuracy=self.fractional_accuracy,
            sub_steps=self.sub_steps,
        )

    def grid_via_deflection_grid_from(
        self, deflection_grid: np.ndarray
    ) -> "Grid2DIterate":
        """
        Returns a new Grid2DIterate from this grid, where the (y,x) coordinates of this grid have a grid of (y,x) values,
        termed the deflection grid, subtracted from them to determine the new grid of (y,x) values.

        This is used by PyAutoLens to perform grid ray-tracing.

        Parameters
        ----------
        deflection_grid
            The grid of (y,x) coordinates which is subtracted from this grid.
        """
        return Grid2DIterate(
            grid=self - deflection_grid,
            mask=self.mask,
            fractional_accuracy=self.fractional_accuracy,
            sub_steps=self.sub_steps,
        )

    def blurring_grid_via_kernel_shape_from(
        self, kernel_shape_native: Tuple[int, int]
    ) -> "Grid2DIterate":
        """
        Returns the blurring grid from a grid and create it as a Grid2DIterate, via an input 2D kernel shape.

        For a full description of blurring grids, checkout *blurring_grid_from*.

        Parameters
        ----------
        kernel_shape_native
            The 2D shape of the kernel which convolves signal from masked pixels to unmasked pixels.
        """

        return Grid2DIterate.blurring_grid_from(
            mask=self.mask,
            kernel_shape_native=kernel_shape_native,
            fractional_accuracy=self.fractional_accuracy,
            sub_steps=self.sub_steps,
        )

    def padded_grid_from(self, kernel_shape_native: Tuple[int, int]) -> "Grid2DIterate":
        """
        When the edge pixels of a mask are unmasked and a convolution is to occur, the signal of edge pixels will be
        'missing' if the grid is used to evaluate the signal via an analytic function.

        To ensure this signal is included the padded grid is used, which is 'buffed' such that it includes all pixels
        whose signal will be convolved into the unmasked pixels given the 2D kernel shape.

        Parameters
        ----------
        kernel_shape_native
            The 2D shape of the kernel which convolves signal from masked pixels to unmasked pixels.
        """
        shape = self.mask.shape

        padded_shape = (
            shape[0] + kernel_shape_native[0] - 1,
            shape[1] + kernel_shape_native[1] - 1,
        )

        padded_mask = Mask2D.unmasked(
            shape_native=padded_shape,
            pixel_scales=self.mask.pixel_scales,
            sub_size=self.mask.sub_size,
        )

        return Grid2DIterate.from_mask(
            mask=padded_mask,
            fractional_accuracy=self.fractional_accuracy,
            sub_steps=self.sub_steps,
        )

    @staticmethod
    def array_at_sub_size_from(func: Callable, cls, mask: Mask2D, sub_size) -> Array2D:

        mask_higher_sub = mask.mask_new_sub_size_from(mask=mask, sub_size=sub_size)

        grid_compute = Grid2D.from_mask(mask=mask_higher_sub)
        array_higher_sub = func(cls, grid_compute)
        return grid_compute.structure_2d_from(result=array_higher_sub).binned.native

    @staticmethod
    def grid_at_sub_size_from(func: Callable, cls, mask: Mask2D, sub_size) -> Grid2D:

        mask_higher_sub = mask.mask_new_sub_size_from(mask=mask, sub_size=sub_size)

        grid_compute = Grid2D.from_mask(mask=mask_higher_sub)
        grid_higher_sub = func(cls, grid_compute)
        return grid_compute.structure_2d_from(result=grid_higher_sub).binned.native

    def threshold_mask_via_arrays_from(
        self, array_lower_sub_2d: Array2D, array_higher_sub_2d: Array2D
    ) -> Mask2D:
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
        array_lower_sub_2d : Array2D
            The results computed by a function using a lower sub-grid size
        array_higher_sub_2d : Array2D
            The results computed by a function using a higher sub-grid size.
        """

        threshold_mask = Mask2D.unmasked(
            shape_native=array_lower_sub_2d.shape_native,
            pixel_scales=array_lower_sub_2d.pixel_scales,
            invert=True,
        )

        threshold_mask = self.threshold_mask_via_arrays_jit_from(
            fractional_accuracy_threshold=self.fractional_accuracy,
            relative_accuracy_threshold=self.relative_accuracy,
            threshold_mask=threshold_mask,
            array_higher_sub_2d=array_higher_sub_2d,
            array_lower_sub_2d=array_lower_sub_2d,
            array_higher_mask=array_higher_sub_2d.mask,
        )

        return Mask2D(
            mask=threshold_mask,
            pixel_scales=array_higher_sub_2d.pixel_scales,
            origin=array_higher_sub_2d.origin,
        )

    @staticmethod
    @numba_util.jit()
    def threshold_mask_via_arrays_jit_from(
        fractional_accuracy_threshold: float,
        relative_accuracy_threshold: Optional[float],
        threshold_mask: Mask2D,
        array_higher_sub_2d: Array2D,
        array_lower_sub_2d: Array2D,
        array_higher_mask: Mask2D,
    ) -> np.ndarray:
        """
        Jitted functioon to determine the fractional mask, which is a mask where:

        - `True` entries signify the function has been evaluated in that pixel to desired accuracy and
           therefore does not need to be iteratively computed at higher levels of sub-gridding.

        - `False` entries signify the function has not been evaluated in that pixel to desired fractional accuracy and
           therefore must be iterative computed at higher levels of sub-gridding to meet this accuracy.
        """

        if fractional_accuracy_threshold is not None:

            for y in range(threshold_mask.shape[0]):
                for x in range(threshold_mask.shape[1]):
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
                            threshold_mask[y, x] = False

        if relative_accuracy_threshold is not None:

            for y in range(threshold_mask.shape[0]):
                for x in range(threshold_mask.shape[1]):
                    if not array_higher_mask[y, x]:

                        if (
                            abs(array_lower_sub_2d[y, x] - array_higher_sub_2d[y, x])
                            > relative_accuracy_threshold
                        ):
                            threshold_mask[y, x] = False

        return threshold_mask

    def iterated_array_from(
        self, func: Callable, cls: object, array_lower_sub_2d: Array2D
    ) -> Array2D:
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

        An example use case of this function is when a "image_2d_from" methods in **PyAutoGalaxy**'s
        `LightProfile` module is comomputed, which by evaluating the function on a higher resolution sub-grids sample
        the analytic light profile at more points and thus more precisely.

        Parameters
        ----------
        func : func
            The function which is iterated over to compute a more precise evaluation.
        cls : cls
            The class the function belongs to.
        grid_lower_sub_2d : Array2D
            The results computed by the function using a lower sub-grid size
        """

        if not np.any(array_lower_sub_2d):
            return array_lower_sub_2d.slim

        iterated_array = np.zeros(shape=self.shape_native)

        threshold_mask_lower_sub = self.mask

        for sub_size in self.sub_steps[:-1]:

            array_higher_sub = self.array_at_sub_size_from(
                func=func, cls=cls, mask=threshold_mask_lower_sub, sub_size=sub_size
            )

            try:

                threshold_mask_higher_sub = self.threshold_mask_via_arrays_from(
                    array_lower_sub_2d=array_lower_sub_2d,
                    array_higher_sub_2d=array_higher_sub,
                )

                iterated_array = self.iterated_array_jit_from(
                    iterated_array=iterated_array,
                    threshold_mask_higher_sub=threshold_mask_higher_sub,
                    threshold_mask_lower_sub=threshold_mask_lower_sub,
                    array_higher_sub_2d=array_higher_sub,
                )

            except ZeroDivisionError:

                return self.return_iterated_array_result(iterated_array=iterated_array)

            if threshold_mask_higher_sub.is_all_true:

                return self.return_iterated_array_result(iterated_array=iterated_array)

            array_lower_sub_2d = array_higher_sub
            threshold_mask_lower_sub = threshold_mask_higher_sub

        array_higher_sub = self.array_at_sub_size_from(
            func=func,
            cls=cls,
            mask=threshold_mask_lower_sub,
            sub_size=self.sub_steps[-1],
        )

        iterated_array_2d = iterated_array + array_higher_sub.binned.native

        return self.return_iterated_array_result(iterated_array=iterated_array_2d)

    def return_iterated_array_result(self, iterated_array: Array2D) -> Array2D:
        """
        Returns the resulting iterated array, by mapping it to 1D and then passing it back as an `Array2D` structure.

        Parameters
        ----------
        iterated_array

        Returns
        -------
        iterated_array
            The resulting array computed via iteration.
        """

        iterated_array_1d = array_2d_util.array_2d_slim_from(
            mask_2d=self.mask, array_2d_native=iterated_array, sub_size=1
        )

        return Array2D(array=iterated_array_1d, mask=self.mask.mask_sub_1)

    @staticmethod
    @numba_util.jit()
    def iterated_array_jit_from(
        iterated_array: Array2D,
        threshold_mask_higher_sub: Mask2D,
        threshold_mask_lower_sub: Mask2D,
        array_higher_sub_2d: Array2D,
    ) -> Array2D:
        """
        Create the iterated array from a result array that is computed at a higher sub size leel than the previous grid.

        The iterated array is only updated for pixels where the fractional accuracy is met.
        """

        for y in range(iterated_array.shape[0]):
            for x in range(iterated_array.shape[1]):
                if (
                    threshold_mask_higher_sub[y, x]
                    and not threshold_mask_lower_sub[y, x]
                ):
                    iterated_array[y, x] = array_higher_sub_2d[y, x]

        return iterated_array

    def threshold_mask_via_grids_from(
        self, grid_lower_sub_2d: Grid2D, grid_higher_sub_2d: Grid2D
    ) -> Mask2D:
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
        grid_lower_sub_2d : Array2D
            The results computed by a function using a lower sub-grid size
        grid_higher_sub_2d : grids.Array2D
            The results computed by a function using a higher sub-grid size.
        """

        threshold_mask = Mask2D.unmasked(
            shape_native=grid_lower_sub_2d.shape_native,
            pixel_scales=grid_lower_sub_2d.pixel_scales,
            invert=True,
        )

        threshold_mask = self.threshold_mask_via_grids_jit_from(
            fractional_accuracy_threshold=self.fractional_accuracy,
            relative_accuracy_threshold=self.relative_accuracy,
            threshold_mask=threshold_mask,
            grid_higher_sub_2d=grid_higher_sub_2d,
            grid_lower_sub_2d=grid_lower_sub_2d,
            grid_higher_mask=grid_higher_sub_2d.mask,
        )

        return Mask2D(
            mask=threshold_mask,
            pixel_scales=grid_higher_sub_2d.pixel_scales,
            origin=grid_higher_sub_2d.origin,
        )

    @staticmethod
    @numba_util.jit()
    def threshold_mask_via_grids_jit_from(
        fractional_accuracy_threshold: float,
        relative_accuracy_threshold: float,
        threshold_mask: Mask2D,
        grid_higher_sub_2d: Grid2D,
        grid_lower_sub_2d: Grid2D,
        grid_higher_mask: Mask2D,
    ) -> Mask2D:
        """
        Jitted function to determine the fractional mask, which is a mask where:

        - `True` entries signify the function has been evaluated in that pixel to desired fractional accuracy and
           therefore does not need to be iteratively computed at higher levels of sub-gridding.

        - `False` entries signify the function has not been evaluated in that pixel to desired fractional accuracy and
           therefore must be iterative computed at higher levels of sub-gridding to meet this accuracy.
        """

        for y in range(threshold_mask.shape[0]):
            for x in range(threshold_mask.shape[1]):
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
                        threshold_mask[y, x] = False

        if relative_accuracy_threshold is not None:

            for y in range(threshold_mask.shape[0]):
                for x in range(threshold_mask.shape[1]):
                    if not grid_higher_mask[y, x]:

                        relative_accuracy_y = abs(
                            grid_lower_sub_2d[y, x, 0] - grid_higher_sub_2d[y, x, 0]
                        )
                        relative_accuracy_x = abs(
                            grid_lower_sub_2d[y, x, 1] - grid_higher_sub_2d[y, x, 1]
                        )

                        relative_accuracy = max(
                            relative_accuracy_y, relative_accuracy_x
                        )

                        if relative_accuracy > relative_accuracy_threshold:
                            threshold_mask[y, x] = False

        return threshold_mask

    def iterated_grid_from(
        self, func: Callable, cls: object, grid_lower_sub_2d: Grid2D
    ) -> Grid2D:
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

        An example use case of this function is when a "deflections_yx_2d_from" methods in **PyAutoLens**'s `MassProfile`
        module is computed, which by evaluating the function on a higher resolution sub-grid samples the analytic
        mass profile at more points and thus more precisely.

        Parameters
        ----------
        func
            The function which is iterated over to compute a more precise evaluation.
        cls
            The class the function belongs to.
        grid_lower_sub_2d
            The results computed by the function using a lower sub-grid size
        """

        if not np.any(grid_lower_sub_2d):
            return grid_lower_sub_2d.slim

        iterated_grid = np.zeros(shape=(self.shape_native[0], self.shape_native[1], 2))

        threshold_mask_lower_sub = self.mask

        for sub_size in self.sub_steps[:-1]:

            grid_higher_sub = self.grid_at_sub_size_from(
                func=func, cls=cls, mask=threshold_mask_lower_sub, sub_size=sub_size
            )

            threshold_mask_higher_sub = self.threshold_mask_via_grids_from(
                grid_lower_sub_2d=grid_lower_sub_2d, grid_higher_sub_2d=grid_higher_sub
            )

            iterated_grid = self.iterated_grid_jit_from(
                iterated_grid=iterated_grid,
                threshold_mask_higher_sub=threshold_mask_higher_sub,
                threshold_mask_lower_sub=threshold_mask_lower_sub,
                grid_higher_sub_2d=grid_higher_sub,
            )

            if threshold_mask_higher_sub.is_all_true:

                iterated_grid_1d = grid_2d_util.grid_2d_slim_from(
                    mask=self.mask, grid_2d_native=iterated_grid, sub_size=1
                )

                return Grid2D(grid=iterated_grid_1d, mask=self.mask.mask_sub_1)

            grid_lower_sub_2d = grid_higher_sub
            threshold_mask_lower_sub = threshold_mask_higher_sub

        grid_higher_sub = self.grid_at_sub_size_from(
            func=func,
            cls=cls,
            mask=threshold_mask_lower_sub,
            sub_size=self.sub_steps[-1],
        )

        iterated_grid_2d = iterated_grid + grid_higher_sub.binned.native

        iterated_grid_1d = grid_2d_util.grid_2d_slim_from(
            mask=self.mask, grid_2d_native=iterated_grid_2d, sub_size=1
        )

        return Grid2D(grid=iterated_grid_1d, mask=self.mask.mask_sub_1)

    @staticmethod
    @numba_util.jit()
    def iterated_grid_jit_from(
        iterated_grid: Grid2D,
        threshold_mask_higher_sub: Mask2D,
        threshold_mask_lower_sub: Mask2D,
        grid_higher_sub_2d: Grid2D,
    ) -> Grid2D:
        """
        Create the iterated grid from a result grid that is computed at a higher sub size level than the previous grid.

        The iterated grid is only updated for pixels where the fractional accuracy is met in both the (y,x) coodinates.
        """

        for y in range(iterated_grid.shape[0]):
            for x in range(iterated_grid.shape[1]):
                if (
                    threshold_mask_higher_sub[y, x]
                    and not threshold_mask_lower_sub[y, x]
                ):
                    iterated_grid[y, x, :] = grid_higher_sub_2d[y, x, :]

        return iterated_grid

    def iterated_result_from(
        self, func: Callable, cls: object
    ) -> Union[Array2D, Grid2D]:
        """
        Iterate over a function that returns an array or grid of values until the it meets a specified fractional
        accuracy. The function returns a result on a pixel-grid where evaluating it on more points on a higher
        resolution sub-grid followed by binning lead to a more precise evaluation of the function.

        A full description of the iteration method can be found in the functions *iterated_array_from* and
        *iterated_grid_from*. This function computes the result on a grid with a sub-size of 1, and uses its
        shape to call the correct function.

        Parameters
        ----------
        func : func
            The function which is iterated over to compute a more precise evaluation.
        cls : object
            The class the function belongs to.
        """
        result_sub_1_1d = func(cls, self.grid)
        result_sub_1_2d = self.grid.structure_2d_from(
            result=result_sub_1_1d
        ).binned.native

        if len(result_sub_1_2d.shape) == 2:
            return self.iterated_array_from(
                func=func, cls=cls, array_lower_sub_2d=result_sub_1_2d
            )
        elif len(result_sub_1_2d.shape) == 3:
            return self.iterated_grid_from(
                func=func, cls=cls, grid_lower_sub_2d=result_sub_1_2d
            )
