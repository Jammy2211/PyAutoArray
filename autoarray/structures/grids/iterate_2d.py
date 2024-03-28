import numpy as np
from typing import Callable, Union, List, Optional

from autoarray import numba_util
from autoarray.mask.mask_2d import Mask2D
from autoarray.structures.arrays import array_2d_util
from autoarray.structures.arrays.uniform_2d import Array2D
from autoarray.structures.grids import grid_2d_util
from autoarray.structures.grids.uniform_2d import Grid2D


def sub_steps_from(sub_steps):
    if sub_steps is None:
        return [2, 4, 8, 16]
    return sub_steps


@numba_util.jit()
def threshold_mask_via_arrays_jit_from(
    fractional_accuracy_threshold: float,
    relative_accuracy_threshold: Optional[float],
    threshold_mask: np.ndarray,
    array_higher_sub_2d: np.ndarray,
    array_lower_sub_2d: np.ndarray,
    array_higher_mask: np.ndarray,
) -> np.ndarray:
    """
    Jitted function to determine the fractional mask, which is a mask where:

    - ``True`` entries signify the function has been evaluated in that pixel to desired accuracy and
      therefore does not need to be iteratively computed at higher levels of sub-gridding.

    - ``False`` entries signify the function has not been evaluated in that pixel to desired fractional accuracy and
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


@numba_util.jit()
def iterated_array_jit_from(
    iterated_array: np.ndarray,
    threshold_mask_higher_sub: np.ndarray,
    threshold_mask_lower_sub: np.ndarray,
    array_higher_sub_2d: np.ndarray,
) -> np.ndarray:
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

@numba_util.jit()
def threshold_mask_via_grids_jit_from(
    fractional_accuracy_threshold: float,
    relative_accuracy_threshold: float,
    threshold_mask: np.ndarray,
    grid_higher_sub_2d: np.ndarray,
    grid_lower_sub_2d: np.ndarray,
    grid_higher_mask: np.ndarray,
) -> np.ndarray:
    """
    Jitted function to determine the fractional mask, which is a mask where:

    - ``True`` entries signify the function has been evaluated in that pixel to desired fractional accuracy and
      therefore does not need to be iteratively computed at higher levels of sub-gridding.

    - ``False`` entries signify the function has not been evaluated in that pixel to desired fractional accuracy and
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


@numba_util.jit()
def iterated_grid_jit_from(
    iterated_grid: Grid2D,
    threshold_mask_higher_sub: np.ndarray,
    threshold_mask_lower_sub: np.ndarray,
    grid_higher_sub_2d: np.ndarray,
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


class Iterator:
    def __init__(
        self,
        fractional_accuracy: float = 0.9999,
        relative_accuracy: Optional[float] = None,
        sub_steps: List[int] = None,
    ):
        """
        Represents a grid of coordinates as described for the ``Grid2D`` class, but using an iterative sub-grid that
        adapts its resolution when it is input into a function that performs an analytic calculation.

        A ``Grid2D`` represents (y,x) coordinates using a sub-grid, where functions are evaluated once at every coordinate
        on the sub-grid and averaged to give a more precise evaluation an analytic function. A ``Iterator`` does not
        have a specified sub-grid size, but instead iteratively recomputes the analytic function at increasing sub-grid
        sizes until an input fractional accuracy is reached.

        Iteration is performed on a per (y,x) coordinate basis, such that the sub-grid size will adopt low values
        wherever doing so can meet the fractional accuracy and high values only where it is required to meet the
        fractional accuracy. For functions where a wide range of sub-grid sizes allow fractional accuracy to be met
        this ensures the function can be evaluated accurate in a computaionally efficient manner.

        This overcomes a limitation of the ``Grid2D`` class whereby if a small subset of pixels require high levels of
        sub-gridding to be evaluated accuracy, the entire grid would require this sub-grid size thus leading to
        unecessary expensive function evaluations.

        Parameters
        ----------
        values
            The (y,x) coordinates of the grid.
        mask
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
        sub_steps
            The sub-size values used to iteratively evaluated the function at high levels of sub-gridding. If None,
            they are setup as the default values [2, 4, 8, 16].
        store_native
            If True, the ndarray is stored in its native format [total_y_pixels, total_x_pixels, 2]. This avoids
            mapping large data arrays to and from the slim / native formats, which can be a computational bottleneck.
        """

        sub_steps = sub_steps_from(sub_steps=sub_steps)

        self.fractional_accuracy = fractional_accuracy
        self.relative_accuracy = relative_accuracy
        self.sub_steps = sub_steps

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
        value in the result array is within the ``Iterator``'s specified fractional accuracy. The fractional mask thus
        determines whether a pixel on the grid needs to be reevaluated at a higher level of sub-gridding to meet the
        specified fractional accuracy. If it must be re-evaluated, the fractional masks's entry is ``False``.

        The fractional mask is computed by comparing the results evaluated at one level of sub-gridding to another
        at a higher level of sub-griding. Thus, the sub-grid size in chosen on a per-pixel basis until the function
        is evaluated at the specified fractional accuracy.

        Parameters
        ----------
        array_lower_sub_2d
            The results computed by a function using a lower sub-grid size
        array_higher_sub_2d
            The results computed by a function using a higher sub-grid size.
        """

        threshold_mask = Mask2D.all_false(
            shape_native=array_lower_sub_2d.shape_native,
            pixel_scales=array_lower_sub_2d.pixel_scales,
            invert=True,
        )

        threshold_mask = threshold_mask_via_arrays_jit_from(
            fractional_accuracy_threshold=self.fractional_accuracy,
            relative_accuracy_threshold=self.relative_accuracy,
            threshold_mask=np.array(threshold_mask),
            array_higher_sub_2d=np.array(array_higher_sub_2d),
            array_lower_sub_2d=np.array(array_lower_sub_2d),
            array_higher_mask=np.array(array_higher_sub_2d.mask),
        )

        return Mask2D(
            mask=threshold_mask,
            pixel_scales=array_higher_sub_2d.pixel_scales,
            origin=array_higher_sub_2d.origin,
        )

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
        ``LightProfile`` module is comomputed, which by evaluating the function on a higher resolution sub-grids sample
        the analytic light profile at more points and thus more precisely.

        Parameters
        ----------
        func : func
            The function which is iterated over to compute a more precise evaluation.
        cls : cls
            The class the function belongs to.
        grid_lower_sub_2d
            The results computed by the function using a lower sub-grid size
        """

        if not np.any(array_lower_sub_2d):
            return array_lower_sub_2d.slim

        shape_native = array_lower_sub_2d.shape_native
        mask = array_lower_sub_2d.mask

        iterated_array = np.zeros(shape=shape_native)

        threshold_mask_lower_sub = mask

        for sub_size in self.sub_steps[:-1]:
            array_higher_sub = self.array_at_sub_size_from(
                func=func, cls=cls, mask=threshold_mask_lower_sub, sub_size=sub_size
            )

            try:
                threshold_mask_higher_sub = self.threshold_mask_via_arrays_from(
                    array_lower_sub_2d=array_lower_sub_2d,
                    array_higher_sub_2d=array_higher_sub,
                )

                iterated_array = iterated_array_jit_from(
                    iterated_array=iterated_array,
                    threshold_mask_higher_sub=np.array(threshold_mask_higher_sub),
                    threshold_mask_lower_sub=np.array(threshold_mask_lower_sub),
                    array_higher_sub_2d=np.array(array_higher_sub),
                )

            except ZeroDivisionError:
                return self.return_iterated_array_result(iterated_array=iterated_array, mask=mask)

            if threshold_mask_higher_sub.is_all_true:
                return self.return_iterated_array_result(iterated_array=iterated_array, mask=mask)

            array_lower_sub_2d = array_higher_sub
            threshold_mask_lower_sub = threshold_mask_higher_sub

        array_higher_sub = self.array_at_sub_size_from(
            func=func,
            cls=cls,
            mask=threshold_mask_lower_sub,
            sub_size=self.sub_steps[-1],
        )

        iterated_array_2d = iterated_array + array_higher_sub.binned.native

        return self.return_iterated_array_result(iterated_array=iterated_array_2d, mask=mask)

    def return_iterated_array_result(self, iterated_array: Array2D, mask : Mask2D) -> Array2D:
        """
        Returns the resulting iterated array, by mapping it to 1D and then passing it back as an ``Array2D`` structure.

        Parameters
        ----------
        iterated_array

        Returns
        -------
        iterated_array
            The resulting array computed via iteration.
        """

        iterated_array_1d = array_2d_util.array_2d_slim_from(
            mask_2d=np.array(mask),
            array_2d_native=np.array(iterated_array),
            sub_size=1,
        )

        return Array2D(values=iterated_array_1d, mask=mask.derive_mask.sub_1)

    def threshold_mask_via_grids_from(
        self, grid_lower_sub_2d: Grid2D, grid_higher_sub_2d: Grid2D
    ) -> Mask2D:
        """
        Returns a fractional mask from a result array, where the fractional mask describes whether the evaluated
        value in the result array is within the ``Iterator``'s specified fractional accuracy. The fractional mask thus
        determines whether a pixel on the grid needs to be reevaluated at a higher level of sub-gridding to meet the
        specified fractional accuracy. If it must be re-evaluated, the fractional masks's entry is ``False``.

        The fractional mask is computed by comparing the results evaluated at one level of sub-gridding to another
        at a higher level of sub-griding. Thus, the sub-grid size in chosen on a per-pixel basis until the function
        is evaluated at the specified fractional accuracy.

        Parameters
        ----------
        grid_lower_sub_2d
            The results computed by a function using a lower sub-grid size
        grid_higher_sub_2d : grids.Array2D
            The results computed by a function using a higher sub-grid size.
        """

        threshold_mask = Mask2D.all_false(
            shape_native=grid_lower_sub_2d.shape_native,
            pixel_scales=grid_lower_sub_2d.pixel_scales,
            invert=True,
        )

        threshold_mask = threshold_mask_via_grids_jit_from(
            fractional_accuracy_threshold=self.fractional_accuracy,
            relative_accuracy_threshold=self.relative_accuracy,
            threshold_mask=np.array(threshold_mask),
            grid_higher_sub_2d=np.array(grid_higher_sub_2d),
            grid_lower_sub_2d=np.array(grid_lower_sub_2d),
            grid_higher_mask=np.array(grid_higher_sub_2d.mask),
        )

        return Mask2D(
            mask=threshold_mask,
            pixel_scales=grid_higher_sub_2d.pixel_scales,
            origin=grid_higher_sub_2d.origin,
        )

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

        An example use case of this function is when a "deflections_yx_2d_from" methods in **PyAutoLens**'s ``MassProfile``
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

        shape_native = grid_lower_sub_2d.shape_native
        mask = grid_lower_sub_2d.mask

        iterated_grid = np.zeros(shape=(shape_native[0], shape_native[1], 2))

        threshold_mask_lower_sub = mask

        for sub_size in self.sub_steps[:-1]:
            grid_higher_sub = self.grid_at_sub_size_from(
                func=func, cls=cls, mask=threshold_mask_lower_sub, sub_size=sub_size
            )

            threshold_mask_higher_sub = self.threshold_mask_via_grids_from(
                grid_lower_sub_2d=grid_lower_sub_2d, grid_higher_sub_2d=grid_higher_sub
            )

            iterated_grid = iterated_grid_jit_from(
                iterated_grid=iterated_grid,
                threshold_mask_higher_sub=np.array(threshold_mask_higher_sub),
                threshold_mask_lower_sub=np.array(threshold_mask_lower_sub),
                grid_higher_sub_2d=np.array(grid_higher_sub),
            )

            if threshold_mask_higher_sub.is_all_true:
                iterated_grid_1d = grid_2d_util.grid_2d_slim_from(
                    mask=mask, grid_2d_native=iterated_grid, sub_size=1
                )

                return Grid2D(values=iterated_grid_1d, mask=mask.derive_mask.sub_1)

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
            mask=mask, grid_2d_native=iterated_grid_2d, sub_size=1
        )

        return Grid2D(values=iterated_grid_1d, mask=mask.derive_mask.sub_1)

    def iterated_result_from(
        self, func: Callable, cls: object, grid : Grid2D
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
        func
            The function which is iterated over to compute a more precise evaluation.
        cls
            The class the function belongs to.
        grid
            The 2D grid whose values input into the function are iterated over.
        """
        result_sub_1_1d = func(cls, grid)
        result_sub_1_2d = grid.structure_2d_from(
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
