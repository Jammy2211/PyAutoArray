import numpy as np
from typing import Callable, Union, List, Optional

from autoarray import numba_util
from autoarray.mask.mask_2d import Mask2D
from autoarray.operators.over_sample.abstract import AbstractOverSample
from autoarray.operators.over_sample.abstract import AbstractOverSampleFunc
from autoarray.operators.over_sample.uniform import OverSampleUniformFunc
from autoarray.structures.arrays.uniform_2d import Array2D
from autoarray.structures.grids.uniform_2d import Grid2D

from autoarray.structures.arrays import array_2d_util


class OverSampleIterate(AbstractOverSample):
    def __init__(
        self,
        fractional_accuracy: float = 0.9999,
        relative_accuracy: Optional[float] = None,
        sub_steps: List[int] = None,
    ):
        """
        Over samples grid calculations using an iterative sub-grid that increases the sampling until a threshold
        accuracy is met.

        When a 2D grid of (y,x) coordinates is input into a function, the result is evaluated at every coordinate
        on the grid. When the grid is paired to a 2D image (e.g. an `Array2D`) the solution needs to approximate
        the 2D integral of that function in each pixel. Over sample objects define how this over-sampling is performed.

        This object iteratively recomputes the analytic function at increasing sub-grid resolutions until an input
        fractional accuracy is reached. The sub-grid is increase in each pixel, therefore it will gradually better
        approximate the 2D integral after each iteration.

        Iteration is performed on a per pixel basis, such that the sub-grid size will stop at lower values
        in pixels where the fractional accuracy is met quickly. It will only go to high values where high sampling is
        required to meet the accuracy. This ensures the function is evaluated accurately in a computationally
        efficient manner.

        Parameters
        ----------
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
        """

        if sub_steps is None:
            sub_steps = [2, 4, 8, 16]

        self.fractional_accuracy = fractional_accuracy
        self.relative_accuracy = relative_accuracy
        self.sub_steps = sub_steps


# @numba_util.jit()
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


# @numba_util.jit()
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
            if threshold_mask_higher_sub[y, x] and not threshold_mask_lower_sub[y, x]:
                iterated_array[y, x] = array_higher_sub_2d[y, x]

    return iterated_array


class OverSampleIterateFunc(AbstractOverSampleFunc):
    def __init__(
        self,
        mask: Mask2D,
        fractional_accuracy: float = 0.9999,
        relative_accuracy: Optional[float] = None,
        sub_steps: List[int] = None,
    ):
        self.mask = mask
        self.fractional_accuracy = fractional_accuracy
        self.relative_accuracy = relative_accuracy
        self.sub_steps = sub_steps

    @property
    def over_sample(self):
        return OverSampleIterate()

    def array_at_sub_size_from(
        self, func: Callable, cls, mask: Mask2D, sub_size
    ) -> Array2D:
        over_sample_uniform = OverSampleUniformFunc(mask=mask, sub_size=sub_size)

        oversampled_grid = over_sample_uniform.oversampled_grid

        array_higher_sub = func(cls, oversampled_grid)
        # array = self.structure_2d_from(
        #     result=array_higher_sub,
        # )
        return over_sample_uniform.binned_array_2d_from(array=array_higher_sub).native

    def threshold_mask_from(
        self, array_lower_sub_2d: Array2D, array_higher_sub_2d: Array2D
    ) -> Mask2D:
        """
        Returns a fractional mask from a result array, where the fractional mask describes whether the evaluated
        value in the result array is within the ``OverSampleIterate``'s specified fractional accuracy. The fractional mask thus
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
            pixel_scales=array_lower_sub_2d.pixel_scales,
            origin=array_lower_sub_2d.origin,
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

        iterated_array = np.zeros(shape=self.mask.shape_native)

        threshold_mask_lower_sub = self.mask

        for sub_size in self.sub_steps[:-1]:
            array_higher_sub = self.array_at_sub_size_from(
                func=func, cls=cls, mask=threshold_mask_lower_sub, sub_size=sub_size
            )

            try:
                threshold_mask_higher_sub = self.threshold_mask_from(
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
                return self.return_iterated_array_result(
                    iterated_array=iterated_array,
                )

            if threshold_mask_higher_sub.is_all_true:
                return self.return_iterated_array_result(
                    iterated_array=iterated_array,
                )

            array_lower_sub_2d = array_higher_sub
            threshold_mask_lower_sub = threshold_mask_higher_sub
            threshold_mask_higher_sub.pixel_scales = self.mask.pixel_scales

        array_higher_sub = self.array_at_sub_size_from(
            func=func,
            cls=cls,
            mask=threshold_mask_lower_sub,
            sub_size=self.sub_steps[-1],
        )

        iterated_array_2d = iterated_array + array_higher_sub

        return self.return_iterated_array_result(
            iterated_array=iterated_array_2d,
        )

    def return_iterated_array_result(
        self,
        iterated_array: Array2D,
    ) -> Array2D:
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
            mask_2d=np.array(self.mask),
            array_2d_native=np.array(iterated_array),
            sub_size=1,
        )

        return Array2D(values=iterated_array_1d, mask=self.mask)

    def iterated_result_from(
        self, func: Callable, cls: object, grid: Grid2D
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

        # Convert to numpy array so grid does not do iteration.

        result_sub_1_1d = func(cls, np.asarray(grid))

        result_sub_1_2d = self.structure_2d_from(
            result=result_sub_1_1d,
        ).native

        return self.iterated_array_from(
            func=func, cls=cls, array_lower_sub_2d=result_sub_1_2d
        )
