import numpy as np

from autoarray import decorator_util
from autoarray.structures import arrays, grids
from autoarray.mask import mask as msk
from autoarray.util import grid_util
from autoarray import exc


class GridIterator(grids.Grid):
    def __new__(
        cls,
        grid,
        mask,
        fractional_accuracy=0.9999,
        sub_steps=[2, 4, 8, 16],
        store_in_1d=True,
        *args,
        **kwargs,
    ):
        """Represents a grid of coordinates as described for the *Grid* class, but using an iterative sub-grid that
        adapts its resolution when it is input into a function that performs an analytic calculation.

        A *Grid* represents (y,x) coordinates using a sub-grid, where functions are evaluated once at every coordinate
        on the sub-grid and averaged to give a more precise evaluation an analytic function. A *GridIterator* does not
        have a specified sub-grid size, but instead iteratively recomputes the analytic function at increasing sub-grid
        sizes until an input fractional accuracy is reached.

        Iteration is performed on a per (y,x) coordinate basis, such that the sub-grid size will adopt low values
        wherever doing so can meet the fractional accuracy and high values only where it is required to meet the
        fractional accuracy. For functions where a wide range of sub-grid sizes allow fractional accuracy to be met
        this ensures the function can be evaluated accurate in a computaionally efficient manner.

        This overcomes a limitation of the *Grid* class whereby if a small subset of pixels require high levels of
        sub-gridding to be evaluated accuracy, the entire grid would require this sub-grid size thus leading to
        unecessary expensive function evaluations.


        """
        obj = super().__new__(cls=cls, grid=grid, mask=mask, store_in_1d=store_in_1d)
        obj.grid = grids.MaskedGrid.manual_1d(grid=grid, mask=mask)
        obj.fractional_accuracy = fractional_accuracy
        obj.sub_steps = sub_steps
        obj.binned = True
        return obj

    @classmethod
    def manual_1d(
        cls,
        grid,
        shape_2d,
        pixel_scales,
        origin=(0.0, 0.0),
        fractional_accuracy=0.9999,
        sub_steps=[2, 4, 8, 16],
        store_in_1d=True,
    ):

        if type(grid) is list:
            grid = np.asarray(grid)

        if type(pixel_scales) is float:
            pixel_scales = (pixel_scales, pixel_scales)

        if grid.shape[-1] != 2:
            raise exc.GridException(
                "The final dimension of the input grid is not equal to 2 (e.g. the (y,x) coordinates)"
            )

        if 2 < len(grid.shape) > 3:
            raise exc.GridException(
                "The dimensions of the input grid array is not 2 or 3"
            )

        mask = msk.Mask.unmasked(
            shape_2d=shape_2d, pixel_scales=pixel_scales, sub_size=1, origin=origin
        )

        return GridIterator(
            grid=grid,
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
        sub_steps=[2, 4, 8, 16],
        store_in_1d=True,
    ):

        grid_1d = grid_util.grid_1d_via_shape_2d_from(
            shape_2d=shape_2d, pixel_scales=pixel_scales, sub_size=1, origin=origin
        )

        return GridIterator.manual_1d(
            grid=grid_1d,
            fractional_accuracy=fractional_accuracy,
            sub_steps=sub_steps,
            store_in_1d=store_in_1d,
        )

    @classmethod
    def from_mask(
        cls, mask, fractional_accuracy=0.9999, sub_steps=[2, 4, 8, 16], store_in_1d=True
    ):
        """Setup a sub-grid of the unmasked pixels, using a mask and a specified sub-grid size. The center of \
        every unmasked pixel's sub-pixels give the grid's (y,x) scaled coordinates.

        Parameters
        -----------
        mask : Mask
            The mask whose masked pixels are used to setup the sub-pixel grid.
        sub_size : int
            The size (sub_size x sub_size) of each unmasked pixels sub-grid.
        """

        grid_1d = grid_util.grid_1d_via_mask_2d_from(
            mask_2d=mask, pixel_scales=mask.pixel_scales, sub_size=1, origin=mask.origin
        )

        return GridIterator(
            grid=grid_1d,
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
        sub_steps=[2, 4, 8, 16],
        store_in_1d=True,
    ):
        """Setup a blurring-grid from a mask, where a blurring grid consists of all pixels that are masked, but they \
        are close enough to the unmasked pixels that a fraction of their light will be blurred into those pixels \
        via PSF convolution. For example, if our mask is as follows:

        |x|x|x|x|x|x|x|x|x|x|
        |x|x|x|x|x|x|x|x|x|x|     This is an imaging.Mask, where:
        |x|x|x|x|x|x|x|x|x|x|
        |x|x|x|x|x|x|x|x|x|x|     x = True (Pixel is masked and excluded from lens)
        |x|x|x|o|o|o|x|x|x|x|     o = False (Pixel is not masked and included in lens)
        |x|x|x|o|o|o|x|x|x|x|
        |x|x|x|o|o|o|x|x|x|x|
        |x|x|x|x|x|x|x|x|x|x|
        |x|x|x|x|x|x|x|x|x|x|
        |x|x|x|x|x|x|x|x|x|x|

        For a PSF of shape (3,3), the following blurring mask is computed (noting that only pixels that are direct \
        neighbors of the unmasked pixels above will blur light into an unmasked pixel):

        |x|x|x|x|x|x|x|x|x|     This is an example grid.Mask, where:
        |x|x|x|x|x|x|x|x|x|
        |x|x|o|o|o|o|o|x|x|     x = True (Pixel is masked and excluded from lens)
        |x|x|o|x|x|x|o|x|x|     o = False (Pixel is not masked and included in lens)
        |x|x|o|x|x|x|o|x|x|
        |x|x|o|x|x|x|o|x|x|
        |x|x|o|o|o|o|o|x|x|
        |x|x|x|x|x|x|x|x|x|
        |x|x|x|x|x|x|x|x|x|

        Thus, the blurring grid coordinates and indexes will be as follows:

        pixel_scales = 1.0"

        <--- -ve  x  +ve -->
                                                            y     x
        |x|x|x |x |x |x |x |x|x|  |   blurring_grid[0] = [2.0, -2.0]  blurring_grid[9] =  [-1.0, -2.0]
        |x|x|x |x |x |x |x |x|x|  |   blurring_grid[1] = [2.0, -1.0]  blurring_grid[10] = [-1.0,  2.0]
        |x|x|0 |1 |2 |3 |4 |x|x| +ve  blurring_grid[2] = [2.0,  0.0]  blurring_grid[11] = [-2.0, -2.0]
        |x|x|5 |x |x |x |6 |x|x|  y   blurring_grid[3] = [2.0,  1.0]  blurring_grid[12] = [-2.0, -1.0]
        |x|x|7 |x |x |x |8 |x|x| -ve  blurring_grid[4] = [2.0,  2.0]  blurring_grid[13] = [-2.0,  0.0]
        |x|x|9 |x |x |x |10|x|x|  |   blurring_grid[5] = [1.0, -2.0]  blurring_grid[14] = [-2.0,  1.0]
        |x|x|11|12|13|14|15|x|x|  |   blurring_grid[6] = [1.0,  2.0]  blurring_grid[15] = [-2.0,  2.0]
        |x|x|x |x |x |x |x |x|x| \/   blurring_grid[7] = [0.0, -2.0]
        |x|x|x |x |x |x |x |x|x|      blurring_grid[8] = [0.0,  2.0]

        For a PSF of shape (5,5), the following blurring mask is computed (noting that pixels that are 2 pixels from an
        direct unmasked pixels now blur light into an unmasked pixel):

        |x|x|x|x|x|x|x|x|x|     This is an example grid.Mask, where:
        |x|o|o|o|o|o|o|o|x|
        |x|o|o|o|o|o|o|o|x|     x = True (Pixel is masked and excluded from lens)
        |x|o|o|x|x|x|o|o|x|     o = False (Pixel is not masked and included in lens)
        |x|o|o|x|x|x|o|o|x|
        |x|o|o|x|x|x|o|o|x|
        |x|o|o|o|o|o|o|o|x|
        |x|o|o|o|o|o|o|o|x|
        |x|x|x|x|x|x|x|x|x|
        """

        blurring_mask = mask.regions.blurring_mask_from_kernel_shape(
            kernel_shape_2d=kernel_shape_2d
        )

        return GridIterator.from_mask(
            mask=blurring_mask,
            fractional_accuracy=fractional_accuracy,
            sub_steps=sub_steps,
            store_in_1d=store_in_1d,
        )

    def blurring_grid_from_kernel_shape(self, kernel_shape_2d):
        """From this grid, determine the blurring grid.

        The blurring grid gives the (y,x) coordinates of all pixels which are masked but whose light will be blurred
        into unmasked due to 2D convolution. These pixels are determined by this grid's mask and the 2D shape of
        the *Kernel*.

        Parameters
        ----------
        kernel_shape_2d : (int, int)
            The 2D shape of the Kernel used to determine which masked pixel's values will be blurred into the grid's
            unmasked pixels by 2D convolution.
        """

        blurring_mask = self.mask.regions.blurring_mask_from_kernel_shape(
            kernel_shape_2d=kernel_shape_2d
        )

        return GridIterator.from_mask(
            mask=blurring_mask,
            fractional_accuracy=self.fractional_accuracy,
            sub_steps=self.sub_steps,
            store_in_1d=self.store_in_1d,
        )

    def __array_finalize__(self, obj):

        super(GridIterator, self).__array_finalize__(obj)

        if hasattr(obj, "grid"):
            self.grid = obj.grid

        if hasattr(obj, "fractional_accuracy"):
            self.fractional_accuracy = obj.fractional_accuracy

        if hasattr(obj, "sub_steps"):
            self.sub_steps = obj.sub_steps

    def fractional_mask_from_arrays(
        self, array_lower_sub_2d, array_higher_sub_2d
    ) -> msk.Mask:
        """ Compute a fractional mask from a result array, where the fractional mask describes whether the evaluated
        value in the result array is within the *GridIterator*'s specified fractional accuracy. The fractional mask thus
        determines whether a pixel on the grid needs to be reevaluated at a higher level of sub-gridding to meet the
        specified fractional accuracy. If it must be re-evaluated, the fractional masks's entry is *False*.

        The fractional mask is computed by comparing the results evaluated at one level of sub-gridding to another
        at a higher level of sub-griding. Thus, the sub-grid size in chosen on a per-pixel basis until the function
        is evaluated at the specified fractional accuracy.

        Parameters
        ----------
        result_array_lower_sub : arrays.Array
            The results computed by a function using a lower sub-grid size
        result_array_lower_sub : arrays.Array
            The results computed by a function using a lower sub-grid size.
        """

        fractional_mask = msk.Mask.unmasked(
            shape_2d=array_lower_sub_2d.shape_2d, invert=True
        )

        fractional_mask = self.fractional_mask_jit_from_array(
            fractional_accuracy_threshold=self.fractional_accuracy,
            fractional_mask=fractional_mask,
            array_higher_sub_2d=array_higher_sub_2d,
            array_lower_sub_2d=array_lower_sub_2d,
            array_higher_mask_2d=array_higher_sub_2d.mask,
        )

        return msk.Mask(
            mask_2d=fractional_mask,
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
        array_higher_mask_2d,
    ):
        """Jitted functioon to determine the fractional mask, which is a mask where:

        - *True* entries signify the function has been evaluated in that pixel to desired fractional accuracy and
           therefore does not need to be iteratively computed at higher levels of sub-gridding.

        - *False* entries signify the function has not been evaluated in that pixel to desired fractional accuracy and
           therefore must be iterative computed at higher levels of sub-gridding to meet this accuracy."""

        for y in range(fractional_mask.shape[0]):
            for x in range(fractional_mask.shape[1]):
                if not array_higher_mask_2d[y, x]:

                    if array_higher_sub_2d[y, x] > 0:

                        fractional_accuracy = (
                            array_lower_sub_2d[y, x] / array_higher_sub_2d[y, x]
                        )

                    else:

                        fractional_accuracy = 1.0

                    if fractional_accuracy > 1.0:
                        fractional_accuracy = 1.0 / fractional_accuracy

                    if fractional_accuracy < fractional_accuracy_threshold:
                        fractional_mask[y, x] = False

        return fractional_mask

    def iterated_array_from_func(
        self, func, profile, array_lower_sub_2d
    ) -> arrays.Array:
        """Iterate over a function that returns an array of values until the it meets a specified fractional accuracy.
        The function returns a result on a pixel-grid where evaluating it on more points on a higher resolution
        sub-grid followed by binning lead to a more precise evaluation of the function.

        The function is first called for a sub-grid size of 1 and a higher resolution grid. The ratio of values give
        the fractional accuracy of each function evaluation. Pixels which do not meet the fractional accuracy are
        iteratively revaluated on higher resolution sub-grids. This is repeated until all pixels meet the fractional
        accuracy or the highest sub-size specified in the *sub_steps* attribute is computed.

        If the function return all zeros, the iteration is terminated early given that all levels of sub-gridding will
        return zeros. This occurs when a function is missing optional objects that contribute to the calculation.

        An example use case of this function is when a "profile_image_from_grid" methods in **PyAutoGalaxy**'s
        *LightProfile* module is comomputed, which by evaluating the function on a higher resolution sub-grids sample
        the analytic light profile at more points and thus more precisely.

        Parameters
        ----------
        func : func
            The function which is iterated over to compute a more precise evaluation."""

        if not np.any(array_lower_sub_2d):
            return array_lower_sub_2d.in_1d

        iterated_array = np.zeros(shape=self.shape_2d)

        fractional_mask_lower_sub = self.mask

        for sub_size in self.sub_steps:

            mask_higher_sub = fractional_mask_lower_sub.mapping.mask_new_sub_size_from_mask(
                mask=fractional_mask_lower_sub, sub_size=sub_size
            )

            grid_compute = grids.Grid.from_mask(mask=mask_higher_sub)
            array_higher_sub = func(profile, grid_compute)
            array_higher_sub = grid_compute.structure_from_result(
                result=array_higher_sub
            ).in_2d_binned

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
                return self.mask.mapping.array_stored_1d_from_array_2d(
                    array_2d=iterated_array
                )

            array_lower_sub_2d = array_higher_sub
            fractional_mask_lower_sub = fractional_mask_higher_sub

        return self.mask.mapping.array_stored_1d_from_array_2d(
            array_2d=iterated_array + array_higher_sub.in_2d_binned
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
        value in the result array is within the *GridIterator*'s specified fractional accuracy. The fractional mask thus
        determines whether a pixel on the grid needs to be reevaluated at a higher level of sub-gridding to meet the
        specified fractional accuracy. If it must be re-evaluated, the fractional masks's entry is *False*.

        The fractional mask is computed by comparing the results evaluated at one level of sub-gridding to another
        at a higher level of sub-griding. Thus, the sub-grid size in chosen on a per-pixel basis until the function
        is evaluated at the specified fractional accuracy.

        Parameters
        ----------
        result_array_lower_sub : arrays.Array
            The results computed by a function using a lower sub-grid size
        result_array_lower_sub : grids.Array
            The results computed by a function using a lower sub-grid size.
        """

        fractional_mask = msk.Mask.unmasked(
            shape_2d=grid_lower_sub_2d.shape_2d, invert=True
        )

        fractional_mask = self.fractional_mask_jit_from_grid(
            fractional_accuracy_threshold=self.fractional_accuracy,
            fractional_mask=fractional_mask,
            grid_higher_sub_2d=grid_higher_sub_2d,
            grid_lower_sub_2d=grid_lower_sub_2d,
            grid_higher_mask_2d=grid_higher_sub_2d.mask,
        )

        return msk.Mask(
            mask_2d=fractional_mask,
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
        grid_higher_mask_2d,
    ):
        """Jitted functioon to determine the fractional mask, which is a mask where:

        - *True* entries signify the function has been evaluated in that pixel to desired fractional accuracy and
           therefore does not need to be iteratively computed at higher levels of sub-gridding.

        - *False* entries signify the function has not been evaluated in that pixel to desired fractional accuracy and
           therefore must be iterative computed at higher levels of sub-gridding to meet this accuracy."""

        for y in range(fractional_mask.shape[0]):
            for x in range(fractional_mask.shape[1]):
                if not grid_higher_mask_2d[y, x]:

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

    def iterated_grid_from_func(self, func, profile, grid_lower_sub_2d):
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

        An example use case of this function is when a "deflections_from_grid" methods in **PyAutoLens**'s *MassProfile*
        module is computed, which by evaluating the function on a higher resolution sub-grid samples the analytic
        mass profile at more points and thus more precisely.

        Parameters
        ----------
        func : func
            The function which is iterated over to compute a more precise evaluation."""

        if not np.any(grid_lower_sub_2d):
            return grid_lower_sub_2d.in_1d

        iterated_grid = np.zeros(shape=(self.shape_2d[0], self.shape_2d[1], 2))

        fractional_mask_lower_sub = self.mask

        for sub_size in self.sub_steps:

            mask_higher_sub = fractional_mask_lower_sub.mapping.mask_new_sub_size_from_mask(
                mask=fractional_mask_lower_sub, sub_size=sub_size
            )

            grid_compute = grids.Grid.from_mask(mask=mask_higher_sub)
            grid_higher_sub = func(profile, grid_compute)
            grid_higher_sub = grid_compute.structure_from_result(
                result=grid_higher_sub
            ).in_2d_binned

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
                return self.mask.mapping.grid_stored_1d_from_grid_2d(
                    grid_2d=iterated_grid
                )

            grid_lower_sub_2d = grid_higher_sub
            fractional_mask_lower_sub = fractional_mask_higher_sub

        return self.mask.mapping.grid_stored_1d_from_grid_2d(
            grid_2d=iterated_grid + grid_higher_sub.in_2d_binned
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

    def iterated_result_from_func(self, func, profile):
        """Iterate over a function that returns an array or grid of values until the it meets a specified fractional
        accuracy. The function returns a result on a pixel-grid where evaluating it on more points on a higher
        resolution sub-grid followed by binning lead to a more precise evaluation of the function.

        A full description of the iteration method can be found in the functions *iterated_array_from_func* and
        *iterated_grid_from_func*. This function computes the result on a grid with a sub-size of 1, and uses its
        shape to call the correct function.

        Parameters
        ----------
        func : func
            The function which is iterated over to compute a more precise evaluation."""
        result_sub_1_1d = func(profile, self.grid)
        result_sub_1_2d = self.structure_from_result(
            result=result_sub_1_1d
        ).in_2d_binned

        if len(result_sub_1_2d.shape) == 2:
            return self.iterated_array_from_func(
                func=func, profile=profile, array_lower_sub_2d=result_sub_1_2d
            )
        elif len(result_sub_1_2d.shape) == 3:
            return self.iterated_grid_from_func(
                func=func, profile=profile, grid_lower_sub_2d=result_sub_1_2d
            )
