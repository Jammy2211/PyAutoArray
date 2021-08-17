import numpy as np
import scipy.spatial.qhull as qhull

from autoconf import conf
from autoarray.structures.grids.two_d.abstract_grid_2d import AbstractGrid2D
from autoarray.structures.grids.two_d.grid_2d import Grid2D
from autoarray.structures.grids.two_d.grid_2d import Grid2DTransformed
from autoarray.structures.grids.two_d.grid_2d import Grid2DTransformedNumpy
from autoarray.mask.mask_2d import Mask2D

from autoarray.structures.arrays.two_d import array_2d as a2d

from autoarray.structures.grids import abstract_grid
from autoarray.geometry import geometry_util
from autoarray.structures.grids.two_d import grid_2d_util


class Grid2DInterpolate(AbstractGrid2D):
    def __new__(cls, grid, mask, pixel_scales_interp, *args, **kwargs):
        """
        Represents a grid of coordinates as described in the `Grid2D` class, but allows for a sparse grid to be used
        to evaluate functions on the grid, the results of which are then interpolated to the grid's native resolution.

        This sparse grid, termed the 'interpolation grid', is computed from the full resolution grid and an input
        pixel_scales_interp. The interpolation grid is laid over the full resolution grid, with all unmasked
        pixels used to set up the interpolation grid. The neighbors of masked pixels are also included, to ensure the
        interpolation evaluate pixels at the edge of the mask

        The decision whether to evaluate the function using the sparse grid and interpolate to th full resolution grid
        is made in the `grid_like_to_grid` decorator. For every function that can receive a Grid2DInterpolate, there is
        an entry in the 'interpolate.ini' config file where a bool determines if the interpolation is used.

        For functions which can be evaluated fast the interpolation should be turned off, ensuring the calculation is
        accurate and precise. However, if the function is slow to evaluate (e.g. it requires numerical integration)
        its bool in this config file should be True, such that the interpolation method is used instead.

        This feature is used in the light profiles and mass profiles of the projects PyAutoGalaxy and PyAutoLens.
        For example, for many mass profiles computing their deflection angles requires expensive numerical integration.
        However, the deflection angles do not vary much locally, so drastically fewer function evaluations can be
        performed by calculating it on a sparse grid interpolating to the full resolution grid.

        Parameters
        ----------
        grid
            The (y,x) coordinates of the grid.
        mask :Mask2D
            The 2D mask associated with the grid, defining the pixels each grid coordinate is paired with and
            originates from.
        pixel_scales_interp : float
            The resolution of the sparse grid used to evaluate the function, from which the results are interpolated
            to the full resolution grid.
        """

        obj = grid.view(cls)
        obj.mask = mask
        obj.pixel_scales_interp = pixel_scales_interp

        rescale_factor = mask.pixel_scale / pixel_scales_interp[0]

        mask = mask.mask_sub_1

        rescaled_mask = mask.rescaled_mask_from_rescale_factor(
            rescale_factor=rescale_factor
        )

        mask_interp = rescaled_mask.edge_buffed_mask

        grid_interp = grid_2d_util.grid_2d_slim_via_mask_from(
            mask_2d=mask_interp,
            pixel_scales=pixel_scales_interp,
            sub_size=1,
            origin=mask.origin,
        )

        obj.grid_interp = Grid2D.manual_mask(grid=grid_interp, mask=mask_interp)

        obj.vtx, obj.wts = obj.interp_weight_list

        return obj

    def __array_finalize__(self, obj):

        super().__array_finalize__(obj)

        if hasattr(obj, "pixel_scales_interp"):
            self.pixel_scales_interp = obj.pixel_scales_interp

        if hasattr(obj, "grid_interp"):
            self.grid_interp = obj.grid_interp

        if hasattr(obj, "vtx"):
            self.vtx = obj.vtx

        if hasattr(obj, "wts"):
            self.wts = obj.wts

    def _new_structure(self, grid, mask):
        """
        Conveninence method for creating a new instance of the Grid2DInterpolate class from this grid.

        This method is used in the 'slim', 'native', etc. convenience methods. By overwriting this method such that a
        Grid2DInterpolate is created the slim and native methods will return instances of the Grid2DInterpolate.

        Parameters
        ----------
        grid or list
            The (y,x) coordinates of the grid input as an ndarray of shape [total_sub_coordinates, 2] or list of lists.
        mask :Mask2D
            The 2D mask associated with the grid, defining the pixels each grid coordinate is paired with and
            originates from.
        """
        return Grid2DInterpolate(
            grid=grid, mask=mask, pixel_scales_interp=self.pixel_scales_interp
        )

    @classmethod
    def manual_slim(
        cls,
        grid,
        shape_native,
        pixel_scales,
        pixel_scales_interp,
        sub_size=1,
        origin=(0.0, 0.0),
    ):
        """
        Create a Grid2DInterpolate (see `Grid2DInterpolate.__new__`) by inputting the grid coordinates in 1D, for
        example:

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
        pixel_scales_interp or float
            The resolution of the sparse grid used to evaluate the function, from which the results are interpolated
            to the full resolution grid.
        origin
            The origin of the grid's mask.
        """
        grid = abstract_grid.convert_grid(grid=grid)
        pixel_scales = geometry_util.convert_pixel_scales_2d(pixel_scales=pixel_scales)
        pixel_scales_interp = geometry_util.convert_pixel_scales_2d(
            pixel_scales=pixel_scales_interp
        )

        mask = Mask2D.unmasked(
            shape_native=shape_native,
            pixel_scales=pixel_scales,
            sub_size=sub_size,
            origin=origin,
        )

        return Grid2DInterpolate(
            grid=grid, mask=mask, pixel_scales_interp=pixel_scales_interp
        )

    @classmethod
    def uniform(
        cls,
        shape_native,
        pixel_scales,
        pixel_scales_interp,
        sub_size=1,
        origin=(0.0, 0.0),
    ):
        """
        Create a Grid2DInterpolate (see `Grid2DInterpolate.__new__`) as a uniform grid of (y,x) values given an input
        shape_native and pixel scale of the grid:

        Parameters
        ----------
        shape_native
            The 2D shape of the uniform grid and the mask that it is paired with.
        pixel_scales
            The (y,x) scaled units to pixel units conversion factors of every pixel. If this is input as a ``float``,
            it is converted to a (float, float) structure.
        pixel_scales_interp : float
            The resolution of the sparse grid used to evaluate the function, from which the results are interpolated
            to the full resolution grid.
        origin
            The origin of the grid's mask.
        """

        pixel_scales = geometry_util.convert_pixel_scales_2d(pixel_scales=pixel_scales)
        pixel_scales_interp = geometry_util.convert_pixel_scales_2d(
            pixel_scales=pixel_scales_interp
        )

        grid_slim = grid_2d_util.grid_2d_slim_via_shape_native_from(
            shape_native=shape_native,
            pixel_scales=pixel_scales,
            sub_size=sub_size,
            origin=origin,
        )

        return Grid2DInterpolate.manual_slim(
            grid=grid_slim,
            shape_native=shape_native,
            pixel_scales=pixel_scales,
            pixel_scales_interp=pixel_scales_interp,
            sub_size=sub_size,
            origin=origin,
        )

    @classmethod
    def from_mask(cls, mask, pixel_scales_interp):
        """
        Create a Grid2DInterpolate (see `Grid2DInterpolate.__new__`) from a mask, where only unmasked pixels are
        included in the grid if the grid is represented in 2D masked values are (0.0, 0.0).

        The mask's pixel_scales and origin properties are used to compute the grid (y,x) coordinates.

        Parameters
        ----------
        mask : Mask2D
            The mask whose masked pixels are used to setup the sub-pixel grid.
        pixel_scales_interp : float
            The resolution of the sparse grid used to evaluate the function, from which the results are interpolated
            to the full resolution grid.
        """

        pixel_scales_interp = geometry_util.convert_pixel_scales_2d(
            pixel_scales=pixel_scales_interp
        )

        grid_slim = grid_2d_util.grid_2d_slim_via_mask_from(
            mask_2d=mask,
            pixel_scales=mask.pixel_scales,
            sub_size=mask.sub_size,
            origin=mask.origin,
        )

        return Grid2DInterpolate(
            grid=grid_slim, mask=mask, pixel_scales_interp=pixel_scales_interp
        )

    @classmethod
    def blurring_grid_from_mask_and_kernel_shape(
        cls, mask, kernel_shape_native, pixel_scales_interp
    ):
        """
        Setup a blurring-grid from a mask, where a blurring grid consists of all pixels that are masked and
        therefore have their values set to (0.0, 0.0), but are close enough to the unmasked pixels that their values
        will be convolved into the unmasked those pixels. This when computing images from
        light profile objects.

        See *Grid2D.blurring_grid_from_mask_and_kernel_shape* for a full description of a blurring grid. This
        method creates the blurring grid as a Grid2DInterpolate.

        Parameters
        ----------
        mask : Mask2D
            The mask whose masked pixels are used to setup the blurring grid.
        kernel_shape_native
            The 2D shape of the kernel which convolves signal from masked pixels to unmasked pixels.
        pixel_scales_interp : float
            The resolution of the sparse grid used to evaluate the function, from which the results are interpolated
            to the full resolution grid.
        """

        blurring_mask = mask.blurring_mask_from_kernel_shape(
            kernel_shape_native=kernel_shape_native
        )

        return cls.from_mask(
            mask=blurring_mask, pixel_scales_interp=pixel_scales_interp
        )

    def blurring_grid_from_kernel_shape(self, kernel_shape_native):
        """
        Returns the blurring grid from a grid and create it as a GridItnterpolate, via an input 2D kernel shape.

        For a full description of blurring grids, checkout *blurring_grid_from_mask_and_kernel_shape*.

        Parameters
        ----------
        kernel_shape_native
            The 2D shape of the kernel which convolves signal from masked pixels to unmasked pixels.
        """

        return Grid2DInterpolate.blurring_grid_from_mask_and_kernel_shape(
            mask=self.mask,
            kernel_shape_native=kernel_shape_native,
            pixel_scales_interp=self.pixel_scales_interp,
        )

    def padded_grid_from_kernel_shape(self, kernel_shape_native):
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

        return Grid2DInterpolate.from_mask(
            mask=padded_mask, pixel_scales_interp=self.pixel_scales_interp
        )

    @property
    def interp_weight_list(self):
        """
        The weight_list of the interpolation scheme between the interpolation grid and grid at native resolution.
        """

        tri = qhull.Delaunay(self.grid_interp.slim)
        simplex = tri.find_simplex(self.slim)
        # noinspection PyUnresolvedReferences
        vertices = np.take(tri.simplices, simplex, axis=0)
        temp = np.take(tri.transform, simplex, axis=0)
        delta = self.slim - temp[:, 2]
        bary = np.einsum("njk,nk->nj", temp[:, :2, :], delta)

        return vertices, np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))

    def result_from_func(self, func, cls):
        """
        Return the result of a function evaluation for a function which uses the grid to return either an
        Array2D or a Grid2D. The function uses the input Grid2DInterpolate as follows:

        1. If the function's entry in the 'interpolate.ini' config file is False, the function will not be evaluated
        using the interpolation grid and simply use the Grid2D at native resolution.

        2. If the function's entry in the 'interpolate.ini' config file is True, the function will be evaluated using
        the interpolation grid, with this result then interpolated to the Grid2D at native resolution.

        The function may return either an Array2D or Grid2D object, in both cases the interpolation may be used, where
        for the later two independent interpolations are performed on the y and x coordinate Grids.

        Parameters
        ----------
        func : func
            The function which may be evaluated using the interpolation grid.
        cls : object
            The class the function belongs to.
        """

        try:

            interpolate = conf.instance["grids"]["interpolate"][func.__name__][
                cls.__class__.__name__
            ]

        except Exception:

            interpolate = False

        if interpolate:

            result_interp = func(cls, self.grid_interp)
            if len(result_interp.shape) == 1:
                return self.interpolated_array_from_array_interp(
                    array_interp=result_interp
                )
            elif len(result_interp.shape) == 2:
                return self.interpolated_grid_from_grid_interp(
                    grid_interp=result_interp
                )

        else:

            result = func(cls, self)

            return self.structure_2d_from_result(result=result)

    def interpolated_array_from_array_interp(self, array_interp):
        """
        Use the precomputed vertexes and weight_list of a Delaunay gridding to interpolate a set of values computed on
        the interpolation grid to the Grid2DInterpolate's full grid.

        This function is taken from the SciPy interpolation method griddata
        (see https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.griddata.html). It is adapted here
        to reuse pre-computed interpolation vertexes and weight_list for efficiency.

        Parameters
        ----------
        array_interp : Array2D
            The results of the function evaluated using the interpolation grid, which is interpolated to the native
            resolution Array2D.
        """

        interpolated_array = np.einsum(
            "nj,nj->n", np.take(array_interp, self.vtx), self.wts
        )
        return a2d.Array2D(array=interpolated_array, mask=self.mask)

    def interpolated_grid_from_grid_interp(self, grid_interp) -> Grid2D:
        """
        Use the precomputed vertexes and weight_list of a Delaunay gridding to interpolate a grid of (y,x) values values
        computed on  the interpolation grid to the Grid2DInterpolate's full grid.

        This function is taken from the SciPy interpolation method griddata
        (see https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.griddata.html). It is adapted here
        to reuse pre-computed interpolation vertexes and weight_list for efficiency.

        Parameters
        ----------
        grid_interp : Grid2D
            The results of the function evaluated using the interpolation grid, which is interpolated to the native
            resolution Grid2D.
        """
        y_values = self.interpolated_array_from_array_interp(
            array_interp=grid_interp[:, 0]
        )
        x_values = self.interpolated_array_from_array_interp(
            array_interp=grid_interp[:, 1]
        )
        grid = np.asarray([y_values, x_values]).T
        return Grid2D(grid=grid, mask=self.mask)

    def structure_2d_from_result(self, result: np.ndarray):
        """
        Convert a result from an ndarray to an aa.Array2D or aa.Grid2D structure, where the conversion depends on
        type(result) as follows:

        - 1D np.ndarray   -> aa.Array2D
        - 2D np.ndarray   -> aa.Grid2D

        This function is used by the grid_2d_to_structure decorator to convert the output result of a function
        to an autoarray structure when a `Grid2D` instance is passed to the decorated function.

        Parameters
        ----------
        result
            The input result (e.g. of a decorated function) that is converted to a PyAutoArray structure.
        """

        if len(result.shape) == 1:
            return a2d.Array2D(array=result, mask=self.mask)
        else:
            if isinstance(result, Grid2DTransformedNumpy):
                return Grid2DTransformed(grid=result, mask=self.mask)
            return Grid2D(grid=result, mask=self.mask)

    def structure_2d_list_from_result_list(self, result_list: list):
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
        return [self.structure_2d_from_result(result=result) for result in result_list]
