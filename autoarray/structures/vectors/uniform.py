import logging
import numpy as np
from typing import List, Tuple, Union

from autoarray.structures.arrays.uniform_2d import Array2D
from autoarray.structures.grids.uniform_2d import Grid2D
from autoarray.structures.vectors.abstract import AbstractVectorYX2D

from autoarray.mask.mask_2d import Mask2D
from autoarray.structures.grids import grid_2d_util
from autoarray.geometry import geometry_util

from autoarray import type as ty

logging.basicConfig()
logger = logging.getLogger(__name__)


class VectorYX2D(AbstractVectorYX2D):
    def __new__(
        cls,
        vectors: Union[np.ndarray, List[Tuple[float, float]]],
        grid: Union[Grid2D, List],
        mask,
    ):
        """
        A collection of (y,x) vectors which are located on a regular 2D grid of (y,x) coordinates.

        The vectors are paired to a uniform 2D mask of pixels and sub-pixels. Each vector corresponds to a value at
        the centre of a sub-pixel in an unmasked pixel.

        The `VectorYX2D` is ordered such that pixels begin from the top-row of the corresponding mask and go right
        and down. The positive y-axis is upwards and positive x-axis to the right.

        The (y,x) vectors are stored as a NumPy array which has the `slim` and `native shapes described below.
        Irrespective of this shape, the last dimension of the data structure storing the vectors is always shape 2,
        corresponding to the y and x vectors. [total_vectors, 2].

        Calculations should use the NumPy array structure wherever possible for efficient calculations.

        The vectors input to this function can have any of the following forms (they will be converted to the 1D NumPy
        array structure and can be converted back using the object's properties):

        [[vector_0_y, vector_0_x], [vector_1_y, vector_1_x]]
        [(vector_0_y, vector_0_x), (vector_1_y, vector_1_x)]

        If your vector field lies on a 2D irregular grid of data the `VectorFieldIrregular2D` data structure should be
        used.


        **Case 1 (sub-size=1, slim)**

        The Vector2D is an ndarray of shape [total_unmasked_pixels, 2].

        The first element of the ndarray corresponds to the pixel index, for example:

        - vector[3, 0:2] = the 4th unmasked pixel's y and x values.
        - vector[6, 0:2] = the 7th unmasked pixel's y and x values.

        Below is a visual illustration of a vector, where a total of 10 pixels are unmasked and are included in
        the vector.

         x x x x x x x x x x
         x x x x x x x x x x     This is an example `Mask2D`, where:
         x x x x x x x x x x
         x x x x O O x x x x     x = `True` (Pixel is masked and excluded from the vector)
         x x x O O O O x x x     O = `False` (Pixel is not masked and included in the vector)
         x x x O O O O x x x
         x x x x x x x x x x
         x x x x x x x x x x
         x x x x x x x x x x
         x x x x x x x x x x

        The mask pixel index's will come out like this (and the direction of scaled values is highlighted
        around the mask).

        pixel_scales = 1.0"

        <--- -ve  x  +ve -->
                                                        y      x
         x x x x x x x x x x  ^   vector[0, :] = 0
         x x x x x x x x x x  I   vector[1, :] = 1
         x x x x x x x x x x  I   vector[2, :] = 2
         x x x x 0 1 x x x x +ve  vector[3, :] = 3
         x x x 2 3 4 5 x x x  y   vector[4, :] = 4
         x x x 6 7 8 9 x x x -ve  vector[5, :] = 5
         x x x x x x x x x x  I   vector[6, :] = 6
         x x x x x x x x x x  I   vector[7, :] = 7
         x x x x x x x x x x \/   vector[8, :] = 8
         x x x x x x x x x x      vector[9, :] = 9


        **Case 2 (sub-size>1, slim)**

        If the masks's sub size is > 1, the vector is defined as a sub-vector where each entry corresponds to the
        values at the centre of each sub-pixel of an unmasked pixel.

        The sub-vector indexes are ordered such that pixels begin from the first (top-left) sub-pixel in the first
        unmasked pixel. Indexes then go over the sub-pixels in each unmasked pixel, for every unmasked pixel.
        Therefore, the sub-vector is an ndarray of shape [total_unmasked_pixels*(sub_array_shape)**2, 2]. For example:

        - vector[9, 0:2] - using a 2x2 sub-vector, gives the 3rd unmasked pixel's 2nd sub-pixel y and x values.
        - vector[9, 0:2] - using a 3x3 sub-vector, gives the 2nd unmasked pixel's 1st sub-pixel y and x values.
        - vector[27, 0:2] - using a 3x3 sub-vector, gives the 4th unmasked pixel's 1st sub-pixel y and x values.

        Below is a visual illustration of a sub vector. Indexing of each sub-pixel goes from the top-left corner. In
        contrast to the vector above, our illustration below restricts the mask to just 2 pixels, to keep the
        illustration brief.

         x x x x x x x x x x
         x x x x x x x x x x     This is an example `Mask2D`, where:
         x x x x x x x x x x
         x x x x x x x x x x     x = `True` (Pixel is masked and excluded from lens)
         x x x x O O x x x x     O = `False` (Pixel is not masked and included in lens)
         x x x x x x x x x x
         x x x x x x x x x x
         x x x x x x x x x x
         x x x x x x x x x x
         x x x x x x x x x x

        Our vector with a sub-size looks like it did before:

        <--- -ve  x  +ve -->

         x x x x x x x x x x  ^
         x x x x x x x x x x  I
         x x x x x x x x x x  I
         x x x x x x x x x x +ve
         x x x 0 1 x x x x x  y
         x x x x x x x x x x -ve
         x x x x x x x x x x  I
         x x x x x x x x x x  I
         x x x x x x x x x x \/
         x x x x x x x x x x

        However, if the sub-size is 2,each unmasked pixel has a set of sub-pixels with values. For example, for pixel 0,
        if `sub_size=2`, it has 4 values on a 2x2 sub-vector:

        Pixel 0 - (2x2):

               vector[0, 0:2] = y and x values of first sub-pixel in pixel 0.
        I0I1I  vector[1, 0:2] = y and x values of first sub-pixel in pixel 1.
        I2I3I  vector[2, 0:2] = y and x values of first sub-pixel in pixel 2.
               vector[3, 0:2] = y and x values of first sub-pixel in pixel 3.

        If we used a sub_size of 3, for the first pixel we we would create a 3x3 sub-vector:


                 vector[0] = y and x values of first sub-pixel in pixel 0.
                 vector[1] = y and x values of first sub-pixel in pixel 1.
                 vector[2] = y and x values of first sub-pixel in pixel 2.
        I0I1I2I  vector[3] = y and x values of first sub-pixel in pixel 3.
        I3I4I5I  vector[4] = y and x values of first sub-pixel in pixel 4.
        I6I7I8I  vector[5] = y and x values of first sub-pixel in pixel 5.
                 vector[6] = y and x values of first sub-pixel in pixel 6.
                 vector[7] = y and x values of first sub-pixel in pixel 7.
                 vector[8] = y and x values of first sub-pixel in pixel 8.


        **Case 3 (sub_size=1, native)**

        The Vector2D has the same properties as Case 1, but is stored as an an ndarray of shape
        [total_y_values, total_x_values, 2].

        All masked entries on the vector have values of 0.0.

        For the following example mask:

         x x x x x x x x x x
         x x x x x x x x x x     This is an example `Mask2D`, where:
         x x x x x x x x x x
         x x x x O O x x x x     x = `True` (Pixel is masked and excluded from the vector)
         x x x O O O O x x x     O = `False` (Pixel is not masked and included in the vector)
         x x x O O O O x x x
         x x x x x x x x x x
         x x x x x x x x x x
         x x x x x x x x x x
         x x x x x x x x x x

        For the example above:

            - vector[0,0, 0:2] = [0.0, 0.0] (it is masked, thus zero)
            - vector[0,0, 0:2] = [0.0, 0.0] (it is masked, thus zero)
            - vector[3,3, 0:2] = [0.0, 0.0] (it is masked, thus zero)
            - vector[3,3, 0:2] = [0.0, 0.0] (it is masked, thus zero)
            - vector[3,4, 0:2] = [0, 0]
            - vector[3,4, 0:2] = [-1, -1]


        **Case 4: (sub_size>, native)**

        The properties of this vector can be derived by combining Case's 2 and 3 above, whereby the vector is stored as
        an ndarray of shape [total_y_values*sub_size, total_x_values*sub_size, 2].

        All sub-pixels in masked pixels have values 0.0.

        Parameters
        ----------
        vectors
            The 2D (y,x) vectors on a regular grid that represent the vector-field.
        grid
            The regular grid of (y,x) coordinates where each vector is located.
        mask
            The 2D mask associated with the array, defining the pixels each array value is paired with and
            originates from.
        """

        if len(vectors) == 0:
            return []

        if type(vectors) is list:
            vectors = np.asarray(vectors)

        obj = vectors.view(cls)
        obj.grid = Grid2D(grid=grid, mask=mask)
        obj.mask = mask

        return obj

    def __array_finalize__(self, obj):

        if hasattr(obj, "mask"):
            self.mask = obj.mask

        if hasattr(obj, "grid"):
            self.grid = obj.grid

    @classmethod
    def manual_slim(
        cls,
        vectors: Union[np.ndarray, List[List], List[Tuple]],
        shape_native: Tuple[int, int],
        pixel_scales: ty.PixelScales,
        sub_size: int = 1,
        origin: Tuple[float, float] = (0.0, 0.0),
    ) -> "VectorYX2D":
        """
        Create a VectorYX2D (see *VectorYX2D.__new__*) by inputting the vector in 1D, for example:

        vectors=np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
        vectors=[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]]

        The `VectorYX2D` object assumes a uniform `Grid2D` which is computed from the input `shape_native`,
        `pixel_scales` and `origin`.

        From 1D input the method cannot determine the 2D shape of the grid and its mask, thus the `shape_native` must be
        input into this method. The mask is setup as a unmasked `Mask2D` of shape_native.

        Parameters
        ----------
        vectors
            The (y,x) vectors input as an ndarray of shape [total_unmasked_pixells*(sub_size**2), 2] or a list of lists.
        shape_native
            The 2D shape of the mask the grid is paired with.
        pixel_scales
            The (y,x) scaled units to pixel units conversion factors of every pixel. If this is input as a ``float``,
            it is converted to a (float, float) structure.
        sub_size
            The size (sub_size x sub_size) of each unmasked pixels sub-grid.
        origin
            The origin of the grid's mask.
        """

        pixel_scales = geometry_util.convert_pixel_scales_2d(pixel_scales=pixel_scales)

        grid = Grid2D.uniform(
            shape_native=shape_native,
            pixel_scales=pixel_scales,
            sub_size=sub_size,
            origin=origin,
        )

        mask = Mask2D.unmasked(
            shape_native=shape_native,
            pixel_scales=pixel_scales,
            sub_size=sub_size,
            origin=origin,
        )

        vectors = grid_2d_util.convert_grid_2d(grid_2d=vectors, mask_2d=mask)

        return cls(vectors=vectors, grid=grid, mask=mask)

    @classmethod
    def manual_native(
        cls,
        vectors: Union[np.ndarray, List],
        pixel_scales: ty.PixelScales,
        sub_size: int = 1,
        origin: Tuple[float, float] = (0.0, 0.0),
    ) -> "VectorYX2D":
        """
        Create a VectorYX2D (see *VectorYX2D.__new__*) by inputting the grid coordinates in 2D, for example:

        vectors=np.ndarray([[[1.0, 1.0], [2.0, 2.0]],
                         [[3.0, 3.0], [4.0, 4.0]]])

        vectors=[[[1.0, 1.0], [2.0, 2.0]],
                [[3.0, 3.0], [4.0, 4.0]]]

        The `VectorYX2D` object assumes a uniform `Grid2D` which is computed from the mask's `shape_native`,
        `pixel_scales` and `origin`.

        The 2D shape of the grid and its mask are determined from the input grid and the mask is setup as an
        unmasked `Mask2D` of shape_native.

        Parameters
        ----------
        grid
            The (y,x) coordinates of the grid input as an ndarray of shape
            [total_y_coordinates*sub_size, total_x_pixel*sub_size, 2] or a list of lists.
        pixel_scales
            The (y,x) scaled units to pixel units conversion factors of every pixel. If this is input as a ``float``,
            it is converted to a (float, float) structure.
        sub_size
            The size (sub_size x sub_size) of each unmasked pixels sub-grid.
        origin
            The origin of the grid's mask.
        """

        vectors = grid_2d_util.convert_grid(grid=vectors)

        pixel_scales = geometry_util.convert_pixel_scales_2d(pixel_scales=pixel_scales)

        shape_native = (
            int(vectors.shape[0] / sub_size),
            int(vectors.shape[1] / sub_size),
        )

        mask = Mask2D.unmasked(
            shape_native=shape_native,
            pixel_scales=pixel_scales,
            sub_size=sub_size,
            origin=origin,
        )

        grid = Grid2D.uniform(
            shape_native=shape_native,
            pixel_scales=pixel_scales,
            sub_size=sub_size,
            origin=origin,
        )

        vectors = grid_2d_util.convert_grid_2d(grid_2d=vectors, mask_2d=mask)

        return VectorYX2D(vectors=vectors, grid=grid, mask=mask)

    @classmethod
    def manual_mask(
        cls, vectors: Union[np.ndarray, List], mask: Mask2D
    ) -> "VectorYX2D":
        """
        Create a VectorYX2D (see *VectorYX2D.__new__*) by inputting the vectors in 1D or 2D with its mask,
        for example:

        mask = Mask2D([[True, False, False, False])
        array=np.array([1.0, 2.0, 3.0])

        Parameters
        ----------
        array
            The values of the array input as an ndarray of shape [total_unmasked_pixels*(sub_size**2)] or a list of
            lists.
        mask
            The mask whose masked pixels are used to setup the sub-pixel grid.
        """

        grid = Grid2D.from_mask(mask=mask)

        vectors = grid_2d_util.convert_grid_2d(grid_2d=vectors, mask_2d=mask)
        return VectorYX2D(vectors=vectors, grid=grid, mask=mask)

    @classmethod
    def full(
        cls,
        fill_value: float,
        shape_native: Tuple[int, int],
        pixel_scales: ty.PixelScales,
        sub_size: int = 1,
        origin: Tuple[float, float] = (0.0, 0.0),
    ) -> "VectorYX2D":
        """
        Create a `VectorYX2D` (see `AbstractVectorYX2D.__new__`) where all values are filled with an input fill value,
        analogous to the method np.full().

        From 1D input the method cannot determine the 2D shape of the array and its mask, thus the shape_native must be
        input into this method. The mask is setup as a unmasked `Mask2D` of shape_native.

        Parameters
        ----------
        fill_value
            The value all array elements are filled with.
        pixel_scales
            The (y,x) scaled units to pixel units conversion factors of every pixel. If this is input as a ``float``,
            it is converted to a (float, float) structure.
        sub_size
            The size (sub_size x sub_size) of each unmasked pixels sub-grid.
        origin
            The origin of the grid's mask.
        origin
            The (y,x) scaled units origin of the mask's coordinate system.
        """
        if sub_size is not None:
            shape_native = (shape_native[0] * sub_size, shape_native[1] * sub_size)

        return cls.manual_native(
            vectors=np.full(
                fill_value=fill_value, shape=(shape_native[0], shape_native[1], 2)
            ),
            pixel_scales=pixel_scales,
            sub_size=sub_size,
            origin=origin,
        )

    @classmethod
    def ones(
        cls,
        shape_native: Tuple[int, int],
        pixel_scales: ty.PixelScales,
        sub_size: int = 1,
        origin: Tuple[float, float] = (0.0, 0.0),
    ) -> "VectorYX2D":
        """
        Create a `VectorYX2D` (see `AbstractVectorYX2D.__new__`) where all values are filled with 1.0, analogous to
        the method np.ones().

        From 1D input the method cannot determine the 2D shape of the array and its mask, thus the shape_native must be
        input into this method. The mask is setup as a unmasked `Mask2D` of shape_native.

        Parameters
        ----------
        fill_value
            The value all array elements are filled with.
        pixel_scales
            The (y,x) scaled units to pixel units conversion factors of every pixel. If this is input as a ``float``,
            it is converted to a (float, float) structure.
        sub_size
            The size (sub_size x sub_size) of each unmasked pixels sub-grid.
        origin
            The (y,x) scaled units origin of the mask's coordinate system.
        """
        return cls.full(
            fill_value=1.0,
            shape_native=shape_native,
            pixel_scales=pixel_scales,
            sub_size=sub_size,
            origin=origin,
        )

    @classmethod
    def zeros(
        cls,
        shape_native: Tuple[int, int],
        pixel_scales: ty.PixelScales,
        sub_size: int = 1,
        origin: Tuple[float, float] = (0.0, 0.0),
    ) -> "VectorYX2D":
        """
        Create a `VectorYX2D` (see `AbstractVectorYX2D.__new__`) where all values are filled with 1.0, analogous to
        the method np.zeros().

        From 1D input the method cannot determine the 2D shape of the array and its mask, thus the shape_native must be
        input into this method. The mask is setup as a unmasked `Mask2D` of shape_native.

        Parameters
        ----------
        fill_value
            The value all array elements are filled with.
        pixel_scales
            The (y,x) scaled units to pixel units conversion factors of every pixel. If this is input as a ``float``,
            it is converted to a (float, float) structure.
        sub_size
            The size (sub_size x sub_size) of each unmasked pixels sub-grid.
        origin
            The (y,x) scaled units origin of the mask's coordinate system.
        """
        return cls.full(
            fill_value=0.0,
            shape_native=shape_native,
            pixel_scales=pixel_scales,
            sub_size=sub_size,
            origin=origin,
        )

    @property
    def slim(self) -> "VectorYX2D":
        """
        Return a `VectorYX2D` where the data is stored its `slim` representation, which is an ndarray of shape
        [total_unmasked_pixels * sub_size**2, 2].

        If it is already stored in its `slim` representation it is returned as it is. If not, it is  mapped from
        `native` to `slim` and returned as a new `Array2D`.
        """
        if len(self.shape) == 2:
            return self

        vectors_2d_slim = grid_2d_util.grid_2d_slim_from(
            grid_2d_native=self, mask=self.mask, sub_size=self.mask.sub_size
        )

        return VectorYX2D(vectors=vectors_2d_slim, grid=self.grid.slim, mask=self.mask)

    @property
    def native(self) -> "VectorYX2D":
        """
        Return a `VectorYX2D` where the data is stored in its `native` representation, which is an ndarray of shape
        [sub_size*total_y_pixels, sub_size*total_x_pixels, 2].

        If it is already stored in its `native` representation it is return as it is. If not, it is mapped from
        `slim` to `native` and returned as a new `Grid2D`.
        """

        if len(self.shape) != 2:
            return self

        vectors_2d_native = grid_2d_util.grid_2d_native_from(
            grid_2d_slim=self, mask_2d=self.mask, sub_size=self.mask.sub_size
        )

        return VectorYX2D(
            vectors=vectors_2d_native, grid=self.grid.native, mask=self.mask
        )

    @property
    def binned(self) -> "VectorYX2D":
        """
        Convenience method to access the binned-up vectors as a Vector2D stored in its `slim` or `native` format.

        The binning up process converts a grid from (y,x) values where each value is a coordinate on the sub-grid to
        (y,x) values where each coordinate is at the centre of its mask (e.g. a grid with a sub_size of 1). This is
        performed by taking the mean of all (y,x) values in each sub pixel.

        If the grid is stored in 1D it is return as is. If it is stored in 2D, it must first be mapped from 2D to 1D.
        """

        vector_2d_slim_binned_y = np.multiply(
            self.mask.sub_fraction,
            self.slim[:, 0].reshape(-1, self.mask.sub_length).sum(axis=1),
        )

        vector_2d_slim_binned_x = np.multiply(
            self.mask.sub_fraction,
            self.slim[:, 1].reshape(-1, self.mask.sub_length).sum(axis=1),
        )

        return VectorYX2D(
            vectors=np.stack(
                (vector_2d_slim_binned_y, vector_2d_slim_binned_x), axis=-1
            ),
            grid=self.grid.binned,
            mask=self.mask.derive_mask.sub_1,
        )

    def apply_mask(self, mask: Mask2D) -> "VectorYX2D":
        return VectorYX2D.manual_mask(vectors=self.native, mask=mask)

    @property
    def magnitudes(self) -> Array2D:
        """
        Returns the magnitude of every vector which are computed as sqrt(y**2 + x**2).
        """
        return Array2D(
            array=np.sqrt(self[:, 0] ** 2.0 + self[:, 1] ** 2.0), mask=self.mask
        )

    @property
    def y(self) -> Array2D:
        """
        Returns the y vector values as an `Array2D` object.
        """
        return Array2D(array=self.slim[:, 0], mask=self.mask)

    @property
    def x(self) -> Array2D:
        """
        Returns the y vector values as an `Array2D` object.
        """
        return Array2D(array=self.slim[:, 1], mask=self.mask)
