import logging
import numpy as np
from typing import List, Tuple, Union

from autoarray.structures.arrays.two_d.array_2d import Array2D
from autoarray.structures.grids.two_d.grid_2d import Grid2D
from autoarray.structures.vector_fields.abstract import AbstractVectorField2D
from autoarray.structures.arrays.values import ValuesIrregular

from autoarray import exc
from autoarray.structures.arrays.two_d import array_2d_util
from autoarray.structures.grids.two_d import grid_2d_util

logging.basicConfig()
logger = logging.getLogger(__name__)


class VectorField2D(AbstractVectorField2D):
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

        The `VectorField2D` is ordered such that pixels begin from the top-row of the corresponding mask and go right 
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

        Case 1: [sub-size=1, slim]:
        ---------------------------

        The Vector2D is an ndarray of shape [total_unmasked_pixels, 2].

        The first element of the ndarray corresponds to the pixel index, for example:

        - vector[3, 0:2] = the 4th unmasked pixel's y and x values.
        - vector[6, 0:2] = the 7th unmasked pixel's y and x values.

        Below is a visual illustration of a vector, where a total of 10 pixels are unmasked and are included in
        the vector.

        IxIxIxIxIxIxIxIxIxIxI
        IxIxIxIxIxIxIxIxIxIxI     This is an example `Mask2D`, where:
        IxIxIxIxIxIxIxIxIxIxI
        IxIxIxIxIoIoIxIxIxIxI     x = `True` (Pixel is masked and excluded from the vector)
        IxIxIxIoIoIoIoIxIxIxI     o = `False` (Pixel is not masked and included in the vector)
        IxIxIxIoIoIoIoIxIxIxI
        IxIxIxIxIxIxIxIxIxIxI
        IxIxIxIxIxIxIxIxIxIxI
        IxIxIxIxIxIxIxIxIxIxI
        IxIxIxIxIxIxIxIxIxIxI

        The mask pixel index's will come out like this (and the direction of scaled values is highlighted
        around the mask).

        pixel_scales = 1.0"

        <--- -ve  x  +ve -->
                                                        y      x
        IxIxIxIxIxIxIxIxIxIxI  ^   vector[0, :] = 0
        IxIxIxIxIxIxIxIxIxIxI  I   vector[1, :] = 1
        IxIxIxIxIxIxIxIxIxIxI  I   vector[2, :] = 2
        IxIxIxIxI0I1IxIxIxIxI +ve  vector[3, :] = 3
        IxIxIxI2I3I4I5IxIxIxI  y   vector[4, :] = 4
        IxIxIxI6I7I8I9IxIxIxI -ve  vector[5, :] = 5
        IxIxIxIxIxIxIxIxIxIxI  I   vector[6, :] = 6
        IxIxIxIxIxIxIxIxIxIxI  I   vector[7, :] = 7
        IxIxIxIxIxIxIxIxIxIxI \/   vector[8, :] = 8
        IxIxIxIxIxIxIxIxIxIxI      vector[9, :] = 9

        Case 2: [sub-size>1, slim]:
        ------------------

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

        IxIxIxIxIxIxIxIxIxIxI
        IxIxIxIxIxIxIxIxIxIxI     This is an example `Mask2D`, where:
        IxIxIxIxIxIxIxIxIxIxI
        IxIxIxIxIxIxIxIxIxIxI     x = `True` (Pixel is masked and excluded from lens)
        IxIxIxIxIoIoIxIxIxIxI     o = `False` (Pixel is not masked and included in lens)
        IxIxIxIxIxIxIxIxIxIxI
        IxIxIxIxIxIxIxIxIxIxI
        IxIxIxIxIxIxIxIxIxIxI
        IxIxIxIxIxIxIxIxIxIxI
        IxIxIxIxIxIxIxIxIxIxI

        Our vector with a sub-size looks like it did before:

        pixel_scales = 1.0"

        <--- -ve  x  +ve -->

        IxIxIxIxIxIxIxIxIxIxI  ^
        IxIxIxIxIxIxIxIxIxIxI  I
        IxIxIxIxIxIxIxIxIxIxI  I
        IxIxIxIxIxIxIxIxIxIxI +ve
        IxIxIxI0I1IxIxIxIxIxI  y
        IxIxIxIxIxIxIxIxIxIxI -ve
        IxIxIxIxIxIxIxIxIxIxI  I
        IxIxIxIxIxIxIxIxIxIxI  I
        IxIxIxIxIxIxIxIxIxIxI \/
        IxIxIxIxIxIxIxIxIxIxI

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

        Case 3: [sub_size=1, native]
        ----------------------------

        The Vector2D has the same properties as Case 1, but is stored as an an ndarray of shape
        [total_y_values, total_x_values, 2].

        All masked entries on the vector have values of 0.0.

        For the following example mask:

        IxIxIxIxIxIxIxIxIxIxI
        IxIxIxIxIxIxIxIxIxIxI     This is an example `Mask2D`, where:
        IxIxIxIxIxIxIxIxIxIxI
        IxIxIxIxIoIoIxIxIxIxI     x = `True` (Pixel is masked and excluded from the vector)
        IxIxIxIoIoIoIoIxIxIxI     o = `False` (Pixel is not masked and included in the vector)
        IxIxIxIoIoIoIoIxIxIxI
        IxIxIxIxIxIxIxIxIxIxI
        IxIxIxIxIxIxIxIxIxIxI
        IxIxIxIxIxIxIxIxIxIxI
        IxIxIxIxIxIxIxIxIxIxI

        - vector[0,0, 0:2] = [0.0, 0.0] (it is masked, thus zero)
        - vector[0,0, 0:2] = [0.0, 0.0] (it is masked, thus zero)
        - vector[3,3, 0:2] = [0.0, 0.0] (it is masked, thus zero)
        - vector[3,3, 0:2] = [0.0, 0.0] (it is masked, thus zero)
        - vector[3,4, 0:2] = [0, 0]
        - vector[3,4, 0:2] = [-1, -1]

        Case 4: [sub_size>, native]
        ---------------------------

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

        return obj

    def __array_finalize__(self, obj):

        if hasattr(obj, "mask"):
            self.mask = obj.mask

        if hasattr(obj, "grid"):
            self.grid = obj.grid

    @property
    def magnitudes(self) -> Array2D:
        """
        Returns the magnitude of every vector which are computed as sqrt(y**2 + x**2).
        """
        return Array2D(
            array=np.sqrt(self[:, 0] ** 2.0 + self[:, 1] ** 2.0), mask=self.mask
        )
