import logging

import numpy as np

from autoarray.structures import grids
from autoarray.util import grid_util

logging.basicConfig()
logger = logging.getLogger(__name__)


class VectorFieldIrregular(np.ndarray):
    def __new__(
        cls, vectors: np.ndarray or [(float, float)], grid: grids.GridIrregular or list
    ):
        """
        A collection of (y,x) vectors which are located on an irregular grid of (y,x) coordinates.

        The `VectorFieldIrregular` stores the (y,x) vector vectors as a 2D NumPy array of shape [total_vectors, 2].
        Index information is stored so that this array can be mapped to a list of tuples structure.

        Calculations should use the NumPy array structure wherever possible for efficient calculations.

        The vectors input to this function can have any of the following forms (they will be converted to the 1D NumPy
        array structure and can be converted back using the object's properties):

        [[vector_0_y, vector_0_x], [vector_1_y, vector_1_x]]
        [(vector_0_y, vector_0_x), (vector_1_y, vector_1_x)]

        If your vector field lies on a 2D uniform grid of data the `VectorField` data structure should be used.

        Parameters
        ----------
        vectors : np.ndarray or [(float, float)]
            The 2D (y,x) vectors on an irregular grid that represent the vector-field.
        grid : grids.GridIrregular
            The irregular grid of (y,x) coordinates where each vector is located.
        """

        if len(vectors) == 0:
            return []

        if type(vectors) is list:
            vectors = np.asarray(vectors)

        obj = vectors.view(cls)
        obj.grid = grids.GridIrregular(grid=grid)

        return obj

    def __array_finalize__(self, obj):

        if hasattr(obj, "grid"):
            self.grid = obj.grid

    @property
    def in_1d(self):
        """The VectorFieldIrregular in its 1D representation, an ndarray of shape [total_vectors, 2]."""
        return self

    @property
    def in_1d_list(self):
        """Return the (y,x) vecotrs in list of structure [(vector_0_y, vector_0_x), ...]."""
        return [tuple(vector) for vector in self.in_1d]

    @property
    def average(self):
        return(np.mean(self))

    @property
    def average_magnitude(self):
        return np.sqrt(self[:, 0].average ** 2 + self[:, 1].average ** 2)

    @property
    def average_angle(self):
        return 0.5*np.arctan2(self[:, 0].average, self[:, 1].average)*(180/np.pi)

    def vectors_from_grid_within_radius(self, radius, centre):

        mask = grid_util.mask_of_points_within_radius(grid=self.grid, radius=radius, centre=centre)

        grid_inside = []
        vector_inside = []

        for i in range(len(mask)):
            if mask[i] == True:
                grid_inside.append(self.grid[i])
                vector_inside.append(self[i])

        grid_inside = grids.GridIrregular(grid=np.asarray(grid_inside))

        return VectorFieldIrregular(grid=grid_inside, vectors=np.asarray(vector_inside))
