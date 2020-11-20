import logging

import numpy as np

from matplotlib.patches import Ellipse
from autoarray.structures.arrays import values
from autoarray.structures import arrays, grids

logging.basicConfig()
logger = logging.getLogger(__name__)


class VectorFieldIrregular(np.ndarray):
    def __new__(
        cls, vectors: np.ndarray or [(float, float)], grid: grids.GridIrregular or list
    ):
        """
        A collection of (y,x) vectors which are located on an irregular grid of (y,x) coordinates.

        The `VectorFieldIrregular` stores the (y,x) vector vectors as a 2D NumPy array of shape [total_vectors, 2].
        This array can be mapped to a list of tuples structure.

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
    def ellipticities(self):
        return values.Values(values=np.sqrt(self[:, 0] ** 2 + self[:, 1] ** 2.0))

    @property
    def semi_major_axes(self):
        return values.Values(values=3 * (1 + self.ellipticities))

    @property
    def semi_minor_axes(self):
        return values.Values(values=3 * (1 - self.ellipticities))

    @property
    def phis(self):
        return values.Values(
            values=np.arctan2(self[:, 0], self[:, 1]) * 180.0 / np.pi / 2.0
        )

    @property
    def elliptical_patches(self):

        return [
            Ellipse(xy=(x, y), width=semi_major_axis, height=semi_minor_axis, angle=phi)
            for x, y, semi_major_axis, semi_minor_axis, phi in zip(
                self.grid[:, 1],
                self.grid[:, 0],
                self.semi_major_axes,
                self.semi_minor_axes,
                self.phis,
            )
        ]

    @property
    def in_1d(self):
        """The VectorFieldIrregular in its 1D representation, an ndarray of shape [total_vectors, 2]."""
        return self

    @property
    def in_1d_list(self):
        """Return the (y,x) vecotrs in list of structure [(vector_0_y, vector_0_x), ...]."""
        return [tuple(vector) for vector in self.in_1d]
