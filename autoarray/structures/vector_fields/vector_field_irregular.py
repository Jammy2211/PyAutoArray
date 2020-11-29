import logging

import numpy as np
import typing

from autoarray.util import grid_util, mask_util

from matplotlib.patches import Ellipse
from autoarray.structures.arrays import values
from autoarray.structures import arrays, grids

from autoarray import exc

logging.basicConfig()
logger = logging.getLogger(__name__)


class VectorFieldIrregular(np.ndarray):
    def __new__(
        cls, vectors: np.ndarray or [(float, float)], grid: grids.GridIrregular or list
    ):
        """
        A collection of (y,x) vectors which are located on an irregular grid of (y,x) coordinates.

        The (y,x) vectors are stored as a 2D NumPy array of shape [total_vectors, 2]. This array can be mapped to a
        list of tuples structure.

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
    def ellipticities(self) -> values.Values:
        """
        If we treat this vector field as a set of weak lensing shear measurements, the galaxy ellipticity each vector
        corresponds too.
        """
        return values.Values(values=np.sqrt(self[:, 0] ** 2 + self[:, 1] ** 2.0))

    @property
    def semi_major_axes(self) -> values.Values:
        """
        If we treat this vector field as a set of weak lensing shear measurements, the semi-major axis of each
        galaxy ellipticity that each vector corresponds too.
        """
        return values.Values(values=3 * (1 + self.ellipticities))

    @property
    def semi_minor_axes(self) -> values.Values:
        """
        If we treat this vector field as a set of weak lensing shear measurements, the semi-minor axis of each
        galaxy ellipticity that each vector corresponds too.
        """
        return values.Values(values=3 * (1 - self.ellipticities))

    @property
    def phis(self) -> values.Values:
        """
        If we treat this vector field as a set of weak lensing shear measurements, the position angle phi defined
        counter clockwise from the positive x-axis of each galaxy ellipticity that each vector corresponds too.
        """
        return values.Values(
            values=np.arctan2(self[:, 0], self[:, 1]) * 180.0 / np.pi / 2.0
        )

    @property
    def elliptical_patches(self) -> typing.List[Ellipse]:
        """
        If we treat this vector field as a set of weak lensing shear measurements, the elliptical patch representing
        each galaxy ellipticity. This patch is used for visualizing an ellipse of each galaxy in an image.
        """

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
    def in_1d(self) -> np.ndarray:
        """
        The vector-field in its 1D representation, an ndarray of shape [total_vectors, 2].
        """
        return self

    @property
    def in_1d_list(self) -> typing.List[typing.Tuple]:
        """
        The vector-field in its list representation, as list of (y,x) vector tuples in a structure
        [(vector_0_y, vector_0_x), ...].
        """
        return [tuple(vector) for vector in self.in_1d]

    @property
    def average_magnitude(self) -> float:
        """
        The average magnitude of the vector field, where averaging is performed on the (vector_y, vector_x) components.
        """
        return np.sqrt(np.mean(self[:, 0]) ** 2 + np.mean(self[:, 1]) ** 2)

    @property
    def average_phi(self) -> float:
        """
        The average angle of the vector field, where averaging is performed on the (vector_y, vector_x) components.
        """
        return (
            0.5 * np.arctan2(np.mean(self[:, 0]), np.mean(self[:, 1])) * (180 / np.pi)
        )

    def vectors_within_radius(
        self, radius: float, centre: typing.Tuple[float, float] = (0.0, 0.0)
    ) -> "VectorFieldIrregular":
        """
        Returns a new `VectorFieldIrregular` object which has had all vectors outside of a circle of input radius
        around an  input (y,x) centre removed.

        Parameters
        ----------
        radius : float
            The radius of the circle outside of which vectors are removed.
        centre : float
            The centre of the circle outside of which vectors are removed.

        Returns
        -------
        VectorFieldIrregular
            The vector field where all vectors outside of the input radius are removed.

        """
        squared_distances = self.grid.distances_from_coordinate(coordinate=centre)
        mask = squared_distances < radius

        if np.all(mask == False):
            raise exc.VectorFieldException(
                "The input radius removed all vectors / points on the grid."
            )

        return VectorFieldIrregular(
            vectors=self[mask], grid=grids.GridIrregular(self.grid[mask])
        )
