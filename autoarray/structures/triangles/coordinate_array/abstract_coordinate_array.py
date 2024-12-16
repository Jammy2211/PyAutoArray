from abc import abstractmethod, ABC

import numpy as np

from autoarray.structures.triangles.abstract import HEIGHT_FACTOR, AbstractTriangles
from autoconf import cached_property


class AbstractCoordinateArray(AbstractTriangles, ABC):
    def __init__(
        self,
        coordinates: np.ndarray,
        side_length: float = 1.0,
        x_offset: float = 0.0,
        y_offset: float = 0.0,
        flipped: bool = False,
    ):
        """
        Represents a set of triangles by integer coordinates.

        Parameters
        ----------
        coordinates
            Integer x y coordinates for each triangle.
        side_length
            The side length of the triangles.
        flipped
            Whether the triangles are flipped upside down.
        y_offset
            An y_offset to apply to the y coordinates so that up-sampled triangles align.
        """
        self.coordinates = coordinates
        self.side_length = side_length
        self.flipped = flipped

        self.scaling_factors = self.numpy.array(
            [0.5 * side_length, HEIGHT_FACTOR * side_length]
        )
        self.x_offset = x_offset
        self.y_offset = y_offset

    @property
    @abstractmethod
    def numpy(self):
        pass

    @cached_property
    def vertex_coordinates(self) -> np.ndarray:
        """
        The vertices of the triangles as an Nx3x2 array.
        """
        coordinates = self.coordinates
        return self.numpy.concatenate(
            [
                coordinates + self.flip_array * np.array([0, 1], dtype=np.int32),
                coordinates + self.flip_array * np.array([1, -1], dtype=np.int32),
                coordinates + self.flip_array * np.array([-1, -1], dtype=np.int32),
            ],
            dtype=np.int32,
        )

    @cached_property
    def triangles(self) -> np.ndarray:
        """
        The vertices of the triangles as an Nx3x2 array.
        """
        centres = self.centres
        return self.numpy.stack(
            (
                centres
                + self.flip_array
                * self.numpy.array(
                    [0.0, 0.5 * self.side_length * HEIGHT_FACTOR],
                ),
                centres
                + self.flip_array
                * self.numpy.array(
                    [0.5 * self.side_length, -0.5 * self.side_length * HEIGHT_FACTOR]
                ),
                centres
                + self.flip_array
                * self.numpy.array(
                    [-0.5 * self.side_length, -0.5 * self.side_length * HEIGHT_FACTOR]
                ),
            ),
            axis=1,
        )

    @property
    def centres(self) -> np.ndarray:
        """
        The centres of the triangles.
        """
        return self.scaling_factors * self.coordinates + self.numpy.array(
            [self.x_offset, self.y_offset]
        )

    @cached_property
    def flip_mask(self) -> np.ndarray:
        """
        A mask for the triangles that are flipped.

        Every other triangle is flipped so that they tessellate.
        """
        mask = (self.coordinates[:, 0] + self.coordinates[:, 1]) % 2 != 0
        if self.flipped:
            mask = ~mask
        return mask

    @cached_property
    @abstractmethod
    def flip_array(self) -> np.ndarray:
        """
        An array of 1s and -1s to flip the triangles.
        """

    def __iter__(self):
        return iter(self.triangles)

    @cached_property
    @abstractmethod
    def _vertices_and_indices(self):
        pass

    @property
    def vertices(self) -> np.ndarray:
        """
        The unique vertices of the triangles.
        """
        return self._vertices_and_indices[0]

    @property
    def indices(self) -> np.ndarray:
        """
        The indices of the vertices of the triangles.
        """
        return self._vertices_and_indices[1]

    def with_vertices(self, vertices: np.ndarray) -> AbstractTriangles:
        """
        Create a new set of triangles with the vertices replaced.

        Parameters
        ----------
        vertices
            The new vertices to use.

        Returns
        -------
        The new set of triangles with the new vertices.
        """

    @property
    def means(self):
        return self.numpy.mean(self.triangles, axis=1)

    @property
    def area(self):
        return (3**0.5 / 4 * self.side_length**2) * len(self)

    def __len__(self):
        return self.numpy.count_nonzero(~self.numpy.isnan(self.coordinates).any(axis=1))
