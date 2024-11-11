from abc import ABC, abstractmethod

import numpy as np

from autoarray.structures.triangles.abstract import HEIGHT_FACTOR, AbstractTriangles
from autoconf import cached_property


class AbstractCoordinateArray(ABC):
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

        self.scaling_factors = np.array(
            [0.5 * side_length, HEIGHT_FACTOR * side_length]
        )
        self.x_offset = x_offset
        self.y_offset = y_offset

    @cached_property
    def triangles(self) -> np.ndarray:
        """
        The vertices of the triangles as an Nx3x2 array.
        """
        centres = self.centres
        return np.stack(
            (
                centres
                + self.flip_array
                * np.array(
                    [0.0, 0.5 * self.side_length * HEIGHT_FACTOR],
                ),
                centres
                + self.flip_array
                * np.array(
                    [0.5 * self.side_length, -0.5 * self.side_length * HEIGHT_FACTOR]
                ),
                centres
                + self.flip_array
                * np.array(
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
        return self.scaling_factors * self.coordinates + np.array(
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

    @classmethod
    def for_limits_and_scale(
        cls,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        scale: float = 1.0,
        **_,
    ):
        coordinates = []
        x = x_min
        while x < x_max:
            y = y_min
            while y < y_max:
                coordinates.append([x, y])
                y += scale
            x += scale

        x_mean = (x_min + x_max) / 2
        y_mean = (y_min + y_max) / 2

        return cls(
            coordinates=np.array(coordinates),
            side_length=scale,
            x_offset=x_mean,
            y_offset=y_mean,
        )

    @property
    def means(self):
        return np.mean(self.triangles, axis=1)

    @property
    def area(self):
        return (3**0.5 / 4 * self.side_length**2) * len(self)

    def __len__(self):
        return np.count_nonzero(~np.isnan(self.coordinates).any(axis=1))
