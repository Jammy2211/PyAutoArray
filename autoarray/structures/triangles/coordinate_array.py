import numpy as np

from autoarray.structures.triangles.abstract import HEIGHT_FACTOR
from autoarray.structures.triangles.array import ArrayTriangles
from autoconf import cached_property


class CoordinateArrayTriangles:
    def __init__(
        self,
        coordinates: np.ndarray,
        side_length: float,
        flipped: bool = False,
        x_offset: float = 0.0,
        y_offset: float = 0.0,
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
        return self.scaling_factors * self.coordinates + np.array([0.0, self.y_offset])

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
    def flip_array(self) -> np.ndarray:
        """
        An array of 1s and -1s to flip the triangles.
        """
        array = np.ones(self.coordinates.shape[0])
        array[self.flip_mask] = -1

        return array[:, np.newaxis]

    def __iter__(self):
        return iter(self.triangles)

    def up_sample(self) -> "CoordinateArrayTriangles":
        """
        Up-sample the triangles by adding a new vertex at the midpoint of each edge.
        """
        new_coordinates = np.zeros((4 * self.coordinates.shape[0], 2))
        n_normal = 4 * np.sum(~self.flip_mask)

        new_coordinates[:n_normal] = np.vstack(
            (
                2 * self.coordinates[~self.flip_mask],
                2 * self.coordinates[~self.flip_mask] + np.array([1, 0]),
                2 * self.coordinates[~self.flip_mask] + np.array([-1, 0]),
                2 * self.coordinates[~self.flip_mask] + np.array([0, 1]),
            )
        )
        new_coordinates[n_normal:] = np.vstack(
            (
                2 * self.coordinates[self.flip_mask],
                2 * self.coordinates[self.flip_mask] + np.array([1, 1]),
                2 * self.coordinates[self.flip_mask] + np.array([-1, 1]),
                2 * self.coordinates[self.flip_mask] + np.array([0, 1]),
            )
        )

        return CoordinateArrayTriangles(
            coordinates=new_coordinates,
            side_length=self.side_length / 2,
            flipped=True,
            y_offset=self.y_offset + -0.25 * HEIGHT_FACTOR * self.side_length,
        )

    def neighborhood(self) -> "CoordinateArrayTriangles":
        """
        Create a new set of triangles that are the neighborhood of the current triangles.

        Ensures that the new triangles are unique.
        """
        new_coordinates = np.zeros((4 * self.coordinates.shape[0], 2))
        n_normal = 4 * np.sum(~self.flip_mask)

        new_coordinates[:n_normal] = np.vstack(
            (
                self.coordinates[~self.flip_mask],
                self.coordinates[~self.flip_mask] + np.array([1, 0]),
                self.coordinates[~self.flip_mask] + np.array([-1, 0]),
                self.coordinates[~self.flip_mask] + np.array([0, -1]),
            )
        )
        new_coordinates[n_normal:] = np.vstack(
            (
                self.coordinates[self.flip_mask],
                self.coordinates[self.flip_mask] + np.array([1, 0]),
                self.coordinates[self.flip_mask] + np.array([-1, 0]),
                self.coordinates[self.flip_mask] + np.array([0, 1]),
            )
        )
        return CoordinateArrayTriangles(
            coordinates=np.unique(new_coordinates, axis=0),
            side_length=self.side_length,
            flipped=self.flipped,
            y_offset=self.y_offset,
        )

    @property
    def vertices(self) -> np.ndarray:
        """
        The unique vertices of the triangles.
        """
        return np.unique(self.triangles.reshape((-1, 2)), axis=0)

    @property
    def indices(self) -> np.ndarray:
        """
        The indices of the vertices of the triangles.
        """
        flat_triangles = self.triangles.reshape(-1, 2)
        vertices, inverse_indices = np.unique(
            flat_triangles, axis=0, return_inverse=True
        )
        indices = inverse_indices.reshape(-1, 3)
        return indices

    def with_vertices(self, vertices: np.ndarray) -> ArrayTriangles:
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
        return ArrayTriangles(
            indices=self.indices,
            vertices=vertices,
        )

    def for_indexes(self, indexes: np.ndarray) -> "CoordinateArrayTriangles":
        """
        Create a new CoordinateArrayTriangles containing triangles corresponding to the given indexes

        Parameters
        ----------
        indexes
            The indexes of the triangles to include in the new CoordinateArrayTriangles.

        Returns
        -------
        The new CoordinateArrayTriangles instance.
        """
        return CoordinateArrayTriangles(
            coordinates=self.coordinates[indexes],
            side_length=self.side_length,
            flipped=self.flipped,
            y_offset=self.y_offset,
        )

    @classmethod
    def for_limits_and_scale(
        cls,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        **_,
    ):
        x_mean = (x_min + x_max) / 2
        y_mean = (y_min + y_max) / 2

        max_side_length = max(x_max - x_min, y_max - y_min)

        return cls(
            coordinates=np.array(
                [
                    [-1, 0],
                    [0, 0],
                    [1, 0],
                ]
            ),
            side_length=max_side_length / HEIGHT_FACTOR,
            x_offset=x_mean,
            y_offset=y_mean,
        )
