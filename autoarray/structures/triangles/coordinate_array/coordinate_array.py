import numpy as np

from autoarray.structures.triangles.abstract import HEIGHT_FACTOR
from autoarray.structures.triangles.coordinate_array.abstract_coordinate_array import (
    AbstractCoordinateArray,
)
from autoarray.structures.triangles.array import ArrayTriangles
from autoarray.structures.triangles.shape import Shape
from autoconf import cached_property


class CoordinateArrayTriangles(AbstractCoordinateArray):
    @cached_property
    def flip_array(self) -> np.ndarray:
        """
        An array of 1s and -1s to flip the triangles.
        """
        array = np.ones(
            self.coordinates.shape[0],
            dtype=np.int32,
        )
        array[self.flip_mask] = -1

        return array[:, np.newaxis]

    def up_sample(self) -> "CoordinateArrayTriangles":
        """
        Up-sample the triangles by adding a new vertex at the midpoint of each edge.
        """
        new_coordinates = np.zeros(
            (4 * self.coordinates.shape[0], 2),
            dtype=np.int32,
        )
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
            y_offset=self.y_offset + -0.25 * HEIGHT_FACTOR * self.side_length,
            x_offset=self.x_offset,
            flipped=True,
        )

    def neighborhood(self) -> "CoordinateArrayTriangles":
        """
        Create a new set of triangles that are the neighborhood of the current triangles.

        Ensures that the new triangles are unique.
        """
        new_coordinates = np.zeros(
            (4 * self.coordinates.shape[0], 2),
            dtype=np.int32,
        )
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
            y_offset=self.y_offset,
            x_offset=self.x_offset,
            flipped=self.flipped,
        )

    @cached_property
    def _vertices_and_indices(self):
        flat_triangles = self.triangles.reshape(-1, 2)
        vertices, inverse_indices = np.unique(
            flat_triangles,
            axis=0,
            return_inverse=True,
        )
        indices = inverse_indices.reshape(-1, 3)
        return vertices, indices

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
            y_offset=self.y_offset,
            x_offset=self.x_offset,
            flipped=self.flipped,
        )

    def containing_indices(self, shape: Shape) -> np.ndarray:
        """
        Find the triangles that insect with a given shape.

        Parameters
        ----------
        shape
            The shape

        Returns
        -------
        The indices of triangles that intersect the shape.
        """
        return self.with_vertices(self.vertices).containing_indices(shape)
