from jax import numpy as np

from autoarray.structures.triangles.abstract import HEIGHT_FACTOR
from autoarray.structures.triangles.abstract_coordinate_array import (
    AbstractCoordinateArray,
)
from autoarray.structures.triangles.jax_array import ArrayTriangles
from autoarray.numpy_wrapper import register_pytree_node_class
from autoconf import cached_property


@register_pytree_node_class
class CoordinateArrayTriangles(AbstractCoordinateArray):
    def tree_flatten(self):
        return (
            self.coordinates,
            self.side_length,
            self.x_offset,
            self.y_offset,
        ), (self.flipped,)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """
        Create a prior from a flattened PyTree

        Parameters
        ----------
        aux_data
            Auxiliary information that remains unchanged including
            the keys of the dict
        children
            Child objects subject to change

        Returns
        -------
        An instance of this class
        """
        return cls(*children, flipped=aux_data[0])

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
    def flip_array(self) -> np.ndarray:
        """
        An array of 1s and -1s to flip the triangles.
        """
        array = np.where(self.flip_mask, -1.0, 1.0)
        return array[:, None]

    def __iter__(self):
        return iter(self.triangles)

    def up_sample(self) -> "CoordinateArrayTriangles":
        """
        Up-sample the triangles by adding a new vertex at the midpoint of each edge.
        """
        coordinates = self.coordinates
        flip_mask = self.flip_mask

        coordinates = 2 * coordinates

        n = coordinates.shape[0]

        shift0 = np.zeros((n, 2))
        shift3 = np.tile(np.array([0, 1]), (n, 1))
        shift1 = np.stack([np.ones(n), np.where(flip_mask, 1.0, 0.0)], axis=1)
        shift2 = np.stack([-np.ones(n), np.where(flip_mask, 1.0, 0.0)], axis=1)
        shifts = np.stack([shift0, shift1, shift2, shift3], axis=1)

        coordinates_expanded = coordinates[:, None, :]
        new_coordinates = coordinates_expanded + shifts
        new_coordinates = new_coordinates.reshape(-1, 2)

        return CoordinateArrayTriangles(
            coordinates=new_coordinates,
            side_length=self.side_length / 2,
            flipped=True,
            y_offset=self.y_offset + -0.25 * HEIGHT_FACTOR * self.side_length,
            x_offset=self.x_offset,
        )

    def neighborhood(self) -> "CoordinateArrayTriangles":
        """
        Create a new set of triangles that are the neighborhood of the current triangles.

        Ensures that the new triangles are unique.
        """
        coordinates = self.coordinates
        flip_mask = self.flip_mask

        shift0 = np.zeros((coordinates.shape[0], 2))
        shift1 = np.tile(np.array([1, 0]), (coordinates.shape[0], 1))
        shift2 = np.tile(np.array([-1, 0]), (coordinates.shape[0], 1))

        shift3 = np.where(
            flip_mask[:, None],
            np.tile(np.array([0, 1]), (coordinates.shape[0], 1)),
            np.tile(np.array([0, -1]), (coordinates.shape[0], 1)),
        )

        shifts = np.stack([shift0, shift1, shift2, shift3], axis=1)

        coordinates_expanded = coordinates[:, None, :]
        new_coordinates = coordinates_expanded + shifts

        new_coordinates = new_coordinates.reshape(-1, 2)

        return CoordinateArrayTriangles(
            coordinates=np.unique(
                new_coordinates,
                axis=0,
                size=4 * self.coordinates.shape[0],
                fill_value=np.nan,
            ),
            side_length=self.side_length,
            flipped=self.flipped,
            y_offset=self.y_offset,
            x_offset=self.x_offset,
        )

    @cached_property
    def _vertices_and_indices(self):
        flat_triangles = self.triangles.reshape(-1, 2)
        vertices, inverse_indices = np.unique(
            flat_triangles,
            axis=0,
            return_inverse=True,
            size=3 * self.coordinates.shape[0],
            equal_nan=True,
            fill_value=np.nan,
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
        mask = indexes == -1
        safe_indexes = np.where(mask, 0, indexes)
        coordinates = np.take(self.coordinates, safe_indexes, axis=0)
        coordinates = np.where(mask[:, None], np.nan, coordinates)

        return CoordinateArrayTriangles(
            coordinates=coordinates,
            side_length=self.side_length,
            y_offset=self.y_offset,
            x_offset=self.x_offset,
            flipped=self.flipped,
        )
