from abc import ABC

import numpy as np
import jax.numpy as jnp
import jax

from autoarray.structures.triangles.abstract import HEIGHT_FACTOR
from autoarray.structures.triangles.abstract import AbstractTriangles
from autoarray.structures.triangles.array import ArrayTriangles
from autoarray.numpy_wrapper import register_pytree_node_class
from autoconf import cached_property

jax.config.update("jax_enable_x64", True)


@register_pytree_node_class
class CoordinateArrayTriangles(AbstractTriangles, ABC):

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

        self.scaling_factors = jnp.array(
            [0.5 * side_length, HEIGHT_FACTOR * side_length]
        )
        self.x_offset = x_offset
        self.y_offset = y_offset

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
        x_shift = int(2 * x_min / scale)
        y_shift = int(y_min / (HEIGHT_FACTOR * scale))

        coordinates = []

        for x in range(x_shift, int(2 * x_max / scale) + 1):
            for y in range(y_shift - 1, int(y_max / (HEIGHT_FACTOR * scale)) + 2):
                coordinates.append([x, y])

        return cls(
            coordinates=jnp.array(coordinates),
            side_length=scale,
        )

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

    def __len__(self):
        return jnp.count_nonzero(~jnp.isnan(self.coordinates).any(axis=1))

    def __iter__(self):
        return iter(self.triangles)

    @property
    def centres(self) -> jnp.ndarray:
        """
        The centres of the triangles.
        """
        centres = self.scaling_factors * self.coordinates + jnp.array(
            [self.x_offset, self.y_offset]
        )
        return centres

    @cached_property
    def vertex_coordinates(self) -> np.ndarray:
        """
        The vertices of the triangles as an Nx3x2 array.
        """
        coordinates = self.coordinates
        return jnp.concatenate(
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
        return jnp.stack(
            (
                centres
                + self.flip_array
                * jnp.array(
                    [0.0, 0.5 * self.side_length * HEIGHT_FACTOR],
                ),
                centres
                + self.flip_array
                * jnp.array(
                    [0.5 * self.side_length, -0.5 * self.side_length * HEIGHT_FACTOR]
                ),
                centres
                + self.flip_array
                * jnp.array(
                    [-0.5 * self.side_length, -0.5 * self.side_length * HEIGHT_FACTOR]
                ),
            ),
            axis=1,
        )

    @cached_property
    def flip_mask(self) -> jnp.ndarray:
        """
        A mask for the triangles that are flipped.

        Every other triangle is flipped so that they tessellate.
        """
        mask = (self.coordinates[:, 0] + self.coordinates[:, 1]) % 2 != 0
        if self.flipped:
            mask = ~mask
        return mask

    @cached_property
    def flip_array(self) -> jnp.ndarray:
        """
        An array of 1s and -1s to flip the triangles.
        """
        array = jnp.where(self.flip_mask, -1, 1)
        return array[:, None]

    def up_sample(self) -> "CoordinateArrayTriangles":
        """
        Up-sample the triangles by adding a new vertex at the midpoint of each edge.
        """
        coordinates = self.coordinates
        flip_mask = self.flip_mask

        coordinates = 2 * coordinates

        n = coordinates.shape[0]

        shift0 = jnp.zeros((n, 2))
        shift3 = jnp.tile(jnp.array([0, 1]), (n, 1))
        shift1 = jnp.stack([jnp.ones(n), jnp.where(flip_mask, 1, 0)], axis=1)
        shift2 = jnp.stack([-jnp.ones(n), jnp.where(flip_mask, 1, 0)], axis=1)
        shifts = jnp.stack([shift0, shift1, shift2, shift3], axis=1)

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

        Ensures that the new triangles are unique and adjusts the mask accordingly.
        """
        coordinates = self.coordinates
        flip_mask = self.flip_mask

        shift0 = jnp.zeros((coordinates.shape[0], 2))
        shift1 = jnp.tile(jnp.array([1, 0]), (coordinates.shape[0], 1))
        shift2 = jnp.tile(jnp.array([-1, 0]), (coordinates.shape[0], 1))
        shift3 = jnp.where(
            flip_mask[:, None],
            jnp.tile(jnp.array([0, 1]), (coordinates.shape[0], 1)),
            jnp.tile(jnp.array([0, -1]), (coordinates.shape[0], 1)),
        )

        shifts = jnp.stack([shift0, shift1, shift2, shift3], axis=1)

        coordinates_expanded = coordinates[:, None, :]
        new_coordinates = coordinates_expanded + shifts
        new_coordinates = new_coordinates.reshape(-1, 2)

        expected_size = 4 * coordinates.shape[0]
        unique_coords, indices = jnp.unique(
            new_coordinates,
            axis=0,
            size=expected_size,
            fill_value=jnp.nan,
            return_index=True,
        )

        return CoordinateArrayTriangles(
            coordinates=unique_coords,
            side_length=self.side_length,
            flipped=self.flipped,
            y_offset=self.y_offset,
            x_offset=self.x_offset,
        )

    @cached_property
    def _vertices_and_indices(self):
        flat_triangles = self.triangles.reshape(-1, 2)
        vertices, inverse_indices = jnp.unique(
            flat_triangles,
            axis=0,
            return_inverse=True,
            size=3 * self.coordinates.shape[0],
            equal_nan=True,
            fill_value=jnp.nan,
        )

        nan_mask = jnp.isnan(vertices).any(axis=1)
        inverse_indices = jnp.where(nan_mask[inverse_indices], -1, inverse_indices)

        indices = inverse_indices.reshape(-1, 3)
        return vertices, indices

    def with_vertices(self, vertices: jnp.ndarray) -> ArrayTriangles:
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

    def for_indexes(self, indexes: jnp.ndarray) -> "CoordinateArrayTriangles":
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
        safe_indexes = jnp.where(mask, 0, indexes)
        coordinates = jnp.take(self.coordinates, safe_indexes, axis=0)
        coordinates = jnp.where(mask[:, None], jnp.nan, coordinates)

        return CoordinateArrayTriangles(
            coordinates=coordinates,
            side_length=self.side_length,
            y_offset=self.y_offset,
            x_offset=self.x_offset,
            flipped=self.flipped,
        )

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

    @property
    def means(self):
        return jnp.mean(self.triangles, axis=1)

    @property
    def area(self):
        return (3**0.5 / 4 * self.side_length**2) * len(self)
