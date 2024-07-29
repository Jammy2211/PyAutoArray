from typing import Tuple

import jax
from jax import numpy as np, lax
from jax.tree_util import register_pytree_node_class
from jax import jit

from autoarray.structures.triangles.abstract import AbstractTriangles


@register_pytree_node_class
class ArrayTriangles(AbstractTriangles):
    @property
    @jit
    def triangles(self):
        def valid_triangle(index):
            return lax.cond(
                np.any(index == -1),
                lambda _: np.full((3, 2), np.nan, dtype=np.float32),
                lambda idx: self.vertices[idx],
                operand=index,
            )

        return jax.vmap(valid_triangle)(self.indices)

    @property
    @jit
    def means(self):
        return np.mean(self.triangles, axis=1)

    @jit
    def containing_indices(self, point: Tuple[float, float]) -> np.ndarray:
        """
        Find the triangles that contain a given point.

        Parameters
        ----------
        point
            The point to find the containing triangles for.

        Returns
        -------
        The triangles that contain the point.
        """
        x, y = point

        triangles = self.triangles

        y1, x1 = triangles[:, 0, 1], triangles[:, 0, 0]
        y2, x2 = triangles[:, 1, 1], triangles[:, 1, 0]
        y3, x3 = triangles[:, 2, 1], triangles[:, 2, 0]

        denominator = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)

        a = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / denominator
        b = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / denominator
        c = 1 - a - b

        inside = (0 <= a) & (a <= 1) & (0 <= b) & (b <= 1) & (0 <= c) & (c <= 1)

        return np.where(inside, size=5, fill_value=-1)[0]

    @jit
    def for_indexes(self, indexes: np.ndarray) -> "ArrayTriangles":
        """
        Create a new ArrayTriangles containing indices and vertices corresponding to the given indexes
        but without duplicate vertices.

        Parameters
        ----------
        indexes
            The indexes of the triangles to include in the new ArrayTriangles.

        Returns
        -------
        The new ArrayTriangles instance.
        """

        def valid_indices(index):
            return lax.cond(
                index == -1,
                lambda _: np.full((3,), -1, dtype=np.int32),
                lambda idx: self.indices[idx],
                operand=index,
            )

        selected_indices = jax.vmap(valid_indices)(indexes)

        flat_indices = selected_indices.flatten()

        def valid_vertices(index):
            return lax.cond(
                index == -1,
                lambda _: np.full((2,), np.nan, dtype=np.float32),
                lambda idx: self.vertices[idx],
                operand=index,
            )

        selected_vertices = jax.vmap(valid_vertices)(flat_indices)

        unique_vertices, inv_indices = np.unique(
            selected_vertices,
            axis=0,
            return_inverse=True,
            size=selected_vertices.shape[0],
            fill_value=np.nan,
            equal_nan=False,
        )

        def swap_nan(index):
            return lax.cond(
                np.any(np.isnan(unique_vertices[index])),
                lambda _: np.array([-1], dtype=np.int32),
                lambda idx: idx,
                operand=index,
            )

        inv_indices = jax.vmap(swap_nan)(inv_indices)

        new_indices = inv_indices.reshape(selected_indices.shape)

        return ArrayTriangles(indices=new_indices, vertices=unique_vertices)

    @jit
    def up_sample(self) -> "ArrayTriangles":
        """
        Up-sample the triangles by adding a new vertex at the midpoint of each edge.

        This means each triangle becomes four smaller triangles.
        """
        triangles = self.triangles

        m01 = (triangles[:, 0] + triangles[:, 1]) / 2
        m12 = (triangles[:, 1] + triangles[:, 2]) / 2
        m20 = (triangles[:, 2] + triangles[:, 0]) / 2

        new_triangles = np.concatenate(
            [
                np.stack([triangles[:, 0], m01, m20], axis=1),
                np.stack([triangles[:, 1], m12, m01], axis=1),
                np.stack([triangles[:, 2], m20, m12], axis=1),
                np.stack([m01, m12, m20], axis=1),
            ],
            axis=0,
        )

        unique_vertices, inverse_indices = np.unique(
            new_triangles.reshape(-1, 2),
            axis=0,
            return_inverse=True,
            size=6 * triangles.shape[0],
        )
        new_indices = inverse_indices.reshape(-1, 3)

        return ArrayTriangles(
            indices=new_indices,
            vertices=unique_vertices,
        )

    @jit
    def neighborhood(self) -> "ArrayTriangles":
        """
        Create a new set of triangles that are the neighborhood of the current triangles.

        Includes the current triangles and the triangles that share an edge with the current triangles.
        """
        triangles = self.triangles

        new_v0 = triangles[:, 1] + triangles[:, 2] - triangles[:, 0]
        new_v1 = triangles[:, 0] + triangles[:, 2] - triangles[:, 1]
        new_v2 = triangles[:, 0] + triangles[:, 1] - triangles[:, 2]

        new_triangles = np.concatenate(
            [
                np.stack([new_v0, triangles[:, 1], triangles[:, 2]], axis=1),
                np.stack([triangles[:, 0], new_v1, triangles[:, 2]], axis=1),
                np.stack([triangles[:, 0], triangles[:, 1], new_v2], axis=1),
                triangles,
            ],
            axis=0,
        )

        max_new_triangles = 6 * triangles.shape[0]

        unique_vertices, inverse_indices = np.unique(
            new_triangles.reshape(-1, 2),
            axis=0,
            return_inverse=True,
            size=max_new_triangles,
        )
        new_indices = inverse_indices.reshape(-1, 3)

        new_indices_sorted = np.sort(new_indices, axis=1)

        unique_triangles_indices, unique_index_positions = np.unique(
            new_indices_sorted,
            axis=0,
            return_index=True,
            size=max_new_triangles,
        )

        return ArrayTriangles(
            indices=unique_triangles_indices,
            vertices=unique_vertices,
        )

    @jit
    def with_vertices(self, vertices: np.ndarray) -> "ArrayTriangles":
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

    def __iter__(self):
        return iter(self.triangles)

    def tree_flatten(self):
        """
        Flatten this model as a PyTree.
        """
        return (self.indices, self.vertices), ()

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """
        Unflatten a PyTree into a model.
        """
        return cls(
            indices=children[0],
            vertices=children[1],
        )

    @classmethod
    def for_limits_and_scale(
        cls,
        y_min: float,
        y_max: float,
        x_min: float,
        x_max: float,
        scale: float,
    ) -> "AbstractTriangles":
        triangles = super().for_limits_and_scale(
            y_min,
            y_max,
            x_min,
            x_max,
            scale,
        )
        return cls(
            indices=np.array(triangles.indices),
            vertices=np.array(triangles.vertices),
        )
