import jax
from jax import numpy as np, lax
from jax.tree_util import register_pytree_node_class
from jax import jit

from autoarray.structures.triangles.abstract import AbstractTriangles
from autoarray.structures.triangles.shape import Shape

MAX_CONTAINING_SIZE = 10


def remove_duplicates(new_triangles):
    unique_vertices, inverse_indices = np.unique(
        new_triangles.reshape(-1, 2),
        axis=0,
        return_inverse=True,
        size=int(1.5 * new_triangles.shape[0]),
        fill_value=np.nan,
        equal_nan=True,
    )

    def swap_nan(index):
        return lax.cond(
            np.any(np.isnan(unique_vertices[index])),
            lambda _: np.array([-1], dtype=np.int32),
            lambda idx: idx,
            operand=index,
        )

    inverse_indices = jax.vmap(swap_nan)(inverse_indices)

    new_indices = inverse_indices.reshape(-1, 3)

    new_indices_sorted = np.sort(new_indices, axis=1)

    unique_triangles_indices = np.unique(
        new_indices_sorted,
        axis=0,
        size=new_indices_sorted.shape[0],
        fill_value=np.array(
            [-1, -1, -1],
            dtype=np.int32,
        ),
    )

    return unique_triangles_indices, unique_vertices


@register_pytree_node_class
class ArrayTriangles(AbstractTriangles):
    def __init__(
        self,
        indices,
        vertices,
        max_containing_size=MAX_CONTAINING_SIZE,
    ):
        super().__init__(indices, vertices)
        self.max_containing_size = max_containing_size

    @property
    def numpy(self):
        return np

    @property
    @jit
    def triangles(self) -> np.ndarray:
        """
        The triangles as a 3x2 array of vertices.
        """

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
    def means(self) -> np.ndarray:
        """
        The mean of each triangle.
        """
        return np.mean(self.triangles, axis=1)

    @jit
    def containing_indices(self, shape: Shape) -> np.ndarray:
        """
        Find the triangles that insect with a given shape.

        Parameters
        ----------
        shape
            The shape

        Returns
        -------
        The triangles that intersect the shape.
        """
        inside = shape.mask(self.triangles)

        return np.where(
            inside,
            size=self.max_containing_size,
            fill_value=-1,
        )[0]

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
            equal_nan=True,
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

        new_indices_sorted = np.sort(new_indices, axis=1)

        unique_triangles_indices = np.unique(
            new_indices_sorted,
            axis=0,
            size=new_indices_sorted.shape[0],
            fill_value=np.array(
                [-1, -1, -1],
                dtype=np.int32,
            ),
        )

        return ArrayTriangles(
            indices=unique_triangles_indices,
            vertices=unique_vertices,
        )

    @jit
    def up_sample(self) -> "ArrayTriangles":
        """
        Up-sample the triangles by adding a new vertex at the midpoint of each edge.

        This means each triangle becomes four smaller triangles.
        """
        new_indices, unique_vertices = remove_duplicates(self._up_sample_triangle())

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
        new_indices, unique_vertices = remove_duplicates(self._neighborhood_triangles())

        return ArrayTriangles(
            indices=new_indices,
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
        return (
            self.indices,
            self.vertices,
        ), (self.max_containing_size,)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """
        Unflatten a PyTree into a model.
        """
        return cls(
            indices=children[0],
            vertices=children[1],
            max_containing_size=aux_data[0],
        )

    @classmethod
    def for_limits_and_scale(
        cls,
        y_min: float,
        y_max: float,
        x_min: float,
        x_max: float,
        scale: float,
        max_containing_size=MAX_CONTAINING_SIZE,
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
            max_containing_size=max_containing_size,
        )
