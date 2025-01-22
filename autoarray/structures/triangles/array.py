import numpy as np

from autoarray.structures.triangles.abstract import AbstractTriangles
from autoarray.structures.triangles.shape import Shape


class ArrayTriangles(AbstractTriangles):
    @property
    def triangles(self):
        return self.vertices[self.indices]

    @property
    def numpy(self):
        return np

    @property
    def means(self):
        return np.mean(self.triangles, axis=1)

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

        return np.where(inside)[0]

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
        selected_indices = self.indices[indexes]

        flat_indices = selected_indices.flatten()
        unique_vertices, inverse_indices = np.unique(
            self.vertices[flat_indices], axis=0, return_inverse=True
        )

        new_indices = inverse_indices.reshape(selected_indices.shape)

        return ArrayTriangles(indices=new_indices, vertices=unique_vertices)

    def up_sample(self) -> "ArrayTriangles":
        """
        Up-sample the triangles by adding a new vertex at the midpoint of each edge.

        This means each triangle becomes four smaller triangles.
        """
        unique_vertices, inverse_indices = np.unique(
            self._up_sample_triangle().reshape(-1, 2), axis=0, return_inverse=True
        )
        new_indices = inverse_indices.reshape(-1, 3)

        return ArrayTriangles(
            indices=new_indices,
            vertices=unique_vertices,
        )

    def neighborhood(self) -> "ArrayTriangles":
        """
        Create a new set of triangles that are the neighborhood of the current triangles.

        Includes the current triangles and the triangles that share an edge with the current triangles.
        """
        unique_vertices, inverse_indices = np.unique(
            self._neighborhood_triangles().reshape(-1, 2),
            axis=0,
            return_inverse=True,
        )
        new_indices = inverse_indices.reshape(-1, 3)

        new_indices_sorted = np.sort(new_indices, axis=1)

        unique_triangles_indices, unique_index_positions = np.unique(
            new_indices_sorted, axis=0, return_index=True
        )

        return ArrayTriangles(
            indices=unique_triangles_indices,
            vertices=unique_vertices,
        )

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
