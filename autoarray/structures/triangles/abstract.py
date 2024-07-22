from abc import abstractmethod, ABC

import numpy as np

from autoarray import Grid2D

HEIGHT_FACTOR = 3**0.5 / 2


class AbstractTriangles(ABC):
    def __init__(
        self,
        indices: np.ndarray,
        vertices: np.ndarray,
    ):
        """
        Represents a set of triangles in efficient NumPy arrays.

        Parameters
        ----------
        indices
            The indices of the vertices of the triangles. This is a 2D array where each row is a triangle
            with the three indices of the vertices.
        vertices
            The vertices of the triangles.
        """
        self.indices = indices
        self.vertices = vertices

    @classmethod
    def for_limits_and_scale(
        cls,
        y_min: float,
        y_max: float,
        x_min: float,
        x_max: float,
        scale: float,
    ) -> "AbstractTriangles":
        height = scale * HEIGHT_FACTOR

        vertices = []
        indices = []
        vertex_dict = {}

        def add_vertex(v):
            if v not in vertex_dict:
                vertex_dict[v] = len(vertices)
                vertices.append(v)
            return vertex_dict[v]

        rows = []
        for row_y in np.arange(y_min, y_max + height, height):
            row = []
            offset = (len(rows) % 2) * scale / 2
            for col_x in np.arange(x_min - offset, x_max + scale, scale):
                row.append((row_y, col_x))
            rows.append(row)

        for i in range(len(rows) - 1):
            row = rows[i]
            next_row = rows[i + 1]
            for j in range(len(row)):
                if i % 2 == 0 and j < len(next_row) - 1:
                    t1 = [
                        add_vertex(row[j]),
                        add_vertex(next_row[j]),
                        add_vertex(next_row[j + 1]),
                    ]
                    if j < len(row) - 1:
                        t2 = [
                            add_vertex(row[j]),
                            add_vertex(row[j + 1]),
                            add_vertex(next_row[j + 1]),
                        ]
                        indices.append(t2)
                elif i % 2 == 1 and j < len(next_row) - 1:
                    t1 = [
                        add_vertex(row[j]),
                        add_vertex(next_row[j]),
                        add_vertex(row[j + 1]),
                    ]
                    indices.append(t1)
                    if j < len(next_row) - 1:
                        t2 = [
                            add_vertex(next_row[j]),
                            add_vertex(next_row[j + 1]),
                            add_vertex(row[j + 1]),
                        ]
                        indices.append(t2)
                else:
                    continue
                indices.append(t1)

        vertices = np.array(vertices)
        indices = np.array(indices)

        return cls(
            indices=indices,
            vertices=vertices,
        )

    @classmethod
    def for_grid(cls, grid: Grid2D) -> "AbstractTriangles":
        """
        Create a grid of equilateral triangles from a regular grid.

        Parameters
        ----------
        grid
            The regular grid to convert to a grid of triangles.

        Returns
        -------
        The grid of triangles.
        """

        scale = grid.pixel_scale

        y = grid[:, 0]
        x = grid[:, 1]

        y_min = y.min()
        y_max = y.max()
        x_min = x.min()
        x_max = x.max()

        return cls.for_limits_and_scale(y_min, y_max, x_min, x_max, scale)

    @abstractmethod
    def with_vertices(self, vertices: np.ndarray) -> "AbstractTriangles":
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

    @abstractmethod
    def for_indexes(self, indexes: np.ndarray) -> "AbstractTriangles":
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
