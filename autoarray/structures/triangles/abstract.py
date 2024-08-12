from abc import abstractmethod, ABC
from typing import Tuple

import numpy as np

from autoarray import Grid2D

HEIGHT_FACTOR = 3**0.5 / 2


class AbstractTriangles(ABC):
    def __init__(
        self,
        indices,
        vertices,
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

    def _containing_mask(self, point):
        x, y = point

        triangles = self.triangles

        y1, x1 = triangles[:, 0, 1], triangles[:, 0, 0]
        y2, x2 = triangles[:, 1, 1], triangles[:, 1, 0]
        y3, x3 = triangles[:, 2, 1], triangles[:, 2, 0]

        denominator = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)

        a = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / denominator
        b = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / denominator
        c = 1 - a - b

        return (0 <= a) & (a <= 1) & (0 <= b) & (b <= 1) & (0 <= c) & (c <= 1)

    def __str__(self):
        return f"{self.__class__.__name__} with {len(self.indices)} triangles"

    def __repr__(self):
        return str(self)

    @property
    @abstractmethod
    def triangles(self):
        pass

    @classmethod
    def for_limits_and_scale(
        cls,
        y_min: float,
        y_max: float,
        x_min: float,
        x_max: float,
        scale: float,
        **kwargs,
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
            **kwargs,
        )

    @classmethod
    def for_grid(
        cls,
        grid: Grid2D,
        **kwargs,
    ) -> "AbstractTriangles":
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

        return cls.for_limits_and_scale(
            y_min,
            y_max,
            x_min,
            x_max,
            scale,
            **kwargs,
        )

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

    @abstractmethod
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

    @abstractmethod
    def neighborhood(self) -> "AbstractTriangles":
        """
        Create a new set of triangles that are the neighborhood of the current triangles.

        Includes the current triangles and the triangles that share an edge with the current triangles.
        """

    @abstractmethod
    def up_sample(self) -> "AbstractTriangles":
        """
        Up-sample the triangles by adding a new vertex at the midpoint of each edge.

        This means each triangle becomes four smaller triangles.
        """

    @property
    @abstractmethod
    def means(self):
        pass
