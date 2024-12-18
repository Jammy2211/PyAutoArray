from abc import abstractmethod

import numpy as np

from autoarray import Grid2D, AbstractTriangles
from autoarray.structures.triangles.abstract import HEIGHT_FACTOR


class AbstractArrayTriangles(AbstractTriangles):
    def __init__(
        self,
        indices,
        vertices,
        **kwargs,
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
        self._indices = indices
        self._vertices = vertices

    @property
    def indices(self):
        return self._indices

    @property
    def vertices(self):
        return self._vertices

    def __len__(self):
        return len(self.triangles)

    @property
    def area(self) -> float:
        """
        The total area covered by the triangles.
        """
        triangles = self.triangles
        return (
            0.5
            * np.abs(
                (triangles[:, 0, 0] * (triangles[:, 1, 1] - triangles[:, 2, 1]))
                + (triangles[:, 1, 0] * (triangles[:, 2, 1] - triangles[:, 0, 1]))
                + (triangles[:, 2, 0] * (triangles[:, 0, 1] - triangles[:, 1, 1]))
            ).sum()
        )

    @property
    @abstractmethod
    def numpy(self):
        pass

    def _up_sample_triangle(self):
        triangles = self.triangles

        m01 = (triangles[:, 0] + triangles[:, 1]) / 2
        m12 = (triangles[:, 1] + triangles[:, 2]) / 2
        m20 = (triangles[:, 2] + triangles[:, 0]) / 2

        return self.numpy.concatenate(
            [
                self.numpy.stack([triangles[:, 1], m12, m01], axis=1),
                self.numpy.stack([triangles[:, 2], m20, m12], axis=1),
                self.numpy.stack([m01, m12, m20], axis=1),
                self.numpy.stack([triangles[:, 0], m01, m20], axis=1),
            ],
            axis=0,
        )

    def _neighborhood_triangles(self):
        triangles = self.triangles

        new_v0 = triangles[:, 1] + triangles[:, 2] - triangles[:, 0]
        new_v1 = triangles[:, 0] + triangles[:, 2] - triangles[:, 1]
        new_v2 = triangles[:, 0] + triangles[:, 1] - triangles[:, 2]

        return self.numpy.concatenate(
            [
                self.numpy.stack([new_v0, triangles[:, 1], triangles[:, 2]], axis=1),
                self.numpy.stack([triangles[:, 0], new_v1, triangles[:, 2]], axis=1),
                self.numpy.stack([triangles[:, 0], triangles[:, 1], new_v2], axis=1),
                triangles,
            ],
            axis=0,
        )

    def __str__(self):
        return f"{self.__class__.__name__} with {len(self.indices)} triangles"

    def __repr__(self):
        return str(self)

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
