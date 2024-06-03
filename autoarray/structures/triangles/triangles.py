from typing import List, Tuple

from autoarray import Grid2D, Grid2DIrregular
from autoarray.structures.triangles.abstract_triangles import AbstractTriangles
from autoarray.structures.triangles.triangle import Triangle
from autoconf import cached_property
import numpy as np

HEIGHT_FACTOR = 3**0.5 / 2


class Triangles(AbstractTriangles):
    def __init__(
        self,
        rows: List[List[Tuple[float, float]]],
    ):
        """
        Represents a grid of equilateral triangles in the image plane.

        Rows are offset by half a pixel width. Every other row is one pixel shorter.
        Rows are spaced such that if all points are joined by lines, equilateral triangles are formed.

        Parameters
        ----------
        rows
            A list of rows, each containing a list of points.
            These rows are offset by half a pixel width.
        """
        self.rows = rows

    @cached_property
    def long(self) -> int:
        """
        The length of the longest row of points.
        """
        return max(map(len, self.rows))

    @cached_property
    def triangles(self) -> List[Triangle]:
        """
        A list of triangles in the image plane.
        """
        triangles = []

        for i, row in enumerate(self.rows):
            row = self.rows[i]

            if len(row) == self.long:
                row = row[1:-1]

            if i != 0:
                triangles.extend(make_triangles(row, self.rows[i - 1]))

            try:
                triangles.extend(make_triangles(row, self.rows[i + 1]))
            except IndexError:
                pass

        return triangles

    @cached_property
    def grid_2d(self) -> Grid2DIrregular:
        """
        A 2D grid comprising the coordinates of the vertices of the triangles.
        """
        return Grid2DIrregular(
            values=[pair for row in self.rows for pair in row],
        )

    def with_updated_grid(self, grid: Grid2DIrregular):
        assert len(grid) == len(self.grid_2d)

        rows = []
        start = 0
        for row in self.rows:
            rows.append(grid[start : start + len(row)])
            start += len(row)

        return Triangles(rows).triangles

    @classmethod
    def for_grid(cls, grid: Grid2D) -> "Triangles":
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

        height = scale * HEIGHT_FACTOR

        rows = []
        for y in np.arange(y_min, y_max + height, height):
            row = []
            offset = (len(rows) % 2) * scale / 2
            for x in np.arange(x_min - offset, x_max + scale, scale):
                row.append(
                    (y, x),
                )
            rows.append(row)

        return Triangles(rows)


def make_triangles(
    short_row: List[Tuple[float, float]],
    long_row: List[Tuple[float, float]],
) -> List[Triangle]:
    """
    Create a list of triangles from two rows of points.

    Parameters
    ----------
    short_row
        The row of points with the shorter length.
    long_row
        The row of points with the longer length.

    Returns
    -------
    A list of triangles.
    """
    return [
        Triangle(point, long_row[i], long_row[i + 1])
        for i, point in enumerate(short_row)
    ]
