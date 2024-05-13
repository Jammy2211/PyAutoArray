from typing import List, Tuple

from autoarray import Grid2D, Grid2DIrregular
from autoarray.structures.triangles.triangle import Triangle
from autoconf import cached_property
import numpy as np

HEIGHT_FACTOR = 3**0.5 / 2


class Triangles:
    def __init__(
        self,
        rows: List[List[Tuple[float, float]]],
    ):
        self.rows = rows

    @cached_property
    def long(self):
        return max(map(len, self.rows))

    def containing(self, point: Tuple[float, float]):
        return [triangle for triangle in self.triangles if triangle.contains(point)]

    @cached_property
    def triangles(self):
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

        return Triangles(rows)

    @classmethod
    def for_grid(cls, grid: Grid2D):
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


def make_triangles(short_row, long_row):
    return [
        Triangle(point, long_row[i], long_row[i + 1])
        for i, point in enumerate(short_row)
    ]
