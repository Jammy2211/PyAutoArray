from autoarray import Grid2D, Grid2DIrregular
from autoconf import cached_property
import numpy as np


HEIGHT_FACTOR = 3**0.5 / 2


class Triangles:
    def __init__(
        self,
        y_min,
        y_max,
        x_min,
        x_max,
        scale,
    ):
        self.y_min = y_min
        self.x_min = x_min
        self.y_max = y_max
        self.x_max = x_max
        self.scale = scale

    @cached_property
    def rows(self):
        rows = []
        for y in np.arange(self.y_min, self.y_max + self.height, self.height):
            row = []
            offset = (len(rows) % 2) * self.scale / 2
            for x in np.arange(
                self.x_min - offset, self.x_max + self.scale, self.scale
            ):
                row.append(
                    (y, x),
                )
            rows.append(row)
        return rows

    @cached_property
    def long(self):
        return max(map(len, self.rows))

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
    def grid_2d(self):
        return Grid2DIrregular(
            values=[pair for row in self.rows for pair in row],
        )

    @cached_property
    def height(self):
        return HEIGHT_FACTOR * self.scale

    @classmethod
    def for_grid(cls, grid: Grid2D):
        scale = grid.pixel_scale
        y = grid[:, 0]
        x = grid[:, 1]

        y_min = y.min()
        y_max = y.max()

        x_min = x.min()
        x_max = x.max()

        return Triangles(y_min, y_max, x_min, x_max, scale)


def make_triangles(short_row, long_row):
    return [(point, long_row[i], long_row[i + 1]) for i, point in enumerate(short_row)]
