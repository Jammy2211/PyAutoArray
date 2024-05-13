from autoarray import Grid2D
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
