import numpy as np
from typing import Tuple


class AbstractGeometry2D:
    @property
    def extent(self) -> np.ndarray:
        raise NotImplementedError

    @property
    def extent_square(self) -> Tuple[float, float, float, float]:
        """
        Returns an extent where the y and x distances from each edge are the same.

        This ensures that a uniform grid with square pixels can be laid over this extent, such that an
        `interpolation_grid` can be computed which has square pixels. This is not necessary, but benefits visualization.
        """

        y_mean = 0.5 * (self.extent[2] + self.extent[3])
        y_half_length = 0.5 * (self.extent[3] - self.extent[2])

        x_mean = 0.5 * (self.extent[0] + self.extent[1])
        x_half_length = 0.5 * (self.extent[1] - self.extent[0])

        half_length = np.max([y_half_length, x_half_length])

        y0 = y_mean - half_length
        y1 = y_mean + half_length

        x0 = x_mean - half_length
        x1 = x_mean + half_length

        return (x0, x1, y0, y1)
