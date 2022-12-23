import numpy as np
from typing import Optional, Tuple

from autoarray.structures.abstract_structure import Structure
from autoarray.structures.grids.uniform_2d import Grid2D


class Abstract2DMesh(Structure):
    @property
    def parameters(self) -> int:
        return self.pixels

    @property
    def pixels(self) -> int:
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

    def interpolation_grid_from(
        self,
        shape_native: Tuple[int, int] = (401, 401),
        extent: Optional[Tuple[float, float, float, float]] = None,
    ) -> Grid2D:
        """
        Returns a 2D grid of (y,x) coordinates on to which a reconstruction from a pixelization (e.g. a `Delaunay`,
        `Voronoi`) can be interpolated.

        The interpolation grid is computed from the pixelization's `extent`, which describes the [x0, x1, y0, y1]
        extent that the pixelization covers. This `extent` is converted to an `extent_square` such
        that `x1 - x0 = y1 - y1`, ensuring that the interpolation grid can have uniform square pixels.

        Parameters
        ----------
        shape_native
            The (y,x) shape of the interpolation grid.
        extent
            The (x0, x1, y0, y1) extent of the grid in scaled coordinates over which the grid is created if it
            is input.
        """

        extent = self.extent_square if extent is None else extent

        return Grid2D.manual_extent(extent=extent, shape_native=shape_native)
