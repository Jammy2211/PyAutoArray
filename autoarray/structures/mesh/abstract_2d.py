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

        extent = self.geometry.extent_square if extent is None else extent

        return Grid2D.manual_extent(extent=extent, shape_native=shape_native)
