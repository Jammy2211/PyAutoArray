import numpy as np
from typing import Optional

from autoarray.inversion.pixelization.mesh.rectangular import RectangularMagnification

from autoarray.structures.grids.irregular_2d import Grid2DIrregular
from autoarray.structures.grids.uniform_2d import Grid2D
from autoarray.structures.mesh.rectangular_2d_uniform import Mesh2DRectangularUniform


class RectangularUniform(RectangularMagnification):

    def mesh_grid_from(
        self,
        source_plane_data_grid: Optional[Grid2D] = None,
        source_plane_mesh_grid: Optional[Grid2D] = None,
        xp=np,
    ) -> Mesh2DRectangularUniform:
        """
        Return the rectangular `source_plane_mesh_grid` as a `Mesh2DRectangular` object, which provides additional
        functionality for perform operatons that exploit the geometry of a rectangular pixelization.

        Parameters
        ----------
        source_plane_data_grid
            The (y,x) grid of coordinates over which the rectangular pixelization is overlaid, where this grid may have
            had exterior pixels relocated to its edge via the border.
        source_plane_mesh_grid
            Not used for a rectangular pixelization, because the pixelization grid in the `source` frame is computed
            by overlaying the `source_plane_data_grid` with the rectangular pixelization.
        """
        return Mesh2DRectangularUniform.overlay_grid(
            shape_native=self.shape,
            grid=Grid2DIrregular(source_plane_data_grid.over_sampled),
            xp=xp,
        )
