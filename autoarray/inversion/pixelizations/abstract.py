import numpy as np
from typing import Dict, Optional

from autoarray.inversion.pixelizations.settings import SettingsPixelization
from autoarray.structures.grids.two_d.grid_2d import Grid2D
from autoarray.structures.grids.two_d.grid_2d import Grid2DSparse
from autoarray.preloads import Preloads

from autoarray.numba_util import profile_func


class AbstractPixelization:
    def __init__(self):
        """
        Abstract base class for a pixelization, which discretizes a grid of (y,x) coordinates into pixels.
        """

    def mapper_from(
        self, grid: Grid2D, border: np.ndarray, profiling_dict: Optional[Dict] = None
    ):
        raise NotImplementedError("pixelization_mapper_from should be overridden")

    def __eq__(self, other):
        return self.__dict__ == other.__dict__ and self.__class__ is other.__class__

    @profile_func
    def relocate_grid_via_border(
        self,
        grid: Grid2D,
        settings: SettingsPixelization = SettingsPixelization(),
        preloads: Preloads = Preloads(),
    ):
        """
        Relocates all coordinates of an input grid that are outside the pixelization's border to the edge of the
        border (see `abstract_grid_2d.relocated_grid_from()` for a full description of how relocation works.

        This is used in the project PyAutoLens to relocate the coordinates that are ray-traced near the centre of mass
        of galaxies, which are heavily demagnified and may trace to outskirts of the source-plane well beyond the
        border.

        Parameters
        ----------
        grid
            A 2D (y,x) grid of coordinates, whose coordinates outside the pixelization's border are relocated
            to the edge of this border.
        """
        if preloads.relocated_grid is None:

            if settings.use_border:
                return grid.relocated_grid_from(grid=grid)
            return grid

        else:

            return preloads.relocated_grid

    def relocate_pixelization_grid_via_border(
        self,
        grid: Grid2D,
        pixelization_grid: Grid2DSparse,
        settings: SettingsPixelization = SettingsPixelization(),
    ):
        raise NotImplementedError

    def make_pixelization_grid(
        self, relocated_grid=None, relocated_pixelization_grid=None
    ):
        raise NotImplementedError

    def weight_map_from(self, hyper_image: np.ndarray):

        raise NotImplementedError()

    def __str__(self):
        return "\n".join(["{}: {}".format(k, v) for k, v in self.__dict__.items()])

    def __repr__(self):
        return "{}\n{}".format(self.__class__.__name__, str(self))
