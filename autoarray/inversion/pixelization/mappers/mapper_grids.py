from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING, Dict, Optional

if TYPE_CHECKING:
    from autoarray import Preloads

from autoarray.structures.grids.uniform_2d import Grid2D
from autoarray.structures.grids.sparse_2d import Grid2DSparse
from autoarray.structures.mesh.abstract_2d import Abstract2DMesh
from autoarray.inversion.pixelization.settings import SettingsPixelization


class MapperGrids:
    def __init__(
        self,
        source_grid_slim: Grid2D,
        source_mesh_grid: Abstract2DMesh = None,
        data_mesh_grid: Grid2DSparse = None,
        hyper_data: np.ndarray = None,
        settings: SettingsPixelization = SettingsPixelization(),
        preloads: Preloads = None,
        profiling_dict: Optional[Dict] = None,
    ):
        """
        Groups the different grids used by `Mesh` objects, the `mesh` package and the `pixelization` package, which
        create the following four grids:

        - `data_grid_slim`: the grid defining where data-points in frame of the data are.

        - `source_grid_slim`: the grid defining where the mapped coordinates of these data-points in the source-frame
         of the linear object are.

        - `dataset_mesh_grid`: the grid defining where the linear object parameters (e.g. what are used as pixels of
        the mapper) are in the data frame.

        - `source_mesh_grid`: the grid defining where the mapped coordinates of the linear object parameters
        are in the source frame.

        Read the docstrings of the `mesh` package for more information is this is unclear.

         This grouped set of grids are input into  `Mapper` objects, in order to determine the mappings between the
         masked data grid's data points (`data_grid_slim` and `source_grid_slim`) and the mesh's pixels
         (`data_mesh_grid` and `source_mesh_grid`).

        Parameters
        ----------
        source_grid_slim
            A 2D grid of (y,x) coordinates associated with the unmasked 2D data after it has been transformed to the
            `source` reference frame.
        source_mesh_grid
            The 2D grid of (y,x) centres of every pixelization pixel in the `source` frame.
        data_mesh_grid
            The sparse set of (y,x) coordinates computed from the unmasked data in the `data` frame. This has a
            transformation applied to it to create the `source_mesh_grid`.
        hyper_data
            An image which is used to determine the `data_mesh_grid` and therefore adapt the distribution of
            pixels of the Delaunay grid to the data it discretizes.
        settings
            Settings controlling the pixelization for example if a border is used to relocate its exterior coordinates.
        preloads
            Preloads in memory certain arrays which may be known beforehand in order to speed up the calculation,
            for example the `source_mesh_grid` could be preloaded.
        profiling_dict
            A dictionary which contains timing of certain functions calls which is used for profiling.
        """

        from autoarray.preloads import Preloads

        self.source_grid_slim = source_grid_slim
        self.source_mesh_grid = source_mesh_grid
        self.data_mesh_grid = data_mesh_grid
        self.hyper_data = hyper_data
        self.settings = settings
        self.preloads = preloads or Preloads()
        self.profiling_dict = profiling_dict
