import itertools
import numpy as np
from typing import Dict, List, Optional

from autoconf import cached_property

from autoarray.inversion.linear_obj import LinearObj
from autoarray.inversion.linear_obj import UniqueMappings
from autoarray.structures.grids.two_d.grid_2d_pixelization import Grid2DRectangular
from autoarray.structures.grids.two_d.grid_2d_pixelization import Grid2DDelaunay
from autoarray.structures.grids.two_d.grid_2d_pixelization import Grid2DVoronoi

from autoarray.numba_util import profile_func
from autoarray.inversion.mappers import mapper_util


def mapper_from(
    source_grid_slim,
    source_pixelization_grid,
    data_pixelization_grid=None,
    hyper_data=None,
):

    if isinstance(source_pixelization_grid, Grid2DRectangular):
        from autoarray.inversion.mappers.rectangular import MapperRectangular

        return MapperRectangular(
            source_grid_slim=source_grid_slim,
            source_pixelization_grid=source_pixelization_grid,
            data_pixelization_grid=data_pixelization_grid,
            hyper_image=hyper_data,
        )
    elif isinstance(source_pixelization_grid, Grid2DDelaunay):
        from autoarray.inversion.mappers.delaunay import MapperDelaunay

        return MapperDelaunay(
            source_grid_slim=source_grid_slim,
            source_pixelization_grid=source_pixelization_grid,
            data_pixelization_grid=data_pixelization_grid,
            hyper_image=hyper_data,
        )
    elif isinstance(source_pixelization_grid, Grid2DVoronoi):
        from autoarray.inversion.mappers.voronoi import MapperVoronoi

        return MapperVoronoi(
            source_grid_slim=source_grid_slim,
            source_pixelization_grid=source_pixelization_grid,
            data_pixelization_grid=data_pixelization_grid,
            hyper_image=hyper_data,
        )
