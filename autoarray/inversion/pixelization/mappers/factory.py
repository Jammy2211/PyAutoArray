from typing import Dict, Optional

from autoarray.inversion.pixelization.mappers.mapper_grids import MapperGrids
from autoarray.inversion.regularization.abstract import AbstractRegularization
from autoarray.structures.mesh.rectangular_2d import Mesh2DRectangular
from autoarray.structures.mesh.delaunay_2d import Mesh2DDelaunay
from autoarray.structures.mesh.voronoi_2d import Mesh2DVoronoi


def mapper_from(
    mapper_grids: MapperGrids,
    regularization: Optional[AbstractRegularization],
    profiling_dict: Optional[Dict] = None,
):

    from autoarray.inversion.pixelization.mappers.rectangular import (
        MapperRectangularNoInterp,
    )
    from autoarray.inversion.pixelization.mappers.delaunay import MapperDelaunay
    from autoarray.inversion.pixelization.mappers.voronoi import MapperVoronoi
    from autoarray.inversion.pixelization.mappers.voronoi import MapperVoronoiNoInterp

    if isinstance(mapper_grids.source_mesh_grid, Mesh2DRectangular):

        return MapperRectangularNoInterp(
            mapper_grids=mapper_grids,
            regularization=regularization,
            profiling_dict=profiling_dict,
        )
    elif isinstance(mapper_grids.source_mesh_grid, Mesh2DDelaunay):

        return MapperDelaunay(
            mapper_grids=mapper_grids,
            regularization=regularization,
            profiling_dict=profiling_dict,
        )
    elif isinstance(mapper_grids.source_mesh_grid, Mesh2DVoronoi):

        if mapper_grids.source_mesh_grid.uses_interpolation:

            return MapperVoronoi(
                mapper_grids=mapper_grids,
                regularization=regularization,
                profiling_dict=profiling_dict,
            )

        return MapperVoronoiNoInterp(
            mapper_grids=mapper_grids,
            regularization=regularization,
            profiling_dict=profiling_dict,
        )