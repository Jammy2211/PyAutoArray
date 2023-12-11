from typing import Dict, Optional

from autoarray.inversion.pixelization.mappers.mapper_grids import MapperGrids
from autoarray.inversion.regularization.abstract import AbstractRegularization
from autoarray.structures.mesh.rectangular_2d import Mesh2DRectangular
from autoarray.structures.mesh.delaunay_2d import Mesh2DDelaunay
from autoarray.structures.mesh.voronoi_2d import Mesh2DVoronoi


def mapper_from(
    mapper_grids: MapperGrids,
    regularization: Optional[AbstractRegularization],
    run_time_dict: Optional[Dict] = None,
):
    """
    Factory which given input `MapperGrids` and `Regularization` objects creates a `Mapper`.

    A `Mapper` determines the mappings between a masked dataset's pixels and pixels of a linear object pixelization.
    The mapper is used in order to fit a dataset via an inversion. Docstrings in the packages `linear_obj`, `mesh`,
    `pixelization`, `mapper_grids` `mapper` and `inversion` provide more details.

    This factory inspects the type of mesh contained in the `MapperGrids` and uses this to determine the type of
    `Mapper` it creates. For example, if a Delaunay mesh is used, a `MapperDelaunay` is created.

    Parameters
    ----------
    mapper_grids
        An object containing the data grid and mesh grid in both the data-frame and source-frame used by the
        mapper to map data-points to linear object parameters.
    regularization
        The regularization scheme which may be applied to this linear object in order to smooth its solution,
        which for a mapper smooths neighboring pixels on the mesh.
    run_time_dict
        A dictionary which contains timing of certain functions calls which is used for profiling.

    Returns
    -------
    A mapper whose type is determined by the input `mapper_grids` mesh type.
    """
    from autoarray.inversion.pixelization.mappers.rectangular import (
        MapperRectangularNoInterp,
    )
    from autoarray.inversion.pixelization.mappers.delaunay import MapperDelaunay
    from autoarray.inversion.pixelization.mappers.voronoi import MapperVoronoi
    from autoarray.inversion.pixelization.mappers.voronoi import MapperVoronoiNoInterp

    if isinstance(mapper_grids.source_plane_mesh_grid, Mesh2DRectangular):
        return MapperRectangularNoInterp(
            mapper_grids=mapper_grids,
            regularization=regularization,
            run_time_dict=run_time_dict,
        )
    elif isinstance(mapper_grids.source_plane_mesh_grid, Mesh2DDelaunay):
        return MapperDelaunay(
            mapper_grids=mapper_grids,
            regularization=regularization,
            run_time_dict=run_time_dict,
        )
    elif isinstance(mapper_grids.source_plane_mesh_grid, Mesh2DVoronoi):
        if mapper_grids.source_plane_mesh_grid.uses_interpolation:
            return MapperVoronoi(
                mapper_grids=mapper_grids,
                regularization=regularization,
                run_time_dict=run_time_dict,
            )

        return MapperVoronoiNoInterp(
            mapper_grids=mapper_grids,
            regularization=regularization,
            run_time_dict=run_time_dict,
        )
