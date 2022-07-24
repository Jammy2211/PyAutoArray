from autoarray.structures.mesh.rectangular_2d import Mesh2DRectangular
from autoarray.structures.mesh.delaunay_2d import Mesh2DDelaunay
from autoarray.structures.mesh.voronoi_2d import Mesh2DVoronoi


def mapper_from(
    source_grid_slim, source_mesh_grid, data_mesh_grid=None, hyper_data=None
):

    from autoarray.inversion.pixelization.mappers.rectangular import (
        MapperRectangularNoInterp,
    )
    from autoarray.inversion.pixelization.mappers.delaunay import MapperDelaunay
    from autoarray.inversion.pixelization.mappers.voronoi import MapperVoronoi
    from autoarray.inversion.pixelization.mappers.voronoi import MapperVoronoiNoInterp

    if isinstance(source_mesh_grid, Mesh2DRectangular):

        return MapperRectangularNoInterp(
            source_grid_slim=source_grid_slim,
            source_mesh_grid=source_mesh_grid,
            data_mesh_grid=data_mesh_grid,
            hyper_image=hyper_data,
        )
    elif isinstance(source_mesh_grid, Mesh2DDelaunay):

        return MapperDelaunay(
            source_grid_slim=source_grid_slim,
            source_mesh_grid=source_mesh_grid,
            data_mesh_grid=data_mesh_grid,
            hyper_image=hyper_data,
        )
    elif isinstance(source_mesh_grid, Mesh2DVoronoi):

        if source_mesh_grid.uses_interpolation:

            return MapperVoronoi(
                source_grid_slim=source_grid_slim,
                source_mesh_grid=source_mesh_grid,
                data_mesh_grid=data_mesh_grid,
                hyper_image=hyper_data,
            )

        else:

            return MapperVoronoiNoInterp(
                source_grid_slim=source_grid_slim,
                source_mesh_grid=source_mesh_grid,
                data_mesh_grid=data_mesh_grid,
                hyper_image=hyper_data,
            )
