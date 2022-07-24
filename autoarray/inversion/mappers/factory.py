from autoarray.structures.grids.grid_2d_mesh import Grid2DRectangular
from autoarray.structures.grids.grid_2d_mesh import Grid2DDelaunay
from autoarray.structures.grids.grid_2d_mesh import Grid2DVoronoi


def mapper_from(
    source_grid_slim, source_mesh_grid, data_mesh_grid=None, hyper_data=None
):

    from autoarray.inversion.mappers.rectangular import MapperRectangularNoInterp
    from autoarray.inversion.mappers.delaunay import MapperDelaunay
    from autoarray.inversion.mappers.voronoi import MapperVoronoi
    from autoarray.inversion.mappers.voronoi import MapperVoronoiNoInterp

    if isinstance(source_mesh_grid, Grid2DRectangular):

        return MapperRectangularNoInterp(
            source_grid_slim=source_grid_slim,
            source_mesh_grid=source_mesh_grid,
            data_mesh_grid=data_mesh_grid,
            hyper_image=hyper_data,
        )
    elif isinstance(source_mesh_grid, Grid2DDelaunay):

        return MapperDelaunay(
            source_grid_slim=source_grid_slim,
            source_mesh_grid=source_mesh_grid,
            data_mesh_grid=data_mesh_grid,
            hyper_image=hyper_data,
        )
    elif isinstance(source_mesh_grid, Grid2DVoronoi):

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
