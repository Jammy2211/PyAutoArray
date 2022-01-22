from autoarray.structures.grids.two_d.grid_2d_pixelization import Grid2DRectangular
from autoarray.structures.grids.two_d.grid_2d_pixelization import Grid2DDelaunay
from autoarray.structures.grids.two_d.grid_2d_pixelization import Grid2DVoronoi


def mapper_from(
    source_grid_slim,
    source_pixelization_grid,
    data_pixelization_grid=None,
    hyper_data=None,
):

    if isinstance(source_pixelization_grid, Grid2DRectangular):
        from autoarray.inversion.mappers.rectangular import MapperRectangularNoInterp

        return MapperRectangularNoInterp(
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


        if source_pixelization_grid.uses_interpolation:

            from autoarray.inversion.mappers.voronoi import MapperVoronoi

            return MapperVoronoi(
                source_grid_slim=source_grid_slim,
                source_pixelization_grid=source_pixelization_grid,
                data_pixelization_grid=data_pixelization_grid,
                hyper_image=hyper_data,
            )

        else:

            from autoarray.inversion.mappers.voronoi import MapperVoronoiNoInterp

            return MapperVoronoiNoInterp(
                source_grid_slim=source_grid_slim,
                source_pixelization_grid=source_pixelization_grid,
                data_pixelization_grid=data_pixelization_grid,
                hyper_image=hyper_data,
            )
