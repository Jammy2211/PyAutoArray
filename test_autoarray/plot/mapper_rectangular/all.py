import autoarray as aa
import autoarray.plot as aplt

grid_7x7 = aa.Grid.uniform(shape_2d=(7, 7), pixel_scales=0.3)
grid_3x3 = aa.Grid.uniform(shape_2d=(3, 3), pixel_scales=1.0)
rectangular_grid = aa.GridRectangular.overlay_grid(grid=grid_3x3, shape_2d=(3, 3))
rectangular_mapper = aa.Mapper(grid=grid_7x7, pixelization_grid=rectangular_grid)

aplt.MapperObj(
    mapper=rectangular_mapper,
    include=aplt.Include(
        inversion_pixelization_grid=True, inversion_grid=True, inversion_border=True
    ),
    image_pixel_indexes=[[(0, 0), (0, 1)], [(1, 2)]],
    source_pixel_indexes=[[0, 1], [2]],
)
