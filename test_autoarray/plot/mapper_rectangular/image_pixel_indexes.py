import autoarray as aa
import autoarray.plot as aplt

grid_7x7 = aa.Grid2D.uniform(shape_native=(7, 7), pixel_scales=0.3)
grid_3x3 = aa.Grid2D.uniform(shape_native=(3, 3), pixel_scales=1.0)
rectangular_grid = aa.Grid2DRectangular.overlay_grid(grid=grid_3x3, shape_native=(3, 3))
rectangular_mapper = aa.Mapper(
    source_grid_slim=grid_7x7, source_pixelization_grid=rectangular_grid
)

aplt.MapperObj(mapper=rectangular_mapper, full_indexes=[0, 1])
aplt.MapperObj(mapper=rectangular_mapper, full_indexes=[[0, 1]])
aplt.MapperObj(mapper=rectangular_mapper, full_indexes=[[0, 1], [2]])
aplt.MapperObj(mapper=rectangular_mapper, full_indexes=[[(0, 0), (0, 1)], [(1, 2)]])
