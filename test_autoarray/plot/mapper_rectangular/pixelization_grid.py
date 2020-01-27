import autoarray as aa
import autoarray.plot as aplt

grid_7x7 = aa.grid.uniform(shape_2d=(7, 7), pixel_scales=0.3)
grid_3x3 = aa.grid.uniform(shape_2d=(3, 3), pixel_scales=1.0)
rectangular_grid = aa.grid_rectangular.overlay_grid(grid=grid_3x3, shape_2d=(3, 3))
rectangular_mapper = aa.mapper(grid=grid_7x7, pixelization_grid=rectangular_grid)

aplt.mapper_obj(mapper=rectangular_mapper, include_pixelization_grid=True)
