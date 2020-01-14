import autoarray as aa

grid = aa.grid.uniform(shape_2d=(11, 11), pixel_scales=1.0)

aa.plot.grid(grid=grid)

grid = aa.grid.uniform(shape_2d=(11, 11), pixel_scales=1.0, origin=(5.0, 5.0))

aa.plot.grid(grid=grid, symmetric_around_centre=False)

aa.plot.grid(grid=grid, unit_conversion_factor=10.0, symmetric_around_centre=False)
