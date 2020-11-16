import autoarray as aa
import autoarray.plot as aplt
import numpy as np

grid = aa.Grid.uniform(shape_2d=(11, 11), pixel_scales=1.0)

aplt.Grid(grid=grid, axis_limits=[-1.5, 1.5, -2.5, 2.5])
