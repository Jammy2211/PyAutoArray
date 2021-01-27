import autoarray as aa
import autoarray.plot as aplt
import numpy as np

grid = aa.Grid2D.uniform(shape_native=(11, 11), pixel_scales=1.0)

aplt.Grid2D(grid=grid, lines=[(1.0, 1.0), (2.0, 2.0)])

aplt.Grid2D(grid=grid, lines=[[(1.0, 1.0), (2.0, 2.0)], [(2.0, 4.0), (5.0, 6.0)]])
