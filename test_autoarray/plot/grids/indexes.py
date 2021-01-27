import autoarray as aa
import autoarray.plot as aplt
import numpy as np

grid = aa.Grid2D.uniform(shape_native=(11, 11), pixel_scales=1.0)

aplt.Grid2D(grid=grid, indexes=[0, 1, 2, 14])

aplt.Grid2D(grid=grid, indexes=[(0, 0), (2, 3), (4, 1), (8, 5)])

aplt.Grid2D(grid=grid, indexes=[[(0, 0), (2, 3)], [(4, 1), (8, 5)]])

aplt.Grid2D(grid=grid, indexes=[[[0, 0], [2, 3]], [[4, 1], [8, 5]]])
