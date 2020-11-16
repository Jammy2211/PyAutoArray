import autoarray as aa
import autoarray.plot as aplt
import numpy as np

grid = aa.Grid.uniform(shape_2d=(11, 11), pixel_scales=1.0)

color_array = np.linspace(start=0.0, stop=1.0, num=grid.shape_1d)

aplt.Grid(
    grid=grid,
    color_array=color_array,
    axis_limits=[-1.5, 1.5, -2.5, 2.5],
    lines=[[(1.0, 1.0), (2.0, 2.0)], [(2.0, 4.0), (5.0, 6.0)]],
    indexes=[0, 1, 2, 14],
    symmetric_around_centre=False,
)

aplt.Grid(
    grid=grid,
    color_array=color_array,
    axis_limits=[-1.5, 1.5, -2.5, 2.5],
    lines=[[(1.0, 1.0), (2.0, 2.0)], [(2.0, 4.0), (5.0, 6.0)]],
    indexes=[0, 1, 2, 14],
    symmetric_around_centre=True,
)
