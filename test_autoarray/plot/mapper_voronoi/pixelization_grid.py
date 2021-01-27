import autoarray as aa
import autoarray.plot as aplt
import numpy as np


grid_7x7 = aa.Grid2D.uniform(shape_native=(7, 7), pixel_scales=0.25)
grid_9 = aa.Grid2D.manual_slim(
    grid=[
        [0.6, -0.3],
        [0.5, -0.8],
        [0.2, 0.1],
        [0.0, 0.5],
        [-0.3, -0.8],
        [-0.6, -0.5],
        [-0.4, -1.1],
        [-1.2, 0.8],
        [-1.5, 0.9],
    ],
    shape_native=(3, 3),
    pixel_scales=1.0,
)
voronoi_grid = aa.Grid2DVoronoi(
    grid=grid_9,
    nearest_pixelization_index_for_slim_index=np.zeros(
        shape=grid_7x7.shape_slim, dtype="int"
    ),
)
voronoi_mapper = aa.Mapper(
    source_grid_slim=grid_7x7, source_pixelization_grid=voronoi_grid
)

aplt.MapperObj(mapper=voronoi_mapper, include_pixelization_grid=True)
