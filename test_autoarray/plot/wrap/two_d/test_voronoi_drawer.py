import autoarray.plot as aplt

import numpy as np


def test__draws_voronoi_pixels_for_sensible_input(voronoi_mapper_9_3x3):
    voronoi_drawer = aplt.VoronoiDrawer(linewidth=0.5, edgecolor="r", alpha=1.0)

    voronoi_drawer.draw_voronoi_pixels(
        mapper=voronoi_mapper_9_3x3,
        pixel_values=None,
        units=None,
        cmap=aplt.Cmap(),
        colorbar=None,
    )

    values = np.ones(9)
    values[0] = 0.0

    voronoi_drawer.draw_voronoi_pixels(
        mapper=voronoi_mapper_9_3x3,
        pixel_values=values,
        units=None,
        cmap=aplt.Cmap(),
        colorbar=aplt.Colorbar(fraction=0.1, pad=0.05),
    )
