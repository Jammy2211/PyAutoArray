import autoarray.plot as aplt

import numpy as np


def test__draws_delaunay_pixels_for_sensible_input(delaunay_mapper_9_3x3):
    delaunay_drawer = aplt.DelaunayDrawer(linewidth=0.5, edgecolor="r", alpha=1.0)

    delaunay_drawer.draw_delaunay_pixels(
        mapper=delaunay_mapper_9_3x3,
        pixel_values=np.ones(9),
        units=aplt.Units(),
        cmap=aplt.Cmap(),
        colorbar=None,
    )

    values = np.ones(9)
    values[0] = 0.0

    delaunay_drawer.draw_delaunay_pixels(
        mapper=delaunay_mapper_9_3x3,
        pixel_values=values,
        units=aplt.Units(),
        cmap=aplt.Cmap(),
        colorbar=aplt.Colorbar(fraction=0.1, pad=0.05),
    )
