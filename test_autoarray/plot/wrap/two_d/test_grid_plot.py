import autoarray as aa
import autoarray.plot as aplt

import matplotlib.pyplot as plt
import numpy as np


def test___from_config_or_via_manual_input():

    grid_plot = aplt.GridPlot()

    assert grid_plot.config_dict["linewidth"] == 3
    assert grid_plot.config_dict["c"] == "k"

    grid_plot = aplt.GridPlot(c=["k", "b"])

    assert grid_plot.config_dict["linewidth"] == 3
    assert grid_plot.config_dict["c"] == ["k", "b"]

    grid_plot = aplt.GridPlot()
    grid_plot.is_for_subplot = True

    assert grid_plot.config_dict["linewidth"] == 1
    assert grid_plot.config_dict["c"] == "k"

    grid_plot = aplt.GridPlot(style=".")
    grid_plot.is_for_subplot = True

    assert grid_plot.config_dict["linewidth"] == 1
    assert grid_plot.config_dict["c"] == "k"


def test__plot_rectangular_grid_lines__draws_for_valid_extent_and_shape():

    line = aplt.GridPlot(linewidth=2, linestyle="--", c="k")

    line.plot_rectangular_grid_lines(extent=[0.0, 1.0, 0.0, 1.0], shape_native=(3, 2))
    line.plot_rectangular_grid_lines(
        extent=[-4.0, 8.0, -3.0, 10.0], shape_native=(8, 3)
    )


def test__plot_grid_list():

    line = aplt.GridPlot(linewidth=2, linestyle="--", c="k")

    line.plot_grid_list(grid_list=[aa.Grid2DIrregular([(1.0, 1.0), (2.0, 2.0)])])
    line.plot_grid_list(
        grid_list=[
            aa.Grid2DIrregular([(1.0, 1.0), (2.0, 2.0)]),
            aa.Grid2DIrregular([(3.0, 3.0)]),
        ]
    )


def test__errorbar_colored_grid__lists_of_coordinates_or_equivalent_2d_grids__with_color_array():

    errorbar = aplt.GridErrorbar(marker="x", c="k")

    cmap = plt.get_cmap("jet")

    errorbar.errorbar_grid_colored(
        grid=aa.Grid2DIrregular(
            [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0), (4.0, 4.0), (5.0, 5.0)]
        ),
        color_array=np.array([2.0, 2.0, 2.0, 2.0, 2.0]),
        y_errors=[1.0] * 5,
        x_errors=[1.0] * 5,
        cmap=cmap,
    )
    errorbar.errorbar_grid_colored(
        grid=aa.Grid2D.uniform(shape_native=(3, 2), pixel_scales=1.0),
        color_array=np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0]),
        cmap=cmap,
        y_errors=[1.0] * 6,
        x_errors=[1.0] * 6,
    )
