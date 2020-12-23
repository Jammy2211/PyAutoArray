from autoconf import conf
import autoarray as aa
import autoarray.plot as aplt
from autoarray.plot import wrap_mat

from os import path

from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pytest
import os, shutil
import numpy as np

directory = path.dirname(path.realpath(__file__))


def test__all_load_correctly():

    origin_scatter = aplt.OriginScatter()
    origin_scatter.scatter_grid(grid=aa.Grid.uniform(shape_2d=(3, 3), pixel_scales=1.0))

    assert origin_scatter.kwargs["s"] == 80

    mask_scatter = aplt.MaskScatter()
    mask_scatter.scatter_grid(grid=aa.Grid.uniform(shape_2d=(3, 3), pixel_scales=1.0))

    assert mask_scatter.kwargs["s"] == 12

    border_scatter = aplt.BorderScatter()
    border_scatter.scatter_grid(grid=aa.Grid.uniform(shape_2d=(3, 3), pixel_scales=1.0))

    assert border_scatter.kwargs["s"] == 13

    positions_scatter = aplt.PositionsScatter()
    positions_scatter.scatter_grid(
        grid=aa.Grid.uniform(shape_2d=(3, 3), pixel_scales=1.0)
    )

    assert positions_scatter.kwargs["s"] == 15

    index_scatter = aplt.IndexScatter()
    index_scatter.scatter_grid(grid=aa.Grid.uniform(shape_2d=(3, 3), pixel_scales=1.0))

    assert index_scatter.kwargs["s"] == 20

    pixelization_grid_scatter = aplt.PixelizationGridScatter()
    pixelization_grid_scatter.scatter_grid(
        grid=aa.Grid.uniform(shape_2d=(3, 3), pixel_scales=1.0)
    )

    assert pixelization_grid_scatter.kwargs["s"] == 5

    parallel_overscan_plot = aplt.ParallelOverscanPlot()
    parallel_overscan_plot.plot_y_vs_x(
        y=[1.0, 2.0, 3.0], x=[1.0, 2.0, 3.0], plot_axis_type="linear"
    )

    assert parallel_overscan_plot.kwargs["linewidth"] == 1

    serial_overscan_plot = aplt.SerialOverscanPlot()
    serial_overscan_plot.plot_y_vs_x(
        y=[1.0, 2.0, 3.0], x=[1.0, 2.0, 3.0], plot_axis_type="semilogy"
    )

    assert serial_overscan_plot.kwargs["linewidth"] == 2

    serial_prescan_plot = aplt.SerialPrescanPlot()
    serial_prescan_plot.plot_y_vs_x(
        y=[1.0, 2.0, 3.0], x=[1.0, 2.0, 3.0], plot_axis_type="loglog"
    )

    assert serial_prescan_plot.kwargs["linewidth"] == 3
