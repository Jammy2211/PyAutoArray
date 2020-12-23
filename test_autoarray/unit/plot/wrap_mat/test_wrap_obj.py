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

    mask_scatter = aplt.MaskScatter()
    mask_scatter.scatter_grid(grid=aa.Grid.uniform(shape_2d=(3, 3), pixel_scales=1.0))

    border_scatter = aplt.BorderScatter()
    border_scatter.scatter_grid(grid=aa.Grid.uniform(shape_2d=(3, 3), pixel_scales=1.0))

    positions_scatter = aplt.PositionsScatter()
    positions_scatter.scatter_grid(
        grid=aa.Grid.uniform(shape_2d=(3, 3), pixel_scales=1.0)
    )

    index_scatter = aplt.IndexScatter()
    index_scatter.scatter_grid(grid=aa.Grid.uniform(shape_2d=(3, 3), pixel_scales=1.0))

    pixelization_grid_scatter = aplt.PositionsScatter()
    pixelization_grid_scatter.scatter_grid(
        grid=aa.Grid.uniform(shape_2d=(3, 3), pixel_scales=1.0)
    )

    parallel_overscan_line = aplt.ParallelOverscanLine()
    parallel_overscan_line.draw_y_vs_x(
        y=[1.0, 2.0, 3.0], x=[1.0, 2.0, 3.0], plot_axis_type="linear"
    )

    serial_overscan_line = aplt.SerialOverscanLine()
    serial_overscan_line.draw_y_vs_x(
        y=[1.0, 2.0, 3.0], x=[1.0, 2.0, 3.0], plot_axis_type="semilogy"
    )

    serial_prescan_line = aplt.SerialPrescanLine()
    serial_prescan_line.draw_y_vs_x(
        y=[1.0, 2.0, 3.0], x=[1.0, 2.0, 3.0], plot_axis_type="loglog"
    )
