import numpy as np
from typing import List, Optional, Union

from autoarray.plot.abstract_plotters import AbstractPlotter
from autoarray.plot.visuals.one_d import Visuals1D
from autoarray.plot.visuals.two_d import Visuals2D
from autoarray.plot.mat_plot.one_d import MatPlot1D
from autoarray.plot.mat_plot.two_d import MatPlot2D
from autoarray.plot.auto_labels import AutoLabels
from autoarray.structures.arrays.uniform_1d import Array1D
from autoarray.structures.arrays.uniform_2d import Array2D
from autoarray.structures.grids.uniform_1d import Grid1D
from autoarray.structures.grids.uniform_2d import Grid2D


class Array2DPlotter(AbstractPlotter):
    def __init__(
        self,
        array: Array2D,
        mat_plot_2d: MatPlot2D = None,
        visuals_2d: Visuals2D = None,
    ):
        """
        Plots `Array2D` objects using the matplotlib method `imshow()` and many other matplotlib functions which
        customize the plot's appearance.

        The `mat_plot_2d` attribute wraps matplotlib function calls to make the figure. By default, the settings
        passed to every matplotlib function called are those specified in the `config/visualize/mat_wrap/*.ini` files,
        but a user can manually input values into `MatPlot2d` to customize the figure's appearance.

        Overlaid on the figure are visuals, contained in the `Visuals2D` object. Attributes may be extracted from
        the `Array2D` and plotted via the visuals object.

        Parameters
        ----------
        array
            The 2D array the plotter plot.
        mat_plot_2d
            Contains objects which wrap the matplotlib function calls that make 2D plots.
        visuals_2d
            Contains 2D visuals that can be overlaid on 2D plots.
        """
        super().__init__(visuals_2d=visuals_2d, mat_plot_2d=mat_plot_2d)

        self.array = array

    def figure_2d(self):
        """
        Plots the plotter's `Array2D` object in 2D.
        """
        self.mat_plot_2d.plot_array(
            array=self.array,
            visuals_2d=self.visuals_2d,
            auto_labels=AutoLabels(title="Array2D", filename="array"),
        )


class Grid2DPlotter(AbstractPlotter):
    def __init__(
        self,
        grid: Grid2D,
        mat_plot_2d: MatPlot2D = None,
        visuals_2d: Visuals2D = None,
    ):
        """
        Plots `Grid2D` objects using the matplotlib method `scatter()` and many other matplotlib functions which
        customize the plot's appearance.

        The `mat_plot_2d` attribute wraps matplotlib function calls to make the figure. By default, the settings
        passed to every matplotlib function called are those specified in the `config/visualize/mat_wrap/*.ini` files,
        but a user can manually input values into `MatPlot2d` to customize the figure's appearance.

        Overlaid on the figure are visuals, contained in the `Visuals2D` object. Attributes may be extracted from
        the `Grid2D` and plotted via the visuals object.

        Parameters
        ----------
        grid
            The 2D grid the plotter plot.
        mat_plot_2d
            Contains objects which wrap the matplotlib function calls that make 2D plots.
        visuals_2d
            Contains 2D visuals that can be overlaid on 2D plots.
        """
        super().__init__(visuals_2d=visuals_2d, mat_plot_2d=mat_plot_2d)

        self.grid = grid

    def figure_2d(
        self,
        color_array: np.ndarray = None,
        plot_grid_lines: bool = False,
        plot_over_sampled_grid: bool = False,
    ):
        """
        Plots the plotter's `Grid2D` object in 2D.

        Parameters
        ----------
        color_array
            An array of RGB color values which can be used to give the plotted 2D grid a colorscale (w/ colorbar).
        plot_grid_lines
            If True, a rectangular grid of lines is plotted on the figure showing the pixels which the grid coordinates
            are centred on.
        plot_over_sampled_grid
            If True, the grid is plotted with over-sampled sub-gridded coordinates based on the `sub_size` attribute
            of the grid's over-sampling object.
        """
        self.mat_plot_2d.plot_grid(
            grid=self.grid,
            visuals_2d=self.visuals_2d,
            auto_labels=AutoLabels(title="Grid2D", filename="grid"),
            color_array=color_array,
            plot_grid_lines=plot_grid_lines,
            plot_over_sampled_grid=plot_over_sampled_grid,
        )


class YX1DPlotter(AbstractPlotter):
    def __init__(
        self,
        y: Union[Array1D, List],
        x: Optional[Union[Array1D, Grid1D, List]] = None,
        mat_plot_1d: MatPlot1D = None,
        visuals_1d: Visuals1D = None,
        should_plot_grid: bool = False,
        should_plot_zero: bool = False,
        plot_axis_type: Optional[str] = None,
        plot_yx_dict=None,
        auto_labels=AutoLabels(),
    ):
        """
        Plots two 1D objects using the matplotlib method `plot()` (or a similar method) and many other matplotlib
        functions which customize the plot's appearance.

        The `mat_plot_1d` attribute wraps matplotlib function calls to make the figure. By default, the settings
        passed to every matplotlib function called are those specified in the `config/visualize/mat_wrap/*.ini` files,
        but a user can manually input values into `MatPlot1d` to customize the figure's appearance.

        Overlaid on the figure are visuals, contained in the `Visuals1D` object. Attributes may be extracted from
        the `Array1D` and plotted via the visuals object.

        Parameters
        ----------
        y
            The 1D y values the plotter plot.
        x
            The 1D x values the plotter plot.
        mat_plot_1d
            Contains objects which wrap the matplotlib function calls that make 1D plots.
        visuals_1d
            Contains 1D visuals that can be overlaid on 1D plots.
        """

        if isinstance(y, list):
            y = Array1D.no_mask(values=y, pixel_scales=1.0)

        if isinstance(x, list):
            x = Array1D.no_mask(values=x, pixel_scales=1.0)

        super().__init__(visuals_1d=visuals_1d, mat_plot_1d=mat_plot_1d)

        self.y = y
        self.x = y.grid_radial if x is None else x
        self.should_plot_grid = should_plot_grid
        self.should_plot_zero = should_plot_zero
        self.plot_axis_type = plot_axis_type
        self.plot_yx_dict = plot_yx_dict or {}
        self.auto_labels = auto_labels

    def figure_1d(self):
        """
        Plots the plotter's y and x values in 1D.
        """

        self.mat_plot_1d.plot_yx(
            y=self.y,
            x=self.x,
            visuals_1d=self.visuals_1d,
            auto_labels=self.auto_labels,
            should_plot_grid=self.should_plot_grid,
            should_plot_zero=self.should_plot_zero,
            plot_axis_type_override=self.plot_axis_type,
            **self.plot_yx_dict,
        )
