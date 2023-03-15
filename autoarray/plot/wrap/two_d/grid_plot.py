import matplotlib.pyplot as plt
import numpy as np
import itertools
from typing import List, Union, Tuple

from autoarray.plot.wrap.two_d.abstract import AbstractMatWrap2D
from autoarray.structures.grids.uniform_2d import Grid2D
from autoarray.structures.grids.irregular_2d import Grid2DIrregular


class GridPlot(AbstractMatWrap2D):
    """
    Plots `Grid2D` data structure that are better visualized as solid lines, for example rectangular lines that are
    plotted over an image and grids of (y,x) coordinates as lines (as opposed to a scatter of points
    using the `GridScatter` object).

    This object wraps the following Matplotlib methods:

    - plt.plot: https://matplotlib.org/3.3.3/api/_as_gen/matplotlib.pyplot.plot.html

    Parameters
    ----------
    colors : [str]
        The color or list of colors that the grid is plotted using. For plotting indexes or a grid list, a
        list of colors can be specified which the plot cycles through.
    """

    def plot_rectangular_grid_lines(
        self, extent: Tuple[float, float, float, float], shape_native: Tuple[int, int]
    ):
        """
        Plots a rectangular grid of lines on a plot, using the coordinate system of the figure.

        The size and shape of the grid is specified by the `extent` and `shape_native` properties of a data structure
        which will provide the rectangaular grid lines on a suitable coordinate system for the plot.

        Parameters
        ----------
        extent : (float, float, float, float)
            The extent of the rectangualr grid, with format [xmin, xmax, ymin, ymax]
        shape_native
            The 2D shape of the mask the array is paired with.
        """

        ys = np.linspace(extent[2], extent[3], shape_native[1] + 1)
        xs = np.linspace(extent[0], extent[1], shape_native[0] + 1)

        # grid lines
        for x in xs:
            plt.plot([x, x], [ys[0], ys[-1]], **self.config_dict)
        for y in ys:
            plt.plot([xs[0], xs[-1]], [y, y], **self.config_dict)

    def plot_grid(self, grid: Union[np.ndarray, Grid2D]):
        """
        Plot an input grid of (y,x) coordinates using the matplotlib method `plt.scatter`.

        Parameters
        ----------
        grid : Grid2D
            The grid of (y,x) coordinates that is plotted.
        """

        try:

            color = self.config_dict["c"]

            if isinstance(color, list):
                color = color[0]

            config_dict = self.config_dict
            config_dict.pop("c")

            plt.plot(grid[:, 1], grid[:, 0], c=color, **config_dict)
        except (IndexError, TypeError):
            self.plot_grid_list(grid_list=grid)

    def plot_grid_list(self, grid_list: Union[List[Grid2D], List[Grid2DIrregular]]):
        """
         Plot an input list of grids of (y,x) coordinates using the matplotlib method `plt.line`.

        This method colors each grid in the list the same, so that the different grids are visible in the plot.

         This provides an alternative to `GridScatter.scatter_grid_list` where the plotted grids appear as lines
         instead of scattered points.

         Parameters
         ----------
         grid_list : Grid2DIrregular
             The list of grids of (y,x) coordinates that are plotted.
        """

        if len(grid_list) == 0:
            return None

        color = itertools.cycle(self.config_dict["c"])
        config_dict = self.config_dict
        config_dict.pop("c")

        try:
            # for critical curves/caustics, grid_list[0] corresponds to tangential curves
            # and grid_list[1] corresponds to radial curves
            for i in range(len(grid_list)):
                color_i = next(color)  # one color for all tangential curves and another color for all radial curves
                for grid in grid_list[i]:
                    plt.plot(grid[:, 1], grid[:, 0], c=color_i, **config_dict)
        except IndexError:
            pass
