import matplotlib.pyplot as plt
import numpy as np
import itertools
from typing import List, Union, Optional

from autoarray.plot.wrap.two_d.abstract import AbstractMatWrap2D
from autoarray.structures.grids.uniform_2d import Grid2D
from autoarray.structures.grids.irregular_2d import Grid2DIrregular


class GridErrorbar(AbstractMatWrap2D):
    """
    Plots an input set of grid points with 2D errors, for example (y,x) coordinates or data structures representing 2D
    (y,x) coordinates like a `Grid2D` or `Grid2DIrregular`. Multiple lists of (y,x) coordinates are plotted with
    varying colors.

    This object wraps the following Matplotlib methods:

    - plt.errorbar: https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.errorbar.html

    Parameters
    ----------
    colors : [str]
        The color or list of colors that the grid is plotted using. For plotting indexes or a grid list, a
        list of colors can be specified which the plot cycles through.
    """

    def config_dict_remove_marker(self, config_dict):

        if config_dict.get("fmt") and config_dict.get("marker"):
            config_dict.pop("marker")

        return config_dict

    def errorbar_grid(
        self,
        grid: Union[np.ndarray, Grid2D],
        y_errors: Optional[Union[np.ndarray, List]] = None,
        x_errors: Optional[Union[np.ndarray, List]] = None,
    ):
        """
        Plot an input grid of (y,x) coordinates using the matplotlib method `plt.errorbar`.

        The (y,x) coordinates are plotted as dots, with a line / cross for its errors.

        Parameters
        ----------
        grid : Grid2D
            The grid of (y,x) coordinates that is plotted.
        y_errors
            The y values of the error on every point of the grid that is plotted (e.g. vertically).
        x_errors
            The x values of the error on every point of the grid that is plotted (e.g. horizontally).
        """

        config_dict = self.config_dict

        if len(config_dict["c"]) > 1:
            config_dict["c"] = config_dict["c"][0]

        config_dict = self.config_dict_remove_marker(config_dict=config_dict)

        try:
            plt.errorbar(
                y=grid[:, 0], x=grid[:, 1], yerr=y_errors, xerr=x_errors, **config_dict
            )
        except (IndexError, TypeError):
            return self.errorbar_grid_list(grid_list=grid)

    def errorbar_grid_list(
        self,
        grid_list: Union[List[Grid2D], List[Grid2DIrregular]],
        y_errors: Optional[Union[np.ndarray, List]] = None,
        x_errors: Optional[Union[np.ndarray, List]] = None,
    ):
        """
        Plot an input list of grids of (y,x) coordinates using the matplotlib method `plt.errorbar`.

        The (y,x) coordinates are plotted as dots, with a line / cross for its errors.

        This method colors each grid in each entry of the list the same, so that the different grids are visible in
        the plot.

        Parameters
        ----------
        grid_list
            The list of grids of (y,x) coordinates that are plotted.
        """
        if len(grid_list) == 0:
            return

        color = itertools.cycle(self.config_dict["c"])
        config_dict = self.config_dict
        config_dict.pop("c")

        config_dict = self.config_dict_remove_marker(config_dict=config_dict)

        try:
            for grid in grid_list:
                plt.errorbar(
                    y=grid[:, 0],
                    x=grid[:, 1],
                    yerr=np.asarray(y_errors),
                    xerr=np.asarray(x_errors),
                    c=next(color),
                    **config_dict,
                )
        except IndexError:
            return None

    def errorbar_grid_colored(
        self,
        grid: Union[np.ndarray, Grid2D],
        color_array: np.ndarray,
        cmap: str,
        y_errors: Optional[Union[np.ndarray, List]] = None,
        x_errors: Optional[Union[np.ndarray, List]] = None,
    ):
        """
        Plot an input grid of (y,x) coordinates using the matplotlib method `plt.errorbar`.

        The method colors the errorbared grid according to an input ndarray of color values, using an input colormap.

        Parameters
        ----------
        grid : Grid2D
            The grid of (y,x) coordinates that is plotted.
        color_array : ndarray
            The array of RGB color values used to color the grid.
        cmap
            The Matplotlib colormap used for the grid point coloring.
        """

        config_dict = self.config_dict
        config_dict.pop("c")

        plt.scatter(y=grid[:, 0], x=grid[:, 1], c=color_array, cmap=cmap)

        config_dict = self.config_dict_remove_marker(config_dict=self.config_dict)

        plt.errorbar(
            y=grid[:, 0],
            x=grid[:, 1],
            yerr=np.asarray(y_errors),
            xerr=np.asarray(x_errors),
            zorder=0.0,
            **config_dict,
        )
