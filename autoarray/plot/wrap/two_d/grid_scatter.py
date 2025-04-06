import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
import itertools
from scipy.spatial import ConvexHull
from typing import List, Union


from autoarray.plot.wrap.two_d.abstract import AbstractMatWrap2D
from autoarray.structures.grids.uniform_2d import Grid2D
from autoarray.structures.grids.irregular_2d import Grid2DIrregular
from autoarray.structures.mesh.abstract_2d import Abstract2DMesh


class GridScatter(AbstractMatWrap2D):
    """
    Scatters an input set of grid points, for example (y,x) coordinates or data structures representing 2D (y,x)
    coordinates like a `Grid2D` or `Grid2DIrregular`. List of (y,x) coordinates are plotted with varying colors.

    This object wraps the following Matplotlib methods:

    - plt.scatter: https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.scatter.html

    There are a number of children of this method in the `mat_obj.py` module that plot specific sets of (y,x)
    points. Each of these objects uses uses their own config file and settings so that each has a unique appearance
    on every figure:

    - `OriginScatter`: plots the (y,x) coordinates of the origin of a data structure (e.g. as a black cross).
    - `MaskScatter`: plots a mask over an image, using the `Mask2d` object's (y,x)  `edge` property.
    - `BorderScatter: plots a border over an image, using the `Mask2d` object's (y,x) `border` property.
    - `PositionsScatter`: plots the (y,x) coordinates that are input in a plotter via the `positions` input.
    - `IndexScatter`: plots specific (y,x) coordinates of a grid (or grids) via their 1d or 2d indexes.
    - `MeshGridScatter`: plots the grid of a `Mesh` object (see `autoarray.inversion`).

    Parameters
    ----------
    colors : [str]
        The color or list of colors that the grid is plotted using. For plotting indexes or a grid list, a
        list of colors can be specified which the plot cycles through.
    """

    def scatter_grid(self, grid: Union[np.ndarray, Grid2D]):
        """
        Plot an input grid of (y,x) coordinates using the matplotlib method `plt.scatter`.

        Parameters
        ----------
        grid : Grid2D
            The grid of (y,x) coordinates that is plotted.
        errors
            The error on every point of the grid that is plotted.
        """

        config_dict = self.config_dict

        if len(config_dict["c"]) > 1:
            config_dict["c"] = config_dict["c"][0]

        try:
            plt.scatter(y=grid[:, 0], x=grid[:, 1], **config_dict)
        except (IndexError, TypeError):
            return self.scatter_grid_list(grid_list=grid)

    def scatter_grid_list(self, grid_list: Union[List[Grid2D], List[Grid2DIrregular]]):
        """
        Plot an input list of grids of (y,x) coordinates using the matplotlib method `plt.scatter`.

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

        try:
            for grid in grid_list:
                plt.scatter(y=grid[:, 0], x=grid[:, 1], c=next(color), **config_dict)
        except IndexError:
            return None

    def scatter_grid_colored(
        self, grid: Union[np.ndarray, Grid2D], color_array: np.ndarray, cmap: str
    ):
        """
        Plot an input grid of (y,x) coordinates using the matplotlib method `plt.scatter`.

        The method colors the scattered grid according to an input ndarray of color values, using an input colormap.

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

        plt.scatter(y=grid[:, 0], x=grid[:, 1], c=color_array, cmap=cmap, **config_dict)

    def scatter_grid_indexes(
        self,
        grid: Union[np.ndarray, Grid2D, Grid2DIrregular],
        indexes: np.ndarray,
    ):
        """
        Plot specific points of an input grid of (y,x) coordinates, which are specified according to the 1D or 2D
        indexes of the `Grid2D`.

        This method allows us to color in points on grids that map between one another.

        Parameters
        ----------
        grid : Grid2D
            The grid of (y,x) coordinates that is plotted.
        indexes
            The 1D indexes of the grid that are colored in when plotted.
        """
        color = itertools.cycle(self.config_dict["c"])
        config_dict = self.config_dict
        config_dict.pop("c")

        for index_list in indexes:
            plt.scatter(
                y=grid[index_list, 0],
                x=grid[index_list, 1],
                color=next(color),
                **config_dict,
            )
