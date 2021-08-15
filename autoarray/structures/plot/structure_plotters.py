import numpy as np
from typing import List, Union

from autoarray.plot.abstract_plotters import AbstractPlotter
from autoarray.plot.mat_wrap.visuals import Visuals1D
from autoarray.plot.mat_wrap.visuals import Visuals2D
from autoarray.plot.mat_wrap.include import Include1D
from autoarray.plot.mat_wrap.include import Include2D
from autoarray.plot.mat_wrap.mat_plot import MatPlot1D
from autoarray.plot.mat_wrap.mat_plot import MatPlot2D
from autoarray.plot.mat_wrap.mat_plot import AutoLabels
from autoarray.structures.arrays.one_d.array_1d import Array1D
from autoarray.structures.arrays.two_d.array_2d import Array2D
from autoarray.structures.grids.two_d.grid_2d import Grid2D
from autoarray.structures.grids.two_d.grid_2d_irregular import Grid2DIrregular


class Array2DPlotter(AbstractPlotter):
    def __init__(
        self,
        array: Array2D,
        mat_plot_2d: MatPlot2D = MatPlot2D(),
        visuals_2d: Visuals2D = Visuals2D(),
        include_2d: Include2D = Include2D(),
    ):

        super().__init__(
            visuals_2d=visuals_2d, include_2d=include_2d, mat_plot_2d=mat_plot_2d
        )

        self.array = array

    @property
    def visuals_with_include_2d(self) -> Visuals2D:
        """
        Extracts from an `Array2D` attributes that can be plotted and returns them in a `Visuals` object.

        Only attributes already in `self.visuals_2d` or with `True` entries in the `Include` object are extracted
        for plotting.

        From an `Array2D` the following attributes can be extracted for plotting:

        - origin: the (y,x) origin of the structure's coordinate system.
        - mask: the mask of the structure.
        - border: the border of the structure's mask.

        Parameters
        ----------
        array : Array2D
            The array whose attributes are extracted for plotting.

        Returns
        -------
        Visuals2D
            The collection of attributes that can be plotted by a `Plotter2D` object.
        """
        return self.visuals_2d + self.visuals_2d.__class__(
            origin=self.extract_2d("origin", Grid2DIrregular(grid=[self.array.origin])),
            mask=self.extract_2d("mask", self.array.mask),
            border=self.extract_2d("border", self.array.mask.border_grid_sub_1.binned),
        )

    def figure_2d(self):

        self.mat_plot_2d.plot_array(
            array=self.array,
            visuals_2d=self.visuals_with_include_2d,
            auto_labels=AutoLabels(title="Array2D", filename="array"),
        )


class Grid2DPlotter(AbstractPlotter):
    def __init__(
        self,
        grid: Grid2D,
        mat_plot_2d: MatPlot2D = MatPlot2D(),
        visuals_2d: Visuals2D = Visuals2D(),
        include_2d: Include2D = Include2D(),
    ):
        super().__init__(
            visuals_2d=visuals_2d, include_2d=include_2d, mat_plot_2d=mat_plot_2d
        )

        self.grid = grid

    @property
    def visuals_with_include_2d(self) -> Visuals2D:
        """
        Extracts from a `Grid2D` attributes that can be plotted and return them in a `Visuals` object.

        Only attributes with `True` entries in the `Include` object are extracted for plotting.

        From a `Grid2D` the following attributes can be extracted for plotting:

        - origin: the (y,x) origin of the grid's coordinate system.
        - mask: the mask of the grid.
        - border: the border of the grid's mask.

        Parameters
        ----------
        grid : abstract_grid_2d.AbstractGrid2D
            The grid whose attributes are extracted for plotting.

        Returns
        -------
        Visuals2D
            The collection of attributes that can be plotted by a `Plotter2D` object.
        """
        if not isinstance(self.grid, Grid2D):
            return self.visuals_2d

        return self.visuals_2d + self.visuals_2d.__class__(
            origin=self.extract_2d("origin", Grid2DIrregular(grid=[self.grid.origin]))
        )

    def figure_2d(self, color_array: np.ndarray = None):

        self.mat_plot_2d.plot_grid(
            grid=self.grid,
            visuals_2d=self.visuals_with_include_2d,
            auto_labels=AutoLabels(title="Grid2D", filename="grid"),
            color_array=color_array,
        )


class YX1DPlotter(AbstractPlotter):
    def __init__(
        self,
        y: Union[np.ndarray, List, Array1D],
        x: Union[np.ndarray, List, Array1D],
        mat_plot_1d: MatPlot1D = MatPlot1D(),
        visuals_1d: Visuals1D = Visuals1D(),
        include_1d: Include1D = Include1D(),
    ):

        super().__init__(
            visuals_1d=visuals_1d, include_1d=include_1d, mat_plot_1d=mat_plot_1d
        )

        self.y = y
        self.x = x

    @property
    def visuals_with_include_1d(self) -> Visuals1D:

        return self.visuals_1d + self.visuals_1d.__class__(
            origin=self.extract_1d("origin", self.x.origin),
            mask=self.extract_1d("mask", self.x.mask),
        )

    def figure_1d(self):

        self.mat_plot_1d.plot_yx(
            y=self.y, x=self.x, visuals_1d=self.visuals_1d, auto_labels=AutoLabels()
        )
