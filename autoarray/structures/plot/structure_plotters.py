import numpy as np
from typing import List, Union

from autoarray.plot.abstract_plotters import Plotter
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


class Array2DPlotter(Plotter):
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
    def get_visuals_2d(self) -> Visuals2D:
        return self.get_2d.via_mask_from(mask=self.array.mask)

    def figure_2d(self):

        self.mat_plot_2d.plot_array(
            array=self.array,
            visuals_2d=self.get_visuals_2d,
            auto_labels=AutoLabels(title="Array2D", filename="array"),
        )


class Grid2DPlotter(Plotter):
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
    def get_visuals_2d(self) -> Visuals2D:
        return self.get_2d.via_grid_from(grid=self.grid)

    def figure_2d(self, color_array: np.ndarray = None):

        self.mat_plot_2d.plot_grid(
            grid=self.grid,
            visuals_2d=self.get_visuals_2d,
            auto_labels=AutoLabels(title="Grid2D", filename="grid"),
            color_array=color_array,
        )


class YX1DPlotter(Plotter):
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
    def get_visuals_1d(self) -> Visuals1D:
        return self.get_1d.via_array_1d_from(array_1d=self.x)

    def figure_1d(self):

        self.mat_plot_1d.plot_yx(
            y=self.y, x=self.x, visuals_1d=self.get_visuals_1d, auto_labels=AutoLabels()
        )
