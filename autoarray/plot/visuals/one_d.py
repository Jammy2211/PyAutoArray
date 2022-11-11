import numpy as np
from typing import List, Optional, Union

from autoarray.mask.mask_1d import Mask1D
from autoarray.plot.include.one_d import Include1D
from autoarray.plot.visuals.abstract import AbstractVisuals
from autoarray.structures.arrays.uniform_1d import Array1D
from autoarray.structures.grids.uniform_1d import Grid1D


class Visuals1D(AbstractVisuals):
    def __init__(
        self,
        origin: Optional[Grid1D] = None,
        mask: Optional[Mask1D] = None,
        points: Optional[Grid1D] = None,
        vertical_line: Optional[float] = None,
        shaded_region: Optional[List[Union[List, Array1D, np.ndarray]]] = None,
    ):

        self.origin = origin
        self.mask = mask
        self.points = points
        self.vertical_line = vertical_line
        self.shaded_region = shaded_region

    @property
    def include(self):
        return Include1D()

    def plot_via_plotter(self, plotter):

        if self.points is not None:

            plotter.yx_scatter.scatter_yx(y=self.points, x=np.arange(len(self.points)))

        if self.vertical_line is not None:

            plotter.vertical_line_axvline.axvline_vertical_line(
                vertical_line=self.vertical_line
            )
