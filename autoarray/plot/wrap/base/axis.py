import matplotlib.pyplot as plt
import numpy as np
from typing import List

from autoarray.plot.wrap.base.abstract import AbstractMatWrap


class Axis(AbstractMatWrap):
    def __init__(self, symmetric_source_centre: bool = False, **kwargs):
        """
        Customizes the axis of the plotted figure.

        This object wraps the following Matplotlib method:

        - plt.axis: https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.axis.html

        Parameters
        ----------
        symmetric_source_centre
            If `True`, the axis is symmetric around the centre of the plotted structure's coordinates.
        """

        super().__init__(**kwargs)

        self.symmetric_around_centre = symmetric_source_centre

    def set(self, extent: List[float] = None, grid=None):
        """
        Set the axis limits of the figure the grid is plotted on.

        Parameters
        ----------
        extent
            The extent of the figure which set the axis-limits on the figure the grid is plotted,
            following the format [xmin, xmax, ymin, ymax].
        """

        config_dict = self.config_dict
        extent_dict = config_dict.get("extent")

        if extent_dict is not None:
            config_dict.pop("extent")

        if self.symmetric_around_centre:
            ymin = np.min(grid[:, 0])
            ymax = np.max(grid[:, 0])
            xmin = np.min(grid[:, 1])
            xmax = np.max(grid[:, 1])

            x = np.max([np.abs(xmin), np.abs(xmax)])
            y = np.max([np.abs(ymin), np.abs(ymax)])

            extent_symmetric = [-x, x, -y, y]

            return plt.axis(extent_symmetric, **config_dict)

        else:
            if extent_dict is not None:
                return plt.axis(extent_dict, **config_dict)
            return plt.axis(extent, **config_dict)
