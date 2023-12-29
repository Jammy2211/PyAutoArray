import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional, Union

from autoarray.plot.wrap.two_d.abstract import AbstractMatWrap2D
from autoarray.structures.arrays.uniform_2d import Array2D


class Contour(AbstractMatWrap2D):

    def __init__(
        self,
        manual_levels: Optional[List[float]] = None,
        total_contours : int = 10,
        use_log10 : bool = True,
        **kwargs,
    ):
        """
        Customizes the contours of the plotted figure.

        This object wraps the following Matplotlib method:

        - plt.contours: https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.contours.html

        Parameters
        ----------
        manual_levels
            Manually override the levels at which the contours are plotted.
        total_contours
            The total number of contours plotted, which also determines the spacing between each contour.
        use_log10
            Whether the contours are plotted with a log10 spacing between each contour (alternative is linear).
        """

        super().__init__(**kwargs)

        self.manual_levels = manual_levels
        self.total_contours = total_contours
        self.use_log10 = use_log10

    def levels_from(self, array : Union[np.ndarray, Array2D]) -> Union[np.ndarray, List[float]]:
        """
        The levels at which the contours are plotted, which may be determined in the following ways:

        - Automatically computed from the minimum and maximum values of the array, using a log10 or linear spacing.
        - Overriden by the input `manual_levels` (e.g. if it is not None).

        Returns
        -------
        The levels at which the contours are plotted.
        """
        if self.manual_levels is None:

            if self.use_log10:
                return np.logspace(np.log10(np.min(array)), np.log10(np.max(array)), self.total_contours)
            return np.linspace(np.min(array), np.max(array), self.total_contours)

        return self.manual_levels

    def set(self, array: Union[np.ndarray, Array2D], extent : List[float] = None):
        """
        Plot an input grid of (y,x) coordinates using the matplotlib method `plt.scatter`.

        Parameters
        ----------
        array
            The array of values the contours are plotted over.
        """
        plt.contour(
            array.native[::-1],
            levels=self.levels_from(array),
            extent=extent,
            **self.config_dict
        )