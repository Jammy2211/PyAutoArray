import matplotlib.pyplot as plt
import numpy as np
from typing import List, Union

from autoarray.plot.wrap.two_d.abstract import AbstractMatWrap2D
from autoarray.structures.arrays.uniform_2d import Array2D


class Contour(AbstractMatWrap2D):
    """
    """

    def contour(self, array: Union[np.ndarray, Array2D], extent : List[float] = None):
        """
        Plot an input grid of (y,x) coordinates using the matplotlib method `plt.scatter`.

        Parameters
        ----------
        array
            The array of values the contours are plotted over.
        """

        # levels = np.logspace(np.log10(0.3), np.log10(20.0), 10)

        levels = np.logspace(np.log10(np.min(array)), np.log10(np.max(array)), 10)

        plt.contour(
            #  array.mask.derive_grid.unmasked_sub_1,
            array.native[::-1],
            levels=levels,
            colors="black",
            extent=extent,
            **self.config_dict
        )