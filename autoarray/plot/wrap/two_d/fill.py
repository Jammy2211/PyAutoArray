import logging

import matplotlib.pyplot as plt

from autoarray.plot.wrap.two_d.abstract import AbstractMatWrap2D


logger = logging.getLogger(__name__)


class Fill(AbstractMatWrap2D):
    def __init__(self, **kwargs):
        """
        The settings used to customize plots using fill on a figure

        This object wraps the following Matplotlib methods:

        - plt.fill https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.fill.html

        Parameters
        ----------
        symmetric
            If True, the colormap normalization (e.g. `vmin` and `vmax`) span the same absolute values producing a
            symmetric color bar.
        """

        super().__init__(**kwargs)

    def plot_fill(self, fill_region):

        try:
            y_fill = fill_region[:, 0]
            x_fill = fill_region[:, 1]
        except TypeError:
            y_fill = fill_region[0]
            x_fill = fill_region[1]

        plt.fill(x_fill, y_fill, **self.config_dict)