import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from typing import List, Optional

from autoarray.plot.wrap.base.abstract import AbstractMatWrap

from autoarray import exc


class Colorbar(AbstractMatWrap):
    def __init__(
        self,
        manual_tick_labels: Optional[List[float]] = None,
        manual_tick_values: Optional[List[float]] = None,
        manual_alignment: Optional[str] = None,
        **kwargs,
    ):
        """
        Customizes the colorbar of the plotted figure.

        This object wraps the following Matplotlib method:

        - plt.colorbar: https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.colorbar.html

        The colorbar object `cb` that is created is also customized using the following methods:

        - cb.set_yticklabels: https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.axes.Axes.set_yticklabels.html

        Parameters
        ----------
        manual_tick_labels
            Manually override the colorbar tick labels to an input list of float.
        manual_tick_values
            If the colorbar tick labels are manually specified the locations on the colorbar they appear running 0 -> 1.
        """

        super().__init__(**kwargs)

        self.manual_tick_labels = manual_tick_labels
        self.manual_tick_values = manual_tick_values
        self.manual_alignment = manual_alignment

    def set(self, ax=None, norm=None):
        """
        Set the figure's colorbar, optionally overriding the tick labels and values with manual inputs.
        """

        manual_tick_labels = self.manual_tick_labels
        manual_tick_values = self.manual_tick_values

        if sum(x is not None for x in [manual_tick_values, manual_tick_labels]) == 1:
            raise exc.PlottingException(
                "You can only manually specify the colorbar tick labels and values if both are input."
            )

        if (
            manual_tick_values is None
            and manual_tick_labels is None
            and norm is not None
        ):

            min_value = norm.vmin
            max_value = norm.vmax
            mid_value = (max_value + min_value) / 2.0

            manual_tick_values = [min_value, mid_value, max_value]
            manual_tick_labels = [np.round(value, 2) for value in manual_tick_values]

        if manual_tick_values is None and manual_tick_labels is None:
            cb = plt.colorbar(ax=ax, **self.config_dict)
        else:

            cb = plt.colorbar(ticks=manual_tick_values, ax=ax, **self.config_dict)
            cb.ax.set_yticklabels(
                labels=manual_tick_labels, va=self.manual_alignment or "center"
            )

        return cb

    def set_with_color_values(
        self, cmap: str, color_values: np.ndarray, ax=None, norm=None
    ):
        """
        Set the figure's colorbar using an array of already known color values.

        This method is used for producing the color bar on a Voronoi mesh plot, which is unable to use the in-built
        Matplotlib colorbar method.

        Parameters
        ----------
        cmap
            The colormap used to map normalized data values to RGBA
            colors (see https://matplotlib.org/3.3.2/api/cm_api.html).
        color_values
            The values of the pixels on the Voronoi mesh which are used to create the colorbar.
        """

        mappable = cm.ScalarMappable(cmap=cmap)
        mappable.set_array(color_values)

        manual_tick_labels = self.manual_tick_labels
        manual_tick_values = self.manual_tick_values

        if sum(x is not None for x in [manual_tick_values, manual_tick_labels]) == 1:
            raise exc.PlottingException(
                "You can only manually specify the colorbar tick labels and values if both are input."
            )

        if (
            manual_tick_values is None
            and manual_tick_labels is None
            and norm is not None
        ):

            min_value = norm.vmin
            max_value = norm.vmax
            mid_value = (max_value + min_value) / 2.0

            manual_tick_values = [min_value, mid_value, max_value]
            manual_tick_labels = [np.round(value, 2) for value in manual_tick_values]

        if manual_tick_values is None and manual_tick_labels is None:
            cb = plt.colorbar(
                mappable=mappable,
                ax=ax,
                **self.config_dict,
            )
        else:
            cb = plt.colorbar(
                mappable=mappable,
                ax=ax,
                ticks=manual_tick_values,
                **self.config_dict,
            )
            cb.ax.set_yticklabels(
                labels=manual_tick_labels, va=self.manual_alignment or "center"
            )

        return cb
