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

    def set(self, ax=None):
        """
        Set the figure's colorbar, optionally overriding the tick labels and values with manual inputs.
        """

        if self.manual_tick_values is None and self.manual_tick_labels is None:
            cb = plt.colorbar(**self.config_dict, ax=ax)
        elif (
            self.manual_tick_values is not None and self.manual_tick_labels is not None
        ):
            cb = plt.colorbar(ticks=self.manual_tick_values, ax=ax, **self.config_dict)
            cb.ax.set_yticklabels(
                labels=self.manual_tick_labels, va=self.manual_alignment or "center"
            )
        else:
            raise exc.PlottingException(
                "Only 1 entry of tick_values or tick_labels was input. You must either supply"
                "both the values and labels, or neither."
            )

        return cb

    def set_with_color_values(self, cmap: str, color_values: np.ndarray, ax=None):
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

        if self.manual_tick_values is None and self.manual_tick_labels is None:
            cb = plt.colorbar(mappable=mappable, ax=ax, **self.config_dict)
        elif (
            self.manual_tick_values is not None and self.manual_tick_labels is not None
        ):
            cb = plt.colorbar(
                mappable=mappable,
                ax=ax,
                ticks=self.manual_tick_values,
                **self.config_dict,
            )
            cb.ax.set_yticklabels(labels=self.manual_tick_labels, va=self.manual_alignment or "center")

        return cb
