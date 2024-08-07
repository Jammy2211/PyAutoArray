import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from typing import List, Optional

from autoconf import conf

from autoarray.plot.wrap.base.abstract import AbstractMatWrap
from autoarray.plot.wrap.base.units import Units

from autoarray import exc


class Colorbar(AbstractMatWrap):
    def __init__(
        self,
        manual_tick_labels: Optional[List[float]] = None,
        manual_tick_values: Optional[List[float]] = None,
        manual_alignment: Optional[str] = None,
        manual_unit: Optional[str] = None,
        manual_log10: bool = False,
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
        manual_alignment
            The vertical alignment of the colorbar tick labels, specified via the matplotlib method  `set_yticklabels`
            and input `va`.
        manual_unit
            The unit label that appears next to the colorbar tick labels, which if not input uses a default unit label
            specified as `cb_unit` in the config file `config/visualize/general.yaml.
        """

        super().__init__(**kwargs)

        self.manual_tick_labels = manual_tick_labels
        self.manual_tick_values = manual_tick_values
        self.manual_alignment = manual_alignment
        self.manual_unit = manual_unit
        self.manual_log10 = manual_log10

    @property
    def cb_unit(self):
        if self.manual_unit is None:
            return conf.instance["visualize"]["general"]["units"]["cb_unit"]
        return self.manual_unit

    def tick_values_from(self, norm=None, use_log10: bool = False):
        if (
            sum(
                x is not None
                for x in [self.manual_tick_values, self.manual_tick_labels]
            )
            == 1
        ):
            raise exc.PlottingException(
                "You can only manually specify the colorbar tick labels and values if both are input."
            )

        if self.manual_tick_values is not None:
            return self.manual_tick_values

        if norm is not None:
            min_value = norm.vmin
            max_value = norm.vmax

            if use_log10:
                if min_value < self.log10_min_value:
                    min_value = self.log10_min_value

                log_mid_value = (np.log10(max_value) + np.log10(min_value)) / 2.0
                mid_value = 10**log_mid_value

            else:
                mid_value = (max_value + min_value) / 2.0

            return [min_value, mid_value, max_value]

    def tick_labels_from(
        self,
        units: Units,
        manual_tick_values: List[float],
        cb_unit=None,
    ):
        if manual_tick_values is None:
            return None

        convert_factor = units.colorbar_convert_factor or 1.0

        if self.manual_tick_labels is not None:
            manual_tick_labels = self.manual_tick_labels
        else:
            manual_tick_labels = [
                np.round(value * convert_factor, 2) for value in manual_tick_values
            ]

        if self.manual_log10:
            manual_tick_labels = [
                "{:.0e}".format(label) for label in manual_tick_labels
            ]

            manual_tick_labels = [
                label.replace("1e", "$10^{") + "}$" for label in manual_tick_labels
            ]

            manual_tick_labels = [
                label.replace("{-0", "{-").replace("{+0", "{+").replace("+", "")
                for label in manual_tick_labels
            ]

        if units.colorbar_label is None:
            if cb_unit is None:
                cb_unit = self.cb_unit
        else:
            cb_unit = units.colorbar_label

        middle_index = (len(manual_tick_labels) - 1) // 2
        manual_tick_labels[
            middle_index
        ] = rf"{manual_tick_labels[middle_index]}{cb_unit}"

        return manual_tick_labels

    def set(
        self, units: Units, ax=None, norm=None, cb_unit=None, use_log10: bool = False
    ):
        """
        Set the figure's colorbar, optionally overriding the tick labels and values with manual inputs.
        """

        tick_values = self.tick_values_from(norm=norm, use_log10=use_log10)
        tick_labels = self.tick_labels_from(
            manual_tick_values=tick_values,
            units=units,
            cb_unit=cb_unit,
        )

        if tick_values is None and tick_labels is None:
            cb = plt.colorbar(ax=ax, **self.config_dict)
        else:
            cb = plt.colorbar(ticks=tick_values, ax=ax, **self.config_dict)
            cb.ax.set_yticklabels(
                labels=tick_labels, va=self.manual_alignment or "center"
            )

        return cb

    def set_with_color_values(
        self,
        units: Units,
        cmap: str,
        color_values: np.ndarray,
        ax=None,
        norm=None,
        use_log10: bool = False,
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

        mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
        mappable.set_array(color_values)

        tick_values = self.tick_values_from(norm=norm, use_log10=use_log10)
        tick_labels = self.tick_labels_from(
            manual_tick_values=tick_values,
            units=units,
        )

        if tick_values is None and tick_labels is None:
            cb = plt.colorbar(
                mappable=mappable,
                ax=ax,
                **self.config_dict,
            )
        else:
            cb = plt.colorbar(
                mappable=mappable,
                ax=ax,
                ticks=tick_values,
                **self.config_dict,
            )
            cb.ax.set_yticklabels(
                labels=tick_labels, va=self.manual_alignment or "center"
            )

        return cb
