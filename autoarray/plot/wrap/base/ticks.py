import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
from typing import List, Optional

from autoconf import conf

from autoarray.plot.wrap.base.abstract import AbstractMatWrap
from autoarray.plot.wrap.base.units import Units


class AbstractTicks(AbstractMatWrap):
    def __init__(
        self,
        manual_extent_factor: Optional[float] = None,
        manual_values: Optional[List[float]] = None,
        manual_units: Optional[str] = None,
        manual_suffix: Optional[str] = None,
        **kwargs,
    ):
        """
        The settings used to customize a figure's y and x ticks using the `YTicks` and `XTicks` objects.

        This object wraps the following Matplotlib methods:

        - plt.yticks: https://matplotlib.org/3.3.1/api/_as_gen/matplotlib.pyplot.yticks.html
        - plt.xticks: https://matplotlib.org/3.3.1/api/_as_gen/matplotlib.pyplot.xticks.html

        Parameters
        ----------
        manual_values
            Manually override the tick labels to display the labels as the input list of floats.
        manual_units
            Manually override the units in brackets of the tick label.
        manual_suffix
            A suffix applied to every tick label (e.g. for the suffix `kpc` 0.0 becomes 0.0kpc).
        """
        super().__init__(**kwargs)

        self.manual_extent_factor = manual_extent_factor
        self.manual_values = manual_values
        self.manual_units = manual_units
        self.manual_suffix = manual_suffix

    def tick_values_from(
        self, min_value: float, max_value: float, is_for_1d_plot: bool = False
    ) -> np.ndarray:
        """
        Calculate the ticks used for the yticks or xticks from input values of the minimum and maximum coordinate
        values of the y and x axis.

        Parameters
        ----------
        min_value
            the minimum value of the ticks that figure is plotted using.
        max_value
            the maximum value of the ticks that figure is plotted using.
        """

        if self.manual_values is not None:
            return self.manual_values

        center = max_value - ((max_value - min_value) / 2.0)

        if is_for_1d_plot:
            suffix = "_1d"
        else:
            suffix = "_2d"

        if self.manual_extent_factor is None:
            factor = conf.instance["visualize"][self.config_folder][
                self.__class__.__name__
            ]["manual"][f"extent_factor{suffix}"]

        number_of_ticks = conf.instance["visualize"][self.config_folder][
            self.__class__.__name__
        ]["manual"][f"number_of_ticks{suffix}"]

        value_0 = center - ((center - max_value)) * factor
        value_1 = center + ((min_value - center)) * factor

        return np.linspace(value_0, value_1, number_of_ticks)

    def tick_values_in_units_from(
        self, tick_values, units: Units, round_value: bool = True
    ) -> Optional[np.ndarray]:
        """
        Calculate the labels used for the yticks or xticks from input values of the minimum and maximum coordinate
        values of the y and x axis.

        The values are converted to the `Units` of the figure, via its conversion factor or using data properties.

        Parameters
        ----------
        array
            The array of data that is to be plotted, whose 2D shape is used to determine the tick values in units of
            pixels if this is the units specified by `units`.
        min_value
            the minimum value of the ticks that figure is plotted using.
        max_value
            the maximum value of the ticks that figure is plotted using.
        units
            The units the tick values are plotted using.
        axis
            Whether to use the y or x axis to estimate the tick labels.
        """

        if self.manual_values is not None:
            return np.asarray(self.manual_values)

        ticks_convert_factor = units.ticks_convert_factor or 1.0

        def signif(x, p):
            x = np.asarray(x)
            x_positive = np.where(np.isfinite(x) & (x != 0), np.abs(x), 10 ** (p - 1))
            mags = 10 ** (p - 1 - np.floor(np.log10(x_positive)))
            return np.round(x * mags) / mags

        if round_value:
            return np.asarray(
                [signif(value * ticks_convert_factor, 2) for value in tick_values]
            )
        return np.asarray([value * ticks_convert_factor for value in tick_values])

    def suffix_from(self, units: Units, yunit=None) -> Optional[str]:
        """
        Returns the label of an object, by determining it from the figure units if the label is not manually specified.

        Parameters
        ----------
        units
           The units of the data structure that is plotted which informs the appropriate label text.
        """

        if self.manual_suffix is not None:
            return self.manual_suffix

        if yunit is not None:
            return yunit

        if units.ticks_label is not None:
            return units.ticks_label

        units_conf = conf.instance["visualize"]["general"]["units"]

        if units is None:
            return units_conf["unscaled_symbol"]

        if units.use_scaled:
            return units_conf["scaled_symbol"]

        return units_conf["unscaled_symbol"]

    def labels_with_suffix_from(self, labels: List[str], suffix: str) -> List[str]:
        """
        The labels used for the y and x ticks can be append with a suffix.

        For example, if the labels were [-1.0, 0.0, 1.0] and the suffix is ", the labels with the suffix appended
        is [-1.0", 0.0", 1.0"].

        Parameters
        ----------
        labels
            The y and x labels which are append with the suffix.
        """

        labels = [str(label) for label in labels]

        all_end_0 = True

        for label in labels:
            if not label.endswith(".0"):
                all_end_0 = False

        if all_end_0:
            labels = [label[:-2] for label in labels]

        return [f"{label}{suffix}" for label in labels]


class YTicks(AbstractTicks):
    def set(
        self,
        min_value: float,
        max_value: float,
        units: Units,
        yunit=None,
        is_for_1d_plot: bool = False,
        is_log10: bool = False,
    ):
        """
        Set the y ticks of a figure using the shape of an input `Array2D` object and input units.

        Parameters
        ----------
        array
            The 2D array of data which is plotted.
        min_value
            the minimum value of the yticks that figure is plotted using.
        max_value
            the maximum value of the yticks that figure is plotted using.
        units
            The units of the figure.
        """

        if is_log10:
            if min_value < 0.001:
                min_value = 0.001

            max_value = 10 ** np.ceil(np.log10(max_value))
            number = int(abs(np.log10(max_value) - np.log10(min_value))) + 1
            ticks = np.logspace(np.log10(min_value), np.log10(max_value), number)

            plt.ylim(min_value, max_value)

            labels = self.tick_values_in_units_from(
                tick_values=ticks, units=units, round_value=False
            )
            labels = ["{:.0e}".format(label) for label in labels]

        else:
            ticks = self.tick_values_from(
                min_value=min_value, max_value=max_value, is_for_1d_plot=is_for_1d_plot
            )

            labels = self.tick_values_in_units_from(
                tick_values=ticks,
                units=units,
            )

        suffix = self.suffix_from(units=units, yunit=yunit)
        labels = self.labels_with_suffix_from(labels=labels, suffix=suffix)

        plt.yticks(ticks=ticks, labels=labels, **self.config_dict)

        if self.manual_units is not None:
            plt.gca().yaxis.set_major_formatter(
                FormatStrFormatter(f"{self.manual_units}")
            )


class XTicks(AbstractTicks):
    def set(
        self,
        min_value: float,
        max_value: float,
        units: Units,
        use_integers=False,
        is_for_1d_plot: bool = False,
    ):
        """
        Set the x ticks of a figure using the shape of an input `Array2D` object and input units.

        Parameters
        ----------
        array
            The 2D array of data which is plotted.
        min_value
            the minimum value of the xticks that figure is plotted using.
        max_value
            the maximum value of the xticks that figure is plotted using.
        units
            The units of the figure.
        """

        if use_integers:
            ticks = np.arange(int(max_value - min_value))
            labels = ticks

        else:
            ticks = self.tick_values_from(
                min_value=min_value, max_value=max_value, is_for_1d_plot=is_for_1d_plot
            )

            if not units.use_scaled:
                ticks = ticks.astype("int")

            labels = self.tick_values_in_units_from(
                tick_values=ticks,
                units=units,
            )

            if not units.use_scaled:
                labels = [f"{int(label)}" for label in labels]

        suffix = self.suffix_from(units=units)
        labels = self.labels_with_suffix_from(labels=labels, suffix=suffix)

        plt.xticks(ticks=ticks, labels=labels, **self.config_dict)

        if self.manual_units is not None:
            plt.gca().xaxis.set_major_formatter(
                FormatStrFormatter(f"{self.manual_units}")
            )
