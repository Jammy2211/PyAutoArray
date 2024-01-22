import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
from typing import List, Tuple, Optional

from autoconf import conf

from autoarray.plot.wrap.base.abstract import AbstractMatWrap
from autoarray.plot.wrap.base.units import Units


class TickMaker:
    def __init__(
        self,
        min_value: float,
        max_value: float,
        factor: float,
        number_of_ticks: int,
        units,
    ):
        self.min_value = min_value
        self.max_value = max_value
        self.factor = factor
        self.number_of_ticks = number_of_ticks
        self.units = units

    @property
    def centre(self):
        return self.max_value - ((self.max_value - self.min_value) / 2.0)

    @property
    def tick_values_linear(self):
        value_0 = self.centre - ((self.centre - self.max_value)) * self.factor
        value_1 = self.centre + ((self.min_value - self.centre)) * self.factor

        return np.linspace(value_0, value_1, self.number_of_ticks)

    @property
    def tick_values_log10(self):
        min_value = self.min_value

        if self.min_value < 0.001:
            min_value = 0.001

        min_value = 10 ** np.floor(np.log10(min_value))

        max_value = 10 ** np.ceil(np.log10(self.max_value))
        number = int(abs(np.log10(max_value) - np.log10(min_value))) + 1

        return np.logspace(np.log10(min_value), np.log10(max_value), number)

    @property
    def tick_values_integers(self):
        ticks = np.arange(int(self.max_value - self.min_value))

        if not self.units.use_scaled:
            ticks = ticks.astype("int")

        return ticks


class LabelMaker:
    def __init__(
        self,
        tick_values,
        min_value: float,
        max_value: float,
        units,
        pixels: Optional[int] = None,
        round_sf: int = 2,
        yunit=None,
        xunit=None,
        manual_suffix=None,
    ):
        self.tick_values = tick_values
        self.min_value = min_value
        self.max_value = max_value
        self.units = units
        self.pixels = pixels
        self.convert_factor = self.units.ticks_convert_factor or 1.0
        self.yunit = yunit
        self.xunit = xunit
        self.round_sf = round_sf
        self.manual_suffix = manual_suffix

    @property
    def suffix(self) -> Optional[str]:
        """
        Returns the label of an object, by determining it from the figure units if the label is not manually specified.

        Parameters
        ----------
        units
           The units of the data structure that is plotted which informs the appropriate label text.
        """

        if self.manual_suffix is not None:
            return self.manual_suffix

        if self.yunit is not None:
            return self.yunit

        if self.xunit is not None:
            return self.xunit

        if self.units.ticks_label is not None:
            return self.units.ticks_label

        units_conf = conf.instance["visualize"]["general"]["units"]

        if self.units is None:
            return units_conf["unscaled_symbol"]

        if self.units.use_scaled:
            return units_conf["scaled_symbol"]

        return units_conf["unscaled_symbol"]

    @property
    def span(self):
        return self.max_value - self.min_value

    @property
    def tick_values_rounded(self):
        values = np.asarray(self.tick_values) * self.convert_factor
        values_positive = np.where(
            np.isfinite(values) & (values != 0),
            np.abs(values),
            10 ** (self.round_sf - 1),
        )
        mags = 10 ** (self.round_sf - 1 - np.floor(np.log10(values_positive)))
        return np.round(values * mags) / mags

    @property
    def labels_linear(self):
        if self.units.use_raw:
            return self.with_appended_suffix(self.tick_values_rounded)

        if not self.units.use_scaled and self.yunit is None:
            return self.labels_linear_pixels

        labels = np.asarray([value for value in self.tick_values_rounded])

        if not self.units.use_scaled and self.yunit is None:
            labels = [f"{int(label)}" for label in labels]
        return self.with_appended_suffix(labels)

    @property
    def labels_linear_pixels(self):
        if self.max_value == self.min_value:
            labels = [f"{int(label)}" for label in self.tick_values]
            return self.with_appended_suffix(labels)

        ticks_from_zero = [tick - self.min_value for tick in self.tick_values]
        labels = [(tick / self.span) * self.pixels for tick in ticks_from_zero]

        labels = [f"{int(label)}" for label in labels]

        return self.with_appended_suffix(labels)

    @property
    def labels_log10(self):
        labels = ["{:.0e}".format(label) for label in self.tick_values]
        labels = [label.replace("1e", "$10^{") + "}$" for label in labels]
        labels = [
            label.replace("{-0", "{-").replace("{+0", "{+").replace("+", "")
            for label in labels
        ]
        #     labels = [label.replace("1e", "").replace("-0", "-").replace("+0", "+").replace("+0", "0") for label in labels]

        return self.with_appended_suffix(labels)

    def with_appended_suffix(self, labels):
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

        return [f"{label}{self.suffix}" for label in labels]


class AbstractTicks(AbstractMatWrap):
    def __init__(
        self,
        manual_factor: Optional[float] = None,
        manual_values: Optional[List[float]] = None,
        manual_min_max_value: Optional[Tuple[float, float]] = None,
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

        self.manual_factor = manual_factor
        self.manual_values = manual_values
        self.manual_min_max_value = manual_min_max_value
        self.manual_units = manual_units
        self.manual_suffix = manual_suffix

    def factor_from(self, suffix):
        if self.manual_factor is not None:
            return self.manual_factor
        return conf.instance["visualize"][self.config_folder][self.__class__.__name__][
            "manual"
        ][f"extent_factor{suffix}"]

    def number_of_ticks_from(self, suffix):
        return conf.instance["visualize"][self.config_folder][self.__class__.__name__][
            "manual"
        ][f"number_of_ticks{suffix}"]

    def tick_maker_from(
        self, min_value: float, max_value: float, units, is_for_1d_plot: bool
    ):
        suffix = "_1d" if is_for_1d_plot else "_2d"

        factor = self.factor_from(suffix=suffix)
        number_of_ticks = self.number_of_ticks_from(suffix=suffix)

        return TickMaker(
            min_value=min_value,
            max_value=max_value,
            factor=factor,
            units=units,
            number_of_ticks=number_of_ticks,
        )

    def ticks_from(
        self,
        min_value: float,
        max_value: float,
        units: Units,
        is_log10: bool = False,
        is_for_1d_plot: bool = False,
    ):
        tick_maker = self.tick_maker_from(
            min_value=min_value,
            max_value=max_value,
            units=units,
            is_for_1d_plot=is_for_1d_plot,
        )

        if self.manual_values:
            return self.manual_values
        elif is_log10:
            return tick_maker.tick_values_log10
        return tick_maker.tick_values_linear

    def labels_from(
        self,
        ticks,
        min_value: float,
        max_value: float,
        units,
        yunit,
        xunit,
        pixels: Optional[int] = None,
        is_log10: bool = False,
    ):
        label_maker = LabelMaker(
            tick_values=ticks,
            min_value=min_value,
            max_value=max_value,
            units=units,
            pixels=pixels,
            yunit=yunit,
            xunit=xunit,
            manual_suffix=self.manual_suffix,
        )

        if self.manual_units:
            return ticks
        elif is_log10:
            return label_maker.labels_log10
        return label_maker.labels_linear

    def ticks_and_labels_from(
        self,
        min_value,
        max_value,
        units,
        pixels: Optional[int] = None,
        use_integers: bool = False,
        yunit=None,
        xunit=None,
        is_log10: bool = False,
        is_for_1d_plot: bool = False,
    ):
        if use_integers:
            ticks = np.arange(int(max_value - min_value))
            return ticks, ticks

        ticks = self.ticks_from(
            min_value=min_value,
            max_value=max_value,
            units=units,
            is_log10=is_log10,
            is_for_1d_plot=is_for_1d_plot,
        )

        labels = self.labels_from(
            ticks=ticks,
            min_value=min_value,
            max_value=max_value,
            units=units,
            yunit=yunit,
            xunit=xunit,
            pixels=pixels,
            is_log10=is_log10,
        )
        return ticks, labels


class YTicks(AbstractTicks):
    def set(
        self,
        min_value: float,
        max_value: float,
        units: Units,
        pixels: Optional[int] = None,
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

        if self.manual_min_max_value:
            min_value = self.manual_min_max_value[0]
            max_value = self.manual_min_max_value[1]

        ticks, labels = self.ticks_and_labels_from(
            min_value=min_value,
            max_value=max_value,
            units=units,
            pixels=pixels,
            yunit=yunit,
            is_log10=is_log10,
            is_for_1d_plot=is_for_1d_plot,
        )

        if is_log10:
            plt.ylim(max(min_value, self.log10_min_value), max_value)

        if not is_for_1d_plot and not units.use_scaled:
            labels = reversed(labels)

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
        pixels: Optional[int] = None,
        xunit=None,
        use_integers=False,
        is_for_1d_plot: bool = False,
        is_log10: bool = False,
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

        if self.manual_min_max_value:
            min_value = self.manual_min_max_value[0]
            max_value = self.manual_min_max_value[1]

        ticks, labels = self.ticks_and_labels_from(
            min_value=min_value,
            max_value=max_value,
            pixels=pixels,
            units=units,
            yunit=xunit,
            use_integers=use_integers,
            is_for_1d_plot=is_for_1d_plot,
            is_log10=is_log10,
        )

        plt.xticks(ticks=ticks, labels=labels, **self.config_dict)

        if self.manual_units is not None:
            plt.gca().xaxis.set_major_formatter(
                FormatStrFormatter(f"{self.manual_units}")
            )
