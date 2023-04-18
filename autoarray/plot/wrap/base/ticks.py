import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
from typing import List, Optional

from autoarray.plot.wrap.base.abstract import AbstractMatWrap
from autoarray.plot.wrap.base.units import Units
from autoarray.structures.arrays.uniform_2d import Array2D

from autoarray import exc


class AbstractTicks(AbstractMatWrap):
    def __init__(
        self,
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

        self.manual_values = manual_values
        self.manual_units = manual_units
        self.manual_suffix = manual_suffix

    def tick_values_from(self, min_value: float, max_value: float) -> np.ndarray:
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
        return np.linspace(min_value, max_value, 5)

    def tick_values_in_units_from(
        self,
        array: Array2D,
        min_value: float,
        max_value: float,
        units: Units,
        axis: int,
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
        elif not units.use_scaled:
            try:
                return np.linspace(0, array.shape_native[axis], 5).astype("int")
            except IndexError:
                return np.linspace(0, array.shape_native[0], 5).astype("int")
            except AttributeError:
                try:
                    return np.linspace(0, np.asarray(array).shape[axis], 5).astype(
                        "int"
                    )
                except IndexError:
                    return np.linspace(0, np.asarray(array).shape[0], 5).astype("int")
        elif (units.use_scaled and units.conversion_factor is None) or not units.in_kpc:
            return np.round(np.linspace(min_value, max_value, 5), 2)
        elif units.use_scaled and units.conversion_factor is not None:
            return np.round(
                np.linspace(
                    min_value * units.conversion_factor,
                    max_value * units.conversion_factor,
                    5,
                ),
                2,
            )

        else:
            raise exc.PlottingException(
                "The tick labels cannot be computed using the input options."
            )

    def suffix_from(self, units: Units) -> Optional[str]:
        """
        Returns the label of an object, by determining it from the figure units if the label is not manually specified.

        Parameters
        ----------
        units
           The units of the data structure that is plotted which informs the appropriate label text.
        """

        if self.manual_suffix is not None:
            return self.manual_suffix

        if units is None:
            return ""

        if units.in_kpc is not None and units.use_scaled:
            if units.in_kpc:
                return "kpc"
            return '"'

        return ""

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
        return [f"{label}{suffix}" for label in labels]


class YTicks(AbstractTicks):
    def set(
        self,
        array: Optional[Array2D],
        min_value: float,
        max_value: float,
        units: Units,
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

        ticks = self.tick_values_from(min_value=min_value, max_value=max_value)
        labels = self.tick_values_in_units_from(
            array=array, min_value=min_value, max_value=max_value, units=units, axis=0
        )
        suffix = self.suffix_from(units=units)
        labels = self.labels_with_suffix_from(labels=labels, suffix=suffix)

        plt.yticks(ticks=ticks, labels=labels, **self.config_dict)

        if self.manual_units is not None:
            plt.gca().yaxis.set_major_formatter(
                FormatStrFormatter(f"{self.manual_units}")
            )

        if not units.use_scaled:
            plt.gca().invert_yaxis()


class XTicks(AbstractTicks):
    def set(
        self,
        array: Optional[Array2D],
        min_value: float,
        max_value: float,
        units: Units,
        use_integers=False,
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

            ticks = self.tick_values_from(min_value=min_value, max_value=max_value)

            labels = self.tick_values_in_units_from(
                array=array,
                min_value=min_value,
                max_value=max_value,
                units=units,
                axis=1,
            )

        suffix = self.suffix_from(units=units)
        labels = self.labels_with_suffix_from(labels=labels, suffix=suffix)

        plt.xticks(ticks=ticks, labels=labels, **self.config_dict)

        if self.manual_units is not None:
            plt.gca().xaxis.set_major_formatter(
                FormatStrFormatter(f"{self.manual_units}")
            )
