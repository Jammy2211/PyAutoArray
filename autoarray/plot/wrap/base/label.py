import matplotlib.pyplot as plt
from typing import Optional

from autoarray.plot.wrap.base.abstract import AbstractMatWrap
from autoarray.plot.wrap.base.units import Units


class AbstractLabel(AbstractMatWrap):
    def __init__(self, **kwargs):
        """
        The settings used to customize the figure's title and y and x labels.

        This object wraps the following Matplotlib methods:

        - plt.ylabel: https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.ylabel.html
        - plt.xlabel: https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.xlabel.html

        The y and x labels will automatically be set if not specified, using the input units.

        Parameters
        ----------
        units
            The units the data is plotted using.
        manual_label
            A manual label which overrides the default computed via the units if input.
        """

        super().__init__(**kwargs)

        self.manual_label = self.kwargs.get("label")

    def label_from(self, units: Units) -> Optional[str]:
        """
        Returns the label of an object, by determining it from the figure units if the label is not manually specified.

        Parameters
        ----------
        units
           The units of the data structure that is plotted which informs the appropriate label text.
        """

        if units is None:
            return None

        if units.in_kpc is not None and units.use_scaled:
            if units.in_kpc:
                return "kpc"
            else:
                return "arcsec"

        if units.use_scaled:
            return "scaled"
        return "pixels"


class YLabel(AbstractLabel):
    def set(
        self,
        units: Units,
        include_brackets: bool = True,
        auto_label: Optional[str] = None,
    ):
        """
        Set the y labels of the figure, including the fontsize.

        The y labels are always the distance scales, thus the labels are either arc-seconds or kpc and depending on
        the unit_label the figure is plotted in.

        Parameters
        ----------
        units
            The units of the image that is plotted which informs the appropriate y label text.
        include_brackets
            Whether to include brackets around the y label text of the units.
        """

        config_dict = self.config_dict

        if "label" in self.config_dict:
            config_dict.pop("label")

        if self.manual_label is not None:
            plt.ylabel(ylabel=self.manual_label, **config_dict)
        elif auto_label is not None:
            plt.ylabel(ylabel=auto_label, **config_dict)
        else:
            if include_brackets:
                plt.ylabel(ylabel=f"y ({self.label_from(units=units)})", **config_dict)
            else:
                plt.ylabel(ylabel=self.label_from(units=units), **config_dict)


class XLabel(AbstractLabel):
    def set(
        self,
        units: Units,
        include_brackets: bool = True,
        auto_label: Optional[str] = None,
    ):
        """
        Set the x labels of the figure, including the fontsize.

        The x labels are always the distance scales, thus the labels are either arc-seconds or kpc and depending on
        the unit_label the figure is plotted in.

        Parameters
        ----------
        units
            The units of the image that is plotted which informs the appropriate x label text.
        include_brackets
            Whether to include brackets around the x label text of the units.
        """

        config_dict = self.config_dict

        if "label" in self.config_dict:
            config_dict.pop("label")

        if self.manual_label is not None:
            plt.xlabel(xlabel=self.manual_label, **config_dict)
        elif auto_label is not None:
            plt.xlabel(xlabel=auto_label, **config_dict)
        else:
            if include_brackets:
                plt.xlabel(xlabel=f"x ({self.label_from(units=units)})", **config_dict)
            else:
                plt.xlabel(xlabel=self.label_from(units=units), **config_dict)
