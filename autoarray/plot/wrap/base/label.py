import matplotlib.pyplot as plt
from typing import Optional

from autoconf import conf

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


class YLabel(AbstractLabel):
    def set(
        self,
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
        """

        config_dict = self.config_dict

        if self.manual_label is not None:
            config_dict.pop("ylabel")
            plt.ylabel(ylabel=self.manual_label, **config_dict)
        elif auto_label is not None:
            config_dict.pop("ylabel")
            plt.ylabel(ylabel=auto_label, **config_dict)
        else:
            plt.ylabel(**config_dict)


class XLabel(AbstractLabel):
    def set(
        self,
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
        """

        config_dict = self.config_dict

        if self.manual_label is not None:
            config_dict.pop("xlabel")
            plt.xlabel(xlabel=self.manual_label, **config_dict)
        elif auto_label is not None:
            config_dict.pop("xlabel")
            plt.xlabel(xlabel=auto_label, **config_dict)
        else:
            plt.xlabel(**config_dict)
