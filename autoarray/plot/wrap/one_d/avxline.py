import matplotlib.pyplot as plt
from typing import List, Optional

from autoarray.plot.wrap.one_d.abstract import AbstractMatWrap1D


class AXVLine(AbstractMatWrap1D):
    def __init__(self, no_label=False, **kwargs):
        """
        Plots vertical lines on 1D plot of y versus x using the method `plt.axvline`.

        This method is typically called after `plot_y_vs_x` to add vertical lines to the figure.

        This object wraps the following Matplotlib methods:

        - plt.avxline: https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.axvline.html

        Parameters
        ----------
        vertical_line
            The vertical lines of data that are plotted on the figure.
        label
            Labels for each vertical line used by a `Legend`.
        """
        super().__init__(**kwargs)

        self.no_label = no_label

    def axvline_vertical_line(
        self,
        vertical_line: float,
        vertical_errors: Optional[List[float]] = None,
        label: Optional[str] = None,
    ):
        """
        Plot an input vertical line given by its x coordinate as a float using the method `plt.axvline`.

        Parameters
        ----------
        vertical_line
            The vertical lines of data that are plotted on the figure.
        label
            Labels for each vertical line used by a `Legend`.
        """

        if vertical_line is [] or vertical_line is None:
            return

        if self.no_label:
            label = None

        plt.axvline(x=vertical_line, label=label, **self.config_dict)

        if vertical_errors is not None:
            config_dict = self.config_dict

            if "linestyle" in config_dict:
                config_dict.pop("linestyle")

            plt.axvline(x=vertical_errors[0], linestyle="--", **config_dict)
            plt.axvline(x=vertical_errors[1], linestyle="--", **config_dict)
