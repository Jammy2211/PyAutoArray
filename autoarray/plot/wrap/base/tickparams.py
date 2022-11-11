import matplotlib.pyplot as plt

from autoarray.plot.wrap.base.abstract import AbstractMatWrap


class TickParams(AbstractMatWrap):
    """
    The settings used to customize a figure's y and x ticks parameters.

    This object wraps the following Matplotlib methods:

    - plt.tick_params: https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.tick_params.html
    """

    def set(self):
        """Set the tick_params of the figure using the method `plt.tick_params`."""
        plt.tick_params(**self.config_dict)
