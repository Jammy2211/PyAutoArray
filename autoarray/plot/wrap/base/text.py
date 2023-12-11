import matplotlib.pyplot as plt

from autoarray.plot.wrap.base.abstract import AbstractMatWrap


class Text(AbstractMatWrap):
    """
    The settings used to customize text on the figure.

    This object wraps the following Matplotlib methods:

    - plt.text: https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.text.html
    """

    def set(self):
        if "x" not in self.kwargs and "y" not in self.kwargs and "s" not in self.kwargs:
            return

        plt.text(**self.config_dict)
