import matplotlib.pyplot as plt

from autoarray.plot.wrap.base.abstract import AbstractMatWrap


class Legend(AbstractMatWrap):
    """
    The settings used to include and customize a legend on a figure.

    This object wraps the following Matplotlib methods:

    - plt.legend: https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.legend.html
    """

    def __init__(self, label=None, include=True, **kwargs):
        super().__init__(**kwargs)

        self.label = label
        self.include = include

    def set(self):
        if self.include:
            config_dict = self.config_dict
            config_dict.pop("include") if "include" in config_dict else None
            config_dict.pop("include_2d") if "include_2d" in config_dict else None

            plt.legend(**config_dict)
