import matplotlib.pyplot as plt

from autoarray.plot.wrap.base.abstract import AbstractMatWrap


class Title(AbstractMatWrap):
    def __init__(self, **kwargs):
        """
        The settings used to customize the figure's title.

        This object wraps the following Matplotlib methods:

        - plt.title: https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.title.html

        The title will automatically be set if not specified, using the name of the function used to plot the data.
        """

        super().__init__(**kwargs)

        self.manual_label = self.kwargs.get("label")

    def set(self, auto_title=None, use_log10 : bool = False):

        config_dict = self.config_dict

        label = auto_title if self.manual_label is None else self.manual_label

        if use_log10:
            label = f"{label} (log10)"

        if "label" in config_dict:
            config_dict.pop("label")

        plt.title(label=label, **config_dict)
