import matplotlib.pyplot as plt

from autoarray.plot.wrap.base.abstract import AbstractMatWrap


class Title(AbstractMatWrap):
    def __init__(self, prefix: str = None, disable_log10_label : bool = False, **kwargs):
        """
        The settings used to customize the figure's title.

        This object wraps the following Matplotlib methods:

        - plt.title: https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.title.html

        The title will automatically be set if not specified, using the name of the function used to plot the data.

        Parameters
        ----------
        prefix
            A string that is added before the title, for example to put the name of the dataset and galaxy in the title.
        disable_log10_label
            If True, the (log10) label is not added to the title if the data is plotted on a log-scale.
        """

        super().__init__(**kwargs)

        self.prefix = prefix
        self.disable_log10_label = disable_log10_label
        self.manual_label = self.kwargs.get("label")

    def set(self, auto_title=None, use_log10: bool = False):
        config_dict = self.config_dict

        label = auto_title if self.manual_label is None else self.manual_label

        if self.prefix is not None:
            label = f"{self.prefix} {label}"

        if use_log10 and not self.disable_log10_label:
            label = f"{label} (log10)"

        if "label" in config_dict:
            config_dict.pop("label")

        plt.title(label=label, **config_dict)
