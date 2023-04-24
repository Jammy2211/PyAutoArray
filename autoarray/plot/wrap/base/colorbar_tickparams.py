from autoarray.plot.wrap.base.abstract import AbstractMatWrap


class ColorbarTickParams(AbstractMatWrap):
    """
    Customizes the ticks of the colorbar of the plotted figure.

    This object wraps the following Matplotlib colorbar method:

    - cb.set_yticklabels: https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.axes.Axes.set_yticklabels.html
    """

    def set(self, cb):
        cb.ax.tick_params(**self.config_dict)
