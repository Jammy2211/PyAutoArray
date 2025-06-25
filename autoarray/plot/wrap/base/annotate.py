from autoarray.plot.wrap.base.abstract import AbstractMatWrap


class Annotate(AbstractMatWrap):
    """
    The settings used to customize annotations on the figure.

    This object wraps the following Matplotlib methods:

    - plt.annotate: https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.text.html
    """

    def set(self):

        import matplotlib.pyplot as plt

        if "x" not in self.kwargs and "y" not in self.kwargs and "s" not in self.kwargs:
            return

        plt.annotate(**self.config_dict)
