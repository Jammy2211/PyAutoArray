import matplotlib.pyplot as plt
from typing import Union, Tuple

from autoarray.plot.wrap.base.abstract import AbstractMatWrap

class Figure(AbstractMatWrap):
    """
    Sets up the Matplotlib figure before plotting (this is used when plotting individual figures and subplots).

    This object wraps the following Matplotlib methods:

    - plt.figure: https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.figure.html
    - plt.close: https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.close.html

    It also controls the aspect ratio of the figure plotted.
    """

    @property
    def config_dict(self):
        """
        Creates a config dict of valid inputs of the method `plt.figure` from the object's config_dict.
        """

        config_dict = super().config_dict

        if config_dict["figsize"] == "auto":
            config_dict["figsize"] = None
        elif isinstance(config_dict["figsize"], str):
            config_dict["figsize"] = tuple(
                map(int, config_dict["figsize"][1:-1].split(","))
            )

        return config_dict

    def aspect_for_subplot_from(self, extent):

        ratio = float((extent[1] - extent[0]) / (extent[3] - extent[2]))

        if self.config_dict["aspect"] in "square":
            return ratio
        elif self.config_dict["aspect"] in "auto":
            return 1.0 / ratio
        elif self.config_dict["aspect"] in "equal":
            return 1.0

    def aspect_from(self, shape_native: Union[Tuple[int, int]]) -> Union[float, str]:
        """
        Returns the aspect ratio of the figure from the 2D shape of a data structure.

        This is used to ensure that rectangular arrays are plotted as square figures on sub-plots.

        Parameters
        ----------
        shape_native
            The two dimensional shape of an `Array2D` that is to be plotted.
        """
        if isinstance(self.config_dict["aspect"], str):
            if self.config_dict["aspect"] in "square":
                return float(shape_native[1]) / float(shape_native[0])

        return self.config_dict["aspect"]

    def open(self):
        """
        Wraps the Matplotlib method 'plt.figure' for opening a figure.
        """
        if not plt.fignum_exists(num=1):
            config_dict = self.config_dict
            config_dict.pop("aspect")
            fig = plt.figure(**config_dict)
            return fig, plt.gca()
        return None, None

    def close(self):
        """
        Wraps the Matplotlib method 'plt.close' for closing a figure.
        """
        if plt.fignum_exists(num=1):
            plt.close()