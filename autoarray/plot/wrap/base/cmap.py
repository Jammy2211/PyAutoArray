import copy
import logging
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as colors
import numpy as np

from autoconf import conf

from autoarray.plot.wrap.base.abstract import AbstractMatWrap

from autoarray import exc

logger = logging.getLogger(__name__)


class Cmap(AbstractMatWrap):
    def __init__(self, symmetric: bool = False, **kwargs):
        """
        Customizes the Matplotlib colormap and its normalization.

        This object wraps the following Matplotlib methods:

        - colors.Linear: https://matplotlib.org/3.3.2/tutorials/colors/colormaps.html
        - colors.LogNorm: https://matplotlib.org/3.3.2/tutorials/colors/colormapnorms.html
        - colors.SymLogNorm: https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.colors.SymLogNorm.html

        The cmap that is created is passed into various Matplotlib methods, most notably imshow:

        - https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.imshow.html

        Parameters
        ----------
        symmetric
            If True, the colormap normalization (e.g. `vmin` and `vmax`) span the same absolute values producing a
            symmetric color bar.
        """

        super().__init__(**kwargs)

        self._symmetric = symmetric
        self.symmetric_value = None

    def symmetric_cmap_from(self, symmetric_value=None):
        cmap = copy.copy(self)

        cmap._symmetric = True
        cmap.symmetric_value = symmetric_value

        return cmap

    def vmin_from(self, array: np.ndarray):
        if self.config_dict["vmin"] is None:
            return np.min(array)
        return self.config_dict["vmin"]

    def vmax_from(self, array: np.ndarray):
        if self.config_dict["vmax"] is None:
            return np.max(array)
        return self.config_dict["vmax"]

    def norm_from(self, array: np.ndarray, use_log10: bool = False) -> object:
        """
        Returns the `Normalization` object which scales of the colormap.

        If vmin / vmax are not manually input by the user, the minimum / maximum values of the data being plotted
        are used.

        Parameters
        ----------
        array
            The array of data which is to be plotted.
        """

        vmin = self.vmin_from(array=array)
        vmax = self.vmax_from(array=array)

        if self._symmetric:
            if vmin < 0.0 and vmax > 0.0:
                if self.symmetric_value is None:
                    if abs(vmin) > abs(vmax):
                        vmax = abs(vmin)
                    else:
                        vmin = -vmax
                else:
                    vmin = -self.symmetric_value
                    vmax = self.symmetric_value

        if isinstance(self.config_dict["norm"], colors.Normalize):
            return self.config_dict["norm"]

        if self.config_dict["norm"] in "log" or use_log10:
            if vmin < self.log10_min_value:
                vmin = self.log10_min_value
            return colors.LogNorm(vmin=vmin, vmax=vmax)
        elif self.config_dict["norm"] in "linear":
            return colors.Normalize(vmin=vmin, vmax=vmax)
        elif self.config_dict["norm"] in "symmetric_log":
            return colors.SymLogNorm(
                vmin=vmin,
                vmax=vmax,
                linthresh=self.config_dict["linthresh"],
                linscale=self.config_dict["linscale"],
            )
        elif self.config_dict["norm"] in "diverge":
            return colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

        raise exc.PlottingException(
            "The normalization (norm) supplied to the plotter is not a valid string must be "
            "{linear, log, symmetric_log}"
        )

    @property
    def cmap(self):
        if self.config_dict["cmap"] == "default":
            from autoarray.plot.wrap.segmentdata import segmentdata

            return LinearSegmentedColormap(name="default", segmentdata=segmentdata)

        return self.config_dict["cmap"]
