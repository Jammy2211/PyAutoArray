import copy
import logging
import numpy as np

from autoarray.plot.wrap.base.abstract import AbstractMatWrap
from autoarray import exc

logger = logging.getLogger(__name__)


class Cmap(AbstractMatWrap):
    def __init__(self, symmetric: bool = False, **kwargs):
        super().__init__(**kwargs)
        self._symmetric = symmetric
        self.symmetric_value = None

    @property
    def defaults(self):
        return {
            "cmap": "default",
            "norm": "linear",
            "vmin": None,
            "vmax": None,
            "linthresh": 0.05,
            "linscale": 0.01,
        }

    def symmetric_cmap_from(self, symmetric_value=None):
        cmap = copy.copy(self)
        cmap._symmetric = True
        cmap.symmetric_value = symmetric_value
        return cmap

    def vmin_from(self, array: np.ndarray, use_log10: bool = False) -> float:
        if self.config_dict["norm"] in "log":
            use_log10 = True

        if self.config_dict["vmin"] is None:
            vmin = np.nanmin(array)
        else:
            vmin = self.config_dict["vmin"]

        if use_log10 and (vmin < self.log10_min_value):
            vmin = self.log10_min_value

        return vmin

    def vmax_from(self, array: np.ndarray, use_log10: bool = False) -> float:
        if self.config_dict["norm"] in "log":
            use_log10 = True

        if self.config_dict["vmax"] is None:
            vmax = np.nanmax(array)
        else:
            vmax = self.config_dict["vmax"]

        if use_log10 and (vmax > self.log10_max_value):
            vmax = self.log10_max_value

        return vmax

    def norm_from(self, array: np.ndarray, use_log10: bool = False) -> object:
        import matplotlib.colors as colors

        vmin = self.vmin_from(array=array, use_log10=use_log10)
        vmax = self.vmax_from(array=array, use_log10=use_log10)

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
        from matplotlib.colors import LinearSegmentedColormap

        if self.config_dict["cmap"] == "default":
            from autoarray.plot.wrap.segmentdata import segmentdata

            return LinearSegmentedColormap(name="default", segmentdata=segmentdata)

        return self.config_dict["cmap"]
