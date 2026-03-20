import copy
import numpy as np
from typing import Optional


class Cmap:
    def __init__(
        self,
        cmap: str = "default",
        norm: str = "linear",
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        linthresh: float = 0.05,
        linscale: float = 0.01,
        symmetric: bool = False,
    ):
        self.cmap_name = cmap
        self.norm_type = norm
        self.vmin = vmin
        self.vmax = vmax
        self.linthresh = linthresh
        self.linscale = linscale
        self._symmetric = symmetric
        self.symmetric_value = None

    @property
    def log10_min_value(self) -> float:
        try:
            from autoconf import conf
            return conf.instance["visualize"]["general"]["general"]["log10_min_value"]
        except Exception:
            return 1.0e-4

    @property
    def log10_max_value(self) -> float:
        try:
            from autoconf import conf
            return float(conf.instance["visualize"]["general"]["general"]["log10_max_value"])
        except Exception:
            return 1.0e10

    def symmetric_cmap_from(self, symmetric_value=None):
        cmap = copy.copy(self)
        cmap._symmetric = True
        cmap.symmetric_value = symmetric_value
        return cmap

    def vmin_from(self, array: np.ndarray, use_log10: bool = False) -> float:
        use_log10 = use_log10 or self.norm_type == "log"
        vmin = np.nanmin(array) if self.vmin is None else self.vmin
        if use_log10 and vmin < self.log10_min_value:
            vmin = self.log10_min_value
        return vmin

    def vmax_from(self, array: np.ndarray, use_log10: bool = False) -> float:
        use_log10 = use_log10 or self.norm_type == "log"
        vmax = np.nanmax(array) if self.vmax is None else self.vmax
        if use_log10 and vmax > self.log10_max_value:
            vmax = self.log10_max_value
        return vmax

    def norm_from(self, array: np.ndarray, use_log10: bool = False):
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

        if use_log10 or self.norm_type == "log":
            return colors.LogNorm(vmin=vmin, vmax=vmax)
        elif self.norm_type == "symmetric_log":
            return colors.SymLogNorm(
                vmin=vmin,
                vmax=vmax,
                linthresh=self.linthresh,
                linscale=self.linscale,
            )
        elif self.norm_type == "diverge":
            return colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        else:
            return colors.Normalize(vmin=vmin, vmax=vmax)

    @property
    def cmap(self):
        from matplotlib.colors import LinearSegmentedColormap

        if self.cmap_name == "default":
            from autoarray.plot.wrap.segmentdata import segmentdata
            return LinearSegmentedColormap(name="default", segmentdata=segmentdata)

        return self.cmap_name
